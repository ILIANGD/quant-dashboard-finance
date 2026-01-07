import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import locale

# Tenter de définir la locale pour le formatage
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    pass

from services.data_loader import load_price_history, load_live_quote


# =========================
# Backtesting strategies
# =========================

def backtest_buy_hold(prices: pd.Series) -> pd.Series:
    """Buy & Hold: invested all the time."""
    prices = prices.dropna()
    rets = prices.pct_change().dropna()
    equity = (1 + rets).cumprod()
    equity.name = "Buy & Hold"
    return equity


def backtest_momentum_ma(prices: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """Momentum MA crossover."""
    prices = prices.dropna()
    rets = prices.pct_change().dropna()

    ma_fast = prices.rolling(fast).mean()
    ma_slow = prices.rolling(slow).mean()

    signal = (ma_fast > ma_slow).astype(int)
    position = signal.shift(1).reindex(rets.index).fillna(0)

    strat_rets = position * rets
    equity = (1 + strat_rets).cumprod()
    equity.name = "Momentum"
    return equity


def backtest_breakout(prices: pd.Series, lookback: int = 50) -> pd.Series:
    """N-day breakout."""
    prices = prices.dropna()
    rets = prices.pct_change().dropna()

    rolling_high = prices.shift(1).rolling(lookback, min_periods=lookback).max()
    signal = (prices >= rolling_high).astype(int)

    position = signal.shift(1).reindex(rets.index).fillna(0)
    strat_rets = position * rets

    equity = (1 + strat_rets).cumprod()
    equity.name = f"Breakout_{lookback}d"
    return equity


# =========================
# Metrics
# =========================

def compute_metrics(series: pd.Series | pd.DataFrame) -> dict:
    if series is None: return {}
    if isinstance(series, pd.DataFrame): series = series.iloc[:, 0]

    series = series.dropna()
    rets = series.pct_change().dropna()
    if rets.empty: return {}

    mean_daily = float(rets.mean())
    vol_daily = float(rets.std())
    annual_return = (1 + mean_daily) ** 252 - 1
    annual_vol = vol_daily * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol != 0 else 0.0

    cum = (1 + rets).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = annual_return / downside if downside > 0 else 0.0

    return {
        "mean_daily": mean_daily, "vol_daily": vol_daily,
        "annual_return": annual_return, "annual_vol": annual_vol,
        "sharpe": sharpe, "max_drawdown": max_dd,
        "calmar": calmar, "sortino": sortino,
    }


def metric_card(label: str, value_str: str, help_text: str = None):
    st.caption(f"**{label}**", help=help_text)
    st.markdown(f"<div style='font-size:1.4em; font-weight:600;'>{value_str}</div>", unsafe_allow_html=True)


def display_metrics_grid(metrics: dict):
    if not metrics: return
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Annual Return", f"{metrics['annual_return']:.2%}")
    with c2: metric_card("Volatility", f"{metrics['annual_vol']:.2%}")
    with c3: metric_card("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
    with c4: metric_card("Max Drawdown", f"{metrics['max_drawdown']:.2%}", "Maximum loss from peak to trough")

    c5, c6, c7, _ = st.columns(4)
    with c5: metric_card("Calmar", f"{metrics['calmar']:.2f}")
    with c6: metric_card("Sortino", f"{metrics['sortino']:.2f}")
    with c7: metric_card("Daily Mean", f"{metrics['mean_daily']:.3%}")


# =========================
# ML Forecast Logic
# =========================

def train_forecast_model(prices: pd.Series, horizon: int = 20, use_log: bool = True):
    """
    Trains a Linear Regression model on historical time steps to predict future trend.
    """
    s = prices.dropna()
    if len(s) < 30: return pd.DataFrame(), {}

    # Feature Engineering: X = Integer Time
    y = np.log(s.values) if use_log else s.values
    X = np.arange(len(y)).reshape(-1, 1)

    # Train Model (ML Regression)
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate In-Sample Performance
    y_pred_hist = model.predict(X)
    r2 = r2_score(y, y_pred_hist)
    rmse = np.sqrt(mean_squared_error(y, y_pred_hist))
    if use_log: rmse = np.exp(rmse) 

    # Predict Future
    last_idx = X[-1][0]
    X_future = np.arange(last_idx + 1, last_idx + 1 + horizon).reshape(-1, 1)
    y_future_pred = model.predict(X_future)

    # Confidence Interval
    residuals = y - y_pred_hist
    std_resid = np.std(residuals)
    z_score = 1.96 
    
    lower_log = y_future_pred - z_score * std_resid
    upper_log = y_future_pred + z_score * std_resid

    # Create Date Index for Future
    freq = pd.infer_freq(s.index) or "D"
    future_dates = pd.date_range(start=s.index[-1], periods=horizon+1, freq=freq)[1:]

    # Transform back if Log
    if use_log:
        forecast_vals = np.exp(y_future_pred)
        lower_vals = np.exp(lower_log)
        upper_vals = np.exp(upper_log)
    else:
        forecast_vals = y_future_pred
        lower_vals = lower_log
        upper_vals = upper_log

    df_fc = pd.DataFrame({
        "forecast": forecast_vals,
        "lower": lower_vals,
        "upper": upper_vals
    }, index=future_dates)

    metrics = {
        "r2": r2,
        "rmse": rmse,
        "model": "Linear Regression (Log-Transformed)" if use_log else "Linear Regression"
    }

    return df_fc, metrics


# =========================
# Main Page
# =========================

def single_asset_page():
    st.title("Single Asset Analysis")
    st.autorefresh = st_autorefresh(interval=300_000, key="auto_refresh_5min")

    # ==========================================
    # 1. CONTROLS
    # ==========================================
    with st.container(border=True):
        st.subheader("Controls")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            default_ticker = st.query_params.get("ticker", "BZ=F")
            ticker = st.text_input("Ticker", value=default_ticker)
            if ticker != st.query_params.get("ticker"):
                st.query_params["ticker"] = ticker
            
        with c2: period = st.selectbox("Lookback", ["3mo", "6mo", "1y", "2y", "5y"], index=4)
        with c3: interval = st.selectbox("Freq", ["1d", "1wk", "1mo"], index=0)
        with c4: strategy_name = st.selectbox("Strategy", ["Buy & Hold", "Momentum (MA Cross)", "Breakout"], index=0)

        st.divider()
        
        # Advanced Params
        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown("**Strategy Settings**")
            if "Momentum" in strategy_name:
                sc1, sc2 = st.columns(2)
                fast = sc1.slider("Fast MA", 5, 50, 20)
                slow = sc2.slider("Slow MA", 20, 200, 50)
                lookback = 50
            elif "Breakout" in strategy_name:
                lookback = st.slider("Breakout Lookback", 10, 200, 50)
                fast, slow = 20, 50
            else:
                st.caption("Standard Buy & Hold.")
                fast, slow, lookback = 20, 50, 50
        
        with ac2:
            st.markdown("**Forecast Settings**")
            fc1, fc2 = st.columns(2)
            with fc1: horizon_fc = st.slider("Horizon (periods)", 5, 120, 60)
            with fc2: log_fc = st.toggle("Log-Space Model", value=True)

    # Load Data
    try:
        data = load_price_history(ticker, period=period, interval=interval)
        prices = data["price"].dropna()
        # Remove timezone to avoid merge conflicts
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
    except:
        st.error("Error loading data."); return

    if prices.empty: st.warning("No data available."); return

    # Strategies
    equity_bh = backtest_buy_hold(prices)
    equity_mom = backtest_momentum_ma(prices, fast, slow)
    equity_brk = backtest_breakout(prices, lookback)

    if "Momentum" in strategy_name: equity_sel = equity_mom; strat_label = "Momentum"
    elif "Breakout" in strategy_name: equity_sel = equity_brk; strat_label = "Breakout"
    else: equity_sel = equity_bh; strat_label = "Buy & Hold"

    metrics_asset = compute_metrics(prices)
    metrics_strat = compute_metrics(equity_sel)

    # ==========================================
    # 2. ASSET OVERVIEW
    # ==========================================
    st.divider()
    st.header("Asset Overview")
    
    quote_key = f"q_{ticker}"
    if quote_key not in st.session_state: 
        st.session_state[quote_key] = load_live_quote(ticker)
    
    quote = st.session_state[quote_key]
    curr = quote.get("last_price")
    prev = quote.get("prev_close")
    
    col_p, col_m = st.columns([1, 2])
    with col_p:
        delta = f"{curr - prev:.2f} ({(curr-prev)/prev:.2%})" if (curr and prev) else ""
        st.metric(f"Price ({ticker})", f"{curr:.2f} $" if curr else "N/A", delta)
        if st.button("Refresh"): 
            st.session_state[quote_key] = load_live_quote(ticker)
            st.rerun()
            
    with col_m:
        m1, m2, m3, m4 = st.columns(4)
        with m1: metric_card("Return (Ann.)", f"{metrics_asset['annual_return']:.2%}")
        with m2: metric_card("Volatility", f"{metrics_asset['annual_vol']:.2%}")
        with m3: metric_card("Sharpe", f"{metrics_asset['sharpe']:.2f}")
        with m4: metric_card("Max Drawdown", f"{metrics_asset['max_drawdown']:.2%}", "Max loss")

    # ==========================================
    # 3. PERFORMANCE (Chart Style Double Axe)
    # ==========================================
    st.divider()
    st.subheader(f"Price & Strategy Performance ({strat_label})")
    
    # Prepare Data: Left Axis = Price ($), Right Axis = Base 100
    df_perf = pd.DataFrame({
        "date": prices.index,
        "Asset Price ($)": prices.values,
        "Strategy (Base 100)": equity_sel.values * 100
    })

    # Base Chart
    base = alt.Chart(df_perf).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%d/%m/%Y", grid=True, gridOpacity=0.3))
    )

    # 1. Asset Price Line (Blue, Left Axis) - ADAPTIVE SCALE (zero=False)
    line_asset = base.mark_line(stroke="#4A90E2", interpolate="monotone").encode(
        y=alt.Y(
            "Asset Price ($):Q",
            title="Asset Price ($)",
            scale=alt.Scale(zero=False),  # <--- ZOOM AUTO
            axis=alt.Axis(titleColor="#4A90E2", grid=True, gridOpacity=0.3)
        )
    )

    # 2. Strategy Line (Green, Right Axis) - ADAPTIVE SCALE (zero=False)
    line_strat = base.mark_line(stroke="#2ECC71", interpolate="monotone").encode(
        y=alt.Y(
            "Strategy (Base 100):Q",
            title="Strategy Performance (Base 100)",
            scale=alt.Scale(zero=False), # <--- ZOOM AUTO
            axis=alt.Axis(titleColor="#2ECC71", orient="right", grid=False)
        )
    )

    # Combine with independent scales
    chart_perf = alt.layer(line_asset, line_strat).resolve_scale(
        y='independent'
    ).properties(height=450)

    st.altair_chart(chart_perf, use_container_width=True)
    
    st.caption("Blue: Asset Price (Left Scale) | Green: Strategy rebased to 100 (Right Scale)")
    
    st.subheader("Strategy Metrics")
    display_metrics_grid(metrics_strat)

    # ==========================================
    # 4. FORECAST (Updated Scale)
    # ==========================================
    st.divider()
    st.header("Forecast Analysis")

    fc_df, fc_metrics = train_forecast_model(prices, horizon_fc, log_fc)

    if not fc_df.empty:
        hist_data = prices.iloc[-252:].to_frame("price").reset_index() 
        hist_data.columns = ["date", "price"]
        fc_data = fc_df.reset_index().rename(columns={"index": "date"})
        
        # --- ALTAIR CHART ---
        # Axes with adaptive scale
        x_axis = alt.X("date:T", axis=alt.Axis(format="%d/%m", title="Date", grid=True, gridOpacity=0.3))
        # Important: zero=False for adaptive zoom
        y_axis = alt.Y("price:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="$f", title="Price ($)", grid=True, gridOpacity=0.3))

        c_hist = alt.Chart(hist_data).mark_line().encode(
            x=x_axis, y=y_axis,
            color=alt.value("#4A90E2"), stroke=alt.datum("Prix historique")
        )
        
        c_fc = alt.Chart(fc_data).mark_line(strokeDash=[6, 4]).encode(
            x=x_axis, y=alt.Y("forecast:Q", scale=alt.Scale(zero=False)),
            color=alt.value("#2ECC71"), stroke=alt.datum("Prédiction")
        )
        
        c_band = alt.Chart(fc_data).mark_area(opacity=0.3).encode(
            x=x_axis,
            y=alt.Y("lower:Q", scale=alt.Scale(zero=False)),
            y2="upper:Q",
            color=alt.value("#2ECC71"), fill=alt.datum("Intervalle de confiance 95%") 
        )

        chart = alt.layer(c_hist, c_band, c_fc).properties(height=500).configure_legend(
            orient='bottom', title=None, labelFontSize=12,
        ).configure_axis(grid=True, gridOpacity=0.3)
        
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Model Performance")
        fp1, fp2, fp3 = st.columns(3)
        with fp1: st.info(f"**Algorithm:** {fc_metrics['model']}")
        with fp2: st.metric("R² Score", f"{fc_metrics['r2']:.3f}")
        with fp3: st.metric("RMSE", f"{fc_metrics['rmse']:.2f} $")
        st.markdown(f"**Prediction (+{horizon_fc}d):** {fc_df['forecast'].iloc[-1]:.2f} $")
    else:
        st.warning("Not enough data to train model.")

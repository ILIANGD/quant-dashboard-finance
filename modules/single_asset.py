import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import altair as alt

from services.data_loader import load_price_history, load_live_quote


# =========================
# Backtesting strategies (Inchangé)
# =========================

def backtest_buy_hold(prices: pd.Series) -> pd.Series:
    """Buy & Hold: invested all the time."""
    prices = prices.dropna()
    rets = prices.pct_change().dropna()
    equity = (1 + rets).cumprod()
    equity.name = "Buy & Hold"
    return equity


def backtest_momentum_ma(prices: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """
    Simple momentum MA crossover:
    - Fast MA > Slow MA -> long (1)
    - else -> cash (0)
    """
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
    """
    N-day breakout:
    - Long if today's price breaks above previous N-day high
    - else cash
    """
    prices = prices.dropna()
    rets = prices.pct_change().dropna()

    # IMPORTANT: require full lookback window before generating signals
    rolling_high = prices.shift(1).rolling(lookback, min_periods=lookback).max()
    signal = (prices >= rolling_high).astype(int)

    position = signal.shift(1).reindex(rets.index).fillna(0)
    strat_rets = position * rets

    equity = (1 + strat_rets).cumprod()
    equity.name = f"Breakout_{lookback}d"
    return equity


# =========================
# Metrics (Inchangé)
# =========================

def compute_metrics(series: pd.Series | pd.DataFrame) -> dict:
    """Compute key performance metrics for an asset or strategy."""
    if series is None:
        return {}
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    series = series.dropna()
    rets = series.pct_change().dropna()
    if rets.empty:
        return {}

    mean_daily = float(rets.mean())
    vol_daily = float(rets.std())

    annual_return = (1 + mean_daily) ** 252 - 1
    annual_vol = vol_daily * np.sqrt(252)

    sharpe = float("nan") if (annual_vol == 0 or pd.isna(annual_vol)) else (annual_return / annual_vol)

    cum = (1 + rets).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = float(drawdown.min())

    calmar = float("nan") if max_dd == 0 else (annual_return / abs(max_dd))

    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = annual_return / downside if (downside != 0 and not pd.isna(downside)) else float("nan")

    return {
        "mean_daily": mean_daily,
        "vol_daily": vol_daily,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "sortino": sortino,
    }


METRIC_TOOLTIPS = {
    "Annual return": "Annualized return computed from daily compounded returns over 252 trading days.",
    "Annual volatility": "Annualized volatility: std(daily returns) * sqrt(252).",
    "Sharpe": "Sharpe ratio (risk-free rate assumed 0): annual return / annual volatility.",
    "Max drawdown": "Maximum peak-to-trough decline of the equity curve (in %).",
    "Calmar": "Calmar ratio: annual return / |max drawdown|.",
    "Sortino": "Sortino ratio: annual return / downside volatility (only negative returns).",
    "Daily mean return": "Average daily return (mean of daily returns).",
}


def metric_with_help(label: str, value_str: str):
    tooltip = METRIC_TOOLTIPS.get(label, "")
    st.caption(f"**{label}**", help=tooltip)
    st.markdown(
        f"<div style='font-size:1.4em; font-weight:600;'>{value_str}</div>",
        unsafe_allow_html=True,
    )


def display_metrics_grid(metrics: dict):
    """Helper to display metrics in a 4x2 grid."""
    if not metrics:
        st.info("No metrics available.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_with_help("Annual return", f"{metrics.get('annual_return', float('nan')) * 100:.2f}%")
    with c2: metric_with_help("Annual volatility", f"{metrics.get('annual_vol', float('nan')) * 100:.2f}%")
    with c3: metric_with_help("Sharpe", f"{metrics.get('sharpe', float('nan')):.2f}")
    with c4: metric_with_help("Max drawdown", f"{metrics.get('max_drawdown', float('nan')) * 100:.2f}%")

    c5, c6, c7, _ = st.columns(4)
    with c5: metric_with_help("Calmar", f"{metrics.get('calmar', float('nan')):.2f}")
    with c6: metric_with_help("Sortino", f"{metrics.get('sortino', float('nan')):.2f}")
    with c7: metric_with_help("Daily mean return", f"{metrics.get('mean_daily', float('nan')) * 100:.3f}%")


# =========================
# Forecast
# =========================

def _infer_pandas_freq(index: pd.DatetimeIndex) -> str:
    freq = pd.infer_freq(index)
    if freq:
        return freq
    if len(index) >= 2:
        delta = index[-1] - index[-2]
        if delta.days >= 28:
            return "MS"
        if delta.days >= 6:
            return "W"
    return "D"


def forecast_linear_trend_with_ci(
    prices: pd.Series,
    horizon: int = 20,
    ci_level: float = 0.95,
    use_log: bool = True,
) -> pd.DataFrame:
    """
    Linear regression trend forecast on time index with approximate prediction intervals.
    """
    s = prices.dropna().copy()
    if len(s) < 20:
        return pd.DataFrame()

    y = np.log(s.values) if use_log else s.values
    t = np.arange(len(y), dtype=float)

    b, a = np.polyfit(t, y, deg=1)  # y ≈ b*t + a
    y_hat = a + b * t
    resid = y - y_hat
    sigma = float(np.std(resid, ddof=2)) if len(resid) > 2 else 0.0

    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(round(ci_level, 2), 1.96)

    t_f = np.arange(len(y), len(y) + horizon, dtype=float)
    y_f = a + b * t_f

    lower = y_f - z * sigma
    upper = y_f + z * sigma

    freq = _infer_pandas_freq(s.index)
    future_index = pd.date_range(start=s.index[-1], periods=horizon + 1, freq=freq)[1:]

    if use_log:
        pred = np.exp(y_f)
        lo = np.exp(lower)
        hi = np.exp(upper)
    else:
        pred = y_f
        lo = lower
        hi = upper

    return pd.DataFrame({"forecast": pred, "lower": lo, "upper": hi}, index=future_index)


def plot_forecast_with_band(history: pd.Series, fc: pd.DataFrame):
    hist_df = history.dropna().to_frame("price").reset_index()
    hist_df.columns = ["date", "price"]

    fc_df = fc.reset_index()
    fc_df.columns = ["date", "forecast", "lower", "upper"]

    line_hist = alt.Chart(hist_df).mark_line().encode(x="date:T", y="price:Q")
    band = alt.Chart(fc_df).mark_area(opacity=0.2).encode(x="date:T", y="lower:Q", y2="upper:Q")
    line_fc = alt.Chart(fc_df).mark_line(strokeDash=[4, 4]).encode(x="date:T", y="forecast:Q")

    st.altair_chart(line_hist + band + line_fc, use_container_width=True)


# =========================
# Streamlit page (Refactorisée)
# =========================

def single_asset_page():
    st.title("Single Asset Analysis")

    # Auto refresh every 5 minutes
    st_autorefresh(interval=300_000, key="auto_refresh_5min")

    # ==========================================
    # 1. ZONE DE CONTRÔLE (Encadrée en haut)
    # ==========================================
    with st.container(border=True):
        st.subheader("⚙️ Configuration & Live Data")
        
        # Ligne 1 : Paramètres
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            default_ticker = st.query_params.get("ticker", "BZ=F")
            ticker = st.text_input("Ticker", value=default_ticker)
            if ticker != st.query_params.get("ticker"):
                st.query_params["ticker"] = ticker
            
        with col2:
            period = st.selectbox(
                "Lookback period",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                index=3,
            )

        with col3:
            interval = st.selectbox(
                "Frequency",
                ["1d", "1wk", "1mo"],
                index=0,
                help="1d = daily, 1wk = weekly, 1mo = monthly",
            )

        with col4:
            strategy_name = st.selectbox(
                "Strategy",
                ["Buy & Hold", "Momentum (Fast MA / Slow MA)", "Breakout (N-day high)"],
                index=0,
            )

        # Ligne 2 : Live Quote & Paramètres Stratégie Spécifiques
        st.markdown("---")
        row2_col1, row2_col2 = st.columns([1, 1])
        
        with row2_col1:
            # Gestion du Live Quote
            quote_key = f"live_quote::{ticker}"
            if "quote_refresh_nonce" not in st.session_state:
                st.session_state["quote_refresh_nonce"] = 0
            if quote_key not in st.session_state:
                try:
                    st.session_state[quote_key] = load_live_quote(ticker)
                except Exception:
                    st.session_state[quote_key] = {"last_price": None, "prev_close": None, "asof_utc": "N/A"}

            quote = st.session_state[quote_key]
            last_price = quote.get("last_price")
            prev_close = quote.get("prev_close")
            
            delta_str = ""
            if last_price is not None and prev_close is not None and prev_close != 0:
                diff = last_price - prev_close
                pct = diff / prev_close
                delta_str = f"{diff:.2f} ({pct:.2%})"

            # Affichage Prix + Bouton Refresh
            q1, q2 = st.columns([2, 1], vertical_alignment="bottom")
            with q1:
                 st.metric(
                    label=f"Live Price ({ticker})", 
                    value=f"{float(last_price):.2f} USD" if last_price else "N/A", 
                    delta=delta_str
                )
            with q2:
                if st.button("Refresh Quote"):
                    try:
                        st.session_state[quote_key] = load_live_quote(ticker)
                    except Exception:
                        pass
                    st.rerun()

        with row2_col2:
            # Paramètres dynamiques de la stratégie choisie
            if "Momentum" in strategy_name:
                s1, s2 = st.columns(2)
                fast = s1.slider("Fast MA", 5, 50, 20)
                slow = s2.slider("Slow MA", 20, 200, 50)
                if slow <= fast: st.warning("Slow > Fast required.")
            elif "Breakout" in strategy_name:
                lookback = st.slider("Breakout lookback (days)", 10, 200, 50)
            else:
                st.caption("No extra parameters for Buy & Hold.")
                # Default vars to avoid errors
                fast, slow, lookback = 20, 50, 50

    # Chargement des données (Invisible)
    try:
        data = load_price_history(ticker, period=period, interval=interval)
    except Exception:
        data = pd.DataFrame()

    if data.empty or "price" not in data.columns:
        st.error("Failed to load data for this ticker.")
        return

    prices = data["price"].dropna()
    if prices.empty:
        st.error("No usable price data.")
        return

    # Check data sufficiency
    n_points = int(prices.shape[0])
    if "Breakout" in strategy_name and n_points <= lookback + 5:
        st.warning(f"Not enough data for breakout (Need > {lookback}).")
        return

    # Calculs Stratégies
    equity_bh = backtest_buy_hold(prices)
    equity_mom = backtest_momentum_ma(prices, fast=fast, slow=slow) if "Momentum" in strategy_name else None
    equity_break = backtest_breakout(prices, lookback=lookback) if "Breakout" in strategy_name else None

    # Sélection courbe
    if equity_mom is not None:
        equity_sel = equity_mom.copy(); strat_label = "Momentum"
    elif equity_break is not None:
        equity_sel = equity_break.copy(); strat_label = f"Breakout ({lookback}d)"
    else:
        equity_sel = equity_bh.copy(); strat_label = "Buy & Hold"

    # ==========================================
    # 2. MAIN CHART SECTION
    # ==========================================
    st.divider() # Ligne de séparation
    st.header(f"📈 Price Analysis: Raw vs {strat_label}")

    price_raw = prices.copy()
    if isinstance(price_raw, pd.DataFrame): price_raw = price_raw.iloc[:, 0]
    if isinstance(equity_sel, pd.DataFrame): equity_sel = equity_sel.iloc[:, 0]

    initial_capital = 1000.0
    strategy_value = equity_sel * initial_capital
    price_raw.name = "Raw price"
    strategy_value.name = strat_label

    df_plot = pd.concat([price_raw, strategy_value], axis=1).dropna().reset_index()
    df_plot = df_plot.rename(columns={df_plot.columns[0]: "date"})
    # Rename explicit
    df_plot.columns = ["date", "Raw price", strat_label]

    base = alt.Chart(df_plot).encode(x=alt.X("date:T", title="Date"))
    line_p = base.mark_line().encode(y=alt.Y("Raw price:Q", title="Price (USD)"))
    line_s = base.mark_line(strokeDash=[6, 3], color="orange").encode(
        y=alt.Y(f"{strat_label}:Q", title="Strategy (USD)", axis=alt.Axis(orient="right"))
    )
    st.altair_chart(line_p + line_s, use_container_width=True)

    # ==========================================
    # 3. FORECAST SECTION
    # ==========================================
    st.divider()
    st.header("🔮 AI Forecast")

    enable_fc = st.checkbox("Show Forecast (Linear Trend + CI)", value=False)
    if enable_fc:
        c_f1, c_f2, c_f3 = st.columns(3)
        with c_f1: horizon_fc = st.slider("Horizon", 5, 120, 20)
        with c_f2: ci_fc = st.selectbox("Confidence", [0.90, 0.95, 0.99], index=1)
        with c_f3: log_fc = st.toggle("Use Log Model", value=True)

        fc = forecast_linear_trend_with_ci(prices, horizon=horizon_fc, ci_level=ci_fc, use_log=log_fc)
        if not fc.empty:
            plot_forecast_with_band(prices, fc)
            last_fc = float(fc["forecast"].iloc[-1])
            st.info(f"Prediction (+{horizon_fc} periods): **{last_fc:.2f} USD**")
        else:
            st.warning("Not enough data.")

    # ==========================================
    # 4. METRICS SECTION
    # ==========================================
    st.divider()
    st.header("📊 Performance Metrics")

    # Utilisation des Tabs pour ne pas surcharger la vue
    tab1, tab2, tab3 = st.tabs(["Asset (Raw)", "Buy & Hold Strategy", f"Active Strategy ({strat_label})"])

    asset_metrics = compute_metrics(prices)
    bh_metrics = compute_metrics(equity_bh)
    strat_metrics = compute_metrics(equity_sel)

    with tab1:
        st.caption("Metrics based on the raw asset price variations.")
        display_metrics_grid(asset_metrics)
    
    with tab2:
        st.caption("Metrics if you bought once and held until today.")
        display_metrics_grid(bh_metrics)
        
    with tab3:
        st.caption(f"Metrics for the **{strat_label}** strategy.")
        display_metrics_grid(strat_metrics)

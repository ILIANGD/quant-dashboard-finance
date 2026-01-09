import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import locale

# Tenter de définir la locale pour le formatage
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    pass

from services.data_loader import load_price_history, load_live_quote


# =========================
# ASSET DATABASE
# =========================

ASSET_DB = {
    "GLOBAL & REGIONAL (ETFs/Indices)": {
        "MSCI World (iShares)": "URTH",
        "MSCI ACWI - All Country (iShares)": "ACWI",
        "MSCI Emerging Markets (iShares)": "EEM",
        "Vanguard Total World Stock": "VT",
        "Euro Stoxx 600": "^STOXX",
        "Vanguard FTSE Europe": "VGK",
        "iShares Asia 50": "AIA"
    },
    "US SECTORS (SPDR ETFs)": {
        "Technology (XLK)": "XLK",
        "Financials (XLF)": "XLF",
        "Healthcare (XLV)": "XLV",
        "Energy (XLE)": "XLE",
        "Semiconductors (SOXX)": "SOXX",
        "Consumer Discretionary (XLY)": "XLY",
        "Consumer Staples (XLP)": "XLP",
        "Utilities (XLU)": "XLU",
        "Industrials (XLI)": "XLI"
    },
    "REAL ESTATE (REITs)": {
        "Vanguard Real Estate (US)": "VNQ",
        "iShares Global REIT": "REET",
        "Simon Property Group": "SPG",
        "Prologis": "PLD",
        "Realty Income": "O"
    },
    "BONDS & RATES": {
        "German 10Y Bund Yield": "^TNX",
        "US Treasury Yield 10 Years": "^TNX",
        "US Treasury Yield 30 Years": "^TYX",
        "US Treasury Yield 5 Years": "^FVX",
        "iShares 20+ Year Treasury Bond (TLT)": "TLT",
        "iShares 7-10 Year Treasury Bond (IEF)": "IEF",
        "iShares Core US Aggregate Bond (AGG)": "AGG"
    },
    "COMMODITIES - AGRI": {
        "Cocoa": "CC=F",
        "Coffee": "KC=F",
        "Corn": "ZC=F",
        "Cotton": "CT=F",
        "Lean Hogs": "HE=F",
        "Live Cattle": "LE=F",
        "Lumber": "LBS=F",
        "Oats": "ZO=F",
        "Orange Juice": "OJ=F",
        "Rough Rice": "ZR=F",
        "Soybean Meal": "ZM=F",
        "Soybean Oil": "ZL=F",
        "Soybeans": "ZS=F",
        "Sugar": "SB=F",
        "Wheat": "ZW=F"
    },
    "COMMODITIES - ENERGY": {
        "Brent Crude Oil": "BZ=F",
        "Gasoline (RBOB)": "RB=F",
        "Heating Oil": "HO=F",
        "Natural Gas": "NG=F",
        "WTI Crude Oil": "CL=F",
        "Uranium (URA ETF)": "URA"
    },
    "COMMODITIES - METALS": {
        "Aluminum": "ALI=F",
        "Copper": "HG=F",
        "Gold": "GC=F",
        "Palladium": "PA=F",
        "Platinum": "PL=F",
        "Silver": "SI=F",
        "Lithium (LIT ETF)": "LIT"
    },
    "CRYPTO": {
        "Avalanche": "AVAX-USD",
        "Binance Coin": "BNB-USD",
        "Bitcoin": "BTC-USD",
        "Bitcoin Cash": "BCH-USD",
        "Cardano": "ADA-USD",
        "Chainlink": "LINK-USD",
        "Dogecoin": "DOGE-USD",
        "Ethereum": "ETH-USD",
        "Litecoin": "LTC-USD",
        "Polkadot": "DOT-USD",
        "Polygon": "MATIC-USD",
        "Shiba Inu": "SHIB-USD",
        "Solana": "SOL-USD",
        "Uniswap": "UNI-USD",
        "XRP": "XRP-USD"
    },
    "FOREX - MAJORS": {
        "AUD/USD": "AUDUSD=X",
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "NZD/USD": "NZDUSD=X",
        "USD/CAD": "CAD=X",
        "USD/CHF": "CHF=X",
        "USD/JPY": "JPY=X"
    },
    "FOREX - CROSSES & EXOTICS": {
        "AUD/JPY": "AUDJPY=X",
        "EUR/GBP": "EURGBP=X",
        "EUR/JPY": "EURJPY=X",
        "EUR/SEK": "EURSEK=X",
        "GBP/JPY": "GBPJPY=X",
        "USD/BRL (Brazil)": "BRL=X",
        "USD/CNY (China)": "CNY=X",
        "USD/HKD (Hong Kong)": "HKD=X",
        "USD/INR (India)": "INR=X",
        "USD/KRW (South Korea)": "KRW=X",
        "USD/MXN (Mexico)": "MXN=X",
        "USD/SGD (Singapore)": "SGD=X",
        "USD/TRY (Turkey)": "TRY=X",
        "USD/ZAR (South Africa)": "ZAR=X"
    },
    "INDICES - AMERICAS": {
        "Bovespa (Brazil)": "^BVSP",
        "Dow Jones 30": "^DJI",
        "IPC Mexico": "^MXX",
        "Merval (Argentina)": "^MERV",
        "Nasdaq 100": "^NDX",
        "NYSE Composite": "^NYA",
        "Russell 2000": "^RUT",
        "S&P 500": "^GSPC",
        "TSX Composite (Canada)": "^GSPTSE",
        "VIX (Volatility)": "^VIX"
    },
    "INDICES - ASIA PACIFIC": {
        "ASX 200 (Australia)": "^AXJO",
        "BSE SENSEX (India)": "^BSESN",
        "Hang Seng (Hong Kong)": "^HSI",
        "KOSPI (South Korea)": "^KS11",
        "Nifty 50 (India)": "^NSEI",
        "Nikkei 225 (Japan)": "^N225",
        "Shanghai Composite": "000001.SS",
        "Shenzhen Component": "399001.SZ",
        "STI (Singapore)": "^STI",
        "TAIEX (Taiwan)": "^TWII"
    },
    "INDICES - EUROPE": {
        "AEX (Netherlands)": "^AEX",
        "BEL 20 (Belgium)": "^BFX",
        "CAC 40 (France)": "^FCHI",
        "DAX (Germany)": "^GDAXI",
        "Euro Stoxx 50": "^STOXX50E",
        "FTSE 100 (UK)": "^FTSE",
        "FTSE MIB (Italy)": "FTSEMIB.MI",
        "IBEX 35 (Spain)": "^IBEX",
        "PSI 20 (Portugal)": "^PSI20",
        "SMI (Switzerland)": "^SSMI"
    },
    "STOCKS - FRANCE (CAC 40)": {
        "Air Liquide": "AI.PA",
        "Airbus": "AIR.PA",
        "AXA": "CS.PA",
        "BNP Paribas": "BNP.PA",
        "Danone": "BN.PA",
        "EssilorLuxottica": "EL.PA",
        "Hermes": "RMS.PA",
        "Kering": "KER.PA",
        "L'Oreal": "OR.PA",
        "LVMH": "MC.PA",
        "Orange": "ORA.PA",
        "Pernod Ricard": "RI.PA",
        "Safran": "SAF.PA",
        "Saint-Gobain": "SGO.PA",
        "Sanofi": "SAN.PA",
        "Schneider Electric": "SU.PA",
        "TotalEnergies": "TTE.PA",
        "Vinci": "DG.PA"
    },
    "STOCKS - US TECH & GIANTS": {
        "AMD": "AMD",
        "Adobe": "ADBE",
        "Airbnb": "ABNB",
        "Alphabet (Google)": "GOOGL",
        "Amazon": "AMZN",
        "Apple": "AAPL",
        "Berkshire Hathaway": "BRK-B",
        "Broadcom": "AVGO",
        "Coca-Cola": "KO",
        "Costco": "COST",
        "Eli Lilly": "LLY",
        "Exxon Mobil": "XOM",
        "JPMorgan Chase": "JPM",
        "Johnson & Johnson": "JNJ",
        "Mastercard": "MA",
        "Meta Platforms": "META",
        "Microsoft": "MSFT",
        "Netflix": "NFLX",
        "Nvidia": "NVDA",
        "Oracle": "ORCL",
        "PepsiCo": "PEP",
        "Pfizer": "PFE",
        "Procter & Gamble": "PG",
        "Salesforce": "CRM",
        "Tesla": "TSLA",
        "Uber": "UBER",
        "Visa": "V",
        "Walmart": "WMT"
    }
}

# Flatten for dropdown
FLAT_ASSETS = {}
for category in sorted(ASSET_DB.keys()):
    items = ASSET_DB[category]
    for name in sorted(items.keys()):
        label = f"{category} | {name} ({items[name]})"
        FLAT_ASSETS[label] = items[name]

TICKER_TO_LABEL = {v: k for k, v in FLAT_ASSETS.items()}


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


def backtest_momentum(prices: pd.Series, lookback: int = 50) -> pd.Series:
    """
    Single Parameter Momentum (Trend Following).
    Logic: Buy if Price > SMA(lookback).
    """
    prices = prices.dropna()
    rets = prices.pct_change().dropna()

    sma = prices.rolling(lookback).mean()
    signal = (prices > sma).astype(int)
    
    position = signal.shift(1).reindex(rets.index).fillna(0)

    strat_rets = position * rets
    equity = (1 + strat_rets).cumprod()
    equity.name = f"Momentum_{lookback}d"
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
# ML Forecast Logic (Improved: Weighted Regression)
# =========================

def train_forecast_model(prices: pd.Series, horizon: int = 20, use_log: bool = True):
    """
    Trains a Weighted Linear Regression model on historical time steps.
    Weights are exponential to prioritize recent data over old history.
    """
    s = prices.dropna()
    if len(s) < 30: return pd.DataFrame(), {}

    # Feature Engineering: X = Integer Time
    y = np.log(s.values) if use_log else s.values
    X = np.arange(len(y)).reshape(-1, 1)

    # --- WEIGHTING LOGIC ---
    # Create exponential weights: recent points get significantly higher weight.
    # geomspace from 0.01 to 1.0 ensures the last point is 100x more important than the first.
    weights = np.geomspace(0.01, 1.0, len(y))

    # Train Model (Weighted Linear Regression)
    model = LinearRegression()
    model.fit(X, y, sample_weight=weights)
    
    # Calculate In-Sample Performance
    y_pred_hist = model.predict(X)
    
    # Compute Metrics on original scale for readability
    if use_log:
        y_true = s.values
        y_pred = np.exp(y_pred_hist)
    else:
        y_true = s.values
        y_pred = y_pred_hist

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # Predict Future
    last_idx = X[-1][0]
    X_future = np.arange(last_idx + 1, last_idx + 1 + horizon).reshape(-1, 1)
    y_future_pred = model.predict(X_future)

    # Confidence Interval
    # We estimate residual std dev based on recent errors to avoid overconfidence from the past
    residuals = y - y_pred_hist
    # Weighted std of residuals? Simple std is safer/more conservative usually.
    std_resid = np.std(residuals[-30:]) # Use last 30 residuals for uncertainty to reflect recent volatility
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
        "mae": mae,
        "mape": mape,
        "model": "Weighted Linear Regression (Trend Following)"
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
        
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1:
            # 1. Retrieve ticker from URL
            url_ticker = st.query_params.get("Asset", "BZ=F")
            
            # Determine if it's a Known Asset or Custom
            if url_ticker in TICKER_TO_LABEL:
                default_index = list(FLAT_ASSETS.values()).index(url_ticker) + 1 # +1 for "Custom" option
                default_custom = ""
            else:
                default_index = 0 # "Custom Ticker..."
                default_custom = url_ticker

            # Dropdown List
            options = ["Custom Ticker..."] + list(FLAT_ASSETS.keys())
            selected_label = st.selectbox("Select Asset", options, index=default_index)

            # Logic: Dropdown vs Custom Input
            if selected_label == "Custom Ticker...":
                ticker = st.text_input("Enter Symbol (Yahoo Finance)", value=default_custom)
            else:
                ticker = FLAT_ASSETS[selected_label]
            
            # Sync to URL
            if ticker and ticker != st.query_params.get("Asset"):
                st.query_params["Asset"] = ticker
            
        with c2:
            period = st.selectbox(
                "Lookback", 
                ["5d", "10d", "15d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], 
                index=6
            )
        
        with c3:
            interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
            interval_label = st.selectbox("Frequency", list(interval_map.keys()), index=0)
            interval = interval_map[interval_label]

        with c4:
            # MODIFIED: Renamed option
            strategy_name = st.selectbox("Backtesting Strategy", ["Buy & Hold", "Momentum", "Breakout"], index=0)

        st.divider()
        
        # MODIFIED: Strategy Params Logic
        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown("**Strategy Settings**")
            
            if "Momentum" in strategy_name:
                # NEW: Single lookback parameter for Momentum
                lookback = st.slider("Momentum Lookback", 5, 200, 50, help="Buy if Price > SMA(Lookback)")
            elif "Breakout" in strategy_name:
                lookback = st.slider("Breakout Lookback", 10, 200, 50, help="Buy if Price > High(Lookback)")
            else:
                st.caption("Standard Buy & Hold strategy (no parameters).")
                lookback = 50 # Default safe value
        
        with ac2:
            st.markdown("**Forecast Settings**")
            fc1, fc2 = st.columns(2)
            with fc1: horizon_fc = st.slider("Horizon (days)", 5, 120, 60)
            with fc2: log_fc = st.toggle("Log-Space Model", value=True)

    # Load Data
    try:
        data = load_price_history(ticker, period=period, interval=interval)
        prices = data["price"].dropna()
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
    except:
        st.error(f"Error loading data for ticker: {ticker}"); return

    if prices.empty: st.warning("No data available."); return

    # Strategies Execution
    equity_bh = backtest_buy_hold(prices)
    # MODIFIED: Call new function
    equity_mom = backtest_momentum(prices, lookback)
    equity_brk = backtest_breakout(prices, lookback)

    # Logic Selection
    if "Momentum" in strategy_name: 
        equity_sel = equity_mom
        strat_label = f"Momentum ({lookback}d)"
    elif "Breakout" in strategy_name: 
        equity_sel = equity_brk
        strat_label = f"Breakout ({lookback}d)"
    else: 
        equity_sel = equity_bh
        strat_label = "Buy & Hold"

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
        st.metric(f"Price ({ticker})", f"${curr:.2f}" if curr else "N/A", delta)
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
    # 3. PERFORMANCE (Chart: Optimized Interactive Cursor)
    # ==========================================
    st.divider()
    st.subheader(f"Price & Strategy Performance ({strat_label})")
    
    df_perf = pd.concat([prices, equity_sel * 100], axis=1).dropna()
    df_perf.columns = ["Asset Price ($)", "Strategy (Base 100)"]
    df_perf = df_perf.reset_index().rename(columns={df_perf.index.name or "index": "date"})

    # --- FLUID INTERACTIVITY LOGIC ---
    base = alt.Chart(df_perf).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%d/%m/%Y", grid=True, gridOpacity=0.3))
    )

    hover = alt.selection_point(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty="none",
        clear="mouseout"
    )

    line_asset = base.mark_line(stroke="#4A90E2", interpolate="monotone").encode(
        y=alt.Y(
            "Asset Price ($):Q",
            title="Asset Price ($)",
            scale=alt.Scale(zero=False),
            axis=alt.Axis(titleColor="#4A90E2", grid=True, gridOpacity=0.3)
        )
    )

    line_strat = base.mark_line(stroke="#FF4B4B", interpolate="monotone").encode(
        y=alt.Y(
            "Strategy (Base 100):Q",
            title="Strategy Performance (Base 100)",
            scale=alt.Scale(zero=False),
            axis=alt.Axis(titleColor="#FF4B4B", orient="right", grid=False)
        )
    )

    cursor = base.mark_rule(color="gray", strokeWidth=1.5).encode(
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%d %B %Y"),
            alt.Tooltip("Asset Price ($):Q", title="Asset Price ($)", format="$.2f"),
            alt.Tooltip("Strategy (Base 100):Q", title="Strategy (Base 100)", format=".2f")
        ]
    ).add_params(hover)

    chart_perf = alt.layer(line_asset, line_strat, cursor).resolve_scale(
        y='independent'
    ).properties(height=450)

    st.altair_chart(chart_perf, use_container_width=True)
    
    st.caption("Blue: Asset Price (Left Scale) | Red: Strategy rebased to 100 (Right Scale)")
    
    st.subheader("Strategy Metrics")
    display_metrics_grid(metrics_strat)

    # ==========================================
    # 4. FORECAST (Updated: Metrics + Fluid Interaction)
    # ==========================================
    st.divider()
    st.header("Forecast Analysis")

    fc_df, fc_metrics = train_forecast_model(prices, horizon_fc, log_fc)

    if not fc_df.empty:
        hist_data = prices.iloc[-252:].to_frame("price").reset_index() 
        hist_data.columns = ["date", "price"]
        
        fc_data = fc_df.reset_index().rename(columns={"index": "date"})
        
        x_axis = alt.X("date:T", axis=alt.Axis(format="%d/%m", title="Date", grid=True, gridOpacity=0.3))
        y_axis = alt.Y("price:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="$f", title="Price ($)", grid=True, gridOpacity=0.3))

        hover_fc = alt.selection_point(
            fields=["date"],
            nearest=True,
            on="mouseover",
            empty="none",
            clear="mouseout"
        )

        c_hist = alt.Chart(hist_data).mark_line().encode(
            x=x_axis, y=y_axis,
            color=alt.value("#4A90E2"), stroke=alt.datum("Historic Price")
        )
        
        c_fc = alt.Chart(fc_data).mark_line(strokeDash=[6, 4]).encode(
            x=x_axis, y=alt.Y("forecast:Q", scale=alt.Scale(zero=False)),
            color=alt.value("#FF4B4B"), stroke=alt.datum("Prediction")
        )
        
        c_band = alt.Chart(fc_data).mark_area(opacity=0.2).encode(
            x=x_axis,
            y=alt.Y("lower:Q", scale=alt.Scale(zero=False)),
            y2="upper:Q",
            color=alt.value("#FF4B4B"), fill=alt.datum("95% Confidence Interval")
        )

        # Combo for Unified Tooltip Rule
        df_h_lite = hist_data.copy()
        df_h_lite["Forecast"] = np.nan
        df_h_lite["Lower"] = np.nan
        df_h_lite["Upper"] = np.nan
        df_h_lite = df_h_lite.rename(columns={"price": "Price"})
        
        df_f_lite = fc_data.copy()
        df_f_lite["Price"] = np.nan
        df_f_lite = df_f_lite.rename(columns={"forecast": "Forecast", "lower": "Lower", "upper": "Upper"})
        
        df_combo = pd.concat([df_h_lite, df_f_lite], ignore_index=True)
        
        rule_fc = alt.Chart(df_combo).mark_rule(color="gray", strokeWidth=1.5).encode(
            x="date:T",
            opacity=alt.condition(hover_fc, alt.value(0.6), alt.value(0)),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%d %B %Y"),
                alt.Tooltip("Price:Q", format="$.2f"),
                alt.Tooltip("Forecast:Q", format="$.2f"),
                alt.Tooltip("Lower:Q", format="$.2f"),
                alt.Tooltip("Upper:Q", format="$.2f"),
            ]
        ).add_params(hover_fc)

        chart = alt.layer(c_hist, c_band, c_fc, rule_fc).properties(height=500).configure_legend(
            orient='bottom', title=None, labelFontSize=12,
        ).configure_axis(grid=True, gridOpacity=0.3)
        
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Model Performance")
        fp1, fp2, fp3, fp4, fp5 = st.columns(5)
        with fp1: st.metric("MAE", f"${fc_metrics['mae']:.2f}", help="Mean Absolute Error")
        with fp2: st.metric("MAPE", f"{fc_metrics['mape']:.2%}", help="Mean Absolute Percentage Error")
        with fp3: st.metric("RMSE", f"${fc_metrics['rmse']:.2f}", help="Root Mean Squared Error")
        with fp4: st.metric("R² Score", f"{fc_metrics['r2']:.3f}")
        with fp5: st.metric("Model", fc_metrics['model'])
            
        st.markdown(f"**Prediction (+{horizon_fc}d):** ${fc_df['forecast'].iloc[-1]:.2f}")

    else:
        st.warning("Not enough data to train model.")

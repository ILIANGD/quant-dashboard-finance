import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import altair as alt

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

    rolling_high = prices.shift(1).rolling(lookback).max()
    signal = (prices > rolling_high).astype(int)

    position = signal.shift(1).reindex(rets.index).fillna(0)
    strat_rets = position * rets

    equity = (1 + strat_rets).cumprod()
    equity.name = f"Breakout_{lookback}d"
    return equity


# =========================
# Metrics
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


# =========================
# Forecast
# =========================

def _infer_pandas_freq(index: pd.DatetimeIndex) -> str:
    freq = pd.infer_freq(index)
    if freq:
        return freq
    # fallback
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

    - Model: y(t) = a + b*t (+ noise)
    - y is log(price) if use_log else price
    - CI is approximate using residual std and normal quantile.
    """
    s = prices.dropna().copy()
    if len(s) < 20:
        return pd.DataFrame()

    y = np.log(s.values) if use_log else s.values
    t = np.arange(len(y), dtype=float)

    # Fit line (least squares)
    b, a = np.polyfit(t, y, deg=1)  # y ≈ b*t + a
    y_hat = a + b * t
    resid = y - y_hat
    sigma = float(np.std(resid, ddof=2)) if len(resid) > 2 else 0.0

    # Normal quantile (approx)
    # for 95%: z≈1.96 ; 90%: 1.645 ; 99%: 2.576
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

    out = pd.DataFrame(
        {"forecast": pred, "lower": lo, "upper": hi},
        index=future_index,
    )
    return out


def plot_forecast_with_band(history: pd.Series, fc: pd.DataFrame):
    hist_df = history.dropna().to_frame("price").reset_index()
    hist_df.columns = ["date", "price"]
    hist_df["type"] = "History"

    fc_df = fc.reset_index()
    fc_df.columns = ["date", "forecast", "lower", "upper"]
    fc_df["type"] = "Forecast"

    # Line for history
    line_hist = (
        alt.Chart(hist_df)
        .mark_line()
        .encode(x="date:T", y="price:Q")
    )

    # Band + line for forecast
    band = (
        alt.Chart(fc_df)
        .mark_area(opacity=0.2)
        .encode(x="date:T", y="lower:Q", y2="upper:Q")
    )
    line_fc = (
        alt.Chart(fc_df)
        .mark_line(strokeDash=[4, 4])
        .encode(x="date:T", y="forecast:Q")
    )

    st.altair_chart(line_hist + band + line_fc, use_container_width=True)


# =========================
# Streamlit page
# =========================

def single_asset_page():
    st.title("Single Asset Analysis – Brent (BZ=F)")

    # Auto refresh every 5 minutes
    st_autorefresh(interval=300_000, key="auto_refresh_5min")

    # Controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ticker = st.text_input("Ticker", value="BZ=F")

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

    st.caption(
    f"Frequency meaning: each point corresponds to one {interval} observation "
    f"used both for the price series and for strategy decisions."
    )

    with col4:
        strategy_name = st.selectbox(
            "Strategy",
            ["Buy & Hold", "Momentum (Fast MA / Slow MA)", "Breakout (N-day high)"],
            index=0,
        )

    # Live quote
    try:
        quote = load_live_quote(ticker)
    except Exception:
        quote = {"last_price": None, "asof_utc": "N/A"}

    c1, c2 = st.columns(2)
    with c1:
        if quote["last_price"] is None:
            st.metric("Current price", "N/A")
        else:
            st.metric("Current price", f"{quote['last_price']:.2f} USD")
    with c2:
        st.caption(f"Last update: {quote.get('asof_utc', 'N/A')}")

    # Strategy parameters
    fast, slow = 20, 50
    if "Momentum" in strategy_name:
        p1, p2 = st.columns(2)
        with p1:
            fast = st.slider("Fast MA window (days)", 5, 50, 20, 1)
        with p2:
            slow = st.slider("Slow MA window (days)", 20, 200, 50, 1)
        if slow <= fast:
            st.warning("Slow MA window must be strictly greater than fast MA window.")
            return

    lookback = 50
    if "Breakout" in strategy_name:
        lookback = st.slider(
            "Breakout lookback (days)",
            min_value=10,
            max_value=200,
            value=50,
            step=5,
            help="Long when price breaks above the previous N-day high, otherwise cash.",
        )

    # Load history
    try:
        data = load_price_history(ticker, period=period, interval=interval)
    except Exception:
        data = pd.DataFrame()

    if data.empty or "price" not in data.columns:
        st.error("Failed to load data for this ticker.")
        return

    prices = data["price"].dropna()
    if prices.empty:
        st.error("No usable price data for this ticker.")
        return

    # Backtests
    equity_bh = backtest_buy_hold(prices)
    equity_mom = backtest_momentum_ma(prices, fast=fast, slow=slow) if "Momentum" in strategy_name else None
    equity_break = backtest_breakout(prices, lookback=lookback) if "Breakout" in strategy_name else None

    # Choose strategy curve for the main plot
    if equity_mom is not None:
        equity_sel = equity_mom.copy()
        strat_label = "Momentum Strategy"
    elif equity_break is not None:
        equity_sel = equity_break.copy()
        strat_label = f"Breakout Strategy ({lookback}d)"
    else:
        equity_sel = equity_bh.copy()
        strat_label = "Buy & Hold Strategy"

    st.subheader(
    f"Raw price (USD) vs {strat_label}",
    help=(
        "Each curve is built from discrete observations (points) "
        "at the selected frequency. "
        "The strategy curve reflects cumulative portfolio value."
        )
    )    

    price_raw = prices.copy()
    if isinstance(price_raw, pd.DataFrame):
        price_raw = price_raw.iloc[:, 0]
    if isinstance(equity_sel, pd.DataFrame):
        equity_sel = equity_sel.iloc[:, 0]

    # Strategy value in USD scale
    strategy_value = equity_sel * float(price_raw.iloc[0])

    price_raw.name = "Raw price"
    strategy_value.name = strat_label

    df_plot = pd.concat([price_raw, strategy_value], axis=1).dropna()
    st.line_chart(df_plot)

    st.caption(
    "Each point represents one observation at the selected frequency "
    "(daily, weekly, or monthly). "
    "The raw price curve shows the observed market price in USD, "
    "while the strategy curve shows the simulated portfolio value "
    "based on trading signals applied at each observation date."
    )

    # =========================
    # Forecast
    # =========================
    st.subheader("Forecast")

    enable_fc = st.checkbox("Enable forecast", value=False)
    if enable_fc:
        f1, f2, f3 = st.columns(3)
        with f1:
            horizon = st.slider("Forecast horizon (points)", 5, 120, 20, 5)
        with f2:
            ci_level = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1)
        with f3:
            use_log = st.selectbox("Model space", ["Log-price (recommended)", "Price"], index=0) == "Log-price (recommended)"

        fc = forecast_linear_trend_with_ci(prices, horizon=horizon, ci_level=ci_level, use_log=use_log)
        if fc.empty:
            st.warning("Not enough data to compute a forecast (need at least ~20 points).")
        else:
            plot_forecast_with_band(prices, fc)
            last_fc = float(fc["forecast"].iloc[-1])
            st.caption(f"Last forecast point: {last_fc:.2f} USD  |  CI{int(ci_level*100)}%: [{fc['lower'].iloc[-1]:.2f}, {fc['upper'].iloc[-1]:.2f}] USD")

    st.caption(
    "Performance metrics are computed from the same series shown above. "
    "Returns are calculated between consecutive points of the selected frequency."
    )
    
    # Metrics
    st.subheader("Performance metrics")

    asset_metrics = compute_metrics(prices)
    bh_metrics = compute_metrics(equity_bh)
    mom_metrics = compute_metrics(equity_mom) if equity_mom is not None else None
    break_metrics = compute_metrics(equity_break) if equity_break is not None else None

    st.markdown(f"### Asset ({ticker})")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        metric_with_help("Annual return", f"{asset_metrics.get('annual_return', float('nan')) * 100:.2f}%")
    with a2:
        metric_with_help("Annual volatility", f"{asset_metrics.get('annual_vol', float('nan')) * 100:.2f}%")
    with a3:
        metric_with_help("Sharpe", f"{asset_metrics.get('sharpe', float('nan')):.2f}")
    with a4:
        metric_with_help("Max drawdown", f"{asset_metrics.get('max_drawdown', float('nan')) * 100:.2f}%")

    a5, a6, a7, _ = st.columns(4)
    with a5:
        metric_with_help("Calmar", f"{asset_metrics.get('calmar', float('nan')):.2f}")
    with a6:
        metric_with_help("Sortino", f"{asset_metrics.get('sortino', float('nan')):.2f}")
    with a7:
        metric_with_help("Daily mean return", f"{asset_metrics.get('mean_daily', float('nan')) * 100:.3f}%")

    st.markdown("### Buy & Hold Strategy")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        metric_with_help("Annual return", f"{bh_metrics.get('annual_return', float('nan')) * 100:.2f}%")
    with b2:
        metric_with_help("Annual volatility", f"{bh_metrics.get('annual_vol', float('nan')) * 100:.2f}%")
    with b3:
        metric_with_help("Sharpe", f"{bh_metrics.get('sharpe', float('nan')):.2f}")
    with b4:
        metric_with_help("Max drawdown", f"{bh_metrics.get('max_drawdown', float('nan')) * 100:.2f}%")

    b5, b6, b7, _ = st.columns(4)
    with b5:
        metric_with_help("Calmar", f"{bh_metrics.get('calmar', float('nan')):.2f}")
    with b6:
        metric_with_help("Sortino", f"{bh_metrics.get('sortino', float('nan')):.2f}")
    with b7:
        metric_with_help("Daily mean return", f"{bh_metrics.get('mean_daily', float('nan')) * 100:.3f}%")

    if mom_metrics is not None:
        st.markdown("### Momentum Strategy")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_with_help("Annual return", f"{mom_metrics.get('annual_return', float('nan')) * 100:.2f}%")
        with m2:
            metric_with_help("Annual volatility", f"{mom_metrics.get('annual_vol', float('nan')) * 100:.2f}%")
        with m3:
            metric_with_help("Sharpe", f"{mom_metrics.get('sharpe', float('nan')):.2f}")
        with m4:
            metric_with_help("Max drawdown", f"{mom_metrics.get('max_drawdown', float('nan')) * 100:.2f}%")

        m5, m6, m7, _ = st.columns(4)
        with m5:
            metric_with_help("Calmar", f"{mom_metrics.get('calmar', float('nan')):.2f}")
        with m6:
            metric_with_help("Sortino", f"{mom_metrics.get('sortino', float('nan')):.2f}")
        with m7:
            metric_with_help("Daily mean return", f"{mom_metrics.get('mean_daily', float('nan')) * 100:.3f}%")

    if break_metrics is not None:
        st.markdown("### Breakout Strategy")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric_with_help("Annual return", f"{break_metrics.get('annual_return', float('nan')) * 100:.2f}%")
        with k2:
            metric_with_help("Annual volatility", f"{break_metrics.get('annual_vol', float('nan')) * 100:.2f}%")
        with k3:
            metric_with_help("Sharpe", f"{break_metrics.get('sharpe', float('nan')):.2f}")
        with k4:
            metric_with_help("Max drawdown", f"{break_metrics.get('max_drawdown', float('nan')) * 100:.2f}%")

        k5, k6, k7, _ = st.columns(4)
        with k5:
            metric_with_help("Calmar", f"{break_metrics.get('calmar', float('nan')):.2f}")
        with k6:
            metric_with_help("Sortino", f"{break_metrics.get('sortino', float('nan')):.2f}")
        with k7:
            metric_with_help("Daily mean return", f"{break_metrics.get('mean_daily', float('nan')) * 100:.3f}%")

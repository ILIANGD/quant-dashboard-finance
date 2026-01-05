import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import altair as alt

from services.data_loader import load_price_history, load_live_quote


# =========================
# Tooltips (help ?)
# =========================

PORTFOLIO_TOOLTIPS = {
    "Tickers": "Comma-separated tickers (>= 3). Example: BZ=F, GC=F, HG=F.",
    "Lookback period": "How far back to download historical data.",
    "Frequency": "Sampling frequency used for prices, returns, and rebalancing.",
    "Allocation rule": "Equal weight gives the same weight to each asset. Custom weights are normalized to sum to 1.",
    "Rebalancing frequency": "How often the portfolio is reset to the target weights.",
    "Initial capital": "Starting portfolio value in USD.",
    "Main chart": "Left axis: asset prices. Right axis: portfolio cumulative value (USD).",
    "Normalized comparison": "All series rebased to 100 at the start date (relative comparison).",
    "Portfolio annual return": "Annualized return from daily compounded returns (252 trading days).",
    "Portfolio annual volatility": "Annualized volatility: std(daily returns) × sqrt(252).",
    "Portfolio Sharpe": "Sharpe ratio assuming risk-free rate = 0.",
    "Portfolio max drawdown": "Maximum peak-to-trough decline of portfolio value.",
    "Weighted avg asset vol": "Weighted average of individual asset volatilities (no diversification).",
    "Diversification effect": "1 − (portfolio volatility / weighted average asset volatility).",
    "Correlation matrix": "Pairwise correlations of aligned asset returns at the selected frequency.",
    "Weights used": "Weights after normalization (sum to 1).",
}


def metric_with_help(label: str, value_str: str):
    st.caption(f"**{label}**", help=PORTFOLIO_TOOLTIPS.get(label, ""))
    st.markdown(
        f"<div style='font-size:1.3em; font-weight:600;'>{value_str}</div>",
        unsafe_allow_html=True,
    )


# =========================
# Helpers
# =========================

def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _normalize_weights(w: dict) -> dict:
    """Keep only non-negative numbers, normalize to sum to 1."""
    ww = {}
    for k, v in w.items():
        try:
            vv = float(v)
        except Exception:
            continue
        if vv >= 0:
            ww[k] = vv

    s = float(sum(ww.values()))
    if s <= 0:
        if not ww:
            return {}
        n = len(ww)
        return {k: 1.0 / n for k in ww}

    return {k: v / s for k, v in ww.items()}


def _align_prices(price_dict: dict) -> pd.DataFrame:
    """
    price_dict: {ticker: Series}
    Align on common dates, keep rows where all assets exist.
    """
    df = pd.concat(price_dict, axis=1)
    df.columns = list(price_dict.keys())
    df = df.sort_index()
    df = df.dropna(how="any")
    return df


def rebalance_dates(index: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
    """
    rule: "Daily", "Weekly", "Monthly"
    Returns dates in index at which we rebalance.
    """
    if rule == "Daily":
        return pd.DatetimeIndex(index)

    s = pd.Series(index=index, data=index)

    if rule == "Weekly":
        key = s.index.to_period("W").astype(str)
        return pd.DatetimeIndex(s.groupby(key).first().values)

    if rule == "Monthly":
        key = s.index.to_period("M").astype(str)
        return pd.DatetimeIndex(s.groupby(key).first().values)

    return pd.DatetimeIndex(index)


# =========================
# Portfolio simulation
# =========================

def simulate_portfolio(
    prices_df: pd.DataFrame,
    weights: dict,
    rebal_rule: str = "Monthly",
    initial_capital: float = 10_000.0,
) -> pd.Series:
    """
    - compute returns
    - hold weights until next rebalance date (piecewise-constant weights)
    - portfolio return each period = sum_i w_i * r_i
    """
    prices_df = prices_df.dropna()
    if prices_df.empty:
        return pd.Series(dtype=float)

    rets = prices_df.pct_change().dropna()
    if rets.empty:
        return pd.Series(dtype=float)

    tickers = list(prices_df.columns)

    w = _normalize_weights({t: weights.get(t, 0.0) for t in tickers})
    if not w:
        w = {t: 1.0 / len(tickers) for t in tickers}

    w0 = np.array([w.get(t, 0.0) for t in tickers], dtype=float)

    rdates = set(rebalance_dates(rets.index, rebal_rule))

    current_w = w0.copy()
    port_rets = []

    for dt, row in rets.iterrows():
        if dt in rdates:
            current_w = w0.copy()

        pr = float(np.dot(current_w, row.values))
        port_rets.append(pr)

    port_rets = pd.Series(port_rets, index=rets.index, name="portfolio_ret")
    equity = (1.0 + port_rets).cumprod() * float(initial_capital)
    equity.name = "Portfolio value (USD)"
    return equity


# =========================
# Metrics
# =========================

def compute_metrics(series: pd.Series | pd.DataFrame) -> dict:
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

    annual_return = (1.0 + mean_daily) ** 252 - 1.0
    annual_vol = vol_daily * np.sqrt(252.0)

    sharpe = float("nan")
    if (annual_vol is not None) and np.isfinite(annual_vol) and annual_vol != 0:
        sharpe = annual_return / annual_vol

    cum = (1.0 + rets).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1.0
    max_dd = float(drawdown.min())

    calmar = float("nan") if max_dd == 0 else (annual_return / abs(max_dd))

    downside = rets[rets < 0].std() * np.sqrt(252.0)
    sortino = annual_return / downside if (downside is not None and np.isfinite(downside) and downside != 0) else float("nan")

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


# =========================
# Plots
# =========================

def plot_multi_assets_and_portfolio(prices_df: pd.DataFrame, portfolio_value: pd.Series):
    """
    Requirement:
    - multiple asset prices together
    - cumulative value of portfolio
    """
    prices_df = prices_df.dropna()
    portfolio_value = portfolio_value.dropna()

    common = prices_df.index.intersection(portfolio_value.index)
    prices_df = prices_df.loc[common]
    portfolio_value = portfolio_value.loc[common]

    idx_name = prices_df.index.name or "index"

    long_prices = (
        prices_df.reset_index()
        .melt(id_vars=idx_name, var_name="ticker", value_name="price")
        .rename(columns={idx_name: "date"})
    )

    port_df = portfolio_value.to_frame("portfolio").reset_index()
    port_df = port_df.rename(columns={port_df.columns[0]: "date"})

    base_x = alt.X("date:T", title="Date")

    price_lines = (
        alt.Chart(long_prices)
        .mark_line()
        .encode(
            x=base_x,
            y=alt.Y("price:Q", title="Asset price (USD)"),
            color=alt.Color("ticker:N", title="Assets"),
            tooltip=["date:T", "ticker:N", alt.Tooltip("price:Q", format=".2f")],
        )
    )

    port_line = (
        alt.Chart(port_df)
        .mark_line(strokeDash=[6, 3])
        .encode(
            x=base_x,
            y=alt.Y("portfolio:Q", title="Portfolio value (USD)", axis=alt.Axis(orient="right")),
            tooltip=["date:T", alt.Tooltip("portfolio:Q", format=".2f")],
        )
    )

    st.altair_chart(price_lines + port_line, use_container_width=True)


def plot_corr_heatmap(corr: pd.DataFrame):
    if corr is None or corr.empty:
        return

    corr_reset = corr.reset_index()
    first_col = corr_reset.columns[0]  # "index" if unnamed
    corr_long = (
        corr_reset.melt(id_vars=first_col, var_name="asset2", value_name="corr")
        .rename(columns={first_col: "asset1"})
    )

    heat = (
        alt.Chart(corr_long)
        .mark_rect()
        .encode(
            x=alt.X("asset2:N", title=""),
            y=alt.Y("asset1:N", title=""),
            color=alt.Color("corr:Q"),
            tooltip=["asset1:N", "asset2:N", alt.Tooltip("corr:Q", format=".3f")],
        )
    )
    st.altair_chart(heat, use_container_width=True)


# =========================
# Streamlit page
# =========================

def portfolio_page():
    st.title("Portfolio Module – Multi-Asset Portfolio")

    st_autorefresh(interval=300_000, key="auto_refresh_5min_portfolio")

    DEFAULT_TICKERS = ["BZ=F", "GC=F", "HG=F"]  # Brent, Gold, Copper
    DEFAULT_WEIGHTS = {"BZ=F": 0.40, "GC=F": 0.35, "HG=F": 0.25}

    # ---- Controls ----
    c1, c2, c3 = st.columns(3)

    with c1:
        tickers_txt = st.text_input(
            "Tickers (comma-separated, >= 3)",
            value=", ".join(DEFAULT_TICKERS),
            help=PORTFOLIO_TOOLTIPS["Tickers"],
        )

    with c2:
        period = st.selectbox(
            "Lookback period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            help=PORTFOLIO_TOOLTIPS["Lookback period"],
        )

    with c3:
        interval = st.selectbox(
            "Frequency",
            ["1d", "1wk", "1mo"],
            index=0,
            help=PORTFOLIO_TOOLTIPS["Frequency"],
        )

    st.caption(
        f"Point meaning: each point corresponds to one {interval} observation, used for prices, returns, and rebalancing."
    )

    tickers = [t.strip() for t in tickers_txt.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))

    if len(tickers) < 3:
        st.error("Please provide at least 3 tickers.")
        return

    # ---- Portfolio parameters ----
    st.subheader("Portfolio parameters", help="Set allocation and rebalancing rules for the portfolio.")

    p1, p2, p3 = st.columns(3)
    with p1:
        allocation_rule = st.selectbox(
            "Allocation rule",
            ["Equal weight", "Custom weights"],
            index=1,
            help=PORTFOLIO_TOOLTIPS["Allocation rule"],
        )
    with p2:
        rebal_rule = st.selectbox(
            "Rebalancing frequency",
            ["Daily", "Weekly", "Monthly"],
            index=2,
            help=PORTFOLIO_TOOLTIPS["Rebalancing frequency"],
        )
    with p3:
        initial_capital = st.number_input(
            "Initial capital (USD)",
            min_value=100.0,
            value=10_000.0,
            step=500.0,
            help=PORTFOLIO_TOOLTIPS["Initial capital"],
        )

    # weights input
    if allocation_rule == "Equal weight":
        weights = {t: 1.0 / len(tickers) for t in tickers}
        st.caption(f"Equal weight: {1.0/len(tickers):.3f} per asset.")
    else:
        st.caption("Custom weights are normalized to sum to 1.")
        cols = st.columns(len(tickers))
        weights = {}
        for i, t in enumerate(tickers):
            default_w = float(DEFAULT_WEIGHTS.get(t, 0.0))
            with cols[i]:
                weights[t] = st.number_input(
                    f"{t}",
                    min_value=0.0,
                    value=default_w,
                    step=0.05,
                    help="Weight before normalization.",
                )

    weights = _normalize_weights(weights)
    if not weights:
        st.error("Invalid weights (sum must be > 0).")
        return

    # ---- Live quotes ----
    st.subheader("Current values", help="Latest available quote from yfinance (best-effort).")

    qcols = st.columns(min(3, len(tickers)))
    for i, t in enumerate(tickers):
        with qcols[i % len(qcols)]:
            try:
                q = load_live_quote(t)
                last = _safe_float(q.get("last_price"))
                if last is None:
                    st.metric(f"{t}", "N/A")
                else:
                    st.metric(f"{t}", f"{last:.2f} USD")
                st.caption(f"Last update: {q.get('asof_utc', 'N/A')}")
            except Exception:
                st.metric(f"{t}", "N/A")
                st.caption("Last update: N/A")

    # ---- Load historical prices ----
    st.subheader("Historical data & portfolio simulation", help="Download prices, align dates across assets, then simulate the portfolio.")

    price_series = {}
    failed = []

    for t in tickers:
        try:
            df = load_price_history(t, period=period, interval=interval)
            if df is None or df.empty or "price" not in df.columns:
                failed.append(t)
                continue
            s = df["price"].dropna()
            if s.empty:
                failed.append(t)
                continue
            s.name = t
            price_series[t] = s
        except Exception:
            failed.append(t)

    if failed:
        st.warning(f"Some tickers failed to load and will be excluded: {', '.join(failed)}")

    if len(price_series) < 3:
        st.error("Need at least 3 assets with valid data to build the portfolio.")
        return

    prices_df = _align_prices(price_series)
    if prices_df.empty or prices_df.shape[0] < 30:
        st.error("Not enough aligned data across assets. Try a longer lookback period or different tickers.")
        return

    # keep weights only for assets present
    weights = _normalize_weights({t: weights.get(t, 0.0) for t in prices_df.columns})
    if not weights:
        st.error("Weights invalid after excluding failed tickers.")
        return

    # simulate
    port_value = simulate_portfolio(
        prices_df=prices_df,
        weights=weights,
        rebal_rule=rebal_rule,
        initial_capital=float(initial_capital),
    )
    if port_value.empty:
        st.error("Portfolio simulation failed (no returns).")
        return

    # ---- Main chart ----
    st.subheader(
        "Main chart: asset prices + portfolio cumulative value",
        help=PORTFOLIO_TOOLTIPS["Main chart"],
    )
    plot_multi_assets_and_portfolio(prices_df, port_value)

    st.caption(
        "Curves: multiple asset prices (USD) + the cumulative portfolio value (USD). "
        "Portfolio value starts from the initial capital and is rebalanced at the chosen frequency."
    )

    # ---- Normalized comparison ----
    st.subheader(
        "Comparison view: normalized assets vs portfolio (base 100)",
        help=PORTFOLIO_TOOLTIPS["Normalized comparison"],
    )

    common = prices_df.index.intersection(port_value.index)
    norm_assets = (prices_df.loc[common] / prices_df.loc[common].iloc[0]) * 100.0
    norm_port = (port_value.loc[common] / port_value.loc[common].iloc[0]) * 100.0

    idx_name = norm_assets.index.name or "index"

    norm_long = (
        norm_assets.reset_index()
        .melt(id_vars=idx_name, var_name="series", value_name="value")
        .rename(columns={idx_name: "date"})
    )

    port_long = norm_port.to_frame("value").reset_index()
    port_long = port_long.rename(columns={port_long.columns[0]: "date"})
    port_long["series"] = "Portfolio (base 100)"

    norm_all = pd.concat([norm_long, port_long], ignore_index=True)

    comp = (
        alt.Chart(norm_all)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Normalized value (base 100)"),
            color=alt.Color("series:N", title="Series"),
            tooltip=["date:T", "series:N", alt.Tooltip("value:Q", format=".2f")],
        )
    )
    st.altair_chart(comp, use_container_width=True)

    # ---- Metrics ----
    st.subheader("Portfolio metrics", help="Key risk/return metrics computed from the portfolio value series.")

    port_metrics = compute_metrics(port_value)

    # correlation
    rets = prices_df.pct_change().dropna()
    corr = rets.corr()

    m1, m2, m3 = st.columns(3)

    with m1:
        metric_with_help("Portfolio annual return", f"{port_metrics.get('annual_return', float('nan'))*100:.2f}%")
        metric_with_help("Portfolio annual volatility", f"{port_metrics.get('annual_vol', float('nan'))*100:.2f}%")

    with m2:
        metric_with_help("Portfolio Sharpe", f"{port_metrics.get('sharpe', float('nan')):.2f}")
        metric_with_help("Portfolio max drawdown", f"{port_metrics.get('max_drawdown', float('nan'))*100:.2f}%")

    vol_assets = rets.std() * np.sqrt(252.0)
    w_vec = np.array([weights.get(t, 0.0) for t in prices_df.columns], dtype=float)
    weighted_avg_vol = float(np.dot(w_vec, vol_assets.values)) if len(vol_assets.values) else float("nan")
    portfolio_vol = float(port_metrics.get("annual_vol", float("nan")))

    with m3:
        metric_with_help("Weighted avg asset vol", f"{weighted_avg_vol*100:.2f}%")
        if np.isfinite(portfolio_vol) and np.isfinite(weighted_avg_vol) and weighted_avg_vol > 0:
            div_effect = 1.0 - (portfolio_vol / weighted_avg_vol)
            metric_with_help("Diversification effect", f"{div_effect*100:.2f}%")
        else:
            metric_with_help("Diversification effect", "N/A")

    st.markdown("### Correlation matrix")
    st.caption("**Correlation matrix**", help=PORTFOLIO_TOOLTIPS["Correlation matrix"])
    plot_corr_heatmap(corr)

    st.caption(
        "Diversification effect (rough): 1 − (portfolio volatility / weighted average of individual volatilities). "
        "Correlation matrix computed from aligned returns at the selected frequency."
    )

    # ---- Weights ----
    st.subheader("Weights used (normalized)", help=PORTFOLIO_TOOLTIPS["Weights used"])

    wdf = (
        pd.DataFrame({"ticker": list(weights.keys()), "weight": list(weights.values())})
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    wdf["weight_%"] = (wdf["weight"] * 100.0).round(2)

    st.dataframe(wdf[["ticker", "weight_%"]], hide_index=True)

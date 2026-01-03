import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from services.data_loader import load_price_history, load_live_quote

# ---------- STRATÉGIES DE BACKTEST ----------

def backtest_buy_hold(prices: pd.Series) -> pd.Series:
    """Stratégie Buy & Hold : on achète au début, on garde tout le temps."""
    prices = prices.dropna()
    rets = prices.pct_change().dropna()
    equity = (1 + rets).cumprod()
    equity.name = "Buy & Hold"
    return equity


def backtest_momentum_ma(prices: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """
    Stratégie Momentum simple :
    - MA rapide > MA lente -> investi (position = 1)
    - Sinon -> cash (position = 0)
    """
    prices = prices.dropna()
    rets = prices.pct_change().dropna()

    ma_fast = prices.rolling(fast).mean()
    ma_slow = prices.rolling(slow).mean()

    # Signal 1 si MA_fast > MA_slow, sinon 0
    signal = (ma_fast > ma_slow).astype(int)

    # Position du jour = signal de la veille (pas de look-ahead)
    position = signal.shift(1).reindex(rets.index).fillna(0)

    strat_rets = position * rets
    equity = (1 + strat_rets).cumprod()
    equity.name = "Momentum"
    return equity


def backtest_breakout(prices: pd.Series, lookback: int = 50) -> pd.Series:
    """
    Stratégie Breakout N jours :
    - Long si le prix dépasse le plus haut des N derniers jours
    - Sinon, cash
    """
    prices = prices.dropna()
    rets = prices.pct_change().dropna()

    # Plus haut sur les N derniers jours (hors jour courant)
    rolling_high = prices.shift(1).rolling(lookback).max()

    # Signal : 1 si prix du jour > plus haut N jours précédent, sinon 0
    signal = (prices > rolling_high).astype(int)

    # Position du jour = signal de la veille (pour éviter le look-ahead)
    position = signal.shift(1).reindex(rets.index).fillna(0)

    strat_rets = position * rets
    equity = (1 + strat_rets).cumprod()
    equity.name = f"Breakout_{lookback}d"
    return equity


# ---------- MÉTRIQUES ----------

def compute_metrics(series: pd.Series | pd.DataFrame) -> dict:
    """Compute key performance metrics for an asset or strategy."""
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    series = series.dropna()
    rets = series.pct_change().dropna()
    if rets.empty:
        return {}

    # Basic stats
    mean_daily = float(rets.mean())
    vol_daily = float(rets.std())

    # Annualization (252 = trading days)
    annual_return = (1 + mean_daily) ** 252 - 1
    annual_vol = vol_daily * (252 ** 0.5)

    # Sharpe ratio
    if annual_vol == 0 or pd.isna(annual_vol):
        sharpe = float("nan")
    else:
        sharpe = annual_return / annual_vol

    # Drawdown
    cum = (1 + rets).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = float(drawdown.min())

    # Calmar = return / max_drawdown
    calmar = float("nan") if max_dd == 0 else annual_return / abs(max_dd)

    # Sortino: downside-only volatility
    downside = rets[rets < 0].std() * (252 ** 0.5)
    sortino = annual_return / downside if downside != 0 else float("nan")

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


# ---------- PAGE STREAMLIT ----------

METRIC_TOOLTIPS = {
    "Annual return": (
        "Annualized return computed from daily compounded returns over 252 trading days."
    ),
    "Annual volatility": (
        "Annualized volatility: standard deviation of daily returns multiplied by sqrt(252)."
    ),
    "Sharpe": (
        "Sharpe ratio (risk-free rate assumed 0 here): annual return divided by annual volatility."
    ),
    "Max drawdown": (
        "Maximum drawdown: largest peak-to-trough decline (in %) of the equity curve."
    ),
    "Calmar": (
        "Calmar ratio: annual return divided by the absolute value of max drawdown."
    ),
    "Sortino": (
        "Sortino ratio: annual return divided by downside volatility (only negative returns)."
    ),
    "Daily mean return": (
        "Average daily return (mean of daily returns)."
    ),
}


def metric_with_help(label: str, value_str: str):
    tooltip = METRIC_TOOLTIPS.get(label, "")

    # Affiche le label AVEC le petit (i) automatique de Streamlit
    st.caption(f"**{label}**", help=tooltip)

    # Affiche la valeur juste en dessous
    st.markdown(
        f"<div style='font-size:1.4em; font-weight:600;'>{value_str}</div>",
        unsafe_allow_html=True,
    )


def single_asset_page():
    st.title("Single Asset Analysis – Brent (BZ=F)")

    # Refresh automatique toutes les 5 minutes (300 000 ms)
    st_autorefresh(interval=300_000, key="auto_refresh_5min")

    # ---- Contrôles interactifs (ticker, période, périodicité, stratégie) ----
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

    with col4:
        strategy_name = st.selectbox(
            "Strategy",
            ["Buy & Hold", "Momentum (Fast MA / Slow MA)", "Breakout (N-day high)"],
            index=0,
        )


    quote = load_live_quote(ticker)

    c1, c2 = st.columns(2)
    with c1:
        if quote["last_price"] is None:
            st.metric("Current price", "N/A")
        else:
            st.metric("Current price", f"{quote['last_price']:.2f}")
    
    with c2:
        st.caption(f"Last update: {quote['asof_utc']}")
        
    # ---- Paramètres de stratégie (Momentum) ----
    fast, slow = 20, 50
    if "Momentum" in strategy_name:
        c_fast, c_slow = st.columns(2)
        with c_fast:
            fast = st.slider("Fast MA window (days)", min_value=5, max_value=50, value=20, step=1)
        with c_slow:
            slow = st.slider("Slow MA window (days)", min_value=20, max_value=200, value=50, step=1)
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

    # ---- Backtest automatique (plus de bouton) ----

    # Load data
    data = load_price_history(ticker, period=period, interval=interval)
    if data.empty:
        st.error("Failed to load data for this ticker.")
        return

    prices = data["price"].dropna()
    if prices.empty:
        st.error("No usable price data for this ticker.")
        return

    # Stratégies
    equity_bh = backtest_buy_hold(prices)

    equity_mom = None
    equity_break = None
    
    if "Momentum" in strategy_name:
        equity_mom = backtest_momentum_ma(prices, fast=fast, slow=slow)
    
    if "Breakout" in strategy_name:
        equity_break = backtest_breakout(prices, lookback=lookback)

    # ---------- GRAPHIQUE PRINCIPAL ----------
    # Choix de la stratégie affichée
    if "Momentum" in strategy_name and equity_mom is not None:
        equity_sel = equity_mom.copy()
        strat_label = "Momentum Strategy"
    elif "Breakout" in strategy_name and equity_break is not None:
        equity_sel = equity_break.copy()
        strat_label = f"Breakout Strategy ({lookback}d)"
    else:
        equity_sel = equity_bh.copy()
        strat_label = "Buy & Hold Strategy"

    st.subheader(f"Raw price vs {strat_label}")

    # Raw price
    price_raw = prices.copy()

    # Sécurité : si jamais ce sont des DataFrame
    if isinstance(price_raw, pd.DataFrame):
        price_raw = price_raw.iloc[:, 0]
    if isinstance(equity_sel, pd.DataFrame):
        equity_sel = equity_sel.iloc[:, 0]

    # On exprime la stratégie en 'valeur' comparable au prix :
    # equity_sel commence à 1 -> on lui donne comme valeur initiale le prix initial
    strategy_value = equity_sel * float(price_raw.iloc[0])

    # Noms des séries pour la légende du graphe
    price_raw.name = "Raw price"
    strategy_value.name = strat_label

    # DataFrame final pour le graphe, index = dates
    df_plot = pd.concat([price_raw, strategy_value], axis=1).dropna()

    # Affichage : deux courbes sur le même graphe
    st.line_chart(df_plot)

    # ---------- MÉTRIQUES ----------
    st.subheader("Performance metrics")

    asset_metrics = compute_metrics(prices)
    bh_metrics = compute_metrics(equity_bh)
    mom_metrics = compute_metrics(equity_mom) if equity_mom is not None else None
    break_metrics = compute_metrics(equity_break) if equity_break is not None else None

    # ----- Actif -----
    st.markdown(f"### Asset ({ticker})")
    ca1, ca2, ca3, ca4 = st.columns(4)
    with ca1:
        metric_with_help("Annual return", f"{asset_metrics['annual_return'] * 100:.2f}%")
    with ca2:
        metric_with_help("Annual volatility", f"{asset_metrics['annual_vol'] * 100:.2f}%")
    with ca3:
        metric_with_help("Sharpe", f"{asset_metrics['sharpe']:.2f}")
    with ca4:
        metric_with_help("Max drawdown", f"{asset_metrics['max_drawdown'] * 100:.2f}%")

    ca5, ca6, ca7, _ = st.columns(4)
    with ca5:
        metric_with_help("Calmar", f"{asset_metrics['calmar']:.2f}")
    with ca6:
        metric_with_help("Sortino", f"{asset_metrics['sortino']:.2f}")
    with ca7:
        metric_with_help("Daily mean return", f"{asset_metrics['mean_daily'] * 100:.3f}%")

    # ----- Stratégie Buy & Hold -----
    st.markdown("### Buy & Hold Strategy")
    cb1, cb2, cb3, cb4 = st.columns(4)
    with cb1:
        metric_with_help("Annual return", f"{bh_metrics['annual_return'] * 100:.2f}%")
    with cb2:
        metric_with_help("Annual volatility", f"{bh_metrics['annual_vol'] * 100:.2f}%")
    with cb3:
        metric_with_help("Sharpe", f"{bh_metrics['sharpe']:.2f}")
    with cb4:
        metric_with_help("Max drawdown", f"{bh_metrics['max_drawdown'] * 100:.2f}%")

    cb5, cb6, cb7, _ = st.columns(4)
    with cb5:
        metric_with_help("Calmar", f"{bh_metrics['calmar']:.2f}")
    with cb6:
        metric_with_help("Sortino", f"{bh_metrics['sortino']:.2f}")
    with cb7:
        metric_with_help("Daily mean return", f"{bh_metrics['mean_daily'] * 100:.3f}%")

    # ----- Stratégie Momentum (si présente) -----
    if mom_metrics is not None:
        st.markdown("### Momentum Strategy")
        cm1, cm2, cm3, cm4 = st.columns(4)
        with cm1:
            metric_with_help("Annual return", f"{mom_metrics['annual_return'] * 100:.2f}%")
        with cm2:
            metric_with_help("Annual volatility", f"{mom_metrics['annual_vol'] * 100:.2f}%")
        with cm3:
            metric_with_help("Sharpe", f"{mom_metrics['sharpe']:.2f}")
        with cm4:
            metric_with_help("Max drawdown", f"{mom_metrics['max_drawdown'] * 100:.2f}%")

        cm5, cm6, cm7, _ = st.columns(4)
        with cm5:
            metric_with_help("Calmar", f"{mom_metrics['calmar']:.2f}")
        with cm6:
            metric_with_help("Sortino", f"{mom_metrics['sortino']:.2f}")
        with cm7:
            metric_with_help("Daily mean return", f"{mom_metrics['mean_daily'] * 100:.3f}%")


    if break_metrics is not None:
        st.markdown("### Breakout Strategy")
    
        bx1, bx2, bx3, bx4 = st.columns(4)
        with bx1:
            metric_with_help("Annual return", f"{break_metrics['annual_return'] * 100:.2f}%")
        with bx2:
            metric_with_help("Annual volatility", f"{break_metrics['annual_vol'] * 100:.2f}%")
        with bx3:
            metric_with_help("Sharpe", f"{break_metrics['sharpe']:.2f}")
        with bx4:
            metric_with_help("Max drawdown", f"{break_metrics['max_drawdown'] * 100:.2f}%")
    
        bx5, bx6, bx7, _ = st.columns(4)
        with bx5:
            metric_with_help("Calmar", f"{break_metrics['calmar']:.2f}")
        with bx6:
            metric_with_help("Sortino", f"{break_metrics['sortino']:.2f}")
        with bx7:
            metric_with_help("Daily mean return", f"{break_metrics['mean_daily'] * 100:.3f}%")

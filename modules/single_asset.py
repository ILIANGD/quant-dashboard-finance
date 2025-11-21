import streamlit as st
import pandas as pd
from services.data_loader import load_price_history


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


# ---------- MÉTRIQUES ----------

def compute_metrics(series: pd.Series | pd.DataFrame) -> dict:
    """
    Calcule des métriques standard (rendement annuel, vol, Sharpe, max drawdown)
    à partir d'une série de valeurs (prix ou equity de stratégie).
    """
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    series = series.dropna()
    rets = series.pct_change().dropna()
    if rets.empty:
        return {}

    mean_ret = float(rets.mean())
    vol = float(rets.std())

    if vol == 0 or pd.isna(vol):
        sharpe = float("nan")
    else:
        sharpe = (mean_ret * 252) / (vol * (252 ** 0.5))

    cum = (1 + rets).cumprod()
    running_max = cum.cummax()
    drawdown = (cum / running_max - 1)
    max_dd = float(drawdown.min())

    return {
        "annual_return": (1 + mean_ret) ** 252 - 1,
        "annual_vol": vol * (252 ** 0.5),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


# ---------- PAGE STREAMLIT ----------

def single_asset_page():
    st.title("Single Asset Analysis – Brent (BZ=F)")

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Ticker", value="BZ=F")
    with col2:
        period = st.selectbox(
            "Historique",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
        )
    with col3:
        strategy_name = st.selectbox(
            "Stratégie",
            ["Buy & Hold", "Momentum (MA rapide / MA lente)"],
            index=0,
        )

    fast, slow = 20, 50
    if "Momentum" in strategy_name:
        c_fast, c_slow = st.columns(2)
        with c_fast:
            fast = st.slider("MA rapide (jours)", min_value=5, max_value=50, value=20, step=1)
        with c_slow:
            slow = st.slider("MA lente (jours)", min_value=20, max_value=200, value=50, step=1)
        if slow <= fast:
            st.warning("La MA lente doit être strictement plus grande que la MA rapide.")
            return

    if st.button("Lancer le backtest"):
        data = load_price_history(ticker, period=period, interval="1d")
        if data.empty:
            st.error("Impossible de charger les données pour ce ticker.")
            return

        prices = data["price"].dropna()

        # Prix normalisé pour comparaison (1 au départ)
        price_norm = prices / prices.iloc[0]
        price_norm.name = "Prix normalisé"

        # Stratégies
        equity_bh = backtest_buy_hold(prices)
        equity_mom = None
        if "Momentum" in strategy_name:
            equity_mom = backtest_momentum_ma(prices, fast=fast, slow=slow)

        # ---------- GRAPHIQUE PRINCIPAL ----------
        st.subheader("Évolution du prix et des stratégies")

        df_plot = pd.concat([price_norm, equity_bh], axis=1)
        if equity_mom is not None:
            df_plot = pd.concat([df_plot, equity_mom], axis=1)

        st.line_chart(df_plot)

        # ---------- MÉTRIQUES ----------
        st.subheader("Métriques")

        asset_metrics = compute_metrics(prices)
        bh_metrics = compute_metrics(equity_bh)
        mom_metrics = compute_metrics(equity_mom) if equity_mom is not None else None

        st.markdown("**Actif (Brent)**")
        ca1, ca2, ca3, ca4 = st.columns(4)
        ca1.metric("Rendement annuel", f"{asset_metrics['annual_return']*100:.2f}%")
        ca2.metric("Vol annuel", f"{asset_metrics['annual_vol']*100:.2f}%")
        ca3.metric("Sharpe", f"{asset_metrics['sharpe']:.2f}")
        ca4.metric("Max drawdown", f"{asset_metrics['max_drawdown']*100:.2f}%")

        st.markdown("**Stratégie Buy & Hold**")
        cb1, cb2, cb3, cb4 = st.columns(4)
        cb1.metric("Rendement annuel", f"{bh_metrics['annual_return']*100:.2f}%")
        cb2.metric("Vol annuel", f"{bh_metrics['annual_vol']*100:.2f}%")
        cb3.metric("Sharpe", f"{bh_metrics['sharpe']:.2f}")
        cb4.metric("Max drawdown", f"{bh_metrics['max_drawdown']*100:.2f}%")

        if mom_metrics is not None:
            st.markdown("**Stratégie Momentum**")
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Rendement annuel", f"{mom_metrics['annual_return']*100:.2f}%")
            cm2.metric("Vol annuel", f"{mom_metrics['annual_vol']*100:.2f}%")
            cm3.metric("Sharpe", f"{mom_metrics['sharpe']:.2f}")
            cm4.metric("Max drawdown", f"{mom_metrics['max_drawdown']*100:.2f}%")
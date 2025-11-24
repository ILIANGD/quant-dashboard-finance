import streamlit as st
import pandas as pd
from services.data_loader import load_price_history
import altair as alt


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

def single_asset_page():
    st.title("Single Asset Analysis – Brent (BZ=F)")

    # ---- Contrôles interactifs (ticker, période, périodicité, stratégie) ----
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ticker = st.text_input("Ticker", value="BZ=F")

    with col2:
        period = st.selectbox(
            "Historique",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
        )

    with col3:
        interval = st.selectbox(
            "Périodicité",
            ["1d", "1wk", "1mo"],
            index=0,
            help="1d = daily, 1wk = weekly, 1mo = monthly",
        )

    with col4:
        strategy_name = st.selectbox(
            "Stratégie",
            ["Buy & Hold", "Momentum (MA rapide / MA lente)"],
            index=0,
        )

    # Paramètres de stratégie (Momentum)
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

    # ---- Lancement du backtest ----
    if st.button("Lancer le backtest"):
        data = load_price_history(ticker, period=period, interval=interval)
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
        st.subheader("Prix brut vs valeur cumulée de la stratégie")

        # Courbe 1 : prix brut (non normalisé)
        price_raw = prices.copy()

        # Courbe 2 : stratégie sélectionnée (valeur cumulée)
        if "Momentum" in strategy_name and equity_mom is not None:
            equity_sel = equity_mom.copy()
            strat_label = "Stratégie Momentum"
        else:
            equity_sel = equity_bh.copy()
            strat_label = "Stratégie Buy & Hold"

        # S'assurer que ce sont bien des Series
        if isinstance(price_raw, pd.DataFrame):
            price_raw = price_raw.iloc[:, 0]
        if isinstance(equity_sel, pd.DataFrame):
            equity_sel = equity_sel.iloc[:, 0]

        # Renommer proprement pour le chart
        price_raw = price_raw.rename("price")
        equity_sel = equity_sel.rename("strategy_value")

        df_plot = pd.concat([price_raw, equity_sel], axis=1).dropna()

        # Préparation pour Altair
        df_plot_reset = df_plot.reset_index().rename(columns={"index": "date"})

        import altair as alt

        price_line = (
            alt.Chart(df_plot_reset)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("price:Q", title="Prix brut"),
                tooltip=["date:T", "price:Q"],
            )
        )

        equity_line = (
            alt.Chart(df_plot_reset)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y(
                    "strategy_value:Q",
                    title="Valeur cumulée de la stratégie",
                    axis=alt.Axis(titleColor="red"),
                ),
                color=alt.value("red"),
                tooltip=["date:T", "strategy_value:Q"],
            )
        )

        chart = (
            alt.layer(price_line, equity_line)
            .resolve_scale(y="independent")  # deux axes Y indépendants
            .properties(
                width=900,
                height=400,
                title=f"Prix brut vs {strat_label}",
            )
        )

        st.altair_chart(chart, use_container_width=True)

        # ---------- MÉTRIQUES ----------
        st.subheader("Métriques")

        asset_metrics = compute_metrics(prices)
        bh_metrics = compute_metrics(equity_bh)
        mom_metrics = compute_metrics(equity_mom) if equity_mom is not None else None

        # ----- Actif -----
        st.markdown("### Actif (Brent)")
        ca1, ca2, ca3, ca4 = st.columns(4)
        ca1.metric("Rendement annuel", f"{asset_metrics['annual_return'] * 100:.2f}%")
        ca2.metric("Vol annuel", f"{asset_metrics['annual_vol'] * 100:.2f}%")
        ca3.metric("Sharpe", f"{asset_metrics['sharpe']:.2f}")
        ca4.metric("Max drawdown", f"{asset_metrics['max_drawdown'] * 100:.2f}%")

        ca5, ca6, ca7, _ = st.columns(4)
        ca5.metric("Calmar", f"{asset_metrics['calmar']:.2f}")
        ca6.metric("Sortino", f"{asset_metrics['sortino']:.2f}")
        ca7.metric("Rendement/jour", f"{asset_metrics['mean_daily'] * 100:.3f}%")

        # ----- Stratégie Buy & Hold -----
        st.markdown("### Stratégie Buy & Hold")
        cb1, cb2, cb3, cb4 = st.columns(4)
        cb1.metric("Rendement annuel", f"{bh_metrics['annual_return'] * 100:.2f}%")
        cb2.metric("Vol annuel", f"{bh_metrics['annual_vol'] * 100:.2f}%")
        cb3.metric("Sharpe", f"{bh_metrics['sharpe']:.2f}")
        cb4.metric("Max drawdown", f"{bh_metrics['max_drawdown'] * 100:.2f}%")

        cb5, cb6, cb7, _ = st.columns(4)
        cb5.metric("Calmar", f"{bh_metrics['calmar']:.2f}")
        cb6.metric("Sortino", f"{bh_metrics['sortino']:.2f}")
        cb7.metric("Rendement/jour", f"{bh_metrics['mean_daily'] * 100:.3f}%")

        # ----- Stratégie Momentum (si présente) -----
        if mom_metrics is not None:
            st.markdown("### Stratégie Momentum")

            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Rendement annuel", f"{mom_metrics['annual_return'] * 100:.2f}%")
            cm2.metric("Vol annuel", f"{mom_metrics['annual_vol'] * 100:.2f}%")
            cm3.metric("Sharpe", f"{mom_metrics['sharpe']:.2f}")
            cm4.metric("Max drawdown", f"{mom_metrics['max_drawdown'] * 100:.2f}%")

            cm5, cm6, cm7, _ = st.columns(4)
            cm5.metric("Calmar", f"{mom_metrics['calmar']:.2f}")
            cm6.metric("Sortino", f"{mom_metrics['sortino']:.2f}")
            cm7.metric("Rendement/jour", f"{mom_metrics['mean_daily'] * 100:.3f}%")

import streamlit as st
import pandas as pd
from services.data_loader import load_price_history

def compute_metrics(prices: pd.Series) -> dict:
    returns = prices.pct_change().dropna()
    if returns.empty:
        return {}

    mean_ret = returns.mean()
    vol = returns.std()
    sharpe = (mean_ret * 252) / (vol * (252 ** 0.5)) if vol > 0 else float("nan")

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum / running_max - 1)
    max_dd = drawdown.min()

    return {
        "annual_return": (1 + mean_ret) ** 252 - 1,
        "annual_vol": vol * (252 ** 0.5),
        "sharpe": sharpe,
        "max_drawdown": max_dd
    }

def single_asset_page():
    st.title("Single Asset Analysis")

    col1, col2 = st.columns(2)
    with col1:
        # Brent = "BZ=F"
        ticker = st.text_input("Ticker", value="BZ=F")
    with col2:
        period = st.selectbox(
            "Historique",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )

    if st.button("Charger les données"):
        data = load_price_history(ticker, period=period, interval="1d")
        if data.empty:
            st.error("Impossible de charger les données pour ce ticker.")
            return

        st.subheader("Prix")
        st.line_chart(data["price"])

        metrics = compute_metrics(data["price"])
        if metrics:
            st.subheader("Métriques (approx.)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rendement annuel", f"{metrics['annual_return']*100:.2f}%")
            c2.metric("Vol annuel", f"{metrics['annual_vol']*100:.2f}%")
            c3.metric("Sharpe", f"{metrics['sharpe']:.2f}")
            c4.metric("Max drawdown", f"{metrics['max_drawdown']*100:.2f}%")
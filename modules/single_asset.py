import streamlit as st
from services.data_loader import load_price

def single_asset_page():
    st.title("Single Asset Analysis")

    ticker = st.text_input("Enter a ticker (example: AAPL, BTC-USD, EURUSD=X)", "AAPL")

    if st.button("Load Data"):
        data = load_price(ticker)

        if data.empty:
            st.error("Impossible de charger les données.")
        else:
            st.line_chart(data)
            st.success("Data loaded successfully!")
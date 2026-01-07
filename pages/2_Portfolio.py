import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from modules.portfolio import portfolio_page (Tu décommenteras ça quand Quant B aura fini)

st.set_page_config(page_title="Portfolio Management", layout="wide")

st.title("Construction du Portefeuille")
st.info("Ce module est en cours de développement.")

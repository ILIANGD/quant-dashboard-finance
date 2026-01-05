import streamlit as st

from modules.single_asset import single_asset_page

# Portfolio module (optional for now)
try:
    from modules.portfolio import portfolio_page
    HAS_PORTFOLIO = True
except Exception:
    portfolio_page = None
    HAS_PORTFOLIO = False


st.set_page_config(page_title="Quant Dashboard", layout="wide")

st.sidebar.title("Navigation")

pages = ["Single Asset"]
if HAS_PORTFOLIO:
    pages.append("Portfolio")

page = st.sidebar.radio("Module", pages)

if page == "Single Asset":
    single_asset_page()
elif page == "Portfolio" and HAS_PORTFOLIO:
    portfolio_page()
else:
    st.error("Module not available.")

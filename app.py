# app.py
import streamlit as st

st.set_page_config(page_title="Quant Dashboard", layout="wide")

# ---- Pages ----
from modules.single_asset import single_asset_page
from modules.portfolio import portfolio_page


def home_page():
    st.title("Quant Dashboard")
    st.write("Choose a module above: Single Asset or Portfolio.")


# ---- Top banner navigation ----
with st.container():
    left, mid, right = st.columns([1, 6, 1])
    with mid:
        b1, b2, b3 = st.columns(3)

        if b1.button("Home", use_container_width=True):
            st.session_state["page"] = "Home"
        if b2.button("Single Asset", use_container_width=True):
            st.session_state["page"] = "Single Asset"
        if b3.button("Portfolio", use_container_width=True):
            st.session_state["page"] = "Portfolio"

# Default page
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

st.divider()

# ---- Render selected page ----
page = st.session_state["page"]

if page == "Home":
    home_page()
elif page == "Single Asset":
    single_asset_page()
elif page == "Portfolio":
    portfolio_page()
else:
    home_page()

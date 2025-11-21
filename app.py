import streamlit as st
from modules.single_asset import single_asset_page

st.set_page_config(page_title="Quant Dashboard", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Module", ["Single Asset"])  # plus tard: ["Single Asset", "Portfolio"]

if page == "Single Asset":
    single_asset_page()
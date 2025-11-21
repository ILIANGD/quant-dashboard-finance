import streamlit as st
from modules.single_asset import single_asset_page

st.set_page_config(page_title="Quant Dashboard", layout="wide")

page = st.sidebar.radio("Navigation", ["Single Asset"])

if page == "Single Asset":
    single_asset_page()
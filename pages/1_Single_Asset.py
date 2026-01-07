import streamlit as st
import sys
import os

# On ajoute le dossier parent au "chemin" pour que Python trouve le dossier 'modules'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Maintenant on peut importer ta fonction proprement
from modules.single_asset import single_asset_page

# Config de la page
st.set_page_config(page_title="Single Asset Analysis", layout="wide")

# Lancement
single_asset_page()

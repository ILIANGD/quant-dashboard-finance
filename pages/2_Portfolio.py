import streamlit as st
import sys
from pathlib import Path

# 1. On ajoute le dossier racine au chemin pour que Python trouve "modules"
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

# 2. On importe la fonction principale depuis le module qu'on vient de créer
from modules.portfolio import portfolio_page

# 3. Configuration de la page
st.set_page_config(
    page_title="Portfolio Manager",
    layout="wide"
)

# 4. Lancement de la page
if __name__ == "__main__":
    portfolio_page()

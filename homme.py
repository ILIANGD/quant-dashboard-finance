import streamlit as st

st.set_page_config(
    page_title="Quant Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Quant Research Dashboard")

st.markdown("""
### Bienvenue

Sélectionnez un module dans la barre latérale gauche pour commencer :

* **Single Asset** : Analyse univariée et stratégies.
* **Portfolio** : Gestion de portefeuille.
""")

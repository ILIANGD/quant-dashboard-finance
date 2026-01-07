import streamlit as st

st.set_page_config(
    page_title="Quant Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Quant Research Dashboard")

st.markdown("### Bienvenue sur la plateforme de recherche quantitative.")
st.info("Sélectionnez un module ci-dessous pour démarrer.")

# Création de deux colonnes pour les boutons
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 1. Analyse Univariée")
    # Ce bouton redirige vers la page Single Asset
    st.page_link("pages/1_Single_Asset.py", label="Aller vers Single Asset", icon="📈", use_container_width=True)

with col2:
    st.markdown("#### 2. Gestion de Portefeuille")
    # Ce bouton redirige vers la page Portfolio
    st.page_link("pages/2_Portfolio.py", label="Aller vers Portfolio", icon="💼", use_container_width=True)

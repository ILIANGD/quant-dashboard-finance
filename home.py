import streamlit as st
import pandas as pd
from services.data_loader import load_live_quote

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Quant Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CSS Styling (Clean Professional Look)
# ==========================================
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-weight: 700;
        letter-spacing: -1px;
    }
    h3 {
        font-weight: 600;
        color: #4A90E2;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# Market Pulse Data
# ==========================================
MARKET_TICKERS = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "CAC 40": "^FCHI",
    "Gold": "GC=F",
    "Brent Oil": "BZ=F",
    "Bitcoin": "BTC-USD",
    "EUR/USD": "EURUSD=X"
}

def display_market_pulse():
    st.subheader("Global Market Pulse")
    
    # Create 7 columns for the tickers
    cols = st.columns(len(MARKET_TICKERS))
    
    for i, (name, ticker) in enumerate(MARKET_TICKERS.items()):
        with cols[i]:
            try:
                data = load_live_quote(ticker)
                price = data.get("last_price")
                prev = data.get("prev_close")
                
                if price and prev:
                    delta = price - prev
                    delta_pct = (delta / prev)
                    
                    st.metric(
                        label=name,
                        value=f"{price:,.2f}",
                        delta=f"{delta_pct:.2%}"
                    )
                else:
                    st.metric(label=name, value="N/A")
            except:
                st.metric(label=name, value="Error")
    
    st.markdown("---")

# ==========================================
# Main Content
# ==========================================

def home_page():
    # Hero Section
    st.title("Quantitative Finance Dashboard")
    st.markdown(
        """
        **Deploy institutional-grade analytics to identify market trends, forecast asset prices using machine learning, 
        and engineer diversified portfolios with advanced risk management.**
        
        This platform integrates real-time data, algorithmic backtesting, and predictive modeling to support 
        data-driven investment decisions.
        """
    )
    
    st.markdown("---")
    
    # 1. Market Overview
    display_market_pulse()
    
    # 2. Module Navigation Cards
    st.subheader("Analytics Modules")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.subheader("Single Asset Analysis")
            st.markdown(
                """
                Deep dive into individual financial instruments.
                
                **Key Features:**
                * **Backtesting Engines:** Buy & Hold, Momentum (MA Crossover), Volatility Breakout.
                * **AI Forecasting:** Weighted Linear Regression trends with confidence intervals.
                * **Risk Metrics:** Sharpe Ratio, Sortino, Max Drawdown, Calmar Ratio.
                * **Interactive Charts:** Zoomable price history and strategy performance curves.
                """
            )
            st.info("Navigate to 'Single Asset' in the sidebar to begin.")

    with col2:
        with st.container(border=True):
            st.subheader("Portfolio Management")
            st.markdown(
                """
                Construct, simulate, and optimize multi-asset portfolios.
                
                **Key Features:**
                * **Asset Allocation:** Support for equal-weight or custom capital allocation.
                * **Rebalancing:** Simulate Daily, Weekly, or Monthly rebalancing strategies.
                * **Correlation Analysis:** Heatmaps to identify diversification opportunities.
                * **Relative Performance:** Compare portfolio growth against underlying assets (Base 100).
                """
            )
            st.info("Navigate to 'Portfolio' in the sidebar to begin.")

    # 3. Footer / Disclaimer
    st.markdown("---")
    st.caption(
        """
        **Disclaimer:** This dashboard is for informational and educational purposes only. 
        It does not constitute financial advice. Past performance is not indicative of future results. 
        Market data is provided by Yahoo Finance and may be delayed.
        """
    )

if __name__ == "__main__":
    home_page()

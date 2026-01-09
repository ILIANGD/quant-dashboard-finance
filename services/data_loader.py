import pandas as pd
import yfinance as yf
import streamlit as st
import logging
from datetime import datetime, timezone

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600, show_spinner=False)
def load_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Downloads historical data from Yahoo Finance.
    Cached for 1 hour (3600s) to reduce latency.
    Returns a DataFrame with columns: ['price', 'open', 'high', 'low', 'volume']
    """
    try:
        # auto_adjust=True récupère les prix ajustés (dividendes/splits)
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)

        if df.empty:
            logger.warning(f"No data found for ticker: {ticker}")
            return pd.DataFrame()

        # 1. Gestion des MultiIndex (yfinance renvoie parfois ('Close', 'AAPL'))
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Renommage standardisé
        # On mappe "Close" vers "price" pour l'uniformité du projet
        df = df.rename(columns={
            "Close": "price", 
            "Open": "open", 
            "High": "high", 
            "Low": "low", 
            "Volume": "volume"
        })

        # 3. Suppression de la Timezone (Crucial pour Altair)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 4. Validation
        if "price" not in df.columns:
            return pd.DataFrame()

        # On retourne les colonnes utiles
        cols = [c for c in ["price", "open", "high", "low", "volume"] if c in df.columns]
        return df[cols].dropna()

    except Exception as e:
        logger.error(f"Error loading history for {ticker}: {e}")
        return pd.DataFrame()


def load_live_quote(ticker: str) -> dict:
    """
    Returns a live-ish quote (last price), previous close, and timestamp.
    Not cached to ensure real-time accuracy.
    """
    t = yf.Ticker(ticker)
    last = None
    prev = None

    # 1) Try fast_info (quickest way)
    try:
        fi = getattr(t, "fast_info", {}) or {}
        last = fi.get("last_price", None)
        prev = fi.get("previous_close", None)
        
        # Validation des types
        if last is not None: last = float(last)
        if prev is not None: prev = float(prev)
    except Exception:
        last = None
        prev = None

    # 2) Fallback: Intraday 1m history
    if last is None:
        try:
            df = t.history(period="1d", interval="1m")
            if not df.empty and "Close" in df.columns:
                last = float(df["Close"].dropna().iloc[-1])
        except Exception:
            pass

    # 3) Fallback / Ensure Prev Close: Daily history
    if last is None or prev is None:
        try:
            df_daily = t.history(period="5d", interval="1d")
            if not df_daily.empty and "Close" in df_daily.columns:
                closes = df_daily["Close"].dropna()
                if len(closes) >= 1 and last is None:
                    last = float(closes.iloc[-1])
                
                if len(closes) >= 2 and prev is None:
                    prev = float(closes.iloc[-2])
                elif len(closes) >= 1 and prev is None:
                    prev = float(closes.iloc[-1])
        except Exception:
            pass

    return {
        "last_price": last,
        "prev_close": prev,
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

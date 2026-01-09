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
    
    CRITICAL FIX: Always downloads '1d' data and resamples locally to avoid 
    sparse data issues (straight lines) with Yahoo Finance on Futures/Forex 
    when requesting '1wk' or '1mo'.
    """
    try:
        # 1. FORCE '1d' interval regardless of what is requested to ensure data density
        # auto_adjust=True récupère les prix ajustés (dividendes/splits)
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)

        if df.empty:
            logger.warning(f"No data found for ticker: {ticker}")
            return pd.DataFrame()

        # 2. Gestion des MultiIndex (yfinance renvoie parfois ('Close', 'AAPL'))
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 3. Renommage standardisé
        # On mappe "Close" vers "price" pour l'uniformité du projet
        df = df.rename(columns={
            "Close": "price", 
            "Open": "open", 
            "High": "high", 
            "Low": "low", 
            "Volume": "volume"
        })

        # 4. Suppression de la Timezone (Crucial pour Altair)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 5. Validation de base
        if "price" not in df.columns:
            return pd.DataFrame()

        # 6. RESAMPLING MANUEL (La correction magique) 
        # Si l'utilisateur voulait du Weekly ou Monthly, on le calcule nous-mêmes proprement.
        if interval == "1wk":
            # On prend le dernier prix de la semaine (Vendredi)
            # 'W-FRI' assure que la date affichée est la fin de semaine
            df = df.resample("W-FRI").last().dropna()
        elif interval == "1mo":
            # On prend le dernier prix du mois
            try:
                df = df.resample("ME").last().dropna() # Pandas >= 2.2
            except:
                df = df.resample("M").last().dropna()  # Pandas < 2.2

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

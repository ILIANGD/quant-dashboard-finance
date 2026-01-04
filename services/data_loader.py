import yfinance as yf
import pandas as pd
from datetime import datetime, timezone


def load_live_quote(ticker: str) -> dict:
    """
    Returns a live-ish quote (last price) + timestamp.
    Robust fallback: if no intraday last price, return last daily close.
    """
    t = yf.Ticker(ticker)
    last = None

    # 1) Try fast_info (quick, but can be missing)
    try:
        fi = getattr(t, "fast_info", {}) or {}
        last = fi.get("last_price", None)
        if last is not None:
            last = float(last)
    except Exception:
        last = None

    # 2) Try intraday 1m
    if last is None:
        try:
            df = t.history(period="1d", interval="1m")
            if isinstance(df, pd.DataFrame) and (not df.empty) and ("Close" in df.columns):
                last = float(df["Close"].dropna().iloc[-1])
        except Exception:
            last = None

    # 3) Fallback: last daily close
    if last is None:
        try:
            df = t.history(period="5d", interval="1d")
            if isinstance(df, pd.DataFrame) and (not df.empty) and ("Close" in df.columns):
                last = float(df["Close"].dropna().iloc[-1])
        except Exception:
            last = None

    return {
        "last_price": last,
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def load_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Returns historical prices with a 'price' column (Close).
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Handle multi-index columns sometimes returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame({"price": df["Close"]})
    out.index = pd.to_datetime(out.index)
    return out.dropna()

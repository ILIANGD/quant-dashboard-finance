import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

def load_live_quote(ticker: str) -> dict:
    """
    Returns a live-ish quote (last price) + timestamp.
    yfinance is dynamic (prices update intraday).
    """
    t = yf.Ticker(ticker)

    # Try fast_info first (often quickest)
    last = None
    try:
        fi = getattr(t, "fast_info", {})
        last = fi.get("last_price", None)
    except Exception:
        last = None

    # Fallback: 1-minute data (most reliable)
    if last is None:
        df = t.history(period="1d", interval="1m")
        if not df.empty:
            last = float(df["Close"].iloc[-1])

    return {
        "last_price": last,
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

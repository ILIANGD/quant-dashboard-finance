import yfinance as yf
import pandas as pd
from datetime import datetime, timezone


def load_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Download historical prices from Yahoo Finance via yfinance.

    Returns a DataFrame with:
      - index = DatetimeIndex
      - column 'price' = Close price (float)
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance returns columns like 'Open','High','Low','Close','Adj Close','Volume'
    if "Close" not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["price"] = df["Close"].astype(float)

    # Clean timezone (Streamlit likes naive timestamps)
    try:
        out.index = pd.to_datetime(out.index).tz_localize(None)
    except Exception:
        pass

    return out


def load_live_quote(ticker: str) -> dict:
    """
    Returns a live-ish quote (last price) + timestamp.
    yfinance is dynamic (prices update intraday).
    """
    t = yf.Ticker(ticker)

    last = None

    # Try fast_info first (often quickest)
    try:
        fi = getattr(t, "fast_info", {})
        last = fi.get("last_price", None)
    except Exception:
        last = None

    # Fallback: 1-minute data (more reliable)
    if last is None:
        try:
            df = t.history(period="1d", interval="1m")
            if df is not None and not df.empty and "Close" in df.columns:
                last = float(df["Close"].iloc[-1])
        except Exception:
            last = None

    return {
        "last_price": last,
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

import yfinance as yf
import pandas as pd
from datetime import datetime, timezone


def load_price_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download historical prices from Yahoo Finance via yfinance.

    Parameters
    ----------
    ticker : str
        Asset ticker (e.g. 'BZ=F', 'AAPL', 'EURUSD=X')
    period : str
        Lookback period (e.g. '1mo', '6mo', '1y', '5y')
    interval : str
        Data frequency ('1d', '1wk', '1mo')

    Returns
    -------
    pd.DataFrame
        - index : DatetimeIndex (timezone-naive)
        - column 'price' : close price (float)
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if "Close" not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["price"] = pd.to_numeric(df["Close"], errors="coerce")

    # Ensure clean datetime index (Streamlit / Altair friendly)
    try:
        out.index = pd.to_datetime(out.index).tz_localize(None)
    except Exception:
        pass

    return out.dropna()


def load_live_quote(ticker: str) -> dict:
    """
    Retrieve a live-ish quote (last traded price) and timestamp.

    Uses:
    - fast_info if available
    - fallback to 1-minute intraday data

    Returns
    -------
    dict
        {
            'last_price': float | None,
            'asof_utc': str
        }
    """
    last_price = None

    try:
        t = yf.Ticker(ticker)
    except Exception:
        return {
            "last_price": None,
            "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    # 1) fast_info (fastest, but not always present)
    try:
        fi = getattr(t, "fast_info", {})
        last_price = fi.get("last_price", None)
    except Exception:
        last_price = None

    # 2) fallback: intraday 1-minute data
    if last_price is None:
        try:
            df = t.history(period="1d", interval="1m")
            if df is not None and not df.empty and "Close" in df.columns:
                last_price = float(df["Close"].iloc[-1])
        except Exception:
            last_price = None

    return {
        "last_price": last_price,
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

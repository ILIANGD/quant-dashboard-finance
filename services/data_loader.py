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

    # yfinance sometimes returns MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["price"] = pd.to_numeric(df["Close"], errors="coerce")

    # Ensure clean datetime index (Streamlit / Altair friendly)
    try:
        out.index = pd.to_datetime(out.index).tz_localize(None)
    except Exception:
        # If already tz-naive or conversion fails, keep as-is
        pass

    return out.dropna()


def load_live_quote(ticker: str) -> dict:
    """
    Returns a live-ish quote (last price) + timestamp.

    Robust fallback:
    1) fast_info last_price
    2) intraday 1-minute close (last available)
    3) last daily close (last 5 days)
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
                close_series = df["Close"].dropna()
                if not close_series.empty:
                    last = float(close_series.iloc[-1])
        except Exception:
            last = None

    # 3) Fallback: last daily close
    if last is None:
        try:
            df = t.history(period="5d", interval="1d")
            if isinstance(df, pd.DataFrame) and (not df.empty) and ("Close" in df.columns):
                close_series = df["Close"].dropna()
                if not close_series.empty:
                    last = float(close_series.iloc[-1])
        except Exception:
            last = None

    return {
        "last_price": last,
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
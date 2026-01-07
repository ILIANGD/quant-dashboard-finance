import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

def load_live_quote(ticker: str) -> dict:
    """
    Returns a live-ish quote (last price), previous close, and timestamp.

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
    prev = None

    # 1) Try fast_info (quickest way to get last + prev_close)
    try:
        fi = getattr(t, "fast_info", {}) or {}
        last = fi.get("last_price", None)
        prev = fi.get("previous_close", None)

        if last is not None: last = float(last)
        if prev is not None: prev = float(prev)
    except Exception:
        last = None
        prev = None

    # 2) Try intraday 1m (fallback for last price)
    if last is None:
        try:
            df = t.history(period="1d", interval="1m")
            if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
                last = float(df["Close"].dropna().iloc[-1])
                # Note: intraday history doesn't easily give "yesterday close" unless we query more data
            if isinstance(df, pd.DataFrame) and (not df.empty) and ("Close" in df.columns):
                close_series = df["Close"].dropna()
                if not close_series.empty:
                    last = float(close_series.iloc[-1])
        except Exception:
            pass

    # 3) Fallback / Ensure we have Prev Close: fetch daily history
    # If we missed 'prev' via fast_info, or 'last' is still None
    if last is None or prev is None:
        try:
            df_daily = t.history(period="5d", interval="1d")
            if isinstance(df_daily, pd.DataFrame) and not df_daily.empty and "Close" in df_daily.columns:
                closes = df_daily["Close"].dropna()
                if len(closes) >= 1 and last is None:
                    last = float(closes.iloc[-1])
                if len(closes) >= 2 and prev is None:
                    # The second to last point is the previous close
                    prev = float(closes.iloc[-2])
                elif len(closes) >= 1 and prev is None:
                    # If only 1 day of history, we can't compute change properly
                    prev = float(closes.iloc[-1]) 
            df = t.history(period="5d", interval="1d")
            if isinstance(df, pd.DataFrame) and (not df.empty) and ("Close" in df.columns):
                close_series = df["Close"].dropna()
                if not close_series.empty:
                    last = float(close_series.iloc[-1])
        except Exception:
            pass

    return {
        "last_price": last,
        "prev_close": prev,
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
    }

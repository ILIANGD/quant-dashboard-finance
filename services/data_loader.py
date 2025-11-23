import yfinance as yf
import pandas as pd

def load_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if data.empty or "Close" not in data.columns:
        return pd.DataFrame()

    out = data[["Close"]].rename(columns={"Close": "price"})
    out.index = out.index.tz_localize(None)
    return out

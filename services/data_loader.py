import yfinance as yf
import pandas as pd

def load_price(ticker: str, period="1y", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval)
    if "Close" in data.columns:
        return data["Close"].rename("price")
    else:
        return pd.Series(dtype=float)
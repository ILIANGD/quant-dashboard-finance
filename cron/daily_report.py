import os
from datetime import datetime, timezone

import pandas as pd
from services.data_loader import load_price_history


TICKER = os.getenv("REPORT_TICKER", "BZ=F")
PERIOD = os.getenv("REPORT_PERIOD", "6mo")
INTERVAL = os.getenv("REPORT_INTERVAL", "1d")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def max_drawdown_from_prices(prices: pd.Series) -> float:
    rets = prices.pct_change().dropna()
    if rets.empty:
        return float("nan")
    cum = (1 + rets).cumprod()
    dd = cum / cum.cummax() - 1
    return float(dd.min())


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = load_price_history(TICKER, period=PERIOD, interval=INTERVAL)
    if df is None or df.empty or "price" not in df.columns:
        raise RuntimeError("No data returned by load_price_history")

    prices = df["price"].dropna()
    if prices.empty:
        raise RuntimeError("Empty price series")

    # Daily metrics
    rets = prices.pct_change().dropna()
    vol_daily = float(rets.std())
    vol_annual = vol_daily * (252 ** 0.5)

    open_price = float(prices.iloc[0])
    close_price = float(prices.iloc[-1])
    max_dd = max_drawdown_from_prices(prices)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    out = {
        "asof_utc": now,
        "ticker": TICKER,
        "period": PERIOD,
        "interval": INTERVAL,
        "open_price": open_price,
        "close_price": close_price,
        "annual_volatility": vol_annual,
        "max_drawdown": max_dd,
    }

    # Save report
    filename = f"{report_date}_{TICKER.replace('=','').replace('^','')}_daily_report.csv"
    path = os.path.join(REPORTS_DIR, filename)
    pd.DataFrame([out]).to_csv(path, index=False)

    print(f"Saved report: {path}")


if __name__ == "__main__":
    main()

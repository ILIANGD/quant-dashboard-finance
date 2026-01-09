import os
import sys
from datetime import datetime, timezone
import pandas as pd

# Ajout du dossier parent au path pour trouver le module 'services'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from services.data_loader import load_price_history

# Configuration
TICKER = os.getenv("REPORT_TICKER", "BZ=F")
PERIOD = os.getenv("REPORT_PERIOD", "6mo")
INTERVAL = os.getenv("REPORT_INTERVAL", "1d")
REPORTS_DIR = os.path.join(parent_dir, "reports")

def max_drawdown_from_prices(prices: pd.Series) -> float:
    rets = prices.pct_change().dropna()
    if rets.empty: return float("nan")
    cum = (1 + rets).cumprod()
    dd = cum / cum.cummax() - 1
    return float(dd.min())

def main():
    print(f"Starting report generation for {TICKER}...")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load Data
    try:
        df = load_price_history(TICKER, period=PERIOD, interval=INTERVAL)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if df is None or df.empty or "price" not in df.columns:
        print("No data returned.")
        return

    prices = df["price"].dropna()
    if prices.empty:
        print("Empty price series.")
        return

    # 2. Compute Metrics
    rets = prices.pct_change().dropna()
    vol_daily = float(rets.std())
    vol_annual = vol_daily * (252 ** 0.5)

    open_price = float(prices.iloc[0]) # Price at start of period
    close_price = float(prices.iloc[-1]) # Price today
    max_dd = max_drawdown_from_prices(prices)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    report_data = {
        "generated_at": now_str,
        "ticker": TICKER,
        "period": PERIOD,
        "open_price_period": round(open_price, 2),
        "close_price_current": round(close_price, 2),
        "annual_volatility": round(vol_annual, 4),
        "max_drawdown": round(max_dd, 4),
    }

    # 3. Save to CSV
    # Clean ticker name for filename (e.g. BZ=F -> BZF)
    safe_ticker = TICKER.replace("=", "").replace("^", "")
    filename = f"{date_str}_{safe_ticker}_daily_report.csv"
    file_path = os.path.join(REPORTS_DIR, filename)

    pd.DataFrame([report_data]).to_csv(file_path, index=False)
    print(f"Report saved successfully: {file_path}")

if __name__ == "__main__":
    main()

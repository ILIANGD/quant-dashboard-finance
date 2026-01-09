import os
import sys
import logging
from datetime import datetime, timezone
import pandas as pd

# =========================================================
# 1. Path Configuration (Safety Net)
# =========================================================
# Add the project root to sys.path to ensure 'services' can be imported
# regardless of how the script is executed.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can safely import from services
try:
    from services.data_loader import load_price_history
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import services. Check PYTHONPATH. Details: {e}")
    sys.exit(1)

# =========================================================
# 2. Setup Logging
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================================
# 3. Environment Variables & Constants
# =========================================================
TICKER = os.getenv("REPORT_TICKER", "BZ=F")
PERIOD = os.getenv("REPORT_PERIOD", "6mo")
INTERVAL = os.getenv("REPORT_INTERVAL", "1d")

REPORTS_DIR = os.path.join(project_root, "reports")

# =========================================================
# 4. Helper Functions
# =========================================================

def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculates the Maximum Drawdown of a price series."""
    if prices.empty:
        return float("nan")
    
    # Calculate returns
    rets = prices.pct_change().dropna()
    
    # Calculate cumulative returns (wealth index)
    wealth_index = (1 + rets).cumprod()
    
    # Calculate running maximum
    previous_peaks = wealth_index.cummax()
    
    # Calculate drawdown
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    
    return float(drawdown.min())

# =========================================================
# 5. Main Execution
# =========================================================

def main():
    logger.info(f"Starting daily report generation for {TICKER}...")

    # Ensure output directory exists
    os.makedirs(REPORTS_DIR, exist_ok=True)

    try:
        # Load Data
        df = load_price_history(TICKER, period=PERIOD, interval=INTERVAL)
        
        if df is None or df.empty or "price" not in df.columns:
            logger.error(f"No data returned for ticker {TICKER}. Aborting.")
            sys.exit(1)

        prices = df["price"].dropna()
        if prices.empty:
            logger.error(f"Price series is empty for {TICKER}. Aborting.")
            sys.exit(1)

        # Calculate Metrics
        # 1. Volatility
        rets = prices.pct_change().dropna()
        vol_daily = float(rets.std())
        vol_annual = vol_daily * (252 ** 0.5)

        # 2. Prices
        open_price = float(prices.iloc[0])
        close_price = float(prices.iloc[-1])

        # 3. Drawdown
        max_dd = calculate_max_drawdown(prices)

        # Prepare Data
        now_utc = datetime.now(timezone.utc)
        report_date_str = now_utc.strftime("%Y-%m-%d")
        
        report_data = {
            "generated_at_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "report_date": report_date_str,
            "ticker": TICKER,
            "period_analyzed": PERIOD,
            "interval": INTERVAL,
            "open_price": round(open_price, 4),
            "close_price": round(close_price, 4),
            "annualized_volatility": round(vol_annual, 4),
            "max_drawdown": round(max_dd, 4),
        }

        # Generate Filename
        # Sanitize ticker name (remove = or ^ for filenames)
        safe_ticker = TICKER.replace('=', '').replace('^', '')
        filename = f"{report_date_str}_{safe_ticker}_daily_report.csv"
        file_path = os.path.join(REPORTS_DIR, filename)

        # Save to CSV
        pd.DataFrame([report_data]).to_csv(file_path, index=False)
        
        logger.info(f"SUCCESS: Report saved to {file_path}")
        logger.info(f"Summary: Vol={vol_annual:.2%}, DD={max_dd:.2%}")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

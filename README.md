Markdown

# Quantitative Finance Dashboard

## Project Overview
This project is a professional-grade dashboard designed for asset management simulation. It allows quantitative researchers and portfolio managers to:
* **Monitor** financial assets in real-time using live market data.
* **Analyze** single-asset performance using advanced backtesting strategies and Machine Learning forecasting.
* **Simulate** multi-asset portfolios with custom allocations and risk-adjusted metrics.
* **Automate** daily financial reporting via Linux cron jobs.

Built with **Python (Streamlit)**, this application is designed to be deployed on a Linux server for continuous 24/7 operation.

---

## Architecture & Modules

The project is strictly divided into two quantitative modules, complying with the collaborative requirements.

### Quant A: Single Asset Analysis
* **Focus:** Univariate analysis and predictive modeling.
* **Key Features:**
    * **Dynamic Data:** Real-time retrieval via Yahoo Finance API.
    * **Backtesting Strategies:**
        * *Buy & Hold*
        * *Momentum* (Moving Average Crossover with adjustable windows).
        * *Volatility Breakout* (Price exceeding N-day high).
    * **ML Forecasting:** Weighted Linear Regression (Exponential weighting) to prioritize recent trends, with optional **Log-Space** transformation for compound growth modeling. Includes 95% confidence intervals.
    * **Metrics:** Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio.

### Quant B: Multi-Asset Portfolio
* **Focus:** Multivariate simulation and optimization.
* **Key Features:**
    * **Asset Universe:** Global indices (MSCI World, Emerging Markets), Sectors, Commodities, and Crypto.
    * **Simulation Engine:** Supports *Equal-Weight* and *Custom Weight* allocations.
    * **Rebalancing:** Simulates Daily, Weekly, or Monthly portfolio rebalancing.
    * **Risk Analysis:** Interactive Correlation Heatmap and Diversification Ratio calculation.
    * **Benchmarking:** Relative performance comparison (Base 100) of the portfolio vs. individual assets.

---

## Installation & Setup

### Prerequisites
* Linux Environment (Ubuntu/Debian recommended)
* Python 3.10+
* Git

### 1. Clone the Repository
```bash
git clone [https://github.com/ILIANGD/quant-dashboard-finance.git](https://github.com/ILIANGD/quant-dashboard-finance.git)
cd quant-dashboard-finance
2. Virtual Environment
It is recommended to use a virtual environment to isolate dependencies.

Bash

python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
Bash

pip install -r requirements.txt
4. Run the Dashboard
Bash

streamlit run Home.py
The app will be accessible at http://localhost:8501 (or your VM's public IP).

 Automated Daily Reporting (Cron)
As per project requirements, a daily report containing Volatility, Drawdown, and Open/Close prices is generated automatically at 8:00 PM every day.

Configuration Steps on VM
Permissions: Ensure the shell wrapper is executable.

Bash

chmod +x cron/run_daily_report.sh
Cron Setup: Open your crontab configuration.

Bash

crontab -e
Job Entry: Add the following line to schedule the job:

Plaintext

0 20 * * * /bin/bash /home/ubuntu/quant-dashboard-finance/cron/run_daily_report.sh >> /home/ubuntu/quant-dashboard-finance/cron/cron.log 2>&1
Output
CSV Reports: Saved locally in the reports/ directory (e.g., 2025-01-14_BZF_daily_report.csv).

Logs: Execution logs and errors are stored in cron/cron.log.

 Project Structure
Plaintext

quant-dashboard-finance/
├── cron/                   # Automation scripts
│   ├── daily_report.py     # Core reporting logic (Vol, DD calculation)
│   ├── run_daily_report.sh # Bash wrapper for virtualenv handling
│   └── crontab_config.txt  # Documentation for cron setup
├── modules/                # Core quantitative logic
│   ├── portfolio.py        # Quant B Module (Multi-asset logic)
│   └── single_asset.py     # Quant A Module (Backtest & ML logic)
├── pages/                  # Streamlit navigation pages
│   ├── 1_Single_Asset.py   # UI for Single Asset
│   └── 2_Portfolio.py      # UI for Portfolio
├── services/               # Shared services
│   └── data_loader.py      # Data fetching, Caching & Timezone cleaning
├── reports/                # Generated CSV reports storage
├── Home.py                 # Main application entry point
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
 Tech Stack
Frontend: Streamlit

Data Processing: Pandas, NumPy

Visualization: Altair (Interactive charts)

Financial Data: yfinance (Yahoo Finance API)

Machine Learning: Scikit-learn (Linear Regression)

Automation: Bash, Cron

 Disclaimer
This project is developed for educational purposes as part of a Quantitative Finance course. Market data is provided by Yahoo Finance and may be delayed. Past performance is not indicative of future results.

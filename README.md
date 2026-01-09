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

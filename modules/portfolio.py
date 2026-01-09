import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import altair as alt
import locale
import time

# Tenter de définir la locale pour le formatage
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    pass

# On suppose que ce fichier existe dans votre projet
from services.data_loader import load_price_history

# =========================
# ASSET DATABASE
# =========================

ASSET_DB = {
    "GLOBAL & REGIONAL (ETFs/Indices)": {
        "MSCI World (iShares)": "URTH",
        "MSCI ACWI - All Country (iShares)": "ACWI",
        "MSCI Emerging Markets (iShares)": "EEM",
        "Vanguard Total World Stock": "VT",
        "Euro Stoxx 600": "^STOXX",
        "Vanguard FTSE Europe": "VGK",
        "iShares Asia 50": "AIA"
    },
    "US SECTORS (SPDR ETFs)": {
        "Technology (XLK)": "XLK",
        "Financials (XLF)": "XLF",
        "Healthcare (XLV)": "XLV",
        "Energy (XLE)": "XLE",
        "Semiconductors (SOXX)": "SOXX",
        "Consumer Discretionary (XLY)": "XLY",
        "Consumer Staples (XLP)": "XLP",
        "Utilities (XLU)": "XLU",
        "Industrials (XLI)": "XLI"
    },
    "REAL ESTATE (REITs)": {
        "Vanguard Real Estate (US)": "VNQ",
        "iShares Global REIT": "REET",
        "Simon Property Group": "SPG",
        "Prologis": "PLD",
        "Realty Income": "O"
    },
    "BONDS & RATES": {
        "German 10Y Bund Yield": "^TNX",
        "US Treasury Yield 10 Years": "^TNX",
        "US Treasury Yield 30 Years": "^TYX",
        "US Treasury Yield 5 Years": "^FVX",
        "iShares 20+ Year Treasury Bond (TLT)": "TLT",
        "iShares 7-10 Year Treasury Bond (IEF)": "IEF",
        "iShares Core US Aggregate Bond (AGG)": "AGG"
    },
    "COMMODITIES - AGRI": {
        "Cocoa": "CC=F",
        "Coffee": "KC=F",
        "Corn": "ZC=F",
        "Cotton": "CT=F",
        "Lean Hogs": "HE=F",
        "Live Cattle": "LE=F",
        "Lumber": "LBS=F",
        "Oats": "ZO=F",
        "Orange Juice": "OJ=F",
        "Rough Rice": "ZR=F",
        "Soybean Meal": "ZM=F",
        "Soybean Oil": "ZL=F",
        "Soybeans": "ZS=F",
        "Sugar": "SB=F",
        "Wheat": "ZW=F"
    },
    "COMMODITIES - ENERGY": {
        "Brent Crude Oil": "BZ=F",
        "Gasoline (RBOB)": "RB=F",
        "Heating Oil": "HO=F",
        "Natural Gas": "NG=F",
        "WTI Crude Oil": "CL=F",
        "Uranium (URA ETF)": "URA"
    },
    "COMMODITIES - METALS": {
        "Aluminum": "ALI=F",
        "Copper": "HG=F",
        "Gold": "GC=F",
        "Palladium": "PA=F",
        "Platinum": "PL=F",
        "Silver": "SI=F",
        "Lithium (LIT ETF)": "LIT"
    },
    "CRYPTO": {
        "Avalanche": "AVAX-USD",
        "Binance Coin": "BNB-USD",
        "Bitcoin": "BTC-USD",
        "Bitcoin Cash": "BCH-USD",
        "Cardano": "ADA-USD",
        "Chainlink": "LINK-USD",
        "Dogecoin": "DOGE-USD",
        "Ethereum": "ETH-USD",
        "Litecoin": "LTC-USD",
        "Polkadot": "DOT-USD",
        "Polygon": "MATIC-USD",
        "Shiba Inu": "SHIB-USD",
        "Solana": "SOL-USD",
        "Uniswap": "UNI-USD",
        "XRP": "XRP-USD"
    },
    "FOREX - MAJORS": {
        "AUD/USD": "AUDUSD=X",
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "NZD/USD": "NZDUSD=X",
        "USD/CAD": "CAD=X",
        "USD/CHF": "CHF=X",
        "USD/JPY": "JPY=X"
    },
    "FOREX - CROSSES & EXOTICS": {
        "AUD/JPY": "AUDJPY=X",
        "EUR/GBP": "EURGBP=X",
        "EUR/JPY": "EURJPY=X",
        "EUR/SEK": "EURSEK=X",
        "GBP/JPY": "GBPJPY=X",
        "USD/BRL (Brazil)": "BRL=X",
        "USD/CNY (China)": "CNY=X",
        "USD/HKD (Hong Kong)": "HKD=X",
        "USD/INR (India)": "INR=X",
        "USD/KRW (South Korea)": "KRW=X",
        "USD/MXN (Mexico)": "MXN=X",
        "USD/SGD (Singapore)": "SGD=X",
        "USD/TRY (Turkey)": "TRY=X",
        "USD/ZAR (South Africa)": "ZAR=X"
    },
    "INDICES - AMERICAS": {
        "Bovespa (Brazil)": "^BVSP",
        "Dow Jones 30": "^DJI",
        "IPC Mexico": "^MXX",
        "Merval (Argentina)": "^MERV",
        "Nasdaq 100": "^NDX",
        "NYSE Composite": "^NYA",
        "Russell 2000": "^RUT",
        "S&P 500": "^GSPC",
        "TSX Composite (Canada)": "^GSPTSE",
        "VIX (Volatility)": "^VIX"
    },
    "INDICES - ASIA PACIFIC": {
        "ASX 200 (Australia)": "^AXJO",
        "BSE SENSEX (India)": "^BSESN",
        "Hang Seng (Hong Kong)": "^HSI",
        "KOSPI (South Korea)": "^KS11",
        "Nifty 50 (India)": "^NSEI",
        "Nikkei 225 (Japan)": "^N225",
        "Shanghai Composite": "000001.SS",
        "Shenzhen Component": "399001.SZ",
        "STI (Singapore)": "^STI",
        "TAIEX (Taiwan)": "^TWII"
    },
    "INDICES - EUROPE": {
        "AEX (Netherlands)": "^AEX",
        "BEL 20 (Belgium)": "^BFX",
        "CAC 40 (France)": "^FCHI",
        "DAX (Germany)": "^GDAXI",
        "Euro Stoxx 50": "^STOXX50E",
        "FTSE 100 (UK)": "^FTSE",
        "FTSE MIB (Italy)": "FTSEMIB.MI",
        "IBEX 35 (Spain)": "^IBEX",
        "PSI 20 (Portugal)": "^PSI20",
        "SMI (Switzerland)": "^SSMI"
    },
    "STOCKS - FRANCE (CAC 40)": {
        "Air Liquide": "AI.PA",
        "Airbus": "AIR.PA",
        "AXA": "CS.PA",
        "BNP Paribas": "BNP.PA",
        "Danone": "BN.PA",
        "EssilorLuxottica": "EL.PA",
        "Hermes": "RMS.PA",
        "Kering": "KER.PA",
        "L'Oreal": "OR.PA",
        "LVMH": "MC.PA",
        "Orange": "ORA.PA",
        "Pernod Ricard": "RI.PA",
        "Safran": "SAF.PA",
        "Saint-Gobain": "SGO.PA",
        "Sanofi": "SAN.PA",
        "Schneider Electric": "SU.PA",
        "TotalEnergies": "TTE.PA",
        "Vinci": "DG.PA"
    },
    "STOCKS - US TECH & GIANTS": {
        "AMD": "AMD",
        "Adobe": "ADBE",
        "Airbnb": "ABNB",
        "Alphabet (Google)": "GOOGL",
        "Amazon": "AMZN",
        "Apple": "AAPL",
        "Berkshire Hathaway": "BRK-B",
        "Broadcom": "AVGO",
        "Coca-Cola": "KO",
        "Costco": "COST",
        "Eli Lilly": "LLY",
        "Exxon Mobil": "XOM",
        "JPMorgan Chase": "JPM",
        "Johnson & Johnson": "JNJ",
        "Mastercard": "MA",
        "Meta Platforms": "META",
        "Microsoft": "MSFT",
        "Netflix": "NFLX",
        "Nvidia": "NVDA",
        "Oracle": "ORCL",
        "PepsiCo": "PEP",
        "Pfizer": "PFE",
        "Procter & Gamble": "PG",
        "Salesforce": "CRM",
        "Tesla": "TSLA",
        "Uber": "UBER",
        "Visa": "V",
        "Walmart": "WMT"
    }
}

# Flatten for dropdown
FLAT_ASSETS = {}
for category in sorted(ASSET_DB.keys()):
    items = ASSET_DB[category]
    for name in sorted(items.keys()):
        label = f"{category} | {name} ({items[name]})"
        FLAT_ASSETS[label] = items[name]

TICKER_TO_LABEL = {v: k for k, v in FLAT_ASSETS.items()}

# =========================
# State Persistence Logic
# =========================

def init_session_state():
    """Force initialization of session state variables to ensure persistence."""
    
    # Default values
    DEFAULT_TICKERS = ["BZ=F", "GC=F", "HG=F"] # Brent, Gold, Copper
    DEFAULT_LOOKBACK = "5y"
    DEFAULT_FREQ = "Daily"
    DEFAULT_CAPITAL = 10000.0
    DEFAULT_ALLOC = "Equal Weight"
    DEFAULT_REBAL = "Monthly"

    # 1. Tickers: Priority = Session > URL > Default
    if "p_tickers_list" not in st.session_state:
        # Check URL
        url_tickers_str = st.query_params.get("tickers", "")
        if url_tickers_str:
            url_tickers = [t.strip() for t in url_tickers_str.split(",") if t.strip()]
            # Convert to labels
            labels = []
            for t in url_tickers:
                if t in TICKER_TO_LABEL: labels.append(TICKER_TO_LABEL[t])
            
            if labels:
                st.session_state.p_tickers_list = labels
            else:
                st.session_state.p_tickers_list = [TICKER_TO_LABEL[t] for t in DEFAULT_TICKERS if t in TICKER_TO_LABEL]
        else:
            # Use Default
            st.session_state.p_tickers_list = [TICKER_TO_LABEL[t] for t in DEFAULT_TICKERS if t in TICKER_TO_LABEL]

    # 2. Other settings
    if "p_lookback" not in st.session_state: st.session_state.p_lookback = DEFAULT_LOOKBACK
    if "p_freq" not in st.session_state: st.session_state.p_freq = DEFAULT_FREQ
    if "p_capital" not in st.session_state: st.session_state.p_capital = DEFAULT_CAPITAL
    if "p_alloc" not in st.session_state: st.session_state.p_alloc = DEFAULT_ALLOC
    if "p_rebal" not in st.session_state: st.session_state.p_rebal = DEFAULT_REBAL


# =========================
# Helpers & Logic
# =========================

def _normalize_weights(w: dict) -> dict:
    ww = {}
    for k, v in w.items():
        try:
            vv = float(v)
            if vv >= 0: ww[k] = vv
        except: continue
    s = sum(ww.values())
    if s <= 0: return {k: 1.0/len(ww) for k in ww} if ww else {}
    return {k: v/s for k, v in ww.items()}

def _align_prices(price_dict: dict) -> pd.DataFrame:
    df = pd.concat(price_dict, axis=1)
    df.columns = list(price_dict.keys())
    return df.sort_index().dropna(how="any")

def rebalance_dates(index: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
    if rule == "Daily": return index
    s = pd.Series(index=index, data=index)
    res = s.resample("W").first() if rule == "Weekly" else s.resample("ME").first()
    return pd.DatetimeIndex(res.values)

def simulate_portfolio(prices_df: pd.DataFrame, weights: dict, rebal_rule: str, initial_capital: float) -> pd.Series:
    rets = prices_df.pct_change().dropna()
    if rets.empty: return pd.Series(dtype=float)
    
    tickers = list(prices_df.columns)
    w_target = _normalize_weights({t: weights.get(t, 0.0) for t in tickers})
    w_vec = np.array([w_target.get(t, 0.0) for t in tickers])
    
    rdates = set(rebalance_dates(rets.index, rebal_rule))
    current_w = w_vec.copy()
    port_rets = []
    
    for dt, row in rets.iterrows():
        if dt in rdates:
            current_w = w_vec.copy()
        
        r = row.values
        port_ret = np.dot(current_w, r)
        port_rets.append(port_ret)
        
        current_w = current_w * (1 + r) / (1 + port_ret)

    port_series = pd.Series(port_rets, index=rets.index)
    equity = (1 + port_series).cumprod() * initial_capital
    return equity

def compute_metrics(series: pd.Series) -> dict:
    series = series.dropna()
    rets = series.pct_change().dropna()
    if rets.empty: return {}

    mean_daily = float(rets.mean())
    vol_daily = float(rets.std())
    annual_return = (1 + mean_daily) ** 252 - 1
    annual_vol = vol_daily * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol != 0 else 0.0
    
    cum = (1 + rets).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0
    
    return {
        "annual_return": annual_return, "annual_vol": annual_vol,
        "sharpe": sharpe, "max_drawdown": max_dd, "calmar": calmar
    }

def metric_card(label: str, value_str: str, help_text: str = None):
    st.caption(f"**{label}**", help=help_text)
    st.markdown(f"<div style='font-size:1.4em; font-weight:600;'>{value_str}</div>", unsafe_allow_html=True)


# =========================
# Main Page
# =========================

def portfolio_page():
    st.title("Portfolio Management")
    st.autorefresh = st_autorefresh(interval=300_000, key="auto_refresh_port")

    # 1. Initialize State (Fixes persistence)
    init_session_state()

    # ==========================================
    # 1. CONTROLS
    # ==========================================
    with st.container(border=True):
        st.subheader("Controls")
        
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        
        with c1:
            def clean_label(label):
                if " | " in label: return label.split(" | ", 1)[1]
                return label

            # Widget bound to session_state key 'p_tickers_list'
            selected_labels = st.multiselect(
                "Select Assets (3 to 8)",
                options=FLAT_ASSETS.keys(),
                key="p_tickers_list", # Persistence Key
                max_selections=8,
                format_func=clean_label
            )
            
            # Resolve Tickers
            final_tickers = [FLAT_ASSETS[l] for l in selected_labels]
            final_tickers = list(dict.fromkeys(final_tickers)) 
            
            # Update URL quietly
            new_url_str = ",".join(final_tickers)
            if new_url_str != st.query_params.get("tickers"):
                st.query_params["tickers"] = new_url_str

        with c2:
            period = st.selectbox("Lookback", ["3mo", "6mo", "1y", "2y", "5y", "10y", "max"], key="p_lookback")
        with c3:
            interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
            interval_label = st.selectbox("Frequency", list(interval_map.keys()), key="p_freq")
            interval = interval_map[interval_label]
        with c4:
            initial_capital = st.number_input("Initial Capital ($)", step=1000.0, key="p_capital")

        st.divider()

        sc1, sc2, sc3 = st.columns([1, 1, 2])
        with sc1:
            alloc_type = st.selectbox("Allocation", ["Equal Weight", "Custom Weights"], key="p_alloc")
        with sc2:
            rebal_freq = st.selectbox("Rebalancing", ["Monthly", "Weekly", "Daily"], key="p_rebal")
        
        if len(final_tickers) < 3:
            st.error(f"Please select at least 3 assets. Currently selected: {len(final_tickers)}")
            return
        
        weights = {}
        if alloc_type == "Custom Weights":
            with sc3:
                st.caption("Weights (will be normalized)")
                cols = st.columns(len(final_tickers))
                for i, t in enumerate(final_tickers):
                    with cols[i]:
                        # Keys MUST be dynamic based on ticker to avoid conflicts
                        weights[t] = st.number_input(f"{t}", value=1.0, step=0.1, key=f"w_{t}")
        else:
            weights = {t: 1.0 for t in final_tickers}

    # ==========================================
    # Data Loading (CORRECTED SECTION)
    # ==========================================
    price_data = {}
    
    # FIX: Add a spinner so the user knows it is working
    with st.spinner("Fetching market data..."):
        for t in final_tickers:
            try:
                df = load_price_history(t, period=period, interval=interval)
                if df is not None and not df.empty:
                    s = df["price"].dropna()
                    if s.index.tz is not None: s.index = s.index.tz_localize(None)
                    price_data[t] = s
            except: pass
    
    # FIX: Robust check to avoid "crash" on incomplete data
    if len(price_data) < 3:
        st.warning(
            f"**Syncing data...** found {len(price_data)} valid assets so far. "
            "Please wait or check your internet connection."
        )
        # Optional: Button to manually trigger a retry if stuck
        if st.button("Retry Loading Data"):
            st.rerun()
        return

    prices_df = _align_prices(price_data)
    
    # Simulation
    port_equity = simulate_portfolio(prices_df, weights, rebal_freq, initial_capital)
    port_metrics = compute_metrics(port_equity)

    # Div Ratio
    asset_rets = prices_df.pct_change().dropna()
    asset_vols = asset_rets.std() * np.sqrt(252)
    w_target = _normalize_weights({t: weights.get(t, 0.0) for t in final_tickers})
    weighted_avg_vol = sum([w * asset_vols[t] for t, w in w_target.items() if t in asset_vols])
    port_vol = port_metrics.get('annual_vol', 0)
    div_ratio = weighted_avg_vol / port_vol if port_vol > 0 else 1.0

    # ==========================================
    # 2. OVERVIEW & METRICS
    # ==========================================
    st.divider()
    st.header("Portfolio Overview")
    
    curr_val = port_equity.iloc[-1]
    start_val = port_equity.iloc[0]
    total_ret = (curr_val - start_val) / start_val
    
    c_val, c_met = st.columns([1, 3])
    with c_val:
        st.metric("Total Value", f"${curr_val:,.2f}", f"{total_ret:+.2%}")
        st.caption(f"Initial: ${start_val:,.2f}")
        if st.button("Refresh Data", type="primary"):
            st.rerun()
    
    with c_met:
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: metric_card("Ann. Return", f"{port_metrics['annual_return']:.2%}")
        with m2: metric_card("Volatility", f"{port_metrics['annual_vol']:.2%}")
        with m3: metric_card("Sharpe", f"{port_metrics['sharpe']:.2f}")
        with m4: metric_card("Max Drawdown", f"{port_metrics['max_drawdown']:.2%}")
        with m5: metric_card("Div. Ratio", f"{div_ratio:.2f}", "Diversification Ratio (>1 is better)")

    # ==========================================
    # 3. PERFORMANCE CHART (Fixed Tooltips)
    # ==========================================
    st.divider()
    st.subheader("Relative Performance (Base 100)")
    
    # Data Prep: Create ONE Wide DataFrame
    norm_assets = (prices_df / prices_df.iloc[0]) * 100
    norm_port = (port_equity / port_equity.iloc[0]) * 100
    
    df_wide = norm_assets.copy()
    df_wide["Portfolio"] = norm_port
    df_wide = df_wide.reset_index().rename(columns={df_wide.index.name or "Date": "date"})

    # --- OPTIMIZATION: SMART SAMPLING ---
    MAX_POINTS = 500
    if len(df_wide) > MAX_POINTS:
        step = len(df_wide) // MAX_POINTS + 1
        df_chart_source = df_wide.iloc[::step].copy()
    else:
        df_chart_source = df_wide.copy()

    # --- PANDAS MELT (Long format for lines) ---
    df_long = df_chart_source.melt(id_vars=["date"], var_name="Asset", value_name="Value")
    
    # Colors
    domain = ["Portfolio"] + list(prices_df.columns)
    range_ = ["#FF4B4B"] + ["#4A90E2", "#50C878", "#FFD700", "#9370DB", "#FF8C00", "#00CED1", "#C71585", "#808080"][:len(prices_df.columns)]

    # 1. Base Selection
    hover = alt.selection_point(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty="none",
        clear="mouseout"
    )

    # 2. Dynamic Tooltip List construction
    tooltip_list = [alt.Tooltip("date", title="Date", format="%d/%m/%Y")]
    tooltip_list.append(alt.Tooltip("Portfolio", title="Portfolio", format=".2f"))
    for t in prices_df.columns:
        tooltip_list.append(alt.Tooltip(t, title=t, format=".2f"))

    # 3. Chart Layers
    
    # A. Lines (The curves)
    lines = alt.Chart(df_long).mark_line().encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%d/%m/%Y", grid=True, gridOpacity=0.3)),
        y=alt.Y("Value:Q", scale=alt.Scale(zero=False), axis=alt.Axis(title="Base 100")),
        color=alt.Color("Asset:N", scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(orient="bottom")),
        strokeWidth=alt.condition(alt.datum.Asset == 'Portfolio', alt.value(3), alt.value(1.5)),
        opacity=alt.condition(alt.datum.Asset == 'Portfolio', alt.value(1), alt.value(0.6))
    )

    # B. Selectors (Invisible points that capture mouse & SHOW TOOLTIPS)
    # CORRECTION : Le tooltip est déplacé ici
    selectors = alt.Chart(df_chart_source).mark_point().encode(
        x="date:T",
        opacity=alt.value(0),
        tooltip=tooltip_list 
    ).add_params(hover)

    # C. Rule (The vertical line visual)
    # CORRECTION : On retire le tooltip d'ici, il ne sert plus qu'à l'affichage de la ligne
    rule = alt.Chart(df_chart_source).mark_rule(color="gray").encode(
        x="date:T",
        opacity=alt.condition(hover, alt.value(0.5), alt.value(0))
    ).transform_filter(hover)

    # Combine
    chart = alt.layer(lines, selectors, rule).properties(height=450)

    st.altair_chart(chart, use_container_width=True)

    # ==========================================
    # 4. INDIVIDUAL ASSET METRICS (Table)
    # ==========================================
    st.divider()
    st.subheader("Individual Asset Performance (Table)")
    
    asset_metrics_list = []
    
    for ticker in prices_df.columns:
        m = compute_metrics(prices_df[ticker])
        row = {
            "Asset": ticker,
            "Return (Ann.)": f"{m['annual_return']:.2%}",
            "Volatility": f"{m['annual_vol']:.2%}",
            "Sharpe Ratio": f"{m['sharpe']:.2f}",
            "Max Drawdown": f"{m['max_drawdown']:.2%}"
        }
        asset_metrics_list.append(row)
    
    df_metrics_table = pd.DataFrame(asset_metrics_list)
    
    st.dataframe(
        df_metrics_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Asset": st.column_config.TextColumn("Asset", width="medium"),
            "Return (Ann.)": st.column_config.TextColumn("Return (Ann.)"),
            "Volatility": st.column_config.TextColumn("Volatility"),
            "Sharpe Ratio": st.column_config.TextColumn("Sharpe"),
            "Max Drawdown": st.column_config.TextColumn("Max Drawdown"),
        }
    )

    # ==========================================
    # 5. COMPOSITION & RISK
    # ==========================================
    st.divider()
    st.header("Composition & Risk")
    
    r1, r2 = st.columns(2)
    
    with r1:
        st.subheader("Correlation Matrix")
        corr = prices_df.pct_change().corr()
        
        corr_reset = corr.reset_index()
        cid_var = corr_reset.columns[0]
        corr_melt = corr_reset.melt(id_vars=cid_var).rename(columns={cid_var: "var1", "variable": "var2"})

        heatmap = alt.Chart(corr_melt).mark_rect().encode(
            x=alt.X("var1:N", title=None),
            y=alt.Y("var2:N", title=None),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
            tooltip=["var1", "var2", alt.Tooltip("value:Q", format=".2f")]
        ).properties(height=300)
        
        text = heatmap.mark_text().encode(
            text=alt.Text("value:Q", format=".2f"),
            color=alt.value("black")
        )
        st.altair_chart(heatmap + text, use_container_width=True)

    with r2:
        st.subheader("Effective Weights")
        df_w = pd.DataFrame(list(w_target.items()), columns=["Asset", "Weight"])
        
        pie = alt.Chart(df_w).mark_arc(innerRadius=50).encode(
            theta=alt.Theta("Weight", stack=True),
            color=alt.Color("Asset", scale=alt.Scale(domain=domain, range=range_)),
            tooltip=["Asset", alt.Tooltip("Weight", format=".1%")]
        ).properties(height=300)
        
        st.altair_chart(pie, use_container_width=True)


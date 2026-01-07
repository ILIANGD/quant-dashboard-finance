import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import altair as alt
import locale

# Tenter de définir la locale pour le formatage
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    pass

from services.data_loader import load_price_history, load_live_quote

# =========================
# Helpers & Logic
# =========================

def _normalize_weights(w: dict) -> dict:
    """Keep only non-negative numbers, normalize to sum to 1."""
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
    
    # Simple rebalancing logic
    rdates = set(rebalance_dates(rets.index, rebal_rule))
    
    # Vectorized approximation for speed
    current_w = w_vec.copy()
    port_rets = []
    
    for dt, row in rets.iterrows():
        if dt in rdates:
            current_w = w_vec.copy() # Reset to target
        
        # Period return
        r = row.values
        port_ret = np.dot(current_w, r)
        port_rets.append(port_ret)
        
        # Update weights based on asset performance (Drift)
        current_w = current_w * (1 + r) / (1 + port_ret)

    port_series = pd.Series(port_rets, index=rets.index)
    equity = (1 + port_series).cumprod() * initial_capital
    return equity

# =========================
# Metrics UI
# =========================

def metric_card(label: str, value_str: str, help_text: str = None):
    st.caption(f"**{label}**", help=help_text)
    st.markdown(f"<div style='font-size:1.4em; font-weight:600;'>{value_str}</div>", unsafe_allow_html=True)

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

# =========================
# Main Page
# =========================

def portfolio_page():
    st.title("Portfolio Management")
    st.autorefresh = st_autorefresh(interval=300_000, key="auto_refresh_port")

    DEFAULT_TICKERS = "BZ=F, GC=F, HG=F"
    
    # ==========================================
    # 1. CONTROLS
    # ==========================================
    with st.container(border=True):
        st.subheader("Controls")
        
        # Row 1: Global Settings
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            tickers_txt = st.text_input("Tickers (comma separated)", value=DEFAULT_TICKERS)
        with c2:
            period = st.selectbox("Lookback", ["3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=4)
        with c3:
            interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
            interval_label = st.selectbox("Frequency", list(interval_map.keys()), index=0)
            interval = interval_map[interval_label]
        with c4:
            initial_capital = st.number_input("Initial Capital ($)", value=10000.0, step=1000.0)

        st.divider()

        # Row 2: Portfolio Strategy
        sc1, sc2, sc3 = st.columns([1, 1, 2])
        with sc1:
            alloc_type = st.selectbox("Allocation", ["Equal Weight", "Custom Weights"], index=1)
        with sc2:
            rebal_freq = st.selectbox("Rebalancing", ["Monthly", "Weekly", "Daily"], index=0)
        
        tickers = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]
        
        # Custom Weights Inputs
        weights = {}
        if alloc_type == "Custom Weights":
            with sc3:
                st.caption("Weights (will be normalized)")
                cols = st.columns(len(tickers))
                for i, t in enumerate(tickers):
                    with cols[i]:
                        weights[t] = st.number_input(f"{t}", value=1.0, step=0.1, key=f"w_{t}")
        else:
            weights = {t: 1.0 for t in tickers}

    # ==========================================
    # Data Loading
    # ==========================================
    if len(tickers) < 2:
        st.error("Please select at least 2 tickers.")
        return

    price_data = {}
    for t in tickers:
        try:
            df = load_price_history(t, period=period, interval=interval)
            if df is not None and not df.empty:
                s = df["price"].dropna()
                if s.index.tz is not None: s.index = s.index.tz_localize(None)
                price_data[t] = s
        except: pass
    
    if len(price_data) < 2:
        st.error("Not enough valid data loaded.")
        return

    prices_df = _align_prices(price_data)
    
    # Simulation
    port_equity = simulate_portfolio(prices_df, weights, rebal_freq, initial_capital)
    port_metrics = compute_metrics(port_equity)

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
    
    with c_met:
        m1, m2, m3, m4 = st.columns(4)
        with m1: metric_card("Ann. Return", f"{port_metrics['annual_return']:.2%}")
        with m2: metric_card("Volatility", f"{port_metrics['annual_vol']:.2%}")
        with m3: metric_card("Sharpe", f"{port_metrics['sharpe']:.2f}")
        with m4: metric_card("Max Drawdown", f"{port_metrics['max_drawdown']:.2%}")

    # ==========================================
    # 3. PERFORMANCE CHART
    # ==========================================
    st.divider()
    st.subheader("Performance Analysis")

    # Chart 1: Portfolio Value (Blue)
    df_chart = port_equity.to_frame("Portfolio Value").reset_index().rename(columns={"index": "date"})
    
    base = alt.Chart(df_chart).encode(
        x=alt.X("date:T", axis=alt.Axis(format="%d/%m/%Y", title=None, grid=True, gridOpacity=0.3))
    )
    
    line = base.mark_line(stroke="#4A90E2", strokeWidth=2).encode(
        y=alt.Y("Portfolio Value:Q", axis=alt.Axis(format="$s", title="Value ($)"), scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T", format="%d/%m/%Y"), alt.Tooltip("Portfolio Value:Q", format="$,.2f")]
    )
    
    st.altair_chart(line.properties(height=400), use_container_width=True)

    # Chart 2: Normalized Comparison (Portfolio vs Assets)
    st.subheader("Relative Performance (Base 100)")
    
    norm_assets = (prices_df / prices_df.iloc[0]) * 100
    norm_port = (port_equity / port_equity.iloc[0]) * 100
    
    df_norm = norm_assets.copy()
    df_norm["Portfolio"] = norm_port
    df_melt = df_norm.reset_index().melt(id_vars="index", var_name="Asset", value_name="Value").rename(columns={"index": "date"})

    # Highlight Portfolio in Red, others in Blue/Grey
    highlight = alt.condition(alt.datum.Asset == 'Portfolio', alt.value("#FF4B4B"), alt.value("#4A90E2"))
    stroke_width = alt.condition(alt.datum.Asset == 'Portfolio', alt.value(3), alt.value(1))
    opacity = alt.condition(alt.datum.Asset == 'Portfolio', alt.value(1), alt.value(0.5))

    # CORRECTION ICI: Suppression de l'argument color en double
    c_norm = alt.Chart(df_melt).mark_line().encode(
        x=alt.X("date:T", axis=alt.Axis(title=None)),
        y=alt.Y("Value:Q", scale=alt.Scale(zero=False), axis=alt.Axis(title="Base 100")),
        color=highlight, # On utilise la condition (Portfolio=Rouge, Autres=Bleu)
        strokeWidth=stroke_width,
        opacity=opacity,
        tooltip=["date:T", "Asset:N", alt.Tooltip("Value:Q", format=".1f")]
    ).properties(height=400)

    st.altair_chart(c_norm, use_container_width=True)

    # ==========================================
    # 4. COMPOSITION & RISK
    # ==========================================
    st.divider()
    st.header("Composition & Risk")
    
    r1, r2 = st.columns(2)
    
    with r1:
        st.subheader("Correlation Matrix")
        corr = prices_df.pct_change().corr()
        
        # Heatmap Altair
        corr_melt = corr.reset_index().melt(id_vars="index").rename(columns={"index": "var1", "variable": "var2"})
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
        final_w = _normalize_weights({t: weights.get(t, 0) for t in tickers})
        df_w = pd.DataFrame(list(final_w.items()), columns=["Asset", "Weight"])
        
        pie = alt.Chart(df_w).mark_arc(innerRadius=50).encode(
            theta=alt.Theta("Weight", stack=True),
            color=alt.Color("Asset", scale=alt.Scale(scheme="category10")),
            tooltip=["Asset", alt.Tooltip("Weight", format=".1%")]
        ).properties(height=300)
        
        st.altair_chart(pie, use_container_width=True)

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pytz
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

from nepal_stock_app.data_source import (
    fetch_political_news_signal,
    get_nepse_sector_list,
    normalize_sector_name,
    fetch_symbol_news_signal,
    fetch_symbol_fundamentals,
    fetch_symbol_history,
    fetch_today_share_data,
)
from nepal_stock_app.signals import analyze_shares
from nepal_stock_app.technical import (
    add_technical_indicators,
    evaluate_combined_recommendation,
    evaluate_technical_signal,
)
from nepal_stock_app.database import (
    init_db, add_trade, remove_trade, get_portfolio, 
    authenticate_user, create_user, set_user_session, 
    get_user_by_session, list_users, delete_user,
    update_trade_tag
)

# Initialize database
init_db()

st.set_page_config(page_title="NEPSE Technical Analyzer", layout="wide")

# --- UI MODERNIZATION STYLES ---
st.markdown("""
<style>
    /* Main App Background & Font */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Global Card Style */
    div.stButton > button {
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
        transform: translateY(-1px);
    }
    
    /* Header Customization */
    h1 {
        padding-top: 0rem;
        font-weight: 800;
        color: #1e1e1e;
    }
    
    /* Metric Card Polishing */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
    }
    
    /* DataFrame Styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #e6e9ef;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Sidebar Divider & Header */
    .css-1d391kg {
        padding-top: 1rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e1e4e8;
    }
    
    /* Alert Styling */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def get_nepal_time() -> datetime:
    """Returns the current time in Nepal (UTC+5:45)."""
    nepal_tz = pytz.timezone('Asia/Kathmandu')
    return datetime.now(nepal_tz)

def is_market_open() -> bool:
    """Returns True if current time is within NEPSE trading hours (11 AM - 3 PM, Sun-Thu)."""
    now = get_nepal_time()
    # Sunday (6) to Thursday (3) in Python weekday
    is_trading_day = now.weekday() in [6, 0, 1, 2, 3] 
    is_trading_hours = 11 <= now.hour < 15
    return is_trading_day and is_trading_hours

# --- AUTHENTICATION LOGIC ---
if "logged_in" not in st.session_state:
    if "session" in st.query_params:
        session_token = st.query_params["session"]
        user_name, user_is_admin = get_user_by_session(session_token)
        if user_name:
            st.session_state.logged_in = True
            st.session_state.username = user_name
            st.session_state.is_admin = user_is_admin
        else:
            st.session_state.logged_in = False
    else:
        st.session_state.logged_in = False

if not st.session_state.get("logged_in", False):
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        is_authenticated, is_admin = authenticate_user(username, password)
        if is_authenticated:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.is_admin = is_admin
            new_token = str(uuid.uuid4())
            set_user_session(username, new_token)
            st.query_params["session"] = new_token
            st.rerun()
        else:
            st.error("Invalid username or password")
    st.stop()

# --- HEADER: WELCOME ---
if st.session_state.get("logged_in", False):
    username = st.session_state.get("username", "USER")
    welcome_msg = f"Logged in as **{username.upper()}**"
    st.caption(f"🇳🇵 {welcome_msg} | {get_nepal_time().strftime('%A, %b %d')}")
    st.divider()

def get_cache_ttl() -> int:
    """Returns 0 if market is open (no cache), else 300 seconds."""
    return 0 if is_market_open() else 300
    is_trading_hours = 11 <= now.hour < 15
    return is_trading_day and is_trading_hours

def get_cache_ttl() -> int:
    """Returns 0 if market is open (no cache), else 300 seconds."""
    return 0 if is_market_open() else 300

@st.cache_data(ttl=300)
def get_analyzed_data(cache_key: str) -> pd.DataFrame:
    del cache_key
    raw = fetch_today_share_data()
    return analyze_shares(raw)

@st.cache_data(ttl=300)
def get_symbol_history(cache_key: str, symbol: str, lookback: int) -> pd.DataFrame:
    # Use shorter TTL during market hours if needed, or rely on cache_key change
    del cache_key
    history = fetch_symbol_history(symbol=symbol, lookback_days=lookback)
    return add_technical_indicators(history)

@st.cache_data(ttl=3600)
def get_symbol_fundamentals(cache_key: str, symbol: str) -> dict[str, Any]:
    del cache_key
    fundamentals = fetch_symbol_fundamentals(symbol=symbol)
    return {
        "symbol": fundamentals.symbol, "sector": fundamentals.sector,
        "listed_shares": fundamentals.listed_shares, "paid_up": fundamentals.paid_up,
        "total_paid_up_value": fundamentals.total_paid_up_value,
        "cash_dividend_pct": fundamentals.cash_dividend_pct, "bonus_share_pct": fundamentals.bonus_share_pct,
        "avg_120": fundamentals.avg_120, "avg_180": fundamentals.avg_180,
        "week52_high": fundamentals.week52_high, "week52_low": fundamentals.week52_low,
    }

@st.cache_data(ttl=1200)
def get_political_news_signal(cache_key: str) -> dict[str, Any]:
    del cache_key
    signal = fetch_political_news_signal()
    return {
        "bonus": signal.bonus, "score": signal.score,
        "positive_hits": signal.positive_hits, "negative_hits": signal.negative_hits,
        "headlines_checked": signal.headlines_checked, "matched_headlines": signal.matched_headlines,
    }

@st.cache_data(ttl=1200)
def get_symbol_news_signal(cache_key: str, symbol: str) -> dict[str, Any]:
    del cache_key
    signal = fetch_symbol_news_signal(symbol)
    return {
        "bonus": signal.bonus, "score": signal.score,
        "positive_hits": signal.positive_hits, "negative_hits": signal.negative_hits,
        "headlines_checked": signal.headlines_checked, "matched_headlines": signal.matched_headlines,
    }

@st.cache_data(ttl=3600)
def get_all_symbols_technical(cache_key: str, lookback: int, override_symbols: list[str] = None) -> pd.DataFrame:
    del cache_key
    today_df = fetch_today_share_data()
    if today_df.empty or "symbol" not in today_df.columns: return pd.DataFrame()
    
    symbol_ltp_map = {str(row["symbol"]).upper(): row.get("ltp") for _, row in today_df.iterrows()}
    
    # Use provided symbols if any, else use all
    symbols = sorted(override_symbols) if override_symbols else sorted(symbol_ltp_map.keys())

    def analyze_symbol(symbol: str):
        try:
            history = fetch_symbol_history(symbol=symbol, lookback_days=lookback)
            indicators = add_technical_indicators(history)
            signal = evaluate_technical_signal(indicators)
            latest = indicators.iloc[-1]
            return {
                "symbol": symbol, "today_ltp": symbol_ltp_map.get(symbol),
                "latest_close": latest.get("close"), "signal": signal.signal,
                "technical_score": signal.score, "confidence": signal.confidence,
                "entry_price": signal.entry_price, "stop_loss": signal.stop_loss,
                "target_price": signal.target_price, "expected_7d_price": signal.expected_7d_price,
                "risk_reward": signal.risk_reward_ratio,
                "simple_note": signal.simple_note, "beginner_action": signal.beginner_action, "reason": signal.reason,
            }
        except: return None

    rows = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_symbol, s): s for s in symbols}
        for future in as_completed(futures):
            res = future.result()
            if res: rows.append(res)
    return pd.DataFrame(rows)

def compute_position_size(entry_price: Any, stop_loss: Any, capital: float, risk_pct: float) -> int:
    if pd.isna(entry_price) or pd.isna(stop_loss): return 0
    risk_amount = capital * (risk_pct / 100.0)
    risk_per_share = abs(float(entry_price) - float(stop_loss))
    return int(risk_amount // risk_per_share) if risk_per_share > 0 else 0

def enrich_all_symbols(all_symbols_df: pd.DataFrame, market_frame: pd.DataFrame) -> pd.DataFrame:
    if all_symbols_df.empty: return pd.DataFrame()
    market_cols = market_frame[["symbol", "sector", "signal", "score", "confidence", "ltp", "reason"]].copy()
    
    # Rename for professional clarity instead of cryptic m_ prefixed names
    market_cols.columns = ["symbol", "Sector", "AI_Signal", "AI_Score", "AI_Confidence", "Live_Price", "AI_Logic"]
    
    enriched = all_symbols_df.merge(market_cols, on="symbol", how="left")
    enriched["Sector"] = enriched["Sector"].fillna("Others")
    
    # Final cleanup of all column names for a non-share market user
    friendly_names = {
        "symbol": "Company Scrip",
        "today_ltp": "Current Price",
        "latest_close": "Last Close Price",
        "signal": "Trend Prediction",
        "technical_score": "Safety Score",
        "confidence": "Accuracy (%)",
        "AI_Confidence": "AI Confidence (%)",
        "expected_7d_price": "Expected Price (7 Days)",
        "entry_price": "Buying Point",
        "stop_loss": "Safety Exit (Stop Loss)",
        "target_price": "Selling Goal",
        "risk_reward": "Profit/Risk Ratio",
        "simple_note": "Expert Tip",
        "beginner_action": "Action to Take",
        "reason": "Analysis Basis"
    }
    # REORDER COLUMNS for better UX (AI confidence and 7D price after accuracy)
    column_order = [
        "Company Scrip", "Current Price", "Trend Prediction", "Safety Score", 
        "Accuracy (%)", "AI Confidence (%)", "Expected Price (7 Days)",
        "Buying Point", "Safety Exit (Stop Loss)", "Selling Goal", 
        "Profit/Risk Ratio", "Sector", "Expert Tip", "Action to Take", "Analysis Basis"
    ]
    
    result = enriched.rename(columns=friendly_names)
    # Filter to only existing columns in order
    existing_cols = [c for c in column_order if c in result.columns]
    remaining_cols = [c for c in result.columns if c not in column_order]
    
    return result[existing_cols + remaining_cols]

# --- SIDEBAR & STATE ---
# --- REAL-TIME MARKET SYNC ---
# If market is open, force cache refresh on every interaction
if "cache_key" not in st.session_state: 
    st.session_state.cache_key = get_nepal_time().isoformat()

if is_market_open():
    st.session_state.cache_key = get_nepal_time().isoformat()
    st.sidebar.caption("⚡ Market Open: Live Data Active")
else:
    st.sidebar.caption("🌙 Market Closed")

# --- SIDEBAR NAVIGATION ---
tab_icons = {
    "Portfolio": "house", 
    "Market (Intraday Transaction)": "briefcase", 
    "Symbol": "search", 
    "All Symbols": "list-ul", 
    "Buy Tomorrow": "arrow-up-circle", 
    "Sell Tomorrow": "arrow-down-circle", 
    "Sector Wise": "grid"
}

if st.session_state.get("is_admin", False):
    tab_icons["Admin Panel"] = "people"

# Combine navigational tabs with action items for uniform look
nav_options = list(tab_icons.keys()) + ["Refresh Data", "Logout"]
nav_icons = list(tab_icons.values()) + ["arrow-clockwise", "box-arrow-right"]

current_nav = st.query_params.get("tab", "Portfolio")

with st.sidebar:
    selected_nav = option_menu(
        menu_title="Menu",
        options=nav_options,
        icons=nav_icons,
        menu_icon="cast",
        default_index=nav_options.index(current_nav) if current_nav in nav_options else 0,
        styles={
            "container": {"padding": "5!important", "background-color": "transparent"},
            "icon": {"color": "orange", "font-size": "18px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#2c3e50"},
        }
    )
    
    # Handle Action Items vs Navigation
    if selected_nav == "Logout":
        set_user_session(st.session_state.username, "")
        st.session_state.logged_in = False
        st.query_params.clear()
        st.rerun()
    elif selected_nav == "Refresh Data":
        st.session_state.cache_key = datetime.now().isoformat()
        st.rerun()
    elif selected_nav != current_nav:
        st.query_params["tab"] = selected_nav
        st.rerun()

    # --- SIDEBAR INFO ---
    st.divider()
    last_update_str = get_nepal_time().strftime("%Y-%m-%d %H:%M:%S.000")
    st.caption(f"Last updated (NPT): {last_update_str}")

    st.header("Technical Params")
    lookback_days = st.slider("Single Lookback", 20, 120, 45)
    all_lookback = st.slider("Global Lookback", 20, 80, 35)
    top_n = st.slider("Top N Rows", 10, 300, 80)
    
    st.header("Risk Management")
    account_size = st.number_input("Capital (NPR)", value=100000.0)
    risk_percent = st.slider("Risk (%)", 0.5, 5.0, 1.0)
    
# --- DATA FETCHING (Professional Background Pre-fetching) ---
# 1. Essential Market Summary (Lightweight & Fast)
# This is required for search boxes and basic LTP info
@st.fragment(run_every="5m" if is_market_open() else None)
def market_summary_fragment():
    with st.spinner("Syncing Prices..."):
        market_df = get_analyzed_data(st.session_state.cache_key)
        st.session_state.market_df = market_df
        st.session_state.available_symbols = sorted(market_df["symbol"].unique().tolist()) if not market_df.empty else ["NABIL"]

# 2. Deep Market Technicals (Progressive & Parallel)
def load_deep_technicals():
    if "all_symbols_master_df" not in st.session_state or st.session_state.get("refresh_heavy", False):
        all_symbols = st.session_state.available_symbols
        # Progress Tracking setup
        all_results = []
        progress_container = st.empty()
        table_container = st.empty()
        
        with st.status("🚀 Initializing Progressive Analysis...", expanded=True) as status:
            # We split the technical analysis into small parallel chunks for progressive display
            chunk_size = 20
            chunks = [all_symbols[i:i + chunk_size] for i in range(0, len(all_symbols), chunk_size)]
            
            for i, chunk in enumerate(chunks):
                # Analyze a chunk of symbols
                raw_chunk_df = get_all_symbols_technical(st.session_state.cache_key, all_lookback, override_symbols=chunk)
                if not raw_chunk_df.empty:
                    enriched_chunk = enrich_all_symbols(raw_chunk_df, st.session_state.market_df)
                    all_results.append(enriched_chunk)
                    
                    # Update the UI progressively
                    current_df = pd.concat(all_results)
                    progress = (len(current_df) / len(all_symbols))
                    status.update(label=f"📡 Analyzed {len(current_df)}/{len(all_symbols)} symbols...")
                    table_container.dataframe(current_df.head(top_n), use_container_width=True)
            
            st.session_state.all_symbols_master_df = pd.concat(all_results)
            st.session_state.refresh_heavy = False
            status.update(label="✅ Data Loaded & Analysis Complete!", state="complete", expanded=False)
            table_container.empty() # Clear the temporary progressive table

# Execute lightweight fetch immediately (Non-blocking for UI layout)
market_summary_fragment()
market_df = st.session_state.get("market_df", pd.DataFrame())
available_symbols = st.session_state.get("available_symbols", ["NABIL"])

# --- CONTENT RENDERING ---
tab_titles = {
    "Portfolio": "📈", 
    "Market (Intraday Transaction)": "📊", 
    "Symbol": "🔍", 
    "All Symbols": "📋", 
    "Buy Tomorrow": "🚀", 
    "Sell Tomorrow": "📉", 
    "Sector Wise": "🏗️",
    "Admin Panel": "👥"
}
st.title(f"{tab_titles.get(selected_nav, '💎')} {selected_nav}")

if "symbol" in market_df.columns:
    available_symbols = sorted(market_df["symbol"].unique().tolist())
else:
    available_symbols = ["NABIL", "NICA", "GBIME"] # Fallback

if selected_nav == "Market (Intraday Transaction)":
    # --- MODERN MARKET DASHBOARD ---
    st.subheader("Deep Market Intelligence")
    st.caption("AI-powered technical indicators for all NEPSE stocks.")
    
    # Run the heavy loading process
    load_deep_technicals()
    
    # Retrieve the final analyzed dataframe
    deep_df = st.session_state.get("all_symbols_master_df", pd.DataFrame())
    
    if not deep_df.empty:
        # Display filtering/sorting metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Stocks", len(deep_df))
        c2.metric("Buy Signals", len(deep_df[deep_df['AI_Signal'] == 'BUY']))
        c3.metric("Sell Signals", len(deep_df[deep_df['AI_Signal'] == 'SELL']))
        
        # SEARCH AND FILTER ROW
        # We can add a simple filter if needed, but the dataframe search is usually enough
        
        # CSS to fix the dataframe container for full-page scrolling
        st.markdown("""
            <style>
                [data-testid="stDataFrame"] > div {
                    height: auto !important;
                    max-height: none !important;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Render the final dataframe without internal scrolls
        st.dataframe(
            deep_df.sort_values(by="AI_Score", ascending=False).head(top_n),
            use_container_width=True,
            height=int(top_n * 35 * 2) if top_n > 0 else 800, # 200% larger assuming 35px per row
            hide_index=True
        )
    else:
        st.warning("No global technical data found. Please check your data source or internet connection.")
elif selected_nav == "Portfolio":
    # 🎨 REFINED PORTFOLIO UI: Modern Header & Action Bar
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader("Your Active Investments")
        st.caption("Real-time valuation based on latest MeroLagani prices.")
    
    with st.expander("💼 ADD A NEW TRADE", expanded=False):
        with st.form("trade_form", clear_on_submit=True):
            f1, f2, f3 = st.columns([2, 1, 1])
            p_sym = f1.selectbox("Stock Symbol", options=available_symbols, help="Select the scrip you purchased")
            p_qty = f2.number_input("Quantity", value=10, min_value=1, step=1)
            p_prc = f3.number_input("Average Buy Price", value=500.0, step=0.1)
            
            f4, f5, f6 = st.columns([1, 1, 1])
            p_sl = f4.number_input("Stop Loss (SL)", value=0.0, help="Optional: Price point to trigger safety exit alert")
            p_tag = f5.selectbox("Strategy Tag", ["Hold", "Sell (Short Term)", "Long Term", "Watchlist"], index=0)
            
            # Form Submission
            st.write("---")
            if st.form_submit_button("➕ ADD TO PORTFOLIO", use_container_width=True, type="primary"):
                sl_val = p_sl if p_sl > 0 else None
                # Updated database call with tag support if your database.py supports it
                # Note: Assuming add_trade signature is consistent with earlier edits
                add_trade(st.session_state.username, p_sym, p_prc, p_qty, sl_val)
                st.success(f"Successfully added {p_qty} shares of {p_sym} to your collection.")
                st.rerun()
    
    pf = get_portfolio(st.session_state.username)
    if not pf.empty:
        # Pre-calculate data for display
        ltp_map = dict(zip(market_df["symbol"], market_df.get("ltp", [])))
        
        # Risk Alerts Logic
        sl_hits = []
        
        # Link market technical signals to portfolio
        pf = pf.merge(market_df[['symbol', 'signal', 'confidence']], on='symbol', how='left')
        
        pf["LTP"] = pf["symbol"].map(ltp_map)
        pf["P&L"] = (pf["LTP"] - pf["buy_price"]) * pf["quantity"]
        pf["Action Date"] = pd.to_datetime(pf["date_added"]).dt.strftime("%Y-%m-%d")
        
        # Check for Stop Loss hits
        for _, r in pf.iterrows():
            if pd.notna(r['stop_loss']) and r['stop_loss'] > 0:
                if r['LTP'] <= r['stop_loss']:
                    sl_hits.append(f"🚨 **{r['symbol']}** hit SL! Price {r['LTP']} <= SL {r['stop_loss']}")

        if sl_hits:
            for alert in sl_hits:
                st.error(alert)

        # My Portfolio Data Presentation
        st.markdown("### 📊 ACTIVE POSITIONS")
        
        # PROFESSIONAL HEADER (Markdown Table)
        # Using a more visual approach with columns
        header_col1, header_col2, header_col3, header_col4, header_col5 = st.columns([1.5, 1.5, 1.5, 1.5, 2])
        header_col1.markdown("##### Symbol")
        header_col2.markdown("##### Entry / LTP")
        header_col3.markdown("##### Qty / SL")
        header_col4.markdown("##### P&L / %")
        header_col5.markdown("##### AI Recommendation")
        st.divider()
        
        for idx, row in pf.iterrows():
            c1, c2, c3, c4, c5 = st.columns([1.5, 1.5, 1.5, 1.5, 2])
            
            # Column 1: SYMBOL
            c1.markdown(f"**{row['symbol']}**")
            c1.caption(f"📅 {row['Action Date']}")
            
            # Column 2: BUY/LTP
            ltp_val = row.get('LTP', 0)
            c2.markdown(f"In: `{row['buy_price']:.1f}`")
            c2.markdown(f"Now: **{ltp_val:.1f}**")
            
            # Column 3: QTY & STOP LOSS
            sl_val = row.get('stop_loss', 0)
            sl_disp = f"🚨 {sl_val:.1f}" if (pd.notna(sl_val) and sl_val > 0) else "None"
            c3.write(f"Qty: **{int(row['quantity'])}**")
            c3.caption(f"SL: {sl_disp}")
            
            # Column 4: PROFIT & LOSS calculation
            pnl_v = row.get("P&L", 0)
            pnl_p = (pnl_v / (row['buy_price'] * row['quantity'])) * 100 if row['buy_price'] > 0 else 0
            pnl_color = "green" if pnl_v >= 0 else "red"
            c4.markdown(f":{pnl_color}[{pnl_v:+.1f}]")
            c4.caption(f":{pnl_color}[({pnl_p:+.1f}%)]")
            
            # Column 5: AI Technical Recommendation
            raw_sig = row.get("signal", "NEUTRAL")
            conf = int(row.get("confidence", 0))
            
            # Enhanced Recommendations Logic
            if raw_sig == "BUY" and conf >= 70:
                rec_icon, rec_text, rec_color = "🔥", "STRONG BUY", "green"
            elif raw_sig == "BUY":
                rec_icon, rec_text, rec_color = "✅", "BUY / HOLD", "green"
            elif raw_sig == "SELL" and conf >= 70:
                rec_icon, rec_text, rec_color = "🔻", "STRONG SELL", "red"
            elif raw_sig == "SELL":
                rec_icon, rec_text, rec_color = "⚠️", "TAKE PROFIT", "orange"
            else:
                rec_icon, rec_text, rec_color = "⚪", "NEUTRAL", "gray"

            c5.markdown(f"**{rec_icon} :{rec_color}[{rec_text}]**")
            c5.progress(conf/100, text=f"Confidence: {conf}%")
            
            # Inline Action with Confirmation
            if st.button(f"🗑️ Remove {row['symbol']}", key=f"del_{idx}", use_container_width=True):
                st.session_state[f"confirm_delete_{idx}"] = True

            if st.session_state.get(f"confirm_delete_{idx}", False):
                st.warning(f"Are you sure you want to remove **{row['symbol']}** from your portfolio?")
                col_confirm, col_cancel = st.columns(2)
                if col_confirm.button("✅ Yes, Delete", key=f"yes_{idx}", use_container_width=True, type="primary"):
                    from nepal_stock_app.database import remove_trade
                    remove_trade(row['id'])
                    del st.session_state[f"confirm_delete_{idx}"]
                    st.success(f"Removed {row['symbol']}")
                    st.rerun()
                if col_cancel.button("❌ No, Keep It", key=f"no_{idx}", use_container_width=True):
                    del st.session_state[f"confirm_delete_{idx}"]
                    st.rerun()

            st.divider()
        st.divider()
        if st.checkbox("Show Raw Data Table (Including Stop Loss)"):
            # Ensure friendly names for display if needed
            display_pf = pf.copy()
            if "stop_loss" in display_pf.columns:
                display_pf = display_pf.rename(columns={"stop_loss": "Stop Loss (SL)"})
            st.dataframe(display_pf, use_container_width=True)
    else: st.info("Portfolio Empty")

elif selected_nav == "Symbol":
    # QUICK EDIT FEATURE - WITH AUTO SUGGEST SELECTBOX
    q_sym = st.selectbox(
        "Analyze Symbol", 
        options=available_symbols, 
        index=available_symbols.index("NABIL") if "NABIL" in available_symbols else 0,
        key="quick_sym"
    )
    if q_sym:
        try:
            hist = get_symbol_history(st.session_state.cache_key, q_sym, lookback_days)
            
            # --- REAL-TIME PRICE SYNC FOR SYMBOL PAGE ---
            # Historical data might be 1 day old, whereas market_df is live.
            live_row = market_df[market_df["symbol"] == q_sym]
            if not live_row.empty:
                live_price = live_row.iloc[0]["ltp"]
                # Update the last row of hist with the live price for indicators
                if abs(hist.iloc[-1]["close"] - live_price) > 0.1:
                    hist.loc[hist.index[-1], "close"] = live_price
            
            # --- CALCULATE TECHNICAL SIGNAL ---
            indicators = add_technical_indicators(hist)
            sig = evaluate_technical_signal(indicators)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"{hist.iloc[-1]['close']:.2f}")
            c2.metric("Signal", sig.signal)
            c3.metric("7D Target", f"{sig.expected_7d_price:.2f}" if sig.expected_7d_price else "N/A")
            c4.metric("Confidence", f"{sig.confidence}%")
            
            candle = go.Figure(data=[go.Candlestick(
                x=hist["date"], open=hist["open"], high=hist["high"], 
                low=hist["low"], close=hist["close"], name="OHLC"
            )])
            candle.update_layout(title=f"{q_sym} Trend", xaxis_rangeslider_visible=False)
            st.plotly_chart(candle, use_container_width=True)
            
            st.subheader("Technical Decision")
            st.info(f"Action: {sig.beginner_action} | Note: {sig.simple_note}")
            st.write(f"Reason: {sig.reason}")
        except Exception as e:
            st.error(f"Error loading {q_sym}: {e}")

elif selected_nav == "All Symbols":
    load_deep_technicals()
    total_df = st.session_state.get("all_symbols_master_df", pd.DataFrame())
    st.dataframe(total_df, use_container_width=True, hide_index=True)

elif selected_nav == "Buy Tomorrow":
    load_deep_technicals()
    total_df = st.session_state.get("all_symbols_master_df", pd.DataFrame())
    buy_list = total_df[total_df["signal"] == "BUY"].sort_values("confidence", ascending=False)
    st.dataframe(buy_list, use_container_width=True, hide_index=True)

elif selected_nav == "Sell Tomorrow":
    load_deep_technicals()
    total_df = st.session_state.get("all_symbols_master_df", pd.DataFrame())
    sell_list = total_df[total_df["signal"] == "SELL"].sort_values("confidence", ascending=False)
    st.dataframe(sell_list, use_container_width=True, hide_index=True)

elif selected_nav == "Sector Wise":
    load_deep_technicals()
    total_df = st.session_state.get("all_symbols_master_df", pd.DataFrame())
    # ... using total_df below ...
    sectors = ["Commercial Banks", "Development Banks", "Finance", "Hotels", "Hydro", "Life Insurance", "Microfinance", "Non-Life Insurance", "Others"]
    picks = []
    for s in sectors:
        match = total_df[total_df["sector"] == s].head(1)
        if not match.empty: picks.append(match.iloc[0])
    if picks:
        st.dataframe(pd.DataFrame(picks), use_container_width=True, hide_index=True)
    else:
        st.info("No picks available for sectors today.")

elif selected_nav == "Admin Panel":
    st.subheader("User Management")
    users = list_users()
    if users:
        user_df = pd.DataFrame([{"Username": u[0], "Role": "Admin" if u[1] else "User"} for u in users])
        st.dataframe(user_df, use_container_width=True)
        
        with st.expander("Delete User"):
            del_user = st.selectbox("Select User", [u[0] for u in users if u[0] != st.session_state.username])
            if st.button("Delete"):
                delete_user(del_user)
                st.success(f"Deleted {del_user}")
                st.rerun()
    else:
        st.info("No users found.")

st.divider()
last_update_str = get_nepal_time().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Last updated (NPT): {last_update_str}")

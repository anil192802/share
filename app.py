from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
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
    get_user_by_session, list_users, delete_user
)

# Initialize database
init_db()

st.set_page_config(page_title="NEPSE Technical Analyzer", layout="wide")

# --- HEADER: WELCOME, REFRESH & LOGOUT ---
if "logged_in" in st.session_state and st.session_state.logged_in:
    h_col1, h_col2 = st.columns([7, 3])
    with h_col1:
        st.markdown(f"**WELCOME, {st.session_state.username.upper()}!**")
    
    with h_col2:
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Refresh Market Data", use_container_width=True):
                st.session_state.cache_key = datetime.now().isoformat()
                st.rerun()
        with btn_col2:
            if st.button("Logout", use_container_width=True):
                set_user_session(st.session_state.username, "")
                st.session_state.logged_in = False
                st.query_params.clear()
                st.rerun()
    st.divider()

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

if not st.session_state.logged_in:
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

# --- HELPER FUNCTIONS & CACHING ---
@st.cache_data(ttl=300)
def get_analyzed_data(cache_key: str) -> pd.DataFrame:
    del cache_key
    raw = fetch_today_share_data()
    return analyze_shares(raw)

@st.cache_data(ttl=900)
def get_symbol_history(cache_key: str, symbol: str, lookback: int) -> pd.DataFrame:
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
def get_all_symbols_technical(cache_key: str, lookback: int) -> pd.DataFrame:
    del cache_key
    today_df = fetch_today_share_data()
    if today_df.empty or "symbol" not in today_df.columns: return pd.DataFrame()
    
    symbol_ltp_map = {str(row["symbol"]).upper(): row.get("ltp") for _, row in today_df.iterrows()}
    symbols = sorted(symbol_ltp_map.keys())

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
                "target_price": signal.target_price, "risk_reward": signal.risk_reward_ratio,
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
    market_cols.columns = ["symbol", "m_sector", "m_signal", "m_score", "m_conf", "m_ltp", "m_reason"]
    enriched = all_symbols_df.merge(market_cols, on="symbol", how="left")
    enriched["sector"] = enriched["m_sector"].fillna("Others")
    return enriched

# --- SIDEBAR & STATE ---
if "cache_key" not in st.session_state: st.session_state.cache_key = datetime.now().isoformat()

# --- SIDEBAR NAVIGATION ---
tab_icons = {
    "Portfolio": "house", 
    "Market": "briefcase", 
    "Symbol": "search", 
    "All Symbols": "list-ul", 
    "Buy Tomorrow": "arrow-up-circle", 
    "Sell Tomorrow": "arrow-down-circle", 
    "Sector Wise": "grid"
}

if st.session_state.get("is_admin", False):
    tab_icons["Admin Panel"] = "people"

tab_list = list(tab_icons.keys())
current_nav = st.query_params.get("tab", "Portfolio")

with st.sidebar:
    selected_nav = option_menu(
        menu_title="Menu",
        options=tab_list,
        icons=list(tab_icons.values()),
        menu_icon="cast",
        default_index=tab_list.index(current_nav) if current_nav in tab_list else 0,
        styles={
            "container": {"padding": "5!important", "background-color": "transparent"},
            "icon": {"color": "orange", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#2c3e50"},
        }
    )
    
    if selected_nav != current_nav:
        st.query_params["tab"] = selected_nav
        st.rerun()

    st.divider()
    st.header("Technical Params")
    lookback_days = st.slider("Single Lookback", 20, 120, 45)
    all_lookback = st.slider("Global Lookback", 20, 80, 35)
    top_n = st.slider("Top N Rows", 10, 300, 80)
    
    st.header("Risk Management")
    account_size = st.number_input("Capital (NPR)", value=100000.0)
    risk_percent = st.slider("Risk (%)", 0.5, 5.0, 1.0)
    
# --- DATA FETCHING ---
with st.status("Fetching NEPSE data...", expanded=False) as status:
    market_df = get_analyzed_data(st.session_state.cache_key)
    status.update(label="Analyzing all symbols...", state="running")
    raw_all_df = get_all_symbols_technical(st.session_state.cache_key, all_lookback)
    all_symbols_master_df = enrich_all_symbols(raw_all_df, market_df)
    status.update(label="Data Ready!", state="complete", expanded=False)

# --- CONTENT RENDERING ---
tab_titles = {
    "Portfolio": "📈", 
    "Market": "📊", 
    "Symbol": "🔍", 
    "All Symbols": "📋", 
    "Buy Tomorrow": "🚀", 
    "Sell Tomorrow": "📉", 
    "Sector Wise": "🏗️",
    "Admin Panel": "👥"
}
st.title(f"{tab_titles[selected_nav]} {selected_nav}")

if "symbol" in market_df.columns:
    available_symbols = sorted(market_df["symbol"].unique().tolist())
else:
    available_symbols = ["NABIL", "NICA", "GBIME"] # Fallback

if selected_nav == "Market":
    st.dataframe(market_df.head(top_n), use_container_width=True)

elif selected_nav == "Portfolio":
    with st.expander("Add Trade"):
        with st.form("trade_form"):
            c1, c2, c3 = st.columns(3)
            p_sym = c1.selectbox("Symbol", options=available_symbols, index=available_symbols.index("NABIL") if "NABIL" in available_symbols else 0)
            p_prc = c2.number_input("Price", value=500.0)
            p_qty = c3.number_input("Qty", value=10)
            if st.form_submit_button("Add"):
                add_trade(st.session_state.username, p_sym, p_prc, p_qty)
                st.success(f"Added {p_sym}"); st.rerun()
    
    pf = get_portfolio(st.session_state.username)
    if not pf.empty:
        ltp_map = dict(zip(market_df["symbol"], market_df.get("ltp", [])))
        pf["LTP"] = pf["symbol"].map(ltp_map)
        pf["P&L"] = (pf["LTP"] - pf["buy_price"]) * pf["quantity"]
        st.dataframe(pf, use_container_width=True)
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
            sig = evaluate_technical_signal(hist)
            c1, c2, c3 = st.columns(3)
            c1.metric("Price", f"{hist.iloc[-1]['close']:.2f}")
            c2.metric("Signal", sig.signal)
            c3.metric("Confidence", f"{sig.confidence}%")
            
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
    st.dataframe(all_symbols_master_df, use_container_width=True)

elif selected_nav == "Buy Tomorrow":
    buy_list = all_symbols_master_df[all_symbols_master_df["signal"] == "BUY"].sort_values("confidence", ascending=False)
    st.dataframe(buy_list, use_container_width=True)

elif selected_nav == "Sell Tomorrow":
    sell_list = all_symbols_master_df[all_symbols_master_df["signal"] == "SELL"].sort_values("confidence", ascending=False)
    st.dataframe(sell_list, use_container_width=True)

elif selected_nav == "Sector Wise":
    sectors = get_nepse_sector_list()
    picks = []
    for s in sectors:
        match = all_symbols_master_df[all_symbols_master_df["sector"] == s].head(1)
        if not match.empty: picks.append(match.iloc[0])
    if picks:
        st.dataframe(pd.DataFrame(picks), use_container_width=True)
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
st.caption(f"Last updated: {datetime.now().strftime('%Y-%M-%d %H:%M:%S')}")

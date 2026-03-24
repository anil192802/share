from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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

st.set_page_config(page_title="NEPSE Technical Analyzer", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid username or password")
    st.stop()

st.title("NEPSE Technical Analyzer")
st.caption("Trend, candle, and indicator-based signal engine for Nepal stocks.")
st.info("No model can guarantee profit. Use stop loss, small position size, and discipline.")

with st.sidebar:
    st.header("User")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.header("Market Screener")
    signal_filter = st.selectbox("Intraday Signal", ["ALL", "BUY", "SELL", "HOLD"], index=0)
    top_n = st.slider("Top rows", min_value=10, max_value=300, value=80, step=10)

    st.header("Symbol Technical")
    symbol_input = st.text_input("Symbol", value="NABIL")
    lookback_days = st.slider("Lookback days", min_value=20, max_value=120, value=45, step=5)
    all_symbols_lookback_days = st.slider(
        "All symbols lookback (faster)",
        min_value=20,
        max_value=80,
        value=35,
        step=5,
    )
    all_symbols_max_load = st.slider(
        "All symbols max load",
        min_value=50,
        max_value=400,
        value=220,
        step=10,
    )

    st.header("Risk Settings")
    account_size = st.number_input("Total money (NPR)", min_value=1000.0, value=100000.0, step=1000.0)
    risk_percent = st.slider("Risk per trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

    st.header("Context Bonus")
    political_bonus = st.slider("Political scenario bonus", min_value=-2, max_value=2, value=0, step=1)
    news_bonus = st.slider("Symbol news bonus", min_value=-2, max_value=2, value=0, step=1)

    refresh = st.button("Refresh data")

if "cache_key" not in st.session_state:
    st.session_state.cache_key = datetime.now().isoformat()

if refresh:
    st.session_state.cache_key = datetime.now().isoformat()


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
        "symbol": fundamentals.symbol,
        "sector": fundamentals.sector,
        "listed_shares": fundamentals.listed_shares,
        "paid_up": fundamentals.paid_up,
        "total_paid_up_value": fundamentals.total_paid_up_value,
        "cash_dividend_pct": fundamentals.cash_dividend_pct,
        "bonus_share_pct": fundamentals.bonus_share_pct,
        "avg_120": fundamentals.avg_120,
        "avg_180": fundamentals.avg_180,
        "week52_high": fundamentals.week52_high,
        "week52_low": fundamentals.week52_low,
    }


@st.cache_data(ttl=1200)
def get_political_news_signal(cache_key: str) -> dict[str, Any]:
    del cache_key
    signal = fetch_political_news_signal()
    return {
        "bonus": signal.bonus,
        "score": signal.score,
        "positive_hits": signal.positive_hits,
        "negative_hits": signal.negative_hits,
        "headlines_checked": signal.headlines_checked,
        "matched_headlines": signal.matched_headlines,
    }


@st.cache_data(ttl=1200)
def get_symbol_news_signal(cache_key: str, symbol: str) -> dict[str, Any]:
    del cache_key
    signal = fetch_symbol_news_signal(symbol)
    return {
        "bonus": signal.bonus,
        "score": signal.score,
        "positive_hits": signal.positive_hits,
        "negative_hits": signal.negative_hits,
        "headlines_checked": signal.headlines_checked,
        "matched_headlines": signal.matched_headlines,
    }


def clamp_bonus(value: int) -> int:
    return max(-2, min(2, int(value)))


@st.cache_data(ttl=3600)
def get_all_symbols_technical(cache_key: str, lookback: int) -> pd.DataFrame:
    del cache_key
    empty_columns = [
        "symbol",
        "today_ltp",
        "latest_close",
        "signal",
        "confidence",
        "entry_price",
        "stop_loss",
        "target_price",
        "risk_reward",
        "simple_note",
        "beginner_action",
        "reason",
    ]

    today_df = fetch_today_share_data()
    if today_df.empty or "symbol" not in today_df.columns:
        return pd.DataFrame(columns=empty_columns)

    symbol_ltp_map: dict[str, Any] = {
        str(row["symbol"]).upper(): row.get("ltp") for _, row in today_df.iterrows()
    }
    symbols = sorted(symbol_ltp_map.keys())
    if not symbols:
        return pd.DataFrame(columns=empty_columns)

    def analyze_symbol(symbol: str) -> dict[str, Any]:
        try:
            history = fetch_symbol_history(symbol=symbol, lookback_days=lookback)
            indicators = add_technical_indicators(history)
            signal = evaluate_technical_signal(indicators)
            latest = indicators.iloc[-1]

            return {
                "symbol": symbol,
                "today_ltp": symbol_ltp_map.get(symbol),
                "latest_close": latest.get("close"),
                "signal": signal.signal,
                "technical_score": signal.score,
                "confidence": signal.confidence,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "target_price": signal.target_price,
                "risk_reward": signal.risk_reward_ratio,
                "simple_note": signal.simple_note,
                "beginner_action": signal.beginner_action,
                "reason": signal.reason,
            }
        except Exception:
            return {
                "symbol": symbol,
                "today_ltp": symbol_ltp_map.get(symbol),
                "latest_close": pd.NA,
                "signal": "NO DATA",
                "technical_score": 0,
                "confidence": 0,
                "entry_price": pd.NA,
                "stop_loss": pd.NA,
                "target_price": pd.NA,
                "risk_reward": pd.NA,
                "simple_note": "Could not load enough history",
                "beginner_action": "Skip",
                "reason": "History fetch failed",
            }

    rows: list[dict[str, Any]] = []
    max_workers = min(8, len(symbols))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(analyze_symbol, symbol): symbol for symbol in symbols}
        for future in as_completed(future_to_symbol):
            rows.append(future.result())

    if not rows:
        return pd.DataFrame(columns=empty_columns)

    result = pd.DataFrame(rows)
    for column in empty_columns:
        if column not in result.columns:
            result[column] = pd.NA

    signal_order = {"BUY": 0, "HOLD": 1, "SELL": 2, "NO DATA": 3}
    result["_order"] = result["signal"].map(signal_order).fillna(9)
    result = result.sort_values(by=["_order", "confidence"], ascending=[True, False]).drop(columns=["_order"])
    result = result.reset_index(drop=True)
    return result


def load_all_symbols_technical_with_progress(
    lookback: int,
    max_symbols: int,
    on_progress: Any | None = None,
) -> pd.DataFrame:
    empty_columns = [
        "symbol",
        "today_ltp",
        "latest_close",
        "signal",
        "technical_score",
        "confidence",
        "entry_price",
        "stop_loss",
        "target_price",
        "risk_reward",
        "simple_note",
        "beginner_action",
        "reason",
    ]

    today_df = fetch_today_share_data()
    if today_df.empty or "symbol" not in today_df.columns:
        return pd.DataFrame(columns=empty_columns)

    symbol_ltp_map: dict[str, Any] = {
        str(row["symbol"]).upper(): row.get("ltp") for _, row in today_df.iterrows()
    }
    symbols = sorted(symbol_ltp_map.keys())
    if max_symbols > 0:
        symbols = symbols[:max_symbols]
    if not symbols:
        return pd.DataFrame(columns=empty_columns)

    def analyze_symbol(symbol: str) -> dict[str, Any]:
        try:
            history = fetch_symbol_history(symbol=symbol, lookback_days=lookback)
            indicators = add_technical_indicators(history)
            signal = evaluate_technical_signal(indicators)
            latest = indicators.iloc[-1]

            return {
                "symbol": symbol,
                "today_ltp": symbol_ltp_map.get(symbol),
                "latest_close": latest.get("close"),
                "signal": signal.signal,
                "technical_score": signal.score,
                "confidence": signal.confidence,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "target_price": signal.target_price,
                "risk_reward": signal.risk_reward_ratio,
                "simple_note": signal.simple_note,
                "beginner_action": signal.beginner_action,
                "reason": signal.reason,
            }
        except Exception:
            return {
                "symbol": symbol,
                "today_ltp": symbol_ltp_map.get(symbol),
                "latest_close": pd.NA,
                "signal": "NO DATA",
                "technical_score": 0,
                "confidence": 0,
                "entry_price": pd.NA,
                "stop_loss": pd.NA,
                "target_price": pd.NA,
                "risk_reward": pd.NA,
                "simple_note": "Could not load enough history",
                "beginner_action": "Skip",
                "reason": "History fetch failed",
            }

    rows: list[dict[str, Any]] = []
    total = len(symbols)
    processed = 0

    max_workers = min(8, total)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(analyze_symbol, symbol): symbol for symbol in symbols}
        for future in as_completed(future_to_symbol):
            rows.append(future.result())
            processed += 1
            if on_progress is not None:
                on_progress(processed, total)

    if not rows:
        return pd.DataFrame(columns=empty_columns)

    result = pd.DataFrame(rows)
    for column in empty_columns:
        if column not in result.columns:
            result[column] = pd.NA

    signal_order = {"BUY": 0, "HOLD": 1, "SELL": 2, "NO DATA": 3}
    result["_order"] = result["signal"].map(signal_order).fillna(9)
    result = result.sort_values(by=["_order", "confidence"], ascending=[True, False]).drop(columns=["_order"])
    return result.reset_index(drop=True)


def compute_position_size(entry_price: Any, stop_loss: Any, capital: float, risk_pct: float) -> int:
    if pd.isna(entry_price) or pd.isna(stop_loss):
        return 0
    risk_amount = capital * (risk_pct / 100.0)
    risk_per_share = abs(float(entry_price) - float(stop_loss))
    if risk_per_share <= 0:
        return 0
    qty = int(risk_amount // risk_per_share)
    return max(qty, 0)


def build_fallback_all_symbols_from_market(market_frame: pd.DataFrame) -> pd.DataFrame:
    if market_frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "today_ltp",
                "latest_close",
                "signal",
                "technical_score",
                "confidence",
                "entry_price",
                "stop_loss",
                "target_price",
                "risk_reward",
                "simple_note",
                "beginner_action",
                "reason",
            ]
        )

    fallback = pd.DataFrame()
    fallback["symbol"] = market_frame["symbol"]
    fallback["sector"] = market_frame.get("sector", "Others")
    fallback["today_ltp"] = market_frame["ltp"]
    fallback["latest_close"] = market_frame["ltp"]
    fallback["signal"] = market_frame["signal"]
    fallback["technical_score"] = market_frame.get("score", 0)
    fallback["confidence"] = market_frame.get("confidence", 50)
    fallback["entry_price"] = market_frame["ltp"]
    fallback["stop_loss"] = market_frame["ltp"] * 0.97
    fallback["target_price"] = market_frame["ltp"] * 1.05
    fallback["risk_reward"] = 5 / 3
    fallback["simple_note"] = "Using intraday fallback signal"
    fallback["beginner_action"] = fallback["signal"].map(
        {"BUY": "Buy small", "SELL": "Book profit / avoid", "HOLD": "Wait"}
    ).fillna("Wait")
    fallback["reason"] = market_frame.get("reason", "Intraday fallback")
    return fallback


def enrich_all_symbols_with_market_fallback(all_symbols_df: pd.DataFrame, market_frame: pd.DataFrame) -> pd.DataFrame:
    market_cols = market_frame[["symbol", "sector", "signal", "score", "confidence", "ltp", "reason"]].rename(
        columns={
            "sector": "market_sector",
            "signal": "market_signal",
            "score": "market_score",
            "confidence": "market_confidence",
            "ltp": "market_ltp",
            "reason": "market_reason",
        }
    )

    if all_symbols_df.empty:
        return build_fallback_all_symbols_from_market(market_frame)

    enriched = all_symbols_df.merge(market_cols, on="symbol", how="left")

    no_data_mask = enriched["signal"].astype(str).eq("NO DATA")
    enriched["signal"] = enriched["signal"].where(~no_data_mask, enriched["market_signal"].fillna("HOLD"))
    enriched["technical_score"] = enriched["technical_score"].where(
        ~no_data_mask,
        enriched["market_score"].fillna(0),
    )
    enriched["confidence"] = enriched["confidence"].where(
        ~no_data_mask,
        enriched["market_confidence"].fillna(50),
    )
    if "sector" not in enriched.columns:
        enriched["sector"] = enriched["market_sector"]
    else:
        enriched["sector"] = enriched["sector"].fillna(enriched["market_sector"])
    enriched["today_ltp"] = enriched["today_ltp"].fillna(enriched["market_ltp"])
    enriched["latest_close"] = enriched["latest_close"].fillna(enriched["market_ltp"])
    enriched["entry_price"] = enriched["entry_price"].fillna(enriched["market_ltp"])
    enriched["stop_loss"] = enriched["stop_loss"].fillna(enriched["market_ltp"] * 0.97)
    enriched["target_price"] = enriched["target_price"].fillna(enriched["market_ltp"] * 1.05)
    enriched["risk_reward"] = enriched["risk_reward"].fillna(5 / 3)
    enriched["simple_note"] = enriched["simple_note"].where(~no_data_mask, "Using intraday fallback signal")
    enriched["beginner_action"] = enriched["beginner_action"].where(
        ~no_data_mask,
        enriched["signal"].map({"BUY": "Buy small", "SELL": "Book profit / avoid", "HOLD": "Wait"}).fillna("Wait"),
    )
    enriched["reason"] = enriched["reason"].where(~no_data_mask, enriched["market_reason"].fillna("Intraday fallback"))

    enriched = enriched.drop(
        columns=["market_sector", "market_signal", "market_score", "market_confidence", "market_ltp", "market_reason"],
        errors="ignore",
    )
    return enriched


try:
    market_df = get_analyzed_data(st.session_state.cache_key)
except Exception as exc:
    st.error(f"Could not fetch market data: {exc}")
    st.stop()

all_symbols_master_df = build_fallback_all_symbols_from_market(market_df)
all_symbols_master_df = enrich_all_symbols_with_market_fallback(all_symbols_master_df, market_df)
all_symbols_error = None

analyzed_df = market_df.copy()
if signal_filter != "ALL":
    analyzed_df = analyzed_df[analyzed_df["signal"] == signal_filter]

tab_market, tab_symbol, tab_all, tab_buy_tomorrow, tab_sell_tomorrow, tab_buy_sector = st.tabs(
    [
        "Market Signals",
        "Symbol Technical Analysis",
        "Daily All Symbols",
        "What to Buy Tomorrow",
        "What to Sell Tomorrow",
        "Buy Sector Wise",
    ]
)

with tab_market:
    st.metric("Total Scrips", len(analyzed_df))
    summary = analyzed_df["signal"].value_counts().rename_axis("signal").reset_index(name="count")
    st.subheader("Signal Summary")
    st.dataframe(summary, width="stretch", hide_index=True)

    st.subheader("Today Intraday Analysis")
    st.dataframe(
        analyzed_df.head(top_n)[
            ["symbol", "ltp", "open", "high", "low", "pct_change", "signal", "confidence", "reason"]
        ],
        width="stretch",
        hide_index=True,
    )

with tab_symbol:
    if not symbol_input.strip():
        st.warning("Enter a symbol in the sidebar to run technical analysis.")
        st.stop()

    try:
        symbol_df = get_symbol_history(st.session_state.cache_key, symbol_input.strip().upper(), lookback_days)
    except Exception as exc:
        st.error(f"Could not fetch historical data for {symbol_input.upper()}: {exc}")
        st.stop()

    technical_signal = evaluate_technical_signal(symbol_df)
    latest = symbol_df.iloc[-1]
    political_news = get_political_news_signal(st.session_state.cache_key)
    symbol_news = get_symbol_news_signal(st.session_state.cache_key, symbol_input.strip().upper())

    try:
        fundamentals = get_symbol_fundamentals(st.session_state.cache_key, symbol_input.strip().upper())
    except Exception:
        fundamentals = {
            "symbol": symbol_input.strip().upper(),
            "sector": None,
            "listed_shares": None,
            "paid_up": None,
            "total_paid_up_value": None,
            "cash_dividend_pct": None,
            "bonus_share_pct": None,
            "avg_120": None,
            "avg_180": None,
            "week52_high": None,
            "week52_low": None,
        }

    intraday_match = market_df[market_df["symbol"] == symbol_input.strip().upper()]
    intraday_score = int(intraday_match.iloc[0]["score"]) if not intraday_match.empty else 0
    effective_political_bonus = clamp_bonus(political_bonus + int(political_news.get("bonus", 0)))
    effective_symbol_bonus = clamp_bonus(news_bonus + int(symbol_news.get("bonus", 0)))

    combined = evaluate_combined_recommendation(
        indicator_df=symbol_df,
        technical_signal=technical_signal,
        intraday_score=intraday_score,
        political_bonus=effective_political_bonus,
        news_bonus=effective_symbol_bonus,
        fundamental_inputs=fundamentals,
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Symbol", symbol_input.strip().upper())
    col2.metric("Latest Close", f"{latest['close']:.2f}")
    col3.metric("Signal", technical_signal.signal)
    col4.metric("Confidence", f"{technical_signal.confidence}%")
    col5.metric(
        "Stop Loss",
        f"{technical_signal.stop_loss:.2f}" if technical_signal.stop_loss is not None else "-",
    )
    col6.metric(
        "Target",
        f"{technical_signal.target_price:.2f}" if technical_signal.target_price is not None else "-",
    )

    st.caption(f"Simple note: {technical_signal.simple_note}")
    st.caption(f"Beginner action: {technical_signal.beginner_action}")
    if technical_signal.risk_reward_ratio is not None:
        st.caption(f"Risk/Reward ratio: 1:{technical_signal.risk_reward_ratio:.2f}")
    st.caption(f"Reason: {technical_signal.reason}")

    suggested_qty = compute_position_size(
        technical_signal.entry_price,
        technical_signal.stop_loss,
        account_size,
        risk_percent,
    )
    if suggested_qty > 0 and technical_signal.entry_price is not None:
        est_cost = suggested_qty * technical_signal.entry_price
        st.success(
            f"Simple plan: Max quantity ≈ {suggested_qty} shares for your risk setting. "
            f"Estimated buy amount ≈ NPR {est_cost:,.0f}."
        )

    st.markdown("**Combined short-term and stable long-term recommendation**")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Short-term Signal", combined.short_term_signal)
    s2.metric("Stable Long-term Signal", combined.long_term_signal)
    s3.metric("Short-term Score", f"{combined.short_term_score:.2f}")
    s4.metric("Long-term Score", f"{combined.long_term_score:.2f}")

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Fundamental Score", combined.fundamental_score)
    f2.metric("Stability Score", combined.stability_score)
    f3.metric("Sector", fundamentals.get("sector") or "-")
    f4.metric("Cash Dividend %", f"{fundamentals['cash_dividend_pct']:.2f}" if fundamentals.get("cash_dividend_pct") is not None else "-")

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Political Bonus (auto + manual)", f"{effective_political_bonus:+d}")
    n2.metric("Symbol Bonus (auto + manual)", f"{effective_symbol_bonus:+d}")
    n3.metric("Political headlines checked", int(political_news.get("headlines_checked", 0)))
    n4.metric("Symbol headlines checked", int(symbol_news.get("headlines_checked", 0)))

    if political_news.get("matched_headlines"):
        st.caption("Political/market news signals:")
        for headline in political_news["matched_headlines"][:3]:
            st.write(f"- {headline}")

    if symbol_news.get("matched_headlines"):
        st.caption(f"{symbol_input.strip().upper()} news signals:")
        for headline in symbol_news["matched_headlines"][:3]:
            st.write(f"- {headline}")

    if combined.expected_profit_per_share is not None and combined.expected_loss_per_share is not None:
        expected_profit_total = combined.expected_profit_per_share * suggested_qty
        expected_loss_total = combined.expected_loss_per_share * suggested_qty
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Expected Profit/Share", f"NPR {combined.expected_profit_per_share:.2f}")
        p2.metric("Expected Loss/Share", f"NPR {combined.expected_loss_per_share:.2f}")
        p3.metric("Expected Profit (qty)", f"NPR {expected_profit_total:,.0f}")
        p4.metric("Expected Loss (qty)", f"NPR {expected_loss_total:,.0f}")

    st.caption(f"Combined rationale: {combined.rationale}")

    st.markdown("**Quick Checklist (Beginner)**")
    st.write("1) Check signal is BUY and confidence is above 60%")
    st.write("2) Never buy without stop loss")
    st.write("3) Risk only small amount per trade")
    st.write("4) If price hits stop loss, exit without emotion")

    candle = go.Figure(
        data=[
            go.Candlestick(
                x=symbol_df["date"],
                open=symbol_df["open"],
                high=symbol_df["high"],
                low=symbol_df["low"],
                close=symbol_df["close"],
                name="OHLC",
            ),
            go.Scatter(x=symbol_df["date"], y=symbol_df["sma20"], mode="lines", name="SMA20"),
            go.Scatter(x=symbol_df["date"], y=symbol_df["sma50"], mode="lines", name="SMA50"),
            go.Scatter(
                x=symbol_df["date"],
                y=symbol_df["bb_upper"],
                mode="lines",
                name="BB Upper",
                line={"dash": "dot"},
            ),
            go.Scatter(
                x=symbol_df["date"],
                y=symbol_df["bb_lower"],
                mode="lines",
                name="BB Lower",
                line={"dash": "dot"},
            ),
        ]
    )
    candle.update_layout(title=f"{symbol_input.strip().upper()} Candles + Trend Bands", xaxis_rangeslider_visible=False)
    st.plotly_chart(candle, use_container_width=True)

    indicators = go.Figure()
    indicators.add_trace(go.Scatter(x=symbol_df["date"], y=symbol_df["macd"], mode="lines", name="MACD"))
    indicators.add_trace(
        go.Scatter(x=symbol_df["date"], y=symbol_df["macd_signal"], mode="lines", name="MACD Signal")
    )
    indicators.add_trace(go.Scatter(x=symbol_df["date"], y=symbol_df["rsi14"], mode="lines", name="RSI14"))
    indicators.update_layout(title="Indicator Panel (MACD / RSI)")
    st.plotly_chart(indicators, use_container_width=True)

    st.subheader("History with indicators")
    st.dataframe(
        symbol_df[
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "sma20",
                "sma50",
                "macd",
                "macd_signal",
                "rsi14",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

with tab_all:
    st.subheader("Daily Buy/Sell List for All Symbols")
    st.caption(
        "Simple meaning: BUY = possible entry, SELL = possible exit/avoid, HOLD = wait. "
        "Stop Loss means where to exit if trade goes wrong."
    )

    all_symbols_df = all_symbols_master_df.copy()

    if all_symbols_error:
        st.error(f"Could not load Daily All Symbols data: {all_symbols_error}")

    if all_symbols_df.empty:
        st.warning("No symbol data is available right now. Please click Refresh data and try again.")
    else:
        all_symbols_df["max_qty"] = all_symbols_df.apply(
            lambda row: compute_position_size(row.get("entry_price"), row.get("stop_loss"), account_size, risk_percent),
            axis=1,
        )
        all_symbols_df["estimated_buy_amount"] = all_symbols_df.apply(
            lambda row: (row.get("entry_price") * row.get("max_qty"))
            if pd.notna(row.get("entry_price")) and row.get("max_qty", 0) > 0
            else pd.NA,
            axis=1,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Symbols", len(all_symbols_df))
        c2.metric("BUY", int((all_symbols_df["signal"] == "BUY").sum()))
        c3.metric("SELL", int((all_symbols_df["signal"] == "SELL").sum()))
        c4.metric("HOLD", int((all_symbols_df["signal"] == "HOLD").sum()))

        display_df = all_symbols_df.rename(
            columns={
                "symbol": "Symbol",
                "today_ltp": "Today Price",
                "latest_close": "Latest Close",
                "signal": "Signal",
                "confidence": "Confidence %",
                "entry_price": "Entry",
                "stop_loss": "Stop Loss",
                "target_price": "Target",
                "risk_reward": "Risk/Reward",
                "simple_note": "Simple Meaning",
                "beginner_action": "Action for Beginner",
                "max_qty": "Max Qty (risk-based)",
                "estimated_buy_amount": "Est. Buy Amount",
                "reason": "Why this signal",
            }
        )

        st.dataframe(display_df, width="stretch", hide_index=True)

        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download daily buy/sell list (CSV)",
            data=csv_data,
            file_name="nepse_daily_all_symbols_signals.csv",
            mime="text/csv",
        )

with tab_buy_tomorrow:
    st.subheader("What to Buy Tomorrow")
    st.caption("Symbols with BUY signal from current technical engine for next-session planning.")

    all_symbols_df = all_symbols_master_df.copy()

    if all_symbols_error:
        st.error(f"Could not load tomorrow buy list: {all_symbols_error}")

    if all_symbols_df.empty:
        st.warning("No data available to build tomorrow buy list.")
    else:
        political_news = get_political_news_signal(st.session_state.cache_key)
        auto_political_bonus = int(political_news.get("bonus", 0))

        buy_df = all_symbols_df.copy()
        buy_df = buy_df[buy_df["signal"].isin(["BUY", "HOLD"])].copy()
        buy_df = buy_df.sort_values(by=["confidence", "risk_reward"], ascending=[False, False]).head(top_n).reset_index(drop=True)

        symbol_news_bonus: list[int] = []
        news_examples: list[str] = []
        news_fetch_limit = min(30, len(buy_df))
        for idx, row in buy_df.iterrows():
            if idx >= news_fetch_limit:
                symbol_news_bonus.append(0)
                news_examples.append("-")
                continue

            symbol = str(row.get("symbol", ""))
            symbol_news = get_symbol_news_signal(st.session_state.cache_key, symbol)
            symbol_bonus = int(symbol_news.get("bonus", 0))
            symbol_news_bonus.append(symbol_bonus)
            matched = symbol_news.get("matched_headlines", [])
            news_examples.append(matched[0] if matched else "-")

        buy_df["political_bonus_auto"] = auto_political_bonus
        buy_df["symbol_bonus_auto"] = symbol_news_bonus
        buy_df["tomorrow_score"] = buy_df["technical_score"] + buy_df["political_bonus_auto"] + buy_df["symbol_bonus_auto"]
        buy_df["tomorrow_signal"] = buy_df["tomorrow_score"].apply(lambda value: "BUY" if value >= 2 else "HOLD")
        buy_df["news_example"] = news_examples

        strict_buy_df = buy_df[buy_df["tomorrow_signal"] == "BUY"].copy().reset_index(drop=True)
        if strict_buy_df.empty:
            st.info("No strong BUY signals right now. Showing top watchlist candidates for tomorrow.")
            display_buy_df = buy_df.sort_values(by=["tomorrow_score", "confidence"], ascending=[False, False]).head(top_n)
        else:
            display_buy_df = strict_buy_df.sort_values(by=["tomorrow_score", "confidence"], ascending=[False, False])

        st.metric("BUY candidates", len(strict_buy_df))
        if display_buy_df.empty:
            st.info("No symbols available for tomorrow buy analysis.")
        else:
            st.dataframe(
                display_buy_df[
                    [
                        "symbol",
                        "today_ltp",
                        "technical_score",
                        "political_bonus_auto",
                        "symbol_bonus_auto",
                        "tomorrow_score",
                        "tomorrow_signal",
                        "confidence",
                        "entry_price",
                        "stop_loss",
                        "target_price",
                        "risk_reward",
                        "news_example",
                        "simple_note",
                        "reason",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

with tab_sell_tomorrow:
    st.subheader("What to Sell Tomorrow")
    st.caption("Symbols with SELL signal from current technical engine for next-session planning.")

    all_symbols_df = all_symbols_master_df.copy()

    if all_symbols_error:
        st.error(f"Could not load tomorrow sell list: {all_symbols_error}")

    if all_symbols_df.empty:
        st.warning("No data available to build tomorrow sell list.")
    else:
        political_news = get_political_news_signal(st.session_state.cache_key)
        auto_political_bonus = int(political_news.get("bonus", 0))

        sell_df = all_symbols_df.copy()
        sell_df = sell_df[sell_df["signal"].isin(["SELL", "HOLD"])].copy()
        sell_df = sell_df.sort_values(by=["confidence", "risk_reward"], ascending=[False, True]).head(top_n).reset_index(drop=True)

        symbol_news_bonus: list[int] = []
        news_examples: list[str] = []
        news_fetch_limit = min(30, len(sell_df))
        for idx, row in sell_df.iterrows():
            if idx >= news_fetch_limit:
                symbol_news_bonus.append(0)
                news_examples.append("-")
                continue

            symbol = str(row.get("symbol", ""))
            symbol_news = get_symbol_news_signal(st.session_state.cache_key, symbol)
            symbol_bonus = int(symbol_news.get("bonus", 0))
            symbol_news_bonus.append(symbol_bonus)
            matched = symbol_news.get("matched_headlines", [])
            news_examples.append(matched[0] if matched else "-")

        sell_df["political_bonus_auto"] = auto_political_bonus
        sell_df["symbol_bonus_auto"] = symbol_news_bonus
        sell_df["tomorrow_score"] = sell_df["technical_score"] + sell_df["political_bonus_auto"] + sell_df["symbol_bonus_auto"]
        sell_df["tomorrow_signal"] = sell_df["tomorrow_score"].apply(lambda value: "SELL" if value <= -2 else "HOLD")
        sell_df["news_example"] = news_examples

        strict_sell_df = sell_df[sell_df["tomorrow_signal"] == "SELL"].copy().reset_index(drop=True)
        if strict_sell_df.empty:
            st.info("No strong SELL signals right now. Showing top risk watchlist for tomorrow.")
            display_sell_df = sell_df.sort_values(by=["tomorrow_score", "confidence"], ascending=[True, False]).head(top_n)
        else:
            display_sell_df = strict_sell_df.sort_values(by=["tomorrow_score", "confidence"], ascending=[True, False])

        st.metric("SELL candidates", len(strict_sell_df))
        if display_sell_df.empty:
            st.info("No symbols available for tomorrow sell analysis.")
        else:
            st.dataframe(
                display_sell_df[
                    [
                        "symbol",
                        "today_ltp",
                        "technical_score",
                        "political_bonus_auto",
                        "symbol_bonus_auto",
                        "tomorrow_score",
                        "tomorrow_signal",
                        "confidence",
                        "entry_price",
                        "stop_loss",
                        "target_price",
                        "risk_reward",
                        "news_example",
                        "simple_note",
                        "reason",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

with tab_buy_sector:
    st.subheader("Buy Sector Wise")
    st.caption("Daily list with one top symbol per NEPSE sector, quantity fixed to 1 share.")

    all_symbols_df = all_symbols_master_df.copy()
    if all_symbols_error:
        st.error(f"Could not load sector-wise buy data: {all_symbols_error}")
    if all_symbols_df.empty:
        st.warning("No symbol data available right now. Please click Refresh data and try again.")
    else:
        target_sectors = get_nepse_sector_list()

        candidates = all_symbols_df.copy()
        priority_map = {"BUY": 0, "HOLD": 1, "SELL": 2}
        candidates["_priority"] = candidates["signal"].map(priority_map).fillna(9)
        candidates = candidates.sort_values(by=["_priority", "technical_score", "confidence"], ascending=[True, False, False])

        sector_picks: dict[str, dict[str, Any]] = {}
        max_symbols_to_scan = min(250, len(candidates))

        for _, row in candidates.head(max_symbols_to_scan).iterrows():
            symbol = str(row.get("symbol", "")).upper()
            if not symbol:
                continue

            sector = normalize_sector_name(row.get("sector"))

            if sector not in target_sectors:
                sector = "Others"

            if sector in sector_picks:
                continue

            sector_picks[sector] = {
                "sector": sector,
                "symbol": symbol,
                "signal": row.get("signal"),
                "technical_score": row.get("technical_score"),
                "confidence": row.get("confidence"),
                "today_ltp": row.get("today_ltp"),
                "entry_price": row.get("entry_price"),
                "stop_loss": row.get("stop_loss"),
                "target_price": row.get("target_price"),
                "buy_qty": 1,
                "estimated_amount": row.get("today_ltp") if pd.notna(row.get("today_ltp")) else row.get("entry_price"),
                "reason": row.get("reason"),
            }

            if len(sector_picks) == len(target_sectors):
                break

        sector_rows: list[dict[str, Any]] = []
        for sector in target_sectors:
            if sector in sector_picks:
                sector_rows.append(sector_picks[sector])
            else:
                sector_rows.append(
                    {
                        "sector": sector,
                        "symbol": "-",
                        "signal": "NO PICK",
                        "technical_score": pd.NA,
                        "confidence": pd.NA,
                        "today_ltp": pd.NA,
                        "entry_price": pd.NA,
                        "stop_loss": pd.NA,
                        "target_price": pd.NA,
                        "buy_qty": 1,
                        "estimated_amount": pd.NA,
                        "reason": "No candidate found from current daily data",
                    }
                )

        sector_pick_df = pd.DataFrame(sector_rows)

        st.metric("Total sectors", len(target_sectors))
        st.metric("Sectors with picks", int((sector_pick_df["symbol"] != "-").sum()))
        st.dataframe(
            sector_pick_df[
                [
                    "sector",
                    "symbol",
                    "signal",
                    "technical_score",
                    "confidence",
                    "today_ltp",
                    "entry_price",
                    "stop_loss",
                    "target_price",
                    "buy_qty",
                    "estimated_amount",
                    "reason",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

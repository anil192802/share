from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TechnicalSignal:
    signal: str
    score: int
    confidence: int
    reason: str
    entry_price: float | None = None
    stop_loss: float | None = None
    target_price: float | None = None
    expected_7d_price: float | None = None
    pivot_points: dict[str, float] | None = None 
    simple_note: str = ""
    beginner_action: str = ""
    risk_reward_ratio: float | None = None


@dataclass
class CombinedRecommendation:
    short_term_signal: str
    long_term_signal: str
    short_term_score: float
    long_term_score: float
    fundamental_score: int
    stability_score: int
    expected_profit_per_share: float | None
    expected_loss_per_share: float | None
    rationale: str


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_technical_indicators(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy().sort_values("date").reset_index(drop=True)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["sma20"] = close.rolling(20).mean()
    df["sma50"] = close.rolling(50).mean()
    df["ema12"] = close.ewm(span=12, adjust=False).mean()
    df["ema26"] = close.ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["rsi14"] = _rsi(close, period=14)

    rolling_std20 = close.rolling(20).std()
    df["bb_mid"] = df["sma20"]
    df["bb_upper"] = df["sma20"] + (2 * rolling_std20)
    df["bb_lower"] = df["sma20"] - (2 * rolling_std20)

    if "volume" in df.columns:
        df["vwma20"] = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
        df["vol_sma20"] = df["volume"].rolling(20).mean()
        df["volume_breakout"] = df["volume"] > (df["vol_sma20"] * 1.5)
    else:
        df["vwma20"] = pd.NA
        df["vol_sma20"] = pd.NA
        df["volume_breakout"] = False

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    body = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    df["hammer"] = (lower_wick >= (2 * body)) & (upper_wick <= body)

    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    bullish_engulfing = (
        (prev_close < prev_open)
        & (df["close"] > df["open"])
        & (df["open"] <= prev_close)
        & (df["close"] >= prev_open)
    )
    bearish_engulfing = (
        (prev_close > prev_open)
        & (df["close"] < df["open"])
        & (df["open"] >= prev_close)
        & (df["close"] <= prev_open)
    )

    df["bullish_engulfing"] = bullish_engulfing
    df["bearish_engulfing"] = bearish_engulfing

    return df


def evaluate_technical_signal(indicator_df: pd.DataFrame) -> TechnicalSignal:
    if indicator_df.empty:
        return TechnicalSignal(
            "HOLD",
            0,
            50,
            "No historical rows available",
            simple_note="Not enough data",
            beginner_action="Wait",
        )

    latest = indicator_df.iloc[-1]
    score = 0
    reasons: list[str] = []

    close = latest.get("close")
    sma20 = latest.get("sma20")
    sma50 = latest.get("sma50")
    rsi14 = latest.get("rsi14")
    macd = latest.get("macd")
    macd_signal = latest.get("macd_signal")
    bb_upper = latest.get("bb_upper")
    bb_lower = latest.get("bb_lower")
    vwma20 = latest.get("vwma20")
    volume_breakout = latest.get("volume_breakout")

    if pd.notna(close) and pd.notna(sma20):
        if close > sma20:
            score += 1
            reasons.append("Price above SMA20")
        else:
            score -= 1
            reasons.append("Price below SMA20")

    if pd.notna(sma20) and pd.notna(sma50):
        if sma20 > sma50:
            score += 1
            reasons.append("Medium trend bullish (SMA20 > SMA50)")
        else:
            score -= 1
            reasons.append("Medium trend bearish (SMA20 < SMA50)")

    if pd.notna(rsi14):
        if rsi14 < 35:
            score += 1
            reasons.append("RSI in oversold zone")
        elif rsi14 > 70:
            score -= 1
            reasons.append("RSI in overbought zone")

    if pd.notna(macd) and pd.notna(macd_signal):
        if macd > macd_signal:
            score += 1
            reasons.append("MACD above signal line")
        else:
            score -= 1
            reasons.append("MACD below signal line")

    if pd.notna(close) and pd.notna(bb_lower) and close < bb_lower:
        score += 1
        reasons.append("Price under lower Bollinger band")
    elif pd.notna(close) and pd.notna(bb_upper) and close > bb_upper:
        score -= 1
        reasons.append("Price above upper Bollinger band")

    if bool(latest.get("bullish_engulfing", False)):
        score += 1
        reasons.append("Bullish engulfing candle")
    if bool(latest.get("bearish_engulfing", False)):
        score -= 1
        reasons.append("Bearish engulfing candle")
    if bool(latest.get("hammer", False)):
        score += 1
        reasons.append("Hammer candle support")

    if pd.notna(close) and pd.notna(vwma20):
        if close > vwma20:
            score += 1
            reasons.append("Price above VWMA (Volume Weighted Moving Average)")
        else:
            score -= 1
            reasons.append("Price below VWMA")

    if pd.notna(volume_breakout) and bool(volume_breakout):
        if pd.notna(close) and close > latest.get("open", close):
            score += 1
            reasons.append("Bullish High Volume Breakout")
        elif pd.notna(close) and close < latest.get("open", close):
            score -= 1
            reasons.append("Bearish High Volume Selloff")

    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = min(90, 50 + (abs(score) * 10))
    if not reasons:
        reasons.append("Insufficient indicator readiness")

    atr14 = latest.get("atr14")
    risk_buffer = None
    if pd.notna(atr14):
        risk_buffer = float(atr14) * 1.5
    if pd.notna(close):
        fallback_buffer = float(close) * 0.03
        risk_buffer = max(risk_buffer or 0.0, fallback_buffer)

    entry_price = float(close) if pd.notna(close) else None
    stop_loss = None
    target_price = None
    expected_7d_price = None
    simple_note = "Wait and watch"
    beginner_action = "Wait"
    risk_reward_ratio = None

    if entry_price is not None and risk_buffer is not None and risk_buffer > 0:
        # Simple Momentum-based 7-Day Projection
        # Based on average daily return of last 5 days
        past_5d = indicator_df.tail(6)["close"].pct_change().dropna()
        avg_ret = float(past_5d.mean()) if not past_5d.empty else 0.0
        # Dampen the momentum to be conservative (e.g. 50% of recent trend)
        dampened_7d_ret = avg_ret * 3.5 # 0.5 * 7 days
        expected_7d_price = entry_price * (1 + dampened_7d_ret)

        if signal == "BUY":
            stop_loss = max(0.0, entry_price - risk_buffer)
            target_price = entry_price + (risk_buffer * 2)
            # Ensure 7D projection aligns with technical target
            expected_7d_price = max(expected_7d_price, entry_price + (risk_buffer * 0.5))
            simple_note = "Possible buy setup"
            beginner_action = "Buy small (step by step)"
        elif signal == "SELL":
            stop_loss = entry_price + risk_buffer
            target_price = max(0.0, entry_price - (risk_buffer * 2))
            # Ensure 7D projection aligns with technical drop
            expected_7d_price = min(expected_7d_price, entry_price - (risk_buffer * 0.5))
            simple_note = "Possible sell/avoid setup"
            beginner_action = "Book profit / Avoid new buy"
        else:
            stop_loss = max(0.0, entry_price - risk_buffer)
            target_price = entry_price + risk_buffer
            simple_note = "No strong trend now"
            beginner_action = "Wait"

    if entry_price is not None and stop_loss is not None and target_price is not None:
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        if risk > 0:
            risk_reward_ratio = reward / risk

    # --- PIVOT POINTS CALCULATION (Professional Levels) ---
    pivots = None
    h, l, c = latest.get("high"), latest.get("low"), latest.get("close")
    if pd.notna(h) and pd.notna(l) and pd.notna(c):
        pp = (h + l + c) / 3
        pivots = {
            "PP": pp,
            "R1": (2 * pp) - l,
            "S1": (2 * pp) - h,
            "R2": pp + (h - l),
            "S2": pp - (h - l)
        }

    return TechnicalSignal(
        signal=signal,
        score=score,
        confidence=confidence,
        reason="; ".join(reasons),
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        expected_7d_price=expected_7d_price,
        pivot_points=pivots,
        simple_note=simple_note,
        beginner_action=beginner_action,
        risk_reward_ratio=risk_reward_ratio,
    )


def _score_to_signal(score: float) -> str:
    if score >= 2:
        return "BUY"
    if score <= -2:
        return "SELL"
    return "HOLD"


def evaluate_combined_recommendation(
    indicator_df: pd.DataFrame,
    technical_signal: TechnicalSignal,
    intraday_score: int,
    political_bonus: int,
    news_bonus: int,
    fundamental_inputs: dict[str, float | int | None],
) -> CombinedRecommendation:
    if indicator_df.empty:
        return CombinedRecommendation(
            short_term_signal="HOLD",
            long_term_signal="STABLE HOLD",
            short_term_score=0.0,
            long_term_score=0.0,
            fundamental_score=0,
            stability_score=0,
            expected_profit_per_share=None,
            expected_loss_per_share=None,
            rationale="Not enough data for combined recommendation",
        )

    latest = indicator_df.iloc[-1]
    close = latest.get("close")

    fundamental_score = 0
    stability_score = 0
    reasons: list[str] = []

    avg_180 = fundamental_inputs.get("avg_180")
    if pd.notna(close) and avg_180 is not None:
        if float(close) >= float(avg_180):
            fundamental_score += 2
            reasons.append("Price is above 180-day average")
        else:
            fundamental_score -= 2
            reasons.append("Price is below 180-day average")

    cash_dividend_pct = fundamental_inputs.get("cash_dividend_pct")
    if cash_dividend_pct is not None and float(cash_dividend_pct) >= 5:
        fundamental_score += 1
        reasons.append("Healthy cash dividend history")

    bonus_share_pct = fundamental_inputs.get("bonus_share_pct")
    if bonus_share_pct is not None and float(bonus_share_pct) >= 5:
        fundamental_score += 1
        reasons.append("Bonus share support present")

    week52_high = fundamental_inputs.get("week52_high")
    week52_low = fundamental_inputs.get("week52_low")
    if (
        pd.notna(close)
        and week52_high is not None
        and week52_low is not None
        and float(week52_high) > float(week52_low)
    ):
        position = (float(close) - float(week52_low)) / (float(week52_high) - float(week52_low))
        if position <= 0.35:
            fundamental_score += 1
            reasons.append("Close is near 52-week support zone")
        elif position >= 0.85:
            fundamental_score -= 1
            reasons.append("Close is near 52-week resistance zone")

    listed_shares = fundamental_inputs.get("listed_shares")
    if listed_shares is not None and float(listed_shares) >= 10_000_000:
        stability_score += 1
        reasons.append("Higher listed shares can improve liquidity")

    returns_20 = indicator_df["close"].pct_change().tail(20).dropna()
    if not returns_20.empty:
        volatility = float(returns_20.std())
        if volatility <= 0.025:
            stability_score += 1
            reasons.append("Recent volatility is low")
        elif volatility >= 0.05:
            stability_score -= 1
            reasons.append("Recent volatility is high")

    short_term_score = (
        (0.50 * float(intraday_score))
        + (0.35 * float(technical_signal.score))
        + (0.15 * float(fundamental_score))
        + float(political_bonus)
        + float(news_bonus)
    )
    long_term_score = (
        (0.20 * float(intraday_score))
        + (0.35 * float(technical_signal.score))
        + (0.45 * float(fundamental_score))
        + (0.20 * float(stability_score))
        + (0.50 * float(political_bonus))
        + (0.50 * float(news_bonus))
    )

    short_term_signal = _score_to_signal(short_term_score)
    long_term_signal = f"STABLE {_score_to_signal(long_term_score)}"

    expected_profit_per_share = None
    expected_loss_per_share = None
    if technical_signal.entry_price is not None and technical_signal.target_price is not None:
        expected_profit_per_share = max(0.0, technical_signal.target_price - technical_signal.entry_price)
    if technical_signal.entry_price is not None and technical_signal.stop_loss is not None:
        expected_loss_per_share = max(0.0, technical_signal.entry_price - technical_signal.stop_loss)

    if not reasons:
        reasons.append("Neutral fundamental and stability profile")

    reasons.append(f"Political bonus: {political_bonus:+d}")
    reasons.append(f"News bonus: {news_bonus:+d}")

    return CombinedRecommendation(
        short_term_signal=short_term_signal,
        long_term_signal=long_term_signal,
        short_term_score=short_term_score,
        long_term_score=long_term_score,
        fundamental_score=fundamental_score,
        stability_score=stability_score,
        expected_profit_per_share=expected_profit_per_share,
        expected_loss_per_share=expected_loss_per_share,
        rationale="; ".join(reasons),
    )

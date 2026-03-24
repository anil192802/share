from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SignalResult:
    signal: str
    score: int
    confidence: int
    reason: str


def evaluate_signal(row: pd.Series) -> SignalResult:
    ltp = row.get("ltp")
    open_price = row.get("open")
    high = row.get("high")
    low = row.get("low")
    pct_change = row.get("pct_change")

    if pd.isna(ltp):
        return SignalResult("HOLD", 0, 50, "Missing LTP data")

    score = 0
    reasons: list[str] = []

    if pd.notna(open_price) and open_price > 0:
        momentum = (ltp - open_price) / open_price
        if momentum >= 0.03:
            score += 2
            reasons.append("Strong rise from open")
        elif momentum >= 0.01:
            score += 1
            reasons.append("Price above open")
        elif momentum <= -0.03:
            score -= 2
            reasons.append("Strong fall from open")
        elif momentum <= -0.01:
            score -= 1
            reasons.append("Price below open")

    if pd.notna(high) and pd.notna(low) and high > low:
        position = (ltp - low) / (high - low)
        if position >= 0.8:
            score += 1
            reasons.append("Trading near day high")
        elif position <= 0.2:
            score -= 1
            reasons.append("Trading near day low")

    if pd.notna(pct_change):
        if pct_change >= 2:
            score += 1
            reasons.append("Positive day change")
        elif pct_change <= -2:
            score -= 1
            reasons.append("Negative day change")

    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = min(95, 50 + (abs(score) * 15))

    if not reasons:
        reasons.append("Limited intraday signal")

    return SignalResult(signal, score, confidence, "; ".join(reasons))


def analyze_shares(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        result = evaluate_signal(row)
        records.append(
            {
                **row.to_dict(),
                "signal": result.signal,
                "score": result.score,
                "confidence": result.confidence,
                "reason": result.reason,
            }
        )

    result_df = pd.DataFrame(records)

    signal_order = {"BUY": 0, "HOLD": 1, "SELL": 2}
    result_df["_signal_order"] = result_df["signal"].map(signal_order)
    result_df = result_df.sort_values(
        by=["_signal_order", "confidence", "pct_change"], ascending=[True, False, False]
    )
    result_df = result_df.drop(columns=["_signal_order"]).reset_index(drop=True)
    return result_df

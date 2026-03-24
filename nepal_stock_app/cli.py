from __future__ import annotations

import argparse

from nepal_stock_app.data_source import fetch_symbol_fundamentals, fetch_symbol_history, fetch_today_share_data
from nepal_stock_app.signals import analyze_shares
from nepal_stock_app.technical import (
    add_technical_indicators,
    evaluate_combined_recommendation,
    evaluate_technical_signal,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch today's NEPSE shares and generate buy/sell/hold signals."
    )
    parser.add_argument(
        "--signal",
        default="all",
        choices=["all", "buy", "sell", "hold"],
        help="Filter results by signal type.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of rows to show.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="",
        help="Run technical analysis for one symbol (example: NABIL).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=45,
        help="Lookback trading days for symbol technical analysis.",
    )
    parser.add_argument(
        "--political-bonus",
        type=int,
        default=0,
        choices=[-2, -1, 0, 1, 2],
        help="Political scenario bonus impact from -2 to +2.",
    )
    parser.add_argument(
        "--news-bonus",
        type=int,
        default=0,
        choices=[-2, -1, 0, 1, 2],
        help="Symbol news bonus impact from -2 to +2.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.symbol:
        history = fetch_symbol_history(args.symbol, lookback_days=args.days)
        indicators = add_technical_indicators(history)
        signal = evaluate_technical_signal(indicators)
        fundamentals = fetch_symbol_fundamentals(args.symbol)

        intraday_score = 0
        try:
            intraday_df = analyze_shares(fetch_today_share_data())
            symbol_upper = args.symbol.strip().upper()
            match = intraday_df[intraday_df["symbol"] == symbol_upper]
            if not match.empty:
                intraday_score = int(match.iloc[0]["score"])
        except Exception:
            intraday_score = 0

        combined = evaluate_combined_recommendation(
            indicator_df=indicators,
            technical_signal=signal,
            intraday_score=intraday_score,
            political_bonus=args.political_bonus,
            news_bonus=args.news_bonus,
            fundamental_inputs={
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
            },
        )

        latest = indicators.iloc[-1]
        print(f"Symbol: {args.symbol.upper()}")
        print(f"Date:   {latest['date'].date()}")
        print(f"Close:  {latest['close']:.2f}")
        print(f"Signal: {signal.signal} (confidence {signal.confidence}%)")
        print(f"Score:  {signal.score}")
        print(f"Why:    {signal.reason}")
        print(f"\nShort-term Signal:      {combined.short_term_signal} ({combined.short_term_score:.2f})")
        print(f"Stable Long-term Signal:{combined.long_term_signal} ({combined.long_term_score:.2f})")
        print(f"Fundamental Score:      {combined.fundamental_score}")
        print(f"Stability Score:        {combined.stability_score}")
        print(f"Political Bonus:        {args.political_bonus:+d}")
        print(f"News Bonus:             {args.news_bonus:+d}")

        if combined.expected_profit_per_share is not None:
            print(f"Expected Profit/Share:  {combined.expected_profit_per_share:.2f}")
        if combined.expected_loss_per_share is not None:
            print(f"Expected Loss/Share:    {combined.expected_loss_per_share:.2f}")
        print(f"Rationale:              {combined.rationale}")

        if args.csv:
            indicators.to_csv(args.csv, index=False)
            print(f"\nSaved technical history to: {args.csv}")
        return

    raw_data = fetch_today_share_data()
    analyzed = analyze_shares(raw_data)

    if args.signal != "all":
        analyzed = analyzed[analyzed["signal"] == args.signal.upper()]

    output = analyzed.head(args.top)
    print(output[["symbol", "ltp", "pct_change", "signal", "confidence", "reason"]].to_string(index=False))

    if args.csv:
        analyzed.to_csv(args.csv, index=False)
        print(f"\nSaved full results to: {args.csv}")


if __name__ == "__main__":
    main()

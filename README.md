# NEPSE Technical Analyzer

App to fetch Nepal stock data, analyze multi-day trends, and produce **BUY / SELL / HOLD** signals.

## What it does

- Fetches today's share table from public NEPSE market pages.
- Fetches date-wise historical snapshots for a selected symbol.
- Normalizes key fields: symbol, LTP, open, high, low, % change, volume.
- Applies intraday and multi-day technical signal strategies.
- Adds a combined recommendation engine with:
  - Fundamental snapshot (sector, dividend, listed shares, 52-week zone, long averages)
  - Political scenario bonus (auto news + manual adjustment)
  - Symbol news bonus (auto news + manual adjustment)
  - Short-term signal and stable long-term signal
  - Expected profit/loss per share and for risk-based quantity
- Shows results in:
  - Streamlit UI
  - CLI output

> No strategy can guarantee profit. Use as decision support and manage risk.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run web app

```bash
streamlit run app.py
```

In the Streamlit sidebar, use **Context Bonus** controls to reflect current political scenario and symbol-specific news impact.
The app also auto-reads headlines from ShareSansar, MeroLagani and market pages to calculate a base political and symbol news bonus.

## Run CLI

```bash
python -m nepal_stock_app.cli --top 30
python -m nepal_stock_app.cli --signal buy --top 20
python -m nepal_stock_app.cli --csv results.csv
python -m nepal_stock_app.cli --symbol NABIL --days 60
python -m nepal_stock_app.cli --symbol FMDBL --days 60 --political-bonus 1 --news-bonus -1
```

## Signal logic

### Intraday screener

- LTP vs Open (momentum)
- Position within High-Low day range
- Percent day change

Score mapping:

- score >= 2 -> BUY
- score <= -2 -> SELL
- otherwise -> HOLD

### Technical analyzer (historical)

- SMA20 / SMA50 trend structure
- EMA/MACD momentum crossover
- RSI(14) overbought / oversold zones
- Bollinger band positioning
- Candlestick signals (engulfing, hammer)

The technical score combines these factors and maps to BUY/SELL/HOLD with confidence.

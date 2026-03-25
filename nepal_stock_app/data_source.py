from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
import re
from typing import Optional

import pandas as pd
import requests

TODAY_SHARE_PRICE_URL = "https://www.sharesansar.com/today-share-price"
MEROLAGANI_LIVE_MARKET_URL = "https://merolagani.com/LatestMarket.aspx"
AJAX_TODAY_SHARE_PRICE_URL = "https://www.sharesansar.com/ajaxtodayshareprice"
COMPANY_PAGE_URL_TEMPLATE = "https://www.sharesansar.com/company/{symbol}"
COMPANY_PRICE_HISTORY_URL = "https://www.sharesansar.com/company-price-history"
MEROLAGANI_HOME_URL = "https://merolagani.com"
SHARESANSAR_HOME_URL = "https://www.sharesansar.com"
SHARESANSAR_MARKET_NEWS_URL = "https://www.sharesansar.com/category/market-news"

NEPSE_SECTOR_LIST = [
    "Commercial Bank",
    "Development Bank",
    "Finance",
    "Microfinance",
    "Hydropower",
    "Life Insurance",
    "Non Life Insurance",
    "Hotels And Tourism",
    "Investment",
    "Manufacturing And Processing",
    "Mutual Fund",
    "Others",
    "Trading",
]

SECTOR_ALIAS_MAP = {
    "commercial bank": "Commercial Bank",
    "development bank": "Development Bank",
    "finance": "Finance",
    "microfinance": "Microfinance",
    "hydropower": "Hydropower",
    "life insurance": "Life Insurance",
    "non life insurance": "Non Life Insurance",
    "non-life insurance": "Non Life Insurance",
    "hotel and tourism": "Hotels And Tourism",
    "hotels and tourism": "Hotels And Tourism",
    "investment": "Investment",
    "manufacturing and processing": "Manufacturing And Processing",
    "mutual fund": "Mutual Fund",
    "others": "Others",
    "other": "Others",
    "trading": "Trading",
}


@dataclass
class FetchConfig:
    url: str = TODAY_SHARE_PRICE_URL
    timeout_seconds: int = 20


@dataclass
class FundamentalSnapshot:
    symbol: str
    sector: str | None = None
    listed_shares: float | None = None
    paid_up: float | None = None
    total_paid_up_value: float | None = None
    cash_dividend_pct: float | None = None
    bonus_share_pct: float | None = None
    avg_120: float | None = None
    avg_180: float | None = None
    week52_high: float | None = None
    week52_low: float | None = None


@dataclass
class NewsSignal:
    bonus: int
    score: int
    positive_hits: int
    negative_hits: int
    headlines_checked: int
    matched_headlines: list[str]


def _default_headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0"
    }


def get_nepse_sector_list() -> list[str]:
    return list(NEPSE_SECTOR_LIST)


def normalize_sector_name(raw_sector: str | None) -> str:
    if not raw_sector:
        return "Others"
    cleaned = re.sub(r"\s+", " ", str(raw_sector)).strip().lower()
    return SECTOR_ALIAS_MAP.get(cleaned, str(raw_sector).strip().title())


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for original in df.columns:
        name = str(original).strip().lower()
        if name in {"symbol", "scrip"}:
            renamed[original] = "symbol"
        elif "ltp" in name:
            renamed[original] = "ltp"
        elif name.startswith("open"):
            renamed[original] = "open"
        elif name.startswith("high"):
            renamed[original] = "high"
        elif name.startswith("low"):
            renamed[original] = "low"
        elif "%" in name and "change" in name:
            renamed[original] = "pct_change"
        elif "change" == name:
            renamed[original] = "pct_change"
        elif "qty" in name or "volume" in name:
            renamed[original] = "volume"
        elif "sector" in name:
            renamed[original] = "sector"
    return df.rename(columns=renamed)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def _extract_csrf_token(html_text: str) -> str:
    match = re.search(r'name="_token"\s+value="([^"]+)"', html_text)
    if not match:
        raise RuntimeError("Could not find CSRF token on source page.")
    return match.group(1)


def _extract_as_of_date(html_text: str, fallback: Optional[str]) -> str:
    match = re.search(r"As of\s*:\s*(\d{4}-\d{2}-\d{2})", html_text)
    if match:
        return match.group(1)
    if fallback:
        return fallback
    return date.today().isoformat()


def _extract_company_id(html_text: str) -> str:
    match = re.search(r'<div id="companyid"[^>]*>(\d+)</div>', html_text)
    if not match:
        raise RuntimeError("Could not find company id on company page.")
    return match.group(1)


def _extract_meta_token(html_text: str) -> str:
    match = re.search(r'<meta\s+name="_token"\s+content="([^"]+)"', html_text)
    if not match:
        raise RuntimeError("Could not find company CSRF token in page meta.")
    return match.group(1)


def _extract_single_value(html_text: str, label: str) -> str | None:
    pattern = rf"<td[^>]*>\s*{re.escape(label)}\s*</td>\s*<td[^>]*>(.*?)</td>"
    match = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    value_html = match.group(1)
    value = re.sub(r"<[^>]+>", "", value_html)
    return re.sub(r"\s+", " ", value).strip() or None


def _extract_float_from_text(raw: str | None) -> float | None:
    if not raw:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", raw)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_sector(html_text: str) -> str | None:
    match = re.search(r"Sector:\s*</[^>]+>\s*([^<\n]+)", html_text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip() or None


def _extract_symbol_market_snapshot(html_text: str) -> dict[str, float | None]:
    compact = re.sub(r"\s+", " ", html_text)

    avg120_match = re.search(r"120 Days Average\s*:\s*([0-9,.]+)", compact, flags=re.IGNORECASE)
    avg180_match = re.search(r"180 Days Average\s*:\s*([0-9,.]+)", compact, flags=re.IGNORECASE)
    high_low_match = re.search(
        r"52\s*Week\s*High-Low\s*:\s*([0-9,.]+)\s*-\s*([0-9,.]+)",
        compact,
        flags=re.IGNORECASE,
    )

    week52_high = _extract_float_from_text(high_low_match.group(1)) if high_low_match else None
    week52_low = _extract_float_from_text(high_low_match.group(2)) if high_low_match else None

    return {
        "avg_120": _extract_float_from_text(avg120_match.group(1)) if avg120_match else None,
        "avg_180": _extract_float_from_text(avg180_match.group(1)) if avg180_match else None,
        "week52_high": week52_high,
        "week52_low": week52_low,
    }


def _extract_anchor_texts(html_text: str) -> list[str]:
    matches = re.findall(r"<a[^>]*>(.*?)</a>", html_text, flags=re.IGNORECASE | re.DOTALL)
    results: list[str] = []
    for item in matches:
        plain = re.sub(r"<[^>]+>", "", item)
        plain = re.sub(r"&nbsp;", " ", plain, flags=re.IGNORECASE)
        plain = re.sub(r"\s+", " ", plain).strip()
        if len(plain) < 12 or len(plain) > 220:
            continue
        if plain.lower() in {"read more", "login", "register", "home"}:
            continue
        results.append(plain)
    deduped = list(dict.fromkeys(results))
    return deduped


def _keyword_sentiment_score(
    headlines: list[str],
    positive_keywords: list[str],
    negative_keywords: list[str],
    must_contain: list[str] | None = None,
) -> NewsSignal:
    positive_hits = 0
    negative_hits = 0
    matched_headlines: list[str] = []

    must_contain = must_contain or []
    must_contain_lower = [item.lower() for item in must_contain if item]
    positive_lower = [item.lower() for item in positive_keywords]
    negative_lower = [item.lower() for item in negative_keywords]

    for headline in headlines:
        text = headline.lower()
        if must_contain_lower and not any(token in text for token in must_contain_lower):
            continue

        local_score = 0
        if any(keyword in text for keyword in positive_lower):
            positive_hits += 1
            local_score += 1
        if any(keyword in text for keyword in negative_lower):
            negative_hits += 1
            local_score -= 1

        if local_score != 0 and len(matched_headlines) < 5:
            matched_headlines.append(headline)

    score = positive_hits - negative_hits
    if score >= 3:
        bonus = 2
    elif score >= 1:
        bonus = 1
    elif score <= -3:
        bonus = -2
    elif score <= -1:
        bonus = -1
    else:
        bonus = 0

    return NewsSignal(
        bonus=bonus,
        score=score,
        positive_hits=positive_hits,
        negative_hits=negative_hits,
        headlines_checked=len(headlines),
        matched_headlines=matched_headlines,
    )


def _fetch_headlines_from_url(url: str, timeout_seconds: int) -> list[str]:
    try:
        response = requests.get(url, timeout=timeout_seconds, headers=_default_headers())
        response.raise_for_status()
        return _extract_anchor_texts(response.text)
    except Exception:
        return []


def fetch_political_news_signal(config: Optional[FetchConfig] = None) -> NewsSignal:
    cfg = config or FetchConfig()
    sources = [
        SHARESANSAR_MARKET_NEWS_URL,
        SHARESANSAR_HOME_URL,
        MEROLAGANI_HOME_URL,
    ]

    headlines: list[str] = []
    for source in sources:
        headlines.extend(_fetch_headlines_from_url(source, cfg.timeout_seconds))

    positive_keywords = [
        "policy support",
        "stable government",
        "agreement",
        "recovery",
        "growth",
        "rate cut",
        "liquidity",
        "bullish",
        "capital market reform",
    ]
    negative_keywords = [
        "political crisis",
        "government collapse",
        "conflict",
        "strike",
        "protest",
        "rate hike",
        "instability",
        "corruption",
        "war",
        "bearish",
    ]
    context_tokens = [
        "government",
        "political",
        "policy",
        "budget",
        "interest rate",
        "market",
        "nepal",
    ]

    return _keyword_sentiment_score(
        headlines=headlines,
        positive_keywords=positive_keywords,
        negative_keywords=negative_keywords,
        must_contain=context_tokens,
    )


def fetch_symbol_news_signal(
    symbol: str,
    config: Optional[FetchConfig] = None,
) -> NewsSignal:
    cfg = config or FetchConfig()
    symbol_upper = symbol.strip().upper()
    if not symbol_upper:
        return NewsSignal(0, 0, 0, 0, 0, [])

    sources = [
        COMPANY_PAGE_URL_TEMPLATE.format(symbol=symbol_upper.lower()),
        SHARESANSAR_HOME_URL,
        MEROLAGANI_HOME_URL,
    ]
    headlines: list[str] = []
    for source in sources:
        headlines.extend(_fetch_headlines_from_url(source, cfg.timeout_seconds))

    positive_keywords = [
        "profit",
        "dividend",
        "bonus",
        "rights",
        "approval",
        "growth",
        "expansion",
        "acquisition",
        "record",
        "uptrend",
        "buy",
    ]
    negative_keywords = [
        "loss",
        "penalty",
        "investigation",
        "default",
        "downgrade",
        "decline",
        "suspension",
        "delist",
        "fine",
        "sell",
    ]

    return _keyword_sentiment_score(
        headlines=headlines,
        positive_keywords=positive_keywords,
        negative_keywords=negative_keywords,
        must_contain=[symbol_upper],
    )


def _build_price_history_payload(
    company_id: str,
    csrf_token: str,
    start: int,
    length: int,
) -> dict[str, str]:
    payload: dict[str, str] = {
        "company": company_id,
        "_token": csrf_token,
        "draw": "1",
        "start": str(start),
        "length": str(length),
        "search[value]": "",
        "search[regex]": "false",
        "order[0][column]": "0",
        "order[0][dir]": "desc",
    }
    for idx in range(6):
        payload[f"columns[{idx}][data]"] = str(idx)
        payload[f"columns[{idx}][name]"] = ""
        payload[f"columns[{idx}][searchable]"] = "true"
        payload[f"columns[{idx}][orderable]"] = "true"
        payload[f"columns[{idx}][search][value]"] = ""
        payload[f"columns[{idx}][search][regex]"] = "false"
    return payload


def _parse_share_table(html_text: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html_text))
    if not tables:
        raise RuntimeError("No tabular share data found on source page.")

    parsed = None
    for table in tables:
        normalized = _normalize_columns(table)
        if {"symbol", "ltp"}.issubset(set(normalized.columns)):
            parsed = normalized
            break

    if parsed is None:
        raise RuntimeError("Could not find expected columns (symbol, ltp) in source data.")

    expected = ["symbol", "sector", "ltp", "open", "high", "low", "pct_change", "volume"]

    cleaned_columns: dict[str, pd.Series] = {}
    for column in expected:
        if column not in parsed.columns:
            cleaned_columns[column] = pd.Series([pd.NA] * len(parsed), index=parsed.index)
            continue

        selected = parsed[column]
        if isinstance(selected, pd.DataFrame):
            cleaned_columns[column] = selected.iloc[:, 0]
        else:
            cleaned_columns[column] = selected

    output = pd.DataFrame(cleaned_columns)
    output["symbol"] = output["symbol"].astype(str).str.strip()
    output["sector"] = output["sector"].astype(str).str.strip()

    for numeric_column in ["ltp", "open", "high", "low", "pct_change", "volume"]:
        output[numeric_column] = _to_numeric(output[numeric_column])

    output = output.dropna(subset=["symbol", "ltp"])
    output = output[output["symbol"] != ""]
    output.loc[output["sector"].str.lower().isin(["", "nan", "none", "<na>"]), "sector"] = "Others"
    output = output.reset_index(drop=True)
    return output


def fetch_share_data_for_date(
    target_date: Optional[str] = None,
    config: Optional[FetchConfig] = None,
) -> pd.DataFrame:
    cfg = config or FetchConfig()
    session = requests.Session()

    # Use MeroLagani for real-time data if target_date is not specified (current day)
    if not target_date:
        try:
            live_response = session.get(MEROLAGANI_LIVE_MARKET_URL, timeout=cfg.timeout_seconds, headers=_default_headers())
            live_response.raise_for_status()
            parsed_live = _parse_share_table(live_response.text)
            if not parsed_live.empty:
                return parsed_live
        except Exception:
            # Fallback to ShareSansar if MeroLagani fails
            pass

    base_response = session.get(cfg.url, timeout=cfg.timeout_seconds, headers=_default_headers())
    base_response.raise_for_status()

    html_text = base_response.text
    if target_date:
        token = _extract_csrf_token(base_response.text)
        payload = {"_token": token, "sector": "all_sec", "date": target_date}
        date_response = session.post(
            AJAX_TODAY_SHARE_PRICE_URL,
            data=payload,
            timeout=cfg.timeout_seconds,
            headers={
                **_default_headers(),
                "Referer": cfg.url,
                "X-Requested-With": "XMLHttpRequest",
            },
        )
        date_response.raise_for_status()
        html_text = date_response.text

    parsed = _parse_share_table(html_text)
    parsed["snapshot_date"] = _extract_as_of_date(html_text, target_date)
    return parsed


def fetch_today_share_data(config: Optional[FetchConfig] = None) -> pd.DataFrame:
    return fetch_share_data_for_date(target_date=None, config=config)


def fetch_symbol_history(
    symbol: str,
    lookback_days: int = 45,
    max_calendar_days: int = 120,
    config: Optional[FetchConfig] = None,
) -> pd.DataFrame:
    del max_calendar_days
    cfg = config or FetchConfig()
    symbol_upper = symbol.strip().upper()
    if not symbol_upper:
        raise ValueError("Symbol is required for history fetch.")

    session = requests.Session()
    company_url = COMPANY_PAGE_URL_TEMPLATE.format(symbol=symbol_upper.lower())
    company_response = session.get(company_url, timeout=cfg.timeout_seconds, headers=_default_headers())
    company_response.raise_for_status()

    company_id = _extract_company_id(company_response.text)
    csrf_token = _extract_meta_token(company_response.text)

    rows: list[dict] = []
    start = 0
    page_size = 50

    while len(rows) < lookback_days:
        payload = _build_price_history_payload(
            company_id=company_id,
            csrf_token=csrf_token,
            start=start,
            length=page_size,
        )
        history_response = session.post(
            COMPANY_PRICE_HISTORY_URL,
            data=payload,
            timeout=cfg.timeout_seconds,
            headers={
                **_default_headers(),
                "X-CSRF-Token": csrf_token,
                "X-Requested-With": "XMLHttpRequest",
                "Referer": company_url,
            },
        )
        history_response.raise_for_status()

        payload_json = history_response.json()
        batch = payload_json.get("data", [])
        if not batch:
            break

        rows.extend(batch)

        total = int(payload_json.get("recordsFiltered", 0) or 0)
        start += page_size
        if start >= total:
            break

    if not rows:
        raise RuntimeError(f"No historical records found for symbol: {symbol_upper}")

    history = pd.DataFrame(rows)
    history = history.rename(
        columns={
            "published_date": "date",
            "traded_quantity": "volume",
            "per_change": "pct_change",
        }
    )
    history["symbol"] = symbol_upper

    for numeric_column in ["open", "high", "low", "close", "volume", "pct_change"]:
        history[numeric_column] = _to_numeric(history[numeric_column])

    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    history = history[["date", "symbol", "open", "high", "low", "close", "pct_change", "volume"]]

    if lookback_days > 0:
        history = history.tail(lookback_days).reset_index(drop=True)

    return history


def fetch_symbol_fundamentals(
    symbol: str,
    config: Optional[FetchConfig] = None,
) -> FundamentalSnapshot:
    cfg = config or FetchConfig()
    symbol_upper = symbol.strip().upper()
    if not symbol_upper:
        raise ValueError("Symbol is required for fundamentals fetch.")

    company_url = COMPANY_PAGE_URL_TEMPLATE.format(symbol=symbol_upper.lower())
    response = requests.get(company_url, timeout=cfg.timeout_seconds, headers=_default_headers())
    response.raise_for_status()

    html_text = response.text
    market_snapshot = _extract_symbol_market_snapshot(html_text)

    return FundamentalSnapshot(
        symbol=symbol_upper,
        sector=_extract_sector(html_text),
        listed_shares=_extract_float_from_text(_extract_single_value(html_text, "Listed Shares")),
        paid_up=_extract_float_from_text(_extract_single_value(html_text, "Paid Up")),
        total_paid_up_value=_extract_float_from_text(_extract_single_value(html_text, "Total Paid Up Value")),
        cash_dividend_pct=_extract_float_from_text(_extract_single_value(html_text, "Cash Dividend")),
        bonus_share_pct=_extract_float_from_text(_extract_single_value(html_text, "Bonus Share")),
        avg_120=market_snapshot.get("avg_120"),
        avg_180=market_snapshot.get("avg_180"),
        week52_high=market_snapshot.get("week52_high"),
        week52_low=market_snapshot.get("week52_low"),
    )

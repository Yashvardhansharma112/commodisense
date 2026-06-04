"""
News Collector — ingests commodity news from GDELT API and RSS feeds,
stores deduplicated articles in DuckDB (table: news_raw).

Usage:
    python data/collector_news.py                # incremental (last 7 days)
    python data/collector_news.py --backfill     # last 30 days
    python data/collector_news.py --months 12    # full 12-month GDELT history
    python data/collector_news.py --months 6     # last 6 months
"""

import argparse
import hashlib
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.db import get_conn, init_schema

# ── constants ──────────────────────────────────────────────────────────────────

GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

# Query terms per commodity — broader queries yield more articles
COMMODITY_QUERIES: dict[str, str] = {
    "CL=F":    "crude oil petroleum OPEC production barrel brent WTI",
    "NG=F":    "natural gas LNG pipeline energy storage Henry Hub",
    "GC=F":    "gold price bullion federal reserve inflation treasury yield",
    "ZW=F":    "wheat grain harvest Ukraine Russia Black Sea export flour",
    "ZC=F":    "corn maize crop yield USDA ethanol feed grain",
    "ZS=F":    "soybean soy crop Brazil Argentina harvest USDA crush",
    "CT=F":    "cotton textile crop India Pakistan fiber apparel",
    "SB=F":    "sugar cane ethanol Brazil harvest mill sweetener",
    "USDINR=X":"USD INR rupee India RBI Reserve Bank forex currency",
    "HG=F":    "copper mining Chile China demand LME industrial metal",
}

# Keywords for tag assignment — broader matching catches more articles
COMMODITY_KEYWORDS: dict[str, list[str]] = {
    "CL=F":    ["crude", "petroleum", "opec", "oil price", "barrel", "brent", "wti",
                "oil market", "oil supply", "energy price", "refinery"],
    "NG=F":    ["natural gas", "lng", "pipeline", "gas price", "gas storage",
                "henry hub", "gas supply", "gas demand", "gas inventory"],
    "GC=F":    ["gold", "bullion", "gold price", "fed rate", "federal reserve",
                "inflation", "treasury yield", "safe haven", "gold rally"],
    "ZW=F":    ["wheat", "grain", "flour", "ukraine wheat", "black sea", "wheat price",
                "grain export", "wheat supply", "winter wheat", "spring wheat"],
    "ZC=F":    ["corn", "maize", "corn price", "corn crop", "ethanol", "usda corn",
                "corn yield", "corn supply", "corn demand", "feed grain"],
    "ZS=F":    ["soybean", "soy", "soya", "soy price", "soy crop", "soy crush",
                "soybean meal", "soy oil", "brazil soy", "argentina soy"],
    "CT=F":    ["cotton", "textile", "cotton price", "cotton crop", "fiber",
                "cotton supply", "cotton demand", "apparel", "cotton india"],
    "SB=F":    ["sugar", "cane", "sugar price", "ethanol", "sucrose", "sweetener",
                "sugar crop", "brazil sugar", "sugar supply", "sugar demand"],
    "USDINR=X":["rupee", "rbi", "inr", "usd inr", "india currency", "forex india",
                "reserve bank india", "indian rupee", "usd/inr", "india rate"],
    "HG=F":    ["copper", "copper price", "copper mine", "lme copper", "copper demand",
                "copper supply", "chile copper", "china copper", "industrial metal",
                "copper inventory", "copper futures"],
}

RSS_FEEDS = [
    "https://www.forexlive.com/feed/news",           # FX + commodities, 25+ daily
    "https://oilprice.com/rss/main",                 # energy + commodity news
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",  # CNBC markets
    "https://www.ft.com/rss/home",                   # FT business headlines
]

LOG_PATH = Path(__file__).parent / "logs" / "news.log"

# ── logging ────────────────────────────────────────────────────────────────────

LOG_PATH.parent.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── helpers ────────────────────────────────────────────────────────────────────


def _url_id(url: str) -> str:
    """Deterministic short ID from URL hash (used as primary key)."""
    return hashlib.sha256(url.encode()).hexdigest()[:32]


def _tag_commodity(text: str) -> str:
    """Return comma-separated commodity symbols found in text (keyword match)."""
    lower = text.lower()
    tags = [sym for sym, kws in COMMODITY_KEYWORDS.items() if any(k in lower for k in kws)]
    return ",".join(tags)


def _insert_article(
    article_id: str,
    source: str,
    published_date: datetime | None,
    title: str,
    summary: str,
    url: str,
    commodity_tags: str,
) -> bool:
    """
    Insert one article. Returns True if inserted, False if duplicate/error.
    """
    conn = get_conn()
    try:
        # Check existence first — DuckDB has no changes() function
        exists = conn.execute(
            "SELECT 1 FROM news_raw WHERE id = ?", [article_id]
        ).fetchone()
        if exists:
            return False
        conn.execute(
            """
            INSERT INTO news_raw
                (id, source, published_date, title, summary, url, commodity_tags,
                 sentiment_score, processed)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, FALSE)
            """,
            [article_id, source, published_date, title, summary, url, commodity_tags],
        )
        return True
    except Exception as exc:
        log.debug("Insert error for %s: %s", article_id, exc)
        return False
    finally:
        conn.close()


# ── GDELT ──────────────────────────────────────────────────────────────────────

# GDELT datetime format
_GDELT_DT_FMT = "%Y%m%d%H%M%S"
# Window size for history chunking — 2 weeks balances coverage vs API calls
_WINDOW_DAYS = 14


def _fmt_gdelt_dt(dt: datetime) -> str:
    return dt.strftime(_GDELT_DT_FMT)


def _fetch_gdelt_window(
    query: str,
    start: datetime,
    end: datetime,
    max_records: int = 250,
    retries: int = 4,
) -> list[dict]:
    """
    Fetch articles from GDELT v2 doc API for a query within a date window.
    Retries with exponential backoff on 429 (rate limit) and transient errors.
    Returns list of article dicts, empty list on persistent error.
    """
    params = {
        "query":         query,
        "mode":          "artlist",
        "maxrecords":    max_records,
        "format":        "json",
        "startdatetime": _fmt_gdelt_dt(start),
        "enddatetime":   _fmt_gdelt_dt(end),
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(GDELT_BASE, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 10 * (2 ** (attempt - 1))  # 10s, 20s, 40s, 80s
                log.warning("GDELT 429 rate limit — waiting %ds (attempt %d/%d)",
                            wait, attempt, retries)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data.get("articles") or []
        except Exception as exc:
            if attempt < retries:
                wait = 5 * attempt
                log.debug("GDELT transient error attempt %d/%d — retrying in %ds: %s",
                          attempt, retries, wait, exc)
                time.sleep(wait)
            else:
                log.warning("GDELT error '%s' [%s-%s]: %s",
                            query[:35], start.date(), end.date(), exc)
    return []


def _process_gdelt_articles(articles: list[dict], symbol: str) -> int:
    """Insert a batch of GDELT article dicts. Returns count of new rows."""
    inserted = 0
    for art in articles:
        url = art.get("url", "")
        if not url:
            continue
        pub_raw = art.get("seendate", "")
        try:
            pub_dt = datetime.strptime(pub_raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            pub_dt = datetime.now(timezone.utc)

        title   = art.get("title", "")
        summary = art.get("socialimage", "") or ""
        tags    = _tag_commodity(title + " " + summary)
        if symbol not in tags:
            tags = (tags + "," + symbol).strip(",")

        if _insert_article(_url_id(url), "gdelt", pub_dt, title, summary, url, tags):
            inserted += 1
    return inserted


def collect_gdelt(days_back: int = 7) -> int:
    """
    Collect recent articles from GDELT for all commodities (single window).
    Used for daily incremental runs. Returns count inserted.
    """
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    inserted_total = 0

    for symbol, query in COMMODITY_QUERIES.items():
        articles = _fetch_gdelt_window(query, start, end)
        n = _process_gdelt_articles(articles, symbol)
        log.info("GDELT %s [%dd]: %d articles, %d new", symbol, days_back, len(articles), n)
        inserted_total += n
        time.sleep(1)

    return inserted_total


def collect_gdelt_history(months: int = 12) -> int:
    """
    Backfill GDELT history by chunking the date range into 2-week windows.

    Each window fetches up to 250 articles per commodity query, giving
    much better coverage than a single query over a long range.

    Args:
        months: Number of months of history to fetch (default: 12).

    Returns:
        Total number of new articles inserted.
    """
    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=months * 30)

    # Build list of (window_start, window_end) pairs, oldest first
    windows: list[tuple[datetime, datetime]] = []
    cursor = start_dt
    while cursor < end_dt:
        window_end = min(cursor + timedelta(days=_WINDOW_DAYS), end_dt)
        windows.append((cursor, window_end))
        cursor = window_end

    total_windows  = len(windows)
    total_commodities = len(COMMODITY_QUERIES)
    total_calls    = total_windows * total_commodities

    # Preflight: check historical endpoint is accessible before committing to full run
    test_arts = _fetch_gdelt_window("oil", windows[0][0], windows[0][1], max_records=1, retries=1)
    if not test_arts and windows:
        # One silent failure is normal (no articles), but a 429 means rate limited
        import requests as _req
        probe = _req.get(GDELT_BASE, params={
            "query": "oil", "mode": "artlist", "maxrecords": 1, "format": "json",
            "startdatetime": _fmt_gdelt_dt(windows[0][0]),
            "enddatetime":   _fmt_gdelt_dt(windows[0][1]),
        }, timeout=10)
        if probe.status_code == 429:
            msg = (
                "GDELT historical API is rate-limited (429). "
                "The current-events endpoint works fine but the date-range endpoint "
                "has a separate quota. Wait a few hours (or until next UTC day) then retry:\n"
                "  python data/collector_news.py --months 12"
            )
            log.error(msg)
            print(f"\nERROR: {msg}")
            return 0

    log.info(
        "GDELT history: %d months -> %d windows x %d commodities = %d API calls (~%ds)",
        months, total_windows, total_commodities, total_calls, total_calls * 2,
    )
    print(f"Fetching {months} months of GDELT history: "
          f"{total_windows} windows x {total_commodities} commodities = {total_calls} calls "
          f"(~{total_calls * 2 // 60} min)")

    inserted_total = 0
    call_num = 0

    for win_start, win_end in windows:
        window_inserted = 0
        for symbol, query in COMMODITY_QUERIES.items():
            call_num += 1
            articles = _fetch_gdelt_window(query, win_start, win_end)
            n = _process_gdelt_articles(articles, symbol)
            window_inserted += n
            time.sleep(2)  # 2 req/s — conservative to avoid 429s

        inserted_total += window_inserted
        pct = call_num / total_calls * 100
        log.info(
            "[%d/%d calls, %.0f%%] window %s->%s: %d new articles (total so far: %d)",
            call_num, total_calls, pct,
            win_start.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d"),
            window_inserted, inserted_total,
        )
        print(f"  [{pct:3.0f}%] {win_start.date()} -> {win_end.date()}: "
              f"{window_inserted} new  (total: {inserted_total})")

    print(f"\nGDELT history complete: {inserted_total} new articles over {months} months")
    return inserted_total


# ── RSS ────────────────────────────────────────────────────────────────────────


def collect_rss() -> int:
    """Collect articles from all RSS feeds. Returns count inserted."""
    inserted_total = 0
    for feed_url in RSS_FEEDS:
        log.info("RSS: fetching %s", feed_url)
        try:
            feed = feedparser.parse(feed_url)
        except Exception as exc:
            log.warning("RSS parse error %s: %s", feed_url, exc)
            continue

        source = feed.feed.get("title", feed_url)
        for entry in feed.entries:
            url = entry.get("link", "")
            if not url:
                continue

            title = entry.get("title", "")
            summary = entry.get("summary", "")
            published = entry.get("published_parsed")
            pub_dt = (
                datetime(*published[:6], tzinfo=timezone.utc)
                if published
                else datetime.now(timezone.utc)
            )
            tags = _tag_commodity(title + " " + summary)
            ok = _insert_article(_url_id(url), source, pub_dt, title, summary, url, tags)
            if ok:
                inserted_total += 1

    return inserted_total


# ── public API ─────────────────────────────────────────────────────────────────


def get_unprocessed_news(limit: int = 100) -> pd.DataFrame:
    """
    Return unprocessed articles from news_raw, ordered by published_date desc.

    Args:
        limit: Max rows to return.

    Returns:
        DataFrame with all news_raw columns for unprocessed=FALSE rows.
    """
    conn = get_conn()
    df = conn.execute(
        "SELECT * FROM news_raw WHERE processed = FALSE ORDER BY published_date DESC LIMIT ?",
        [limit],
    ).df()
    conn.close()
    return df


def mark_processed(ids: list[str]) -> None:
    """
    Mark a list of article IDs as processed (sentiment scored).

    Args:
        ids: List of news_raw.id strings to update.
    """
    if not ids:
        return
    conn = get_conn()
    placeholders = ",".join(["?"] * len(ids))
    conn.execute(
        f"UPDATE news_raw SET processed = TRUE WHERE id IN ({placeholders})", ids
    )
    conn.close()


# ── main run ───────────────────────────────────────────────────────────────────


def run(backfill: bool = False, months: int = None) -> None:
    """
    Collect news from GDELT and RSS.

    Args:
        backfill: Fetch last 30 days via single GDELT query (fast).
        months:   Fetch N months of GDELT history via windowed queries (thorough).
                  When set, takes precedence over backfill.
    """
    init_schema()

    if months:
        log.info("Starting GDELT history backfill (%d months)", months)
        gdelt_count = collect_gdelt_history(months=months)
    else:
        days_back = 30 if backfill else 7
        log.info("Starting news collection (days_back=%d)", days_back)
        gdelt_count = collect_gdelt(days_back=days_back)

    rss_count = collect_rss()
    total = gdelt_count + rss_count

    print(f"News collected: {gdelt_count} GDELT + {rss_count} RSS = {total} new articles")
    log.info("News run complete: %d total inserted", total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CommodiSense news collector")
    parser.add_argument("--backfill", action="store_true",
                        help="Fetch last 30 days via single GDELT query")
    parser.add_argument("--months", type=int, default=None,
                        help="Fetch N months of GDELT history via windowed 2-week chunks "
                             "(e.g. --months 12). Slow but thorough (~4-5 min for 12 months).")
    args = parser.parse_args()
    run(backfill=args.backfill, months=args.months)

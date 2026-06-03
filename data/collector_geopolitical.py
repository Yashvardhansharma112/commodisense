"""
Geopolitical Collector — ingests geopolitical risk signals from:
  1. Wikipedia Current Events RSS (zero-cost, no API key)
  2. EIA Weekly Petroleum Supply Report (free, public)
  3. Hardcoded OPEC meeting calendar (2024-2025)

Scores each event 0-1 for commodity supply risk and stores in DuckDB
(table: geopolitical_events).

Usage:
    python data/collector_geopolitical.py            # incremental
    python data/collector_geopolitical.py --backfill # parse more history
"""

import argparse
import logging
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import feedparser
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.db import get_conn, init_schema

# ── constants ──────────────────────────────────────────────────────────────────

LOG_PATH = Path(__file__).parent / "logs" / "geopolitical.log"

# Risk keyword scoring — matched against lowercased headline text
RISK_KEYWORDS: dict[str, tuple[float, str]] = {
    # HIGH RISK (0.7–1.0)
    "sanctions":          (0.90, "policy"),
    "embargo":            (0.85, "policy"),
    "pipeline attack":    (0.95, "conflict"),
    "pipeline explosion": (0.95, "conflict"),
    "port closure":       (0.80, "logistics"),
    "export ban":         (0.85, "policy"),
    "production halt":    (0.85, "supply"),
    "war declared":       (0.95, "conflict"),
    "missile strike":     (0.90, "conflict"),
    # MEDIUM RISK (0.4–0.6)
    "protest":            (0.45, "social"),
    "strike":             (0.50, "labor"),
    "election":           (0.40, "political"),
    "drought declared":   (0.55, "weather"),
    "supply disruption":  (0.60, "supply"),
    "production cut":     (0.65, "supply"),
    "output reduced":     (0.55, "supply"),
    "conflict":           (0.50, "conflict"),
    "tension":            (0.45, "political"),
    # LOW RISK (0.0–0.3)
    "meeting":            (0.15, "policy"),
    "forecast":           (0.10, "analysis"),
    "review":             (0.10, "policy"),
    "statement":          (0.10, "political"),
    "agreement":          (0.25, "policy"),
    "deal":               (0.20, "policy"),
}

# Map headline keywords to affected commodities
COMMODITY_KEYWORDS: dict[str, list[str]] = {
    "CL=F":    ["oil", "petroleum", "opec", "crude", "brent", "wti", "barrel"],
    "NG=F":    ["natural gas", "lng", "gas pipeline", "gas supply"],
    "GC=F":    ["gold", "bullion", "safe haven"],
    "ZW=F":    ["wheat", "grain", "ukraine", "black sea", "russia wheat"],
    "ZC=F":    ["corn", "maize", "ethanol"],
    "ZS=F":    ["soybean", "soy", "soya"],
    "CT=F":    ["cotton", "textile"],
    "SB=F":    ["sugar", "cane"],
    "USDINR=X":["rupee", "india", "rbi", "inr"],
    "HG=F":    ["copper", "chile copper", "copper mine"],
}

# OPEC meeting dates — hardcoded calendar (update annually)
# Format: (date_str, headline)
OPEC_MEETINGS: list[tuple[str, str]] = [
    ("2024-06-02", "OPEC+ ministerial meeting — production policy review"),
    ("2024-11-26", "OPEC+ ministerial meeting — Q1 2025 output levels"),
    ("2025-03-03", "OPEC+ monitoring committee meeting"),
    ("2025-05-28", "OPEC+ ministerial meeting — production quota adjustment"),
    ("2025-11-05", "OPEC+ ministerial meeting — 2026 output strategy"),
]

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


def _score_headline(text: str) -> tuple[float, str]:
    """
    Score a headline for commodity supply risk using keyword rules.

    Returns:
        (risk_score, event_type) — score in [0, 1], type from RISK_KEYWORDS
    """
    lower = text.lower()
    best_score = 0.05  # minimum background noise
    best_type = "general"
    for kw, (score, evt_type) in RISK_KEYWORDS.items():
        if kw in lower and score > best_score:
            best_score = score
            best_type = evt_type
    return round(best_score, 2), best_type


def _tag_commodities(text: str) -> list[str]:
    """Return list of commodity symbols mentioned in text."""
    lower = text.lower()
    return [sym for sym, kws in COMMODITY_KEYWORDS.items() if any(k in lower for k in kws)]


def _insert_event(
    event_date: date,
    event_type: str,
    region: str,
    commodity: str,
    risk_score: float,
    headline: str,
    source: str,
) -> None:
    """Insert one geopolitical event row (no deduplication — events are additive)."""
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO geopolitical_events
                (date, event_type, region, commodity, risk_score, headline, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [event_date, event_type, region, commodity, risk_score, headline, source],
        )
    except Exception as exc:
        log.debug("Geopolitical insert error: %s", exc)
    finally:
        conn.close()


# ── Wikipedia Current Events ───────────────────────────────────────────────────


def _collect_wikipedia_events() -> int:
    """
    Scrape Wikipedia Current Events RSS for geopolitical headlines.
    Returns count of events inserted.
    """
    url = "https://en.wikipedia.org/w/index.php?title=Portal:Current_events&action=raw"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        text = resp.text
    except Exception as exc:
        log.warning("Wikipedia fetch error: %s", exc)
        return 0

    # Extract headline lines — Wikipedia raw format has ==YYYY Month DD== date headers
    # and * [[Link|Headline]] bullet lines
    count = 0
    current_date = date.today()
    date_pattern = re.compile(r"==+\s*(\d{4} \w+ \d{1,2})\s*==+")
    bullet_pattern = re.compile(r"^\*+\s*(.+)$", re.MULTILINE)

    for line in text.splitlines():
        date_match = date_pattern.search(line)
        if date_match:
            try:
                current_date = datetime.strptime(date_match.group(1), "%Y %B %d").date()
            except ValueError:
                pass
            continue

        bullet_match = bullet_pattern.match(line)
        if not bullet_match:
            continue

        # Strip wiki markup from headline
        headline = re.sub(r"\[\[([^\]|]+\|)?([^\]]+)\]\]", r"\2", bullet_match.group(1))
        headline = re.sub(r"<[^>]+>", "", headline).strip()
        if len(headline) < 20:
            continue

        score, evt_type = _score_headline(headline)
        commodities = _tag_commodities(headline) or ["CL=F"]  # default to oil for geopolitical

        for commodity in commodities:
            _insert_event(
                current_date, evt_type, "global", commodity, score, headline, "wikipedia"
            )
            count += 1

    return count


# ── EIA Petroleum Report ───────────────────────────────────────────────────────


def _collect_eia_petroleum() -> int:
    """
    Fetch EIA weekly petroleum supply report headlines via their RSS feed.
    Returns count of events inserted.
    """
    eia_rss = "https://www.eia.gov/rss/markets.xml"
    try:
        feed = feedparser.parse(eia_rss)
    except Exception as exc:
        log.warning("EIA RSS error: %s", exc)
        return 0

    count = 0
    for entry in feed.entries:
        title = entry.get("title", "")
        pub = entry.get("published_parsed")
        pub_date = datetime(*pub[:3]).date() if pub else date.today()

        score, evt_type = _score_headline(title)
        # EIA data primarily affects energy commodities
        for commodity in ["CL=F", "NG=F"]:
            _insert_event(pub_date, evt_type, "us_energy", commodity, score, title, "eia")
            count += 1

    return count


# ── OPEC Calendar ──────────────────────────────────────────────────────────────


def _insert_opec_meetings() -> int:
    """Insert OPEC meeting events from hardcoded calendar. Returns count inserted."""
    count = 0
    for date_str, headline in OPEC_MEETINGS:
        meeting_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        for commodity in ["CL=F", "NG=F"]:
            _insert_event(
                meeting_date, "policy", "opec", commodity, 0.50, headline, "opec_calendar"
            )
            count += 1
    return count


# ── public API ─────────────────────────────────────────────────────────────────


def get_risk_score(commodity: str, days: int = 30) -> float:
    """
    Return the average risk score for a commodity over the last N days.

    Args:
        commodity: Ticker symbol, e.g. "CL=F"
        days:      Look-back window in calendar days

    Returns:
        Float in [0, 1] — average risk score (0=no risk, 1=max risk).
        Returns 0.05 (baseline) if no events found.
    """
    cutoff = date.today() - timedelta(days=days)
    conn = get_conn()
    row = conn.execute(
        """
        SELECT AVG(risk_score) FROM geopolitical_events
        WHERE commodity = ? AND date >= ?
        """,
        [commodity, cutoff],
    ).fetchone()
    conn.close()
    return round(float(row[0]) if row and row[0] else 0.05, 3)


# ── main run ───────────────────────────────────────────────────────────────────


def run(backfill: bool = False) -> None:
    """
    Collect geopolitical signals from all sources.
    """
    init_schema()

    wiki_count = _collect_wikipedia_events()
    log.info("Wikipedia events: %d", wiki_count)

    eia_count = _collect_eia_petroleum()
    log.info("EIA petroleum events: %d", eia_count)

    opec_count = _insert_opec_meetings()
    log.info("OPEC calendar events: %d", opec_count)

    total = wiki_count + eia_count + opec_count
    print(f"Geopolitical: {wiki_count} Wikipedia + {eia_count} EIA + {opec_count} OPEC = {total} events")
    log.info("Geopolitical run complete: %d total", total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CommodiSense geopolitical collector")
    parser.add_argument("--backfill", action="store_true", help="Full history scan")
    args = parser.parse_args()
    run(backfill=args.backfill)

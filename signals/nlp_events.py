"""
Event Extractor — uses spaCy + rule-based patterns to detect commodity-relevant
events in news headlines and classify them as BULLISH / BEARISH / NEUTRAL.

Usage:
    python signals/nlp_events.py          # process recent news_raw articles
    python signals/nlp_events.py --limit 200
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.db import get_conn, init_schema

LOG_PATH = Path(__file__).parent.parent / "data" / "logs" / "events.log"
LOG_PATH.parent.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── event pattern definitions ──────────────────────────────────────────────────

# Each entry: (event_type, trigger_phrases, default_direction, base_severity)
EVENT_PATTERNS: list[tuple[str, list[str], str, int]] = [
    ("SUPPLY_SHOCK",    ["production cut", "harvest failure", "pipeline explosion",
                         "pipeline attack", "port strike", "port closure",
                         "sanctions imposed", "export ban", "output cut",
                         "supply disruption", "refinery fire", "mine closure"],     "BULLISH",  4),
    ("SUPPLY_INCREASE", ["production increase", "record output", "supply glut",
                         "oversupply", "output raised", "inventory build",
                         "stockpile rise"],                                          "BEARISH",  3),
    ("DEMAND_SURGE",    ["record imports", "stockpile build", "demand forecast raised",
                         "strong demand", "demand surge", "buying spree"],           "BULLISH",  3),
    ("DEMAND_DROP",     ["demand falls", "demand drop", "weak demand",
                         "economic slowdown", "recession fears", "demand cut"],      "BEARISH",  3),
    ("POLICY_CHANGE",   ["opec decision", "fed rate", "interest rate hike",
                         "interest rate cut", "tariff imposed", "trade deal",
                         "subsidy cut", "subsidy increase", "central bank"],         "NEUTRAL",  2),
    ("WEATHER_EVENT",   ["drought", "flood", "frost", "la niña", "el niño",
                         "monsoon failure", "heatwave", "crop damage",
                         "hurricane", "cyclone", "typhoon"],                         "BULLISH",  4),
    ("GEOPOLITICAL",    ["war", "armed conflict", "sanctions", "embargo",
                         "coup", "invasion", "airstrike", "blockade"],               "BULLISH",  5),
]

# Commodity-specific policy direction overrides
# (event_type, commodity) → direction
POLICY_DIRECTION_OVERRIDES: dict[tuple[str, str], str] = {
    ("POLICY_CHANGE", "GC=F"):    "BULLISH",   # rate cuts → gold up
    ("POLICY_CHANGE", "USDINR=X"):"BEARISH",   # rate hikes → stronger USD → bearish INR
    ("POLICY_CHANGE", "CL=F"):    "BEARISH",   # trade deal → supply up
}

# Region → commodities most affected by weather in that region
REGION_COMMODITY_WEATHER: dict[str, list[str]] = {
    "ukraine": ["ZW=F", "ZC=F"],
    "russia":  ["ZW=F"],
    "brazil":  ["ZS=F", "CT=F", "SB=F"],
    "india":   ["CT=F", "SB=F"],
    "us":      ["ZC=F", "ZS=F", "CL=F", "NG=F"],
    "texas":   ["CL=F", "NG=F"],
    "chile":   ["HG=F"],
    "middle east": ["CL=F", "NG=F"],
    "opec":    ["CL=F", "NG=F"],
    "gulf":    ["CL=F", "NG=F"],
}

# Commodity keywords for tagging (same as news collector)
COMMODITY_KEYWORDS: dict[str, list[str]] = {
    "CL=F":    ["oil", "petroleum", "crude", "opec", "brent", "wti"],
    "NG=F":    ["natural gas", "lng", "gas pipeline"],
    "GC=F":    ["gold", "bullion", "safe haven"],
    "ZW=F":    ["wheat", "grain", "flour"],
    "ZC=F":    ["corn", "maize"],
    "ZS=F":    ["soybean", "soy"],
    "CT=F":    ["cotton"],
    "SB=F":    ["sugar", "cane"],
    "USDINR=X":["rupee", "inr", "india forex"],
    "HG=F":    ["copper"],
}

# ── spaCy loader (lazy) ────────────────────────────────────────────────────────

_nlp = None


def _load_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            log.warning("en_core_web_sm not found — run: python -m spacy download en_core_web_sm")
            _nlp = None
    return _nlp


# ── helpers ────────────────────────────────────────────────────────────────────


def _detect_commodities(text: str) -> list[str]:
    lower = text.lower()
    return [sym for sym, kws in COMMODITY_KEYWORDS.items() if any(k in lower for k in kws)]


def _detect_location(text: str) -> str:
    """Extract first recognised location from text using spaCy GPE entities."""
    nlp = _load_nlp()
    if nlp is None:
        return "unknown"
    doc = nlp(text[:300])
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            return ent.text
    return "unknown"


def _resolve_direction(event_type: str, commodities: list[str], default: str) -> str:
    """Apply commodity-specific overrides to the default direction."""
    if not commodities:
        return default
    for commodity in commodities:
        override = POLICY_DIRECTION_OVERRIDES.get((event_type, commodity))
        if override:
            return override
    return default


def _severity_from_text(text: str, base: int) -> int:
    """Bump severity +1 if text contains intensifiers."""
    intensifiers = ["massive", "unprecedented", "historic", "emergency",
                    "catastrophic", "record", "major", "severe"]
    lower = text.lower()
    bump = sum(1 for w in intensifiers if w in lower)
    return min(5, base + bump)


# ── public API ─────────────────────────────────────────────────────────────────


def extract_events(text: str, event_date: str) -> list[dict]:
    """
    Extract commodity-relevant events from a text string.

    Args:
        text:       Article headline or summary.
        event_date: ISO date string "YYYY-MM-DD" for the event.

    Returns:
        List of dicts with keys: date, headline, event_type, commodity,
        location, severity, direction, source.
    """
    lower = text.lower()
    events: list[dict] = []

    for evt_type, phrases, default_direction, base_severity in EVENT_PATTERNS:
        matched_phrase = next((p for p in phrases if p in lower), None)
        if not matched_phrase:
            continue

        commodities = _detect_commodities(text)
        if not commodities:
            # For weather/geopolitical, try to infer commodity from location
            location = _detect_location(text)
            loc_lower = location.lower()
            for region, syms in REGION_COMMODITY_WEATHER.items():
                if region in loc_lower:
                    commodities = syms
                    break

        if not commodities:
            commodities = ["CL=F"]  # fallback to crude oil as most globally traded

        direction = _resolve_direction(evt_type, commodities, default_direction)
        severity = _severity_from_text(text, base_severity)
        location = _detect_location(text)

        for commodity in commodities:
            events.append({
                "date": event_date,
                "headline": text[:500],
                "event_type": evt_type,
                "commodity": commodity,
                "location": location,
                "severity": severity,
                "direction": direction,
                "source": "nlp_events",
            })

    return events


def process_batch(limit: int = 100) -> int:
    """
    Extract events from recent news_raw articles and store in extracted_events.

    Args:
        limit: Max articles to scan.

    Returns:
        Count of events extracted.
    """
    conn = get_conn()
    df = conn.execute(
        """
        SELECT id, title, summary, published_date
        FROM news_raw
        ORDER BY published_date DESC
        LIMIT ?
        """,
        [limit],
    ).df()
    conn.close()

    if df.empty:
        return 0

    total_events = 0
    conn = get_conn()
    for _, row in df.iterrows():
        text = f"{row.get('title', '')} {row.get('summary', '')}".strip()
        pub = str(row.get("published_date", date.today()))[:10]
        events = extract_events(text, pub)
        for evt in events:
            try:
                conn.execute(
                    """
                    INSERT INTO extracted_events
                        (date, headline, event_type, commodity, location,
                         severity, direction, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        evt["date"], evt["headline"], evt["event_type"],
                        evt["commodity"], evt["location"], evt["severity"],
                        evt["direction"], evt["source"],
                    ],
                )
                total_events += 1
            except Exception as exc:
                log.debug("Event insert error: %s", exc)
    conn.close()

    log.info("Extracted %d events from %d articles", total_events, len(df))
    return total_events


def get_event_features(commodity: str, days: int = 30) -> pd.DataFrame:
    """
    Return aggregated event features for a commodity over a date window.

    Args:
        commodity: Ticker symbol, e.g. "CL=F"
        days:      Look-back window in calendar days

    Returns:
        DataFrame with one row per date, columns:
        event_count, bullish_count, bearish_count, max_severity,
        direction_score (bullish=+1, bearish=-1, neutral=0, summed),
        supply_shock_flag (1 if any SUPPLY_SHOCK that day),
        policy_change_flag (1 if any POLICY_CHANGE that day)
    """
    cutoff = date.today() - timedelta(days=days)
    conn = get_conn()
    df = conn.execute(
        """
        SELECT date, event_type, direction, severity
        FROM extracted_events
        WHERE commodity = ? AND date >= ?
        ORDER BY date
        """,
        [commodity, cutoff],
    ).df()
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=[
            "date", "event_count", "bullish_count", "bearish_count",
            "max_severity", "direction_score", "supply_shock_flag", "policy_change_flag",
        ])

    df["dir_score"] = df["direction"].map({"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}).fillna(0)

    agg = df.groupby("date").agg(
        event_count=("event_type", "count"),
        bullish_count=("direction", lambda x: (x == "BULLISH").sum()),
        bearish_count=("direction", lambda x: (x == "BEARISH").sum()),
        max_severity=("severity", "max"),
        direction_score=("dir_score", "sum"),
        supply_shock_flag=("event_type", lambda x: int((x == "SUPPLY_SHOCK").any())),
        policy_change_flag=("event_type", lambda x: int((x == "POLICY_CHANGE").any())),
    ).reset_index()

    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    init_schema()
    n = process_batch(limit=args.limit)
    print(f"Extracted {n} events")

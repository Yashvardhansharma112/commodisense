"""
USDA NASS Agricultural Data Collector.
Free API key required: https://quickstats.nass.usda.gov/api
Set USDA_API_KEY in your .env file.

Collects weekly crop progress (% good + % excellent) and quarterly stocks for:
  ZC=F (Corn), ZW=F (Wheat), ZS=F (Soybeans), CT=F (Cotton)

Usage:
    python data/collector_usda.py
    python data/collector_usda.py --backfill
"""

import logging
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.db import get_conn

log = logging.getLogger(__name__)

USDA_API_BASE = "https://quickstats.nass.usda.gov/api/api_GET/"

USDA_COMMODITY_MAP = {
    "ZC=F": "CORN",
    "ZW=F": "WHEAT",
    "ZS=F": "SOYBEANS",
    "CT=F": "COTTON",
}

# (statisticcat_desc, short_desc_contains, date_type)
# date_type: "weekly" → week_ending; "quarterly" → year+reference_period_desc; "annual" → year-01-01
USDA_QUERIES = [
    ("CONDITION",  "PCT GOOD",                        "weekly"),
    ("CONDITION",  "PCT EXCELLENT",                   "weekly"),
    ("STOCKS",     "GRAIN - STOCKS, MEASURED IN BU",  "quarterly"),
    ("PRODUCTION", "MEASURED IN BU",                  "annual"),   # annual crop yield (WASDE proxy)
]

# USDA FAS PSD (no key needed) — global supply/demand for ending stocks-to-use ratio
FAS_PSD_BASE = "https://apps.fas.usda.gov/psdonline/api/v1/data"
FAS_COMMODITY_MAP = {
    "ZC=F": "Corn",
    "ZW=F": "Wheat",
    "ZS=F": "Soybean Oil",  # closest FAS mapping for soybeans
    "ZS_grain": "Soybeans",
}

_PERIOD_MONTH = {
    "FIRST OF JAN": "01", "FIRST OF MAR": "03", "FIRST OF JUN": "06",
    "FIRST OF SEP": "09", "FIRST OF DEC": "12",
    "MAR": "03", "JUN": "06", "SEP": "09", "DEC": "12",
}


def _parse_quarterly_date(year: str, period: str) -> str | None:
    """Convert USDA year + reference_period_desc to YYYY-MM-01 date string."""
    period_upper = str(period).strip().upper()
    for key, month in _PERIOD_MONTH.items():
        if key in period_upper:
            return f"{year}-{month}-01"
    return None


def _fetch_usda(api_key: str, commodity: str, stat_cat: str,
                desc_contains: str, date_type: str, start_year: int) -> pd.DataFrame:
    params = {
        "key":               api_key,
        "commodity_desc":    commodity,
        "statisticcat_desc": stat_cat,
        "year__GE":          str(start_year),
        "agg_level_desc":    "NATIONAL",
        "format":            "JSON",
    }
    try:
        resp = requests.get(USDA_API_BASE, params=params, timeout=45)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Filter by short_desc
        if "short_desc" in df.columns:
            df = df[df["short_desc"].str.upper().str.contains(desc_contains.upper(), na=False)]
        if df.empty:
            return pd.DataFrame()

        df["value"] = pd.to_numeric(
            df["Value"].astype(str).str.replace(",", ""), errors="coerce"
        )

        if date_type == "weekly":
            df["date"] = pd.to_datetime(df["week_ending"], errors="coerce").dt.date
        elif date_type == "annual":
            # Annual: use Jan 1 of the marketing year
            df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01", errors="coerce").dt.date
        else:
            # Quarterly: construct from year + reference_period_desc
            df["date"] = df.apply(
                lambda r: _parse_quarterly_date(r.get("year", ""), r.get("reference_period_desc", "")),
                axis=1,
            )
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

        return (
            df[["date", "value", "short_desc"]]
            .rename(columns={"short_desc": "metric"})
            .dropna(subset=["date", "value"])
            .sort_values("date")
            .reset_index(drop=True)
        )
    except Exception as exc:
        log.debug("USDA fetch error (%s %s): %s", commodity, stat_cat, exc)
        return pd.DataFrame()


def run(backfill: bool = False) -> str:
    api_key = os.environ.get("USDA_API_KEY", "").strip()
    if not api_key:
        log.info("USDA_API_KEY not set — skipping USDA collection (get free key at quickstats.nass.usda.gov/api)")
        return "USDA: skipped (USDA_API_KEY not set)"

    start_year = date.today().year - 5 if backfill else date.today().year - 2
    conn  = get_conn()
    total = 0

    for symbol, commodity in USDA_COMMODITY_MAP.items():
        for stat_cat, desc_contains, date_type in USDA_QUERIES:
            df = _fetch_usda(api_key, commodity, stat_cat, desc_contains, date_type, start_year)
            if df.empty:
                continue

            df = df.sort_values("date").reset_index(drop=True)
            df["yoy_chg_pct"] = df["value"].pct_change(periods=52) * 100

            for _, row in df.iterrows():
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO usda_crop
                            (date, commodity, metric, value, yoy_chg_pct)
                        VALUES (?, ?, ?, ?, ?)
                    """, [
                        row["date"], symbol, row["metric"], row["value"],
                        None if pd.isna(row["yoy_chg_pct"]) else row["yoy_chg_pct"],
                    ])
                    total += 1
                except Exception as exc:
                    log.debug("USDA insert error: %s", exc)

        log.info("USDA %s (%s): done", symbol, commodity)

    conn.close()
    return f"USDA: stored {total} rows"


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()
    from data.db import init_schema
    init_schema()
    print(run(backfill=args.backfill))

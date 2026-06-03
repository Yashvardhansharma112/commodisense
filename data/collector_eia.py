"""
EIA Energy Inventory Collector — US crude oil & natural gas storage.
Free API key required: https://www.eia.gov/opendata/register.php
Set EIA_API_KEY in your .env file.

Series:
  PET.WCRSTUS1.W          Weekly US crude oil stocks (thousand barrels)
  NG.NW2_EPG0_SWO_R48_BCF.W  Natural gas working storage (Bcf)

Usage:
    python data/collector_eia.py
    python data/collector_eia.py --backfill
"""

import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.db import get_conn

log = logging.getLogger(__name__)

EIA_API_BASE = "https://api.eia.gov/v2/seriesid"

EIA_SERIES = {
    "PET.WCRSTUS1.W":               "crude_stocks",
    "NG.NW2_EPG0_SWO_R48_BCF.W":   "natgas_storage",
}


def _fetch_eia_series(series_id: str, api_key: str, start: str) -> pd.DataFrame:
    url = f"{EIA_API_BASE}/{series_id}"
    params = {
        "api_key":              api_key,
        "data[0]":              "value",
        "start":                start,
        "sort[0][column]":      "period",
        "sort[0][direction]":   "asc",
        "length":               5000,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json().get("response", {}).get("data", [])
        if not rows:
            log.warning("EIA %s: no data returned", series_id)
            return pd.DataFrame()
        df = pd.DataFrame(rows)[["period", "value"]]
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"]  = pd.to_datetime(df["date"], errors="coerce").dt.date
        return df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    except Exception as exc:
        log.warning("EIA %s fetch error: %s", series_id, exc)
        return pd.DataFrame()


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add week-over-week change and z-score vs 5-year rolling average."""
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["chg_1w"]          = df["value"].diff()
    rolling_mean          = df["value"].rolling(260, min_periods=52).mean()
    rolling_std           = df["value"].rolling(260, min_periods=52).std().replace(0, 1)
    df["vs_5yr_avg"]      = (df["value"] - rolling_mean) / rolling_std
    return df[["date", "value", "chg_1w", "vs_5yr_avg"]]


def run(backfill: bool = False) -> str:
    api_key = os.environ.get("EIA_API_KEY", "").strip()
    if not api_key:
        log.info("EIA_API_KEY not set — skipping EIA collection (get free key at eia.gov/opendata)")
        return "EIA: skipped (EIA_API_KEY not set)"

    start = "2010-01-01" if backfill else (date.today() - timedelta(days=120)).isoformat()
    conn  = get_conn()
    total = 0

    for series_id, series_name in EIA_SERIES.items():
        df = _fetch_eia_series(series_id, api_key, start)
        if df.empty:
            continue
        df = _enrich(df)
        for _, row in df.iterrows():
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO eia_inventory (date, series, value, chg_1w, vs_5yr_avg)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    row["date"], series_name, row["value"],
                    None if pd.isna(row["chg_1w"])     else row["chg_1w"],
                    None if pd.isna(row["vs_5yr_avg"]) else row["vs_5yr_avg"],
                ])
                total += 1
            except Exception as exc:
                log.debug("EIA insert error: %s", exc)
        log.info("EIA %s: %d rows stored", series_name, len(df))

    conn.close()
    return f"EIA: stored {total} rows"


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

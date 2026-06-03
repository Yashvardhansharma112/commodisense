"""
CFTC Commitment of Traders (COT) Collector — disaggregated futures report.
No API key required. Data is published weekly (Friday 3:30 PM ET).

Usage:
    python data/collector_cot.py              # YTD + current year
    python data/collector_cot.py --backfill   # last 5 years
"""

import io
import logging
import sys
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.db import get_conn

log = logging.getLogger(__name__)

# Lowercase partial matches → symbol (longer keys first avoids substring collisions)
COT_NAME_MAP = {
    "crude oil, light sweet":   "CL=F",
    "natural gas (nymex)":      "NG=F",  # pre-2022 CFTC naming
    "henry hub penultimate":    "NG=F",  # NYMEX Henry Hub variant
    "nat gas nyme":             "NG=F",  # 2022+ NYMEX Natural Gas
    "gold":                     "GC=F",
    "copper- #1":               "HG=F",  # actual CFTC name (not "copper-grade")
    "wheat-srw":                "ZW=F",
    "corn":                     "ZC=F",
    "soybeans":                 "ZS=F",
    "cotton no. 2":             "CT=F",
    "sugar no. 11":             "SB=F",
}

# Ordered from most- to least-specific so "corn" doesn't match "acorn" etc.
_SORTED_KEYS = sorted(COT_NAME_MAP.keys(), key=len, reverse=True)

REQUIRED_COLS = [
    "Market_and_Exchange_Names",
    "As_of_Date_In_Form_YYYY-MM-DD",
    "Open_Interest_All",
    "Prod_Merc_Positions_Long_All",
    "Prod_Merc_Positions_Short_All",
    "M_Money_Positions_Long_All",
    "M_Money_Positions_Short_All",
]
ALT_DATE_COL = "Report_Date_as_YYYY-MM-DD"


def _fetch_zip(url: str) -> pd.DataFrame:
    try:
        resp = requests.get(url, timeout=90)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("COT download failed (%s): %s", url, exc)
        return pd.DataFrame()

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            txt_files = [f for f in zf.namelist() if f.lower().endswith(".txt")]
            if not txt_files:
                log.warning("No .txt in zip: %s", url)
                return pd.DataFrame()
            with zf.open(txt_files[0]) as f:
                return pd.read_csv(f, low_memory=False)
    except Exception as exc:
        log.warning("COT zip parse error: %s", exc)
        return pd.DataFrame()


def _match_symbol(name: str) -> str | None:
    lower = name.lower()
    for key in _SORTED_KEYS:
        if key in lower:
            return COT_NAME_MAP[key]
    return None


def _parse_and_store(raw: pd.DataFrame) -> int:
    if raw.empty:
        return 0

    raw.columns = raw.columns.str.strip()

    # Accept alternate date column name
    if "As_of_Date_In_Form_YYYY-MM-DD" not in raw.columns and ALT_DATE_COL in raw.columns:
        raw = raw.rename(columns={ALT_DATE_COL: "As_of_Date_In_Form_YYYY-MM-DD"})

    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        log.error("COT missing columns: %s (available: %s)", missing, list(raw.columns[:10]))
        return 0

    df = raw[REQUIRED_COLS].copy()
    df.columns = ["market_name", "date", "open_interest",
                  "comm_long", "comm_short", "mm_long", "mm_short"]

    df["symbol"] = df["market_name"].apply(_match_symbol)
    df = df[df["symbol"].notna()].copy()
    if df.empty:
        return 0

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    for col in ["open_interest", "comm_long", "comm_short", "mm_long", "mm_short"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ""), errors="coerce"
        ).fillna(0)

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    df["commercial_net_long"] = df["comm_long"] - df["comm_short"]
    df["mm_net_long"]         = df["mm_long"]   - df["mm_short"]
    df["commercial_net_pct"]  = df["commercial_net_long"] / df["open_interest"].replace(0, 1)
    df["mm_net_pct"]          = df["mm_net_long"] / df["open_interest"].replace(0, 1)
    df["commercial_chg_1w"]   = df.groupby("symbol")["commercial_net_long"].diff()
    df["mm_chg_1w"]           = df.groupby("symbol")["mm_net_long"].diff()

    conn = get_conn()
    inserted = 0
    for _, row in df.iterrows():
        try:
            conn.execute("""
                INSERT OR REPLACE INTO cot_data
                    (date, symbol, commercial_net_long, commercial_net_pct,
                     mm_net_long, mm_net_pct, commercial_chg_1w, mm_chg_1w, open_interest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                row["date"], row["symbol"],
                row["commercial_net_long"], row["commercial_net_pct"],
                row["mm_net_long"],         row["mm_net_pct"],
                None if pd.isna(row["commercial_chg_1w"]) else row["commercial_chg_1w"],
                None if pd.isna(row["mm_chg_1w"])         else row["mm_chg_1w"],
                row["open_interest"],
            ])
            inserted += 1
        except Exception as exc:
            log.debug("COT insert error: %s", exc)
    conn.close()
    return inserted


def run(backfill: bool = False) -> str:
    current_year = date.today().year
    total = 0

    start_year = current_year - 4 if backfill else current_year
    for year in range(start_year, current_year + 1):
        url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
        raw = _fetch_zip(url)
        n = _parse_and_store(raw)
        log.info("COT %d: %d rows", year, n)
        total += n

    return f"COT: stored {total} rows"


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()
    from data.db import init_schema
    init_schema()
    print(run(backfill=args.backfill))

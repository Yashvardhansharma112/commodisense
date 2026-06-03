"""
FRED Macroeconomic Data Collector — Federal Reserve Economic Data.
Uses public CSV endpoints — no API key required.

Series collected:
  DTWEXBGS  Broad USD index (weekly)
  T10YIE    10-yr breakeven inflation (daily)
  VIXCLS    CBOE VIX (daily)
  GS10      10-yr Treasury yield (daily)
  STLFSI2   St. Louis Financial Stress Index (weekly)
  INDPRO    Industrial Production Index (monthly)
  FEDFUNDS  Fed Funds Effective Rate (monthly)

Usage:
    python data/collector_fred.py
    python data/collector_fred.py --backfill
"""

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.db import get_conn

log = logging.getLogger(__name__)

FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

FRED_SERIES = {
    # Note: DTWEXBGS (DXY) and VIXCLS are fetched via yfinance fallback
    # because FRED's CSV endpoint is unreliable for these high-traffic series.
    "T10YIE":   "inflation_exp",
    "GS10":     "treasury_10y",
    "STLFSI2":  "financial_stress",
    "INDPRO":   "indpro",
    "FEDFUNDS": "fedfunds",
}

# Fetched via yfinance when FRED CSV endpoint is unavailable
YFINANCE_FALLBACK = {
    "DX-Y.NYB": "dxy",
    "^VIX":     "vix",
}


def _fetch_series(series_id: str) -> pd.DataFrame:
    url = f"{FRED_CSV_BASE}?id={series_id}"
    try:
        resp = requests.get(url, timeout=90)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"]  = pd.to_datetime(df["date"], errors="coerce").dt.date
        return df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    except Exception as exc:
        log.warning("FRED %s fetch error: %s", series_id, exc)
        return pd.DataFrame(columns=["date", "value"])


def _build_daily_frame(all_series: dict[str, pd.DataFrame], start_date: str) -> pd.DataFrame:
    """Forward-fill all series onto a daily calendar from start_date to today."""
    start = pd.to_datetime(start_date).date()
    end   = date.today()
    date_range = pd.date_range(start, end, freq="D").date
    base = pd.DataFrame({"date": date_range})

    # FRED series: key=series_id, value df has "value" column
    for series_id, col_name in FRED_SERIES.items():
        df = all_series.get(series_id, pd.DataFrame())
        if df.empty:
            base[col_name] = None
            continue
        df_col = df.rename(columns={"value": col_name})[["date", col_name]]
        base = base.merge(df_col, on="date", how="left")
        base[col_name] = base[col_name].ffill()

    # yfinance fallback series: key=col_name, df already has col_name column
    for col_name in YFINANCE_FALLBACK.values():
        df = all_series.get(col_name, pd.DataFrame())
        if df.empty or col_name not in df.columns:
            base[col_name] = None
            continue
        base = base.merge(df[["date", col_name]], on="date", how="left")
        base[col_name] = base[col_name].ffill()

    return base


def _fetch_yfinance(ticker: str, start_date: str) -> pd.DataFrame:
    """Fallback: fetch a single series from Yahoo Finance."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame(columns=["date", "value"])
        df = df[["Close"]].copy()
        df.columns = ["value"]
        df.index = pd.to_datetime(df.index).date
        df = df.reset_index().rename(columns={"index": "date"})
        return df.dropna(subset=["value"])
    except Exception as exc:
        log.warning("yfinance %s error: %s", ticker, exc)
        return pd.DataFrame(columns=["date", "value"])


def run(backfill: bool = False) -> str:
    start_date = "2010-01-01" if backfill else (date.today() - timedelta(days=120)).isoformat()
    log.info("Fetching FRED macro series (start=%s)...", start_date)

    all_series: dict[str, pd.DataFrame] = {}
    for series_id in FRED_SERIES:
        df = _fetch_series(series_id)
        if not df.empty:
            df = df[df["date"] >= pd.to_datetime(start_date).date()]
        all_series[series_id] = df
        log.info("  FRED %s: %d rows", series_id, len(df))

    # Fetch DXY and VIX via yfinance (FRED CSV endpoint is unreliable for these)
    for ticker, col_name in YFINANCE_FALLBACK.items():
        df = _fetch_yfinance(ticker, start_date)
        all_series[col_name] = df.rename(columns={"value": col_name}) if not df.empty else pd.DataFrame()
        log.info("  yfinance %s → %s: %d rows", ticker, col_name, len(df))

    # Rebuild combined FRED_SERIES map including yfinance fallback columns
    all_col_names = dict(FRED_SERIES)
    all_col_names.update({v: v for v in YFINANCE_FALLBACK.values()})

    daily = _build_daily_frame(all_series, start_date)

    conn = get_conn()
    inserted = 0
    for _, row in daily.iterrows():
        try:
            conn.execute("""
                INSERT OR REPLACE INTO fred_data
                    (date, dxy, inflation_exp, vix, treasury_10y,
                     financial_stress, indpro, fedfunds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                row["date"],
                row.get("dxy"),
                row.get("inflation_exp"),
                row.get("vix"),
                row.get("treasury_10y"),
                row.get("financial_stress"),
                row.get("indpro"),
                row.get("fedfunds"),
            ])
            inserted += 1
        except Exception as exc:
            log.debug("FRED insert error: %s", exc)
    conn.close()
    return f"FRED: stored {inserted} rows"


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()
    from data.db import init_schema
    init_schema()
    print(run(backfill=args.backfill))

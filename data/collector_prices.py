"""
Price Collector — downloads OHLCV data for all 10 commodities via yfinance
and stores incrementally in DuckDB (table: prices).

Usage:
    python data/collector_prices.py            # incremental update
    python data/collector_prices.py --backfill # 5-year history
"""

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.db import get_conn, init_schema

# ── constants ──────────────────────────────────────────────────────────────────

SYMBOLS = [
    "CL=F",    # Crude Oil
    "NG=F",    # Natural Gas
    "GC=F",    # Gold
    "ZW=F",    # Wheat
    "ZC=F",    # Corn
    "ZS=F",    # Soybeans
    "CT=F",    # Cotton
    "SB=F",    # Sugar
    "USDINR=X",# USD/INR
    "HG=F",    # Copper
]

BACKFILL_YEARS = 5
LOG_PATH = Path(__file__).parent / "logs" / "prices.log"

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


def _last_stored_date(symbol: str) -> date | None:
    """Return the most recent date we have in DB for a symbol, or None."""
    conn = get_conn()
    row = conn.execute(
        "SELECT MAX(date) FROM prices WHERE symbol = ?", [symbol]
    ).fetchone()
    conn.close()
    return row[0] if row and row[0] else None


def _fetch_with_retry(symbol: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """Download OHLCV from yfinance with up to `retries` attempts."""
    for attempt in range(1, retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, auto_adjust=False)
            if df.empty:
                log.warning("%s: empty response (attempt %d/%d)", symbol, attempt, retries)
            else:
                return df
        except Exception as exc:
            log.warning("%s: download error (attempt %d/%d): %s", symbol, attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)  # exponential back-off
    return pd.DataFrame()


def _upsert_prices(df: pd.DataFrame, symbol: str) -> int:
    """Insert rows that don't already exist. Returns count of new rows inserted."""
    if df.empty:
        return 0

    # Normalize column names from yfinance
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Drop weekends/holidays — yfinance already does this but be defensive
    df = df[df["date"].apply(lambda d: d.weekday() < 5)]

    conn = get_conn()
    inserted = 0
    for _, row in df.iterrows():
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO prices
                    (date, symbol, open, high, low, close, volume, adj_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    row["date"],
                    symbol,
                    float(row.get("open", 0) or 0),
                    float(row.get("high", 0) or 0),
                    float(row.get("low", 0) or 0),
                    float(row.get("close", 0) or 0),
                    float(row.get("volume", 0) or 0),
                    float(row.get("adj_close", row.get("close", 0)) or 0),
                ],
            )
            inserted += 1
        except Exception as exc:
            log.debug("Skip row %s %s: %s", symbol, row["date"], exc)
    conn.close()
    return inserted


# ── public API ─────────────────────────────────────────────────────────────────


def get_price_history(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    Return historical OHLCV data for a symbol from DuckDB.

    Args:
        symbol: Commodity ticker, e.g. "GC=F"
        days:   Number of calendar days to look back (default 365)

    Returns:
        DataFrame with columns: date, open, high, low, close, volume, adj_close
    """
    cutoff = date.today() - timedelta(days=days)
    conn = get_conn()
    df = conn.execute(
        "SELECT * FROM prices WHERE symbol = ? AND date >= ? ORDER BY date",
        [symbol, cutoff],
    ).df()
    conn.close()
    return df


def get_latest_price(symbol: str) -> dict:
    """
    Return the most recent price row for a symbol.

    Returns:
        dict with keys: price, change_pct, date
        Returns empty dict if symbol has no data.
    """
    conn = get_conn()
    rows = conn.execute(
        "SELECT date, close, open FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT 2",
        [symbol],
    ).fetchall()
    conn.close()

    if not rows:
        return {}

    latest = rows[0]
    prev_close = rows[1][1] if len(rows) > 1 else latest[2]
    change_pct = ((latest[1] - prev_close) / prev_close * 100) if prev_close else 0.0

    return {
        "price": latest[1],
        "change_pct": round(change_pct, 2),
        "date": str(latest[0]),
    }


# ── main run ───────────────────────────────────────────────────────────────────


def run(backfill: bool = False) -> None:
    """
    Main entry point. Downloads price data for all symbols.

    Args:
        backfill: If True, fetch BACKFILL_YEARS of history regardless of what's stored.
    """
    init_schema()
    total_rows = 0
    symbols_ok = 0
    end_date = date.today().isoformat()

    for symbol in SYMBOLS:
        if backfill:
            start_date = (date.today() - timedelta(days=365 * BACKFILL_YEARS)).isoformat()
        else:
            last = _last_stored_date(symbol)
            # Start from day after last stored date, or 30 days back as a safe default
            start_date = (
                (last + timedelta(days=1)).isoformat()
                if last
                else (date.today() - timedelta(days=30)).isoformat()
            )

        log.info("%s: fetching %s -> %s", symbol, start_date, end_date)
        df = _fetch_with_retry(symbol, start=start_date, end=end_date)

        if df.empty:
            log.warning("%s: no data returned, skipping", symbol)
            continue

        inserted = _upsert_prices(df, symbol)
        log.info("%s: inserted %d new rows", symbol, inserted)
        total_rows += inserted
        symbols_ok += 1

    print(f"Downloaded {total_rows} rows for {symbols_ok} symbols")
    log.info("Run complete: %d rows / %d symbols", total_rows, symbols_ok)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CommodiSense price collector")
    parser.add_argument("--backfill", action="store_true", help="Fetch full 5-year history")
    args = parser.parse_args()
    run(backfill=args.backfill)

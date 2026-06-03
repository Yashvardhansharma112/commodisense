"""
Price Feature Engineer — computes all technical, momentum, seasonality, and
cross-commodity features from stored price data.

All features are derived from DuckDB prices table — no live API calls.

Usage (standalone):
    python signals/price_features.py --symbol GC=F --date 2024-06-01
"""

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.db import get_conn

log = logging.getLogger(__name__)

# ── commodity metadata ─────────────────────────────────────────────────────────

SYMBOL_NAMES: dict[str, str] = {
    "CL=F":    "Crude Oil",
    "NG=F":    "Natural Gas",
    "GC=F":    "Gold",
    "ZW=F":    "Wheat",
    "ZC=F":    "Corn",
    "ZS=F":    "Soybeans",
    "CT=F":    "Cotton",
    "SB=F":    "Sugar",
    "USDINR=X":"USD/INR",
    "HG=F":    "Copper",
}

ALL_SYMBOLS = list(SYMBOL_NAMES.keys())

# Harvest season windows (month_start, month_end) — inclusive
HARVEST_SEASONS: dict[str, list[tuple[int, int]]] = {
    "ZW=F":    [(6, 8)],          # Northern hemisphere wheat: June–August
    "ZC=F":    [(9, 11)],         # US corn: September–November
    "ZS=F":    [(9, 11), (3, 5)], # US + Brazil soy harvest windows
    "CT=F":    [(9, 12)],         # US cotton: September–December
    "SB=F":    [(4, 6), (10, 12)],# Brazil + India sugar
}

# OPEC meeting dates (ISO strings) — extend annually
OPEC_MEETING_DATES: list[str] = [
    "2024-06-02", "2024-11-26",
    "2025-03-03", "2025-05-28", "2025-11-05",
    "2026-03-02", "2026-06-01",
]

# ── helpers ────────────────────────────────────────────────────────────────────


def _load_prices(symbol: str, days: int = 400) -> pd.DataFrame:
    """
    Load OHLCV data for a symbol from DuckDB.
    Returns DataFrame sorted by date ascending with at least `days` rows of buffer.
    """
    cutoff = date.today() - timedelta(days=days)
    conn = get_conn()
    df = conn.execute(
        "SELECT date, open, high, low, close, volume, adj_close FROM prices "
        "WHERE symbol = ? AND date >= ? ORDER BY date",
        [symbol, cutoff],
    ).df()
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _load_all_prices_latest() -> dict[str, float]:
    """Return latest close price for every symbol (used for cross-commodity ratios)."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT symbol, close FROM prices p
        WHERE date = (SELECT MAX(date) FROM prices p2 WHERE p2.symbol = p.symbol)
        """
    ).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def _days_to_next_opec(as_of: date) -> int:
    """Return calendar days until the next OPEC meeting on or after `as_of`."""
    future = [
        (datetime.strptime(d, "%Y-%m-%d").date() - as_of).days
        for d in OPEC_MEETING_DATES
        if datetime.strptime(d, "%Y-%m-%d").date() >= as_of
    ]
    return min(future) if future else 180  # default if no upcoming date in list


def _harvest_season_flag(symbol: str, month: int) -> int:
    """Return 1 if `month` falls within any harvest window for the symbol."""
    windows = HARVEST_SEASONS.get(symbol, [])
    for start_m, end_m in windows:
        if start_m <= month <= end_m:
            return 1
    return 0


def _compute_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append technical analysis columns using pandas-ta.
    Works on a copy of df — returns augmented DataFrame.
    """
    try:
        import pandas_ta as ta

        df = df.copy()
        df.ta.rsi(length=14, append=True)           # RSI_14
        df.ta.macd(fast=12, slow=26, signal=9, append=True)   # MACD_12_26_9, etc.
        df.ta.bbands(length=20, std=2, append=True) # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
        df.ta.atr(length=14, append=True)           # ATRr_14
        df.ta.sma(length=20, append=True)           # SMA_20
        df.ta.sma(length=50, append=True)           # SMA_50

    except ImportError:
        log.warning("pandas-ta not installed — TA features will be NaN")

    return df


def _safe(val) -> float:
    """Return 0.0 for NaN/None values to keep feature vector clean."""
    if val is None:
        return 0.0
    try:
        v = float(val)
        return 0.0 if (v != v) else v  # NaN check without numpy
    except (TypeError, ValueError):
        return 0.0


# ── public API ─────────────────────────────────────────────────────────────────


def get_price_features(symbol: str, as_of_date: str = None) -> dict:
    """
    Compute all price-based features for a symbol on a given date.

    Args:
        symbol:      Commodity ticker, e.g. "GC=F"
        as_of_date:  ISO date string. Defaults to today.

    Returns:
        Flat dict of feature_name → float value.
        All values are guaranteed non-NaN (NaN → 0.0).
    """
    target_date = (
        datetime.strptime(as_of_date, "%Y-%m-%d").date()
        if as_of_date
        else date.today()
    )

    df = _load_prices(symbol, days=400)
    if df.empty or len(df) < 20:
        log.warning("%s: insufficient price history for feature engineering", symbol)
        return {}

    df = _compute_ta_features(df)

    # Locate the row nearest to target_date (T-1 to avoid lookahead)
    df["_date"] = df["date"].dt.date
    available = df[df["_date"] <= target_date]
    if available.empty:
        return {}
    row = available.iloc[-1]
    idx = available.index[-1]

    close = _safe(row["close"])
    if close == 0:
        return {}

    # ── momentum / returns ──
    def _pct_change(lookback_days: int) -> float:
        past = df[df["_date"] <= (target_date - timedelta(days=lookback_days))]
        if past.empty:
            return 0.0
        past_close = _safe(past.iloc[-1]["close"])
        return round((close - past_close) / past_close * 100, 4) if past_close else 0.0

    ret_1d  = _pct_change(1)
    ret_7d  = _pct_change(7)
    ret_14d = _pct_change(14)
    ret_30d = _pct_change(30)
    ret_60d = _pct_change(60)
    momentum_score = float(np.sign(ret_7d) + np.sign(ret_30d))  # -2 to +2

    # ── technical indicators from pandas-ta ──
    rsi = _safe(row.get("RSI_14"))

    macd = _safe(row.get("MACD_12_26_9"))
    macd_signal_line = _safe(row.get("MACDs_12_26_9"))
    macd_signal = 1 if macd > macd_signal_line else (-1 if macd < macd_signal_line else 0)

    bb_lower = _safe(row.get("BBL_20_2.0"))
    bb_upper = _safe(row.get("BBU_20_2.0"))
    bb_range = bb_upper - bb_lower
    bb_position = ((close - bb_lower) / bb_range) if bb_range > 0 else 0.5

    atr = _safe(row.get("ATRr_14"))
    atr_pct = (atr / close * 100) if close > 0 else 0.0

    sma20 = _safe(row.get("SMA_20"))
    sma50 = _safe(row.get("SMA_50"))
    sma_20_50_cross = 1 if sma20 > sma50 else -1

    # ── seasonality ──
    month = target_date.month
    day_of_week = target_date.weekday()  # 0=Monday
    month_sin = float(np.sin(2 * np.pi * month / 12))
    month_cos = float(np.cos(2 * np.pi * month / 12))
    harvest_flag = _harvest_season_flag(symbol, month)

    # Oil/gas: days to next OPEC meeting
    days_opec = _days_to_next_opec(target_date) if symbol in ("CL=F", "NG=F") else 0

    # ── cross-commodity features ──
    latest_prices = _load_all_prices_latest()
    cl_price = latest_prices.get("CL=F", 0)
    gc_price = latest_prices.get("GC=F", 0)
    oil_gold_ratio = round(cl_price / gc_price, 6) if gc_price > 0 else 0.0

    # DXY proxy: inverted gold price normalised (gold up → USD weak)
    gc_hist_mean = df["close"].mean() if not df.empty else 1.0
    dxy_proxy = round(1 - (gc_price / gc_hist_mean) if gc_hist_mean > 0 else 0.5, 4)

    return {
        # Technical
        "rsi_14":           round(rsi, 4),
        "macd_signal":      macd_signal,
        "bb_position":      round(bb_position, 4),
        "atr_14":           round(atr, 4),
        "atr_pct":          round(atr_pct, 4),
        "sma_20_50_cross":  sma_20_50_cross,
        # Momentum
        "return_1d":        ret_1d,
        "return_7d":        ret_7d,
        "return_14d":       ret_14d,
        "return_30d":       ret_30d,
        "return_60d":       ret_60d,
        "momentum_score":   momentum_score,
        # Seasonality
        "month_sin":        round(month_sin, 4),
        "month_cos":        round(month_cos, 4),
        "day_of_week":      day_of_week,
        "harvest_season_flag": harvest_flag,
        "days_to_opec_meeting": days_opec,
        # Cross-commodity
        "oil_gold_ratio":   oil_gold_ratio,
        "dxy_proxy":        dxy_proxy,
    }


def build_feature_matrix(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Build a feature matrix for model training — one row per trading day.

    Args:
        symbol:     Commodity ticker
        start_date: ISO date string "YYYY-MM-DD"
        end_date:   ISO date string "YYYY-MM-DD"

    Returns:
        DataFrame with one row per date, all price feature columns.
        Does NOT include target variable — caller adds that.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Load full price history once
    df_prices = _load_prices(symbol, days=(end - start).days + 500)
    if df_prices.empty:
        return pd.DataFrame()

    df_prices = _compute_ta_features(df_prices)
    df_prices["_date"] = df_prices["date"].dt.date

    latest_prices = _load_all_prices_latest()
    cl_price = latest_prices.get("CL=F", 0)
    gc_price = latest_prices.get("GC=F", 0)
    gc_hist_mean = df_prices["close"].mean() if not df_prices.empty else 1.0

    rows: list[dict] = []
    for _, price_row in df_prices.iterrows():
        row_date = price_row["_date"]
        if not (start <= row_date <= end):
            continue

        close = _safe(price_row["close"])
        if close == 0:
            continue

        # Returns — look back within df_prices to avoid reloading
        def _ret(days: int) -> float:
            past = df_prices[df_prices["_date"] <= (row_date - timedelta(days=days))]
            if past.empty:
                return 0.0
            pc = _safe(past.iloc[-1]["close"])
            return round((close - pc) / pc * 100, 4) if pc else 0.0

        ret_1d  = _ret(1)
        ret_7d  = _ret(7)
        ret_14d = _ret(14)
        ret_30d = _ret(30)
        ret_60d = _ret(60)

        rsi = _safe(price_row.get("RSI_14"))
        macd = _safe(price_row.get("MACD_12_26_9"))
        macd_sig = _safe(price_row.get("MACDs_12_26_9"))
        bb_lower = _safe(price_row.get("BBL_20_2.0"))
        bb_upper = _safe(price_row.get("BBU_20_2.0"))
        bb_range = bb_upper - bb_lower
        atr = _safe(price_row.get("ATRr_14"))
        sma20 = _safe(price_row.get("SMA_20"))
        sma50 = _safe(price_row.get("SMA_50"))

        month = row_date.month
        rows.append({
            "date":                 row_date,
            "rsi_14":               round(rsi, 4),
            "macd_signal":          1 if macd > macd_sig else (-1 if macd < macd_sig else 0),
            "bb_position":          round((close - bb_lower) / bb_range, 4) if bb_range > 0 else 0.5,
            "atr_14":               round(atr, 4),
            "atr_pct":              round(atr / close * 100, 4) if close > 0 else 0.0,
            "sma_20_50_cross":      1 if sma20 > sma50 else -1,
            "return_1d":            ret_1d,
            "return_7d":            ret_7d,
            "return_14d":           ret_14d,
            "return_30d":           ret_30d,
            "return_60d":           ret_60d,
            "momentum_score":       float(np.sign(ret_7d) + np.sign(ret_30d)),
            "month_sin":            round(float(np.sin(2 * np.pi * month / 12)), 4),
            "month_cos":            round(float(np.cos(2 * np.pi * month / 12)), 4),
            "day_of_week":          row_date.weekday(),
            "harvest_season_flag":  _harvest_season_flag(symbol, month),
            "days_to_opec_meeting": _days_to_next_opec(row_date) if symbol in ("CL=F", "NG=F") else 0,
            "oil_gold_ratio":       round(cl_price / gc_price, 6) if gc_price > 0 else 0.0,
            "dxy_proxy":            round(1 - gc_price / gc_hist_mean, 4) if gc_hist_mean > 0 else 0.5,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="GC=F")
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    features = get_price_features(args.symbol, args.date)
    print(json.dumps(features, indent=2))

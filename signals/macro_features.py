"""
Macro Feature Engineering — COT, FRED, EIA, USDA signals.

No lookahead guarantee: all features use data available at or before as_of_date.
Missing data returns zero (model learns to weight it accordingly via has_*_data flags).

Public API:
    build_macro_dataframe(symbol, start_date, end_date) → pd.DataFrame  (training)
    get_macro_features(symbol, as_of_date)              → dict            (inference)
"""

import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.db import get_conn

log = logging.getLogger(__name__)

# Symbols that have EIA inventory data
EIA_SYMBOL_MAP = {
    "CL=F": "crude_stocks",
    "NG=F": "natgas_storage",
}

# Symbols that have USDA crop data
USDA_SYMBOLS = {"ZW=F", "ZC=F", "ZS=F", "CT=F"}

# All macro feature names — used to guarantee consistent columns across training and inference
ALL_MACRO_FEATURES = [
    # COT
    "cot_commercial_net",
    "cot_commercial_net_pct",
    "cot_mm_net",
    "cot_mm_net_pct",
    "cot_commercial_chg_1w",
    "cot_mm_chg_1w",
    "cot_open_interest",
    "has_cot_data",
    # FRED
    "fred_dxy",
    "fred_dxy_chg_1w",
    "fred_dxy_chg_4w",
    "fred_inflation_exp",
    "fred_vix",
    "fred_vix_chg_1w",
    "fred_vix_high",
    "fred_treasury_10y",
    "fred_financial_stress",
    "fred_indpro",
    "fred_fedfunds",
    "fred_yield_inv",
    "fred_china_pmi",
    "fred_copper_basis",
    "has_fred_data",
    # EIA
    "eia_crude_stocks",
    "eia_crude_chg_1w",
    "eia_crude_vs_5yr",
    "eia_crude_draw",
    "eia_natgas_stocks",
    "eia_natgas_chg_1w",
    "eia_natgas_vs_5yr",
    "eia_natgas_draw",
    "has_eia_data",
    # USDA
    "usda_crop_good_exc",
    "usda_crop_good_exc_chg",
    "usda_stocks",
    "usda_stocks_yoy",
    "usda_production",
    "has_usda_data",
]

_ZERO_ROW = {k: 0.0 for k in ALL_MACRO_FEATURES}


# ── training dataframes ────────────────────────────────────────────────────────


def _load_cot(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    conn = get_conn()
    df = conn.execute("""
        SELECT date,
               commercial_net_long  AS cot_commercial_net,
               commercial_net_pct   AS cot_commercial_net_pct,
               mm_net_long          AS cot_mm_net,
               mm_net_pct           AS cot_mm_net_pct,
               commercial_chg_1w    AS cot_commercial_chg_1w,
               mm_chg_1w            AS cot_mm_chg_1w,
               open_interest        AS cot_open_interest
        FROM cot_data
        WHERE symbol = ? AND date >= ? AND date <= ?
        ORDER BY date
    """, [symbol, start_date, end_date]).df()
    conn.close()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["has_cot_data"] = 1
    return df.sort_values("date").reset_index(drop=True)


def _load_fred(start_date: str, end_date: str) -> pd.DataFrame:
    conn = get_conn()
    # Try to select new columns; fall back gracefully if they don't exist yet
    try:
        df = conn.execute("""
            SELECT date, dxy, inflation_exp, vix, treasury_10y,
                   financial_stress, indpro, fedfunds, china_pmi, copper_basis
            FROM fred_data
            WHERE date >= ? AND date <= ?
            ORDER BY date
        """, [start_date, end_date]).df()
    except Exception:
        df = conn.execute("""
            SELECT date, dxy, inflation_exp, vix, treasury_10y,
                   financial_stress, indpro, fedfunds
            FROM fred_data
            WHERE date >= ? AND date <= ?
            ORDER BY date
        """, [start_date, end_date]).df()
        df["china_pmi"]    = None
        df["copper_basis"] = None
    conn.close()
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)

    for col in df.columns[1:]:
        df[col] = df[col].ffill()

    df["fred_dxy_chg_1w"]  = df["dxy"].diff(5)
    df["fred_dxy_chg_4w"]  = df["dxy"].diff(20)
    df["fred_vix_chg_1w"]  = df["vix"].diff(5)
    df["fred_vix_high"]    = (df["vix"] > 25).astype(float)
    fedfunds_safe          = df["fedfunds"].fillna(0)
    t10y_safe              = df["treasury_10y"].fillna(0)
    df["fred_yield_inv"]   = (t10y_safe < fedfunds_safe).astype(float)
    df["has_fred_data"]    = 1

    return df.rename(columns={
        "dxy":              "fred_dxy",
        "inflation_exp":    "fred_inflation_exp",
        "vix":              "fred_vix",
        "treasury_10y":     "fred_treasury_10y",
        "financial_stress": "fred_financial_stress",
        "indpro":           "fred_indpro",
        "fedfunds":         "fred_fedfunds",
        "china_pmi":        "fred_china_pmi",
        "copper_basis":     "fred_copper_basis",
    })


def _load_eia(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    series_name = EIA_SYMBOL_MAP.get(symbol)
    if not series_name:
        return pd.DataFrame()

    conn = get_conn()
    df = conn.execute("""
        SELECT date, value, chg_1w, vs_5yr_avg
        FROM eia_inventory
        WHERE series = ? AND date >= ? AND date <= ?
        ORDER BY date
    """, [series_name, start_date, end_date]).df()
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.date
    prefix = "eia_crude" if symbol == "CL=F" else "eia_natgas"
    df = df.rename(columns={
        "value":      f"{prefix}_stocks",
        "chg_1w":     f"{prefix}_chg_1w",
        "vs_5yr_avg": f"{prefix}_vs_5yr",
    })
    # Drawdown flag: inventory fell (bullish supply signal)
    chg_col = f"{prefix}_chg_1w"
    df[f"{prefix}_draw"] = (df[chg_col].fillna(0) < -500).astype(float)
    df["has_eia_data"] = 1
    return df.sort_values("date").reset_index(drop=True)


def _load_usda(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    if symbol not in USDA_SYMBOLS:
        return pd.DataFrame()

    conn = get_conn()
    df = conn.execute("""
        SELECT date, metric, value, yoy_chg_pct
        FROM usda_crop
        WHERE commodity = ? AND date >= ? AND date <= ?
        ORDER BY date
    """, [symbol, start_date, end_date]).df()
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Crop condition: sum % good + % excellent per date
    cond = (
        df[df["metric"].str.upper().str.contains("PCT GOOD|PCT EXCELLENT", na=False)]
        .groupby("date")["value"].sum()
        .reset_index()
        .rename(columns={"value": "usda_crop_good_exc"})
        .sort_values("date")
    )
    cond["usda_crop_good_exc_chg"] = cond["usda_crop_good_exc"].diff()

    # Stocks
    stk = (
        df[df["metric"].str.upper().str.contains("STOCKS", na=False)]
        .groupby("date")
        .agg(usda_stocks=("value", "mean"), usda_stocks_yoy=("yoy_chg_pct", "mean"))
        .reset_index()
        .sort_values("date")
    )

    # Annual production (forward-filled across year)
    prd = (
        df[df["metric"].str.upper().str.contains("PRODUCTION", na=False)]
        .groupby("date")
        .agg(usda_production=("value", "mean"))
        .reset_index()
        .sort_values("date")
    )

    parts = [p for p in [cond, stk, prd] if not p.empty]
    if not parts:
        return pd.DataFrame()
    result = parts[0]
    for p in parts[1:]:
        result = result.merge(p, on="date", how="outer")

    result["has_usda_data"] = 1
    return result.sort_values("date").reset_index(drop=True)


def _safe_merge(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Left-merge other onto base by date, zero-fill NaN."""
    if other.empty:
        return base
    merged = base.merge(other, on="date", how="left")
    return merged.fillna(0)


def build_macro_dataframe(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Assemble all macro feature columns for a symbol over a date range.
    Returns a DataFrame keyed on 'date' with one row per calendar day
    that has at least one non-zero macro feature. Missing data → zeros.

    Designed for left-joining onto the price feature matrix in feature_builder.
    """
    cot   = _load_cot(symbol, start_date, end_date)
    fred  = _load_fred(start_date, end_date)
    eia   = _load_eia(symbol, start_date, end_date)
    usda  = _load_usda(symbol, start_date, end_date)

    if all(df.empty for df in [cot, fred, eia, usda]):
        return pd.DataFrame()

    # Use FRED as the date spine (widest coverage); fall back to other sources
    if not fred.empty:
        base = fred[["date"]].copy()
    elif not cot.empty:
        base = cot[["date"]].copy()
    else:
        base = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D").date})

    df = base.copy()
    df = _safe_merge(df, cot)
    df = _safe_merge(df, fred)
    df = _safe_merge(df, eia)
    df = _safe_merge(df, usda)

    # Ensure all expected columns present
    for col in ALL_MACRO_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    # COT and EIA are weekly; forward-fill within the merged frame
    cot_cols  = [c for c in df.columns if c.startswith("cot_")]
    eia_cols  = [c for c in df.columns if c.startswith("eia_")]
    usda_cols = [c for c in df.columns if c.startswith("usda_")]
    for col_group in [cot_cols, eia_cols, usda_cols]:
        df[col_group] = df[col_group].replace(0, float("nan")).ffill().fillna(0)

    # Drop columns that are >95% zero — they have no signal and add noise.
    # This auto-excludes EIA/USDA when no API keys are set.
    feature_cols = [c for c in ALL_MACRO_FEATURES if c in df.columns]
    nonzero_frac = (df[feature_cols].abs() > 0).mean()
    active_cols  = nonzero_frac[nonzero_frac >= 0.05].index.tolist()
    if not active_cols:
        return pd.DataFrame()

    return df[["date"] + active_cols].sort_values("date").reset_index(drop=True)


# ── inference: single-row feature dict ────────────────────────────────────────


def get_macro_features(symbol: str, as_of_date: str = None) -> dict:
    """
    Return a flat dict of all macro features for the given symbol and date.
    Guaranteed to return all keys in ALL_MACRO_FEATURES (zeros for missing data).
    """
    target = as_of_date or date.today().isoformat()
    conn   = get_conn()
    result = dict(_ZERO_ROW)

    # ── COT ──────────────────────────────────────────────────────────────────
    row = conn.execute("""
        SELECT commercial_net_long, commercial_net_pct, mm_net_long, mm_net_pct,
               commercial_chg_1w, mm_chg_1w, open_interest
        FROM cot_data WHERE symbol = ? AND date <= ?
        ORDER BY date DESC LIMIT 1
    """, [symbol, target]).fetchone()

    if row:
        result.update({
            "cot_commercial_net":     row[0] or 0,
            "cot_commercial_net_pct": row[1] or 0,
            "cot_mm_net":             row[2] or 0,
            "cot_mm_net_pct":         row[3] or 0,
            "cot_commercial_chg_1w":  row[4] or 0,
            "cot_mm_chg_1w":          row[5] or 0,
            "cot_open_interest":      row[6] or 0,
            "has_cot_data":           1.0,
        })

    # ── FRED ─────────────────────────────────────────────────────────────────
    try:
        fred_now = conn.execute("""
            SELECT dxy, inflation_exp, vix, treasury_10y, financial_stress,
                   indpro, fedfunds, china_pmi, copper_basis
            FROM fred_data WHERE date <= ? ORDER BY date DESC LIMIT 1
        """, [target]).fetchone()
    except Exception:
        fred_now = conn.execute("""
            SELECT dxy, inflation_exp, vix, treasury_10y, financial_stress,
                   indpro, fedfunds
            FROM fred_data WHERE date <= ? ORDER BY date DESC LIMIT 1
        """, [target]).fetchone()
        fred_now = (fred_now + (None, None)) if fred_now else None

    week_ago = (datetime.strptime(target, "%Y-%m-%d").date() - timedelta(days=7)).isoformat()
    fred_wk   = conn.execute("""
        SELECT dxy, vix FROM fred_data WHERE date <= ? ORDER BY date DESC LIMIT 1
    """, [week_ago]).fetchone()

    month_ago = (datetime.strptime(target, "%Y-%m-%d").date() - timedelta(days=28)).isoformat()
    fred_mo   = conn.execute("""
        SELECT dxy FROM fred_data WHERE date <= ? ORDER BY date DESC LIMIT 1
    """, [month_ago]).fetchone()

    if fred_now:
        dxy   = fred_now[0] or 0
        vix   = fred_now[2] or 0
        t10y  = fred_now[3] or 0
        ff    = fred_now[6] or 0
        dxy_w = (fred_wk[0] or dxy) if fred_wk else dxy
        vix_w = (fred_wk[1] or vix) if fred_wk else vix
        dxy_m = (fred_mo[0] or dxy) if fred_mo else dxy
        result.update({
            "fred_dxy":              dxy,
            "fred_dxy_chg_1w":       dxy - dxy_w,
            "fred_dxy_chg_4w":       dxy - dxy_m,
            "fred_inflation_exp":    fred_now[1] or 0,
            "fred_vix":              vix,
            "fred_vix_chg_1w":       vix - vix_w,
            "fred_vix_high":         float(vix > 25),
            "fred_treasury_10y":     t10y,
            "fred_financial_stress": fred_now[4] or 0,
            "fred_indpro":           fred_now[5] or 0,
            "fred_fedfunds":         ff,
            "fred_yield_inv":        float(t10y < ff),
            "fred_china_pmi":        float(fred_now[7]) if fred_now[7] is not None else 0,
            "fred_copper_basis":     float(fred_now[8]) if fred_now[8] is not None else 0,
            "has_fred_data":         1.0,
        })

    # ── EIA ──────────────────────────────────────────────────────────────────
    series_name = EIA_SYMBOL_MAP.get(symbol)
    prefix      = "eia_crude" if symbol == "CL=F" else "eia_natgas"
    if series_name:
        eia_row = conn.execute("""
            SELECT value, chg_1w, vs_5yr_avg FROM eia_inventory
            WHERE series = ? AND date <= ? ORDER BY date DESC LIMIT 1
        """, [series_name, target]).fetchone()
        if eia_row:
            chg = eia_row[1] or 0
            result.update({
                f"{prefix}_stocks":  eia_row[0] or 0,
                f"{prefix}_chg_1w":  chg,
                f"{prefix}_vs_5yr":  eia_row[2] or 0,
                f"{prefix}_draw":    float(chg < -500),
                "has_eia_data":      1.0,
            })

    # ── USDA ─────────────────────────────────────────────────────────────────
    if symbol in USDA_SYMBOLS:
        latest_date_row = conn.execute("""
            SELECT MAX(date) FROM usda_crop WHERE commodity = ? AND date <= ?
        """, [symbol, target]).fetchone()
        latest = latest_date_row[0] if latest_date_row and latest_date_row[0] else None

        if latest:
            cond_row = conn.execute("""
                SELECT SUM(value) FROM usda_crop
                WHERE commodity = ? AND date = ?
                  AND (UPPER(metric) LIKE '%PCT GOOD%' OR UPPER(metric) LIKE '%PCT EXCELLENT%')
            """, [symbol, latest]).fetchone()

            stk_row = conn.execute("""
                SELECT AVG(value), AVG(yoy_chg_pct) FROM usda_crop
                WHERE commodity = ? AND date = ? AND UPPER(metric) LIKE '%STOCKS%'
            """, [symbol, latest]).fetchone()

            # Previous week for crop condition change
            prev_date = (datetime.strptime(str(latest), "%Y-%m-%d").date() - timedelta(days=7)).isoformat()
            prev_cond = conn.execute("""
                SELECT SUM(value) FROM usda_crop
                WHERE commodity = ? AND date = ?
                  AND (UPPER(metric) LIKE '%PCT GOOD%' OR UPPER(metric) LIKE '%PCT EXCELLENT%')
            """, [symbol, prev_date]).fetchone()

            prod_row = conn.execute("""
                SELECT value FROM usda_crop
                WHERE commodity = ? AND UPPER(metric) LIKE '%PRODUCTION%'
                  AND date <= ?
                ORDER BY date DESC LIMIT 1
            """, [symbol, target]).fetchone()

            crop_now  = float(cond_row[0])  if cond_row and cond_row[0]  else 0
            crop_prev = float(prev_cond[0]) if prev_cond and prev_cond[0] else crop_now
            result.update({
                "usda_crop_good_exc":     crop_now,
                "usda_crop_good_exc_chg": crop_now - crop_prev,
                "usda_stocks":            float(stk_row[0]) if stk_row and stk_row[0] else 0,
                "usda_stocks_yoy":        float(stk_row[1]) if stk_row and stk_row[1] else 0,
                "usda_production":        float(prod_row[0]) if prod_row and prod_row[0] else 0,
                "has_usda_data":          1.0,
            })

    conn.close()
    return result

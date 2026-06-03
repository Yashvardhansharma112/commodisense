"""
Weather Features — thin wrapper that surfaces weather_features table data
as commodity-specific signals for the feature builder.

The heavy lifting (fetching + engineering drought_index etc.) is done in
data/collector_weather.py. This module just shapes the data for ML consumption.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.db import get_conn

# Which regions matter most per commodity
COMMODITY_REGIONS: dict[str, list[str]] = {
    "CL=F":    ["middle_east_oil", "texas_energy"],
    "NG=F":    ["texas_energy", "middle_east_oil"],
    "GC=F":    ["south_africa_gold"],
    "ZW=F":    ["black_sea_ukraine"],
    "ZC=F":    ["us_corn_belt", "black_sea_ukraine"],
    "ZS=F":    ["us_corn_belt", "brazil_soy"],
    "CT=F":    ["india_monsoon", "brazil_soy"],
    "SB=F":    ["india_monsoon", "brazil_soy"],
    "USDINR=X":["india_monsoon"],
    "HG=F":    ["chile_copper"],
}


def get_weather_features(commodity: str, days: int = 90) -> dict:
    """
    Return the latest aggregated weather signals for a commodity.

    Averages drought_index, heat_stress_days, and precip_anomaly_pct across
    the commodity's primary regions over the last 30 days.

    Args:
        commodity: Ticker symbol, e.g. "ZW=F"
        days:      Look-back window in calendar days (used for region filter)

    Returns:
        Dict with keys: drought_index, heat_stress_days, precip_anomaly_pct.
        Returns zeros if no data found.
    """
    regions = COMMODITY_REGIONS.get(commodity, [])
    if not regions:
        return {"drought_index": 0.0, "heat_stress_days": 0, "precip_anomaly_pct": 0.0}

    cutoff = date.today() - timedelta(days=30)  # use last 30 days for signal
    placeholders = ",".join(["?"] * len(regions))

    conn = get_conn()
    df = conn.execute(
        f"""
        SELECT drought_index, heat_stress_days, precip_anomaly_pct
        FROM weather_features
        WHERE commodity = ?
          AND region IN ({placeholders})
          AND date >= ?
        """,
        [commodity] + regions + [cutoff],
    ).df()
    conn.close()

    if df.empty:
        return {"drought_index": 0.0, "heat_stress_days": 0, "precip_anomaly_pct": 0.0}

    return {
        "drought_index":      round(float(df["drought_index"].mean()), 4),
        "heat_stress_days":   int(df["heat_stress_days"].mean()),
        "precip_anomaly_pct": round(float(df["precip_anomaly_pct"].mean()), 2),
    }


def get_weather_dataframe(commodity: str, days: int = 90) -> pd.DataFrame:
    """
    Return time-series weather data for a commodity (all relevant regions).
    Used by the feature builder to join weather signals into the training matrix.
    """
    regions = COMMODITY_REGIONS.get(commodity, [])
    if not regions:
        return pd.DataFrame()

    cutoff = date.today() - timedelta(days=days)
    placeholders = ",".join(["?"] * len(regions))

    conn = get_conn()
    df = conn.execute(
        f"""
        SELECT date, region,
               drought_index, heat_stress_days, precip_anomaly_pct
        FROM weather_features
        WHERE commodity = ?
          AND region IN ({placeholders})
          AND date >= ?
        ORDER BY date
        """,
        [commodity] + regions + [cutoff],
    ).df()
    conn.close()

    if df.empty:
        return df

    # Average across regions per date
    return (
        df.groupby("date")
        .agg(
            drought_index=("drought_index", "mean"),
            heat_stress_days=("heat_stress_days", "mean"),
            precip_anomaly_pct=("precip_anomaly_pct", "mean"),
        )
        .reset_index()
        .sort_values("date")
    )

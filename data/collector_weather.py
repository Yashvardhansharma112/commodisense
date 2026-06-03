"""
Weather Collector — fetches daily weather for 8 commodity-critical regions
via Open-Meteo (free, no API key) and stores engineered features in DuckDB
(table: weather_features).

Usage:
    python data/collector_weather.py            # incremental
    python data/collector_weather.py --backfill # 2-year history
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.db import get_conn, init_schema

# ── constants ──────────────────────────────────────────────────────────────────

REGIONS: dict[str, dict] = {
    "black_sea_ukraine": {
        "lat": 49.0, "lon": 32.0,
        "commodities": ["ZW=F", "ZC=F"],
    },
    "us_corn_belt": {
        "lat": 41.5, "lon": -93.6,
        "commodities": ["ZC=F", "ZS=F"],
    },
    "brazil_soy": {
        "lat": -15.0, "lon": -47.9,
        "commodities": ["ZS=F", "CT=F"],
    },
    "india_monsoon": {
        "lat": 20.6, "lon": 78.9,
        "commodities": ["CT=F", "SB=F"],
    },
    "middle_east_oil": {
        "lat": 26.0, "lon": 50.5,
        "commodities": ["CL=F", "NG=F"],
    },
    "texas_energy": {
        "lat": 31.0, "lon": -99.0,
        "commodities": ["CL=F", "NG=F"],
    },
    "south_africa_gold": {
        "lat": -26.2, "lon": 28.0,
        "commodities": ["GC=F"],
    },
    "chile_copper": {
        "lat": -33.4, "lon": -70.6,
        "commodities": ["HG=F"],
    },
}

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    # soil_moisture omitted: not available at daily resolution in archive API
]

BACKFILL_YEARS = 2
LOG_PATH = Path(__file__).parent / "logs" / "weather.log"
HEAT_STRESS_THRESHOLD_C = 35.0

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

# ── Open-Meteo client with caching + auto-retry ────────────────────────────────

_cache_session = requests_cache.CachedSession(".weather_cache", expire_after=3600)
_retry_session = retry(_cache_session, retries=3, backoff_factor=0.5)
_om_client = openmeteo_requests.Client(session=_retry_session)

# ── helpers ────────────────────────────────────────────────────────────────────


def _last_stored_date(region: str, commodity: str) -> date | None:
    """Return the most recent date stored for a given region+commodity pair."""
    conn = get_conn()
    row = conn.execute(
        "SELECT MAX(date) FROM weather_features WHERE region = ? AND commodity = ?",
        [region, commodity],
    ).fetchone()
    conn.close()
    return row[0] if row and row[0] else None


def _fetch_open_meteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily weather from Open-Meteo for a lat/lon over a date range.

    Returns DataFrame indexed by date with columns matching DAILY_VARS.
    Returns empty DataFrame on error.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": DAILY_VARS,
        "timezone": "UTC",
    }

    # Use forecast endpoint for recent/future dates, archive for historical
    today = date.today().isoformat()
    if end >= today:
        forecast_url = "https://api.open-meteo.com/v1/forecast"
        try:
            responses = _om_client.weather_api(forecast_url, params=params)
        except Exception:
            responses = _om_client.weather_api(url, params=params)
    else:
        responses = _om_client.weather_api(url, params=params)

    resp = responses[0]
    daily = resp.Daily()

    records: dict[str, list] = {"date": []}
    # Build date range from API response time bounds
    dates = pd.date_range(
        start=pd.Timestamp(daily.Time(), unit="s", tz="UTC"),
        end=pd.Timestamp(daily.TimeEnd(), unit="s", tz="UTC"),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left",
    ).date.tolist()
    records["date"] = dates

    for i, var in enumerate(DAILY_VARS):
        records[var] = daily.Variables(i).ValuesAsNumpy().tolist()

    return pd.DataFrame(records)


def _compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add drought_index, heat_stress_days, and precip_anomaly_pct columns.

    drought_index:       rolling 30-day precip deficit vs 5-yr mean
                         (here we use available history as proxy for 5-yr mean)
    heat_stress_days:    count of days with temp_max > 35°C in last 30 rows
    precip_anomaly_pct:  (current_30d_avg - historical_avg) / historical_avg * 100
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    precip = df["precipitation_sum"].fillna(0)
    temp_max = df["temperature_2m_max"].fillna(0)

    # Rolling 30-day precipitation sum
    rolling_precip = precip.rolling(30, min_periods=1).sum()
    # Historical average precipitation (using all available data as proxy)
    hist_avg_precip = precip.mean() * 30  # scale to 30-day window
    hist_avg_precip = max(hist_avg_precip, 0.01)  # avoid divide-by-zero

    df["drought_index"] = ((hist_avg_precip - rolling_precip) / hist_avg_precip).clip(0, 1)

    # Heat stress: days above threshold in last 30-row window
    df["heat_stress_days"] = (
        (temp_max > HEAT_STRESS_THRESHOLD_C)
        .rolling(30, min_periods=1)
        .sum()
        .astype(int)
    )

    # Precipitation anomaly: rolling 30d vs historical average
    df["precip_anomaly_pct"] = (
        (rolling_precip - hist_avg_precip) / hist_avg_precip * 100
    ).round(1)

    return df


def _upsert_weather(df: pd.DataFrame, region: str, commodity: str) -> int:
    """Upsert weather feature rows into DuckDB. Returns count of new rows."""
    if df.empty:
        return 0

    conn = get_conn()
    inserted = 0
    for _, row in df.iterrows():
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO weather_features
                    (date, region, commodity,
                     temp_max, temp_min, precipitation, soil_moisture,
                     drought_index, heat_stress_days, precip_anomaly_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    row["date"],
                    region,
                    commodity,
                    float(row.get("temperature_2m_max") or 0),
                    float(row.get("temperature_2m_min") or 0),
                    float(row.get("precipitation_sum") or 0),
                    0.0,  # soil_moisture not available from archive API
                    float(row.get("drought_index") or 0),
                    int(row.get("heat_stress_days") or 0),
                    float(row.get("precip_anomaly_pct") or 0),
                ],
            )
            inserted += 1
        except Exception as exc:
            log.debug("Weather insert error %s/%s/%s: %s", region, commodity, row["date"], exc)
    conn.close()
    return inserted


# ── public API ─────────────────────────────────────────────────────────────────


def get_weather_signal(commodity: str, days: int = 90) -> pd.DataFrame:
    """
    Return weather features for all regions relevant to a commodity.

    Args:
        commodity: Ticker symbol, e.g. "ZW=F"
        days:      Look-back window in calendar days

    Returns:
        DataFrame with columns: date, region, drought_index, heat_stress_days,
        precip_anomaly_pct (and raw weather columns).
    """
    cutoff = date.today() - timedelta(days=days)
    conn = get_conn()
    df = conn.execute(
        """
        SELECT * FROM weather_features
        WHERE commodity = ? AND date >= ?
        ORDER BY region, date
        """,
        [commodity, cutoff],
    ).df()
    conn.close()
    return df


# ── main run ───────────────────────────────────────────────────────────────────


def run(backfill: bool = False) -> None:
    """
    Collect weather data for all regions and commodities.
    Backfill=True fetches 2 years of history.
    """
    init_schema()
    total_inserted = 0
    end_date = date.today().isoformat()

    for region, cfg in REGIONS.items():
        lat, lon = cfg["lat"], cfg["lon"]
        for commodity in cfg["commodities"]:
            if backfill:
                start_date = (
                    date.today() - timedelta(days=365 * BACKFILL_YEARS)
                ).isoformat()
            else:
                last = _last_stored_date(region, commodity)
                start_date = (
                    (last + timedelta(days=1)).isoformat()
                    if last
                    else (date.today() - timedelta(days=14)).isoformat()
                )

            log.info("Weather %s/%s: %s -> %s", region, commodity, start_date, end_date)
            try:
                df_raw = _fetch_open_meteo(lat, lon, start_date, end_date)
            except Exception as exc:
                log.error("Failed to fetch %s: %s", region, exc)
                continue

            df_eng = _compute_engineered_features(df_raw)
            n = _upsert_weather(df_eng, region, commodity)
            log.info("  inserted %d rows", n)
            total_inserted += n

    print(f"Weather: inserted {total_inserted} rows across {len(REGIONS)} regions")
    log.info("Weather run complete: %d rows", total_inserted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CommodiSense weather collector")
    parser.add_argument("--backfill", action="store_true", help="Fetch 2-year history")
    args = parser.parse_args()
    run(backfill=args.backfill)

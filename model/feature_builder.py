"""
Feature Builder — assembles all signals (price, sentiment, events, weather,
geopolitical) into a single feature matrix per commodity.

CRITICAL: zero lookahead. All signal windows use T-1 to T-N only.
Target variable uses T+7 and T+30 prices (shifted forward, excluded from features).

Usage:
    from model.feature_builder import build_training_data, build_prediction_features
"""

import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.db import get_conn
from signals.price_features import build_feature_matrix, ALL_SYMBOLS
from signals.weather_features import get_weather_dataframe

log = logging.getLogger(__name__)

# Per-commodity direction thresholds — calibrated to each asset's typical volatility.
# USDINR is a managed float (rarely moves ±2% in 7 days → extreme STABLE imbalance).
# NG=F is highly volatile → needs wider threshold to avoid noise.
DIRECTION_THRESHOLDS: dict[str, float] = {
    "CL=F":     2.0,
    "NG=F":     3.5,
    "GC=F":     1.5,
    "ZW=F":     2.0,
    "ZC=F":     2.0,
    "ZS=F":     2.0,
    "CT=F":     2.0,
    "SB=F":     2.0,
    "USDINR=X": 0.4,
    "HG=F":     2.0,
}
DIRECTION_THRESHOLD_PCT = 2.0  # fallback

# ── helpers ────────────────────────────────────────────────────────────────────


def _load_prices_for_target(symbol: str) -> pd.DataFrame:
    """Load close prices with enough future rows to compute T+7 and T+30 targets."""
    conn = get_conn()
    df = conn.execute(
        "SELECT date, close FROM prices WHERE symbol = ? ORDER BY date",
        [symbol],
    ).df()
    conn.close()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)


def _compute_targets(price_df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """
    Compute direction_7d and direction_30d target columns.

    Labels:
         1  (UP)     if future price > current * 1.02
         0  (STABLE) if within ±2%
        -1  (DOWN)   if future price < current * 0.98
    """
    df = price_df.copy().sort_values("date").reset_index(drop=True)
    closes = df["close"].values
    threshold = DIRECTION_THRESHOLDS.get(symbol, DIRECTION_THRESHOLD_PCT) if symbol else DIRECTION_THRESHOLD_PCT

    def _direction(current: float, future: float) -> int:
        if future == 0 or current == 0:
            return 0
        chg = (future - current) / current * 100
        if chg > threshold:
            return 1
        if chg < -threshold:
            return -1
        return 0

    dir_7d, dir_30d = [], []
    n = len(closes)
    for i in range(n):
        # Find the index approximately 7 / 30 trading days forward
        # Use calendar-day shifted date to find the nearest actual price row
        fwd7  = df[df["date"] >= (df.at[i, "date"] + timedelta(days=7))].head(1)
        fwd30 = df[df["date"] >= (df.at[i, "date"] + timedelta(days=30))].head(1)

        dir_7d.append(
            _direction(closes[i], float(fwd7["close"].values[0])) if not fwd7.empty else None
        )
        dir_30d.append(
            _direction(closes[i], float(fwd30["close"].values[0])) if not fwd30.empty else None
        )

    df["direction_7d"]  = dir_7d
    df["direction_30d"] = dir_30d
    return df


def _load_sentiment_series(symbol: str) -> pd.DataFrame:
    """Load daily sentiment aggregates for a commodity from DuckDB."""
    conn = get_conn()
    df = conn.execute(
        """
        SELECT date, sentiment_score, article_count, positive_count
        FROM sentiment_daily
        WHERE commodity = ?
        ORDER BY date
        """,
        [symbol],
    ).df()
    conn.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    # Rolling aggregates
    df["sentiment_3d"]       = df["sentiment_score"].rolling(3, min_periods=1).mean()
    df["sentiment_7d"]       = df["sentiment_score"].rolling(7, min_periods=1).mean()
    df["article_count_7d"]   = df["article_count"].rolling(7, min_periods=1).sum()
    df["positive_ratio_7d"]  = (
        df["positive_count"].rolling(7, min_periods=1).sum()
        / df["article_count_7d"].replace(0, 1)
    )
    return df.rename(columns={"sentiment_score": "sentiment_score_1d"})


def _load_event_series(symbol: str) -> pd.DataFrame:
    """Load daily event aggregates for a commodity from DuckDB."""
    conn = get_conn()
    df = conn.execute(
        """
        SELECT date, event_type, direction, severity
        FROM extracted_events
        WHERE commodity = ?
        ORDER BY date
        """,
        [symbol],
    ).df()
    conn.close()
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["dir_score"] = df["direction"].map({"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}).fillna(0)

    agg = df.groupby("date").agg(
        bullish_events_7d  =("direction", lambda x: int((x == "BULLISH").sum())),
        bearish_events_7d  =("direction", lambda x: int((x == "BEARISH").sum())),
        max_severity_7d    =("severity",  "max"),
        direction_score_7d =("dir_score", "sum"),
        supply_shock_flag  =("event_type", lambda x: int((x == "SUPPLY_SHOCK").any())),
        policy_change_flag =("event_type", lambda x: int((x == "POLICY_CHANGE").any())),
    ).reset_index()

    # Rolling 7-day window for event counts
    agg = agg.sort_values("date").reset_index(drop=True)
    for col in ["bullish_events_7d", "bearish_events_7d", "direction_score_7d"]:
        agg[col] = agg[col].rolling(7, min_periods=1).sum()
    return agg


def _load_geo_series(symbol: str) -> pd.DataFrame:
    """Load rolling geopolitical risk scores for a commodity."""
    conn = get_conn()
    df = conn.execute(
        "SELECT date, risk_score FROM geopolitical_events WHERE commodity = ? ORDER BY date",
        [symbol],
    ).df()
    conn.close()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    agg = df.groupby("date")["risk_score"].mean().reset_index()
    agg = agg.sort_values("date").reset_index(drop=True)
    agg["risk_score_7d"]  = agg["risk_score"].rolling(7,  min_periods=1).mean()
    agg["risk_score_30d"] = agg["risk_score"].rolling(30, min_periods=1).mean()
    return agg[["date", "risk_score_7d", "risk_score_30d"]]


def _safe_merge(base: pd.DataFrame, other: pd.DataFrame, on: str = "date") -> pd.DataFrame:
    """Left-join `other` onto `base`, filling NaN with 0."""
    if other.empty:
        return base
    merged = base.merge(other, on=on, how="left")
    merged = merged.fillna(0)
    return merged


# ── public API ─────────────────────────────────────────────────────────────────


def build_training_data(
    symbol: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Assemble the full feature matrix + targets for a commodity.

    Uses all available history in DuckDB. No lookahead: signal features
    reflect data known at close of each trading day.

    Args:
        symbol: Commodity ticker, e.g. "ZW=F"

    Returns:
        (X, y_7d, y_30d) where:
          X      — DataFrame, one row per date, all feature columns
          y_7d   — Series of direction labels {-1, 0, 1} for 7-day horizon
          y_30d  — Series of direction labels {-1, 0, 1} for 30-day horizon
    """
    # Price features (covers ~5yr history)
    end_date   = date.today().isoformat()
    start_date = (date.today() - timedelta(days=365 * 5)).isoformat()
    price_feat = build_feature_matrix(symbol, start_date, end_date)
    if price_feat.empty:
        log.warning("%s: no price features available", symbol)
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=int)

    # Targets — computed from raw close prices with per-commodity threshold
    prices = _load_prices_for_target(symbol)
    targets = _compute_targets(prices, symbol=symbol)[["date", "direction_7d", "direction_30d"]]

    # All signal series
    sentiment = _load_sentiment_series(symbol)
    events    = _load_event_series(symbol)
    geo       = _load_geo_series(symbol)
    weather   = get_weather_dataframe(symbol, days=365 * 5)
    if not weather.empty:
        weather["date"] = pd.to_datetime(weather["date"]).dt.date

    # Merge everything onto price_feat (left join → zero-fill missing signal days)
    df = price_feat.copy()
    df = _safe_merge(df, targets,   on="date")
    df = _safe_merge(df, sentiment[["date", "sentiment_score_1d", "sentiment_3d",
                                     "sentiment_7d", "article_count_7d",
                                     "positive_ratio_7d"]] if not sentiment.empty else pd.DataFrame(),
                    on="date")
    df = _safe_merge(df, events,  on="date")
    df = _safe_merge(df, geo,     on="date")
    df = _safe_merge(df, weather, on="date")

    # Add binary indicator: 1 on days where we have real news signal, 0 elsewhere.
    # This lets the model learn "trust sentiment when has_news_signal=1" rather than
    # treating zero-padded sentiment rows as neutral-sentiment days.
    if "sentiment_score_1d" in df.columns:
        df["has_news_signal"] = (df["sentiment_score_1d"].abs() > 0.01).astype(int)
    else:
        df["has_news_signal"] = 0

    # Drop rows where targets are unavailable (last 30 days have no T+30 target)
    df = df.dropna(subset=["direction_7d", "direction_30d"])
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in
                    ("date", "direction_7d", "direction_30d")]

    X     = df[feature_cols].fillna(0).astype(float)
    y_7d  = df["direction_7d"].astype(int)
    y_30d = df["direction_30d"].astype(int)

    log.info("%s: training data shape %s, class dist 7d: %s",
             symbol, X.shape, y_7d.value_counts().to_dict())
    return X, y_7d, y_30d


def build_prediction_features(symbol: str, as_of_date: str = None) -> pd.Series:
    """
    Build a single-row feature vector for inference.

    Uses only data available up to (and including) as_of_date.
    No future data touches this vector.

    Args:
        symbol:      Commodity ticker
        as_of_date:  ISO date string. Defaults to today.

    Returns:
        pd.Series with the same feature names as build_training_data returns.
    """
    from signals.price_features import get_price_features
    from signals.weather_features import get_weather_features

    target_date = as_of_date or date.today().isoformat()

    # Price features (T-1 based internally)
    price_f = get_price_features(symbol, target_date)

    # Sentiment: last 7 days before target_date
    cutoff = (datetime.strptime(target_date, "%Y-%m-%d").date() - timedelta(days=7)).isoformat()
    conn = get_conn()
    sent_rows = conn.execute(
        """
        SELECT date, sentiment_score, article_count, positive_count
        FROM sentiment_daily
        WHERE commodity = ? AND date >= ? AND date <= ?
        ORDER BY date DESC
        """,
        [symbol, cutoff, target_date],
    ).df()
    conn.close()

    sentiment_1d = float(sent_rows.iloc[0]["sentiment_score"]) if not sent_rows.empty else 0.0
    sentiment_3d = float(sent_rows.head(3)["sentiment_score"].mean()) if len(sent_rows) >= 1 else 0.0
    sentiment_7d = float(sent_rows["sentiment_score"].mean()) if not sent_rows.empty else 0.0
    article_count_7d = int(sent_rows["article_count"].sum()) if not sent_rows.empty else 0
    positive_ratio_7d = (
        float(sent_rows["positive_count"].sum() / max(article_count_7d, 1))
        if not sent_rows.empty else 0.0
    )

    # Events: last 7 days
    conn = get_conn()
    evt_rows = conn.execute(
        """
        SELECT event_type, direction, severity
        FROM extracted_events
        WHERE commodity = ? AND date >= ? AND date <= ?
        """,
        [symbol, cutoff, target_date],
    ).df()
    conn.close()

    bullish_events_7d  = int((evt_rows["direction"] == "BULLISH").sum()) if not evt_rows.empty else 0
    bearish_events_7d  = int((evt_rows["direction"] == "BEARISH").sum()) if not evt_rows.empty else 0
    max_severity_7d    = int(evt_rows["severity"].max()) if not evt_rows.empty else 0
    dir_map = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}
    direction_score_7d = int(evt_rows["direction"].map(dir_map).fillna(0).sum()) if not evt_rows.empty else 0
    supply_shock_flag  = int((evt_rows["event_type"] == "SUPPLY_SHOCK").any()) if not evt_rows.empty else 0
    policy_change_flag = int((evt_rows["event_type"] == "POLICY_CHANGE").any()) if not evt_rows.empty else 0

    # Geopolitical risk
    cutoff_30 = (datetime.strptime(target_date, "%Y-%m-%d").date() - timedelta(days=30)).isoformat()
    conn = get_conn()
    geo_rows = conn.execute(
        "SELECT risk_score FROM geopolitical_events WHERE commodity = ? AND date >= ? AND date <= ?",
        [symbol, cutoff_30, target_date],
    ).df()
    conn.close()
    risk_score_7d  = float(geo_rows.tail(7)["risk_score"].mean())  if not geo_rows.empty else 0.05
    risk_score_30d = float(geo_rows["risk_score"].mean())           if not geo_rows.empty else 0.05

    # Weather
    weather_f = get_weather_features(symbol, days=90)

    features = {
        **price_f,
        "sentiment_score_1d":  sentiment_1d,
        "sentiment_3d":        sentiment_3d,
        "sentiment_7d":        sentiment_7d,
        "article_count_7d":    article_count_7d,
        "positive_ratio_7d":   positive_ratio_7d,
        "bullish_events_7d":   bullish_events_7d,
        "bearish_events_7d":   bearish_events_7d,
        "max_severity_7d":     max_severity_7d,
        "direction_score_7d":  direction_score_7d,
        "supply_shock_flag":   supply_shock_flag,
        "policy_change_flag":  policy_change_flag,
        "risk_score_7d":       risk_score_7d,
        "risk_score_30d":      risk_score_30d,
        **weather_f,
    }

    return pd.Series(features)

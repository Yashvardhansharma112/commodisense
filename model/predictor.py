"""
Predictor — loads saved XGBoost + LightGBM models and generates forecasts
at inference time. Runs entirely on CPU.

Usage:
    python model/predictor.py --symbol ZW=F
    python model/predictor.py --all
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.feature_builder import build_prediction_features
from data.db import get_conn

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"

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

# Human-readable labels for SHAP feature display
FEATURE_LABELS: dict[str, str] = {
    "rsi_14":              "RSI (14-day)",
    "macd_signal":         "MACD crossover",
    "bb_position":         "Bollinger Band position",
    "atr_14":              "Average True Range",
    "atr_pct":             "Volatility %",
    "sma_20_50_cross":     "SMA 20/50 crossover",
    "return_1d":           "1-day return %",
    "return_7d":           "7-day return %",
    "return_30d":          "30-day return %",
    "momentum_score":      "Momentum score",
    "month_sin":           "Seasonal cycle (sin)",
    "month_cos":           "Seasonal cycle (cos)",
    "harvest_season_flag": "Harvest season",
    "days_to_opec_meeting":"Days to OPEC meeting",
    "oil_gold_ratio":      "Oil/Gold ratio",
    "dxy_proxy":           "USD strength proxy",
    "sentiment_score_1d":  "News sentiment (24h)",
    "sentiment_3d":        "News sentiment (3-day)",
    "sentiment_7d":        "News sentiment (7-day)",
    "article_count_7d":    "Article volume (7-day)",
    "positive_ratio_7d":   "Positive news ratio",
    "bullish_events_7d":   "Bullish events (7-day)",
    "bearish_events_7d":   "Bearish events (7-day)",
    "max_severity_7d":     "Max event severity",
    "direction_score_7d":  "Net event direction",
    "supply_shock_flag":   "Supply shock detected",
    "policy_change_flag":  "Policy change detected",
    "risk_score_7d":       "Geopolitical risk (7-day)",
    "risk_score_30d":      "Geopolitical risk (30-day)",
    "drought_index":       "Drought index",
    "heat_stress_days":    "Heat stress days",
    "precip_anomaly_pct":  "Precipitation anomaly %",
}

# Expected return by predicted direction (base, adjusted per-commodity)
DIRECTION_EXPECTED_RETURN: dict[str, float] = {
    "UP":     3.0,
    "STABLE": 0.0,
    "DOWN":  -3.0,
}

# ── model cache (loaded once per process) ─────────────────────────────────────

_model_cache: dict[str, dict] = {}


def _load_models(symbol: str, horizon: str = "7d") -> dict | None:
    """
    Load XGBoost, LightGBM, scaler, and feature names for a symbol.
    Caches in memory for the process lifetime.

    Returns None if models not found (not trained yet).
    """
    cache_key = f"{symbol}_{horizon}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    xgb_path    = MODELS_DIR / f"xgb_{symbol}_{horizon}.pkl"
    lgbm_path   = MODELS_DIR / f"lgbm_{symbol}_{horizon}.pkl"
    scaler_path = MODELS_DIR / f"scaler_{symbol}_{horizon}.pkl"
    feat_path   = MODELS_DIR / f"feature_names_{symbol}_{horizon}.json"

    if not all(p.exists() for p in [xgb_path, lgbm_path, scaler_path, feat_path]):
        log.warning("Models not found for %s %s — run model/trainer.py first", symbol, horizon)
        return None

    with open(xgb_path, "rb") as f:
        xgb_model = pickle.load(f)
    with open(lgbm_path, "rb") as f:
        lgbm_model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(feat_path) as f:
        feature_names = json.load(f)

    bundle = {
        "xgb":      xgb_model,
        "lgbm":     lgbm_model,
        "scaler":   scaler,
        "features": feature_names,
    }
    _model_cache[cache_key] = bundle
    return bundle


def _get_shap_top5(xgb_model, X_row: np.ndarray, feature_names: list[str], pred_class: int) -> list[dict]:
    """
    Compute SHAP values for XGBoost and return top 5 features by |shap_value|
    for the predicted class.
    """
    try:
        import shap
        explainer  = shap.TreeExplainer(xgb_model)
        shap_vals  = explainer.shap_values(X_row)  # shape: (n_classes, n_features) or (1, n_classes, n_features)

        # shap_values shape varies by XGBoost version
        if isinstance(shap_vals, list):
            vals = shap_vals[pred_class][0]  # for predicted class
        else:
            vals = shap_vals[0, :, pred_class] if shap_vals.ndim == 3 else shap_vals[0]

        top_idx = np.argsort(np.abs(vals))[::-1][:5]
        result = []
        for i in top_idx:
            fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            fval  = float(X_row[0][i])
            shap_v = float(vals[i])
            result.append({
                "feature": fname,
                "label":   FEATURE_LABELS.get(fname, fname),
                "value":   round(fval, 4),
                "impact":  "BULLISH" if shap_v > 0 else "BEARISH",
                "weight":  round(abs(shap_v), 4),
            })
        return result

    except Exception as exc:
        log.debug("SHAP error: %s", exc)
        return []


def _get_current_price(symbol: str) -> tuple[float, float]:
    """Return (current_close, atr_pct) from latest DB row."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT close FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT 2",
        [symbol],
    ).fetchall()
    conn.close()
    if not rows:
        return 0.0, 0.02
    close = float(rows[0][0])
    # Rough ATR proxy: |today - yesterday| / today
    atr_pct = abs(float(rows[0][0]) - float(rows[1][0])) / close if len(rows) > 1 and close > 0 else 0.02
    return close, atr_pct


# ── public API ─────────────────────────────────────────────────────────────────


def predict(symbol: str, as_of_date: str = None) -> dict:
    """
    Generate a forecast for a single commodity.

    Args:
        symbol:      Commodity ticker, e.g. "ZW=F"
        as_of_date:  ISO date string. Defaults to today.

    Returns:
        Forecast dict with symbol, current price, 7d + 30d forecasts,
        top_signals, and confidence levels. Returns error dict if models
        are not trained.
    """
    as_of = as_of_date or date.today().isoformat()

    bundle_7d  = _load_models(symbol, "7d")
    bundle_30d = _load_models(symbol, "30d")

    if bundle_7d is None:
        return {"symbol": symbol, "error": "models_not_trained", "as_of_date": as_of}

    # Build feature vector
    features_series = build_prediction_features(symbol, as_of)
    if features_series.empty:
        return {"symbol": symbol, "error": "no_features", "as_of_date": as_of}

    # Align to trained feature names
    feat_names_7d = bundle_7d["features"]
    X_raw = features_series.reindex(feat_names_7d, fill_value=0).values.reshape(1, -1)
    X_scaled_7d = bundle_7d["scaler"].transform(pd.DataFrame(X_raw, columns=feat_names_7d))

    # Ensemble prediction — 7d
    X_df_7d = pd.DataFrame(X_scaled_7d, columns=feat_names_7d)
    xgb_proba_7d  = bundle_7d["xgb"].predict_proba(X_df_7d)[0]
    lgbm_proba_7d = bundle_7d["lgbm"].predict_proba(X_df_7d)[0]
    ensemble_proba_7d = (xgb_proba_7d + lgbm_proba_7d) / 2
    pred_class_7d = int(ensemble_proba_7d.argmax())
    # Map encoded class back: 0=DOWN, 1=STABLE, 2=UP
    direction_map = {0: "DOWN", 1: "STABLE", 2: "UP"}
    direction_7d  = direction_map[pred_class_7d]
    prob_7d       = float(ensemble_proba_7d[pred_class_7d])

    # Ensemble prediction — 30d (may not be trained)
    direction_30d, prob_30d = "STABLE", 0.5
    if bundle_30d:
        feat_names_30d = bundle_30d["features"]
        X_raw_30d = features_series.reindex(feat_names_30d, fill_value=0).values.reshape(1, -1)
        X_scaled_30d = bundle_30d["scaler"].transform(pd.DataFrame(X_raw_30d, columns=feat_names_30d))
        X_df_30d = pd.DataFrame(X_scaled_30d, columns=feat_names_30d)
        xgb_proba_30d  = bundle_30d["xgb"].predict_proba(X_df_30d)[0]
        lgbm_proba_30d = bundle_30d["lgbm"].predict_proba(X_df_30d)[0]
        ensemble_proba_30d = (xgb_proba_30d + lgbm_proba_30d) / 2
        pred_class_30d = int(ensemble_proba_30d.argmax())
        direction_30d  = direction_map[pred_class_30d]
        prob_30d       = float(ensemble_proba_30d[pred_class_30d])

    # Confidence tier — tuned thresholds (validated via 3.5yr walk-forward backtest)
    # HIGH fires ~9% of time with ~74% accuracy vs 45% overall
    def _confidence(prob: float) -> str:
        if prob >= 0.55:
            return "HIGH"
        if prob >= 0.45:
            return "MEDIUM"
        return "LOW"

    # Signal confirmation: require 2+ independent signals to issue HIGH confidence.
    def _confirmed_confidence(prob: float, direction: str, feat: pd.Series) -> str:
        base = _confidence(prob)
        if base == "LOW":
            return "LOW"
        confirming = 0
        # Signal 1: price momentum agrees
        mom = float(feat.get("momentum_score", 0) or 0)
        ret7 = float(feat.get("return_7d", 0) or 0)
        if direction == "UP"   and (mom > 0 or ret7 > 0):  confirming += 1
        if direction == "DOWN" and (mom < 0 or ret7 < 0):  confirming += 1
        # Signal 2: COT commercial positioning agrees (commercials = smart money)
        cot_net = float(feat.get("cot_commercial_net_pct", 0) or 0)
        cot_chg = float(feat.get("cot_commercial_chg_1w", 0) or 0)
        if direction == "UP"   and (cot_net > 0.05 or cot_chg > 0):  confirming += 1
        if direction == "DOWN" and (cot_net < -0.05 or cot_chg < 0): confirming += 1
        # Signal 3: EIA supply signal agrees (for CL=F and NG=F)
        eia_draw  = float(feat.get("eia_crude_draw", 0) or feat.get("eia_natgas_draw", 0) or 0)
        eia_vs5yr = float(feat.get("eia_crude_vs_5yr", 0) or feat.get("eia_natgas_vs_5yr", 0) or 0)
        if direction == "UP"   and (eia_draw > 0 or eia_vs5yr < -0.5): confirming += 1
        if direction == "DOWN" and eia_vs5yr > 0.5:                     confirming += 1
        # Signal 4: USDA crop condition trend agrees (for grain/ag symbols)
        crop_chg = float(feat.get("usda_crop_good_exc_chg", 0) or 0)
        if direction == "DOWN" and crop_chg < -2:  confirming += 1
        if direction == "UP"   and crop_chg >  2:  confirming += 1
        # 2+ signals → HIGH; 0 signals → LOW; 1 signal → MEDIUM
        if confirming >= 2:
            return "HIGH"
        if confirming == 0:
            return "LOW"
        return "MEDIUM"

    # Price range using ATR
    current_price, atr_pct = _get_current_price(symbol)
    exp_ret = DIRECTION_EXPECTED_RETURN.get(direction_7d, 0.0) / 100
    price_range_low  = round(current_price * (1 + exp_ret - 1.5 * atr_pct), 2)
    price_range_high = round(current_price * (1 + exp_ret + 1.5 * atr_pct), 2)

    # SHAP top signals
    top_signals = _get_shap_top5(bundle_7d["xgb"], X_scaled_7d, feat_names_7d, pred_class_7d)

    conf_7d  = _confirmed_confidence(prob_7d,  direction_7d,  features_series)
    conf_30d = _confirmed_confidence(prob_30d, direction_30d, features_series)

    # Symbols where 7d model has known accuracy issues — surface a warning.
    UNRELIABLE_7D = {"ZC=F", "HG=F"}
    model_warning = (
        "7d model accuracy is low for this symbol — use 30d forecast instead"
        if symbol in UNRELIABLE_7D else None
    )

    return {
        "symbol":           symbol,
        "commodity_name":   SYMBOL_NAMES.get(symbol, symbol),
        "as_of_date":       as_of,
        "current_price":    current_price,
        "forecast_7d": {
            "direction":        direction_7d,
            "probability":      round(prob_7d, 4),
            "price_range_low":  price_range_low,
            "price_range_high": price_range_high,
            "confidence":       conf_7d,
            "model_warning":    model_warning,
        },
        "forecast_30d": {
            "direction":   direction_30d,
            "probability": round(prob_30d, 4),
            "confidence":  conf_30d,
        },
        "top_signals": top_signals,
    }


def predict_all(as_of_date: str = None) -> dict[str, dict]:
    """
    Generate forecasts for all 10 commodities and save to DuckDB.

    Returns:
        Dict mapping symbol → forecast dict.
    """
    from signals.price_features import ALL_SYMBOLS

    results = {}
    for symbol in ALL_SYMBOLS:
        try:
            fc = predict(symbol, as_of_date)
            results[symbol] = fc
            if "error" not in fc:
                _save_forecast(fc)
        except Exception as exc:
            log.error("predict %s failed: %s", symbol, exc)
            results[symbol] = {"symbol": symbol, "error": str(exc)}

    return results


def _save_forecast(fc: dict) -> None:
    """Persist a forecast to DuckDB for accuracy tracking."""
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO accuracy_log
                (date, symbol, forecast_direction, actual_direction, was_correct, confidence)
            VALUES (?, ?, ?, NULL, NULL, ?)
            """,
            [
                fc["as_of_date"],
                fc["symbol"],
                fc["forecast_7d"]["direction"],
                fc["forecast_7d"]["confidence"],
            ],
        )
    except Exception as exc:
        log.debug("Forecast save error: %s", exc)
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CommodiSense predictor")
    parser.add_argument("--symbol", default=None, help="Single symbol to predict")
    parser.add_argument("--all",    action="store_true", help="Predict all symbols")
    parser.add_argument("--date",   default=None, help="As-of date YYYY-MM-DD")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.all:
        results = predict_all(args.date)
        for sym, fc in results.items():
            if "error" not in fc:
                d7 = fc["forecast_7d"]
                print(f"{sym:<12} {d7['direction']:<7} {d7['probability']:.0%} [{d7['confidence']}]")
    elif args.symbol:
        fc = predict(args.symbol, args.date)
        print(json.dumps(fc, indent=2, default=str))
    else:
        parser.print_help()

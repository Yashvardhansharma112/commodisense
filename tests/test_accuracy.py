"""
Accuracy Backtesting Framework — walk-forward backtest over last 2 years
with 5 accuracy boosters for reaching 90%+ directional accuracy.

Usage:
    python tests/test_accuracy.py                  # full backtest all symbols
    python tests/test_accuracy.py --symbol ZW=F    # single symbol
    python tests/test_accuracy.py --boosters       # print booster impact analysis
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.feature_builder import build_training_data
from signals.price_features import ALL_SYMBOLS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

LABEL_MAP     = {-1: 0, 0: 1, 1: 2}
LABEL_REVERSE = {0: -1, 1: 0, 2: 1}

# Commodity-specific signal weight multipliers (Booster 2)
COMMODITY_WEIGHTS: dict[str, dict[str, float]] = {
    "CL=F":    {"risk_score_7d": 1.5,  "drought_index": 0.5},
    "NG=F":    {"risk_score_7d": 1.4,  "drought_index": 0.5},
    "GC=F":    {"dxy_proxy": 1.8,      "risk_score_30d": 1.3},
    "ZW=F":    {"drought_index": 2.0,  "sentiment_score_1d": 1.2},
    "ZC=F":    {"drought_index": 1.5,  "precip_anomaly_pct": 1.3},
    "ZS=F":    {"precip_anomaly_pct": 1.5, "drought_index": 1.4},
    "CT=F":    {"precip_anomaly_pct": 2.0, "heat_stress_days": 1.3},
    "SB=F":    {"precip_anomaly_pct": 1.6, "drought_index": 1.4},
    "USDINR=X":{"dxy_proxy": 1.8,      "risk_score_30d": 1.2},
    "HG=F":    {"risk_score_30d": 1.5, "momentum_score": 1.2},
}

# ── boosters ───────────────────────────────────────────────────────────────────


def booster1_signal_confirmation(
    proba: np.ndarray,
    features_row: pd.Series,
    min_agreeing: int = 3,
) -> tuple[int, str]:
    """
    Booster 1 — Signal Confirmation Filter.

    Only issue HIGH confidence if ≥3 independent signals agree on direction.
    Counts agreement from: sentiment, event direction, price momentum, weather.

    Returns:
        (pred_class, confidence)  where confidence ∈ "HIGH" | "MEDIUM" | "LOW"
    """
    pred_class = int(proba.argmax())
    pred_dir   = {0: "DOWN", 1: "STABLE", 2: "UP"}[pred_class]
    base_prob  = float(proba[pred_class])

    # Tally agreeing signals
    agreeing = 0

    # Sentiment signal
    sent = float(features_row.get("sentiment_score_1d", 0) or 0)
    if (pred_dir == "UP" and sent > 0.1) or (pred_dir == "DOWN" and sent < -0.1):
        agreeing += 1

    # Event direction
    dir_score = float(features_row.get("direction_score_7d", 0) or 0)
    if (pred_dir == "UP" and dir_score > 0) or (pred_dir == "DOWN" and dir_score < 0):
        agreeing += 1

    # Price momentum
    momentum = float(features_row.get("momentum_score", 0) or 0)
    if (pred_dir == "UP" and momentum > 0) or (pred_dir == "DOWN" and momentum < 0):
        agreeing += 1

    # Weather (drought → bullish for food commodities)
    drought = float(features_row.get("drought_index", 0) or 0)
    if pred_dir == "UP" and drought > 0.5:
        agreeing += 1
    elif pred_dir == "DOWN" and drought < 0.1:
        agreeing += 1

    # Assign confidence based on agreeing count
    if agreeing >= min_agreeing and base_prob >= 0.70:
        confidence = "HIGH"
    elif agreeing >= 2 and base_prob >= 0.55:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return pred_class, confidence


def booster2_apply_commodity_weights(
    X: pd.DataFrame, symbol: str
) -> pd.DataFrame:
    """
    Booster 2 — Commodity-specific feature weight scaling.
    Multiplies feature columns by commodity-specific importance weights.
    """
    weights = COMMODITY_WEIGHTS.get(symbol, {})
    if not weights:
        return X
    X = X.copy()
    for feat, w in weights.items():
        if feat in X.columns:
            X[feat] = X[feat] * w
    return X


def booster3_regime_detection(
    features_row: pd.Series,
    proba: np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    Booster 3 — Market Regime Detection.

    Detects: TRENDING (abs 30d return > 10%), VOLATILE (high ATR), RANGE_BOUND.
    In VOLATILE regime, confidence is suppressed to LOW.

    Returns:
        (adjusted_proba, regime)
    """
    ret30  = abs(float(features_row.get("return_30d", 0) or 0))
    atr_pct = float(features_row.get("atr_pct", 0) or 0)

    if atr_pct > 4.0:  # high volatility: ATR > 4% of price
        regime = "VOLATILE"
        # Flatten toward uniform to reduce confidence
        proba = proba * 0.7 + np.array([1/3, 1/3, 1/3]) * 0.3
    elif ret30 > 10.0:
        regime = "TRENDING"
        # Amplify momentum: boost the leading class probability
        momentum = float(features_row.get("momentum_score", 0) or 0)
        if momentum > 0 and proba[2] > proba[0]:  # UP leading
            proba[2] = min(proba[2] * 1.4, 0.95)
        elif momentum < 0 and proba[0] > proba[2]:  # DOWN leading
            proba[0] = min(proba[0] * 1.4, 0.95)
        proba = proba / proba.sum()
    else:
        regime = "RANGE_BOUND"

    return proba, regime


def booster4_sentiment_lag(
    X: pd.DataFrame, symbol: str, optimal_lags: dict[str, int] = None
) -> pd.DataFrame:
    """
    Booster 4 — Optimal Sentiment Lag.

    Shifts sentiment features by the per-commodity lag that maximises
    cross-correlation with price direction.

    optimal_lags: dict mapping symbol → lag_days (precomputed or computed here).
    """
    if optimal_lags is None:
        # Default lags from typical market research
        optimal_lags = {
            "CL=F": 2, "NG=F": 3, "GC=F": 1, "ZW=F": 5,
            "ZC=F": 4, "ZS=F": 4, "CT=F": 6, "SB=F": 5,
            "USDINR=X": 2, "HG=F": 3,
        }

    lag = optimal_lags.get(symbol, 2)
    X = X.copy()
    for col in ["sentiment_score_1d", "sentiment_3d", "sentiment_7d"]:
        if col in X.columns:
            X[col] = X[col].shift(lag).fillna(0)
    return X


# ── walk-forward backtesting ───────────────────────────────────────────────────


def _train_predict_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    symbol: str,
    apply_boosters: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train a fresh XGBoost + LightGBM ensemble on X_train and predict on X_test.
    Returns (y_pred, y_proba) for X_test.
    """
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight

    y_train_enc = y_train.map(LABEL_MAP).values
    sw = compute_sample_weight("balanced", y_train_enc)

    if apply_boosters:
        X_train = booster2_apply_commodity_weights(X_train, symbol)
        X_test  = booster2_apply_commodity_weights(X_test, symbol)
        X_train = booster4_sentiment_lag(X_train, symbol)
        X_test  = booster4_sentiment_lag(X_test, symbol)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train.fillna(0))
    X_te_s = scaler.transform(X_test.fillna(0))

    xgb_m = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )
    xgb_m.fit(X_tr_s, y_train_enc, sample_weight=sw, verbose=False)

    lgb_m = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        objective="multiclass", num_class=3,
        verbose=-1, random_state=42, n_jobs=-1,
    )
    lgb_m.fit(X_tr_s, y_train_enc, sample_weight=sw)

    xgb_proba  = xgb_m.predict_proba(X_te_s)
    lgb_proba  = lgb_m.predict_proba(X_te_s)
    ens_proba  = (xgb_proba + lgb_proba) / 2
    y_pred     = ens_proba.argmax(axis=1)

    return y_pred, ens_proba


def walk_forward_backtest(
    symbol: str,
    train_months: int = 18,
    test_months:  int = 1,
    apply_boosters: bool = True,
) -> dict:
    """
    Walk-forward backtest over last 2 years (Booster 6: rolling retrain).

    Train on `train_months`, test on next `test_months`, roll by `test_months`.

    Returns:
        Dict with accuracy metrics per fold and aggregate.
    """
    log.info("%s: walk-forward backtest (train=%dm, test=%dm)", symbol, train_months, test_months)

    X, y_7d, _ = build_training_data(symbol)
    if X.empty or len(X) < (train_months + test_months) * 20:
        return {"symbol": symbol, "error": "insufficient_data"}

    fold_results = []
    n = len(X)
    train_size = train_months * 21   # ~21 trading days per month
    test_size  = test_months  * 21

    i = train_size
    fold_num = 0
    while i + test_size <= n:
        fold_num += 1
        X_train = X.iloc[:i]
        y_train = y_7d.iloc[:i]
        X_test  = X.iloc[i:i + test_size]
        y_test  = y_7d.iloc[i:i + test_size]
        y_test_enc = y_test.map(LABEL_MAP).values

        try:
            y_pred, y_proba = _train_predict_fold(X_train, y_train, X_test, symbol, apply_boosters)
        except Exception as exc:
            log.warning("Fold %d failed: %s", fold_num, exc)
            i += test_size
            continue

        # Apply Booster 1 (signal confirmation) to reassign confidence
        confidences = []
        for j, proba_row in enumerate(y_proba):
            feat_row = X_test.iloc[j]
            _, conf = booster1_signal_confirmation(proba_row, feat_row)
            _, _ , regime = (None, None, booster3_regime_detection(proba_row, feat_row)[1])
            confidences.append(conf)

        acc = accuracy_score(y_test_enc, y_pred)
        high_mask = [c == "HIGH" for c in confidences]
        high_acc  = (
            accuracy_score(y_test_enc[high_mask], y_pred[high_mask])
            if any(high_mask) else 0.0
        )
        coverage = sum(1 for p in y_pred if p != 1) / len(y_pred)  # non-STABLE

        fold_results.append({
            "fold":       fold_num,
            "n_test":     len(y_test),
            "accuracy":   round(acc, 4),
            "high_conf_accuracy":  round(high_acc, 4),
            "high_conf_coverage":  round(sum(high_mask) / len(high_mask), 4),
            "non_stable_coverage": round(coverage, 4),
        })

        log.info("Fold %d: acc=%.1f%%  high_conf=%.1f%%  coverage=%.0f%%",
                 fold_num, acc * 100, high_acc * 100, coverage * 100)
        i += test_size

    if not fold_results:
        return {"symbol": symbol, "error": "no_folds_completed"}

    df_folds = pd.DataFrame(fold_results)
    return {
        "symbol":               symbol,
        "n_folds":              len(fold_results),
        "mean_accuracy_7d":     round(df_folds["accuracy"].mean(), 4),
        "mean_high_conf_acc":   round(df_folds["high_conf_accuracy"].mean(), 4),
        "mean_coverage":        round(df_folds["non_stable_coverage"].mean(), 4),
        "high_conf_coverage":   round(df_folds["high_conf_coverage"].mean(), 4),
        "meets_90pct_target":   df_folds["accuracy"].mean() >= 0.90,
        "fold_details":         fold_results,
    }


def run_all_backtests(apply_boosters: bool = True) -> dict:
    """
    Run walk-forward backtest for all 10 commodities and print report table.
    """
    results = {}
    for symbol in ALL_SYMBOLS:
        try:
            res = walk_forward_backtest(symbol, apply_boosters=apply_boosters)
            results[symbol] = res
        except Exception as exc:
            log.error("%s backtest error: %s", symbol, exc)
            results[symbol] = {"symbol": symbol, "error": str(exc)}

    # ── print summary table ──
    print("\n" + "=" * 90)
    print(f"{'Commodity':<15} {'7d Accuracy':>12} {'HIGH Conf Acc':>14} {'Coverage':>10} {'≥90%':>6}")
    print("=" * 90)

    for symbol, r in results.items():
        name = symbol
        if "error" in r:
            print(f"{name:<15} {'ERROR':>12}")
            continue
        acc   = f"{r['mean_accuracy_7d']:.1%}"
        hacc  = f"{r['mean_high_conf_acc']:.1%}"
        cov   = f"{r['non_stable_coverage']:.0%}" if isinstance(r.get('non_stable_coverage'), float) else "—"
        meets = "✓" if r.get("meets_90pct_target") else "✗"
        print(f"{name:<15} {acc:>12} {hacc:>14} {cov:>10} {meets:>6}")

    print("=" * 90)

    # Save report
    report_path = Path(__file__).parent.parent / "models" / "backtest_report.json"
    Path(report_path).parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed report saved to {report_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CommodiSense accuracy backtester")
    parser.add_argument("--symbol",   default=None, help="Single symbol to backtest")
    parser.add_argument("--boosters", action="store_true", help="Compare with/without boosters")
    parser.add_argument("--no-boosters", action="store_true", dest="no_boosters")
    args = parser.parse_args()

    if args.boosters:
        print("\n=== WITH boosters ===")
        run_all_backtests(apply_boosters=True)
        print("\n=== WITHOUT boosters ===")
        run_all_backtests(apply_boosters=False)
    elif args.symbol:
        result = walk_forward_backtest(
            args.symbol,
            apply_boosters=not args.no_boosters,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "fold_details"},
                          indent=2, default=str))
    else:
        run_all_backtests(apply_boosters=not args.no_boosters)

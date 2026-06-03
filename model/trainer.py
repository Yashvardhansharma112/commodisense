"""
Model Trainer — trains XGBoost + LightGBM ensemble per commodity.
Designed to run on Kaggle free notebooks (GPU available there) but
works on CPU locally.

IMPORTANT: Run this on Kaggle for GPU acceleration, or locally with CPU.
Saves trained models to models/ directory.

Usage:
    python model/trainer.py                    # train all symbols
    python model/trainer.py --symbol GC=F      # train one symbol
    python model/trainer.py --symbol ZW=F --horizon 7d
"""

import argparse
import json
import logging
import pickle
from datetime import date, timedelta
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.feature_builder import build_training_data
from signals.price_features import ALL_SYMBOLS

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# Label encoding: -1 → 0 (DOWN), 0 → 1 (STABLE), 1 → 2 (UP) for XGBoost
LABEL_MAP     = {-1: 0, 0: 1, 1: 2}
LABEL_REVERSE = {0: -1, 1: 0, 2: 1}
LABEL_NAMES   = {0: "DOWN", 1: "STABLE", 2: "UP"}

# ── Phase 6: Booster 2 — commodity-specific feature weight multipliers ─────────
# Applied to sample weights at training time so the model learns that certain
# features matter more for specific commodities.
COMMODITY_FEATURE_WEIGHTS: dict[str, dict[str, float]] = {
    "CL=F":     {"risk_score_7d": 1.5, "risk_score_30d": 1.5, "days_to_opec_meeting": 1.4,
                 "drought_index": 0.5},
    "NG=F":     {"days_to_opec_meeting": 1.4, "return_60d": 1.3, "atr_14": 1.3},
    "GC=F":     {"dxy_proxy": 1.8, "risk_score_7d": 1.3, "sentiment_score_1d": 1.2},
    "ZW=F":     {"drought_index": 2.0, "sentiment_score_1d": 1.2, "precip_anomaly_pct": 1.5},
    "ZC=F":     {"harvest_season_flag": 1.5, "drought_index": 1.8, "precip_anomaly_pct": 1.4},
    "ZS=F":     {"harvest_season_flag": 1.5, "drought_index": 1.6, "precip_anomaly_pct": 1.3},
    "CT=F":     {"harvest_season_flag": 1.6, "heat_stress_days": 2.0, "precip_anomaly_pct": 1.5},
    "SB=F":     {"harvest_season_flag": 1.5, "precip_anomaly_pct": 1.4},
    "USDINR=X": {"return_60d": 1.4, "momentum_score": 1.3, "macd_signal": 1.2},
    "HG=F":     {"risk_score_7d": 1.3, "return_60d": 1.4, "momentum_score": 1.2},
}

# ── model configs ──────────────────────────────────────────────────────────────

XGB_PARAMS = {
    "n_estimators":         500,
    "max_depth":            6,
    "learning_rate":        0.05,
    "subsample":            0.8,
    "colsample_bytree":     0.8,
    "objective":            "multi:softprob",
    "num_class":            3,
    "eval_metric":          "mlogloss",
    "early_stopping_rounds": 50,   # constructor param in XGBoost 3.x
    "random_state":         42,
    "n_jobs":               -1,
}

LGBM_PARAMS = {
    "n_estimators":    500,
    "num_leaves":      31,
    "learning_rate":   0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":    5,
    "objective":       "multiclass",
    "num_class":       3,
    "metric":          "multi_logloss",
    "verbose":         -1,
    "random_state":    42,
    "n_jobs":          -1,
}

# ── helpers ────────────────────────────────────────────────────────────────────


def _encode_labels(y: pd.Series) -> np.ndarray:
    return y.map(LABEL_MAP).values


def _compute_sample_weights(y_encoded: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights. Falls back to uniform if not all 3 classes present."""
    from sklearn.utils.class_weight import compute_sample_weight
    if len(np.unique(y_encoded)) < 3:
        return np.ones(len(y_encoded), dtype=float)
    return compute_sample_weight("balanced", y_encoded)


def _select_top_features(
    X: pd.DataFrame,
    importances: np.ndarray,
    top_n: int = 20,
    min_importance: float = 0.01,
) -> list[str]:
    """Return top_n feature names by importance, filtering below min_importance."""
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    selected = feat_imp[feat_imp >= min_importance].head(top_n).index.tolist()
    if len(selected) < 5:
        selected = feat_imp.head(top_n).index.tolist()
    return selected


def _detect_regime(X: pd.DataFrame) -> np.ndarray:
    """
    Booster 3 — Regime Detection.
    Returns per-row regime array: 0=RANGE_BOUND, 1=TRENDING, 2=VOLATILE.
    Uses ATR% and absolute 30-day return to classify market state.
    Only applied when X has enough rows to compute rolling stats (>60).
    """
    if len(X) < 60 or "atr_pct" not in X.columns:
        return np.zeros(len(X), dtype=int)

    atr_pct   = X["atr_pct"].fillna(0)
    ret_30d   = X.get("return_30d", pd.Series(0, index=X.index)).abs().fillna(0)

    atr_mean  = atr_pct.rolling(60, min_periods=20).mean().fillna(atr_pct.mean())
    atr_std   = atr_pct.rolling(60, min_periods=20).std().fillna(atr_pct.std())
    atr_thresh_volatile = atr_mean + 1.5 * atr_std

    regime = np.zeros(len(X), dtype=int)
    regime[ret_30d.values > 10.0]                      = 1   # TRENDING
    regime[atr_pct.values > atr_thresh_volatile.values] = 2  # VOLATILE
    return regime


def _apply_commodity_weights(
    sample_weights: np.ndarray,
    X: pd.DataFrame,
    symbol: str,
    regime: np.ndarray,
) -> np.ndarray:
    """
    Booster 2+3 combined — scale sample weights by commodity-specific feature
    importance multipliers, then dampen VOLATILE-regime rows (trust nothing when
    the market is in a chaotic state).
    """
    w = sample_weights.copy().astype(float)

    # Commodity-specific: up-weight rows where the key signal is strong
    for feat, mult in COMMODITY_FEATURE_WEIGHTS.get(symbol, {}).items():
        if feat in X.columns:
            signal_strength = X[feat].abs().fillna(0)
            percentile_75   = np.percentile(signal_strength, 75)
            if percentile_75 > 0:
                strong_rows = (signal_strength >= percentile_75).values
                w[strong_rows] *= mult

    # Regime: dampen volatile rows (Booster 3 — "trust nothing when volatile")
    w[regime == 2] *= 0.6
    # Trending rows: trust momentum features more — mild up-weight
    w[regime == 1] *= 1.2

    # Renormalise so total weight is unchanged
    total = w.sum()
    if total > 0:
        w = w / total * len(w)
    return w


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy of predicting UP/DOWN/STABLE direction correctly."""
    return float(np.mean(y_true == y_pred))


def _sharpe_ratio(y_true_raw: pd.Series, y_pred_encoded: np.ndarray) -> float:
    """
    Naive Sharpe: long when model predicts UP, short when DOWN, flat when STABLE.
    Uses true direction as proxy for daily return sign.
    """
    pred_dirs = pd.Series(y_pred_encoded).map(LABEL_REVERSE)
    returns = pred_dirs * y_true_raw.values  # +1 correct, -1 wrong
    mu = returns.mean()
    sigma = returns.std()
    return round(float(mu / sigma * np.sqrt(252)) if sigma > 0 else 0.0, 3)


# ── training ───────────────────────────────────────────────────────────────────


def train_symbol(
    symbol: str,
    horizon: str = "7d",
    add_lag_features: bool = True,
    last_days: int = None,
) -> dict:
    """
    Train XGBoost + LightGBM ensemble for a single commodity and horizon.

    Args:
        symbol:            Commodity ticker, e.g. "ZW=F"
        horizon:           "7d" or "30d"
        add_lag_features:  Add interaction features (accuracy booster)
        last_days:         If set, train only on the most recent N calendar days.
                           Use this when NLP signals only cover a short window —
                           avoids 4+ years of zero-padded sentiment rows diluting the model.

    Returns:
        Dict with accuracy metrics for this symbol/horizon.
    """
    log.info("Training %s | horizon=%s | window=%s",
             symbol, horizon, f"last {last_days}d" if last_days else "full")

    X, y_7d, y_30d = build_training_data(symbol)
    if X.empty:
        log.warning("%s: no training data, skipping", symbol)
        return {"symbol": symbol, "horizon": horizon, "error": "no_data"}

    # Trim to short window — keeps only rows where NLP signals are non-zero
    if last_days is not None:
        cutoff = date.today() - timedelta(days=last_days)
        if "date" in X.columns:
            mask = pd.to_datetime(X["date"]).dt.date >= cutoff
        else:
            # date is the index order — take the last last_days * 0.7 rows (trading days)
            trading_days = int(last_days * 0.71)
            mask = pd.Series([False] * len(X))
            mask.iloc[-trading_days:] = True
        X = X[mask.values].reset_index(drop=True)
        y_7d  = y_7d[mask.values].reset_index(drop=True)
        y_30d = y_30d[mask.values].reset_index(drop=True)
        log.info("%s: trimmed to %d rows (last %d days)", symbol, len(X), last_days)

    y = y_7d if horizon == "7d" else y_30d

    # Skip if one class dominates >95% — model would just memorise the majority class
    class_counts = y.value_counts(normalize=True)
    if class_counts.max() > 0.95:
        log.warning("%s %s: dominant class %.0f%% — skipping (too imbalanced to learn from)",
                    symbol, horizon, class_counts.max() * 100)
        return {"symbol": symbol, "horizon": horizon, "error": "extreme_class_imbalance"}

    # ── Phase 6 Booster 4: lag + interaction features ──
    if add_lag_features:
        X = X.copy()
        # Interaction: sentiment × momentum (strong when both agree)
        if "sentiment_score_1d" in X.columns and "momentum_score" in X.columns:
            X["sentiment_x_momentum"] = X["sentiment_score_1d"] * X["momentum_score"]
        # Interaction: event direction × price momentum
        if "direction_score_7d" in X.columns and "return_7d" in X.columns:
            X["event_x_momentum"] = X["direction_score_7d"] * np.sign(X["return_7d"].fillna(0))
        # Volatility regime flag (standalone feature for the model)
        if "atr_pct" in X.columns and len(X) >= 60:
            atr_mean = X["atr_pct"].rolling(60, min_periods=20).mean().fillna(X["atr_pct"].mean())
            X["high_volatility_flag"] = (X["atr_pct"] > atr_mean * 1.5).astype(int)

    y_enc = _encode_labels(y)
    sample_weights = _compute_sample_weights(y_enc)

    # Phase 6 Boosters 2+3: regime detection + commodity-specific weights
    if len(X) >= 60:
        regime = _detect_regime(X)
        sample_weights = _apply_commodity_weights(sample_weights, X, symbol, regime)
        trending_pct  = (regime == 1).mean() * 100
        volatile_pct  = (regime == 2).mean() * 100
        log.info("%s: regime — %.0f%% trending, %.0f%% volatile, %.0f%% range-bound",
                 symbol, trending_pct, volatile_pct, 100 - trending_pct - volatile_pct)

    # Short-window mode: use fewer folds + lighter model to avoid overfitting
    is_short_window = last_days is not None and len(X) < 200
    n_splits = 3 if is_short_window else 5
    xgb_params_cv = {**XGB_PARAMS, "n_estimators": 200, "max_depth": 3} if is_short_window else XGB_PARAMS

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_accs: list[float] = []
    best_features: list[str] | None = None
    last_fold_idx = n_splits - 1

    # ── cross-validation to find stable feature set ──
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_enc[train_idx], y_enc[val_idx]
        sw_train = sample_weights[train_idx]

        # Skip folds where val set has fewer than 3 samples or missing classes
        if len(y_val) < 3:
            continue

        scaler_fold = StandardScaler()
        X_tr_s = scaler_fold.fit_transform(X_train)
        X_vl_s = scaler_fold.transform(X_val)

        import xgboost as xgb
        xgb_fold = xgb.XGBClassifier(**xgb_params_cv)
        xgb_fold.fit(
            X_tr_s, y_train,
            sample_weight=sw_train,
            eval_set=[(X_vl_s, y_val)],
            verbose=False,
        )

        fold_acc = accuracy_score(y_val, xgb_fold.predict(X_vl_s))
        fold_accs.append(fold_acc)

        if fold == last_fold_idx:
            best_features = _select_top_features(X, xgb_fold.feature_importances_)

    if not fold_accs:
        return {"symbol": symbol, "horizon": horizon, "error": "all_folds_skipped"}

    cv_accuracy = float(np.mean(fold_accs))
    log.info("%s %s: CV accuracy %.3f (folds: %s)",
             symbol, horizon, cv_accuracy, [f"{a:.3f}" for a in fold_accs])

    # Short window: use lighter final model to avoid overfitting on small data
    if is_short_window:
        XGB_PARAMS_BOOSTED  = {**XGB_PARAMS,  "n_estimators": 300, "max_depth": 4, "learning_rate": 0.03}
        LGBM_PARAMS_BOOSTED = {**LGBM_PARAMS, "n_estimators": 300, "num_leaves": 15}
    elif cv_accuracy < 0.90 and add_lag_features:
        log.info("%s: below 90%%, boosting n_estimators to 1000", symbol)
        XGB_PARAMS_BOOSTED  = {**XGB_PARAMS,  "n_estimators": 1000}
        LGBM_PARAMS_BOOSTED = {**LGBM_PARAMS, "n_estimators": 1000}
    else:
        XGB_PARAMS_BOOSTED  = XGB_PARAMS
        LGBM_PARAMS_BOOSTED = LGBM_PARAMS

    # ── final training on full dataset using best_features ──
    X_selected = X[best_features] if best_features else X

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_selected)

    # Short window: 70/30 split to keep a meaningful test set; else 80/20
    test_frac = 0.30 if is_short_window else 0.20
    split = int(len(X_s) * (1 - test_frac))
    X_train_f, X_test_f = X_s[:split], X_s[split:]
    y_train_f, y_test_f = y_enc[:split], y_enc[split:]
    sw_f = sample_weights[:split]

    import xgboost as xgb
    import lightgbm as lgb

    xgb_model = xgb.XGBClassifier(**XGB_PARAMS_BOOSTED)
    xgb_model.fit(
        X_train_f, y_train_f,
        sample_weight=sw_f,
        eval_set=[(X_test_f, y_test_f)],
        verbose=False,
    )

    lgbm_model = lgb.LGBMClassifier(**LGBM_PARAMS_BOOSTED)
    lgbm_model.fit(
        X_train_f, y_train_f,
        sample_weight=sw_f,
        eval_set=[(X_test_f, y_test_f)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

    # Phase 6 Booster 5 — Platt/isotonic calibration on XGBoost
    # Uses the test split as held-out calibration data (cv="prefit")
    cal_cv = min(3, max(2, len(X_train_f) // 100))
    try:
        from sklearn.calibration import CalibratedClassifierCV
        xgb_calibrated = CalibratedClassifierCV(xgb_model, method="isotonic", cv="prefit")
        xgb_calibrated.fit(X_test_f, y_test_f)
    except Exception:
        xgb_calibrated = xgb_model  # fallback: uncalibrated

    # Soft-voting ensemble on test set (calibrated XGB + raw LGBM)
    xgb_proba  = xgb_calibrated.predict_proba(X_test_f)
    lgbm_proba = lgbm_model.predict_proba(X_test_f)
    ensemble_proba = (xgb_proba + lgbm_proba) / 2
    ensemble_pred  = ensemble_proba.argmax(axis=1)

    test_accuracy = _directional_accuracy(y_test_f, ensemble_pred)
    sharpe = _sharpe_ratio(y.iloc[split:].reset_index(drop=True), ensemble_pred)

    # Classification report
    report = classification_report(
        y_test_f, ensemble_pred,
        target_names=["DOWN", "STABLE", "UP"],
        output_dict=True,
    )

    # Feature importance (top 10 for report)
    top10_features = (
        pd.Series(xgb_model.feature_importances_, index=X_selected.columns)
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )

    log.info("%s %s: test accuracy=%.3f, Sharpe=%.2f", symbol, horizon, test_accuracy, sharpe)

    # ── save artifacts ──
    with open(MODELS_DIR / f"xgb_{symbol}_{horizon}.pkl", "wb") as f:
        pickle.dump(xgb_calibrated, f)
    with open(MODELS_DIR / f"lgbm_{symbol}_{horizon}.pkl", "wb") as f:
        pickle.dump(lgbm_model, f)
    with open(MODELS_DIR / f"scaler_{symbol}_{horizon}.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / f"feature_names_{symbol}_{horizon}.json", "w") as f:
        json.dump(X_selected.columns.tolist(), f)

    return {
        "symbol":           symbol,
        "horizon":          horizon,
        "cv_accuracy":      round(cv_accuracy, 4),
        "test_accuracy":    round(test_accuracy, 4),
        "sharpe_ratio":     sharpe,
        "n_features":       len(X_selected.columns),
        "n_train_samples":  split,
        "n_test_samples":   len(X_test_f),
        "top10_features":   top10_features,
        "classification_report": report,
    }


def train_all(horizons: list[str] = None, last_days: int = None) -> dict:
    """
    Train models for all 10 commodities and save accuracy report.

    Args:
        horizons:  List of horizons to train. Default: ["7d", "30d"]
        last_days: If set, train each symbol on only the most recent N days.

    Returns:
        Dict mapping symbol → accuracy metrics per horizon.
    """
    if horizons is None:
        horizons = ["7d", "30d"]

    results: dict = {}
    for symbol in ALL_SYMBOLS:
        results[symbol] = {}
        for horizon in horizons:
            try:
                metrics = train_symbol(symbol, horizon=horizon, last_days=last_days)
                results[symbol][horizon] = metrics
            except Exception as exc:
                log.error("Failed to train %s %s: %s", symbol, horizon, exc)
                results[symbol][horizon] = {"error": str(exc)}

    # Save combined accuracy report
    report_path = MODELS_DIR / "accuracy_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Accuracy report saved to %s", report_path)

    # Print summary table
    print("\n" + "=" * 85)
    print(f"{'Commodity':<15} {'7d Accuracy':>12} {'30d Accuracy':>13} {'Sharpe (7d)':>12} {'Samples':>8}")
    print("=" * 85)
    for symbol, res in results.items():
        r7  = res.get("7d", {})
        r30 = res.get("30d", {})
        acc7  = f"{r7.get('test_accuracy', 0):.1%}" if "test_accuracy" in r7 else "ERR"
        acc30 = f"{r30.get('test_accuracy', 0):.1%}" if "test_accuracy" in r30 else "ERR"
        sh7   = f"{r7.get('sharpe_ratio', 0):.2f}"   if "sharpe_ratio"  in r7 else "ERR"
        n     = r7.get("n_train_samples", 0)
        print(f"{symbol:<15} {acc7:>12} {acc30:>13} {sh7:>12} {n:>8}")
    print("=" * 85)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CommodiSense model trainer")
    parser.add_argument("--symbol",  default=None, help="Single symbol to train")
    parser.add_argument("--horizon", default="both", choices=["7d", "30d", "both"])
    parser.add_argument("--days",    default=None, type=int,
                        help="Train on only the most recent N calendar days (short-window mode)")
    args = parser.parse_args()

    if args.symbol:
        horizons = ["7d", "30d"] if args.horizon == "both" else [args.horizon]
        for h in horizons:
            result = train_symbol(args.symbol, horizon=h, last_days=args.days)
            print(json.dumps({k: v for k, v in result.items()
                               if k != "classification_report"}, indent=2, default=str))
    else:
        train_all(last_days=args.days)

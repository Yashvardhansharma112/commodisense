"""
Daily Pipeline Orchestrator — runs all 9 steps of the CommodiSense
data → signals → forecast → report pipeline.

Designed to run on GitHub Actions at 06:00 UTC Mon–Fri.
Each step is isolated in try/except — one failure does not halt the rest.

Usage:
    python pipeline/daily_run.py            # normal daily run
    python pipeline/daily_run.py --backfill # first-time full history load
    python pipeline/daily_run.py --step 5   # run only step N (for debugging)
"""

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RUN_DATE = date.today().isoformat()
LOG_DIR  = Path(__file__).parent.parent / "data" / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"pipeline_{RUN_DATE}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pipeline")

# ── step helpers ───────────────────────────────────────────────────────────────

_step_results: dict[str, dict] = {}


def _run_step(step_num: int, name: str, fn, *args, **kwargs) -> bool:
    """
    Execute a pipeline step with timing and error isolation.
    Records result in _step_results. Returns True on success.
    """
    log.info("=" * 60)
    log.info("STEP %d: %s", step_num, name)
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = round(time.time() - t0, 1)
        _step_results[name] = {"status": "ok", "elapsed_s": elapsed, "result": str(result)}
        log.info("STEP %d OK (%.1fs): %s", step_num, elapsed, result)
        return True
    except Exception as exc:
        elapsed = round(time.time() - t0, 1)
        _step_results[name] = {"status": "error", "elapsed_s": elapsed, "error": str(exc)}
        log.error("STEP %d FAILED (%.1fs): %s", step_num, elapsed, exc, exc_info=True)
        return False


# ── accuracy logging ───────────────────────────────────────────────────────────


def _log_accuracy() -> str:
    """
    Compare 7-day forecasts made 7 days ago vs today's actual prices.
    Updates accuracy_log.was_correct and actual_direction.
    Returns summary string.
    """
    from data.db import get_conn

    seven_days_ago = (date.today() - timedelta(days=7)).isoformat()
    today_str      = date.today().isoformat()

    conn = get_conn()
    # Load forecasts made 7 days ago that haven't been resolved yet
    pending = conn.execute(
        """
        SELECT date, symbol, forecast_direction, confidence
        FROM accuracy_log
        WHERE date = ? AND was_correct IS NULL
        """,
        [seven_days_ago],
    ).fetchall()

    if not pending:
        conn.close()
        return "No pending forecasts to evaluate"

    correct = 0
    for row_date, symbol, forecast_dir, confidence in pending:
        # Get actual price change from 7 days ago to today
        prices = conn.execute(
            """
            SELECT date, close FROM prices
            WHERE symbol = ? AND date IN (?, ?)
            ORDER BY date
            """,
            [symbol, seven_days_ago, today_str],
        ).fetchall()

        if len(prices) < 2:
            continue

        past_close  = float(prices[0][1])
        today_close = float(prices[1][1])
        change_pct  = (today_close - past_close) / past_close * 100

        if change_pct > 2.0:
            actual_dir = "UP"
        elif change_pct < -2.0:
            actual_dir = "DOWN"
        else:
            actual_dir = "STABLE"

        was_correct = forecast_dir == actual_dir

        conn.execute(
            """
            UPDATE accuracy_log
            SET actual_direction = ?, was_correct = ?
            WHERE date = ? AND symbol = ?
            """,
            [actual_dir, was_correct, seven_days_ago, symbol],
        )
        if was_correct:
            correct += 1

    conn.close()

    total = len(pending)
    accuracy = correct / total if total > 0 else 0
    return f"Accuracy log: {correct}/{total} correct ({accuracy:.0%}) for forecasts from {seven_days_ago}"


def _check_stale_symbols() -> list[str]:
    """Flag symbols with ≥3 consecutive days of pipeline errors."""
    from data.db import get_conn
    cutoff = (date.today() - timedelta(days=3)).isoformat()
    conn = get_conn()
    # Use accuracy_log as proxy: if no new forecast rows in 3 days, symbol is stale
    rows = conn.execute(
        """
        SELECT symbol, COUNT(*) as cnt
        FROM accuracy_log
        WHERE date >= ? AND was_correct IS NOT NULL
        GROUP BY symbol
        """,
        [cutoff],
    ).fetchall()
    conn.close()

    from signals.price_features import ALL_SYMBOLS
    active_symbols = {r[0] for r in rows}
    stale = [s for s in ALL_SYMBOLS if s not in active_symbols]
    if stale:
        log.warning("STALE symbols (no data in 3 days): %s", stale)
    return stale


# ── pipeline steps ─────────────────────────────────────────────────────────────


def run(backfill: bool = False, only_step: int | None = None) -> dict:
    """
    Execute the full daily pipeline.

    Args:
        backfill:   If True, runs all collectors in backfill mode (first run).
        only_step:  If set, runs only that step number (1–9) for debugging.

    Returns:
        Summary dict with step results and timing.
    """
    from data.db import init_schema
    init_schema()

    pipeline_start = time.time()
    log.info("Pipeline start: %s (backfill=%s)", RUN_DATE, backfill)

    # ── imports (deferred to avoid slow startup on import) ──
    from data.collector_prices        import run as run_prices
    from data.collector_news          import run as run_news
    from data.collector_weather       import run as run_weather
    from data.collector_geopolitical  import run as run_geo
    from data.collector_cot           import run as run_cot
    from data.collector_fred          import run as run_fred
    from data.collector_eia           import run as run_eia
    from data.collector_usda          import run as run_usda
    from signals.nlp_sentiment        import process_batch as run_sentiment
    from signals.nlp_events           import process_batch as run_events
    from model.predictor              import predict_all
    from model.explainer              import generate_all_reports

    steps = [
        (1,  "Collect prices",       run_prices,          {"backfill": backfill}),
        (2,  "Collect news",         run_news,            {"backfill": backfill}),
        (3,  "Collect weather",      run_weather,         {"backfill": backfill}),
        (4,  "Collect geopolitical", run_geo,             {"backfill": backfill}),
        (5,  "Collect COT",          run_cot,             {"backfill": backfill}),
        (6,  "Collect FRED macro",   run_fred,            {"backfill": backfill}),
        (7,  "Collect EIA inventory",run_eia,             {"backfill": backfill}),
        (8,  "Collect USDA crop",    run_usda,            {"backfill": backfill}),
        (9,  "Score sentiment",      run_sentiment,       {"limit": 500}),
        (10, "Extract events",       run_events,          {"limit": 500}),
        (11, "Generate forecasts",   predict_all,         {}),
        (12, "Generate reports",     generate_all_reports,{}),
        (13, "Log accuracy",         _log_accuracy,       {}),
    ]

    for step_num, step_name, fn, kwargs in steps:
        if only_step and step_num != only_step:
            continue
        _run_step(step_num, step_name, fn, **kwargs)

    # Post-run diagnostics
    _check_stale_symbols()

    total_elapsed = round(time.time() - pipeline_start, 1)
    errors = [name for name, r in _step_results.items() if r["status"] == "error"]

    summary = {
        "date":          RUN_DATE,
        "total_elapsed": total_elapsed,
        "steps_ok":      len(_step_results) - len(errors),
        "steps_failed":  len(errors),
        "failed_steps":  errors,
        "step_details":  _step_results,
    }

    log.info("=" * 60)
    log.info(
        "Pipeline complete: %.1fs | %d ok / %d failed",
        total_elapsed,
        summary["steps_ok"],
        summary["steps_failed"],
    )
    if errors:
        log.warning("Failed steps: %s", errors)

    # Save run summary to log dir
    summary_path = LOG_DIR / f"pipeline_summary_{RUN_DATE}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CommodiSense daily pipeline")
    parser.add_argument("--backfill",  action="store_true", help="Full history load (first run)")
    parser.add_argument("--step",      type=int, default=None, help="Run only step N (1–9)")
    args = parser.parse_args()
    result = run(backfill=args.backfill, only_step=args.step)
    n_steps = len(steps)
    print(f"\nPipeline {'OK' if result['steps_failed'] == 0 else 'PARTIAL'}: "
          f"{result['steps_ok']}/{n_steps} steps succeeded in {result['total_elapsed']}s")

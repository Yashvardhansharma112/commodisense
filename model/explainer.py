"""
Explainer — generates plain-English 3-sentence forecast reports using
Groq API (llama-3.3-70b-versatile, free tier: 14,400 req/day).
Falls back to a deterministic template if Groq is unavailable.

Reports are cached to data/reports/report_{date}.json.

Usage:
    python model/explainer.py --symbol ZW=F
    python model/explainer.py --all
"""

import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.predictor import predict, predict_all, SYMBOL_NAMES

log = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Groq client (lazy) ─────────────────────────────────────────────────────────

_groq_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        log.warning("GROQ_API_KEY not set — using template fallback")
        return None
    try:
        from groq import Groq
        _groq_client = Groq(api_key=api_key)
        return _groq_client
    except ImportError:
        log.warning("groq package not installed — using template fallback")
        return None


# ── helpers ────────────────────────────────────────────────────────────────────


def _format_signals(signals: list[dict]) -> str:
    """Format top signals as numbered list for the LLM prompt."""
    lines = []
    for i, sig in enumerate(signals[:5], 1):
        label = sig.get("label", sig.get("feature", "unknown"))
        value = sig.get("value", 0)
        impact = sig.get("impact", "NEUTRAL")
        weight = sig.get("weight", 0)
        lines.append(f"  {i}. {label}: {value:.3g} | Impact: {impact} | Weight: {weight:.3f}")
    return "\n".join(lines) if lines else "  (no signal data available)"


def _pick_risk_factor(prediction: dict) -> str:
    """Return the top bearish signal as the risk factor for the report."""
    signals = prediction.get("top_signals", [])
    bearish = [s for s in signals if s.get("impact") == "BEARISH"]
    if bearish:
        return bearish[0].get("label", "adverse signal reversal")
    # Generic risks per commodity type
    symbol = prediction.get("symbol", "")
    risk_map = {
        "CL=F":    "unexpected OPEC output increase",
        "NG=F":    "warmer-than-expected winter forecast",
        "GC=F":    "stronger-than-expected US jobs data",
        "ZW=F":    "favourable Black Sea weather reducing supply fears",
        "ZC=F":    "USDA upward crop estimate revision",
        "ZS=F":    "Brazil harvest beating expectations",
        "CT=F":    "recovery in monsoon rainfall",
        "SB=F":    "Brazil supply-side recovery",
        "USDINR=X":"RBI unexpected rate cut",
        "HG=F":    "China demand slowdown data",
    }
    return risk_map.get(symbol, "unexpected policy reversal")


def _template_report(prediction: dict) -> str:
    """
    Deterministic template-based report. Used when Groq is unavailable.
    No LLM needed — readable and fast.
    """
    name     = prediction.get("commodity_name", prediction.get("symbol", "Commodity"))
    fc7      = prediction.get("forecast_7d", {})
    fc30     = prediction.get("forecast_30d", {})
    direction= fc7.get("direction", "STABLE")
    prob     = fc7.get("probability", 0.5)
    conf     = fc7.get("confidence", "LOW")
    dir30    = fc30.get("direction", "STABLE")
    prob30   = fc30.get("probability", 0.5)
    signals  = prediction.get("top_signals", [])

    sig1 = signals[0] if len(signals) > 0 else {}
    sig2 = signals[1] if len(signals) > 1 else {}
    s1_label = sig1.get("label", "market momentum")
    s1_val   = sig1.get("value", 0)
    s2_label = sig2.get("label", "news sentiment")
    s2_val   = sig2.get("value", 0)
    risk     = _pick_risk_factor(prediction)

    dir_phrase = {
        "UP":     "rise",
        "DOWN":   "fall",
        "STABLE": "remain stable",
    }.get(direction, "remain stable")

    return (
        f"{name} is forecast to {dir_phrase} over the next 7 days "
        f"({prob:.0%} confidence, {conf}); 30-day view is {dir30} ({prob30:.0%}). "
        f"Primary drivers are {s1_label} at {s1_val:.3g} and "
        f"{s2_label} at {s2_val:.3g}. "
        f"Key risk: {risk} could invalidate this forecast."
    )


def _groq_report(prediction: dict) -> str:
    """Call Groq API to generate a 3-sentence analyst report."""
    client = _get_groq_client()
    if client is None:
        return _template_report(prediction)

    name    = prediction.get("commodity_name", prediction.get("symbol"))
    price   = prediction.get("current_price", 0)
    fc7     = prediction.get("forecast_7d", {})
    fc30    = prediction.get("forecast_30d", {})
    signals = prediction.get("top_signals", [])

    prompt = f"""You are a commodity market analyst. Based on the following data signals, write a 3-sentence forecast report. Be specific. Cite the signals. Use numbers.

Commodity: {name}
Current price: {price}
7-day forecast: {fc7.get('direction')} with {fc7.get('probability', 0):.0%} confidence ({fc7.get('confidence')} tier)
30-day forecast: {fc30.get('direction')} with {fc30.get('probability', 0):.0%} confidence

Top 5 driving signals:
{_format_signals(signals)}

Rules:
- Sentence 1: State the forecast and confidence level.
- Sentence 2: Name the top 2 signals and their specific values.
- Sentence 3: Name one risk factor that could invalidate this forecast.
- Write in plain English. No jargon. Max 80 words total.
- Do not use phrases like "based on the data" or "analysis suggests".
- Start directly: "{name} is forecast to..."
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3,
        )
        text = response.choices[0].message.content.strip()
        # Sanity: if response is empty or too short, fall back to template
        if len(text) < 30:
            return _template_report(prediction)
        return text
    except Exception as exc:
        log.warning("Groq API error: %s — using template fallback", exc)
        return _template_report(prediction)


# ── public API ─────────────────────────────────────────────────────────────────


def generate_report(prediction: dict) -> str:
    """
    Generate a plain-English 3-sentence forecast report for a commodity.

    Uses Groq if GROQ_API_KEY is set, otherwise falls back to template.

    Args:
        prediction: Dict returned by predictor.predict()

    Returns:
        3-sentence report string.
    """
    if "error" in prediction:
        return f"{prediction.get('symbol', 'Commodity')}: forecast unavailable ({prediction['error']})."

    return _groq_report(prediction)


def generate_all_reports(as_of_date: str = None) -> dict[str, str]:
    """
    Generate reports for all 10 commodities.
    Calls predict() + generate_report() for each.
    Caches results to data/reports/report_{date}.json.

    Args:
        as_of_date: ISO date string. Defaults to today.

    Returns:
        Dict mapping symbol → report string.
    """
    today = as_of_date or date.today().isoformat()
    cache_path = REPORTS_DIR / f"report_{today}.json"

    # Return cached reports if already generated today
    if cache_path.exists():
        log.info("Loading cached reports from %s", cache_path)
        with open(cache_path) as f:
            return json.load(f)

    forecasts = predict_all(as_of_date)
    reports: dict[str, str] = {}

    for symbol, fc in forecasts.items():
        report = generate_report(fc)
        reports[symbol] = report
        name = SYMBOL_NAMES.get(symbol, symbol)
        log.info("%s: report generated", name)

    # Cache to disk
    with open(cache_path, "w") as f:
        json.dump(reports, f, indent=2)
    log.info("Reports saved to %s", cache_path)

    return reports


def load_latest_reports() -> dict[str, str]:
    """
    Return the most recently generated report file, or empty dict if none.
    Used by the dashboard to display reports without regenerating.
    """
    report_files = sorted(REPORTS_DIR.glob("report_*.json"), reverse=True)
    if not report_files:
        return {}
    with open(report_files[0]) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="CommodiSense explainer")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--date",   default=None)
    args = parser.parse_args()

    if args.all:
        reports = generate_all_reports(args.date)
        for sym, report in reports.items():
            print(f"\n[{sym}]\n{report}")
    elif args.symbol:
        fc = predict(args.symbol, args.date)
        report = generate_report(fc)
        print(report)
    else:
        parser.print_help()

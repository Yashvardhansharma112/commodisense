"""
Explainer — generates structured analyst reports using Groq API
(llama-3.3-70b-versatile, free tier: 14,400 req/day).
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

# Walk-forward backtest accuracy per symbol (from 3.5yr backtest, Jun 2026)
BACKTEST_ACCURACY = {
    "CL=F": 37.1, "NG=F": 39.7, "GC=F": 51.9, "ZW=F": 37.3,
    "ZC=F": 48.8, "ZS=F": 47.3, "CT=F": 47.4, "SB=F": 37.4,
    "USDINR=X": 59.7, "HG=F": 43.7,
}

HIGH_CONF_ACCURACY = {
    "CL=F": 66.7, "NG=F": 82.3, "ZW=F": 56.9, "ZC=F": 100.0,
    "ZS=F": 75.0, "CT=F": 100.0, "SB=F": 58.4, "USDINR=X": 50.0,
}

_groq_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq
        _groq_client = Groq(api_key=api_key)
        return _groq_client
    except ImportError:
        return None


def _load_macro_context() -> dict:
    """Pull latest FRED macro row for context injection into the prompt."""
    try:
        from data.db import get_conn
        conn = get_conn()
        row = conn.execute(
            "SELECT date, dxy, vix, treasury_10y, fedfunds, indpro "
            "FROM fred_data WHERE dxy IS NOT NULL ORDER BY date DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            return {
                "date": str(row[0]), "dxy": row[1], "vix": row[2],
                "t10y": row[3], "fedfunds": row[4], "indpro": row[5],
            }
    except Exception:
        pass
    return {}


def _load_cot_context(symbol: str) -> dict:
    """Pull latest COT row for a symbol."""
    try:
        from data.db import get_conn
        conn = get_conn()
        row = conn.execute(
            "SELECT commercial_net_pct, mm_net_pct, commercial_chg_1w, mm_chg_1w "
            "FROM cot_data WHERE symbol = ? ORDER BY date DESC LIMIT 1",
            [symbol]
        ).fetchone()
        conn.close()
        if row:
            return {
                "commercial_net_pct": row[0], "mm_net_pct": row[1],
                "commercial_chg_1w": row[2], "mm_chg_1w": row[3],
            }
    except Exception:
        pass
    return {}


def _load_eia_context(symbol: str) -> dict:
    """Pull latest EIA inventory for CL=F / NG=F."""
    if symbol not in ("CL=F", "NG=F"):
        return {}
    try:
        from data.db import get_conn
        series = "crude_stocks" if symbol == "CL=F" else "natgas_stocks"
        conn = get_conn()
        row = conn.execute(
            "SELECT value, chg_1w, vs_5yr_avg FROM eia_inventory "
            "WHERE series = ? ORDER BY date DESC LIMIT 1",
            [series]
        ).fetchone()
        conn.close()
        if row:
            return {"value": row[0], "chg_1w": row[1], "vs_5yr_avg": row[2]}
    except Exception:
        pass
    return {}


def _format_signals(signals: list[dict]) -> str:
    lines = []
    for i, sig in enumerate(signals[:5], 1):
        label  = sig.get("label", sig.get("feature", "unknown"))
        value  = sig.get("value", 0)
        impact = sig.get("impact", "NEUTRAL")
        weight = sig.get("weight", 0)
        lines.append(f"  {i}. {label}: {value:.3g} ({impact}, weight {weight:.3f})")
    return "\n".join(lines) if lines else "  (no signal data)"


def _pick_risk_factor(prediction: dict) -> str:
    signals = prediction.get("top_signals", [])
    bearish = [s for s in signals if s.get("impact") == "BEARISH"]
    if bearish:
        return bearish[0].get("label", "adverse signal reversal")
    symbol = prediction.get("symbol", "")
    risk_map = {
        "CL=F":    "unexpected OPEC output increase or demand shock",
        "NG=F":    "warmer-than-expected seasonal forecasts cutting demand",
        "GC=F":    "stronger US jobs data reducing Fed cut expectations",
        "ZW=F":    "favourable Black Sea weather easing supply concerns",
        "ZC=F":    "USDA upward crop estimate revision",
        "ZS=F":    "Brazil harvest exceeding expectations",
        "CT=F":    "recovery in monsoon rainfall improving crop outlook",
        "SB=F":    "Brazil supply-side recovery above estimates",
        "USDINR=X":"RBI unexpected rate cut or foreign inflow surge",
        "HG=F":    "China industrial demand data disappointing",
    }
    return risk_map.get(symbol, "unexpected macro policy reversal")


def _template_report(prediction: dict) -> dict:
    """Structured template report — used when Groq is unavailable."""
    name     = prediction.get("commodity_name", prediction.get("symbol", "Commodity"))
    symbol   = prediction.get("symbol", "")
    price    = prediction.get("current_price", 0)
    fc7      = prediction.get("forecast_7d", {})
    fc30     = prediction.get("forecast_30d", {})
    direction= fc7.get("direction", "STABLE")
    prob     = fc7.get("probability", 0.5)
    conf     = fc7.get("confidence", "LOW")
    dir30    = fc30.get("direction", "STABLE")
    signals  = prediction.get("top_signals", [])
    accuracy = BACKTEST_ACCURACY.get(symbol, 45.0)

    sig1 = signals[0] if signals else {}
    sig2 = signals[1] if len(signals) > 1 else {}
    s1   = f"{sig1.get('label','momentum')} ({sig1.get('value',0):.3g})" if sig1 else "price momentum"
    s2   = f"{sig2.get('label','sentiment')} ({sig2.get('value',0):.3g})" if sig2 else "news sentiment"
    risk = _pick_risk_factor(prediction)

    dir_word = {"UP": "rise", "DOWN": "fall", "STABLE": "remain range-bound"}.get(direction, "remain range-bound")
    dir_emoji = {"UP": "▲", "DOWN": "▼", "STABLE": "◆"}.get(direction, "◆")

    cot = _load_cot_context(symbol)
    cot_line = ""
    if cot:
        comm = cot.get("commercial_net_pct", 0) or 0
        mm   = cot.get("mm_net_pct", 0) or 0
        cot_line = f"Institutional positioning: commercial hedgers {comm:+.1%}, managed money {mm:+.1%}."

    trade_bias = {
        "UP":     f"Bias long {name}. Monitor {s1} for continuation.",
        "DOWN":   f"Bias short {name}. Watch for {risk} as an exit trigger.",
        "STABLE": f"Range-bound. Wait for a directional break before committing.",
    }.get(direction, "No clear trade bias.")

    return {
        "outlook":     f"{dir_emoji} {name} is forecast to {dir_word} over the next 7 days — {prob:.0%} model probability, {conf} confidence. 30-day view: {dir30}. Model historical accuracy: {accuracy:.1f}% (vs 33.3% random).",
        "key_drivers": f"Primary signals driving this call: {s1} and {s2}. {cot_line}",
        "risk":        f"Main downside risk: {risk} could invalidate this forecast.",
        "trade_idea":  trade_bias,
    }


def _groq_report(prediction: dict) -> dict:
    """Call Groq API to generate a structured 4-section analyst report."""
    client = _get_groq_client()
    if client is None:
        return _template_report(prediction)

    name    = prediction.get("commodity_name", prediction.get("symbol"))
    symbol  = prediction.get("symbol", "")
    price   = prediction.get("current_price", 0)
    fc7     = prediction.get("forecast_7d", {})
    fc30    = prediction.get("forecast_30d", {})
    signals = prediction.get("top_signals", [])
    accuracy = BACKTEST_ACCURACY.get(symbol, 45.0)
    hc_acc   = HIGH_CONF_ACCURACY.get(symbol)
    conf     = fc7.get("confidence", "LOW")

    macro = _load_macro_context()
    cot   = _load_cot_context(symbol)
    eia   = _load_eia_context(symbol)

    macro_block = ""
    if macro:
        macro_block = (
            f"Macro context: DXY={macro.get('dxy',0):.1f}, VIX={macro.get('vix',0):.1f}, "
            f"10Y yield={macro.get('t10y',0):.2f}%, Fed Funds={macro.get('fedfunds',0):.2f}%"
        )

    cot_block = ""
    if cot:
        cot_block = (
            f"COT positioning: commercial hedgers {cot.get('commercial_net_pct',0):+.1%} net long "
            f"(week chg: {cot.get('commercial_chg_1w',0):+,.0f}), "
            f"managed money {cot.get('mm_net_pct',0):+.1%} net long"
        )

    eia_block = ""
    if eia:
        label = "Crude stocks" if symbol == "CL=F" else "Nat gas storage"
        eia_block = (
            f"{label}: {eia.get('value',0):,.0f} (week chg: {eia.get('chg_1w',0):+,.0f}, "
            f"vs 5yr avg: {eia.get('vs_5yr_avg',0):+.1f}%)"
        )

    hc_line = f" When confidence is HIGH, this model is right {hc_acc:.0f}% of the time." if hc_acc and conf == "HIGH" else ""

    prompt = f"""You are a professional commodity market analyst writing a structured report for a trading terminal.

COMMODITY: {name} ({symbol})
CURRENT PRICE: ${price:,.2f}
7-DAY FORECAST: {fc7.get('direction')} | Probability: {fc7.get('probability',0):.0%} | Confidence: {conf}
30-DAY FORECAST: {fc30.get('direction')} | Probability: {fc30.get('probability',0):.0%} | Confidence: {fc30.get('confidence','LOW')}
MODEL ACCURACY: {accuracy:.1f}% historical (random = 33.3%).{hc_line}

TOP SIGNALS (SHAP-ranked):
{_format_signals(signals)}

{macro_block}
{cot_block}
{eia_block}

Write EXACTLY this JSON structure — no extra keys, no markdown fences:
{{
  "outlook": "2 sentences. State the directional forecast, probability, confidence tier, and 30-day view. Mention the model accuracy context.",
  "key_drivers": "2 sentences. Name the top 2-3 signals with their actual values. Include COT positioning or EIA inventory if relevant.",
  "risk": "1 sentence. The single most important factor that could invalidate this forecast.",
  "trade_idea": "1-2 sentences. Actionable bias — long/short/wait, entry trigger, what to watch."
}}

Rules:
- Use numbers and specific values everywhere possible
- No filler phrases like "based on the analysis" or "it is worth noting"
- Write like a Bloomberg terminal analyst, not a chatbot
- Total word count: 80-120 words across all 4 fields
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        # Validate all 4 keys present
        for key in ("outlook", "key_drivers", "risk", "trade_idea"):
            if key not in parsed or not parsed[key]:
                raise ValueError(f"Missing key: {key}")
        return parsed

    except Exception as exc:
        log.warning("Groq report failed (%s) — using template", exc)
        return _template_report(prediction)


# ── public API ─────────────────────────────────────────────────────────────────


def generate_report(prediction: dict) -> dict:
    """
    Generate a structured 4-section analyst report for a commodity.

    Returns:
        Dict with keys: outlook, key_drivers, risk, trade_idea
    """
    if "error" in prediction:
        sym = prediction.get("symbol", "Commodity")
        return {
            "outlook":     f"{sym}: forecast unavailable ({prediction['error']}).",
            "key_drivers": "Run the daily pipeline to generate features.",
            "risk":        "No data.",
            "trade_idea":  "No actionable signal.",
        }
    return _groq_report(prediction)


def generate_all_reports(as_of_date: str = None) -> dict[str, dict]:
    today = as_of_date or date.today().isoformat()
    cache_path = REPORTS_DIR / f"report_{today}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        # If cached as old string format, regenerate
        if data and isinstance(next(iter(data.values())), str):
            cache_path.unlink()
        else:
            return data

    forecasts = predict_all(as_of_date)
    reports: dict[str, dict] = {}
    for symbol, fc in forecasts.items():
        reports[symbol] = generate_report(fc)
        log.info("%s: report generated", SYMBOL_NAMES.get(symbol, symbol))

    with open(cache_path, "w") as f:
        json.dump(reports, f, indent=2)
    return reports


def load_latest_reports() -> dict[str, dict]:
    """Return the most recently generated reports, or empty dict."""
    report_files = sorted(REPORTS_DIR.glob("report_*.json"), reverse=True)
    if not report_files:
        return {}
    with open(report_files[0]) as f:
        data = json.load(f)
    # Migrate old string-format cache
    if data and isinstance(next(iter(data.values())), str):
        return {}
    return data


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
        for sym, r in reports.items():
            print(f"\n[{sym}]")
            for k, v in r.items():
                print(f"  {k.upper()}: {v}")
    elif args.symbol:
        fc = predict(args.symbol, args.date)
        r  = generate_report(fc)
        for k, v in r.items():
            print(f"{k.upper()}: {v}")
    else:
        parser.print_help()

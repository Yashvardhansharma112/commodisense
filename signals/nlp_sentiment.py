"""
NLP Sentiment Engine — scores commodity news articles using FinBERT,
aggregates into daily sentiment features per commodity.

Usage:
    python signals/nlp_sentiment.py          # process all unscored articles
    python signals/nlp_sentiment.py --limit 200
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.db import get_conn, init_schema
from data.collector_news import get_unprocessed_news, mark_processed

LOG_PATH = Path(__file__).parent.parent / "data" / "logs" / "sentiment.log"
LOG_PATH.parent.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# Max tokens for FinBERT input (model limit is 512, we use 256 for speed)
MAX_TOKENS = 256

# ── model loading (lazy, cached) ───────────────────────────────────────────────

_pipeline = None


def _load_pipeline():
    """Load FinBERT pipeline once and cache it. Falls back to DistilBERT."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from transformers import pipeline as hf_pipeline

    try:
        log.info("Loading FinBERT (ProsusAI/finbert)...")
        _pipeline = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,          # return all 3 class probabilities
            device=-1,           # CPU
            truncation=True,
            max_length=MAX_TOKENS,
        )
        log.info("FinBERT loaded")
    except Exception as exc:
        log.warning("FinBERT load failed (%s), falling back to DistilBERT", exc)
        _pipeline = hf_pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            top_k=None,
            device=-1,
            truncation=True,
            max_length=MAX_TOKENS,
        )
        log.info("DistilBERT loaded as fallback")

    return _pipeline


# ── keyword-based baseline (for ensemble uncertainty check) ────────────────────

_BULLISH_WORDS = {
    "surge", "rally", "gain", "rise", "boom", "shortage", "record high",
    "supply cut", "output cut", "strong demand", "bullish",
}
_BEARISH_WORDS = {
    "fall", "drop", "crash", "decline", "surplus", "oversupply",
    "demand drop", "weak demand", "bearish", "glut",
}


def _keyword_sentiment(text: str) -> float:
    """Fast keyword-based sentiment score in [-1, +1]."""
    lower = text.lower()
    pos = sum(1 for w in _BULLISH_WORDS if w in lower)
    neg = sum(1 for w in _BEARISH_WORDS if w in lower)
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0


# ── public API ─────────────────────────────────────────────────────────────────


def score_article(text: str) -> float:
    """
    Score a single text string using FinBERT.

    Returns:
        Sentiment score in [-1.0, +1.0].
        positive_prob - negative_prob from FinBERT.
        Falls back to keyword score if model unavailable.
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    try:
        pipe = _load_pipeline()
        results = pipe(text[:512])[0]  # list of {label, score} dicts

        scores = {r["label"].lower(): r["score"] for r in results}

        # FinBERT labels: positive / negative / neutral
        # DistilBERT labels: POSITIVE / NEGATIVE — normalize
        pos = scores.get("positive", scores.get("label_1", 0.0))
        neg = scores.get("negative", scores.get("label_0", 0.0))

        ml_score = pos - neg  # range [-1, +1]

        # Ensemble uncertainty check: if ML and keyword disagree strongly, use neutral
        kw_score = _keyword_sentiment(text)
        if abs(ml_score - kw_score) > 0.4 and abs(kw_score) > 0.1:
            return 0.0

        return round(ml_score, 4)

    except Exception as exc:
        log.debug("score_article error: %s", exc)
        return _keyword_sentiment(text)


def process_batch(limit: int = 100) -> int:
    """
    Score unprocessed articles from news_raw using batched inference and store
    aggregated daily sentiment.

    Batched pipeline call is 10-30x faster than scoring one article at a time.

    Args:
        limit: Max articles to process per call.

    Returns:
        Count of articles processed.
    """
    df = get_unprocessed_news(limit=limit)
    if df.empty:
        log.info("No unprocessed articles found")
        return 0

    log.info("Processing %d articles (batched)...", len(df))

    # Build text list and IDs together
    texts = [
        f"{row.get('title', '')} {row.get('summary', '')}".strip()[:512]
        for _, row in df.iterrows()
    ]
    ids = df["id"].tolist()

    # Single batched pipeline call — far faster than N individual calls
    pipe = _load_pipeline()
    try:
        batch_results = pipe(texts, batch_size=16, truncation=True)
    except Exception as exc:
        log.warning("Batched inference failed (%s), falling back to per-article", exc)
        batch_results = [pipe(t)[0] for t in texts]

    scores: list[float] = []
    for text, result in zip(texts, batch_results):
        try:
            # result is a list of dicts when top_k=None
            label_scores = {r["label"].lower(): r["score"] for r in result}
            pos = label_scores.get("positive", label_scores.get("label_1", 0.0))
            neg = label_scores.get("negative", label_scores.get("label_0", 0.0))
            ml_score = pos - neg

            kw_score = _keyword_sentiment(text)
            if abs(ml_score - kw_score) > 0.4 and abs(kw_score) > 0.1:
                scores.append(0.0)
            else:
                scores.append(round(ml_score, 4))
        except Exception:
            scores.append(_keyword_sentiment(text))

    # Bulk update in one connection
    conn = get_conn()
    for article_id, score in zip(ids, scores):
        conn.execute(
            "UPDATE news_raw SET sentiment_score = ?, processed = TRUE WHERE id = ?",
            [score, article_id],
        )
    conn.close()

    _aggregate_daily_sentiment()
    log.info("Processed %d articles", len(ids))
    return len(ids)


def _aggregate_daily_sentiment() -> None:
    """
    Recompute sentiment_daily table from scored news_raw rows.
    Applies time-decay weights: 1.0 (<24h), 0.5 (24–48h), 0.25 (48–72h).
    """
    conn = get_conn()
    # Get scored articles from last 7 days
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    df = conn.execute(
        """
        SELECT id, published_date, commodity_tags, sentiment_score
        FROM news_raw
        WHERE processed = TRUE
          AND sentiment_score IS NOT NULL
          AND published_date >= ?
        """,
        [cutoff],
    ).df()
    conn.close()

    if df.empty:
        return

    now = datetime.now(timezone.utc)
    rows_to_upsert: list[dict] = []

    # Explode commodity tags — one row per commodity mention
    records = []
    for _, row in df.iterrows():
        tags = str(row.get("commodity_tags") or "").split(",")
        pub = row["published_date"]
        if isinstance(pub, str):
            try:
                pub = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            except Exception:
                pub = now
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)

        age_hours = (now - pub).total_seconds() / 3600
        weight = 1.0 if age_hours < 24 else (0.5 if age_hours < 48 else 0.25)

        for tag in tags:
            tag = tag.strip()
            if tag:
                records.append({
                    "date": pub.date(),
                    "commodity": tag,
                    "score": row["sentiment_score"],
                    "weight": weight,
                })

    if not records:
        return

    df_exp = pd.DataFrame(records)

    # Weighted average per (date, commodity)
    def _wavg(g):
        w = g["weight"]
        s = g["score"]
        total_w = w.sum()
        return {
            "sentiment_score": (s * w).sum() / total_w if total_w > 0 else 0.0,
            "article_count": len(g),
            "positive_count": int((s > 0.1).sum()),
            "negative_count": int((s < -0.1).sum()),
        }

    summary = df_exp.groupby(["date", "commodity"]).apply(_wavg).reset_index()

    conn = get_conn()
    for _, row in summary.iterrows():
        vals = row[0] if isinstance(row[0], dict) else row.to_dict()
        conn.execute(
            """
            INSERT OR REPLACE INTO sentiment_daily
                (date, commodity, sentiment_score, article_count,
                 positive_count, negative_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                row["date"],
                row["commodity"],
                vals.get("sentiment_score", 0.0),
                vals.get("article_count", 0),
                vals.get("positive_count", 0),
                vals.get("negative_count", 0),
            ],
        )
    conn.close()


def get_sentiment_features(commodity: str, days: int = 30) -> pd.DataFrame:
    """
    Return daily sentiment features for a commodity with rolling averages.

    Args:
        commodity: Ticker symbol, e.g. "ZW=F"
        days:      Look-back window in calendar days

    Returns:
        DataFrame with columns: date, sentiment_score, article_count,
        sentiment_3d, sentiment_7d, positive_ratio_7d
    """
    cutoff = date.today() - timedelta(days=days)
    conn = get_conn()
    df = conn.execute(
        """
        SELECT * FROM sentiment_daily
        WHERE commodity = ? AND date >= ?
        ORDER BY date
        """,
        [commodity, cutoff],
    ).df()
    conn.close()

    if df.empty:
        return df

    df = df.sort_values("date").reset_index(drop=True)
    df["sentiment_3d"] = df["sentiment_score"].rolling(3, min_periods=1).mean()
    df["sentiment_7d"] = df["sentiment_score"].rolling(7, min_periods=1).mean()
    df["positive_ratio_7d"] = (
        df["positive_count"].rolling(7, min_periods=1).sum()
        / df["article_count"].rolling(7, min_periods=1).sum().replace(0, 1)
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    init_schema()
    n = process_batch(limit=args.limit)
    print(f"Processed {n} articles")

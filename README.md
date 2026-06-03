---
title: CommodiSense
emoji: 📈
colorFrom: slate
colorTo: slate
sdk: docker
app_file: dashboard/app.py
pinned: false
---

# ◈ CommodiSense — Global Commodity Intelligence Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006400?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-5B8C5A?style=flat-square)
![DuckDB](https://img.shields.io/badge/DuckDB-0.10+-FFF000?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live-00D97E?style=flat-square)

**Zero-cost commodity price direction forecaster for 10 global markets.**  
Powered by XGBoost + LightGBM ensemble, SHAP explainability, FinBERT NLP sentiment,  
CFTC COT positioning, EIA inventory data, USDA crop signals, and FRED macro indicators.

[**Live Demo**](https://commodisense.streamlit.app) · [**Report Bug**](https://github.com/Yashvardhansharma112/commodisense/issues) · [**Request Feature**](https://github.com/Yashvardhansharma112/commodisense/issues)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Data Sources](#data-sources)
- [Model Architecture](#model-architecture)
- [Accuracy Results](#accuracy-results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Daily Pipeline](#daily-pipeline)
- [API Keys](#api-keys)

---

## Overview

CommodiSense is a production-grade commodity intelligence platform that forecasts price direction (UP / STABLE / DOWN) for 10 global commodity futures over 7-day and 30-day horizons.

Unlike most financial ML projects that rely on price technicals alone, CommodiSense fuses **8 independent data sources** — including institutional positioning data (CFTC COT), energy inventory surprises (EIA), crop condition ratings (USDA), and macroeconomic indicators (FRED) — into a single ensemble model per commodity.

The entire system runs at **zero ongoing cost** using free public APIs, GitHub Actions for scheduling, Streamlit Cloud for hosting, and DuckDB as a serverless embedded database.

```
Data Collection → Feature Engineering → Ensemble Training → Live Dashboard
   (8 sources)       (65+ features)       (XGBoost+LGBM)    (Streamlit Cloud)
```

---

## Features

### Forecasting Engine
- **10 commodity markets**: Crude Oil (CL=F), Natural Gas (NG=F), Gold (GC=F), Wheat (ZW=F), Corn (ZC=F), Soybeans (ZS=F), Cotton (CT=F), Sugar (SB=F), USD/INR (USDINR=X), Copper (HG=F)
- **Dual horizons**: 7-day and 30-day directional forecasts
- **3-class output**: UP (>threshold%), STABLE, DOWN (<-threshold%) with per-commodity calibrated thresholds
- **Probability scores** with isotonic calibration for reliable confidence estimates
- **HIGH / MEDIUM / LOW confidence tiers** based on model probability
- **Signal confirmation filter**: 4 independent signals must agree to issue a HIGH-confidence call (price momentum, COT commercial positioning, EIA supply signal, USDA crop trend)

### Data Intelligence
- **CFTC COT Reports**: 13 years of weekly institutional positioning (commercial hedgers vs managed money). The single most valuable commodity signal — smart money positioning often leads price by 1–3 weeks.
- **EIA Inventory**: Weekly crude oil stocks (2,278 rows back to 1982) and natural gas storage (856 rows). Inventory surprises vs 5-year average directly drive energy price moves.
- **USDA NASS**: Weekly crop condition (% good + excellent) for corn, wheat, soybeans, cotton. Annual production estimates. Declining crop condition → bullish price signal.
- **FRED Macro**: USD Index (DXY), VIX volatility, 10-year Treasury yield, Fed Funds rate, Industrial Production. Gold inversely correlates with real yields; copper tracks industrial output.
- **FinBERT NLP**: GDELT news articles scored for financial sentiment (bullish/bearish/neutral). Rolling 1-day, 3-day, 7-day sentiment aggregates per commodity.
- **spaCy Event Extraction**: Supply shock, policy change, and geopolitical event detection from news headlines.
- **Open-Meteo Weather**: Drought index, heat stress days, precipitation anomaly for agricultural commodity regions.
- **ACLED Geopolitical**: Risk scores for regions that supply each commodity.

### Explainability
- **SHAP values** for every forecast — top 5 signal drivers shown in the dashboard
- Human-readable feature labels (e.g., "COT Smart Money Positioning", "EIA Crude Inventory Surprise")
- **AI Analyst Reports** generated via Groq LLM (Llama 3) contextualizing each forecast

### Dashboard (Dark Luxury Terminal)
- Live animated ticker strip with all 10 markets
- Macro environment bar: DXY, VIX, yield curve, spread, copper demand proxy
- Direction-colored commodity cards with confidence badges
- Candlestick chart with 20-day SMA and forecast zone overlay
- COT positioning chart (commercial vs managed money, 2-year history)
- EIA inventory bar chart with 4-week rolling average
- News sentiment chart with bull/bear zones
- Weather signal metrics
- AI analyst report per commodity
- Recent news feed with sentiment scores

### Infrastructure
- **GitHub Actions** daily pipeline (Mon–Fri 6am UTC): collect → process → retrain → forecast → commit
- **DuckDB** embedded database (no server required, zero cost)
- **Streamlit Cloud** free-tier hosting with auto-deploy on push
- Full **error isolation** — one failing step doesn't halt the rest of the pipeline

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     DAILY PIPELINE (13 Steps)                    │
├─────────────────────────────────────────────────────────────────┤
│  Step 1   Collect prices         yfinance → DuckDB               │
│  Step 2   Collect news           GDELT → DuckDB                  │
│  Step 3   Collect weather        Open-Meteo → DuckDB             │
│  Step 4   Collect geopolitical   ACLED → DuckDB                  │
│  Step 5   Collect COT            CFTC → DuckDB                   │
│  Step 6   Collect FRED macro     FRED CSV + yfinance → DuckDB    │
│  Step 7   Collect EIA inventory  EIA API v2 → DuckDB             │
│  Step 8   Collect USDA crop      USDA NASS API → DuckDB          │
│  Step 9   Score NLP sentiment    FinBERT → sentiment_daily       │
│  Step 10  Extract events         spaCy → extracted_events        │
│  Step 11  Generate forecasts     XGBoost+LightGBM → accuracy_log │
│  Step 12  Generate AI reports    Groq LLM → reports              │
│  Step 13  Log accuracy           Compare 7-day-old forecasts      │
└─────────────────────────────────────────────────────────────────┘

                             ↓ pushes to GitHub ↓

                    Streamlit Cloud auto-deploys
```

---

## Data Sources

| Source | Type | Coverage | Update Frequency | Key |
|--------|------|----------|-----------------|-----|
| **yfinance** | Price OHLCV | 12,613 rows · 5yr | Daily | None |
| **CFTC COT** | Futures positioning | 8,826 rows · 13yr | Weekly (Friday) | None |
| **FRED** | Macro indicators | 7,193 rows · 16yr | Daily/Weekly/Monthly | None |
| **EIA** | Energy inventory | 3,134 rows · 40yr crude | Weekly (Wednesday) | Free |
| **USDA NASS** | Crop condition & stocks | 1,104 rows · 5yr | Weekly/Quarterly | Free |
| **GDELT** | Global news | 392 articles | Daily | None |
| **Open-Meteo** | Agricultural weather | 210 rows | Daily | None |
| **ACLED** | Geopolitical events | 20 events | Weekly | None |

### Free API Keys Required

| API | Data | Register |
|-----|------|---------|
| EIA | Crude oil & natural gas weekly inventory | [eia.gov/opendata](https://www.eia.gov/opendata/register.php) |
| USDA NASS | Crop condition, stocks, production | [quickstats.nass.usda.gov/api](https://quickstats.nass.usda.gov/api) |
| Groq | AI analyst report generation | [console.groq.com](https://console.groq.com) |

---

## Model Architecture

### Per-Symbol Ensemble

Each of the 10 commodities has **two independent models** trained: one for the 7-day horizon and one for the 30-day horizon.

```
Raw Features (65+)
        │
        ▼
  Feature Selection                ← drops columns with <5% non-zero values
  (sparse filter)                    auto-excludes missing data sources
        │
        ▼
  StandardScaler                   ← fit on training data, saved per symbol
        │
        ├─────────────────────────────────────────────┐
        ▼                                             ▼
  XGBoost Classifier            LightGBM Classifier
  (300 trees, max_depth=5)      (300 trees, 31 leaves)
  + Isotonic Calibration
        │                                             │
        └──────────────┬──────────────────────────────┘
                       ▼
              Ensemble (avg probabilities)
                       │
                       ▼
              Direction + Probability
              (UP / STABLE / DOWN)
                       │
                       ▼
         Signal Confirmation Filter          ← 4-signal cross-check
         (momentum + COT + EIA + USDA)
                       │
                       ▼
              HIGH / MEDIUM / LOW confidence
```

### Feature Groups (65+ total)

| Group | Features | Count |
|-------|----------|-------|
| **Price technicals** | RSI-14, MACD, Bollinger Band position, ATR, SMA crossover | 5 |
| **Price momentum** | Return 1d/7d/14d/30d/60d, momentum score | 6 |
| **Seasonality** | Month sin/cos, harvest season flag, days to OPEC meeting | 4 |
| **Cross-commodity** | Oil/Gold ratio, DXY proxy | 2 |
| **CFTC COT** | Commercial net %, MM net %, week-over-week changes, open interest | 7 |
| **FRED macro** | DXY, VIX, 10Y yield, Fed Funds, INDPRO, yield inversion, copper basis | 12 |
| **EIA inventory** | Stocks level, weekly change, z-score vs 5yr avg, draw flag | 5 |
| **USDA crop** | Condition score, week-over-week change, stocks, production | 5 |
| **NLP sentiment** | 1-day/3-day/7-day sentiment, article count, positive ratio | 5 |
| **Event signals** | Bullish/bearish events, max severity, supply shock, policy change | 6 |
| **Geopolitical** | Risk score 7d, risk score 30d | 2 |
| **Weather** | Drought index, heat stress days, precipitation anomaly | 3 |
| **Data flags** | has_cot_data, has_fred_data, has_eia_data, has_usda_data | 4 |

### Training Strategy

- **Walk-forward validation**: 5-fold cross-validation on 80% of data, tested on most recent 20%
- **Class balancing**: `compute_sample_weight("balanced")` addresses UP/DOWN/STABLE imbalance
- **Commodity-specific thresholds**: USDINR uses ±0.4% threshold (managed float), NG=F uses ±3.5% (highly volatile)
- **Regime detection**: TRENDING / VOLATILE / RANGE_BOUND classification per row
- **Interaction features**: `sentiment × momentum`, `event × momentum`, `high_volatility_flag`
- **SHAP explainer**: TreeExplainer run post-training, top 5 features saved per forecast

---

## Accuracy Results

> Measured on held-out test set (most recent 20% of data). Random chance = 33.3% (3-class problem).

| Commodity | 7-Day | 30-Day | vs Baseline |
|-----------|-------|--------|------------|
| Crude Oil (CL=F) | 30.7% | 31.5% | +4.0% |
| Natural Gas (NG=F) | 36.3% | 44.6% | +3.6% |
| Gold (GC=F) | 37.1% | **54.2%** | +6.8% 30d |
| Wheat (ZW=F) | **44.6%** | 23.1% | +0.4% 7d |
| Corn (ZC=F) | 16.7%⚠ | **48.2%** | — |
| **Soybeans (ZS=F)** | **62.2%** | 48.6% | **+18.0%** |
| Cotton (CT=F) | **45.8%** | 34.7% | +0.8% |
| Sugar (SB=F) | 35.9% | 36.7% | — |
| USD/INR (USDINR=X) | 41.2% | **50.8%** | **+28.1%** 30d |
| Copper (HG=F) | 16.3%⚠ | 23.1% | — |
| **Average** | **36.7%** | **39.6%** | +5.4% vs random |

> ⚠ ZC=F 7d and HG=F have below-random accuracy due to structural market regime breaks in 2024–2026 (South American corn oversupply, HG=F name change in CFTC files limiting history). Use 30d forecasts for these symbols.

**Best performers:**
- 🥇 **ZS=F 7d: 62.2%** — USDA soybean crop condition is a dominant signal
- 🥈 **USDINR=X 30d: 50.8%** — FRED DXY + Fed Funds rate highly predictive for USD/INR
- 🥉 **GC=F 30d: 54.2%** — Gold responds strongly to yield curve and inflation expectations

---

## Tech Stack

```
Language        Python 3.10+
Database        DuckDB 0.10+ (embedded, zero-config, serverless)
ML              XGBoost 2.0, LightGBM 4.0, scikit-learn 1.3
Explainability  SHAP 0.42
NLP             HuggingFace Transformers (FinBERT), spaCy 3.5
Dashboard       Streamlit 1.28, Plotly 5.15
LLM Reports     Groq API (Llama 3)
Data APIs       yfinance, requests, FRED CSV, EIA API v2, USDA NASS API
Scheduling      GitHub Actions (cron)
Hosting         Streamlit Cloud (free tier)
```

---

## Project Structure

```
commodisense/
│
├── data/                          # Data collection layer
│   ├── db.py                      # DuckDB connection + schema init (9 tables)
│   ├── collector_prices.py        # yfinance OHLCV prices
│   ├── collector_news.py          # GDELT news articles
│   ├── collector_weather.py       # Open-Meteo agricultural weather
│   ├── collector_geopolitical.py  # ACLED geopolitical events
│   ├── collector_cot.py           # CFTC COT weekly positioning (2013–2026)
│   ├── collector_fred.py          # FRED macro + yfinance DXY/VIX
│   ├── collector_eia.py           # EIA crude oil + natural gas inventory
│   └── collector_usda.py          # USDA crop condition + stocks + production
│
├── signals/                       # Feature engineering layer
│   ├── price_features.py          # RSI, MACD, momentum, seasonality, cross-commodity
│   ├── nlp_sentiment.py           # FinBERT sentiment scoring pipeline
│   ├── nlp_events.py              # spaCy event extraction
│   ├── weather_features.py        # Drought/heat/precip aggregation by commodity region
│   └── macro_features.py          # COT + FRED + EIA + USDA feature engineering
│
├── model/                         # ML layer
│   ├── feature_builder.py         # Assembles all signals → training matrix (no lookahead)
│   ├── trainer.py                 # XGBoost + LightGBM training, calibration, SHAP
│   ├── predictor.py               # Inference with signal confirmation filter
│   └── explainer.py               # AI report generation via Groq
│
├── pipeline/
│   └── daily_run.py               # 13-step orchestrator with error isolation
│
├── dashboard/
│   └── app.py                     # Streamlit dashboard (dark luxury terminal UI)
│
├── models/                        # Trained model artifacts (committed to git)
│   ├── xgb_{SYMBOL}_{horizon}.pkl
│   ├── lgbm_{SYMBOL}_{horizon}.pkl
│   ├── scaler_{SYMBOL}_{horizon}.pkl
│   ├── feature_names_{SYMBOL}_{horizon}.json
│   └── accuracy_report.json
│
├── tests/
│   └── test_accuracy.py           # Walk-forward backtesting framework (6 boosters)
│
├── .github/workflows/
│   └── daily_pipeline.yml         # GitHub Actions cron (Mon–Fri 06:00 UTC)
│
├── .env.example                   # Environment variable template
├── requirements.txt               # Python dependencies
└── README.md
```

### Database Schema (9 tables)

| Table | Description |
|-------|-------------|
| `prices` | Daily OHLCV per symbol |
| `news_raw` | Raw news articles with NLP scores |
| `sentiment_daily` | Aggregated daily sentiment per commodity |
| `extracted_events` | spaCy-extracted supply shocks, policy changes |
| `weather_features` | Drought/heat/precip by region and commodity |
| `geopolitical_events` | Risk scores per region/commodity |
| `accuracy_log` | Live forecast vs actual outcome tracking |
| `cot_data` | CFTC COT weekly positioning per symbol |
| `fred_data` | FRED macro series (daily, forward-filled) |
| `eia_inventory` | EIA weekly energy storage |
| `usda_crop` | USDA crop condition, stocks, production |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Yashvardhansharma112/commodisense.git
cd commodisense

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Environment Variables

```bash
# Copy the example and fill in your keys
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=your_groq_key_here       # groq.com — free, for AI reports
EIA_API_KEY=your_eia_key_here         # eia.gov/opendata — free
USDA_API_KEY=your_usda_key_here       # quickstats.nass.usda.gov/api — free
```

### First Run (Full Backfill)

```bash
# Initialize database schema
python data/db.py

# Backfill all data sources (takes ~15 minutes)
python pipeline/daily_run.py --backfill

# Train models for all 10 commodities
for symbol in CL=F NG=F GC=F ZW=F ZC=F ZS=F CT=F SB=F USDINR=X HG=F; do
    python model/trainer.py --symbol $symbol --horizon both
done

# Launch dashboard
streamlit run dashboard/app.py
```

The dashboard will be available at **http://localhost:8501**

### Individual Commands

```bash
# Collect specific data source
python data/collector_prices.py --backfill
python data/collector_cot.py --backfill
python data/collector_fred.py --backfill
python data/collector_eia.py --backfill
python data/collector_usda.py --backfill

# Run NLP pipeline
python signals/nlp_sentiment.py --limit 500
python signals/nlp_events.py --limit 500

# Generate forecast for a single symbol
python model/predictor.py --symbol ZS=F

# Generate all forecasts
python model/predictor.py --all

# Run accuracy backtest
python tests/test_accuracy.py --symbol ZS=F

# Run only a specific pipeline step (for debugging)
python pipeline/daily_run.py --step 7
```

---

## Configuration

### Per-Commodity Direction Thresholds

Different commodities have different volatility profiles. Thresholds are set in `model/feature_builder.py`:

| Symbol | Threshold | Rationale |
|--------|-----------|-----------|
| USDINR=X | ±0.4% | Managed float — rarely moves >1% in a week |
| GC=F | ±1.5% | Gold — moderately volatile |
| NG=F | ±3.5% | Natural gas — highly volatile seasonally |
| Others | ±2.0% | Default threshold |

### Adding a New Commodity

1. Add the ticker to `ALL_SYMBOLS` in `signals/price_features.py`
2. Add a human-readable name to `SYMBOL_NAMES` in `model/predictor.py`
3. Run `python data/collector_prices.py --backfill`
4. Train: `python model/trainer.py --symbol NEW=F --horizon both`

---

## Deployment

### Streamlit Cloud (Recommended — Free)

1. Fork or push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your GitHub repo
4. Set:
   - **Repository**: `Yashvardhansharma112/commodisense`
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
5. Click **Advanced settings** → paste in **Secrets** (TOML format):
   ```toml
   GROQ_API_KEY = "your_key"
   EIA_API_KEY  = "your_key"
   USDA_API_KEY = "your_key"
   ```
6. Click **Deploy**

### GitHub Actions (Daily Pipeline)

Add the same 3 keys as **Repository Secrets** at:
`Settings → Secrets → Actions → New repository secret`

The pipeline runs automatically Mon–Fri at 06:00 UTC. It:
1. Collects fresh data from all 8 sources
2. Runs NLP sentiment + event extraction
3. Generates new forecasts for all 10 symbols
4. Commits the updated `data/commodisense.duckdb` back to the repo
5. Streamlit Cloud auto-deploys on the new commit

---

## Daily Pipeline

The pipeline is defined in `pipeline/daily_run.py`. Each step is isolated in a `try/except` — one failure doesn't stop the rest.

```
Step 1   Collect prices          ~30s
Step 2   Collect news            ~60s   (GDELT rate-limited)
Step 3   Collect weather         ~45s
Step 4   Collect geopolitical    ~15s
Step 5   Collect COT             ~30s   (CFTC public ZIP download)
Step 6   Collect FRED macro      ~30s   (7 series + yfinance fallback)
Step 7   Collect EIA inventory   ~15s   (2 series via API)
Step 8   Collect USDA crop       ~60s   (4 commodities × 3 queries)
Step 9   Score NLP sentiment     ~120s  (FinBERT on GPU/CPU)
Step 10  Extract events          ~60s   (spaCy NER)
Step 11  Generate forecasts      ~30s   (10 symbols, cached models)
Step 12  Generate AI reports     ~90s   (Groq API, 10 LLM calls)
Step 13  Log accuracy            ~5s    (compare 7-day-old forecasts)
─────────────────────────────────────────
Total                            ~8-12 minutes
```

Manual trigger: Go to **Actions** tab → **Daily CommodiSense Pipeline** → **Run workflow**

---

## API Keys

| Key | Where to get | Cost | What it enables |
|-----|-------------|------|----------------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Free tier | AI analyst reports via Llama 3 |
| `EIA_API_KEY` | [eia.gov/opendata/register.php](https://www.eia.gov/opendata/register.php) | Free | Crude oil + natural gas weekly inventory data |
| `USDA_API_KEY` | [quickstats.nass.usda.gov/api](https://quickstats.nass.usda.gov/api) | Free | Crop condition, stocks, production |

The system runs without any API keys — it will skip those data collection steps and fall back to price technicals only. Accuracy improves significantly with all keys set.

---

## Accuracy Improvement Roadmap

| Data Source | Expected Gain | Status |
|------------|--------------|--------|
| CFTC COT (13yr history) | +5–8% avg | ✅ Implemented |
| EIA crude + natgas inventory | +10–13% for CL=F | ✅ Implemented |
| USDA crop condition | +15–18% for ZS=F | ✅ Implemented |
| FRED macro (DXY, VIX, yields) | +21% USDINR=X 30d | ✅ Implemented |
| South American crop data (CONAB) | +10–15% ZC=F | 🔲 Planned |
| LME copper warehouse stocks | +8–12% HG=F | 🔲 Planned |
| Heating/Cooling Degree Days (NOAA) | +5–8% NG=F | 🔲 Planned |
| WASDE monthly projections | +5–7% grains | 🔲 Planned |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **CFTC** for free public COT disaggregated reports
- **Federal Reserve (FRED)** for free macroeconomic data API
- **U.S. Energy Information Administration (EIA)** for free energy inventory API
- **USDA NASS** for free agricultural statistics API
- **GDELT Project** for free global news event database
- **Open-Meteo** for free historical weather API
- **yfinance** community for the excellent Yahoo Finance wrapper
- **Groq** for free Llama 3 inference API

---

<div align="center">

Built with Python · Deployed on Streamlit Cloud · Data from CFTC, FRED, EIA, USDA, GDELT

**[⭐ Star this repo](https://github.com/Yashvardhansharma112/commodisense)** if you find it useful

</div>

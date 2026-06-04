"""
CommodiSense Dashboard — Global Commodity Intelligence Engine
Dark luxury financial terminal UI.

Run: streamlit run dashboard/app.py
Deploy: Streamlit Cloud → main file: dashboard/app.py → secret: GROQ_API_KEY
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.db import get_conn, init_schema
from model.explainer import load_latest_reports, generate_report
from model.predictor import predict, SYMBOL_NAMES

# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CommodiSense",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── design tokens ──────────────────────────────────────────────────────────────

C = {
    "bg":          "#060A0F",
    "surface":     "#0D1117",
    "surface2":    "#161B22",
    "border":      "rgba(255,255,255,0.07)",
    "border_hi":   "rgba(255,255,255,0.14)",
    "up":          "#00D97E",
    "down":        "#FF3B55",
    "stable":      "#7A8899",
    "up_dim":      "rgba(0,217,126,0.12)",
    "down_dim":    "rgba(255,59,85,0.12)",
    "stable_dim":  "rgba(122,136,153,0.10)",
    "accent":      "#3D7FFF",
    "accent_dim":  "rgba(61,127,255,0.12)",
    "gold":        "#FFBB00",
    "text":        "#E6EDF3",
    "text2":       "#8B949E",
    "text3":       "#484F58",
    "conf_high":   "#00D97E",
    "conf_mid":    "#FFBB00",
    "conf_low":    "#7A8899",
}

DIR_COLOR  = {"UP": C["up"],   "DOWN": C["down"],   "STABLE": C["stable"]}
DIR_DIM    = {"UP": C["up_dim"],"DOWN": C["down_dim"],"STABLE": C["stable_dim"]}
DIR_ICON   = {"UP": "▲",       "DOWN": "▼",          "STABLE": "◆"}
CONF_COLOR = {"HIGH": C["conf_high"], "MEDIUM": C["conf_mid"], "LOW": C["conf_low"]}

ALL_SYMBOLS = list(SYMBOL_NAMES.keys())

# ── CSS ────────────────────────────────────────────────────────────────────────

def _inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif;
        background-color: {C['bg']};
        color: {C['text']};
    }}
    .stApp {{ background-color: {C['bg']}; }}
    .block-container {{ padding: 1.2rem 2rem 3rem 2rem; max-width: 1600px; }}

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {{ visibility: hidden; }}
    .stDeployButton {{ display: none; }}
    [data-testid="stSidebar"] {{ background: {C['surface']}; border-right: 1px solid {C['border']}; }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
    ::-webkit-scrollbar-track {{ background: {C['bg']}; }}
    ::-webkit-scrollbar-thumb {{ background: {C['border_hi']}; border-radius: 2px; }}

    /* Buttons */
    .stButton > button {{
        background: transparent;
        border: 1px solid {C['border_hi']};
        color: {C['text2']};
        border-radius: 6px;
        font-size: 0.78rem;
        padding: 4px 10px;
        transition: all 0.15s ease;
        font-family: 'Inter', sans-serif;
    }}
    .stButton > button:hover {{
        border-color: {C['accent']};
        color: {C['accent']};
        background: {C['accent_dim']};
    }}

    /* Metric cards */
    div[data-testid="metric-container"] {{
        background: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 10px;
        padding: 14px 16px;
    }}
    div[data-testid="metric-container"] label {{
        color: {C['text2']} !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {C['text']} !important;
        font-size: 1.3rem !important;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }}

    /* Radio + select */
    .stRadio > div {{ gap: 8px; }}
    .stRadio label {{ font-size: 0.8rem; color: {C['text2']}; }}
    .stSelectbox label {{ color: {C['text2']}; font-size: 0.8rem; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: transparent;
        border-bottom: 1px solid {C['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border: none;
        color: {C['text2']};
        font-size: 0.82rem;
        padding: 6px 14px;
        border-radius: 6px 6px 0 0;
    }}
    .stTabs [aria-selected="true"] {{
        background: {C['surface']} !important;
        color: {C['text']} !important;
        border-bottom: 2px solid {C['accent']};
    }}

    /* Ticker animation */
    @keyframes ticker-scroll {{
        0%   {{ transform: translateX(0); }}
        100% {{ transform: translateX(-50%); }}
    }}
    .ticker-wrap {{
        overflow: hidden;
        background: {C['surface']};
        border-top: 1px solid {C['border']};
        border-bottom: 1px solid {C['border']};
        padding: 8px 0;
        margin: -1rem -2rem 1.4rem -2rem;
    }}
    .ticker-inner {{
        display: flex;
        animation: ticker-scroll 40s linear infinite;
        width: max-content;
    }}
    .ticker-item {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0 28px;
        white-space: nowrap;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        border-right: 1px solid {C['border']};
    }}
    .ticker-sep {{
        padding: 0 28px;
        color: {C['text3']};
        font-size: 0.6rem;
        border-right: 1px solid {C['border']};
    }}

    /* Commodity cards */
    .comm-card {{
        background: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 12px;
        padding: 16px;
        cursor: pointer;
        transition: all 0.18s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }}
    .comm-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        border-radius: 12px 0 0 12px;
    }}
    .comm-card:hover {{
        border-color: {C['border_hi']};
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }}
    .comm-card.active {{
        border-color: {C['accent']} !important;
        background: linear-gradient(135deg, {C['surface']} 0%, rgba(61,127,255,0.05) 100%);
    }}
    .comm-card.up::before {{ background: {C['up']}; }}
    .comm-card.down::before {{ background: {C['down']}; }}
    .comm-card.stable::before {{ background: {C['stable']}; }}

    /* Signal pill */
    .signal-pill {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}

    /* Macro bar */
    .macro-item {{
        text-align: center;
        padding: 10px 16px;
        background: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 8px;
    }}
    .macro-label {{ font-size: 0.65rem; color: {C['text3']}; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 3px; }}
    .macro-value {{ font-size: 1.05rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; color: {C['text']}; }}
    .macro-change {{ font-size: 0.68rem; margin-top: 2px; }}

    /* AI report */
    .ai-report {{
        background: linear-gradient(135deg, {C['surface2']} 0%, rgba(61,127,255,0.04) 100%);
        border: 1px solid {C['border']};
        border-left: 3px solid {C['accent']};
        border-radius: 10px;
        padding: 16px 20px;
        line-height: 1.7;
        font-size: 0.9rem;
        color: {C['text']};
    }}

    /* News row */
    .news-row {{
        padding: 10px 0;
        border-bottom: 1px solid {C['border']};
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }}

    /* COT bar */
    .cot-label {{ font-size: 0.7rem; color: {C['text2']}; margin-bottom: 4px; }}
    .cot-bar-wrap {{
        height: 6px;
        background: {C['surface2']};
        border-radius: 3px;
        overflow: hidden;
        margin-bottom: 10px;
    }}

    /* Section header */
    .section-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid {C['border']};
    }}
    .section-title {{
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {C['text2']};
    }}
    .section-dot {{ width: 6px; height: 6px; border-radius: 50%; background: {C['accent']}; }}

    /* Confidence arc */
    .conf-badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }}
    </style>
    """, unsafe_allow_html=True)


# ── data loaders ───────────────────────────────────────────────────────────────

@st.cache_resource
def _ensure_schema():
    init_schema()

@st.cache_resource
def _ensure_prices():
    """Keep prices up to date on every container start.
    - If DB is empty: full 5-year backfill from yfinance.
    - If DB has data but is stale (latest < yesterday): fetch missing days only.
    Runs once per process lifetime."""
    try:
        import yfinance as yf
        from datetime import date as _date, timedelta

        conn = get_conn()
        row = conn.execute(
            "SELECT COUNT(*), MAX(date) FROM prices"
        ).fetchone()
        conn.close()
        count, latest_date = row[0], row[1]

        yesterday = (_date.today() - timedelta(days=1)).isoformat()

        # Nothing to do if already current
        if count > 100 and latest_date and str(latest_date) >= yesterday:
            return

        # Full backfill for empty DB, incremental top-up otherwise
        period = "5y" if count < 100 else None
        start  = None if period else (str(latest_date) if latest_date else "2020-01-01")

        symbols = list(SYMBOL_NAMES.keys())
        kwargs = dict(auto_adjust=True, progress=False)
        if period:
            kwargs["period"] = period
        else:
            kwargs["start"] = start

        ticker_data = yf.download(symbols, **kwargs)
        if ticker_data.empty:
            return

        conn2 = get_conn()
        for sym in symbols:
            try:
                df = ticker_data.xs(sym, axis=1, level=1) if len(symbols) > 1 else ticker_data.copy()
                if df is None or df.empty:
                    continue
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                for _, r in df.iterrows():
                    try:
                        conn2.execute(
                            "INSERT OR REPLACE INTO prices "
                            "(date,symbol,open,high,low,close,volume,adj_close) "
                            "VALUES (?,?,?,?,?,?,?,?)",
                            [str(r["date"])[:10], sym,
                             float(r.get("open")  or 0), float(r.get("high")  or 0),
                             float(r.get("low")   or 0), float(r.get("close") or 0),
                             float(r.get("volume") or 0), float(r.get("close") or 0)]
                        )
                    except Exception:
                        pass
            except Exception:
                pass
        conn2.close()
    except Exception:
        pass

@st.cache_resource
def _retag_news():
    """Retag all news articles using the expanded keyword lists.
    Fixes articles with empty tags and re-applies broader matching once per process."""
    try:
        from data.collector_news import COMMODITY_KEYWORDS
        conn = get_conn()
        rows = conn.execute(
            "SELECT id, title, summary FROM news_raw"
        ).fetchall()
        updated = 0
        for row_id, title, summary in rows:
            text  = ((title or "") + " " + (summary or "")).lower()
            tags  = [sym for sym, kws in COMMODITY_KEYWORDS.items()
                     if any(k in text for k in kws)]
            if tags:
                conn.execute(
                    "UPDATE news_raw SET commodity_tags = ? WHERE id = ?",
                    [",".join(tags), row_id]
                )
                updated += 1
        conn.close()
    except Exception:
        pass

@st.cache_data(ttl=3600)
def _load_forecast(symbol: str) -> dict:
    return predict(symbol)

@st.cache_data(ttl=3600)
def _load_all_forecasts(symbols: tuple) -> dict:
    return {s: _load_forecast(s) for s in symbols}

@st.cache_data(ttl=3600)
def _load_price_history(symbol: str, days: int = 90) -> pd.DataFrame:
    conn = get_conn()
    df = conn.execute(
        "SELECT date, open, high, low, close FROM prices "
        "WHERE symbol = ? AND date >= ? ORDER BY date",
        [symbol, (date.today() - timedelta(days=days)).isoformat()],
    ).df()
    conn.close()
    return df

@st.cache_data(ttl=3600)
def _load_sentiment_history(symbol: str, days: int = 60) -> pd.DataFrame:
    conn = get_conn()
    df = conn.execute(
        "SELECT date, sentiment_score, article_count FROM sentiment_daily "
        "WHERE commodity = ? AND date >= ? ORDER BY date",
        [symbol, (date.today() - timedelta(days=days)).isoformat()],
    ).df()
    conn.close()
    return df

@st.cache_data(ttl=3600)
def _load_cot_history(symbol: str, weeks: int = 104) -> pd.DataFrame:
    conn = get_conn()
    df = conn.execute(
        "SELECT date, commercial_net_pct, mm_net_pct, open_interest "
        "FROM cot_data WHERE symbol = ? ORDER BY date DESC LIMIT ?",
        [symbol, weeks],
    ).df()
    conn.close()
    return df.sort_values("date").reset_index(drop=True) if not df.empty else df

@st.cache_data(ttl=3600)
def _load_macro_env() -> dict:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT dxy, vix, treasury_10y, fedfunds, financial_stress, copper_basis "
            "FROM fred_data WHERE dxy IS NOT NULL ORDER BY date DESC LIMIT 1"
        ).fetchone()
    except Exception:
        row = None
    conn.close()
    if row:
        return {"dxy": row[0], "vix": row[1], "t10y": row[2],
                "fedfunds": row[3], "stress": row[4], "copper_basis": row[5]}
    return {}

@st.cache_data(ttl=1800)  # 30-min cache — live news
def _load_recent_news(symbol: str, limit: int = 20) -> pd.DataFrame:
    """Fetch live news from Yahoo Finance RSS, fall back to DB."""
    rows = []
    try:
        import feedparser, urllib.parse
        # Some tickers need remapping for Yahoo Finance RSS
        _YF_RSS_MAP = {"USDINR=X": "INR=X"}
        rss_ticker = _YF_RSS_MAP.get(symbol, symbol)
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={urllib.parse.quote(rss_ticker)}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        for entry in feed.entries[:limit]:
            pub = entry.get("published", "")
            try:
                from email.utils import parsedate_to_datetime
                pub_dt = parsedate_to_datetime(pub).strftime("%Y-%m-%d %H:%M")
            except Exception:
                pub_dt = pub[:16]
            rows.append({
                "published_date": pub_dt,
                "title":          entry.get("title", ""),
                "url":            entry.get("link", "#"),
                "sentiment_score": 0.0,
                "source":         "Yahoo Finance",
            })
    except Exception:
        pass

    # Fallback to DB if Yahoo RSS returned nothing
    if not rows:
        conn = get_conn()
        db_df = conn.execute(
            "SELECT published_date, title, url, sentiment_score FROM news_raw "
            "WHERE commodity_tags LIKE ? ORDER BY published_date DESC LIMIT ?",
            [f"%{symbol}%", limit],
        ).df()
        conn.close()
        if not db_df.empty:
            db_df["source"] = "GDELT"
            return db_df

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["published_date", "title", "url", "sentiment_score", "source"])

@st.cache_data(ttl=3600)
def _load_weather(symbol: str) -> dict:
    from signals.weather_features import get_weather_features
    return get_weather_features(symbol, days=30)

@st.cache_data(ttl=3600)
def _load_eia_history(series: str, weeks: int = 52) -> pd.DataFrame:
    conn = get_conn()
    df = conn.execute(
        "SELECT date, value, chg_1w, vs_5yr_avg FROM eia_inventory "
        "WHERE series = ? ORDER BY date DESC LIMIT ?",
        [series, weeks],
    ).df()
    conn.close()
    return df.sort_values("date").reset_index(drop=True) if not df.empty else df

# ── header ─────────────────────────────────────────────────────────────────────

def _render_header():
    now = datetime.now()
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:16px 0 12px 0;border-bottom:1px solid {C['border']};margin-bottom:0;">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="font-size:1.6rem;font-weight:700;letter-spacing:-0.02em;
                    background:linear-gradient(135deg,{C['text']} 0%,{C['accent']} 100%);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
          ◈ CommodiSense
        </div>
        <div style="display:flex;align-items:center;gap:5px;
                    background:{C['surface']};border:1px solid {C['border']};
                    border-radius:20px;padding:3px 10px;">
          <div style="width:6px;height:6px;border-radius:50%;background:{C['up']};
                      box-shadow:0 0 6px {C['up']};animation:pulse 2s infinite;"></div>
          <span style="font-size:0.68rem;color:{C['up']};font-weight:600;letter-spacing:0.06em;">LIVE</span>
        </div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:0.7rem;color:{C['text3']};letter-spacing:0.06em;text-transform:uppercase;">
          Global Commodity Intelligence
        </div>
        <div style="font-size:0.78rem;color:{C['text2']};font-family:'JetBrains Mono',monospace;">
          {now.strftime('%a %d %b %Y  %H:%M')} UTC
        </div>
      </div>
    </div>
    <style>
    @keyframes pulse {{
      0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }}
    }}
    </style>
    """, unsafe_allow_html=True)


# ── ticker strip ───────────────────────────────────────────────────────────────

def _render_ticker(forecasts: dict, horizon_key: str):
    fk = "forecast_7d" if horizon_key == "7d" else "forecast_30d"
    items_html = ""
    for sym in ALL_SYMBOLS:
        fc = forecasts.get(sym, {})
        if "error" in fc or not fc:
            continue
        f     = fc.get(fk, {})
        price = fc.get("current_price", 0)
        dir_  = f.get("direction", "STABLE")
        prob  = f.get("probability", 0)
        icon  = DIR_ICON.get(dir_, "◆")
        col   = DIR_COLOR.get(dir_, C["stable"])
        name  = SYMBOL_NAMES.get(sym, sym).upper()
        items_html += f"""
        <div class="ticker-item">
          <span style="color:{C['text3']};font-size:0.65rem;">{sym}</span>
          <span style="color:{C['text']};font-weight:500;">{name}</span>
          <span style="color:{C['text2']};">${price:,.2f}</span>
          <span style="color:{col};font-weight:600;">{icon} {prob:.0%}</span>
        </div>"""

    # Double for seamless loop
    st.markdown(f"""
    <div class="ticker-wrap">
      <div class="ticker-inner">{items_html}{items_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ── macro environment bar ──────────────────────────────────────────────────────

def _render_macro_bar():
    macro = _load_macro_env()
    if not macro:
        return

    def _change_html(val, neutral=0, invert=False, fmt=".2f", suffix=""):
        if val is None:
            return ""
        diff = val - neutral
        if invert:
            diff = -diff
        col = C["up"] if diff > 0 else (C["down"] if diff < 0 else C["stable"])
        sign = "+" if diff > 0 else ""
        return f'<span style="color:{col}">{sign}{diff:{fmt}}{suffix}</span>'

    vix = macro.get("vix") or 0
    vix_regime = "HIGH FEAR" if vix > 30 else ("CAUTION" if vix > 20 else "CALM")
    vix_col = C["down"] if vix > 30 else (C["gold"] if vix > 20 else C["up"])

    dxy = macro.get("dxy") or 0
    t10y = macro.get("t10y") or 0
    ff = macro.get("fedfunds") or 0
    yield_inv = t10y < ff
    spread = t10y - ff

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-bottom:20px;">
      <div class="macro-item">
        <div class="macro-label">USD Index (DXY)</div>
        <div class="macro-value">{dxy:.1f}</div>
        <div class="macro-change" style="color:{C['text3']}">Broad USD Strength</div>
      </div>
      <div class="macro-item">
        <div class="macro-label">VIX Volatility</div>
        <div class="macro-value" style="color:{vix_col}">{vix:.1f}</div>
        <div class="macro-change" style="color:{vix_col}">{vix_regime}</div>
      </div>
      <div class="macro-item">
        <div class="macro-label">10Y Treasury</div>
        <div class="macro-value">{t10y:.2f}%</div>
        <div class="macro-change" style="color:{C['text3']}">US Yield</div>
      </div>
      <div class="macro-item">
        <div class="macro-label">Fed Funds</div>
        <div class="macro-value">{ff:.2f}%</div>
        <div class="macro-change" style="color:{C['text3']}">Policy Rate</div>
      </div>
      <div class="macro-item">
        <div class="macro-label">Yield Spread</div>
        <div class="macro-value" style="color:{C['down'] if yield_inv else C['up']}">{spread:+.2f}%</div>
        <div class="macro-change" style="color:{C['down'] if yield_inv else C['text3']}">
          {'⚠ INVERTED' if yield_inv else 'Normal'}
        </div>
      </div>
      <div class="macro-item">
        <div class="macro-label">Copper 3M Trend</div>
        <div class="macro-value" style="color:{C['up'] if (macro.get('copper_basis') or 0) > 0 else C['down']}">
          {(macro.get('copper_basis') or 0):+.1f}%
        </div>
        <div class="macro-change" style="color:{C['text3']}">Industrial Demand</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── commodity grid ─────────────────────────────────────────────────────────────

def _render_commodity_grid(forecasts: dict, horizon_key: str, active_sym: str) -> str | None:
    fk = "forecast_7d" if horizon_key == "7d" else "forecast_30d"

    st.markdown(f"""
    <div class="section-header">
      <div class="section-dot"></div>
      <div class="section-title">Market Overview — {horizon_key.upper()} Forecast</div>
    </div>
    """, unsafe_allow_html=True)

    clicked = None
    rows = [ALL_SYMBOLS[i:i+5] for i in range(0, len(ALL_SYMBOLS), 5)]

    for row_syms in rows:
        cols = st.columns(len(row_syms))
        for col, sym in zip(cols, row_syms):
            fc = forecasts.get(sym, {})
            f  = fc.get(fk, {}) if fc and "error" not in fc else {}
            dir_  = f.get("direction", "STABLE")
            conf  = f.get("confidence", "LOW")
            prob  = f.get("probability", 0.5)
            price = fc.get("current_price", 0) if fc else 0
            name  = SYMBOL_NAMES.get(sym, sym)
            icon  = DIR_ICON.get(dir_, "◆")
            dcol  = DIR_COLOR.get(dir_, C["stable"])
            ddim  = DIR_DICT = DIR_DIM.get(dir_, C["stable_dim"])
            ccol  = CONF_COLOR.get(conf, C["conf_low"])
            is_active = sym == active_sym
            warn  = fc.get("forecast_7d", {}).get("model_warning") if fc else None

            with col:
                st.markdown(f"""
                <div class="comm-card {dir_.lower()} {'active' if is_active else ''}"
                     style="background:linear-gradient(145deg,{C['surface']} 0%,{ddim} 100%);">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
                    <div>
                      <div style="font-size:0.65rem;color:{C['text3']};letter-spacing:0.08em;font-family:'JetBrains Mono',monospace;">{sym}</div>
                      <div style="font-size:0.88rem;font-weight:600;color:{C['text']};margin-top:1px;">{name}</div>
                    </div>
                    <div style="background:{ccol}22;border:1px solid {ccol}44;border-radius:4px;
                                padding:2px 6px;font-size:0.6rem;font-weight:700;color:{ccol};
                                letter-spacing:0.06em;">{conf}</div>
                  </div>
                  <div style="font-size:1.05rem;font-weight:600;color:{C['text']};
                              font-family:'JetBrains Mono',monospace;margin-bottom:6px;">
                    ${price:,.2f}
                  </div>
                  <div style="display:flex;align-items:center;gap:6px;">
                    <span style="font-size:1.5rem;color:{dcol};font-weight:700;line-height:1;">{icon}</span>
                    <div>
                      <div style="font-size:0.82rem;color:{dcol};font-weight:600;">{dir_}</div>
                      <div style="font-size:0.65rem;color:{C['text3']};">{prob:.0%} probability</div>
                    </div>
                  </div>
                  {'<div style="margin-top:6px;font-size:0.62rem;color:' + C["gold"] + ';background:' + C["gold"] + '15;border-radius:3px;padding:2px 5px;">⚠ Use 30d model</div>' if warn and horizon_key == "7d" else ''}
                </div>
                """, unsafe_allow_html=True)

                if st.button("Analyze →", key=f"btn_{sym}", use_container_width=True):
                    clicked = sym

    return clicked


# ── deep dive ──────────────────────────────────────────────────────────────────

def _price_chart(symbol: str, days: int, fc: dict, horizon_key: str):
    df = _load_price_history(symbol, days)
    if df.empty:
        st.info("No price history — run the price collector.")
        return

    fk    = "forecast_7d" if horizon_key == "7d" else "forecast_30d"
    fcast = fc.get(fk, {})
    dir_  = fcast.get("direction", "STABLE")
    dcol  = DIR_COLOR.get(dir_, C["stable"])
    low   = fcast.get("price_range_low")
    high  = fcast.get("price_range_high")

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing=dict(line=dict(color=C["up"], width=1), fillcolor=C["up_dim"]),
        decreasing=dict(line=dict(color=C["down"], width=1), fillcolor=C["down_dim"]),
    ))

    # 20-day SMA
    df["sma20"] = df["close"].rolling(20, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["sma20"], mode="lines",
        line=dict(color=C["accent"], width=1.2, dash="dot"),
        name="SMA 20", opacity=0.6,
    ))

    # Forecast zone
    if low and high and not df.empty:
        last_date = pd.to_datetime(df["date"].max())
        fwd = last_date + timedelta(days=7 if horizon_key == "7d" else 30)
        fig.add_shape(type="rect",
            x0=str(last_date.date()), x1=str(fwd.date()),
            y0=low, y1=high,
            fillcolor=dcol, opacity=0.10,
            line=dict(color=dcol, width=1, dash="dot"),
        )
        fig.add_annotation(
            x=str(fwd.date()), y=(low + high) / 2,
            text=f"  {DIR_ICON.get(dir_,'')} {dir_} {fcast.get('probability',0):.0%}",
            showarrow=False, font=dict(color=dcol, size=11, family="JetBrains Mono"),
            bgcolor=C["surface2"], bordercolor=dcol,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        xaxis_rangeslider_visible=False,
        height=360,
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(orientation="h", x=0, y=1.06, font=dict(size=10, color=C["text2"])),
        xaxis=dict(gridcolor=C["surface2"], showgrid=True),
        yaxis=dict(gridcolor=C["surface2"], showgrid=True),
        font=dict(family="Inter", color=C["text2"]),
    )
    st.plotly_chart(fig, use_container_width=True)


def _shap_chart(fc: dict):
    signals = fc.get("top_signals", [])
    if not signals:
        st.caption("No SHAP signals — retrain models to enable.")
        return

    labels  = [s.get("label", s.get("feature", ""))[:32] for s in signals]
    weights = [s["weight"] if s["impact"] == "BULLISH" else -s["weight"] for s in signals]
    colors  = [C["up"] if w > 0 else C["down"] for w in weights]

    fig = go.Figure(go.Bar(
        x=weights, y=labels, orientation="h",
        marker=dict(color=colors, opacity=0.85),
        text=[f"{'▲' if w>0 else '▼'} {abs(w):.3f}" for w in weights],
        textposition="outside", textfont=dict(size=10, family="JetBrains Mono", color=C["text2"]),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        title=dict(text="Top Signal Drivers (SHAP)", font=dict(size=11, color=C["text2"])),
        xaxis=dict(gridcolor=C["surface2"], zeroline=True, zerolinecolor=C["border_hi"],
                   showticklabels=False),
        yaxis=dict(gridcolor="transparent"),
        height=260, margin=dict(l=0, r=40, t=32, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _cot_chart(symbol: str):
    df = _load_cot_history(symbol)
    if df.empty:
        st.caption("No COT data for this symbol.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["commercial_net_pct"] * 100,
        mode="lines", fill="tozeroy",
        line=dict(color=C["up"], width=1.5),
        fillcolor="rgba(0,217,126,0.08)",
        name="Commercial (Smart $)",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["mm_net_pct"] * 100,
        mode="lines", fill="tozeroy",
        line=dict(color=C["accent"], width=1.5),
        fillcolor="rgba(61,127,255,0.08)",
        name="Managed Money",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=C["border_hi"], line_width=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        title=dict(text="COT Positioning — % of Open Interest", font=dict(size=11, color=C["text2"])),
        xaxis=dict(gridcolor=C["surface2"]),
        yaxis=dict(gridcolor=C["surface2"], ticksuffix="%"),
        height=220, margin=dict(l=0, r=0, t=32, b=0),
        legend=dict(orientation="h", x=0, y=1.12, font=dict(size=10)),
        font=dict(family="Inter", color=C["text2"]),
    )
    st.plotly_chart(fig, use_container_width=True)


def _sentiment_chart(symbol: str):
    df = _load_sentiment_history(symbol)
    if df.empty:
        st.caption("No sentiment data — run the NLP processor.")
        return

    colors = [C["up"] if float(s) > 0.1 else (C["down"] if float(s) < -0.1 else C["stable"])
              for s in df["sentiment_score"].fillna(0)]

    fig = go.Figure()
    fig.add_hrect(y0=0.1, y1=1, fillcolor=C["up_dim"], opacity=1, line_width=0)
    fig.add_hrect(y0=-1, y1=-0.1, fillcolor=C["down_dim"], opacity=1, line_width=0)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["sentiment_score"],
        mode="lines+markers",
        line=dict(color=C["text2"], width=1.5),
        marker=dict(color=colors, size=5),
        fill="tozeroy", fillcolor="rgba(139,148,158,0.06)",
        name="Sentiment",
    ))
    fig.add_hline(y=0, line_dash="solid", line_color=C["border_hi"], line_width=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        title=dict(text="News Sentiment (60-day)", font=dict(size=11, color=C["text2"])),
        yaxis=dict(range=[-1, 1], gridcolor=C["surface2"], tickformat=".1f"),
        xaxis=dict(gridcolor=C["surface2"]),
        height=200, margin=dict(l=0, r=0, t=32, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _eia_chart(symbol: str):
    series = {"CL=F": "crude_stocks", "NG=F": "natgas_storage"}.get(symbol)
    if not series:
        return
    df = _load_eia_history(series)
    if df.empty:
        return

    label = "Crude Oil Stocks (Mbbls)" if symbol == "CL=F" else "Natural Gas Storage (Bcf)"
    div = 1000 if symbol == "CL=F" else 1

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=df["value"] / div,
        name=label,
        marker=dict(
            color=[C["down_dim"] if (v or 0) > 0 else C["up_dim"] for v in df.get("chg_1w", [])],
            line=dict(width=0),
        ),
        opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=(df["value"] / div).rolling(4).mean(),
        mode="lines", line=dict(color=C["accent"], width=1.5, dash="dot"),
        name="4-wk avg",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        title=dict(text=label, font=dict(size=11, color=C["text2"])),
        height=200, margin=dict(l=0, r=0, t=32, b=0),
        legend=dict(orientation="h", x=0, y=1.15, font=dict(size=10)),
        xaxis=dict(gridcolor=C["surface2"]),
        yaxis=dict(gridcolor=C["surface2"]),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_deep_dive(symbol: str, days: int, horizon_key: str):
    fc   = _load_forecast(symbol)
    name = SYMBOL_NAMES.get(symbol, symbol)

    if "error" in fc:
        st.warning(f"No forecast for {name} — run `python model/trainer.py --symbol {symbol}`")
        return

    fk     = "forecast_7d" if horizon_key == "7d" else "forecast_30d"
    fcast  = fc.get(fk, {})
    dir_   = fcast.get("direction", "STABLE")
    prob   = fcast.get("probability", 0.5)
    conf   = fcast.get("confidence", "LOW")
    price  = fc.get("current_price", 0)
    dcol   = DIR_COLOR.get(dir_, C["stable"])
    ddim   = DIR_DIM.get(dir_, C["stable_dim"])
    icon   = DIR_ICON.get(dir_, "◆")
    ccol   = CONF_COLOR.get(conf, C["conf_low"])
    warn   = fcast.get("model_warning")

    # Breadcrumb + headline
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
      <div style="font-size:0.7rem;color:{C['text3']};letter-spacing:0.08em;">ANALYSIS</div>
      <div style="font-size:0.7rem;color:{C['text3']};">›</div>
      <div style="font-size:0.85rem;font-weight:600;color:{C['text']};">{name}</div>
      <div style="font-size:0.65rem;color:{C['text3']};font-family:'JetBrains Mono',monospace;">{symbol}</div>
    </div>
    <div style="display:flex;align-items:center;gap:16px;padding:18px 20px;
                background:linear-gradient(135deg,{C['surface']} 0%,{ddim} 100%);
                border:1px solid {dcol}44;border-radius:12px;margin-bottom:16px;">
      <div style="font-size:3rem;color:{dcol};line-height:1;">{icon}</div>
      <div>
        <div style="font-size:1.9rem;font-weight:700;color:{C['text']};font-family:'JetBrains Mono',monospace;">
          ${price:,.2f}
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
          <span style="font-size:1.1rem;font-weight:700;color:{dcol};">{dir_}</span>
          <span style="background:{ccol}22;border:1px solid {ccol}55;color:{ccol};
                       font-size:0.72rem;font-weight:700;padding:3px 8px;border-radius:20px;">
            {conf} CONF
          </span>
          <span style="font-size:0.85rem;color:{C['text2']};">{prob:.1%} probability · {horizon_key.upper()}</span>
        </div>
        {f'<div style="margin-top:6px;font-size:0.72rem;color:{C["gold"]};background:{C["gold"]}18;padding:4px 10px;border-radius:6px;display:inline-block;">⚠ {warn}</div>' if warn else ''}
      </div>
      {f'''<div style="margin-left:auto;text-align:right;">
        <div style="font-size:0.65rem;color:{C["text3"]};text-transform:uppercase;letter-spacing:0.1em;">Price Target Range</div>
        <div style="font-size:1.1rem;font-weight:600;color:{C["text"]};font-family:'JetBrains Mono',monospace;">
          ${fcast.get("price_range_low",0):,.0f} – ${fcast.get("price_range_high",0):,.0f}
        </div>
      </div>''' if fcast.get("price_range_low") else ''}
    </div>
    """, unsafe_allow_html=True)

    # Main layout: chart | signals
    chart_col, signal_col = st.columns([3, 2])

    with chart_col:
        st.markdown(f'<div class="section-header"><div class="section-dot"></div><div class="section-title">Price Chart</div></div>', unsafe_allow_html=True)
        _price_chart(symbol, days, fc, horizon_key)

    with signal_col:
        st.markdown(f'<div class="section-header"><div class="section-dot"></div><div class="section-title">Signal Drivers</div></div>', unsafe_allow_html=True)
        _shap_chart(fc)

        # Both 7d and 30d forecast side by side
        f7  = fc.get("forecast_7d", {})
        f30 = fc.get("forecast_30d", {})
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px;">
          <div style="background:{C['surface2']};border:1px solid {C['border']};border-radius:8px;padding:10px;text-align:center;">
            <div style="font-size:0.6rem;color:{C['text3']};letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">7-Day</div>
            <div style="font-size:1.1rem;font-weight:700;color:{DIR_COLOR.get(f7.get('direction','STABLE'),C['stable'])};">
              {DIR_ICON.get(f7.get('direction','STABLE'),'◆')} {f7.get('direction','—')}
            </div>
            <div style="font-size:0.7rem;color:{C['text3']};">{f7.get('probability',0):.0%}</div>
          </div>
          <div style="background:{C['surface2']};border:1px solid {C['border']};border-radius:8px;padding:10px;text-align:center;">
            <div style="font-size:0.6rem;color:{C['text3']};letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">30-Day</div>
            <div style="font-size:1.1rem;font-weight:700;color:{DIR_COLOR.get(f30.get('direction','STABLE'),C['stable'])};">
              {DIR_ICON.get(f30.get('direction','STABLE'),'◆')} {f30.get('direction','—')}
            </div>
            <div style="font-size:0.7rem;color:{C['text3']};">{f30.get('probability',0):.0%}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Tabbed data panels
    tab_labels = ["COT Positioning", "Sentiment", "EIA Inventory", "Weather", "AI Report"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _cot_chart(symbol)

    with tabs[1]:
        _sentiment_chart(symbol)

    with tabs[2]:
        _eia_chart(symbol)
        if symbol not in ("CL=F", "NG=F"):
            st.caption("EIA inventory data is available for Crude Oil (CL=F) and Natural Gas (NG=F) only.")

    with tabs[3]:
        weather = _load_weather(symbol)
        if weather and weather.get("drought_index", 0) > 0:
            w1, w2, w3 = st.columns(3)
            w1.metric("Drought Index", f"{weather['drought_index']:.2f}", help="0=normal, 1=extreme drought")
            w2.metric("Heat Stress Days", weather["heat_stress_days"])
            w3.metric("Precip Anomaly", f"{weather['precip_anomaly_pct']:+.1f}%")
        else:
            st.caption("No weather data available. Weather signals apply to agricultural commodities.")

    with tabs[4]:
        reports = load_latest_reports()
        report = reports.get(symbol)
        if not report:
            with st.spinner("Generating AI analysis..."):
                report = generate_report(fc)

        if report and isinstance(report, dict):
            # Header
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;'
                f'padding-bottom:8px;border-bottom:1px solid {C["border"]};">'
                f'<span style="font-size:1rem;">🤖</span>'
                f'<span style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;'
                f'text-transform:uppercase;color:{C["text2"]};">AI Analyst Report</span>'
                f'<span style="font-size:0.6rem;color:{C["text3"]};margin-left:auto;">'
                f'Powered by Llama 3.3 70B · Groq</span></div>',
                unsafe_allow_html=True,
            )
            # One st.markdown per section — avoids nested HTML escaping
            for key, title, color, icon in [
                ("outlook",     "Market Outlook", C["accent"], "◈"),
                ("key_drivers", "Key Drivers",    C["up"],     "▲"),
                ("risk",        "Primary Risk",   C["down"],   "⚠"),
                ("trade_idea",  "Trade Idea",     C["gold"],   "◎"),
            ]:
                text = report.get(key, "")
                if not text:
                    continue
                st.markdown(
                    f'<div style="margin-bottom:12px;padding:14px 18px;'
                    f'background:{C["surface2"]};border-radius:10px;'
                    f'border-left:3px solid {color};">'
                    f'<div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;'
                    f'text-transform:uppercase;color:{color};margin-bottom:6px;">'
                    f'{icon}&nbsp; {title}</div>'
                    f'<div style="font-size:0.88rem;color:{C["text"]};line-height:1.65;">'
                    f'{text}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("AI report unavailable — set GROQ_API_KEY in your environment.")


# ── news feed ──────────────────────────────────────────────────────────────────

def _render_news(symbol: str):
    name = SYMBOL_NAMES.get(symbol, symbol)
    df   = _load_recent_news(symbol)
    source_label = df["source"].iloc[0] if not df.empty and "source" in df.columns else "GDELT"
    live_badge = (
        f'<span style="background:{C["up"]}22;color:{C["up"]};border-radius:4px;'
        f'padding:2px 7px;font-size:0.6rem;font-weight:700;letter-spacing:0.06em;'
        f'margin-left:8px;">LIVE</span>'
        if source_label == "Yahoo Finance" else ""
    )

    st.markdown(
        f'<div class="section-header" style="margin-top:8px;">'
        f'<div class="section-dot"></div>'
        f'<div class="section-title">News — {name}{live_badge}</div>'
        f'<div style="font-size:0.6rem;color:{C["text3"]};margin-left:auto;">'
        f'{source_label}</div></div>',
        unsafe_allow_html=True,
    )

    if df.empty:
        st.caption("No news available for this commodity.")
        return

    for _, row in df.iterrows():
        score = float(row.get("sentiment_score") or 0)
        has_score = abs(score) > 0.01
        scol  = C["up"] if score > 0.1 else (C["down"] if score < -0.1 else C["stable"])
        sign  = "+" if score > 0 else ""
        title = str(row.get("title", ""))[:130]
        url   = str(row.get("url", "#"))
        pub   = str(row.get("published_date", ""))[:16]

        score_html = (
            f'<span style="background:{scol}22;color:{scol};border-radius:4px;'
            f'padding:2px 6px;font-size:0.68rem;font-weight:600;'
            f'font-family:\'JetBrains Mono\',monospace;">{sign}{score:.2f}</span>'
            if has_score else
            f'<span style="color:{C["text3"]};font-size:0.68rem;">—</span>'
        )
        st.markdown(
            f'<div class="news-row">'
            f'<div style="min-width:90px;font-size:0.68rem;color:{C["text3"]};'
            f'font-family:\'JetBrains Mono\',monospace;padding-top:1px;">{pub}</div>'
            f'<div style="min-width:44px;text-align:center;">{score_html}</div>'
            f'<div style="flex:1;font-size:0.84rem;color:{C["text"]};">'
            f'<a href="{url}" target="_blank" style="color:{C["text"]};text-decoration:none;"'
            f' onmouseover="this.style.color=\'{C["accent"]}\'"'
            f' onmouseout="this.style.color=\'{C["text"]}\'">{title}</a>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


# ── sidebar controls ───────────────────────────────────────────────────────────

def _render_sidebar() -> tuple[str, int]:
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:12px 0 16px 0;border-bottom:1px solid {C['border']};margin-bottom:16px;">
          <div style="font-size:1.1rem;font-weight:700;
                      background:linear-gradient(135deg,{C['text']} 0%,{C['accent']} 100%);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            ◈ CommodiSense
          </div>
          <div style="font-size:0.65rem;color:{C['text3']};margin-top:3px;letter-spacing:0.06em;">
            COMMODITY INTELLIGENCE
          </div>
        </div>
        """, unsafe_allow_html=True)

        horizon = st.radio("Forecast Horizon", ["7d", "30d"], index=0,
                           format_func=lambda x: "7-Day" if x == "7d" else "30-Day")

        days = st.slider("Chart History", 30, 365, 90, step=15,
                         format="%d days")

        st.markdown("---")

        if st.button("↺ Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown(f"""
        <div style="margin-top:16px;">
          <div style="font-size:0.65rem;color:{C['text3']};letter-spacing:0.08em;
                      text-transform:uppercase;margin-bottom:10px;">Data Sources</div>
        """, unsafe_allow_html=True)

        sources = [
            ("Prices", "yfinance", "12,613 rows"),
            ("COT", "CFTC", "8,826 rows"),
            ("Macro", "FRED", "7,193 rows"),
            ("EIA", "DOE", "3,134 rows"),
            ("USDA", "NASS", "1,104 rows"),
            ("News", "GDELT", "392 articles"),
            ("Weather", "Open-Meteo", "210 rows"),
        ]
        for name, src, count in sources:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:5px 0;border-bottom:1px solid {C['border']};">
              <div style="font-size:0.72rem;color:{C['text2']};font-weight:500;">{name}</div>
              <div style="text-align:right;">
                <div style="font-size:0.62rem;color:{C['text3']};font-family:'JetBrains Mono',monospace;">{count}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:20px;padding:10px;background:{C['surface']};
                    border:1px solid {C['border']};border-radius:8px;font-size:0.65rem;
                    color:{C['text3']};line-height:1.6;">
          <div style="color:{C['text2']};font-weight:600;margin-bottom:4px;">Pipeline</div>
          GitHub Actions · Mon–Fri 06:00 UTC<br>
          XGBoost + LightGBM ensemble<br>
          SHAP explainability · FinBERT NLP
        </div>
        """, unsafe_allow_html=True)

    return horizon, days


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    _ensure_schema()
    _ensure_prices()
    _retag_news()
    _inject_css()
    _render_header()

    horizon, days = _render_sidebar()

    # Load all forecasts at once
    forecasts = _load_all_forecasts(tuple(ALL_SYMBOLS))

    # Ticker strip
    _render_ticker(forecasts, horizon)

    # Macro environment
    _render_macro_bar()

    # Commodity grid — track active symbol in session state
    clicked = _render_commodity_grid(forecasts, horizon,
                                     st.session_state.get("active_sym", ALL_SYMBOLS[0]))

    if clicked:
        st.session_state["active_sym"] = clicked

    active = st.session_state.get("active_sym")
    if not active:
        active = ALL_SYMBOLS[0]
        st.session_state["active_sym"] = active

    # Divider
    st.markdown(f'<div style="height:1px;background:linear-gradient(90deg,transparent,{C["border_hi"]},transparent);margin:20px 0;"></div>', unsafe_allow_html=True)

    # Deep dive
    _render_deep_dive(active, days, horizon)

    # News
    st.markdown(f'<div style="height:1px;background:linear-gradient(90deg,transparent,{C["border_hi"]},transparent);margin:20px 0 16px;"></div>', unsafe_allow_html=True)
    _render_news(active)


if __name__ == "__main__":
    main()

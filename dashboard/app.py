"""
CommodiSense Dashboard — Streamlit app serving commodity forecasts,
signal breakdowns, AI reports, and recent news.

Run locally:
    streamlit run dashboard/app.py

Deploy: Streamlit Cloud → connect GitHub repo → set main file: dashboard/app.py
        Add secret GROQ_API_KEY in Streamlit Cloud settings.
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make project root importable whether launched from repo root or dashboard/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.db import get_conn, init_schema
from model.explainer import load_latest_reports, generate_report
from model.predictor import predict, SYMBOL_NAMES

# ── page config (must be first Streamlit call) ─────────────────────────────────

st.set_page_config(
    page_title="CommodiSense",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ──────────────────────────────────────────────────────────────────

ALL_SYMBOLS  = list(SYMBOL_NAMES.keys())
COLOR_UP     = "#00C896"
COLOR_DOWN   = "#FF4B4B"
COLOR_STABLE = "#888888"
COLOR_BG     = "#0E1117"

DIRECTION_ARROW = {"UP": "↑", "DOWN": "↓", "STABLE": "→"}
DIRECTION_COLOR = {"UP": COLOR_UP, "DOWN": COLOR_DOWN, "STABLE": COLOR_STABLE}

CONF_COLOR = {"HIGH": "#00C896", "MEDIUM": "#F5A623", "LOW": "#888888"}

# ── caching ────────────────────────────────────────────────────────────────────


@st.cache_resource
def _ensure_schema():
    """Initialize DB schema once per process."""
    init_schema()


@st.cache_data(ttl=3600)
def _load_forecast(symbol: str) -> dict:
    """Cache forecast for 1 hour — expensive ML inference."""
    return predict(symbol)


@st.cache_data(ttl=3600)
def _load_price_history(symbol: str, days: int = 90) -> pd.DataFrame:
    cutoff = date.today() - timedelta(days=days)
    conn = get_conn()
    df = conn.execute(
        "SELECT date, open, high, low, close FROM prices WHERE symbol = ? AND date >= ? ORDER BY date",
        [symbol, cutoff],
    ).df()
    conn.close()
    return df


@st.cache_data(ttl=3600)
def _load_sentiment_history(symbol: str, days: int = 30) -> pd.DataFrame:
    cutoff = date.today() - timedelta(days=days)
    conn = get_conn()
    df = conn.execute(
        "SELECT date, sentiment_score, article_count FROM sentiment_daily "
        "WHERE commodity = ? AND date >= ? ORDER BY date",
        [symbol, cutoff],
    ).df()
    conn.close()
    return df


@st.cache_data(ttl=3600)
def _load_recent_news(symbol: str, limit: int = 20) -> pd.DataFrame:
    conn = get_conn()
    df = conn.execute(
        """
        SELECT published_date, title, url, sentiment_score, commodity_tags
        FROM news_raw
        WHERE commodity_tags LIKE ?
        ORDER BY published_date DESC
        LIMIT ?
        """,
        [f"%{symbol}%", limit],
    ).df()
    conn.close()
    return df


@st.cache_data(ttl=3600)
def _load_weather_latest(symbol: str) -> dict:
    from signals.weather_features import get_weather_features
    return get_weather_features(symbol, days=30)


@st.cache_data(ttl=3600)
def _load_all_forecasts_cached(symbols_tuple: tuple) -> dict:
    """Load forecasts for a fixed set of symbols (tuple for hashability)."""
    return {sym: _load_forecast(sym) for sym in symbols_tuple}


# ── helpers ────────────────────────────────────────────────────────────────────


def _direction_badge(direction: str, confidence: str) -> str:
    arrow  = DIRECTION_ARROW.get(direction, "→")
    color  = DIRECTION_COLOR.get(direction, COLOR_STABLE)
    ccolor = CONF_COLOR.get(confidence, COLOR_STABLE)
    return (
        f'<span style="color:{color};font-size:1.4em;font-weight:bold">{arrow}</span>'
        f' <span style="background:{ccolor};color:#000;border-radius:4px;'
        f'padding:2px 6px;font-size:0.75em;font-weight:600">{confidence}</span>'
    )


def _sentiment_color(score: float) -> str:
    if score > 0.1:
        return COLOR_UP
    if score < -0.1:
        return COLOR_DOWN
    return COLOR_STABLE


# ── sidebar ────────────────────────────────────────────────────────────────────


def _render_sidebar() -> tuple[list[str], str, int, bool]:
    st.sidebar.title("⚙️ Controls")

    selected = st.sidebar.multiselect(
        "Commodities",
        options=ALL_SYMBOLS,
        default=ALL_SYMBOLS,
        format_func=lambda s: SYMBOL_NAMES.get(s, s),
    )
    if not selected:
        selected = ALL_SYMBOLS

    horizon = st.sidebar.radio("Forecast horizon", ["7-day", "30-day"], index=0)

    history_days = st.sidebar.slider("Price chart (days)", 30, 365, 90, step=15)

    refresh = st.sidebar.button("🔄 Refresh data", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last update: {date.today().isoformat()}")
    st.sidebar.caption("Data: yfinance · GDELT · Open-Meteo · ACLED")

    return selected, horizon, history_days, refresh


# ── Section 1: Market Overview ─────────────────────────────────────────────────


def _render_overview(selected_symbols: list[str], horizon_key: str) -> str | None:
    """
    Render a grid of metric cards for all selected commodities.
    Returns the symbol the user clicked to deep-dive, or None.
    """
    st.subheader("Market Overview")

    forecasts = _load_all_forecasts_cached(tuple(selected_symbols))

    # 5 columns per row
    cols_per_row = 5
    rows = [selected_symbols[i : i + cols_per_row]
            for i in range(0, len(selected_symbols), cols_per_row)]

    clicked = None
    for row_syms in rows:
        cols = st.columns(len(row_syms))
        for col, sym in zip(cols, row_syms):
            fc = forecasts.get(sym, {})
            if "error" in fc:
                with col:
                    st.error(f"{SYMBOL_NAMES.get(sym, sym)}\nNo model")
                continue

            fc_key  = "forecast_7d" if horizon_key == "7-day" else "forecast_30d"
            fcast   = fc.get(fc_key, {})
            direction  = fcast.get("direction", "STABLE")
            confidence = fcast.get("confidence", "LOW")
            prob       = fcast.get("probability", 0.5)
            price      = fc.get("current_price", 0)
            name       = SYMBOL_NAMES.get(sym, sym)

            arrow  = DIRECTION_ARROW.get(direction, "→")
            dcolor = DIRECTION_COLOR.get(direction, COLOR_STABLE)
            ccolor = CONF_COLOR.get(confidence, "#888")

            bg = {"UP": "#0a2e1a", "DOWN": "#2e0a0a", "STABLE": "#1a1a1a"}.get(direction, "#1a1a1a")

            with col:
                st.markdown(
                    f"""
                    <div style="background:{bg};border-radius:8px;padding:12px 10px;
                                border:1px solid {dcolor}33;margin-bottom:4px;">
                      <div style="font-size:0.75em;color:#aaa;margin-bottom:2px">{sym}</div>
                      <div style="font-size:1.0em;font-weight:600;color:#fff">{name}</div>
                      <div style="font-size:1.3em;color:#ddd;margin:4px 0">${price:,.2f}</div>
                      <div style="font-size:1.6em;color:{dcolor};font-weight:700">{arrow}
                        <span style="font-size:0.6em;color:{ccolor};background:{ccolor}22;
                                     border-radius:3px;padding:1px 5px">{confidence}</span>
                      </div>
                      <div style="font-size:0.7em;color:#888">{prob:.0%} conf · {horizon_key}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Deep dive →", key=f"btn_{sym}", use_container_width=True):
                    clicked = sym

    return clicked


# ── Section 2: Deep Dive ───────────────────────────────────────────────────────


def _render_price_chart(symbol: str, history_days: int, fc: dict, horizon_key: str):
    df = _load_price_history(symbol, history_days)
    if df.empty:
        st.info("No price history in database. Run the price collector first.")
        return

    fc_key  = "forecast_7d" if horizon_key == "7-day" else "forecast_30d"
    fcast   = fc.get(fc_key, {})
    direction = fcast.get("direction", "STABLE")
    low  = fcast.get("price_range_low", None)
    high = fcast.get("price_range_high", None)
    dcolor = DIRECTION_COLOR.get(direction, COLOR_STABLE)

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name="Price",
        increasing_line_color=COLOR_UP,
        decreasing_line_color=COLOR_DOWN,
    ))

    # Forecast range band (extend 7 or 30 days from last date)
    if low and high and not df.empty:
        last_date = pd.to_datetime(df["date"].max())
        days_fwd  = 7 if horizon_key == "7-day" else 30
        fwd_date  = last_date + timedelta(days=days_fwd)
        last_close = float(df["close"].iloc[-1])
        fig.add_shape(type="rect",
            x0=str(last_date.date()), x1=str(fwd_date.date()),
            y0=low, y1=high,
            fillcolor=dcolor, opacity=0.12,
            line=dict(color=dcolor, width=1, dash="dot"),
        )
        fig.add_annotation(
            x=str(fwd_date.date()), y=(low + high) / 2,
            text=f"{direction} {fcast.get('probability', 0):.0%}",
            showarrow=False, font=dict(color=dcolor, size=11),
            bgcolor="#0E1117", bordercolor=dcolor,
        )

    name = SYMBOL_NAMES.get(symbol, symbol)
    fig.update_layout(
        template="plotly_dark",
        title=f"{name} — last {history_days} days",
        xaxis_rangeslider_visible=False,
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_shap_chart(fc: dict):
    signals = fc.get("top_signals", [])
    if not signals:
        st.info("Run model/trainer.py to enable SHAP signal breakdown.")
        return

    labels  = [s.get("label", s.get("feature", ""))[:30] for s in signals]
    weights = [s["weight"] if s["impact"] == "BULLISH" else -s["weight"] for s in signals]
    colors  = [COLOR_UP if w > 0 else COLOR_DOWN for w in weights]

    fig = go.Figure(go.Bar(
        x=weights,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{abs(w):.3f}" for w in weights],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Top Signal Drivers (SHAP)",
        xaxis_title="Impact (+ bullish, − bearish)",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_sentiment_chart(symbol: str):
    df = _load_sentiment_history(symbol, days=30)
    if df.empty:
        st.caption("No sentiment data yet — run the NLP sentiment processor.")
        return

    colors = [_sentiment_color(float(s)) for s in df["sentiment_score"].fillna(0)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["sentiment_score"],
        mode="lines+markers",
        line=dict(color="#5B9BD5", width=2),
        marker=dict(color=colors, size=6),
        name="Sentiment",
        fill="tozeroy",
        fillcolor="rgba(91,155,213,0.1)",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#555")
    fig.update_layout(
        template="plotly_dark",
        title="News Sentiment (30-day)",
        yaxis=dict(range=[-1, 1], tickformat=".1f"),
        height=220,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_deep_dive(symbol: str, history_days: int, horizon_key: str):
    fc   = _load_forecast(symbol)
    name = SYMBOL_NAMES.get(symbol, symbol)

    if "error" in fc:
        st.warning(f"No forecast for {name} — models not trained yet. "
                   "Run `python model/trainer.py` after backfilling data.")
        return

    fc_key    = "forecast_7d" if horizon_key == "7-day" else "forecast_30d"
    fcast     = fc.get(fc_key, {})
    direction = fcast.get("direction", "STABLE")
    prob      = fcast.get("probability", 0.5)
    confidence= fcast.get("confidence", "LOW")
    price     = fc.get("current_price", 0)

    dcolor = DIRECTION_COLOR.get(direction, COLOR_STABLE)
    arrow  = DIRECTION_ARROW.get(direction, "→")

    st.markdown(f"## {name} Deep Dive")

    # Headline metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${price:,.2f}")
    m2.metric(f"{horizon_key} Forecast",
              f"{arrow} {direction}",
              delta=f"{prob:.0%} conf",
              delta_color="normal" if direction == "UP" else ("inverse" if direction == "DOWN" else "off"))
    m3.metric("Confidence", confidence)
    if horizon_key == "7-day" and fcast.get("price_range_low"):
        m4.metric("Price Range",
                  f"${fcast['price_range_low']:,.0f} – ${fcast['price_range_high']:,.0f}")

    st.markdown("---")

    # Price chart + SHAP side by side
    c_chart, c_shap = st.columns([3, 2])
    with c_chart:
        _render_price_chart(symbol, history_days, fc, horizon_key)
    with c_shap:
        _render_shap_chart(fc)

    # AI Report
    reports = load_latest_reports()
    report_text = reports.get(symbol, "")
    if not report_text:
        # Generate on demand if no cached report
        with st.spinner("Generating AI report..."):
            report_text = generate_report(fc)

    st.markdown(
        f"""
        <div style="background:#161B22;border-left:3px solid {dcolor};
                    border-radius:6px;padding:14px 18px;margin:8px 0;">
          <div style="color:#888;font-size:0.75em;margin-bottom:6px">
            🤖 AI Analyst Report
          </div>
          <div style="color:#E6EDF3;font-size:0.95em;line-height:1.6">{report_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Weather + Sentiment
    w_col, s_col = st.columns(2)
    with w_col:
        st.markdown("**Weather Signals**")
        weather = _load_weather_latest(symbol)
        if weather and weather.get("drought_index", 0) > 0:
            w1, w2, w3 = st.columns(3)
            w1.metric("Drought Index", f"{weather['drought_index']:.2f}")
            w2.metric("Heat Stress Days", weather["heat_stress_days"])
            w3.metric("Precip Anomaly", f"{weather['precip_anomaly_pct']:+.1f}%")
        else:
            st.caption("No weather data — run the weather collector.")

    with s_col:
        _render_sentiment_chart(symbol)


# ── Section 3: Recent News ─────────────────────────────────────────────────────


def _render_news(symbol: str):
    st.markdown("---")
    st.subheader(f"Recent News — {SYMBOL_NAMES.get(symbol, symbol)}")
    df = _load_recent_news(symbol, limit=20)

    if df.empty:
        st.info("No news data yet — run the news collector.")
        return

    for _, row in df.iterrows():
        score = float(row.get("sentiment_score") or 0)
        sc    = _sentiment_color(score)
        title = str(row.get("title", ""))
        url   = str(row.get("url", "#"))
        pub   = str(row.get("published_date", ""))[:10]
        tags  = str(row.get("commodity_tags", ""))

        score_display = f'<span style="color:{sc};font-weight:600">{score:+.2f}</span>'

        st.markdown(
            f"""
            <div style="padding:6px 0;border-bottom:1px solid #21262d">
              <span style="color:#888;font-size:0.75em">{pub}</span>
              &nbsp;{score_display}&nbsp;
              <a href="{url}" target="_blank"
                 style="color:#58A6FF;text-decoration:none">{title[:120]}</a>
              <span style="color:#555;font-size:0.7em;margin-left:8px">{tags}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── main app ───────────────────────────────────────────────────────────────────


def main():
    _ensure_schema()

    # Custom CSS for dark polish
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; }
        div[data-testid="metric-container"] {
            background: #161B22;
            border: 1px solid #30363D;
            border-radius: 8px;
            padding: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📊 CommodiSense")
    st.caption("Global Commodity Intelligence Engine · zero cost · open source")

    selected, horizon, history_days, _ = _render_sidebar()

    # Section 1: Overview grid
    clicked_symbol = _render_overview(selected, horizon)

    # Determine which symbol to deep-dive
    # Use session state to persist selection across reruns
    if clicked_symbol:
        st.session_state["deep_dive_symbol"] = clicked_symbol

    deep_sym = st.session_state.get("deep_dive_symbol")

    # If nothing selected yet, default to first selected symbol
    if not deep_sym or deep_sym not in selected:
        deep_sym = selected[0] if selected else None
        if deep_sym:
            st.session_state["deep_dive_symbol"] = deep_sym

    # Section 2 + 3 only render if a symbol is active
    if deep_sym:
        st.markdown("---")
        _render_deep_dive(deep_sym, history_days, horizon)
        _render_news(deep_sym)


if __name__ == "__main__":
    main()

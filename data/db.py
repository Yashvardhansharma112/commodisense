"""
Shared DuckDB connection and schema initialization for CommodiSense.
All collectors import get_conn() from here.
"""

import duckdb
from pathlib import Path

DB_PATH = Path(__file__).parent / "commodisense.duckdb"


def get_conn() -> duckdb.DuckDBPyConnection:
    """Return a persistent DuckDB connection. Creates the DB file if missing."""
    return duckdb.connect(str(DB_PATH))


def init_schema() -> None:
    """Create all tables if they don't already exist. Safe to call repeatedly."""
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            date        DATE    NOT NULL,
            symbol      TEXT    NOT NULL,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            volume      DOUBLE,
            adj_close   DOUBLE,
            PRIMARY KEY (date, symbol)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_raw (
            id              TEXT PRIMARY KEY,
            source          TEXT,
            published_date  TIMESTAMP,
            title           TEXT,
            summary         TEXT,
            url             TEXT,
            commodity_tags  TEXT,
            sentiment_score DOUBLE,
            processed       BOOLEAN DEFAULT FALSE
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_features (
            date                DATE    NOT NULL,
            region              TEXT    NOT NULL,
            commodity           TEXT    NOT NULL,
            temp_max            DOUBLE,
            temp_min            DOUBLE,
            precipitation       DOUBLE,
            soil_moisture       DOUBLE,
            drought_index       DOUBLE,
            heat_stress_days    INTEGER,
            precip_anomaly_pct  DOUBLE,
            PRIMARY KEY (date, region, commodity)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS geopolitical_events (
            date        DATE    NOT NULL,
            event_type  TEXT,
            region      TEXT,
            commodity   TEXT,
            risk_score  DOUBLE,
            headline    TEXT,
            source      TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_daily (
            date            DATE    NOT NULL,
            commodity       TEXT    NOT NULL,
            sentiment_score DOUBLE,
            article_count   INTEGER,
            positive_count  INTEGER,
            negative_count  INTEGER,
            PRIMARY KEY (date, commodity)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS extracted_events (
            date        DATE,
            headline    TEXT,
            event_type  TEXT,
            commodity   TEXT,
            location    TEXT,
            severity    INTEGER,
            direction   TEXT,
            source      TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS accuracy_log (
            date                DATE,
            symbol              TEXT,
            forecast_direction  TEXT,
            actual_direction    TEXT,
            was_correct         BOOLEAN,
            confidence          TEXT
        )
    """)

    conn.close()


if __name__ == "__main__":
    init_schema()
    print(f"Schema initialized at {DB_PATH}")

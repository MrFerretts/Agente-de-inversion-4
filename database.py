"""
╔══════════════════════════════════════════════════════════════════╗
║         PATO QUANT — DATABASE MANAGER (SUPABASE)                ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import logging
from typing import List, Dict
import pandas as pd

logger = logging.getLogger("PatoQuant.DB")

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_engine = None

def get_engine():
    global _engine
    if _engine is not None:
        return _engine
    from sqlalchemy import create_engine
    url = os.getenv("SUPABASE_DB_URL", "")
    if not url:
        raise ValueError("SUPABASE_DB_URL no configurada")
    _engine = create_engine(url, pool_size=5, max_overflow=10,
                             pool_timeout=15, pool_pre_ping=True)
    return _engine


def query_df(sql: str, params: dict = None) -> pd.DataFrame:
    """SELECT → DataFrame. Sin warnings de pandas (usa SQLAlchemy)."""
    try:
        from sqlalchemy import text
        with get_engine().connect() as conn:
            return pd.read_sql_query(text(sql), conn, params=params or {})
    except Exception as e:
        logger.warning(f"⚠️ Query falló: {e}")
        return pd.DataFrame()


def execute(sql: str, params: tuple = ()):
    """INSERT/UPDATE/DELETE con psycopg2."""
    try:
        import psycopg2
        url = os.getenv("SUPABASE_DB_URL", "")
        conn = psycopg2.connect(url, connect_timeout=15)
        cur  = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"❌ Execute falló: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# TABLAS
# ─────────────────────────────────────────────────────────────────────────────

def init_tables():
    ddl = [
        """CREATE TABLE IF NOT EXISTS scan_results (
            id             SERIAL PRIMARY KEY,
            ticker         VARCHAR(20)  NOT NULL,
            timestamp      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            price          NUMERIC(12,4),
            change_pct     NUMERIC(8,4),
            score          NUMERIC(8,2),
            recommendation VARCHAR(30),
            rsi            NUMERIC(6,2),
            adx            NUMERIC(6,2),
            rvol           NUMERIC(6,3),
            macd           NUMERIC(10,6),
            bb_position    NUMERIC(6,4),
            volume         BIGINT,
            raw_data       JSONB
        )""",
        """CREATE TABLE IF NOT EXISTS alerts_sent (
            id          SERIAL PRIMARY KEY,
            ticker      VARCHAR(20)  NOT NULL,
            timestamp   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            alert_type  VARCHAR(50),
            message     TEXT,
            channel     VARCHAR(20)
        )""",
        """CREATE TABLE IF NOT EXISTS watchlist (
            id          SERIAL PRIMARY KEY,
            ticker      VARCHAR(20)  NOT NULL UNIQUE,
            asset_type  VARCHAR(10)  NOT NULL DEFAULT 'stock',
            added_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            added_by    VARCHAR(20)  NOT NULL DEFAULT 'user',
            is_active   BOOLEAN      NOT NULL DEFAULT TRUE
        )""",
        """CREATE TABLE IF NOT EXISTS ml_signals (
            id          SERIAL PRIMARY KEY,
            ticker      VARCHAR(20)  NOT NULL,
            timestamp   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            signal      VARCHAR(10),
            probability NUMERIC(6,4),
            model_type  VARCHAR(30),
            features    JSONB
        )""",
        """CREATE TABLE IF NOT EXISTS agent_log (
            id          SERIAL PRIMARY KEY,
            timestamp   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            action      VARCHAR(10)  NOT NULL,
            ticker      VARCHAR(20)  NOT NULL,
            reason      TEXT,
            score       NUMERIC(8,2)
        )""",
        """CREATE TABLE IF NOT EXISTS trades (
            id           SERIAL PRIMARY KEY,
            ticker       VARCHAR(20)  NOT NULL,
            action       VARCHAR(10)  NOT NULL,
            qty          NUMERIC(18,6) NOT NULL,
            price        NUMERIC(18,4) NOT NULL,
            value_usd    NUMERIC(18,4) GENERATED ALWAYS AS (qty * price) STORED,
            pnl_pct      NUMERIC(10,4),
            stop_loss    NUMERIC(18,4),
            take_profit  NUMERIC(18,4),
            reason       TEXT,
            order_id     VARCHAR(64),
            status       VARCHAR(20)  NOT NULL DEFAULT 'filled',
            timestamp    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        )""",
        "CREATE INDEX IF NOT EXISTS idx_scan_ticker_ts ON scan_results (ticker, timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts_sent (timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_trades_ticker_ts ON trades (ticker, timestamp DESC)",
        "ALTER TABLE trades ENABLE ROW LEVEL SECURITY",
    ]
    for sql in ddl:
        try:
            execute(sql)
        except Exception as e:
            logger.warning(f"⚠️ DDL: {e}")
    logger.info("✅ Tablas Supabase listas")


# ─────────────────────────────────────────────────────────────────────────────
# WATCHLIST
# ─────────────────────────────────────────────────────────────────────────────

def get_watchlist() -> List[Dict]:
    """
    Retorna tickers activos.
    IMPORTANTE: incluye 'category' como alias de 'asset_type'
    para compatibilidad con scheduler.py.
    """
    df = query_df("""
        SELECT ticker, asset_type, added_at, added_by
        FROM watchlist
        WHERE is_active = TRUE
        ORDER BY added_at ASC
    """)
    if df.empty:
        return []
    records = df.to_dict("records")
    for r in records:
        r["category"] = r.get("asset_type", "stock")  # ← alias crítico
    return records


def get_watchlist_tickers() -> List[str]:
    df = query_df("SELECT ticker FROM watchlist WHERE is_active = TRUE")
    return df["ticker"].tolist() if not df.empty else []


def add_ticker(ticker: str, asset_type: str = "stock", added_by: str = "agent"):
    execute("""
        INSERT INTO watchlist (ticker, asset_type, added_by)
        VALUES (%s, %s, %s)
        ON CONFLICT (ticker) DO UPDATE SET is_active = TRUE
    """, (ticker.upper(), asset_type, added_by))


def remove_ticker(ticker: str):
    execute("UPDATE watchlist SET is_active = FALSE WHERE ticker = %s",
            (ticker.upper(),))


def sync_watchlist_from_json(json_path: str = "data/watchlist.json"):
    import json
    from pathlib import Path
    if not Path(json_path).exists():
        logger.warning(f"⚠️ {json_path} no encontrado")
        return
    with open(json_path) as f:
        data = json.load(f)
    stocks = data.get("stocks", [])
    crypto = data.get("crypto", [])
    for t in stocks: add_ticker(t, "stock",  "json_sync")
    for t in crypto: add_ticker(t, "crypto", "json_sync")
    logger.info(f"✅ Watchlist sincronizada: {len(stocks)} stocks + {len(crypto)} crypto")


# ─────────────────────────────────────────────────────────────────────────────
# SCAN
# ─────────────────────────────────────────────────────────────────────────────

def save_scan_result(ticker: str, data: Dict):
    import json
    execute("""
        INSERT INTO scan_results
            (ticker, price, change_pct, score, recommendation,
             rsi, adx, rvol, macd, bb_position, volume, raw_data)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        ticker,
        data.get("price"),       data.get("change_pct"),
        data.get("score"),       data.get("recommendation"),
        data.get("rsi"),         data.get("adx"),
        data.get("rvol"),        data.get("macd"),
        data.get("bb_position"), data.get("volume"),
        json.dumps(data.get("extra", {})),
    ))


def get_latest_scan() -> pd.DataFrame:
    return query_df("""
        SELECT DISTINCT ON (ticker)
            ticker, timestamp, price, change_pct, score,
            recommendation, rsi, adx, rvol
        FROM scan_results
        WHERE timestamp >= NOW() - INTERVAL '2 hours'
        ORDER BY ticker, timestamp DESC
    """)


def get_ticker_history(ticker: str, days: int = 3) -> pd.DataFrame:
    return query_df("""
        SELECT timestamp, score, rsi, rvol, price
        FROM scan_results
        WHERE ticker = :ticker
          AND timestamp >= NOW() - INTERVAL :interval
        ORDER BY timestamp ASC
    """, {"ticker": ticker, "interval": f"{days} days"})


def get_top_picks(n: int = 5) -> pd.DataFrame:
    df = query_df("""
        SELECT DISTINCT ON (ticker)
            ticker, timestamp, price, change_pct,
            score, recommendation, rsi, adx, rvol
        FROM scan_results
        WHERE timestamp >= NOW() - INTERVAL '2 hours'
          AND score > 0
        ORDER BY ticker, timestamp DESC
    """)
    if df.empty:
        return df
    return df.sort_values("score", ascending=False).head(n)


# ─────────────────────────────────────────────────────────────────────────────
# ML SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

def save_ml_signal(ticker: str, probability: float, signal: str = None,
                    model_type: str = "ensemble", features: Dict = None):
    import json
    execute("""
        INSERT INTO ml_signals (ticker, signal, probability, model_type, features)
        VALUES (%s,%s,%s,%s,%s)
    """, (ticker, signal, probability, model_type, json.dumps(features or {})))


def get_recent_ml_signals(ticker: str = None, hours: int = 24) -> pd.DataFrame:
    if ticker:
        return query_df("""
            SELECT ticker, timestamp, signal, probability, model_type, features
            FROM ml_signals
            WHERE ticker = :ticker AND timestamp >= NOW() - INTERVAL :interval
            ORDER BY timestamp DESC
        """, {"ticker": ticker, "interval": f"{hours} hours"})
    return query_df("""
        SELECT ticker, timestamp, signal, probability, model_type, features
        FROM ml_signals
        WHERE timestamp >= NOW() - INTERVAL :interval
        ORDER BY timestamp DESC
    """, {"interval": f"{hours} hours"})


# ─────────────────────────────────────────────────────────────────────────────
# ALERTAS
# ─────────────────────────────────────────────────────────────────────────────

def save_alert(ticker: str, alert_type: str, message: str, channel: str = "telegram"):
    execute("""
        INSERT INTO alerts_sent (ticker, alert_type, message, channel)
        VALUES (%s,%s,%s,%s)
    """, (ticker, alert_type, message, channel))


def get_recent_alerts(hours: int = 24) -> pd.DataFrame:
    return query_df("""
        SELECT ticker, timestamp, alert_type, message, channel
        FROM alerts_sent
        WHERE timestamp >= NOW() - INTERVAL :interval
        ORDER BY timestamp DESC LIMIT 50
    """, {"interval": f"{hours} hours"})


def alert_cooldown_ok(ticker: str, alert_type: str, minutes: int = 30) -> bool:
    df = query_df("""
        SELECT COUNT(*) AS cnt FROM alerts_sent
        WHERE ticker = :ticker
          AND alert_type = :atype
          AND timestamp >= NOW() - INTERVAL :interval
    """, {"ticker": ticker, "atype": alert_type, "interval": f"{minutes} minutes"})
    if df.empty:
        return True
    return int(df["cnt"].iloc[0]) == 0


# ─────────────────────────────────────────────────────────────────────────────
# AGENTE
# ─────────────────────────────────────────────────────────────────────────────

def log_agent_action(action: str, ticker: str, reason: str, score: float = 0):
    execute("""
        INSERT INTO agent_log (action, ticker, reason, score)
        VALUES (%s,%s,%s,%s)
    """, (action.upper(), ticker, reason, score))


def get_agent_log(limit: int = 20) -> pd.DataFrame:
    return query_df("""
        SELECT timestamp, action, ticker, reason, score
        FROM agent_log ORDER BY timestamp DESC LIMIT :limit
    """, {"limit": limit})


# ─────────────────────────────────────────────────────────────────────────────
# TRADES (fuente de verdad persistente — sobrevive redeploys de Railway,
# a diferencia de data/performance_history.json que vive en filesystem efímero)
# ─────────────────────────────────────────────────────────────────────────────

def save_trade(ticker: str, action: str, qty: float, price: float,
                pnl_pct: float = None, stop_loss: float = None,
                take_profit: float = None, reason: str = None,
                order_id: str = None, status: str = "filled"):
    execute("""
        INSERT INTO trades
            (ticker, action, qty, price, pnl_pct, stop_loss, take_profit,
             reason, order_id, status)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        ticker, action.upper(), qty, price, pnl_pct, stop_loss, take_profit,
        reason, order_id, status,
    ))


def get_trades(ticker: str = None, days: int = None, limit: int = 200) -> pd.DataFrame:
    where = []
    params = {"limit": limit}
    if ticker:
        where.append("ticker = :ticker")
        params["ticker"] = ticker
    if days:
        where.append("timestamp >= NOW() - INTERVAL :interval")
        params["interval"] = f"{days} days"
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    return query_df(f"""
        SELECT ticker, action, qty, price, value_usd, pnl_pct, stop_loss,
               take_profit, reason, order_id, status, timestamp
        FROM trades
        {where_sql}
        ORDER BY timestamp DESC
        LIMIT :limit
    """, params)


def get_trades_summary() -> Dict:
    """
    Métricas agregadas calculadas directamente en Supabase — sirven como
    fuente persistente para performance real (win rate, profit factor),
    a diferencia de PerformanceTracker que vive en el JSON local efímero.
    """
    df = query_df("""
        SELECT action, qty, price, pnl_pct, timestamp
        FROM trades
        WHERE action = 'SELL' AND pnl_pct IS NOT NULL
    """)
    if df.empty:
        return {"total_trades": 0, "win_rate_pct": 0.0, "profit_factor": 0.0,
                "avg_win_pct": 0.0, "avg_loss_pct": 0.0}

    wins = df[df["pnl_pct"] > 0]
    losses = df[df["pnl_pct"] <= 0]
    gross_profit = (wins["pnl_pct"] * wins["qty"] * wins["price"] / 100).sum()
    gross_loss = abs((losses["pnl_pct"] * losses["qty"] * losses["price"] / 100).sum())

    return {
        "total_trades": len(df),
        "win_rate_pct": round(len(wins) / len(df) * 100, 1) if len(df) else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0,
        "avg_win_pct": round(wins["pnl_pct"].mean(), 2) if not wins.empty else 0.0,
        "avg_loss_pct": round(losses["pnl_pct"].mean(), 2) if not losses.empty else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATUS
# ─────────────────────────────────────────────────────────────────────────────

def get_scheduler_status() -> Dict:
    df = query_df("""
        SELECT MAX(timestamp) AS last_scan,
               COUNT(DISTINCT timestamp) AS total_scans,
               COUNT(DISTINCT ticker)    AS tickers_scanned
        FROM scan_results
    """)
    if df.empty or df["last_scan"].iloc[0] is None:
        return {"running": False, "status": "inactive", "label": "Sin datos",
                "last_scan": None, "age_min": None, "total_scans": 0}

    last_ts = pd.to_datetime(df["last_scan"].iloc[0])
    total   = int(df["total_scans"].iloc[0])
    now     = pd.Timestamp.utcnow().tz_localize(None)
    if last_ts.tzinfo is not None:
        last_ts = last_ts.tz_convert(None)
    age = (now - last_ts).total_seconds() / 60

    if age < 8:   status, label = "active",   f"Activo · hace {age:.0f} min"
    elif age < 30: status, label = "waiting",  f"Esperando · hace {age:.0f} min"
    else:          status, label = "inactive", f"Inactivo · hace {age:.0f} min"

    return {"running": status == "active", "status": status, "label": label,
            "last_scan": str(df["last_scan"].iloc[0]), "age_min": round(age, 1),
            "total_scans": total, "tickers_scanned": int(df["tickers_scanned"].iloc[0])}


# ─────────────────────────────────────────────────────────────────────────────
# LIMPIEZA
# ─────────────────────────────────────────────────────────────────────────────

def cleanup_old_data(days: int = 7):
    execute("DELETE FROM scan_results WHERE timestamp < NOW() - INTERVAL %s",
            (f"{days} days",))
    execute("DELETE FROM alerts_sent   WHERE timestamp < NOW() - INTERVAL %s",
            (f"{days} days",))
    logger.info(f"✅ Limpieza completada — datos > {days} días eliminados")

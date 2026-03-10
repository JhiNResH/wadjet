"""
Supabase / PostgreSQL client for Wadjet.
Handles connection pooling, schema migrations, and CRUD operations.
"""

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger("wadjet.db")

# ─── Connection ────────────────────────────────────────────────────────────────

_RAW_DB_URL = os.environ.get(
    "DATABASE_URL",
    os.environ["DATABASE_URL"],
)
# psycopg2 doesn't understand pgbouncer=true — strip it
DATABASE_URL = _RAW_DB_URL.split("?")[0]

_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        logger.info("Initializing connection pool...")
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=DATABASE_URL,
        )
        logger.info("Connection pool ready")
    return _pool


@contextmanager
def get_cursor():
    pool = get_pool()
    conn = pool.getconn()
    try:
        conn.autocommit = False
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ─── Schema Migrations ─────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Agent behavior profiles
CREATE TABLE IF NOT EXISTS wadjet_agent_profiles (
    id              SERIAL PRIMARY KEY,
    address         TEXT NOT NULL UNIQUE,
    behavior_type   TEXT NOT NULL,
    avg_daily_volume FLOAT,
    counterparties  INT,
    risk_tolerance  TEXT,
    dependencies    JSONB DEFAULT '[]'::jsonb,
    survival_history TEXT,
    metrics         JSONB DEFAULT '{}'::jsonb,
    raw_profile     JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Simulation results (Monte Carlo outputs)
CREATE TABLE IF NOT EXISTS wadjet_simulation_results (
    id              SERIAL PRIMARY KEY,
    address         TEXT NOT NULL,
    scenarios       JSONB NOT NULL DEFAULT '[]'::jsonb,
    cascade_risk    JSONB NOT NULL DEFAULT '[]'::jsonb,
    cluster_id      INT,
    resilience_score FLOAT,
    simulation_runs INT DEFAULT 0,
    cached_until    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Unique index for fast lookup
CREATE UNIQUE INDEX IF NOT EXISTS idx_sim_address
    ON wadjet_simulation_results (address);

-- Transaction relationship graph edges
CREATE TABLE IF NOT EXISTS wadjet_tx_graph (
    id              SERIAL PRIMARY KEY,
    from_address    TEXT NOT NULL,
    to_address      TEXT NOT NULL,
    weight          FLOAT DEFAULT 1.0,
    edge_type       TEXT DEFAULT 'transfer',
    tx_count        INT DEFAULT 1,
    total_value     FLOAT DEFAULT 0.0,
    last_seen       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (from_address, to_address)
);

CREATE INDEX IF NOT EXISTS idx_graph_from ON wadjet_tx_graph (from_address);
CREATE INDEX IF NOT EXISTS idx_graph_to   ON wadjet_tx_graph (to_address);

-- Hidden clusters
CREATE TABLE IF NOT EXISTS wadjet_clusters (
    id              SERIAL PRIMARY KEY,
    cluster_id      INT NOT NULL UNIQUE,
    member_addresses JSONB NOT NULL DEFAULT '[]'::jsonb,
    cluster_type    TEXT DEFAULT 'unknown',
    risk_score      FLOAT DEFAULT 0.0,
    detected_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Daily cron run logs
CREATE TABLE IF NOT EXISTS cron_logs (
    id               SERIAL PRIMARY KEY,
    run_id           TEXT NOT NULL,
    status           TEXT NOT NULL,
    agents_profiled  INT DEFAULT 0,
    new_risk_flags   INT DEFAULT 0,
    error_count      INT DEFAULT 0,
    errors           JSONB DEFAULT '[]'::jsonb,
    duration_seconds FLOAT,
    extra            JSONB DEFAULT '{}'::jsonb,
    ran_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_cron_logs_run_id
    ON cron_logs (run_id);

-- Daily token snapshots for delta computation
CREATE TABLE IF NOT EXISTS wadjet_daily_snapshots (
    id               BIGSERIAL PRIMARY KEY,
    token_address    TEXT NOT NULL,
    wallet_address   TEXT,
    snapshot_date    DATE NOT NULL DEFAULT CURRENT_DATE,
    -- GoPlus signals
    top10_holder_pct FLOAT,
    holder_count     INT,
    lp_locked_pct    FLOAT,
    creator_percent  FLOAT,
    owner_percent    FLOAT,
    -- DexScreener signals
    price_usd        FLOAT,
    liquidity_usd    FLOAT,
    volume_24h       FLOAT,
    market_cap       FLOAT,
    -- ACP signals
    trust_score      FLOAT,
    total_jobs       INT,
    completion_rate  FLOAT,
    -- Metadata
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(token_address, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_token_date
    ON wadjet_daily_snapshots(token_address, snapshot_date DESC);

-- Watchlist for flagged tokens
CREATE TABLE IF NOT EXISTS wadjet_watchlist (
    id             BIGSERIAL PRIMARY KEY,
    token_address  TEXT NOT NULL UNIQUE,
    wallet_address TEXT,
    agent_name     TEXT,
    reason         TEXT NOT NULL,
    severity       TEXT NOT NULL DEFAULT 'medium',
    added_at       TIMESTAMPTZ DEFAULT NOW(),
    last_checked   TIMESTAMPTZ,
    status         TEXT DEFAULT 'active',
    notes          TEXT
);

-- Sentinel alerts: sell signals, dump patterns, confirmed rugs
CREATE TABLE IF NOT EXISTS wadjet_alerts (
    id             BIGSERIAL PRIMARY KEY,
    token_address  TEXT NOT NULL,
    wallet_address TEXT,
    agent_name     TEXT,
    alert_type     TEXT NOT NULL,  -- 'watchlist_added', 'sell_signal', 'dump_pattern', 'confirmed_rug'
    severity       TEXT NOT NULL,  -- 'info', 'warning', 'critical'
    details        JSONB,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_token
    ON wadjet_alerts(token_address, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity
    ON wadjet_alerts(severity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_type
    ON wadjet_alerts(alert_type, created_at DESC);
"""


def ensure_schema():
    """Run CREATE TABLE IF NOT EXISTS migrations."""
    with get_cursor() as cur:
        cur.execute(SCHEMA_SQL)
    logger.info("Schema ensured")


# ─── Agent Profiles ────────────────────────────────────────────────────────────

def upsert_agent_profile(profile: dict) -> None:
    address = profile["agent"]
    sql = """
        INSERT INTO wadjet_agent_profiles
            (address, behavior_type, avg_daily_volume, counterparties,
             risk_tolerance, dependencies, survival_history, metrics, raw_profile, updated_at)
        VALUES
            (%(address)s, %(behavior_type)s, %(avg_daily_volume)s, %(counterparties)s,
             %(risk_tolerance)s, %(dependencies)s, %(survival_history)s, %(metrics)s, %(raw_profile)s, NOW())
        ON CONFLICT (address) DO UPDATE SET
            behavior_type    = EXCLUDED.behavior_type,
            avg_daily_volume = EXCLUDED.avg_daily_volume,
            counterparties   = EXCLUDED.counterparties,
            risk_tolerance   = EXCLUDED.risk_tolerance,
            dependencies     = EXCLUDED.dependencies,
            survival_history = EXCLUDED.survival_history,
            metrics          = EXCLUDED.metrics,
            raw_profile      = EXCLUDED.raw_profile,
            updated_at       = NOW()
    """
    with get_cursor() as cur:
        cur.execute(sql, {
            "address": address,
            "behavior_type": profile.get("behavior_type", "normal"),
            "avg_daily_volume": profile.get("avg_daily_volume"),
            "counterparties": profile.get("counterparties"),
            "risk_tolerance": profile.get("risk_tolerance", "medium"),
            "dependencies": json.dumps(profile.get("dependencies", [])),
            "survival_history": profile.get("survival_history", ""),
            "metrics": json.dumps(profile.get("metrics", {})),
            "raw_profile": json.dumps(profile),
        })


def get_agent_profile(address: str) -> Optional[dict]:
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM wadjet_agent_profiles WHERE address = %s",
            (address.lower(),)
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_all_profiles(limit: int = 500) -> list[dict]:
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM wadjet_agent_profiles ORDER BY updated_at DESC LIMIT %s",
            (limit,)
        )
        return [dict(r) for r in cur.fetchall()]


# ─── Simulation Results ────────────────────────────────────────────────────────

def upsert_simulation_result(result: dict, cache_hours: int = 24) -> None:
    from datetime import timedelta
    cached_until = datetime.now(timezone.utc) + timedelta(hours=cache_hours)
    sql = """
        INSERT INTO wadjet_simulation_results
            (address, scenarios, cascade_risk, cluster_id, resilience_score,
             simulation_runs, cached_until, updated_at)
        VALUES
            (%(address)s, %(scenarios)s, %(cascade_risk)s, %(cluster_id)s,
             %(resilience_score)s, %(simulation_runs)s, %(cached_until)s, NOW())
        ON CONFLICT (address) DO UPDATE SET
            scenarios        = EXCLUDED.scenarios,
            cascade_risk     = EXCLUDED.cascade_risk,
            cluster_id       = EXCLUDED.cluster_id,
            resilience_score = EXCLUDED.resilience_score,
            simulation_runs  = EXCLUDED.simulation_runs,
            cached_until     = EXCLUDED.cached_until,
            updated_at       = NOW()
    """
    with get_cursor() as cur:
        cur.execute(sql, {
            "address": result["agent"],
            "scenarios": json.dumps(result.get("scenarios", [])),
            "cascade_risk": json.dumps(result.get("cascade_risk", [])),
            "cluster_id": result.get("cluster_id"),
            "resilience_score": result.get("resilience_score"),
            "simulation_runs": result.get("simulation_runs", 0),
            "cached_until": cached_until,
        })


def get_simulation_result(address: str) -> Optional[dict]:
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT * FROM wadjet_simulation_results
            WHERE address = %s AND cached_until > NOW()
            """,
            (address.lower(),)
        )
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        # Deserialize JSONB
        for field in ("scenarios", "cascade_risk"):
            if isinstance(d[field], str):
                d[field] = json.loads(d[field])
        return d


# ─── TX Graph ─────────────────────────────────────────────────────────────────

def upsert_graph_edges(edges: list[dict]) -> None:
    sql = """
        INSERT INTO wadjet_tx_graph
            (from_address, to_address, weight, edge_type, tx_count, total_value, last_seen)
        VALUES
            (%(from_address)s, %(to_address)s, %(weight)s, %(edge_type)s,
             %(tx_count)s, %(total_value)s, NOW())
        ON CONFLICT (from_address, to_address) DO UPDATE SET
            weight      = GREATEST(EXCLUDED.weight, wadjet_tx_graph.weight),
            tx_count    = wadjet_tx_graph.tx_count + EXCLUDED.tx_count,
            total_value = wadjet_tx_graph.total_value + EXCLUDED.total_value,
            last_seen   = NOW()
    """
    with get_cursor() as cur:
        cur.executemany(sql, edges)


def get_graph_edges(address: str) -> list[dict]:
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT * FROM wadjet_tx_graph
            WHERE from_address = %s OR to_address = %s
            ORDER BY weight DESC
            """,
            (address.lower(), address.lower())
        )
        return [dict(r) for r in cur.fetchall()]


# ─── Clusters ─────────────────────────────────────────────────────────────────

def upsert_clusters(clusters: list[dict]) -> None:
    sql = """
        INSERT INTO wadjet_clusters
            (cluster_id, member_addresses, cluster_type, risk_score, detected_at)
        VALUES
            (%(cluster_id)s, %(member_addresses)s, %(cluster_type)s, %(risk_score)s, NOW())
        ON CONFLICT (cluster_id) DO UPDATE SET
            member_addresses = EXCLUDED.member_addresses,
            cluster_type     = EXCLUDED.cluster_type,
            risk_score       = EXCLUDED.risk_score,
            detected_at      = NOW()
    """
    with get_cursor() as cur:
        cur.executemany(sql, [
            {
                "cluster_id": c["cluster_id"],
                "member_addresses": json.dumps(c.get("members", [])),
                "cluster_type": c.get("cluster_type", "unknown"),
                "risk_score": c.get("risk_score", 0.0),
            }
            for c in clusters
        ])


def get_all_clusters() -> list[dict]:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM wadjet_clusters ORDER BY risk_score DESC")
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            if isinstance(r.get("member_addresses"), str):
                r["member_addresses"] = json.loads(r["member_addresses"])
        return rows


# ─── Agent Score table (read-only) ────────────────────────────────────────────

def fetch_agent_scores(limit: int = 500) -> list[dict]:
    """Read existing agent data from the agent_scores table."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM agent_scores ORDER BY last_updated DESC NULLS LAST LIMIT %s",
            (limit,)
        )
        return [dict(r) for r in cur.fetchall()]


# ─── Daily Snapshots ──────────────────────────────────────────────────────────

def upsert_daily_snapshot(snapshot: dict) -> None:
    """Upsert today's snapshot for a token. snapshot dict keys match column names."""
    sql = """
        INSERT INTO wadjet_daily_snapshots
            (token_address, wallet_address, snapshot_date,
             top10_holder_pct, holder_count, lp_locked_pct, creator_percent, owner_percent,
             price_usd, liquidity_usd, volume_24h, market_cap,
             trust_score, total_jobs, completion_rate)
        VALUES
            (%(token_address)s, %(wallet_address)s, CURRENT_DATE,
             %(top10_holder_pct)s, %(holder_count)s, %(lp_locked_pct)s, %(creator_percent)s, %(owner_percent)s,
             %(price_usd)s, %(liquidity_usd)s, %(volume_24h)s, %(market_cap)s,
             %(trust_score)s, %(total_jobs)s, %(completion_rate)s)
        ON CONFLICT (token_address, snapshot_date) DO UPDATE SET
            wallet_address   = EXCLUDED.wallet_address,
            top10_holder_pct = EXCLUDED.top10_holder_pct,
            holder_count     = EXCLUDED.holder_count,
            lp_locked_pct    = EXCLUDED.lp_locked_pct,
            creator_percent  = EXCLUDED.creator_percent,
            owner_percent    = EXCLUDED.owner_percent,
            price_usd        = EXCLUDED.price_usd,
            liquidity_usd    = EXCLUDED.liquidity_usd,
            volume_24h       = EXCLUDED.volume_24h,
            market_cap       = EXCLUDED.market_cap,
            trust_score      = EXCLUDED.trust_score,
            total_jobs       = EXCLUDED.total_jobs,
            completion_rate  = EXCLUDED.completion_rate,
            created_at       = NOW()
    """
    with get_cursor() as cur:
        cur.execute(sql, {
            "token_address":   snapshot.get("token_address", ""),
            "wallet_address":  snapshot.get("wallet_address"),
            "top10_holder_pct": snapshot.get("top10_holder_pct"),
            "holder_count":    snapshot.get("holder_count"),
            "lp_locked_pct":   snapshot.get("lp_locked_pct"),
            "creator_percent": snapshot.get("creator_percent"),
            "owner_percent":   snapshot.get("owner_percent"),
            "price_usd":       snapshot.get("price_usd"),
            "liquidity_usd":   snapshot.get("liquidity_usd"),
            "volume_24h":      snapshot.get("volume_24h"),
            "market_cap":      snapshot.get("market_cap"),
            "trust_score":     snapshot.get("trust_score"),
            "total_jobs":      snapshot.get("total_jobs"),
            "completion_rate": snapshot.get("completion_rate"),
        })


def get_snapshot_for_date(token_address: str, days_ago: int = 1) -> Optional[dict]:
    """
    Fetch snapshot for a token from `days_ago` days back.
    days_ago=1 → yesterday, days_ago=0 → today.
    """
    sql = """
        SELECT * FROM wadjet_daily_snapshots
        WHERE token_address = %s
          AND snapshot_date = CURRENT_DATE - INTERVAL %s
        LIMIT 1
    """
    with get_cursor() as cur:
        cur.execute(sql, (token_address.lower(), f"{days_ago} days"))
        row = cur.fetchone()
        return dict(row) if row else None


def get_last_two_snapshots(token_address: str) -> list[dict]:
    """Fetch the last 2 snapshots for delta computation (most recent first)."""
    sql = """
        SELECT * FROM wadjet_daily_snapshots
        WHERE token_address = %s
        ORDER BY snapshot_date DESC
        LIMIT 2
    """
    with get_cursor() as cur:
        cur.execute(sql, (token_address.lower(),))
        return [dict(r) for r in cur.fetchall()]


# ─── Watchlist ────────────────────────────────────────────────────────────────

def upsert_watchlist_item(item: dict) -> None:
    """Insert or update a watchlist entry. Conflict on token_address → update reason/severity."""
    sql = """
        INSERT INTO wadjet_watchlist
            (token_address, wallet_address, agent_name, reason, severity, last_checked)
        VALUES
            (%(token_address)s, %(wallet_address)s, %(agent_name)s,
             %(reason)s, %(severity)s, NOW())
        ON CONFLICT (token_address) DO UPDATE SET
            reason       = EXCLUDED.reason,
            severity     = EXCLUDED.severity,
            wallet_address = COALESCE(EXCLUDED.wallet_address, wadjet_watchlist.wallet_address),
            agent_name   = COALESCE(EXCLUDED.agent_name, wadjet_watchlist.agent_name),
            last_checked = NOW(),
            status       = 'active'
    """
    with get_cursor() as cur:
        cur.execute(sql, {
            "token_address": item.get("token_address", "").lower(),
            "wallet_address": item.get("wallet_address"),
            "agent_name":    item.get("agent_name"),
            "reason":        item.get("reason", ""),
            "severity":      item.get("severity", "medium"),
        })


def get_watchlist(status: str = "active", limit: int = 200) -> list[dict]:
    """Fetch active watchlist items sorted by severity."""
    SEVERITY_ORDER = "CASE severity WHEN 'critical' THEN 1 WHEN 'high' THEN 2 WHEN 'medium' THEN 3 ELSE 4 END"
    sql = f"""
        SELECT * FROM wadjet_watchlist
        WHERE status = %s
        ORDER BY {SEVERITY_ORDER}, added_at DESC
        LIMIT %s
    """
    with get_cursor() as cur:
        cur.execute(sql, (status, limit))
        return [dict(r) for r in cur.fetchall()]


def get_watchlist_item(token_address: str) -> Optional[dict]:
    """Fetch a single watchlist entry by token address."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM wadjet_watchlist WHERE token_address = %s",
            (token_address.lower(),)
        )
        row = cur.fetchone()
        return dict(row) if row else None


# ─── Sentinel Alerts ──────────────────────────────────────────────────────────

def get_alerts(
    severity: Optional[str] = None,
    alert_type: Optional[str] = None,
    token_address: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """
    Fetch recent alerts, optionally filtered by severity, type, or token.
    Always returns newest-first.
    """
    conditions = []
    params: list = []

    if severity:
        conditions.append("severity = %s")
        params.append(severity)
    if alert_type:
        conditions.append("alert_type = %s")
        params.append(alert_type)
    if token_address:
        conditions.append("token_address = %s")
        params.append(token_address.lower())

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"""
        SELECT * FROM wadjet_alerts
        {where}
        ORDER BY created_at DESC
        LIMIT %s
    """
    params.append(min(limit, 500))

    with get_cursor() as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def get_alert_counts() -> dict:
    """Return count of alerts grouped by severity and alert_type."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) AS total,
                COUNT(CASE WHEN severity = 'critical' THEN 1 END) AS critical,
                COUNT(CASE WHEN severity = 'warning'  THEN 1 END) AS warning,
                COUNT(CASE WHEN severity = 'info'     THEN 1 END) AS info,
                COUNT(CASE WHEN alert_type = 'confirmed_rug'  THEN 1 END) AS confirmed_rugs,
                COUNT(CASE WHEN alert_type = 'dump_pattern'   THEN 1 END) AS dump_patterns,
                COUNT(CASE WHEN alert_type = 'sell_signal'    THEN 1 END) AS sell_signals,
                COUNT(CASE WHEN alert_type = 'watchlist_added' THEN 1 END) AS watchlist_added,
                MAX(created_at) AS last_alert_at
            FROM wadjet_alerts
        """)
        row = cur.fetchone()
        if not row:
            return {}
        d = dict(row)
        # Convert to ints/floats for JSON serialisation
        for k, v in d.items():
            if k == "last_alert_at":
                d[k] = str(v) if v else None
            elif v is not None:
                try:
                    d[k] = int(v)
                except (TypeError, ValueError):
                    pass
        return d


def get_last_stage1_scan() -> Optional[str]:
    """Return the timestamp of the last successful sentinel stage1 run."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT ran_at FROM cron_logs
            WHERE run_id LIKE 'sentinel-stage1-%' AND status = 'ok'
            ORDER BY ran_at DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        return str(row["ran_at"]) if row else None


def get_last_stage2_check() -> Optional[str]:
    """Return the timestamp of the last successful sentinel stage2 run."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT ran_at FROM cron_logs
            WHERE run_id LIKE 'sentinel-stage2-%' AND status = 'ok'
            ORDER BY ran_at DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        return str(row["ran_at"]) if row else None


def fetch_top_agents_for_cron(limit: int = 500) -> list[dict]:
    """
    Fetch the top agents ranked by trust_score + query_volume for daily re-profiling.
    Falls back gracefully if columns don't exist.
    """
    # Try trust_score + query_volume ordering first
    for order_sql in (
        "trust_score DESC NULLS LAST, query_volume DESC NULLS LAST",
        "trust_score DESC NULLS LAST",
        "last_updated DESC NULLS LAST",
    ):
        try:
            with get_cursor() as cur:
                cur.execute(
                    f"SELECT * FROM agent_scores ORDER BY {order_sql} LIMIT %s",
                    (limit,)
                )
                return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.debug(f"Order by '{order_sql}' failed: {e} — trying fallback")
    return []

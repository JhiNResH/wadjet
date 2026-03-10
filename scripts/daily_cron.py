#!/usr/bin/env python3
"""
Wadjet Daily Cron — standalone re-profiling job.

Usage:
    python scripts/daily_cron.py

Environment:
    DATABASE_URL      — Supabase PostgreSQL connection string
    ALCHEMY_API_KEY   — Alchemy API key
    CRON_API_KEY      — (optional) same key used by FastAPI /cron/run-daily
    MAX_AGENTS        — (optional) override top-N limit (default: 500)
    MAX_RUNTIME_MIN   — (optional) max runtime in minutes (default: 30)

Railway: set as Cron job or call POST /cron/run-daily from GitHub Actions.
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone

# ─── Bootstrap path so we can import wadjet modules ──────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PKG_ROOT)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("wadjet.cron")

# ─── Config ───────────────────────────────────────────────────────────────────
MAX_AGENTS = int(os.environ.get("MAX_AGENTS", "500"))
MAX_RUNTIME_SECONDS = int(os.environ.get("MAX_RUNTIME_MIN", "30")) * 60

# ─── Alchemy retry helpers ────────────────────────────────────────────────────

ALCHEMY_MAX_RETRIES = 4
ALCHEMY_BASE_DELAY  = 2.0   # seconds; doubles on each retry


def with_alchemy_retry(fn, *args, label="call", **kwargs):
    """
    Call `fn(*args, **kwargs)` with exponential back-off on Alchemy 429s.
    """
    delay = ALCHEMY_BASE_DELAY
    for attempt in range(1, ALCHEMY_MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            is_rate_limit = (
                "429" in msg
                or "rate limit" in msg
                or "too many requests" in msg
            )
            if is_rate_limit and attempt < ALCHEMY_MAX_RETRIES:
                logger.warning(
                    f"Alchemy rate-limit on {label} (attempt {attempt}/{ALCHEMY_MAX_RETRIES}). "
                    f"Sleeping {delay:.1f}s …"
                )
                time.sleep(delay)
                delay *= 2
            else:
                raise


# ─── Cron log writer ─────────────────────────────────────────────────────────

def write_cron_log(
    run_id: str,
    status: str,
    agents_profiled: int,
    new_risk_flags: int,
    errors: list[str],
    duration_seconds: float,
    extra: dict | None = None,
) -> None:
    """
    Upsert a row into `cron_logs` table.
    Creates the table first if it doesn't exist.
    """
    try:
        from db.supabase_client import get_cursor

        CREATE_SQL = """
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
        """

        INSERT_SQL = """
            INSERT INTO cron_logs
                (run_id, status, agents_profiled, new_risk_flags,
                 error_count, errors, duration_seconds, extra, ran_at)
            VALUES
                (%(run_id)s, %(status)s, %(agents_profiled)s, %(new_risk_flags)s,
                 %(error_count)s, %(errors)s::jsonb, %(duration_seconds)s,
                 %(extra)s::jsonb, NOW())
            ON CONFLICT (run_id) DO UPDATE SET
                status           = EXCLUDED.status,
                agents_profiled  = EXCLUDED.agents_profiled,
                new_risk_flags   = EXCLUDED.new_risk_flags,
                error_count      = EXCLUDED.error_count,
                errors           = EXCLUDED.errors,
                duration_seconds = EXCLUDED.duration_seconds,
                extra            = EXCLUDED.extra,
                ran_at           = NOW()
        """

        import json
        with get_cursor() as cur:
            cur.execute(CREATE_SQL)
            cur.execute(INSERT_SQL, {
                "run_id":           run_id,
                "status":           status,
                "agents_profiled":  agents_profiled,
                "new_risk_flags":   new_risk_flags,
                "error_count":      len(errors),
                "errors":           json.dumps(errors[:50]),   # cap to 50 entries
                "duration_seconds": round(duration_seconds, 2),
                "extra":            json.dumps(extra or {}),
            })
        logger.info(f"Cron log written: run_id={run_id} status={status}")
    except Exception as e:
        logger.error(f"Failed to write cron log: {e}")


# ─── Risk flag detector ───────────────────────────────────────────────────────

def count_risk_flags(profiles: list[dict], previous_profiles: dict[str, dict]) -> int:
    """
    Count newly detected risk flags (high/critical behavior types that were
    previously 'normal' or 'market_maker').
    """
    risky_types = {"whale", "bot", "wash_trader", "exit_liquidity", "rug_pull"}
    safe_types  = {"normal", "market_maker", "accumulator"}
    new_flags = 0
    for p in profiles:
        addr = p.get("agent", "")
        new_type = p.get("behavior_type", "normal")
        old_type = previous_profiles.get(addr, {}).get("behavior_type", "normal")
        if new_type in risky_types and old_type in safe_types:
            new_flags += 1
    return new_flags


# ─── Main cron logic ──────────────────────────────────────────────────────────

def run_daily_cron(limit: int = MAX_AGENTS) -> dict:
    """
    Full daily re-profiling pipeline.
    Returns a summary dict suitable for the FastAPI endpoint response.
    """
    run_id = f"daily-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    start_time = time.time()
    errors: list[str] = []
    snapshots_stored = 0
    watchlist_added  = 0

    logger.info("=" * 60)
    logger.info(f"WADJET DAILY CRON STARTED  run_id={run_id}")
    logger.info(f"Max agents={limit}  Max runtime={MAX_RUNTIME_SECONDS}s")
    logger.info("=" * 60)

    def elapsed() -> float:
        return time.time() - start_time

    def time_left() -> float:
        return MAX_RUNTIME_SECONDS - elapsed()

    # ── Step 0: ensure schema (idempotent) ───────────────────────────────────
    try:
        from db.supabase_client import ensure_schema
        ensure_schema()
    except Exception as e:
        logger.error(f"Schema ensure failed: {e}")
        errors.append(f"schema: {e}")

    # ── Step 1: snapshot current profiles (for diff) ─────────────────────────
    previous_profiles: dict[str, dict] = {}
    try:
        from db.supabase_client import get_all_profiles
        rows = get_all_profiles(limit=limit)
        previous_profiles = {r["address"]: r for r in rows}
        logger.info(f"Snapshot: {len(previous_profiles)} existing profiles")
    except Exception as e:
        logger.warning(f"Failed to snapshot previous profiles: {e}")
        errors.append(f"snapshot: {e}")

    # ── Step 2: build fresh profiles ──────────────────────────────────────────
    profiles: list[dict] = []
    if time_left() < 60:
        errors.append("timeout: not enough time to profile agents")
    else:
        try:
            # Patch alchemy client to wrap calls with retry logic
            _patch_alchemy_client()

            from profiler.profile_builder import build_profiles
            logger.info(f"Building profiles for top {limit} agents …")
            profiles = build_profiles(limit=limit, delay_between=0.5)
            logger.info(f"Profiles built: {len(profiles)}")
        except Exception as e:
            logger.error(f"Profiling step failed: {e}", exc_info=True)
            errors.append(f"profiling: {e}")

    # ── Step 3: Store daily snapshots + compute deltas + watchlist triggers ───
    snapshots_stored = 0
    watchlist_added  = 0
    if profiles:
        try:
            from db.supabase_client import (
                upsert_daily_snapshot,
                get_snapshot_for_date,
                upsert_watchlist_item,
            )

            for profile in profiles:
                if time_left() < 60:
                    logger.warning("Approaching timeout — stopping snapshot storage")
                    break
                try:
                    token_addr  = profile.get("agent", "").lower()
                    wallet_addr = profile.get("wallet_address")

                    # ── Build snapshot from profile raw data ─────────────
                    raw     = profile.get("raw_profile", {}) or {}
                    gp_data = raw.get("goplus", {}) or {}
                    dex     = raw.get("dex", {}) or {}
                    acp     = raw.get("acp", {}) or {}

                    snapshot = {
                        "token_address":   token_addr,
                        "wallet_address":  wallet_addr,
                        # GoPlus
                        "top10_holder_pct": gp_data.get("top10_holder_pct"),
                        "holder_count":     gp_data.get("holder_count"),
                        "lp_locked_pct":    gp_data.get("lp_locked_pct"),
                        "creator_percent":  gp_data.get("creator_percent"),
                        "owner_percent":    gp_data.get("owner_percent"),
                        # DexScreener
                        "price_usd":     dex.get("price_usd"),
                        "liquidity_usd": dex.get("liquidity_usd"),
                        "volume_24h":    dex.get("volume_24h"),
                        "market_cap":    dex.get("market_cap"),
                        # ACP
                        "trust_score":    acp.get("trust_score"),
                        "total_jobs":     acp.get("total_jobs"),
                        "completion_rate": acp.get("completion_rate"),
                    }
                    upsert_daily_snapshot(snapshot)
                    snapshots_stored += 1

                    # ── Compute deltas vs yesterday ───────────────────────
                    yesterday = get_snapshot_for_date(token_addr, days_ago=1)
                    if not yesterday:
                        continue  # no history yet — skip delta/watchlist

                    def _pct_delta(today_val, yest_val):
                        """Relative delta: (today - yest) / yest if yest > 0."""
                        if yest_val and yest_val != 0:
                            return (today_val - yest_val) / abs(yest_val)
                        return None

                    today_top10    = snapshot.get("top10_holder_pct") or 0
                    yest_top10     = yesterday.get("top10_holder_pct") or 0
                    today_liq      = snapshot.get("liquidity_usd") or 0
                    yest_liq       = yesterday.get("liquidity_usd") or 0
                    today_vol      = snapshot.get("volume_24h") or 0
                    yest_vol       = yesterday.get("volume_24h") or 0
                    today_price    = snapshot.get("price_usd") or 0
                    yest_price     = yesterday.get("price_usd") or 0
                    today_holders  = snapshot.get("holder_count") or 0
                    yest_holders   = yesterday.get("holder_count") or 0
                    today_creator  = snapshot.get("creator_percent") or 0
                    yest_creator   = yesterday.get("creator_percent") or 0

                    holder_conc_delta  = today_top10 - yest_top10
                    liquidity_delta    = _pct_delta(today_liq, yest_liq)
                    volume_delta       = _pct_delta(today_vol, yest_vol)
                    price_delta        = _pct_delta(today_price, yest_price)
                    holder_count_delta = today_holders - yest_holders
                    creator_pct_delta  = today_creator - yest_creator

                    # Holder count mass-exit pct
                    holder_exit_pct = (
                        holder_count_delta / yest_holders
                        if yest_holders > 0 else 0
                    )

                    # ── Watchlist trigger logic ───────────────────────────
                    triggers = []

                    if holder_conc_delta > 0.15:
                        triggers.append(
                            f"holder_concentration_delta: +{holder_conc_delta:.2f} (>0.15 threshold)"
                        )
                    if creator_pct_delta > 0.05:
                        triggers.append(
                            f"creator_percent_delta: +{creator_pct_delta:.4f} (>0.05 threshold)"
                        )
                    if liquidity_delta is not None and liquidity_delta < -0.5:
                        triggers.append(
                            f"liquidity_delta: {liquidity_delta:.2%} (<-50% threshold)"
                        )
                    if (volume_delta is not None and volume_delta > 5.0
                            and price_delta is not None and price_delta > 1.0):
                        triggers.append(
                            f"pump_pattern: volume_delta={volume_delta:.1f}x, "
                            f"price_delta=+{price_delta:.1%}"
                        )
                    if holder_exit_pct < -0.20:
                        triggers.append(
                            f"holder_count_delta: {holder_exit_pct:.1%} (<-20% threshold)"
                        )

                    if triggers:
                        reason   = "; ".join(triggers)
                        # Severity: critical if >1 trigger or any liquidity-drain/pump
                        critical_keywords = ("liquidity_delta", "pump_pattern")
                        severity = "critical" if (
                            len(triggers) > 1
                            or any(k in reason for k in critical_keywords)
                        ) else "high"

                        upsert_watchlist_item({
                            "token_address": token_addr,
                            "wallet_address": wallet_addr,
                            "agent_name":    profile.get("agent_name"),
                            "reason":        reason,
                            "severity":      severity,
                        })
                        watchlist_added += 1
                        logger.warning(
                            f"Watchlist: {token_addr[:10]}… "
                            f"severity={severity} | {reason[:80]}"
                        )

                except Exception as e:
                    err = f"snapshot/watchlist {profile.get('agent', '?')}: {e}"
                    logger.error(err)
                    errors.append(err)

            logger.info(
                f"Snapshots stored: {snapshots_stored}/{len(profiles)}  "
                f"Watchlist added: {watchlist_added}"
            )
        except Exception as e:
            logger.error(f"Snapshot/watchlist step failed: {e}", exc_info=True)
            errors.append(f"snapshots: {e}")
    else:
        logger.warning("Skipping snapshots (no profiles)")

    # ── Step 4 (old Step 3): Monte Carlo simulation ────────────────────────────────────────
    sim_results: list[dict] = []
    stored_sims = 0
    if profiles and time_left() > 120:
        try:
            from simulator.monte_carlo import run_all_simulations, print_simulation_summary
            from db.supabase_client import get_all_clusters, upsert_simulation_result

            clusters    = get_all_clusters()
            cluster_map = {}
            for cluster in clusters:
                for addr in (cluster.get("member_addresses") or []):
                    cluster_map[addr] = cluster["cluster_id"]

            # Use 50 runs (fast enough for daily refresh; full 100 for on-demand)
            logger.info(f"Running Monte Carlo (50 runs) for {len(profiles)} agents …")
            sim_results = run_all_simulations(
                profiles=profiles,
                n_runs=50,
                cluster_map=cluster_map,
            )
            print_simulation_summary(sim_results)

            for result in sim_results:
                if time_left() < 30:
                    logger.warning("Approaching timeout — stopping simulation storage")
                    break
                try:
                    upsert_simulation_result(result, cache_hours=24)
                    stored_sims += 1
                except Exception as e:
                    err = f"store_sim {result.get('agent')}: {e}"
                    logger.error(err)
                    errors.append(err)

            logger.info(f"Simulations stored: {stored_sims}/{len(sim_results)}")
        except Exception as e:
            logger.error(f"Simulation step failed: {e}", exc_info=True)
            errors.append(f"simulation: {e}")
    else:
        reason = "no profiles" if not profiles else "timeout"
        logger.warning(f"Skipping Monte Carlo ({reason})")

    # ── Step 5: Compute summary ────────────────────────────────────────────────
    duration    = elapsed()
    new_flags   = count_risk_flags(profiles, previous_profiles)
    status      = "success" if not errors else ("partial" if profiles else "failed")

    # Behavior distribution
    behavior_dist: dict[str, int] = {}
    for p in profiles:
        bt = p.get("behavior_type", "unknown")
        behavior_dist[bt] = behavior_dist.get(bt, 0) + 1

    summary = {
        "run_id":              run_id,
        "status":              status,
        "agents_profiled":     len(profiles),
        "simulations_run":     len(sim_results),
        "simulations_stored":  stored_sims,
        "snapshots_stored":    snapshots_stored,
        "watchlist_added":     watchlist_added,
        "new_risk_flags":      new_flags,
        "behavior_distribution": behavior_dist,
        "error_count":         len(errors),
        "errors":              errors[:10],    # cap for API response
        "duration_seconds":    round(duration, 2),
        "completed_at":        datetime.now(timezone.utc).isoformat(),
    }

    # ── Step 6: write cron log ─────────────────────────────────────────────────
    write_cron_log(
        run_id=run_id,
        status=status,
        agents_profiled=len(profiles),
        new_risk_flags=new_flags,
        errors=errors,
        duration_seconds=duration,
        extra={
            "simulations_run":    len(sim_results),
            "simulations_stored": stored_sims,
            "snapshots_stored":   snapshots_stored,
            "watchlist_added":    watchlist_added,
            "behavior_distribution": behavior_dist,
        },
    )

    # ── Print final summary ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("WADJET DAILY CRON COMPLETE")
    logger.info(f"  Status:          {status}")
    logger.info(f"  Agents profiled: {len(profiles)}")
    logger.info(f"  Simulations:     {stored_sims}/{len(sim_results)} stored")
    logger.info(f"  Snapshots:       {snapshots_stored} stored")
    logger.info(f"  Watchlist:       {watchlist_added} new entries")
    logger.info(f"  New risk flags:  {new_flags}")
    logger.info(f"  Errors:          {len(errors)}")
    logger.info(f"  Duration:        {duration:.1f}s")
    if behavior_dist:
        logger.info("  Behavior distribution:")
        for bt, cnt in sorted(behavior_dist.items(), key=lambda x: -x[1]):
            logger.info(f"    {bt:<18}: {cnt}")
    if errors:
        logger.info("  Error summary:")
        for err in errors[:5]:
            logger.info(f"    - {err}")
    logger.info("=" * 60)

    return summary


# ─── Alchemy client patcher ──────────────────────────────────────────────────

def _patch_alchemy_client():
    """
    Monkey-patch the alchemy_client module so every outbound call automatically
    retries on 429s / rate-limit errors.
    """
    try:
        import profiler.alchemy_client as _ac

        _orig_transfers   = _ac.get_asset_transfers
        _orig_balance     = _ac.get_balance
        _orig_tx_count    = _ac.get_transaction_count

        def _transfers_retry(address, **kwargs):
            return with_alchemy_retry(
                _orig_transfers, address, label=f"transfers:{address[:8]}", **kwargs
            )

        def _balance_retry(address):
            return with_alchemy_retry(
                _orig_balance, address, label=f"balance:{address[:8]}"
            )

        def _tx_count_retry(address):
            return with_alchemy_retry(
                _orig_tx_count, address, label=f"txcount:{address[:8]}"
            )

        _ac.get_asset_transfers      = _transfers_retry
        _ac.get_balance              = _balance_retry
        _ac.get_transaction_count    = _tx_count_retry

        logger.debug("Alchemy client patched with retry wrappers")
    except Exception as e:
        logger.warning(f"Could not patch alchemy client (non-fatal): {e}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import signal

    # Hard-kill after MAX_RUNTIME_SECONDS + 60s grace
    def _timeout_handler(sig, frame):
        logger.error(f"Hard timeout ({MAX_RUNTIME_SECONDS + 60}s) reached — exiting")
        sys.exit(1)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(MAX_RUNTIME_SECONDS + 60)

    try:
        summary = run_daily_cron()
        exit_code = 0 if summary["status"] == "success" else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unhandled error in daily cron: {e}", exc_info=True)
        sys.exit(1)

"""
Daily scheduler: re-runs simulations for top 500 agents every 24h.
Run this as a long-lived process: python -m wadjet.simulator.scheduler
"""

import logging
import os
import sys
import time
from datetime import datetime

import schedule

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("wadjet.scheduler")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def run_daily_simulation():
    """
    Full pipeline: fetch profiles → run simulations → store results.
    """
    logger.info("="*50)
    logger.info("DAILY SIMULATION RUN STARTED")
    logger.info(f"Time: {datetime.utcnow().isoformat()}Z")
    logger.info("="*50)

    try:
        from profiler.profile_builder import build_profiles
        from simulator.monte_carlo import run_all_simulations, print_simulation_summary
        from db.supabase_client import (
            ensure_schema, upsert_simulation_result, get_all_clusters
        )
        from profiler.graph_builder import get_cluster_id_for_address

        # Ensure DB schema exists
        ensure_schema()

        # Build profiles for top 500 agents
        logger.info("Building agent profiles...")
        profiles = build_profiles(limit=500)

        if not profiles:
            logger.warning("No profiles built — skipping simulation")
            return

        # Load cluster map
        clusters = get_all_clusters()
        cluster_map = {}
        for cluster in clusters:
            for addr in cluster.get("member_addresses", []):
                cluster_map[addr] = cluster["cluster_id"]

        # Run Monte Carlo simulations (50 runs for daily scheduler)
        logger.info(f"Running simulations for {len(profiles)} agents...")
        results = run_all_simulations(
            profiles=profiles,
            n_runs=50,
            cluster_map=cluster_map,
        )

        # Store results
        stored = 0
        for result in results:
            try:
                upsert_simulation_result(result, cache_hours=24)
                stored += 1
            except Exception as e:
                logger.error(f"Failed to store result for {result.get('agent')}: {e}")

        print_simulation_summary(results)
        logger.info(f"Daily simulation complete. {stored}/{len(results)} results stored.")

    except Exception as e:
        logger.error(f"Daily simulation FAILED: {e}", exc_info=True)


def main():
    logger.info("Wadjet scheduler started — running daily at 02:00 UTC")

    # Run immediately on startup
    run_daily_simulation()

    # Schedule daily at 02:00 UTC
    schedule.every().day.at("02:00").do(run_daily_simulation)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()

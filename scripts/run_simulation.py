#!/usr/bin/env python3
"""
Standalone Monte Carlo simulation runner.
Usage:
    cd packages/wadjet
    python scripts/run_simulation.py [--limit 100] [--runs 50]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def main():
    parser = argparse.ArgumentParser(description="Run Wadjet Monte Carlo simulations")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max agents to simulate (default: 100)")
    parser.add_argument("--runs", type=int, default=50,
                        help="Monte Carlo runs per scenario (default: 50)")
    args = parser.parse_args()

    from db.supabase_client import get_all_profiles, get_all_clusters, upsert_simulation_result
    from simulator.monte_carlo import run_all_simulations, print_simulation_summary
    import json

    # Load profiles from DB
    profiles_raw = get_all_profiles(limit=args.limit)
    print(f"Loaded {len(profiles_raw)} profiles from DB")

    if not profiles_raw:
        print("❌ No profiles found. Run the profiler first (scripts/run_profiler.py)")
        return

    # Normalize profiles for simulator
    profiles = []
    for p in profiles_raw:
        deps = p.get("dependencies", [])
        if isinstance(deps, str):
            try:
                deps = json.loads(deps)
            except Exception:
                deps = []
        profiles.append({
            "agent": p["address"],
            "behavior_type": p.get("behavior_type", "normal"),
            "risk_tolerance": p.get("risk_tolerance", "medium"),
            "dependencies": deps,
        })

    # Build cluster map
    clusters = get_all_clusters()
    cluster_map = {}
    for c in clusters:
        for addr in c.get("member_addresses", []):
            cluster_map[addr] = c["cluster_id"]

    # Run simulations
    results = run_all_simulations(
        profiles=profiles,
        n_runs=args.runs,
        cluster_map=cluster_map,
    )

    # Store results
    stored = 0
    for r in results:
        try:
            upsert_simulation_result(r, cache_hours=24)
            stored += 1
        except Exception as e:
            print(f"  Warn: failed to store {r.get('agent', '?')[:10]}: {e}")

    print_simulation_summary(results)
    print(f"✅ Done. {stored}/{len(results)} results stored in DB.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Standalone profiler runner.
Usage:
    cd packages/wadjet
    python scripts/run_profiler.py [--limit 500]
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
    parser = argparse.ArgumentParser(description="Run Wadjet agent profiler")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max agents to profile (default: 500)")
    args = parser.parse_args()

    from db.supabase_client import ensure_schema
    from profiler.profile_builder import build_profiles

    ensure_schema()
    profiles = build_profiles(limit=args.limit)
    print(f"\n✅ Done. Built {len(profiles)} profiles.")


if __name__ == "__main__":
    main()

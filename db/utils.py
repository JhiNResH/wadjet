"""Shared DB utilities for Wadjet scripts."""

import os


def get_db_url() -> str:
    """Get DATABASE_URL with pgbouncer/sslmode params stripped for psycopg2."""
    url = os.environ["DATABASE_URL"]
    # Strip query params that psycopg2 doesn't understand
    if "?" in url:
        base, params = url.split("?", 1)
        # Keep only params psycopg2 supports
        kept = []
        for p in params.split("&"):
            key = p.split("=")[0].lower()
            if key not in ("pgbouncer", "pgbouncer=true"):
                kept.append(p)
        url = base + ("?" + "&".join(kept) if kept else "")
    return url

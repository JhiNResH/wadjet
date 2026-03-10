"""
Agent Profile Builder — orchestrates data fetching + classification + graph.

Entry point: build_profiles(limit=500)
"""

import logging
import time
from typing import Optional

from profiler.alchemy_client import get_asset_transfers, get_balance, get_transaction_count
from profiler.classifier import classify_behavior, compute_risk_tolerance, extract_tx_features
from profiler.graph_builder import (
    build_global_graph,
    compute_dependencies,
    detect_clusters,
    detect_cycles,
    get_cluster_id_for_address,
)
from db.supabase_client import (
    fetch_agent_scores,
    upsert_agent_profile,
    upsert_clusters,
    upsert_graph_edges,
)

logger = logging.getLogger("wadjet.profiler")


def _extract_wallet(row: dict) -> Optional[str]:
    """Extract wallet address from agent_scores row."""
    # Primary: wallet_address column
    for key in ("wallet_address", "walletAddress", "address", "agentAddress"):
        val = row.get(key)
        if val and isinstance(val, str) and val.startswith("0x") and len(val) == 42:
            return val.lower()
    return None


def build_single_profile(
    address: str,
    agent_score_row: Optional[dict] = None,
    all_volumes: Optional[list[float]] = None,
) -> dict:
    """
    Build a complete behavior profile for a single address.
    """
    logger.info(f"Profiling {address}")

    # Fetch transfers (limit to last 200 txs for MVP speed)
    transfers = get_asset_transfers(address, direction="both", max_count=200)

    # Extract behavioral features
    features = extract_tx_features(transfers, address)

    # Classify behavior
    behavior_type = classify_behavior(features, agent_score_row, all_volumes)
    risk_tolerance = compute_risk_tolerance(behavior_type, features)

    # On-chain metrics
    balance = get_balance(address)
    tx_nonce = get_transaction_count(address)

    # Survival history (simple heuristic)
    span_days = features.get("span_days", 0)
    if span_days > 90:
        survival_history = f"Active {span_days:.0f}d, survived market cycles"
    elif span_days > 30:
        survival_history = f"Active {span_days:.0f}d, limited history"
    else:
        survival_history = f"New agent, {span_days:.0f}d old"

    # Compute metrics from features
    holder_concentration = max(0.0, min(1.0, 1.0 - features.get("two_sided", 0.5)))
    liquidity_locked = False  # Would need contract inspection
    tx_volume_spike = features.get("avg_daily_volume", 0) > 1000

    profile = {
        "agent": address,
        "behavior_type": behavior_type,
        "avg_daily_volume": round(features.get("avg_daily_volume", 0), 6),
        "counterparties": features.get("counterparties", 0),
        "risk_tolerance": risk_tolerance,
        "dependencies": [],  # Populated by graph builder
        "survival_history": survival_history,
        "metrics": {
            "holder_concentration": round(holder_concentration, 4),
            "liquidity_locked": liquidity_locked,
            "tx_volume_spike": tx_volume_spike,
            "new_counterparties_30d": features.get("new_counterparties_30d", 0),
            "tx_count": features.get("tx_count", 0),
            "balance_eth": round(balance, 6),
            "tx_nonce": tx_nonce,
            "span_days": round(span_days, 1),
            "buy_sell_ratio": round(features.get("buy_sell_ratio", 0.5), 4),
            "regularity": round(features.get("regularity", 0), 4),
        },
        "_transfers": transfers,  # Temporary, used for graph building
    }

    return profile


def build_profiles(limit: int = 500, delay_between: float = 0.3) -> list[dict]:
    """
    Main entry point: fetch agents from Supabase, profile each one.
    Returns list of profiles.
    """
    logger.info(f"Fetching up to {limit} agents from Supabase...")
    agent_rows = fetch_agent_scores(limit=limit)
    logger.info(f"Found {len(agent_rows)} agents")

    if not agent_rows:
        logger.warning("No agents found in agentScore table — using mock data")
        agent_rows = _mock_agent_rows()

    # Filter agents with wallet addresses
    agents_with_wallets = []
    for row in agent_rows:
        wallet = _extract_wallet(row)
        if wallet:
            agents_with_wallets.append((wallet, row))

    logger.info(f"{len(agents_with_wallets)} agents have wallet addresses")

    # First pass: collect all volumes (for whale detection)
    # We'll estimate from on-chain after profiling

    profiles = []
    all_transfers_by_agent: dict[str, list] = {}

    for i, (address, row) in enumerate(agents_with_wallets):
        try:
            profile = build_single_profile(address, agent_score_row=row)
            all_transfers_by_agent[address] = profile.pop("_transfers", [])
            profiles.append(profile)

            if (i + 1) % 10 == 0:
                logger.info(f"Profiled {i+1}/{len(agents_with_wallets)} agents")

            time.sleep(delay_between)  # Respect rate limits

        except Exception as e:
            logger.error(f"Failed to profile {address}: {e}")

    # Second pass: re-classify with whale detection
    all_volumes = [p.get("avg_daily_volume", 0) for p in profiles]
    for profile in profiles:
        if all_volumes:
            new_type = classify_behavior(
                {
                    "regularity": profile["metrics"].get("regularity", 0),
                    "two_sided": 1 - profile["metrics"].get("holder_concentration", 0.5),
                    "counterparties": profile["counterparties"],
                    "buy_sell_ratio": profile["metrics"].get("buy_sell_ratio", 0.5),
                    "first_tx_hold_hours": 999,
                    "total_volume": profile["avg_daily_volume"] * profile["metrics"].get("span_days", 1),
                    "span_days": profile["metrics"].get("span_days", 0),
                    "tx_count": profile["metrics"].get("tx_count", 0),
                    "avg_interval_s": 0,
                },
                all_volumes=all_volumes,
            )
            profile["behavior_type"] = new_type

    # Build global transaction graph
    logger.info("Building transaction relationship graph...")
    G = build_global_graph(all_transfers_by_agent)
    logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Detect clusters
    clusters = detect_clusters(G)
    logger.info(f"Detected {len(clusters)} clusters")

    # Detect wash trading cycles
    cycles = detect_cycles(G)
    logger.info(f"Detected {len(cycles)} potential wash trading cycles")

    # Enrich profiles with graph data
    for profile in profiles:
        address = profile["agent"]
        profile["dependencies"] = compute_dependencies(G, address)

    # Store graph edges in DB
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            "from_address": u,
            "to_address": v,
            "weight": data.get("weight", 0),
            "edge_type": "transfer",
            "tx_count": data.get("tx_count", 0),
            "total_value": data.get("weight", 0),
        })
    if edges:
        try:
            upsert_graph_edges(edges)
            logger.info(f"Stored {len(edges)} graph edges")
        except Exception as e:
            logger.error(f"Failed to store graph edges: {e}")

    # Store clusters in DB
    if clusters:
        try:
            upsert_clusters(clusters)
            logger.info(f"Stored {len(clusters)} clusters")
        except Exception as e:
            logger.error(f"Failed to store clusters: {e}")

    # Store profiles in DB
    stored = 0
    for profile in profiles:
        try:
            upsert_agent_profile(profile)
            stored += 1
        except Exception as e:
            logger.error(f"Failed to store profile for {profile.get('agent')}: {e}")

    # Print summary stats
    behavior_counts: dict[str, int] = {}
    for p in profiles:
        bt = p.get("behavior_type", "unknown")
        behavior_counts[bt] = behavior_counts.get(bt, 0) + 1

    print("\n" + "="*60)
    print(f"WADJET PROFILING SUMMARY")
    print("="*60)
    print(f"  Total agents fetched:    {len(agent_rows)}")
    print(f"  With wallet addresses:   {len(agents_with_wallets)}")
    print(f"  Profiles built:          {len(profiles)}")
    print(f"  Profiles stored in DB:   {stored}")
    print(f"  Graph nodes/edges:       {G.number_of_nodes()}/{G.number_of_edges()}")
    print(f"  Clusters detected:       {len(clusters)}")
    print(f"  Wash trading cycles:     {len(cycles)}")
    print(f"\n  Behavior Distribution:")
    for bt, count in sorted(behavior_counts.items(), key=lambda x: -x[1]):
        print(f"    {bt:<18}: {count}")
    print("="*60 + "\n")

    return profiles


def _mock_agent_rows() -> list[dict]:
    """Generate mock agent rows for testing when DB is empty."""
    import hashlib
    mock = []
    for i in range(20):
        seed = f"mock_agent_{i}"
        h = hashlib.sha256(seed.encode()).hexdigest()
        address = f"0x{h[:40]}"
        mock.append({
            "walletAddress": address,
            "trustScore": 0.3 + (i % 7) * 0.1,
            "completionRate": 0.5 + (i % 5) * 0.1,
            "totalJobs": i * 10,
        })
    return mock

"""
Transaction Relationship Graph builder.

Builds a directed weighted graph of agent-to-agent fund flows.
Detects:
  - Cycles (potential wash trading)
  - Dependency chains (A depends on B for X% of liquidity)
  - Hidden clusters via connected components
"""

import logging
from collections import defaultdict
from typing import Optional

import networkx as nx

logger = logging.getLogger("wadjet.graph")


def build_graph(transfers: list[dict], focus_address: str = None) -> nx.DiGraph:
    """
    Build a directed weighted graph from transfer records.
    Each edge: from_addr → to_addr, weight = total value transferred.
    """
    G = nx.DiGraph()
    edge_values: dict[tuple, float] = defaultdict(float)
    edge_counts: dict[tuple, int] = defaultdict(int)

    for tx in transfers:
        frm = (tx.get("from") or "").lower().strip()
        to  = (tx.get("to")   or "").lower().strip()
        if not frm or not to or frm == to:
            continue

        try:
            val = float(tx.get("value") or 0)
        except (ValueError, TypeError):
            val = 0.0

        edge_values[(frm, to)] += val
        edge_counts[(frm, to)] += 1

    for (frm, to), weight in edge_values.items():
        G.add_edge(frm, to, weight=weight, tx_count=edge_counts[(frm, to)])

    return G


def detect_cycles(G: nx.DiGraph) -> list[list[str]]:
    """Find simple cycles in the graph (wash trading candidates)."""
    try:
        cycles = list(nx.simple_cycles(G))
        # Filter meaningful cycles (length 2-5)
        return [c for c in cycles if 2 <= len(c) <= 5]
    except Exception as e:
        logger.warning(f"Cycle detection failed: {e}")
        return []


def compute_dependencies(G: nx.DiGraph, address: str) -> list[dict]:
    """
    For a given address, identify its top liquidity sources.
    Returns list of {address, weight, type}.
    """
    if address not in G:
        return []

    addr = address.lower()
    incoming = G.in_edges(addr, data=True)
    total_in = sum(d.get("weight", 0) for _, _, d in incoming)

    deps = []
    for src, _, data in G.in_edges(addr, data=True):
        w = data.get("weight", 0)
        rel_weight = w / (total_in + 1e-9)
        if rel_weight > 0.05:  # Only significant sources
            deps.append({
                "address": src,
                "weight": round(rel_weight, 4),
                "type": "liquidity_source",
                "tx_count": data.get("tx_count", 0),
            })

    deps.sort(key=lambda x: -x["weight"])
    return deps[:10]  # Top 10 dependencies


def detect_clusters(G: nx.DiGraph, min_cluster_size: int = 3) -> list[dict]:
    """
    Detect hidden clusters via weakly connected components.
    Large clusters may indicate coordinated behavior.
    """
    undirected = G.to_undirected()
    clusters = []

    for i, component in enumerate(nx.connected_components(undirected)):
        if len(component) < min_cluster_size:
            continue

        subgraph = G.subgraph(component)

        # Compute cluster risk: cycle density + edge concentration
        cycle_count = len(detect_cycles(subgraph))
        density = nx.density(subgraph)
        risk_score = min(1.0, 0.4 * density + 0.3 * min(cycle_count / 3, 1.0))

        cluster_type = "unknown"
        if cycle_count > 0:
            cluster_type = "wash_trading_ring"
        elif density > 0.5:
            cluster_type = "tight_cluster"
        elif len(component) > 20:
            cluster_type = "large_network"

        clusters.append({
            "cluster_id": i,
            "members": list(component),
            "size": len(component),
            "cluster_type": cluster_type,
            "risk_score": round(risk_score, 4),
            "cycle_count": cycle_count,
            "density": round(density, 4),
        })

    clusters.sort(key=lambda x: -x["risk_score"])
    return clusters


def get_cluster_id_for_address(clusters: list[dict], address: str) -> Optional[int]:
    addr = address.lower()
    for c in clusters:
        if addr in c.get("members", []):
            return c["cluster_id"]
    return None


def build_global_graph(all_transfers_by_agent: dict[str, list[dict]]) -> nx.DiGraph:
    """
    Build a single global graph from all agents' transfer data.
    all_transfers_by_agent: {address: [transfer_list]}
    """
    all_transfers = []
    for txs in all_transfers_by_agent.values():
        all_transfers.extend(txs)
    return build_graph(all_transfers)

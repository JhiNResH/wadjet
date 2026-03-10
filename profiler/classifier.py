"""
Behavior classifier — maps on-chain tx patterns to behavior types.

Behavior types:
  diamond_hands  — held through >20% dips without selling
  paper_hands    — sold within hours of >10% dip
  market_maker   — consistent two-sided activity, many counterparties
  sniper         — buys immediately after deployment, sells within hours
  whale          — top 5% by volume/holdings
  bot            — highly regular tx patterns, fixed intervals
  follower       — trades correlate with whale movements
  rug_deployer   — deployed tokens that went to zero
  normal         — none of the above
"""

import logging
import math
import statistics
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("wadjet.classifier")


# ─── Feature extraction ────────────────────────────────────────────────────────

def _parse_timestamp(ts: Optional[str]) -> Optional[float]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def extract_tx_features(transfers: list[dict], address: str) -> dict:
    """
    Compute behavioral features from raw transfer list.
    Returns a dict of numeric signals.
    """
    addr = address.lower()
    outgoing = [t for t in transfers if (t.get("from") or "").lower() == addr]
    incoming = [t for t in transfers if (t.get("to") or "").lower() == addr]

    # Unique counterparties
    cp_out = {(t.get("to") or "").lower() for t in outgoing if t.get("to")}
    cp_in  = {(t.get("from") or "").lower() for t in incoming if t.get("from")}
    counterparties = cp_out | cp_in

    # Timestamps
    timestamps = []
    for t in transfers:
        ts = _parse_timestamp(t.get("metadata", {}).get("blockTimestamp"))
        if ts:
            timestamps.append(ts)
    timestamps.sort()

    # Inter-tx intervals
    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    avg_interval = statistics.mean(intervals) if intervals else 0
    std_interval = statistics.stdev(intervals) if len(intervals) > 2 else float("inf")

    # Volume metrics
    def _value(t: dict) -> float:
        try:
            return float(t.get("value") or 0)
        except (ValueError, TypeError):
            return 0.0

    total_out = sum(_value(t) for t in outgoing)
    total_in  = sum(_value(t) for t in incoming)
    total_vol = total_out + total_in

    # Days active
    if timestamps:
        span_days = (timestamps[-1] - timestamps[0]) / 86400
    else:
        span_days = 0.0

    avg_daily_volume = total_vol / max(span_days, 1)

    # Buy/sell ratio (incoming=buy, outgoing=sell in DeFi context)
    buy_sell_ratio = total_in / (total_vol + 1e-9)

    # Timing regularity (bot detection)
    regularity = (
        1.0 / (1.0 + std_interval / (avg_interval + 1.0))
        if avg_interval > 0 else 0.0
    )

    # Two-sided activity (market maker signal)
    two_sided = min(len(outgoing), len(incoming)) / (max(len(outgoing), len(incoming)) + 1)

    # First tx to last tx speed (sniper detection)
    first_tx_hold_hours = (timestamps[1] - timestamps[0]) / 3600 if len(timestamps) >= 2 else 999

    # New counterparties in last 30 days
    cutoff_30d = (timestamps[-1] - 30 * 86400) if timestamps else 0
    recent_cps = set()
    for t in transfers:
        ts = _parse_timestamp(t.get("metadata", {}).get("blockTimestamp"))
        if ts and ts >= cutoff_30d:
            cp = (t.get("to") or t.get("from") or "").lower()
            if cp and cp != addr:
                recent_cps.add(cp)

    return {
        "tx_count": len(transfers),
        "outgoing_count": len(outgoing),
        "incoming_count": len(incoming),
        "counterparties": len(counterparties),
        "new_counterparties_30d": len(recent_cps),
        "total_volume": total_vol,
        "total_out": total_out,
        "total_in": total_in,
        "avg_daily_volume": avg_daily_volume,
        "span_days": span_days,
        "buy_sell_ratio": buy_sell_ratio,
        "avg_interval_s": avg_interval,
        "regularity": regularity,          # 0=random, 1=highly regular (bot)
        "two_sided": two_sided,            # 0=one-sided, 1=perfectly balanced
        "first_tx_hold_hours": first_tx_hold_hours,
    }


# ─── Classification logic ──────────────────────────────────────────────────────

def classify_behavior(
    features: dict,
    agent_score_row: Optional[dict] = None,
    all_volumes: Optional[list[float]] = None,
) -> str:
    """
    Classify agent behavior type based on extracted features.
    Priority order: rug_deployer > whale > bot > sniper > market_maker >
                    paper_hands > diamond_hands > follower > normal
    """
    tx_count   = features.get("tx_count", 0)
    regularity = features.get("regularity", 0)
    two_sided  = features.get("two_sided", 0)
    cps        = features.get("counterparties", 0)
    bsr        = features.get("buy_sell_ratio", 0.5)
    fth        = features.get("first_tx_hold_hours", 999)
    vol        = features.get("total_volume", 0)
    span_days  = features.get("span_days", 1)
    avg_iv     = features.get("avg_interval_s", 0)

    # rug_deployer: near-zero completion rate + very low trust
    if agent_score_row:
        comp = agent_score_row.get("completion_rate") or agent_score_row.get("completionRate") or 0
        trust = (agent_score_row.get("trust_score") or agent_score_row.get("trustScore") or 0)
        # trust_score is integer 0-100 in agent_scores
        trust_norm = trust / 100.0 if trust > 1 else trust
        if comp < 0.1 and trust_norm < 0.2 and tx_count > 5:
            return "rug_deployer"

    # whale: top 5% by volume
    if all_volumes and vol > 0:
        pct = sum(1 for v in all_volumes if v <= vol) / len(all_volumes)
        if pct >= 0.95:
            return "whale"

    # bot: high regularity (>0.85) + many txs + consistent interval
    if regularity > 0.85 and tx_count > 20 and avg_iv > 0:
        return "bot"

    # sniper: buys early, holds <4h, mostly outgoing after initial buy
    if fth < 4 and bsr < 0.3 and tx_count < 30:
        return "sniper"

    # market_maker: balanced two-sided + many counterparties
    if two_sided > 0.7 and cps > 20:
        return "market_maker"

    # paper_hands: mostly sells (low buy_sell_ratio) + fast exits
    if bsr < 0.3 and fth < 24:
        return "paper_hands"

    # diamond_hands: mostly buys / holds (high buy_sell_ratio)
    if bsr > 0.75 and span_days > 30:
        return "diamond_hands"

    # follower: relatively few unique counterparties but correlated timing
    # (simplified heuristic — full impl would require cross-agent correlation)
    if cps < 5 and tx_count > 10 and two_sided < 0.4:
        return "follower"

    return "normal"


def compute_risk_tolerance(behavior_type: str, features: dict) -> str:
    high_risk_types = {"sniper", "bot", "rug_deployer"}
    low_risk_types  = {"diamond_hands", "whale"}

    if behavior_type in high_risk_types:
        return "high"
    if behavior_type in low_risk_types:
        return "low"

    bsr = features.get("buy_sell_ratio", 0.5)
    if bsr < 0.3:
        return "high"
    if bsr > 0.7:
        return "low"
    return "medium"

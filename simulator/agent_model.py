"""
Agent behavioral models for the Monte Carlo simulation.

Each behavior type has a stress_response function that returns
survival probability given shock magnitude and current state.
"""

import math
import random
from typing import Optional


# ─── Behavior response parameters ─────────────────────────────────────────────
# Each type: (base_survival, shock_sensitivity, recovery_rate)
# base_survival: probability of surviving a 0% shock (baseline)
# shock_sensitivity: multiplier — how much each unit of shock hurts
# recovery_rate: how fast state recovers per round (0=no recovery, 1=full)

BEHAVIOR_PARAMS = {
    "diamond_hands":  (0.90, 0.30, 0.25),   # Strong holders, absorb dips
    "whale":          (0.75, 0.45, 0.30),   # Big but illiquid exits
    "market_maker":   (0.80, 0.55, 0.40),   # Exposed to both sides
    "normal":         (0.65, 0.60, 0.20),   # Average joe
    "follower":       (0.55, 0.75, 0.15),   # Copies whales, high correlation
    "paper_hands":    (0.40, 0.90, 0.05),   # Sells at first sign of stress
    "bot":            (0.70, 0.50, 0.50),   # Automated, adapts fast
    "sniper":         (0.45, 0.85, 0.10),   # Hit-and-run, high variance
    "rug_deployer":   (0.20, 0.95, 0.00),   # Already compromised
}

DEFAULT_PARAMS = (0.60, 0.65, 0.15)


def get_params(behavior_type: str) -> tuple[float, float, float]:
    return BEHAVIOR_PARAMS.get(behavior_type, DEFAULT_PARAMS)


# ─── Affected factor modifiers ─────────────────────────────────────────────────

FACTOR_SENSITIVITY = {
    # factor_name: {behavior_type: extra_multiplier}
    "price": {
        "diamond_hands": 0.5,   # Half as sensitive to price
        "paper_hands":   1.8,
        "sniper":        1.6,
        "whale":         1.2,
        "market_maker":  1.3,
        "bot":           0.9,
        "follower":      1.4,
        "normal":        1.0,
        "rug_deployer":  1.5,
    },
    "liquidity": {
        "market_maker":  1.8,
        "whale":         1.5,
        "diamond_hands": 0.7,
        "paper_hands":   1.4,
        "sniper":        1.2,
        "bot":           1.1,
        "follower":      1.3,
        "normal":        1.0,
        "rug_deployer":  0.8,
    },
    "gas": {
        "bot":           1.8,   # Bots are heavily impacted by gas
        "sniper":        1.5,
        "market_maker":  1.3,
        "whale":         0.7,   # Whales can absorb gas
        "diamond_hands": 0.6,
        "paper_hands":   1.1,
        "follower":      1.2,
        "normal":        1.0,
        "rug_deployer":  1.0,
    },
    "oracle": {
        "follower":      1.7,   # Depends on external signals
        "bot":           1.6,
        "market_maker":  1.4,
        "normal":        1.0,
        "whale":         0.9,
        "diamond_hands": 0.8,
        "paper_hands":   1.2,
        "sniper":        1.1,
        "rug_deployer":  0.9,
    },
    "counterparty": {
        "market_maker":  1.9,   # Highly dependent on partners
        "follower":      1.6,
        "whale":         1.3,
        "normal":        1.0,
        "bot":           0.9,
        "diamond_hands": 0.7,
        "paper_hands":   1.1,
        "sniper":        0.8,
        "rug_deployer":  0.6,
    },
}


def factor_sensitivity(behavior_type: str, affected_factor: str) -> float:
    """Return the factor sensitivity multiplier for this behavior type."""
    factor_map = FACTOR_SENSITIVITY.get(affected_factor, {})
    return factor_map.get(behavior_type, 1.0)


# ─── Single-run simulation ─────────────────────────────────────────────────────

class AgentState:
    """Mutable state of one agent during a simulation round."""

    def __init__(self, profile: dict):
        self.address = profile.get("agent", "unknown")
        self.behavior_type = profile.get("behavior_type", "normal")
        self.risk_tolerance = profile.get("risk_tolerance", "medium")
        self.dependencies = profile.get("dependencies", [])
        self.health = 1.0          # 1.0 = healthy, 0.0 = dead
        self.survived = True
        self.losses = []           # Loss per round

    def apply_shock(self, shock: float, affected_factor: str, rng: random.Random) -> float:
        """
        Apply a shock to this agent. Returns actual loss applied.
        """
        base_survival, sensitivity, _ = get_params(self.behavior_type)
        factor_mult = factor_sensitivity(self.behavior_type, affected_factor)

        # Effective shock after behavior filtering
        effective_shock = shock * sensitivity * factor_mult

        # Risk tolerance modifier
        rt_mod = {"low": 0.75, "medium": 1.0, "high": 1.25}.get(self.risk_tolerance, 1.0)
        effective_shock *= rt_mod

        # Add stochastic noise
        noise = rng.gauss(0, 0.05)
        effective_shock = max(0.0, min(1.0, effective_shock + noise))

        # Apply loss to health
        loss = effective_shock * self.health
        self.health = max(0.0, self.health - loss)
        self.losses.append(loss)

        # Check survival
        if self.health < 0.1:
            self.survived = False

        return loss

    def recover(self):
        """Apply partial recovery between rounds."""
        _, _, recovery_rate = get_params(self.behavior_type)
        if self.survived and self.health < 1.0:
            self.health = min(1.0, self.health + recovery_rate * (1.0 - self.health))

    @property
    def total_loss(self) -> float:
        return min(1.0, sum(self.losses))


def run_single_simulation(
    profile: dict,
    scenario,
    all_profiles_by_address: dict[str, dict],
    shock: float,
    seed: int,
) -> dict:
    """
    Run one Monte Carlo simulation for a single agent under a scenario.
    Returns: {"survived": bool, "total_loss": float, "cascade_triggered": bool}
    """
    rng = random.Random(seed)
    state = AgentState(profile)

    cascade_triggered = False

    for round_num in range(scenario.rounds):
        if not state.survived:
            break

        # Direct shock (decreases per round as system equilibrates)
        round_shock = shock * (0.7 ** round_num)  # Exponential decay
        state.apply_shock(round_shock, scenario.affected_factor, rng)

        # Cascade: if dependencies fail, additional shock
        for dep in state.dependencies:
            dep_addr = dep.get("address", "")
            dep_profile = all_profiles_by_address.get(dep_addr, {})
            dep_type = dep_profile.get("behavior_type", "normal")

            # Check if dependency is likely failing (simplified)
            dep_base, dep_sens, _ = get_params(dep_type)
            dep_failure_prob = shock * dep_sens * factor_sensitivity(dep_type, scenario.affected_factor)

            if rng.random() < dep_failure_prob * dep.get("weight", 0.5):
                cascade_shock = shock * dep.get("weight", 0.5) * scenario.cascade_multiplier
                cascade_shock *= 0.5 ** round_num  # Attenuate over rounds
                state.apply_shock(cascade_shock, scenario.affected_factor, rng)
                cascade_triggered = True

        state.recover()

    return {
        "survived": state.survived,
        "total_loss": round(state.total_loss, 4),
        "cascade_triggered": cascade_triggered,
        "final_health": round(state.health, 4),
    }

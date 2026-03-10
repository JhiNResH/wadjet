"""
Stress scenarios for Wadjet Monte Carlo simulations.

Each scenario defines:
  - id, name, description
  - initial_shock: dict of factors applied at t=0
  - propagation_fn: how the shock spreads each round
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import random


@dataclass
class Scenario:
    id: str
    name: str
    description: str
    # Initial shock parameters (randomized in MC runs)
    base_shock: float          # e.g. 0.5 = 50% value loss
    shock_std: float           # standard deviation for randomization
    affected_factor: str       # which factor is hit: "price"|"liquidity"|"gas"|"oracle"|"counterparty"
    rounds: int = 4            # how many rounds until equilibrium
    cascade_multiplier: float = 1.2  # how much shock amplifies per cascade


# ─── Scenario Definitions ──────────────────────────────────────────────────────

SCENARIOS: list[Scenario] = [
    Scenario(
        id="S1",
        name="Token Crash",
        description="A token drops 50% in 1 hour",
        base_shock=0.50,
        shock_std=0.15,
        affected_factor="price",
        rounds=4,
        cascade_multiplier=1.3,
    ),
    Scenario(
        id="S2",
        name="Whale Exit",
        description="Largest holder withdraws everything",
        base_shock=0.70,
        shock_std=0.10,
        affected_factor="liquidity",
        rounds=3,
        cascade_multiplier=1.5,
    ),
    Scenario(
        id="S3",
        name="Gas Spike",
        description="Gas costs 10x normal",
        base_shock=0.30,
        shock_std=0.05,
        affected_factor="gas",
        rounds=2,
        cascade_multiplier=1.1,
    ),
    Scenario(
        id="S4",
        name="Oracle Failure",
        description="Trust oracle offline for 4 hours",
        base_shock=0.40,
        shock_std=0.10,
        affected_factor="oracle",
        rounds=3,
        cascade_multiplier=1.2,
    ),
    Scenario(
        id="S5",
        name="Mass Withdrawal",
        description="30% of agents withdraw simultaneously",
        base_shock=0.60,
        shock_std=0.10,
        affected_factor="liquidity",
        rounds=5,
        cascade_multiplier=1.4,
    ),
    Scenario(
        id="S6",
        name="Counterparty Default",
        description="Major trading partner disappears",
        base_shock=0.55,
        shock_std=0.15,
        affected_factor="counterparty",
        rounds=4,
        cascade_multiplier=1.35,
    ),
    Scenario(
        id="S7",
        name="Regulatory Shock",
        description="Jurisdiction blocks agent trading",
        base_shock=0.45,
        shock_std=0.20,
        affected_factor="oracle",
        rounds=5,
        cascade_multiplier=1.15,
    ),
]

SCENARIO_MAP = {s.id: s for s in SCENARIOS}
SCENARIO_NAME_MAP = {s.name: s for s in SCENARIOS}


def get_scenario(identifier: str) -> Optional[Scenario]:
    return SCENARIO_MAP.get(identifier) or SCENARIO_NAME_MAP.get(identifier)


def randomize_shock(scenario: Scenario, seed: Optional[int] = None) -> float:
    """Return a randomized shock magnitude for a Monte Carlo run."""
    rng = random.Random(seed)
    shock = rng.gauss(scenario.base_shock, scenario.shock_std)
    return max(0.05, min(0.99, shock))  # Clamp to [0.05, 0.99]

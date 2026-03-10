"""
Monte Carlo runner for Wadjet stress simulations.

For each (agent, scenario) pair, runs N simulations with randomized parameters
and aggregates results into survival rates, cascade risks, and resilience scores.
"""

import logging
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from .agent_model import run_single_simulation
from .scenarios import SCENARIOS, Scenario, randomize_shock

logger = logging.getLogger("wadjet.montecarlo")

DEFAULT_N_RUNS = 100   # Monte Carlo runs per (agent, scenario)
DEFAULT_MAX_WORKERS = 4


# ─── Single scenario simulation ────────────────────────────────────────────────

def simulate_scenario(
    profile: dict,
    scenario: Scenario,
    all_profiles_by_address: dict[str, dict],
    n_runs: int = DEFAULT_N_RUNS,
    base_seed: int = 42,
) -> dict:
    """
    Run N Monte Carlo simulations for one agent under one scenario.
    Returns aggregated statistics.
    """
    survivors = 0
    total_losses = []
    cascade_count = 0

    for run_i in range(n_runs):
        seed = base_seed + run_i * 31337
        shock = randomize_shock(scenario, seed=seed)

        result = run_single_simulation(
            profile=profile,
            scenario=scenario,
            all_profiles_by_address=all_profiles_by_address,
            shock=shock,
            seed=seed,
        )

        if result["survived"]:
            survivors += 1
        total_losses.append(result["total_loss"])
        if result["cascade_triggered"]:
            cascade_count += 1

    survival_rate = survivors / n_runs
    avg_loss = sum(total_losses) / len(total_losses) if total_losses else 0.0
    losses_sorted = sorted(total_losses)
    var_95 = losses_sorted[int(0.95 * len(losses_sorted))] if losses_sorted else 0.0

    return {
        "scenario_id": scenario.id,
        "name": scenario.name,
        "survival_rate": round(survival_rate, 4),
        "avg_loss": round(avg_loss, 4),
        "var_95": round(var_95, 4),   # 95th-percentile loss
        "cascade_rate": round(cascade_count / n_runs, 4),
        "n_runs": n_runs,
    }


# ─── Cascade risk analysis ─────────────────────────────────────────────────────

def compute_cascade_risk(
    profile: dict,
    all_profiles_by_address: dict[str, dict],
    n_runs: int = 50,
    base_seed: int = 99,
) -> list[dict]:
    """
    For each direct dependency, simulate: if that dep fails (health→0),
    what is this agent's survival probability?
    """
    from .scenarios import Scenario

    cascade_risks = []

    # Create a synthetic "counterparty default" scenario for each dep
    for dep in profile.get("dependencies", [])[:5]:  # Top 5 deps
        dep_addr = dep.get("address", "")
        dep_weight = dep.get("weight", 0.5)

        # Synthesize a high-shock counterparty scenario
        synth_scenario = Scenario(
            id="SYNTH",
            name=f"Dep Fail {dep_addr[:8]}",
            description=f"If {dep_addr[:8]} fails",
            base_shock=dep_weight * 0.9,  # Shock proportional to dependency weight
            shock_std=0.05,
            affected_factor="counterparty",
            rounds=3,
            cascade_multiplier=1.5,
        )

        result = simulate_scenario(
            profile=profile,
            scenario=synth_scenario,
            all_profiles_by_address=all_profiles_by_address,
            n_runs=n_runs,
            base_seed=base_seed,
        )

        cascade_risks.append({
            "if_fails": dep_addr,
            "dependency_weight": dep_weight,
            "this_agent_survival": result["survival_rate"],
            "expected_loss": result["avg_loss"],
        })

    cascade_risks.sort(key=lambda x: x["this_agent_survival"])  # Worst first
    return cascade_risks


# ─── Resilience score ─────────────────────────────────────────────────────────

def compute_resilience_score(scenario_results: list[dict]) -> float:
    """
    Aggregate resilience score across all scenarios.
    0 = fragile (dies in every scenario)
    1 = antifragile (thrives under stress)
    """
    if not scenario_results:
        return 0.5

    # Weighted average of survival rates
    # Weight severe scenarios (high shock) more heavily
    weights = {
        "Token Crash":          1.5,
        "Whale Exit":           1.3,
        "Mass Withdrawal":      1.4,
        "Counterparty Default": 1.2,
        "Oracle Failure":       1.0,
        "Gas Spike":            0.8,
        "Regulatory Shock":     1.1,
    }

    total_w = 0.0
    weighted_survival = 0.0

    for sr in scenario_results:
        w = weights.get(sr["name"], 1.0)
        weighted_survival += sr["survival_rate"] * w
        total_w += w

    avg_survival = weighted_survival / total_w if total_w > 0 else 0.5

    # Penalize high cascade rate
    avg_cascade = sum(s.get("cascade_rate", 0) for s in scenario_results) / len(scenario_results)

    resilience = avg_survival * (1 - 0.3 * avg_cascade)
    return round(max(0.0, min(1.0, resilience)), 4)


# ─── Full agent simulation ─────────────────────────────────────────────────────

def simulate_agent(
    profile: dict,
    all_profiles_by_address: dict[str, dict],
    n_runs: int = DEFAULT_N_RUNS,
    cluster_id: Optional[int] = None,
) -> dict:
    """
    Run all 7 scenarios for one agent.
    Returns complete simulation result.
    """
    address = profile.get("agent", "unknown")
    logger.debug(f"Simulating {address}")

    scenario_results = []
    for scenario in SCENARIOS:
        result = simulate_scenario(
            profile=profile,
            scenario=scenario,
            all_profiles_by_address=all_profiles_by_address,
            n_runs=n_runs,
        )
        scenario_results.append(result)

    cascade_risk = compute_cascade_risk(
        profile=profile,
        all_profiles_by_address=all_profiles_by_address,
        n_runs=max(50, n_runs // 2),
    )

    resilience = compute_resilience_score(scenario_results)

    return {
        "agent": address,
        "scenarios": scenario_results,
        "cascade_risk": cascade_risk,
        "cluster_id": cluster_id,
        "resilience_score": resilience,
        "simulation_runs": n_runs,
        "behavior_type": profile.get("behavior_type", "normal"),
    }


# ─── Batch runner ─────────────────────────────────────────────────────────────

def run_all_simulations(
    profiles: list[dict],
    n_runs: int = DEFAULT_N_RUNS,
    cluster_map: Optional[dict[str, int]] = None,  # address → cluster_id
) -> list[dict]:
    """
    Run simulations for all agents.
    Returns list of simulation result dicts.
    """
    all_profiles_by_address = {p["agent"]: p for p in profiles}
    cluster_map = cluster_map or {}

    results = []
    total = len(profiles)

    for i, profile in enumerate(profiles):
        address = profile.get("agent", "")
        cluster_id = cluster_map.get(address)

        try:
            result = simulate_agent(
                profile=profile,
                all_profiles_by_address=all_profiles_by_address,
                n_runs=n_runs,
                cluster_id=cluster_id,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Simulation failed for {address}: {e}")

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"Simulated {i+1}/{total} agents")

    return results


def print_simulation_summary(results: list[dict]) -> None:
    """Print summary statistics after simulation run."""
    if not results:
        print("No results to summarize.")
        return

    all_resilience = [r["resilience_score"] for r in results]
    avg_resilience = sum(all_resilience) / len(all_resilience)

    # Per-scenario survival
    scenario_survivals: dict[str, list[float]] = {}
    for r in results:
        for sr in r.get("scenarios", []):
            name = sr["name"]
            scenario_survivals.setdefault(name, []).append(sr["survival_rate"])

    print("\n" + "="*65)
    print("WADJET MONTE CARLO SIMULATION SUMMARY")
    print("="*65)
    print(f"  Agents simulated:        {len(results)}")
    print(f"  Avg resilience score:    {avg_resilience:.3f}  (0=fragile, 1=antifragile)")
    print(f"  Agents with resilience")
    print(f"    > 0.7 (robust):        {sum(1 for r in all_resilience if r > 0.7)}")
    print(f"    0.3-0.7 (moderate):    {sum(1 for r in all_resilience if 0.3 <= r <= 0.7)}")
    print(f"    < 0.3 (fragile):       {sum(1 for r in all_resilience if r < 0.3)}")
    print(f"\n  Scenario Survival Rates (avg across agents):")
    for name, survivals in sorted(scenario_survivals.items()):
        avg_surv = sum(survivals) / len(survivals)
        print(f"    {name:<25}: {avg_surv:.1%}")
    print("="*65 + "\n")

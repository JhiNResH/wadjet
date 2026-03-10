# Wadjet — Risk Intelligence System

Wadjet is the risk intelligence layer of Maiat Protocol. It combines ML-powered rug detection (Phase 1) with agent behavior profiling and Monte Carlo stress simulation (Phase 2).

---

## Architecture

```
wadjet/
├── main.py                     # FastAPI service (all endpoints)
├── requirements.txt
├── Dockerfile
│
├── profiler/                   # Phase 2: Agent Profiling
│   ├── __init__.py
│   ├── alchemy_client.py       # Alchemy API (Base chain tx history)
│   ├── classifier.py           # Behavior type classification
│   ├── graph_builder.py        # Tx relationship graph + cluster detection
│   └── profile_builder.py      # Orchestrates profiling pipeline
│
├── simulator/                  # Phase 2: Monte Carlo Engine
│   ├── __init__.py
│   ├── scenarios.py            # 7 stress scenarios (S1–S7)
│   ├── agent_model.py          # Per-behavior stress response models
│   ├── monte_carlo.py          # MC runner + resilience scoring
│   └── scheduler.py            # Daily cron: top 500 agents
│
├── db/                         # Database layer
│   ├── __init__.py
│   └── supabase_client.py      # PostgreSQL connection + CRUD
│
├── models/                     # Phase 1: XGBoost model artifacts
│   ├── wadjet_xgb.joblib
│   ├── wadjet_xgb.onnx
│   └── model_metadata.json
│
├── scripts/                    # Training + data generation
│   ├── train_model.py
│   └── generate_dataset.py
│
└── data/
    └── rug_pull_dataset.csv
```

---

## Phase 2 Components

### Agent Profile Builder

Reads agents from Supabase `agentScore` table → fetches Base chain tx history from Alchemy → builds behavior profiles.

**Behavior Types:**

| Type | Signal | Risk |
|------|--------|------|
| `diamond_hands` | Held through >20% dips without selling | Low |
| `paper_hands` | Sold within hours of >10% dip | High |
| `market_maker` | Consistent two-sided activity, many counterparties | Medium |
| `sniper` | Buys right after deployment, exits within hours | High |
| `whale` | Top 5% by volume/holdings | Medium |
| `bot` | Highly regular tx patterns, fixed intervals | Medium |
| `follower` | Trades correlate with whale movements | High |
| `rug_deployer` | Deployed tokens that went to zero | Critical |
| `normal` | None of the above | Low |

### Monte Carlo Simulation Engine

Runs stress scenarios 50–100× with randomized parameters. Each round:
1. Apply shock → agents react per behavior type
2. Check dependency cascade (if dep fails, propagate shock)
3. Apply recovery rate → advance to next round

**Stress Scenarios:**

| ID | Name | Shock | Factor |
|----|------|-------|--------|
| S1 | Token Crash | 50% price drop in 1h | price |
| S2 | Whale Exit | Largest holder withdraws everything | liquidity |
| S3 | Gas Spike | Gas 10× normal | gas |
| S4 | Oracle Failure | Trust oracle offline 4h | oracle |
| S5 | Mass Withdrawal | 30% of agents withdraw simultaneously | liquidity |
| S6 | Counterparty Default | Major trading partner disappears | counterparty |
| S7 | Regulatory Shock | Jurisdiction blocks agent trading | oracle |

**Resilience Score:** 0 = fragile (fails every scenario), 1 = antifragile

### Transaction Relationship Graph

- **Cycle detection** → wash trading candidates
- **Dependency chains** → A depends on B for 60% of liquidity
- **Cluster detection** → hidden coordinated behavior via connected components

---

## API Endpoints

### Phase 1 (Existing)

```
POST /predict      — XGBoost rug pull prediction
GET  /health       — Health check
GET  /model-info   — Model metadata
```

### Phase 2 (New)

```
GET /wadjet/{address}
    Full risk profile + Monte Carlo simulation results.
    Auto-triggers simulation if not cached.

GET /wadjet/{address}/scenarios
    Detailed breakdown of all 7 stress scenarios.

GET /wadjet/portfolio?agents=0x...,0x...
    Portfolio risk assessment for up to 50 agents.
    Returns avg resilience, fragile/robust counts, portfolio risk score.

GET /wadjet/cascade/{address}
    Cascade risk map:
    - upstream_risks: which dependencies put this agent at risk
    - downstream_impact: which agents this agent's failure would cascade to

GET /wadjet/clusters
    All detected hidden clusters with risk scores and member lists.
```

---

## Running

### Start the API server

```bash
cd packages/wadjet
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### Run profiling pipeline (once or manually)

```bash
cd packages/wadjet
python -c "
from db.supabase_client import ensure_schema
from profiler.profile_builder import build_profiles
ensure_schema()
profiles = build_profiles(limit=500)
print(f'Built {len(profiles)} profiles')
"
```

### Run Monte Carlo simulation (once)

```bash
cd packages/wadjet
python -c "
from db.supabase_client import get_all_profiles
from simulator.monte_carlo import run_all_simulations, print_simulation_summary
from db.supabase_client import upsert_simulation_result

profiles_raw = get_all_profiles(limit=100)
profiles = [{'agent': p['address'], 'behavior_type': p['behavior_type'],
              'risk_tolerance': p['risk_tolerance'], 'dependencies': p.get('dependencies', [])}
            for p in profiles_raw]

results = run_all_simulations(profiles, n_runs=50)
print_simulation_summary(results)
for r in results:
    upsert_simulation_result(r, cache_hours=24)
"
```

### Start daily scheduler (long-running)

```bash
cd packages/wadjet
python -m simulator.scheduler
```

---

## Database Tables (Auto-created)

| Table | Purpose |
|-------|---------|
| `wadjet_agent_profiles` | Behavior profiles per agent |
| `wadjet_simulation_results` | Monte Carlo results, cached 24h |
| `wadjet_tx_graph` | Agent-to-agent fund flow graph |
| `wadjet_clusters` | Hidden cluster groups |

---

## Environment Variables

```env
DATABASE_URL=postgresql://...   # Supabase connection string
ALCHEMY_API_KEY=...             # Alchemy API key for Base chain
PORT=8001                       # FastAPI port
```

---

## Design Decisions

- **Python** for simulation (scipy, networkx, math-heavy)
- **Monte Carlo with 50–100 runs** per scenario: statistically meaningful without being slow
- **24h cache** for simulation results (expensive to compute)
- **Behavior models parameterized** by (base_survival, shock_sensitivity, recovery_rate) — easy to tune
- **Cascade shocks** attenuate exponentially per round — prevents infinite amplification
- **Top 500 agents** for MVP, designed to scale horizontally (parallelize by address shard)

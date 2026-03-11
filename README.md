# Wadjet 🐍

**Predictive risk intelligence for the agent economy.**

Wadjet is the unified data + ML engine behind [Maiat Protocol](https://maiat.io). It predicts rug pulls, monitors on-chain behavior in real-time, and closes the feedback loop automatically.

**Live:** `https://wadjet-production.up.railway.app`  
**Docs:** `https://wadjet-production.up.railway.app/docs` (Swagger UI)

---

## What It Does

### Data Collection
| Module | Frequency | Source |
|--------|-----------|--------|
| ACP Poller | Every 5 min | Virtuals ACP API |
| Price Tracker | Every 15 min | DexScreener |
| Chain Listener | Real-time | Base WSS (EAS + ERC-8004 events) |
| Virtuals Sync | Daily | Virtuals Protocol API |

### Analysis
| Module | What It Does |
|--------|-------------|
| **Predictor** | XGBoost rug detection — 98% accuracy, 50 features, trained on 18K+ real tokens |
| **Profiler** | Agent behavior classification (wash trading, ghost, rug deployer) |
| **Simulator** | Monte Carlo stress testing across 7 market crash scenarios |
| **Sentinel** | Two-stage real-time monitoring: GoPlus scan → sell pattern tracking → confirmed dump |

### Feedback Loop
| Module | What It Does |
|--------|-------------|
| **Auto-Outcomes** | Auto-classifies query results after 7 days (success/failure/scam) via DexScreener |
| **Health Signals** | Completion rate trends, LP drain rate, price volatility |

---

## Quick Start

### Predict rug risk for any token

```bash
curl -X POST https://wadjet-production.up.railway.app/predict/agent \
  -H "Content-Type: application/json" \
  -d '{"token_address": "0xYourTokenAddress"}'
```

Response:
```json
{
  "rug_score": 73,
  "risk_level": "critical",
  "behavior_class": "rug_pull",
  "dex_signals": { "price_usd": 0.0000072, "liquidity_usd": 10414, "holder_count": 382 },
  "goplus_signals": { "is_honeypot": false, "top10_holder_pct": 0.88, "lp_locked_pct": 0 },
  "risk_signals": [
    { "signal": "EXTREME_CONCENTRATION", "severity": "critical", "detail": "Top 10 holders own 88% of supply" },
    { "signal": "LP_UNLOCKED", "severity": "high", "detail": "0% of LP locked" }
  ],
  "summary": "🔴 HIGH RUG RISK (73%). Do NOT interact with this token."
}
```

### Simple prediction (with known features)

```bash
curl -X POST https://wadjet-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"trust_score": 0.8, "total_jobs": 0.5, "completion_rate": 0.9}'
```

---

## API Reference

### Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Predict rug probability from known features |
| `POST` | `/predict/agent` | Auto-fetch all data by token address (DexScreener + GoPlus + Maiat DB) |
| `GET` | `/wadjet/{address}` | Full risk profile + Monte Carlo simulation results |
| `GET` | `/wadjet/{address}/scenarios` | Detailed scenario breakdowns |
| `GET` | `/wadjet/portfolio` | Portfolio risk assessment (`?agents=0x...,0x...`) |
| `GET` | `/wadjet/cascade/{address}` | Cascade/contagion map for an agent |
| `GET` | `/wadjet/clusters` | All detected behavioral clusters |

### Sentinel (Real-time Monitoring)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/sentinel/scan` | Trigger Stage 1 GoPlus batch scan (🔑 X-Cron-Api-Key) |
| `POST` | `/sentinel/check-watchlist` | Trigger Stage 2 sell pattern check (🔑) |
| `GET` | `/sentinel/alerts` | List alerts (`?severity=critical&limit=10`) |
| `GET` | `/sentinel/alerts/{token}` | Alerts for a specific token |
| `GET` | `/sentinel/status` | Scan stats + watchlist count |

### Watchlist

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/watchlist` | All tokens on watchlist |
| `GET` | `/watchlist/{token}` | Watchlist entry for a specific token |

### Indexer / Data Collection

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/cron/index-agents` | Trigger ACP agent indexing (🔑) |
| `POST` | `/cron/sync-virtuals` | Trigger Virtuals token sync (🔑) |
| `POST` | `/cron/track-prices` | Trigger DexScreener price tracking (🔑) |
| `POST` | `/cron/run-daily` | Full daily cron (profiling + simulation + outcomes) (🔑) |
| `POST` | `/cron/auto-outcomes` | Backfill unreported query outcomes (🔑) |
| `GET` | `/indexer/status` | Status of all background pollers |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + model status |
| `GET` | `/model-info` | Model metadata and accuracy metrics |

🔑 = Requires `X-Cron-Api-Key` header

---

## Model Details

### V2 Agent-Enhanced XGBoost

- **Training data:** 18,296 real Uniswap V2 tokens + Virtuals agent tokens
- **Features:** 50 (20 base + 2 meta + 23 Virtuals-specific + 5 dynamic deltas)
- **Accuracy:** 98.1% | **Recall:** 98.8% | **False Negative:** 1.2%
- **Ensemble:** `max(ML_score, rule_based_score)` — prefers false positive over false negative

### Key Features
| Category | Features |
|----------|----------|
| On-chain | holder_concentration, liquidity, LP lock ratio, creator tx pattern |
| DexScreener | price_change, volume, buy/sell ratio, market cap |
| GoPlus | honeypot, mintable, hidden_owner, sell_tax, slippage_modifiable |
| ACP Behavioral | trust_score, total_jobs, completion_rate, unique_buyers |
| Meta | data_completeness, is_ghost_agent |
| Dynamic | holder_delta_1d, liquidity_delta_1d, volume_delta_1d, price_delta_1d, creator_percent_delta_1d |

### Scoring

```
rug_score = max(ML_prediction, rule_based_score) + goplus_delta
```

| Score | Risk Level | Action |
|-------|-----------|--------|
| 0-24 | 🟢 LOW | Proceed |
| 25-49 | 🟡 MEDIUM | Caution |
| 50-69 | 🔴 HIGH | Avoid |
| 70-100 | ⛔ CRITICAL | Do not interact |

### Sentinel Two-Stage Alert System

```
Stage 1 (hourly): GoPlus batch scan all tokens
  → Flag suspicious → add to watchlist

Stage 2 (every 10 min): Track watchlist tokens
  → SELL_SIGNAL → DUMP_PATTERN → CONFIRMED_DUMP → ABANDONMENT
  → Only escalate rug_score after behavioral confirmation
```

Single signals ≠ rug. The system requires behavioral sequence matching before raising alerts.

---

## Architecture

```
                    External APIs
                   ┌─────────────────┐
                   │ Virtuals ACP    │ ← 5min poll
                   │ DexScreener     │ ← 15min poll
                   │ GoPlus          │ ← hourly scan
                   │ Base WSS        │ ← real-time
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │     Wadjet      │
                   │   (FastAPI)     │
                   │                 │
                   │ ┌─────────────┐ │
                   │ │  Predictor  │ │ XGBoost V2
                   │ │  Profiler   │ │ Clustering
                   │ │  Simulator  │ │ Monte Carlo
                   │ │  Sentinel   │ │ 2-stage alerts
                   │ └─────────────┘ │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │    Supabase     │
                   │                 │
                   │ agent_scores    │
                   │ wadjet_alerts   │
                   │ wadjet_watchlist│
                   │ query_logs      │
                   └─────────────────┘
                            │
                   ┌────────▼────────┐
                   │  Maiat Protocol │
                   │   (Vercel)      │
                   │  app.maiat.io   │
                   └─────────────────┘
```

---

## Environment Variables

```env
DATABASE_URL          # Supabase Postgres (required)
CRON_API_KEY          # Protects /cron/* endpoints (required)
BASE_WSS_URL          # Base mainnet WebSocket (Alchemy)
BASE_RPC_URL          # Base mainnet RPC fallback
PORT                  # Server port (default: 8001, Railway overrides)
```

---

## Local Development

```bash
git clone https://github.com/JhiNResH/wadjet.git
cd wadjet
pip install -r requirements.txt
export DATABASE_URL="postgresql://..."
uvicorn main:app --reload --port 8001
```

---

## Related

- [Maiat Protocol](https://github.com/JhiNResH/maiat-protocol) — Trust oracle + API + frontend
- [Maiat SDK](https://www.npmjs.com/package/maiat-sdk) — TypeScript client
- [Dune Dashboard](https://dune.com/jhinresh/maiat-trust-infrastructure-base) — On-chain analytics

---

**Built by [Maiat Protocol](https://maiat.io)** — The trust layer for the agent economy.

# Wadjet — Railway Deployment Guide

## Live Service

🚀 **Production URL:** `https://wadjet-production.up.railway.app`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model status |
| `/predict` | POST | XGBoost rug prediction |
| `/model-info` | GET | Model metadata + feature importance |
| `/wadjet/{address}` | GET | Phase 2: Full agent risk profile |
| `/wadjet/portfolio` | GET | Phase 2: Portfolio risk assessment |
| `/wadjet/clusters` | GET | Phase 2: Hidden cluster detection |
| `/wadjet/cascade/{address}` | GET | Phase 2: Cascade risk map |

## Railway Project

- **Project:** `maiat-indexer`
- **Service:** `wadjet`
- **Environment:** `production`

## Environment Variables (set in Railway)

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | Supabase PostgreSQL connection string |
| `ALCHEMY_API_KEY` | Alchemy API key for on-chain data |
| `PORT` | Auto-set by Railway |

## Redeploy

```bash
cd packages/wadjet
railway up --detach
```

## View Logs

```bash
cd packages/wadjet
railway logs            # runtime logs (streaming)
railway logs --build    # build logs
railway logs --lines 50 # last 50 lines
```

## Local Development

```bash
cd packages/wadjet
pip install -r requirements.txt
PORT=8001 python main.py
curl http://localhost:8001/health
```

## Connecting from Vercel (Next.js)

Add to `.env.local`:
```
WADJET_API_URL=https://wadjet-production.up.railway.app
```

The TypeScript client (`src/lib/wadjet-client.ts`) reads `WADJET_API_URL` first,
falls back to `WADJET_URL`, then to `http://localhost:8001`.

## Architecture Notes

- **Model:** XGBoost CPU-only (`xgboost-cpu`) to avoid 293MB CUDA dependencies
- **Inference:** ~65ms avg latency on Railway's shared instances
- **Circuit breaker:** TypeScript client auto-falls back to rule-based on timeout/error
- **CORS:** Wildcard `*` — restrict to Vercel domain in production if needed

## First Deployment Notes (2026-03-10)

- Initial `xgboost==3.2.0` pulled `nvidia-nccl-cu12` (293MB CUDA dep) → OOM on startup
- Fixed by switching to `xgboost-cpu==3.2.0`
- Healthcheck timeout set to 300s (sklearn/xgboost model loading takes ~10s cold start)

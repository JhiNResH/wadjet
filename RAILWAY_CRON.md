# Wadjet — Cron Setup (Daily + Sentinel)

Three scheduled jobs:
| Job | Schedule | Script / Endpoint |
|-----|----------|-------------------|
| Daily re-profiling | `0 2 * * *` (02:00 UTC) | `scripts/daily_cron.py` / `POST /cron/run-daily` |
| Sentinel Stage 1 (GoPlus scan) | `0 * * * *` (hourly) | `POST /sentinel/scan` |
| Sentinel Stage 2 (watchlist) | `*/10 * * * *` (every 10 min) | `POST /sentinel/check-watchlist` |

Three deployment options — choose whichever fits your setup.

---

## Option A — Railway Cron Service (Recommended)

Railway supports a dedicated "Cron" service type that runs a command on a schedule.

### Steps

1. In your Railway project, click **New Service → Empty Service**.
2. Name it `wadjet-cron`.
3. Connect the same repo / Dockerfile as the `wadjet` service.
4. In **Settings → Deploy → Start Command**, set:
   ```
   python scripts/daily_cron.py
   ```
5. In **Settings → Deploy → Cron Schedule**, set:
   ```
   0 2 * * *
   ```
   (every day at 02:00 UTC)
6. Add the same environment variables as the main service:
   ```
   DATABASE_URL=<your Supabase connection string>
   ALCHEMY_API_KEY=<your key>
   CRON_API_KEY=<choose a secret>
   MAX_AGENTS=500
   MAX_RUNTIME_MIN=30
   ```
7. Deploy — Railway will spin up the container daily, run the script, then shut it down.

> **Tip:** Railway charges only for active cron runtime (~30 min/day ≈ pennies).

---

## Option B — `railway.toml` Cron Block

Add this to `packages/wadjet/railway.toml`:

```toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3

# ─── Daily cron service ───────────────────────────────────────────────────────
[[services]]
name = "wadjet-cron"

  [services.deploy]
  startCommand = "python scripts/daily_cron.py"
  cronSchedule = "0 2 * * *"
```

Then `railway up` — Railway will detect the second service and schedule it.

---

## Option C — GitHub Actions (External Cron)

Use GitHub Actions to POST to `/cron/run-daily` every day at 02:00 UTC.

### `.github/workflows/wadjet-daily-cron.yml`

```yaml
name: Wadjet Daily Cron

on:
  schedule:
    - cron: "0 2 * * *"   # 02:00 UTC daily
  workflow_dispatch:        # allow manual trigger

jobs:
  trigger-cron:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger Wadjet daily profiling
        run: |
          RESPONSE=$(curl -s -w "\n%{http_code}" \
            -X POST "${{ secrets.WADJET_API_URL }}/cron/run-daily" \
            -H "X-Cron-Api-Key: ${{ secrets.WADJET_CRON_API_KEY }}" \
            -H "Content-Type: application/json")

          HTTP_CODE=$(echo "$RESPONSE" | tail -1)
          BODY=$(echo "$RESPONSE" | head -1)

          echo "Response: $BODY"
          echo "HTTP code: $HTTP_CODE"

          if [[ "$HTTP_CODE" != "200" ]]; then
            echo "❌ Cron trigger failed with HTTP $HTTP_CODE"
            exit 1
          fi
          echo "✅ Cron triggered successfully"

      - name: Poll until complete (max 35 min)
        run: |
          for i in $(seq 1 35); do
            sleep 60
            STATUS=$(curl -s \
              "${{ secrets.WADJET_API_URL }}/cron/status" \
              -H "X-Cron-Api-Key: ${{ secrets.WADJET_CRON_API_KEY }}" \
              | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['running'])")
            echo "Minute $i: running=$STATUS"
            if [[ "$STATUS" == "False" ]]; then
              echo "✅ Cron complete"
              exit 0
            fi
          done
          echo "⏰ Timed out waiting for cron"
          exit 1
```

### Required GitHub Secrets

| Secret | Value |
|---|---|
| `WADJET_API_URL` | `https://your-wadjet.up.railway.app` |
| `WADJET_CRON_API_KEY` | Same value as `CRON_API_KEY` env var on Railway |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | ✅ | — | Supabase PostgreSQL URL |
| `ALCHEMY_API_KEY` | ✅ | — | Alchemy API key |
| `ALCHEMY_API_URL` | ❌ | `https://base-mainnet.g.alchemy.com/v2/` | Alchemy base URL |
| `CRON_API_KEY` | ✅ | `wadjet-cron-secret` | Protects cron + sentinel endpoints |
| `MAX_AGENTS` | ❌ | `500` | Max agents to re-profile |
| `MAX_RUNTIME_MIN` | ❌ | `30` | Hard timeout in minutes |
| `GOPLUS_BATCH_SIZE` | ❌ | `10` | Tokens per GoPlus batch (max 10/req) |

---

## Cron Log Table

Runs are logged to the `cron_logs` Supabase table (auto-created on first run):

```sql
SELECT run_id, status, agents_profiled, new_risk_flags,
       error_count, duration_seconds, ran_at
FROM cron_logs
ORDER BY ran_at DESC
LIMIT 10;
```

---

## Manual Run (Local / Testing)

### Daily cron
```bash
cd packages/wadjet
export DATABASE_URL="postgresql://..."
export ALCHEMY_API_KEY="<key>"
export MAX_AGENTS=10          # small batch for testing
python scripts/daily_cron.py
```

### Sentinel Stage 1 (GoPlus scan)
```bash
python scripts/sentinel.py --stage1
```

### Sentinel Stage 2 (watchlist check)
```bash
python scripts/sentinel.py --stage2
```

### Sentinel validation ($ELYS token)
```bash
python scripts/sentinel.py --validate
```

Or via the API:

```bash
# Daily cron
curl -X POST https://your-wadjet.up.railway.app/cron/run-daily \
  -H "X-Cron-Api-Key: wadjet-cron-secret"

# Sentinel Stage 1
curl -X POST https://your-wadjet.up.railway.app/sentinel/scan \
  -H "X-Cron-Api-Key: wadjet-cron-secret"

# Sentinel Stage 2
curl -X POST https://your-wadjet.up.railway.app/sentinel/check-watchlist \
  -H "X-Cron-Api-Key: wadjet-cron-secret"

# Sentinel status
curl https://your-wadjet.up.railway.app/sentinel/status \
  -H "X-Cron-Api-Key: wadjet-cron-secret"

# List critical alerts
curl "https://your-wadjet.up.railway.app/sentinel/alerts?severity=critical"
```

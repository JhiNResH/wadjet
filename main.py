"""
Wadjet — Risk Intelligence Microservice
FastAPI service for ML-powered rug pull detection + Monte Carlo risk simulation.

Phase 1 Endpoints:
  POST /predict    — Predict rug probability for an agent
  GET  /health     — Health check
  GET  /model-info — Model metadata and accuracy metrics

Phase 2 Endpoints (Agent Profiling + Monte Carlo):
  GET  /wadjet/{address}              — Full risk profile + simulation results
  GET  /wadjet/{address}/scenarios    — Detailed scenario breakdowns
  GET  /wadjet/portfolio              — Portfolio risk assessment (?agents=0x...,0x...)
  GET  /wadjet/cascade/{address}      — Cascade map for specific agent
  GET  /wadjet/clusters               — All detected hidden clusters

Phase 4 Endpoints (Sentinel — Real-time On-chain Monitoring):
  POST /sentinel/scan                 — Trigger Stage 1 hourly GoPlus scan (X-Cron-Api-Key)
  POST /sentinel/check-watchlist      — Trigger Stage 2 watchlist check (X-Cron-Api-Key)
  GET  /sentinel/alerts               — List recent alerts (?severity=&alert_type=&limit=)
  GET  /sentinel/alerts/{token}       — Alerts for specific token
  GET  /sentinel/status               — Scan times, watchlist count, alert counts
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import joblib
import numpy as np
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("wadjet")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "wadjet_xgb.joblib"
MODEL_V2_PATH = MODELS_DIR / "wadjet_xgb_v2_agent.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METADATA_V2_PATH = MODELS_DIR / "model_v2_metadata.json"

# ─── Feature ordering (MUST match training) ──────────────────────────────────
# Original 20 features (V1)
FEATURES_V1 = [
    "holder_concentration",
    "liquidity_lock_ratio",
    "creator_tx_pattern",
    "buy_sell_ratio",
    "contract_similarity_score",
    "fund_flow_pattern",
    "price_change_24h",
    "liquidity_usd",
    "volume_24h",
    "total_jobs",
    "completion_rate",
    "trust_score",
    "age_days",
    "lp_drain_rate",
    "deployer_age_days",
    "token_supply_concentration",
    "renounced_ownership",
    "verified_contract",
    "social_presence_score",
    "audit_score",
]

# Extended features for retrained models (V1 + 2 new)
FEATURES = FEATURES_V1 + [
    "data_completeness",   # Fraction of V1 features with real (non-default) values (0–1)
    "is_ghost_agent",      # 1 if total_jobs==0 AND completion_rate==0 AND token exists
]

# Default values for optional features (conservative — assume risky)
FEATURE_DEFAULTS = {
    "holder_concentration": 0.5,
    "liquidity_lock_ratio": 0.5,
    "creator_tx_pattern": 0.5,
    "buy_sell_ratio": 0.5,
    "contract_similarity_score": 0.3,
    "fund_flow_pattern": 0.3,
    "price_change_24h": 0.0,
    "liquidity_usd": 0.3,
    "volume_24h": 0.3,
    "total_jobs": 0.0,
    "completion_rate": 0.5,
    "trust_score": 0.5,
    "age_days": 0.1,
    "lp_drain_rate": 0.0,
    "deployer_age_days": 0.3,
    "token_supply_concentration": 0.3,
    "renounced_ownership": 0,
    "verified_contract": 0,
    "social_presence_score": 0.3,
    "audit_score": 0.0,
    # New features — computed, not supplied by caller
    "data_completeness": 0.0,
    "is_ghost_agent": 0.0,
}

# ─── Risk factor thresholds ────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "holder_concentration": (0.7, "Top holders control >70% of supply"),
    "liquidity_lock_ratio": (0.3, "Less than 30% of liquidity is locked"),
    "creator_tx_pattern": (0.6, "Deployer has suspicious transaction history"),
    "contract_similarity_score": (0.5, "Contract similar to known scams"),
    "fund_flow_pattern": (0.5, "Circular/wash trading detected"),
    "lp_drain_rate": (0.3, "Liquidity being drained"),
    "token_supply_concentration": (0.6, "Deployer holds >60% of token supply"),
}

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Wadjet Rug Prediction API",
    description="ML-powered rug pull detection for Maiat Protocol",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model loading ────────────────────────────────────────────────────────────
model = None
metadata = None
model_v2 = None
metadata_v2 = None


def load_model():
    global model, metadata, model_v2, metadata_v2

    # V1 model
    if not MODEL_PATH.exists():
        logger.warning(f"Model not found at {MODEL_PATH}. Run training first.")
        return False
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Loaded V1 model from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load V1 model: {e}")
        return False

    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        logger.info(f"Loaded V1 metadata v{metadata.get('model_version', '?')}")

    # V2 agent-enhanced model (optional — loads if available)
    if MODEL_V2_PATH.exists():
        try:
            model_v2 = joblib.load(MODEL_V2_PATH)
            logger.info(f"Loaded V2 agent model from {MODEL_V2_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load V2 model (non-fatal): {e}")

    if METADATA_V2_PATH.exists():
        with open(METADATA_V2_PATH) as f:
            metadata_v2 = json.load(f)
        logger.info(f"Loaded V2 metadata v{metadata_v2.get('model_version', '?')}")

    return True


@app.on_event("startup")
async def startup():
    loaded = load_model()
    if not loaded:
        logger.error("❌ Model not loaded — /predict will return 503")


# ─── Request / Response schemas ────────────────────────────────────────────
class PredictRequest(BaseModel):
    """
    Agent features for rug prediction.
    All features are normalized 0-1 unless noted.
    """

    # On-chain / DexScreener signals
    holder_concentration: Optional[float] = Field(None, ge=0, le=1,
        description="Top 10 holders % (0=distributed, 1=one holder)")
    liquidity_lock_ratio: Optional[float] = Field(None, ge=0, le=1,
        description="Fraction of LP tokens locked (0=none, 1=all)")
    creator_tx_pattern: Optional[float] = Field(None, ge=0, le=1,
        description="Deployer risk score (0=trusted, 1=suspicious)")
    buy_sell_ratio: Optional[float] = Field(None, ge=0, le=1,
        description="Buy / (buy+sell) ratio (0.5=balanced)")
    contract_similarity_score: Optional[float] = Field(None, ge=0, le=1,
        description="Similarity to known scam contracts (0=unique, 1=clone)")
    fund_flow_pattern: Optional[float] = Field(None, ge=0, le=1,
        description="Wash trading / circular flow score (0=clean, 1=suspicious)")
    price_change_24h: Optional[float] = Field(None,
        description="24h price change normalized (-1=crashed, 0=stable, +1=pumped)")
    liquidity_usd: Optional[float] = Field(None, ge=0, le=1,
        description="Log-normalized USD liquidity (0=empty, 1=deep)")
    volume_24h: Optional[float] = Field(None, ge=0, le=1,
        description="Log-normalized 24h volume (0=none, 1=high)")
    lp_drain_rate: Optional[float] = Field(None, ge=0, le=1,
        description="Rate of LP removal (0=stable, 1=draining)")
    deployer_age_days: Optional[float] = Field(None, ge=0, le=1,
        description="Log-normalized deployer wallet age (0=new, 1=old)")
    token_supply_concentration: Optional[float] = Field(None, ge=0, le=1,
        description="Fraction of supply held by deployer (0=distributed, 1=concentrated)")
    renounced_ownership: Optional[int] = Field(None, ge=0, le=1,
        description="1 if ownership renounced, 0 if not")
    verified_contract: Optional[int] = Field(None, ge=0, le=1,
        description="1 if contract verified on explorer, 0 if not")
    social_presence_score: Optional[float] = Field(None, ge=0, le=1,
        description="Community/social media activity score (0=none, 1=active)")
    audit_score: Optional[float] = Field(None, ge=0, le=1,
        description="External audit presence/quality (0=none, 1=fully audited)")
    age_days: Optional[float] = Field(None, ge=0, le=1,
        description="Log-normalized project age in days (0=new, 1=old)")

    # Maiat DB signals
    total_jobs: Optional[float] = Field(None, ge=0, le=1,
        description="Log-normalized total jobs completed (0=none, 1=many)")
    completion_rate: Optional[float] = Field(None, ge=0, le=1,
        description="Job completion rate (0=none, 1=perfect)")
    trust_score: Optional[float] = Field(None, ge=0, le=1,
        description="Maiat trust score (0=untrusted, 1=trusted)")

    # GoPlus on-chain security enrichment (optional)
    token_address: Optional[str] = Field(None,
        description="Token contract address for GoPlus security lookup")
    chain_id: Optional[int] = Field(8453,
        description="Chain ID for GoPlus lookup (default: 8453 = Base mainnet)")

    # Metadata
    agent_id: Optional[str] = Field(None, description="Agent identifier for logging")


class RiskFactor(BaseModel):
    feature: str
    value: float
    reason: str
    severity: str  # "low" | "medium" | "high" | "critical"


class GoPlusSignals(BaseModel):
    goplus_malicious: bool = False
    goplus_phishing: bool = False
    goplus_honeypot: bool = False
    goplus_proxy_contract: bool = False
    goplus_owner_can_change_balance: bool = False
    goplus_score_delta: int = 0
    goplus_flags: list[str] = []
    goplus_available: bool = False


class PredictResponse(BaseModel):
    rug_probability: float = Field(..., ge=0, le=1)
    rug_score: int = Field(..., ge=0, le=100)
    risk_level: str  # "low" | "medium" | "high" | "critical"
    behavior_class: str  # "clean" | "suspicious" | "rug_pull"
    confidence: float = Field(..., ge=0, le=1)
    risk_factors: list[RiskFactor]
    goplus: Optional[GoPlusSignals] = None
    summary: str
    model_version: str
    predicted_at: str


# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize_price_change(raw: float) -> float:
    """Normalize raw price change (-1 to +5) to model feature range."""
    return float(np.clip(raw, -1.0, 5.0))


def extract_features(req: PredictRequest) -> tuple:
    """
    Extract feature vector from request, applying defaults for missing values.

    Returns:
        feature_vec      — np.ndarray shape (1, 22) — full feature vector incl. new features
        real_count       — int: how many of the 20 original features had real (non-default) values
        low_data_confidence — bool: True if < 10 of 20 core features were provided
    """
    row = {}
    real_count = 0

    for feat in FEATURES_V1:
        val = getattr(req, feat, None)
        if val is not None:
            real_count += 1
            row[feat] = val
        else:
            row[feat] = FEATURE_DEFAULTS[feat]

    if req.price_change_24h is not None:
        row["price_change_24h"] = normalize_price_change(req.price_change_24h)

    # ── Computed new features ──────────────────────────────────────────────
    data_completeness = real_count / len(FEATURES_V1)  # 0.0 – 1.0
    is_ghost_agent = 1.0 if (
        row["total_jobs"] == 0.0
        and row["completion_rate"] == 0.0
        and (req.agent_id is not None or req.token_address is not None)
    ) else 0.0

    row["data_completeness"] = data_completeness
    row["is_ghost_agent"] = is_ghost_agent

    low_data_confidence = real_count < (len(FEATURES_V1) // 2)  # < 10 of 20

    feature_vec = np.array([[row[f] for f in FEATURES]], dtype=np.float32)
    return feature_vec, real_count, low_data_confidence


def compute_rule_based_score(
    trust_score_norm: float,        # 0–1 (0.0 = no trust)
    total_jobs_norm: float,         # 0–1 (0.0 = no jobs)
    completion_rate: float,         # 0–1
    holder_concentration: float,    # 0–1
    liquidity_norm: float,          # 0–1 log-normalised
    verified_contract: int,         # 0 or 1
    social_presence: float,         # 0–1
) -> int:
    """
    Rule-based rug risk score (0-100).  Mirrors the Vercel endpoint heuristics.
    Higher = riskier.
    """
    score = 0
    if trust_score_norm == 0.0:         score += 20
    if total_jobs_norm == 0.0:          score += 15
    if completion_rate == 0.0:          score += 15
    if holder_concentration > 0.7:      score += 15
    if liquidity_norm < 0.1:            score += 10
    if verified_contract == 0:          score += 5
    if social_presence == 0.0:          score += 5
    return min(score, 100)


def detect_risk_factors(req: PredictRequest) -> list[RiskFactor]:
    """Identify triggered risk factors for human-readable explanation."""
    factors = []

    def get(attr):
        return getattr(req, attr, None)

    def sev(score: float) -> str:
        if score >= 0.8: return "critical"
        if score >= 0.6: return "high"
        if score >= 0.4: return "medium"
        return "low"

    if (v := get("holder_concentration")) is not None and v > 0.7:
        factors.append(RiskFactor(
            feature="holder_concentration", value=v,
            reason=f"Top holders control {v*100:.0f}% of supply",
            severity=sev(v)
        ))

    if (v := get("liquidity_lock_ratio")) is not None and v < 0.3:
        inv = 1 - v
        factors.append(RiskFactor(
            feature="liquidity_lock_ratio", value=v,
            reason=f"Only {v*100:.0f}% of liquidity locked — easy exit",
            severity=sev(inv)
        ))

    if (v := get("creator_tx_pattern")) is not None and v > 0.5:
        factors.append(RiskFactor(
            feature="creator_tx_pattern", value=v,
            reason=f"Deployer risk score: {v*100:.0f}% — suspicious history",
            severity=sev(v)
        ))

    if (v := get("contract_similarity_score")) is not None and v > 0.5:
        factors.append(RiskFactor(
            feature="contract_similarity_score", value=v,
            reason=f"Contract {v*100:.0f}% similar to known scams",
            severity=sev(v)
        ))

    if (v := get("fund_flow_pattern")) is not None and v > 0.5:
        factors.append(RiskFactor(
            feature="fund_flow_pattern", value=v,
            reason=f"Wash trading / circular flow detected ({v*100:.0f}% confidence)",
            severity=sev(v)
        ))

    if (v := get("lp_drain_rate")) is not None and v > 0.3:
        factors.append(RiskFactor(
            feature="lp_drain_rate", value=v,
            reason=f"Liquidity drain rate: {v*100:.0f}% — possible exit in progress",
            severity=sev(v)
        ))

    if (v := get("token_supply_concentration")) is not None and v > 0.6:
        factors.append(RiskFactor(
            feature="token_supply_concentration", value=v,
            reason=f"Deployer holds {v*100:.0f}% of token supply",
            severity=sev(v)
        ))

    if (v := get("price_change_24h")) is not None and v < -0.5:
        factors.append(RiskFactor(
            feature="price_change_24h", value=v,
            reason=f"Price crashed {abs(v)*100:.0f}% in 24h",
            severity="critical" if v < -0.8 else "high"
        ))

    if (v := get("completion_rate")) is not None and v < 0.3:
        factors.append(RiskFactor(
            feature="completion_rate", value=v,
            reason=f"Low job completion rate: {v*100:.0f}%",
            severity=sev(1 - v)
        ))

    if (v := get("trust_score")) is not None and v < 0.3:
        factors.append(RiskFactor(
            feature="trust_score", value=v,
            reason=f"Low Maiat trust score: {v*100:.0f}/100",
            severity=sev(1 - v)
        ))

    if (v := get("renounced_ownership")) is not None and v == 0:
        factors.append(RiskFactor(
            feature="renounced_ownership", value=float(v),
            reason="Ownership NOT renounced — deployer retains control",
            severity="medium"
        ))

    if (v := get("verified_contract")) is not None and v == 0:
        factors.append(RiskFactor(
            feature="verified_contract", value=float(v),
            reason="Contract not verified on block explorer",
            severity="medium"
        ))

    if (v := get("audit_score")) is not None and v < 0.1:
        factors.append(RiskFactor(
            feature="audit_score", value=v,
            reason="No external security audit found",
            severity="medium"
        ))

    # Sort by severity
    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    factors.sort(key=lambda x: sev_order.get(x.severity, 99))

    return factors


def compute_confidence(req: PredictRequest) -> float:
    """Compute prediction confidence based on data completeness (V1 features only)."""
    provided = sum(
        1 for f in FEATURES_V1
        if getattr(req, f, None) is not None
    )
    return round(provided / len(FEATURES_V1), 2)


def classify_behavior(prob: float) -> str:
    if prob >= 0.7: return "rug_pull"
    if prob >= 0.4: return "suspicious"
    return "clean"


PREDICTION_THRESHOLD = 0.25  # Lowered from 0.35 to minimize false negatives (v1.2)

# ─── Phase 3: Agent-specific feature lists ────────────────────────────────────
V2_VIRTUALS_FEATURES = [
    "bonding_curve_position",
    "lp_locked_pct",
    "creator_other_tokens",
    "creator_wallet_age",
    "holder_count_norm",
    "top10_holder_pct",
    "acp_job_count",
    "acp_completion_rate",
    "acp_trust_score",
    "token_age_days_norm",
    "volume_to_mcap_ratio",
    "price_volatility_7d",
    "social_presence",
    "is_honeypot",
    "is_mintable",
    "hidden_owner",
    "slippage_modifiable",
    "buy_tax",
    "sell_tax",
    "creator_percent",
    "owner_percent",
    "has_dex_data",
    "is_virtuals_token",
    # Dynamic delta features (computed from daily snapshots)
    "holder_concentration_delta_1d",
    "liquidity_delta_1d",
    "volume_delta_1d",
    "price_delta_1d",
    "creator_percent_delta_1d",
]
ALL_V2_FEATURES = FEATURES + V2_VIRTUALS_FEATURES


# ─── Phase 3 request/response schemas ─────────────────────────────────────────

class AgentPredictRequest(BaseModel):
    """
    Agent-specific predict request for Virtuals protocol tokens.
    Accepts token_address + wallet_address and auto-fetches on-chain data.
    Or supply pre-fetched features directly (faster, no external calls).
    """
    # Required identifier
    token_address: str = Field(..., description="ERC-20 token contract address (Base)")
    wallet_address: Optional[str] = Field(None, description="Agent wallet / creator address")
    chain_id: Optional[int] = Field(8453, description="Chain ID (default: 8453 = Base)")

    # Pre-fetched override signals (if not provided, auto-fetched from APIs)
    # DexScreener
    liquidity_usd: Optional[float] = Field(None, ge=0, description="USD liquidity")
    volume_24h: Optional[float] = Field(None, ge=0, description="24h trading volume USD")
    market_cap: Optional[float] = Field(None, ge=0, description="Market cap USD")
    price_change_24h: Optional[float] = Field(None, description="24h price change %")
    pair_created_at: Optional[int] = Field(None, description="Pair creation unix ms timestamp")

    # ACP / Maiat DB signals (auto-fetched from DB if wallet_address provided)
    acp_trust_score: Optional[float] = Field(None, ge=0, le=100, description="Trust score 0-100")
    acp_total_jobs: Optional[int] = Field(None, ge=0, description="Total ACP jobs")
    acp_completion_rate: Optional[float] = Field(None, ge=0, le=1, description="Job completion rate")


class AgentRiskSignal(BaseModel):
    signal: str
    severity: str  # "critical" | "high" | "medium" | "info"
    detail: str


class AgentPredictResponse(BaseModel):
    token_address: str
    wallet_address: Optional[str]
    rug_probability: float
    rug_score: int
    risk_level: str
    behavior_class: str
    confidence: float
    model_version: str
    # Agent-specific enrichment
    dex_signals: dict
    goplus_signals: dict
    acp_signals: dict
    risk_signals: list[AgentRiskSignal]
    virtuals_features: dict
    summary: str
    predicted_at: str


# ─── Phase 3 helpers ─────────────────────────────────────────────────────────

async def _fetch_dexscreener(token_address: str, client: httpx.AsyncClient) -> dict:
    """Async DexScreener fetch."""
    try:
        resp = await client.get(
            f"https://api.dexscreener.com/latest/dex/tokens/{token_address}",
            timeout=10.0
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        pairs = data.get("pairs") or []
        if not pairs:
            return {}
        pair = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd", 0) or 0)
        return {
            "price_usd": float(pair.get("priceUsd") or 0),
            "price_change_24h": float((pair.get("priceChange") or {}).get("h24") or 0),
            "price_change_6h": float((pair.get("priceChange") or {}).get("h6") or 0),
            "volume_24h": float((pair.get("volume") or {}).get("h24") or 0),
            "liquidity_usd": float((pair.get("liquidity") or {}).get("usd") or 0),
            "market_cap": float(pair.get("marketCap") or 0),
            "pair_created_at": pair.get("pairCreatedAt"),
            "txns_24h_buys": int((pair.get("txns") or {}).get("h24", {}).get("buys") or 0),
            "txns_24h_sells": int((pair.get("txns") or {}).get("h24", {}).get("sells") or 0),
            "websites": [w.get("url","") for w in (pair.get("info") or {}).get("websites", [])],
            "socials": [s.get("url","") for s in (pair.get("info") or {}).get("socials", [])],
            "base_token": pair.get("baseToken", {}).get("symbol", ""),
            "dex_id": pair.get("dexId", ""),
        }
    except Exception as e:
        logger.debug(f"DexScreener fetch failed: {e}")
        return {}


async def _fetch_goplus_token(token_address: str, client: httpx.AsyncClient) -> dict:
    """Async GoPlus token security fetch."""
    try:
        resp = await client.get(
            f"https://api.gopluslabs.io/api/v1/token_security/8453",
            params={"contract_addresses": token_address},
            timeout=10.0,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        rd = (data.get("result") or {}).get(token_address.lower(), {})
        if not rd:
            return {}

        lp_locked = sum(
            float(h.get("percent", 0)) for h in rd.get("lp_holders", [])
            if h.get("is_locked") == 1
        )

        return {
            "is_honeypot": rd.get("is_honeypot", "0") == "1",
            "is_mintable": rd.get("is_mintable", "0") == "1",
            "hidden_owner": rd.get("hidden_owner", "0") == "1",
            "can_take_back_ownership": rd.get("can_take_back_ownership", "0") == "1",
            "slippage_modifiable": rd.get("slippage_modifiable", "0") == "1",
            "is_open_source": rd.get("is_open_source", "0") == "1",
            "buy_tax": float(rd.get("buy_tax") or 0),
            "sell_tax": float(rd.get("sell_tax") or 0),
            "holder_count": int(rd.get("holder_count") or 0),
            "top10_holder_pct": sum(
                float(h.get("percent", 0)) for h in rd.get("holders", [])[:10]
            ),
            "lp_locked_pct": lp_locked,
            "creator_address": rd.get("creator_address", ""),
            "creator_percent": float(rd.get("creator_percent") or 0),
            "owner_percent": float(rd.get("owner_percent") or 0),
        }
    except Exception as e:
        logger.debug(f"GoPlus fetch failed: {e}")
        return {}


def _fetch_acp_data(wallet_address: str) -> dict:
    """Fetch ACP behavioral data from Supabase."""
    try:
        import psycopg2
        db_url = os.environ.get(
            "DATABASE_URL",
            os.environ["DATABASE_URL"]
        )
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("""
            SELECT trust_score, completion_rate, total_jobs
            FROM agent_scores
            WHERE wallet_address = %s OR token_address = %s
            LIMIT 1
        """, (wallet_address, wallet_address))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return {
                "trust_score": row[0],
                "completion_rate": row[1],
                "total_jobs": row[2],
            }
    except Exception as e:
        logger.debug(f"ACP DB fetch failed: {e}")
    return {}


def _fetch_delta_features(token_address: str) -> dict:
    """
    Query wadjet_daily_snapshots for the last 2 days and compute deltas.
    Returns a dict with 5 delta features (all 0.0 if no history available).
    """
    defaults = {
        "holder_concentration_delta_1d": 0.0,
        "liquidity_delta_1d":            0.0,
        "volume_delta_1d":               0.0,
        "price_delta_1d":                0.0,
        "creator_percent_delta_1d":      0.0,
    }
    try:
        from db.supabase_client import get_last_two_snapshots
        rows = get_last_two_snapshots(token_address.lower())
        if len(rows) < 2:
            return defaults

        today = rows[0]   # most recent
        yest  = rows[1]   # previous

        def _rel(t, y):
            if y and y != 0:
                return (t - y) / abs(y)
            return 0.0

        today_top10  = today.get("top10_holder_pct") or 0.0
        yest_top10   = yest.get("top10_holder_pct") or 0.0
        today_liq    = today.get("liquidity_usd") or 0.0
        yest_liq     = yest.get("liquidity_usd") or 0.0
        today_vol    = today.get("volume_24h") or 0.0
        yest_vol     = yest.get("volume_24h") or 0.0
        today_price  = today.get("price_usd") or 0.0
        yest_price   = yest.get("price_usd") or 0.0
        today_creat  = today.get("creator_percent") or 0.0
        yest_creat   = yest.get("creator_percent") or 0.0

        return {
            "holder_concentration_delta_1d": float(today_top10 - yest_top10),
            "liquidity_delta_1d":            float(_rel(today_liq, yest_liq)),
            "volume_delta_1d":               float(_rel(today_vol, yest_vol)),
            "price_delta_1d":                float(_rel(today_price, yest_price)),
            "creator_percent_delta_1d":      float(today_creat - yest_creat),
        }
    except Exception as e:
        logger.debug(f"Delta fetch failed (non-fatal): {e}")
        return defaults


def _build_v2_feature_vector(dex: dict, gp: dict, acp: dict, req: AgentPredictRequest) -> np.ndarray:
    """Build complete V2 feature vector for agent token prediction."""
    import math

    # Token age
    created_at = req.pair_created_at or dex.get("pair_created_at")
    token_age_days = 0.0
    if created_at:
        try:
            import time as _time
            age_ms = _time.time() * 1000 - float(created_at)
            token_age_days = max(0, age_ms / (1000 * 86400))
        except Exception:
            pass

    # Trading ratios
    buys  = dex.get("txns_24h_buys", 0)
    sells = dex.get("txns_24h_sells", 0)
    buy_sell_ratio = buys / (buys + sells) if (buys + sells) > 0 else 0.5

    volume_24 = req.volume_24h or dex.get("volume_24h", 0)
    mcap      = req.market_cap or dex.get("market_cap", 0)
    liquidity = req.liquidity_usd or dex.get("liquidity_usd", 0)
    price24h  = req.price_change_24h if req.price_change_24h is not None else dex.get("price_change_24h", 0)

    vol_to_mcap = min(volume_24 / mcap, 10.0) if mcap > 0 else 0.0
    price_vol   = abs(price24h) / 100.0

    # Social
    websites = dex.get("websites", [])
    socials  = dex.get("socials", [])
    social_presence = min(1.0, (len(websites) + len(socials)) / 3.0)

    # GoPlus
    lp_locked    = min(gp.get("lp_locked_pct", 0.0), 1.0)
    top10        = min(gp.get("top10_holder_pct", 0.0), 1.0)
    holder_count = gp.get("holder_count", 0)
    is_open_src  = 1 if gp.get("is_open_source") else 0
    is_mintable  = 1 if gp.get("is_mintable") else 0
    hidden_owner = 1 if gp.get("hidden_owner") else 0
    slippage_mod = 1 if gp.get("slippage_modifiable") else 0
    buy_tax      = min(gp.get("buy_tax", 0), 1.0)
    sell_tax     = min(gp.get("sell_tax", 0), 1.0)
    is_honeypot  = 1 if gp.get("is_honeypot") else 0
    creator_pct  = gp.get("creator_percent", 0)
    owner_pct    = gp.get("owner_percent", 0)

    # ACP
    trust_raw  = req.acp_trust_score if req.acp_trust_score is not None else acp.get("trust_score", 0)
    total_jobs = req.acp_total_jobs if req.acp_total_jobs is not None else acp.get("total_jobs", 0)
    comp_rate  = req.acp_completion_rate if req.acp_completion_rate is not None else acp.get("completion_rate", 0)

    acp_trust_norm = min(trust_raw / 100.0, 1.0)
    log_jobs = math.log1p(total_jobs) / math.log1p(100000)
    holder_norm = math.log1p(holder_count) / math.log1p(100000)
    age_norm = math.log1p(token_age_days) / math.log1p(3650)
    liquidity_norm = math.log1p(liquidity) / math.log1p(1_000_000)
    volume_norm = math.log1p(volume_24) / math.log1p(1_000_000)
    price_norm = max(-1.0, min(1.0, price24h / 100.0))
    has_dex = 1 if dex else 0

    # V1 features (indices 0-19)
    v1 = [
        top10,                               # holder_concentration
        lp_locked,                           # liquidity_lock_ratio
        0.5,                                 # creator_tx_pattern (unknown)
        buy_sell_ratio,                      # buy_sell_ratio
        0.3,                                 # contract_similarity_score
        0.0,                                 # fund_flow_pattern
        price_norm,                          # price_change_24h
        liquidity_norm,                      # liquidity_usd
        volume_norm,                         # volume_24h
        log_jobs,                            # total_jobs
        min(comp_rate, 1.0),                 # completion_rate
        acp_trust_norm,                      # trust_score
        age_norm,                            # age_days
        0.0,                                 # lp_drain_rate
        0.5,                                 # deployer_age_days (unknown)
        max(creator_pct, owner_pct),         # token_supply_concentration
        0,                                   # renounced_ownership
        is_open_src,                         # verified_contract
        social_presence,                     # social_presence_score
        0.0,                                 # audit_score
    ]

    # V2 Virtuals features (indices 20-42)
    v2 = [
        0.0,                                 # bonding_curve_position
        lp_locked,                           # lp_locked_pct
        0.0,                                 # creator_other_tokens
        0.5,                                 # creator_wallet_age (unknown)
        holder_norm,                         # holder_count_norm
        top10,                               # top10_holder_pct
        log_jobs,                            # acp_job_count
        min(comp_rate, 1.0),                 # acp_completion_rate
        acp_trust_norm,                      # acp_trust_score
        age_norm,                            # token_age_days_norm
        vol_to_mcap,                         # volume_to_mcap_ratio
        price_vol,                           # price_volatility_7d
        social_presence,                     # social_presence
        float(is_honeypot),                  # is_honeypot
        float(is_mintable),                  # is_mintable
        float(hidden_owner),                 # hidden_owner
        float(slippage_mod),                 # slippage_modifiable
        buy_tax,                             # buy_tax
        sell_tax,                            # sell_tax
        creator_pct,                         # creator_percent
        owner_pct,                           # owner_percent
        float(has_dex),                      # has_dex_data
        1.0,                                 # is_virtuals_token
    ]

    # ── Delta features from daily snapshots ────────────────────────────────
    deltas = _fetch_delta_features(req.token_address)
    v_delta = [
        deltas["holder_concentration_delta_1d"],
        deltas["liquidity_delta_1d"],
        deltas["volume_delta_1d"],
        deltas["price_delta_1d"],
        deltas["creator_percent_delta_1d"],
    ]

    return np.array(v1 + v2 + v_delta, dtype=np.float32)


def _compute_agent_risk_signals(dex: dict, gp: dict, acp: dict, age_days: float) -> list[AgentRiskSignal]:
    """Generate human-readable risk signals for the agent token."""
    signals = []

    if gp.get("is_honeypot"):
        signals.append(AgentRiskSignal(signal="HONEYPOT", severity="critical",
            detail="GoPlus flags this token as a honeypot — you cannot sell"))
    if gp.get("hidden_owner"):
        signals.append(AgentRiskSignal(signal="HIDDEN_OWNER", severity="critical",
            detail="Contract has a hidden owner who can take back control"))
    if gp.get("slippage_modifiable"):
        signals.append(AgentRiskSignal(signal="SLIPPAGE_MODIFIABLE", severity="high",
            detail="Owner can arbitrarily set slippage, blocking sells"))

    sell_tax = gp.get("sell_tax", 0)
    if sell_tax >= 0.5:
        signals.append(AgentRiskSignal(signal="HIGH_SELL_TAX", severity="critical",
            detail=f"Sell tax is {sell_tax:.0%} — effectively blocks selling"))
    elif sell_tax >= 0.1:
        signals.append(AgentRiskSignal(signal="ELEVATED_SELL_TAX", severity="high",
            detail=f"Sell tax is {sell_tax:.0%}"))

    top10 = gp.get("top10_holder_pct", 0)
    if top10 > 0.85:
        signals.append(AgentRiskSignal(signal="EXTREME_CONCENTRATION", severity="critical",
            detail=f"Top 10 holders own {top10:.0%} of supply — extreme dump risk"))
    elif top10 > 0.7:
        signals.append(AgentRiskSignal(signal="HIGH_CONCENTRATION", severity="high",
            detail=f"Top 10 holders own {top10:.0%} of supply"))

    lp_locked = gp.get("lp_locked_pct", 0)
    if lp_locked < 0.1:
        signals.append(AgentRiskSignal(signal="LP_UNLOCKED", severity="high",
            detail=f"Only {lp_locked:.0%} of LP locked — easy liquidity removal"))

    price24h = dex.get("price_change_24h", 0)
    if price24h <= -70:
        signals.append(AgentRiskSignal(signal="PRICE_CRASH", severity="critical",
            detail=f"Price dropped {abs(price24h):.0f}% in 24h"))
    elif price24h <= -40:
        signals.append(AgentRiskSignal(signal="PRICE_DUMP", severity="high",
            detail=f"Price dropped {abs(price24h):.0f}% in 24h"))

    liquidity = dex.get("liquidity_usd", 0)
    mcap = dex.get("market_cap", 0)
    if mcap > 0 and liquidity < 500:
        signals.append(AgentRiskSignal(signal="LOW_LIQUIDITY", severity="high",
            detail=f"Liquidity is only ${liquidity:,.0f} — very easy to manipulate"))

    volume = dex.get("volume_24h", 0)
    if age_days > 14 and volume == 0:
        signals.append(AgentRiskSignal(signal="DEAD_VOLUME", severity="high",
            detail=f"No trading volume for a {age_days:.0f}-day-old token"))

    if mcap > 0 and volume > 0:
        vol_ratio = volume / mcap
        if vol_ratio > 5.0:
            signals.append(AgentRiskSignal(signal="WASH_TRADING", severity="high",
                detail=f"Volume/MCap ratio is {vol_ratio:.1f}x — likely wash trading"))

    trust = acp.get("trust_score", -1)
    jobs  = acp.get("total_jobs", -1)
    if trust == 0 and jobs == 0:
        signals.append(AgentRiskSignal(signal="GHOST_AGENT", severity="medium",
            detail="Agent has zero jobs and zero trust score — inactive/ghost"))

    if not gp.get("is_open_source"):
        signals.append(AgentRiskSignal(signal="UNVERIFIED_CONTRACT", severity="medium",
            detail="Contract source code is not verified on block explorer"))

    if gp.get("is_mintable"):
        signals.append(AgentRiskSignal(signal="MINTABLE", severity="medium",
            detail="Owner can mint unlimited tokens, diluting holders"))

    if not signals:
        signals.append(AgentRiskSignal(signal="NO_MAJOR_FLAGS", severity="info",
            detail="No critical on-chain risk flags detected"))

    return signals


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/model-info")
async def model_info():
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_version": metadata.get("model_version"),
        "training_date": metadata.get("training_date"),
        "algorithm": metadata.get("algorithm"),
        "n_features": metadata.get("n_features"),
        "features": metadata.get("features"),
        "metrics": metadata.get("metrics"),
        "dataset": metadata.get("dataset"),
        "deployment": metadata.get("deployment"),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training script first."
        )

    try:
        feature_vec, real_count, low_data_confidence = extract_features(req)

        # ── Backward-compat: truncate to model's expected n_features ─────────
        n_expected = model.n_features_in_ if hasattr(model, "n_features_in_") else len(FEATURES_V1)
        features_for_model = feature_vec[:, :n_expected]
        prob = float(model.predict_proba(features_for_model)[0][1])

        # ─── Rule-based score (always computed) ──────────────────────────────
        eff_trust      = (req.trust_score if req.trust_score is not None else FEATURE_DEFAULTS["trust_score"])
        eff_jobs       = (req.total_jobs if req.total_jobs is not None else FEATURE_DEFAULTS["total_jobs"])
        eff_completion = (req.completion_rate if req.completion_rate is not None else FEATURE_DEFAULTS["completion_rate"])
        eff_holder     = (req.holder_concentration if req.holder_concentration is not None else FEATURE_DEFAULTS["holder_concentration"])
        eff_liquidity  = (req.liquidity_usd if req.liquidity_usd is not None else FEATURE_DEFAULTS["liquidity_usd"])
        eff_verified   = (req.verified_contract if req.verified_contract is not None else FEATURE_DEFAULTS["verified_contract"])
        eff_social     = (req.social_presence_score if req.social_presence_score is not None else FEATURE_DEFAULTS["social_presence_score"])

        rule_based_score = compute_rule_based_score(
            trust_score_norm=eff_trust,
            total_jobs_norm=eff_jobs,
            completion_rate=eff_completion,
            holder_concentration=eff_holder,
            liquidity_norm=eff_liquidity,
            verified_contract=int(eff_verified),
            social_presence=eff_social,
        )

        # ─── GoPlus enrichment ───────────────────────────────────────────────
        goplus_signals: Optional[GoPlusSignals] = None
        goplus_score_delta = 0

        if req.token_address or req.agent_id:
            try:
                from data_sources.goplus_client import get_goplus_client
                gp = get_goplus_client()
                chain_id = req.chain_id or 8453

                # Use agent_id as wallet address if it looks like an 0x address
                wallet_addr = (
                    req.agent_id
                    if req.agent_id and req.agent_id.startswith("0x") and len(req.agent_id) == 42
                    else None
                )

                combined = await gp.get_combined_risk(
                    wallet_address=wallet_addr or "0x0000000000000000000000000000000000000000",
                    token_address=req.token_address,
                    chain_id=chain_id,
                )
                score_info = gp.compute_goplus_score_delta(
                    combined["wallet"], combined.get("token", {})
                )

                goplus_score_delta = score_info["score_delta"]
                goplus_signals = GoPlusSignals(
                    goplus_malicious=score_info["goplus_malicious"],
                    goplus_phishing=score_info["goplus_phishing"],
                    goplus_honeypot=score_info["goplus_honeypot"],
                    goplus_proxy_contract=score_info["goplus_proxy_contract"],
                    goplus_owner_can_change_balance=score_info["goplus_owner_can_change_balance"],
                    goplus_score_delta=goplus_score_delta,
                    goplus_flags=score_info["goplus_flags"],
                    goplus_available=score_info["goplus_available"],
                )

                if goplus_score_delta > 0:
                    logger.info(
                        f"GoPlus flags for {req.agent_id or req.token_address}: "
                        f"+{goplus_score_delta} rugScore, flags={score_info['goplus_flags']}"
                    )

            except Exception as e:
                logger.warning(f"GoPlus enrichment failed (non-fatal): {e}")

        # ─── Compute final score ─────────────────────────────────────────────
        is_rug = prob >= PREDICTION_THRESHOLD
        ml_rug_score = int(round(prob * 100))

        # Ensemble: take max(ML, rule-based) — rule-based guards against low-data failures
        rug_score = max(ml_rug_score, rule_based_score)

        # GoPlus delta on top of ensemble score
        rug_score = min(100, rug_score + goplus_score_delta)

        # ─── Zero-activity penalty (post-processing floor) ───────────────────
        # If no evidence of activity (both omitted/zero), enforce floor of 40
        zero_jobs  = req.total_jobs is None or req.total_jobs == 0
        zero_trust = req.trust_score is None or req.trust_score == 0
        if zero_jobs and zero_trust:
            rug_score = max(rug_score, 40)

        # Re-derive risk level from augmented score
        aug_prob = rug_score / 100.0
        risk_level = (
            "critical" if aug_prob >= 0.7 else
            "high"     if aug_prob >= 0.5 else
            "medium"   if aug_prob >= 0.35 else
            "low"
        )
        behavior_class = classify_behavior(aug_prob)
        confidence = compute_confidence(req)
        risk_factors = detect_risk_factors(req)

        # Add GoPlus risk factors to the list
        if goplus_signals and goplus_signals.goplus_available:
            if goplus_signals.goplus_malicious:
                risk_factors.insert(0, RiskFactor(
                    feature="goplus_malicious", value=1.0,
                    reason="GoPlus: Address flagged as malicious",
                    severity="critical"
                ))
            if goplus_signals.goplus_phishing:
                risk_factors.insert(0, RiskFactor(
                    feature="goplus_phishing", value=1.0,
                    reason="GoPlus: Address linked to phishing activities",
                    severity="critical"
                ))
            if goplus_signals.goplus_honeypot:
                risk_factors.insert(0, RiskFactor(
                    feature="goplus_honeypot", value=1.0,
                    reason="GoPlus: Token identified as honeypot — cannot sell",
                    severity="critical"
                ))
            if goplus_signals.goplus_owner_can_change_balance:
                risk_factors.append(RiskFactor(
                    feature="goplus_owner_can_change_balance", value=1.0,
                    reason="GoPlus: Contract owner can arbitrarily change balances",
                    severity="high"
                ))

        goplus_note = (
            f" GoPlus added +{goplus_score_delta} to risk score ({', '.join(goplus_signals.goplus_flags)})."
            if goplus_signals and goplus_score_delta > 0
            else ""
        )

        summary = (
            f"High rug pull probability ({aug_prob*100:.0f}%). "
            f"{len([f for f in risk_factors if f.severity in ('critical','high')])} critical/high risk factors."
            f"{goplus_note}"
            if aug_prob >= PREDICTION_THRESHOLD else
            f"Moderate risk ({aug_prob*100:.0f}%). Monitor closely.{goplus_note}" if aug_prob >= 0.35 else
            f"Low rug risk ({aug_prob*100:.0f}%). Proceed with standard due diligence."
        )

        if req.agent_id:
            logger.info(
                f"Prediction: agent={req.agent_id} prob={prob:.4f} "
                f"ml_score={ml_rug_score} rule_based={rule_based_score} "
                f"rug_score={rug_score} (goplus_delta={goplus_score_delta}) "
                f"real_features={real_count}/20 low_data={low_data_confidence} "
                f"risk={risk_level} behavior={behavior_class}"
            )

        return PredictResponse(
            rug_probability=round(aug_prob, 4),
            rug_score=rug_score,
            risk_level=risk_level,
            behavior_class=behavior_class,
            confidence=confidence,
            risk_factors=risk_factors,
            goplus=goplus_signals,
            summary=summary,
            model_version=metadata.get("model_version", "1.0.0") if metadata else "1.0.0",
            predicted_at=datetime.utcnow().isoformat() + "Z",
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ─── Phase 3: Agent Token Prediction ─────────────────────────────────────────

@app.post("/predict/agent", response_model=AgentPredictResponse, tags=["Phase 3: Agent Rug Detection"])
async def predict_agent(req: AgentPredictRequest):
    """
    Predict rug pull risk for a Virtuals protocol agent token.

    Auto-fetches on-chain data from:
    - DexScreener (price, liquidity, volume, social links)
    - GoPlus (honeypot, sell tax, holder concentration, LP lock)
    - Maiat DB (ACP trust score, job completion, activity)

    Uses V2 agent-enhanced XGBoost model if available, falls back to V1.

    Returns enhanced prediction with Virtuals-specific risk signals.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    token_address = req.token_address.lower()

    # ─── Parallel data fetching ─────────────────────────────────────────────
    dex_data: dict = {}
    gp_data: dict = {}
    acp_data: dict = {}

    async with httpx.AsyncClient() as client:
        tasks = [
            _fetch_dexscreener(token_address, client),
            _fetch_goplus_token(token_address, client),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    dex_data = results[0] if isinstance(results[0], dict) else {}
    gp_data  = results[1] if isinstance(results[1], dict) else {}

    # ACP data — sync DB call (quick)
    if req.wallet_address or req.acp_trust_score is None:
        lookup = req.wallet_address or token_address
        try:
            acp_data = _fetch_acp_data(lookup)
        except Exception as e:
            logger.debug(f"ACP fetch non-fatal: {e}")

    # Merge in any pre-provided ACP fields from request
    if req.acp_trust_score is not None:
        acp_data["trust_score"] = req.acp_trust_score
    if req.acp_total_jobs is not None:
        acp_data["total_jobs"] = req.acp_total_jobs
    if req.acp_completion_rate is not None:
        acp_data["completion_rate"] = req.acp_completion_rate

    # ─── Build feature vector ───────────────────────────────────────────────
    try:
        feature_vec = _build_v2_feature_vector(dex_data, gp_data, acp_data, req)
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")

    # ─── Run model ─────────────────────────────────────────────────────────
    active_model = model_v2 or model
    active_metadata = metadata_v2 or metadata
    model_ver = (active_metadata or {}).get("model_version", "1.0.0")
    model_tag = "v2_agent" if model_v2 else "v1_generic"

    try:
        n_expected = active_model.n_features_in_ if hasattr(active_model, 'n_features_in_') else 20
        input_vec = feature_vec[:n_expected].reshape(1, -1)
        rug_prob = float(active_model.predict_proba(input_vec)[0, 1])
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    ml_rug_score = int(round(rug_prob * 100))

    # ─── Rule-based score for agent endpoint ────────────────────────────────
    import math as _math_rb
    acp_trust_raw   = acp_data.get("trust_score", 0) or 0
    acp_total_jobs  = acp_data.get("total_jobs", 0) or 0
    acp_comp_rate   = acp_data.get("completion_rate", 0) or 0
    agent_liq_usd   = req.liquidity_usd or dex_data.get("liquidity_usd", 0) or 0
    agent_liq_norm  = _math_rb.log1p(agent_liq_usd) / _math_rb.log1p(1_000_000)
    agent_top10     = gp_data.get("top10_holder_pct", 0.5) or 0.5
    agent_verified  = 1 if gp_data.get("is_open_source") else 0
    agent_social    = float(len(dex_data.get("websites", [])) + len(dex_data.get("socials", []))) / 3.0
    agent_social    = min(agent_social, 1.0)

    # Normalize trust (0-100 → 0-1) and jobs (log-norm)
    acp_trust_norm = min(acp_trust_raw / 100.0, 1.0)
    acp_jobs_norm  = _math_rb.log1p(acp_total_jobs) / _math_rb.log1p(100_000)

    rule_based_score = compute_rule_based_score(
        trust_score_norm=acp_trust_norm,
        total_jobs_norm=acp_jobs_norm,
        completion_rate=float(acp_comp_rate),
        holder_concentration=float(agent_top10),
        liquidity_norm=agent_liq_norm,
        verified_contract=agent_verified,
        social_presence=agent_social,
    )

    # Ensemble: max(ML, rule-based)
    rug_score = max(ml_rug_score, rule_based_score)

    # ─── Zero-activity penalty ───────────────────────────────────────────────
    zero_agent_jobs  = (req.acp_total_jobs is None or req.acp_total_jobs == 0) and acp_total_jobs == 0
    zero_agent_trust = (req.acp_trust_score is None or req.acp_trust_score == 0) and acp_trust_raw == 0
    if zero_agent_jobs and zero_agent_trust:
        rug_score = max(rug_score, 40)

    rug_prob_final = rug_score / 100.0
    risk_level = (
        "critical" if rug_prob_final >= 0.7 else
        "high"     if rug_prob_final >= 0.5 else
        "medium"   if rug_prob_final >= PREDICTION_THRESHOLD else
        "low"
    )
    behavior_class = classify_behavior(rug_prob_final)

    # ─── Token age for signals ──────────────────────────────────────────────
    import time as _time
    import math as _math
    created_at = req.pair_created_at or dex_data.get("pair_created_at")
    token_age_days = 0.0
    if created_at:
        try:
            token_age_days = max(0, (_time.time() * 1000 - float(created_at)) / (1000 * 86400))
        except Exception:
            pass

    # ─── Risk signals ───────────────────────────────────────────────────────
    risk_signals = _compute_agent_risk_signals(dex_data, gp_data, acp_data, token_age_days)

    # Confidence: based on data completeness
    data_points = sum([
        1 if dex_data and not dex_data.get("no_data") else 0,
        1 if gp_data else 0,
        1 if acp_data else 0,
        1 if req.acp_trust_score is not None else 0,
    ])
    confidence = round(0.4 + (data_points / 4.0) * 0.6, 2)

    # ─── Summary ────────────────────────────────────────────────────────────
    critical_count = sum(1 for s in risk_signals if s.severity == "critical")
    high_count = sum(1 for s in risk_signals if s.severity == "high")

    if rug_prob_final >= 0.7 or critical_count > 0:
        summary = (
            f"🔴 HIGH RUG RISK ({rug_prob_final*100:.0f}%). "
            f"{critical_count} critical + {high_count} high signals. "
            f"Do NOT interact with this token."
        )
    elif rug_prob_final >= PREDICTION_THRESHOLD:
        summary = (
            f"🟠 ELEVATED RISK ({rug_prob_final*100:.0f}%). "
            f"{high_count} high-risk signals detected. Exercise caution."
        )
    else:
        summary = (
            f"🟢 LOW RISK ({rug_prob_final*100:.0f}%). "
            f"No major red flags. Standard due diligence applies."
        )

    logger.info(
        f"Agent prediction: token={token_address[:12]}... "
        f"ml_prob={rug_prob:.3f} ml_score={ml_rug_score} rule_based={rule_based_score} "
        f"rug_score={rug_score} risk={risk_level} "
        f"model={model_tag} signals={len(risk_signals)}"
    )

    # ─── Virtuals feature summary for transparency ─────────────────────────
    virtuals_features = {
        "token_age_days": round(token_age_days, 1),
        "lp_locked_pct": gp_data.get("lp_locked_pct", 0),
        "top10_holder_pct": gp_data.get("top10_holder_pct", 0),
        "holder_count": gp_data.get("holder_count", 0),
        "is_honeypot": gp_data.get("is_honeypot", False),
        "sell_tax": gp_data.get("sell_tax", 0),
        "buy_tax": gp_data.get("buy_tax", 0),
        "acp_trust_score": acp_data.get("trust_score", 0),
        "acp_total_jobs": acp_data.get("total_jobs", 0),
        "acp_completion_rate": acp_data.get("completion_rate", 0),
        "volume_to_mcap_ratio": round(
            dex_data.get("volume_24h", 0) / dex_data.get("market_cap", 1)
            if dex_data.get("market_cap", 0) > 0 else 0, 4
        ),
        "social_presence_count": len(dex_data.get("websites", [])) + len(dex_data.get("socials", [])),
        "model_used": model_tag,
    }

    return AgentPredictResponse(
        token_address=token_address,
        wallet_address=req.wallet_address,
        rug_probability=round(rug_prob_final, 4),
        rug_score=rug_score,
        risk_level=risk_level,
        behavior_class=behavior_class,
        confidence=confidence,
        model_version=f"{model_ver} ({model_tag})",
        dex_signals={
            k: v for k, v in dex_data.items()
            if not k.startswith("_") and k not in ("websites", "socials")
        },
        goplus_signals={k: v for k, v in gp_data.items() if not k.startswith("_")},
        acp_signals=acp_data,
        risk_signals=risk_signals,
        virtuals_features=virtuals_features,
        summary=summary,
        predicted_at=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/risks/top", tags=["Phase 3: Agent Rug Detection"])
async def get_top_risks(limit: int = 50, min_score: float = 0.5):
    """
    Get top riskiest agent tokens from the pre-scanned results.
    Requires: scan_agent_tokens.py to have been run first.
    """
    try:
        import psycopg2
        import psycopg2.extras
        db_url = os.environ.get(
            "DATABASE_URL",
            os.environ["DATABASE_URL"]
        )
        conn = psycopg2.connect(db_url)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT token_address, wallet_address, token_symbol,
                   risk_score, risk_label, risk_signals,
                   liquidity_usd, volume_24h, price_change_24h, market_cap,
                   holder_count, top10_holder_pct,
                   acp_trust_score, acp_total_jobs,
                   token_age_days, scanned_at
            FROM wadjet_agent_token_risks
            WHERE risk_score >= %s
            ORDER BY risk_score DESC
            LIMIT %s
        """, (min_score, min(limit, 200)))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {
            "total": len(rows),
            "min_score_filter": min_score,
            "tokens": [dict(r) for r in rows],
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB query failed: {e}")


@app.get("/risks/summary", tags=["Phase 3: Agent Rug Detection"])
async def get_risks_summary():
    """
    Summary statistics of all scanned agent token risks.
    """
    try:
        import psycopg2
        db_url = os.environ.get(
            "DATABASE_URL",
            os.environ["DATABASE_URL"]
        )
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(*) as total_scanned,
                COUNT(CASE WHEN risk_label = 1 THEN 1 END) as flagged_rugs,
                COUNT(CASE WHEN risk_score >= 0.7 THEN 1 END) as critical_risk,
                COUNT(CASE WHEN risk_score >= 0.5 AND risk_score < 0.7 THEN 1 END) as high_risk,
                COUNT(CASE WHEN risk_score >= 0.35 AND risk_score < 0.5 THEN 1 END) as medium_risk,
                COUNT(CASE WHEN risk_score < 0.35 THEN 1 END) as low_risk,
                ROUND(AVG(risk_score)::numeric, 4) as avg_risk_score,
                MAX(scanned_at) as last_scan
            FROM wadjet_agent_token_risks
        """)
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            return {"total_scanned": 0, "message": "No scan results yet. Run scan_agent_tokens.py first."}
        return {
            "total_scanned": row[0],
            "flagged_rugs": row[1],
            "critical_risk": row[2],
            "high_risk": row[3],
            "medium_risk": row[4],
            "low_risk": row[5],
            "avg_risk_score": float(row[6]) if row[6] else 0,
            "last_scan": str(row[7]) if row[7] else None,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB query failed: {e}")


# ─── Phase 2: Wadjet Risk Intelligence API ────────────────────────────────────

from typing import List as TypingList

# Lazy imports to avoid circular deps and slow startup
def _get_db():
    from db.supabase_client import (
        get_agent_profile, get_simulation_result, get_all_clusters,
        get_graph_edges, ensure_schema
    )
    return {
        "get_agent_profile": get_agent_profile,
        "get_simulation_result": get_simulation_result,
        "get_all_clusters": get_all_clusters,
        "get_graph_edges": get_graph_edges,
        "ensure_schema": ensure_schema,
    }


def _run_fresh_simulation(address: str, profile: dict) -> dict:
    """Run a fresh Monte Carlo simulation for a single agent."""
    from simulator.monte_carlo import simulate_agent
    from db.supabase_client import get_all_profiles, upsert_simulation_result

    all_profiles = get_all_profiles(limit=500)
    all_by_addr = {p["address"]: {
        "agent": p["address"],
        "behavior_type": p.get("behavior_type", "normal"),
        "risk_tolerance": p.get("risk_tolerance", "medium"),
        "dependencies": p.get("dependencies", []) if isinstance(p.get("dependencies"), list) else [],
    } for p in all_profiles}

    agent_profile = {
        "agent": address,
        "behavior_type": profile.get("behavior_type", "normal"),
        "risk_tolerance": profile.get("risk_tolerance", "medium"),
        "dependencies": profile.get("dependencies", []) if isinstance(profile.get("dependencies"), list) else [],
    }

    result = simulate_agent(
        profile=agent_profile,
        all_profiles_by_address=all_by_addr,
        n_runs=50,
    )

    try:
        upsert_simulation_result(result, cache_hours=24)
    except Exception as e:
        logger.warning(f"Failed to cache simulation result: {e}")

    return result


# ─── Response schemas (Phase 2) ────────────────────────────────────────────────

class ScenarioResult(BaseModel):
    scenario_id: str
    name: str
    survival_rate: float
    avg_loss: float
    var_95: float
    cascade_rate: float
    n_runs: int


class CascadeRisk(BaseModel):
    if_fails: str
    dependency_weight: float
    this_agent_survival: float
    expected_loss: float


class AgentRiskProfile(BaseModel):
    address: str
    behavior_type: str
    avg_daily_volume: Optional[float]
    counterparties: Optional[int]
    risk_tolerance: str
    resilience_score: Optional[float]
    cluster_id: Optional[int]
    scenarios: TypingList[ScenarioResult]
    cascade_risk: TypingList[CascadeRisk]
    metrics: dict
    cached: bool
    generated_at: str


class PortfolioRisk(BaseModel):
    total_agents: int
    avg_resilience: float
    fragile_count: int
    robust_count: int
    agents: TypingList[dict]
    portfolio_risk_score: float


# ─── Helper ────────────────────────────────────────────────────────────────────

def _merge_profile_and_simulation(
    address: str,
    profile_row: Optional[dict],
    sim_row: Optional[dict],
    cached: bool,
) -> dict:
    import json as _json

    def _loads(v):
        if isinstance(v, str):
            try:
                return _json.loads(v)
            except Exception:
                return []
        return v or []

    scenarios = _loads(sim_row.get("scenarios")) if sim_row else []
    cascade   = _loads(sim_row.get("cascade_risk")) if sim_row else []
    metrics   = profile_row.get("metrics") if profile_row else {}
    if isinstance(metrics, str):
        try:
            metrics = _json.loads(metrics)
        except Exception:
            metrics = {}

    return {
        "address": address,
        "behavior_type": (profile_row or {}).get("behavior_type", "unknown"),
        "avg_daily_volume": (profile_row or {}).get("avg_daily_volume"),
        "counterparties": (profile_row or {}).get("counterparties"),
        "risk_tolerance": (profile_row or {}).get("risk_tolerance", "medium"),
        "resilience_score": (sim_row or {}).get("resilience_score"),
        "cluster_id": (sim_row or {}).get("cluster_id"),
        "scenarios": scenarios,
        "cascade_risk": cascade,
        "metrics": metrics or {},
        "cached": cached,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


# ─── Wadjet Phase 2 Routes ────────────────────────────────────────────────────
# NOTE: Static routes (/clusters, /portfolio) MUST be declared BEFORE
#       the dynamic /wadjet/{address} route to avoid FastAPI shadowing them.

@app.get("/wadjet/clusters", tags=["Phase 2: Risk Intel"])
async def get_all_clusters_route():
    """All detected hidden clusters with risk scores."""
    db = _get_db()
    try:
        clusters = db["get_all_clusters"]()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch clusters: {e}")
    return {
        "total_clusters": len(clusters),
        "clusters": clusters,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/wadjet/portfolio", tags=["Phase 2: Risk Intel"])
async def get_portfolio_route(agents: str):
    """
    Portfolio risk assessment for a comma-separated list of agent addresses.
    Usage: /wadjet/portfolio?agents=0x123,0x456,0x789
    """
    return await _portfolio_risk_logic(agents)


@app.get("/wadjet/{address}", response_model=AgentRiskProfile, tags=["Phase 2: Risk Intel"])
async def get_agent_risk(address: str):
    """Full risk profile + simulation results for a single agent."""
    addr = address.lower()
    db = _get_db()

    profile_row = None
    sim_row = None
    cached = False

    try:
        profile_row = db["get_agent_profile"](addr)
    except Exception as e:
        logger.warning(f"Profile fetch failed for {addr}: {e}")

    try:
        sim_row = db["get_simulation_result"](addr)
        if sim_row:
            cached = True
    except Exception as e:
        logger.warning(f"Sim result fetch failed for {addr}: {e}")

    # If no simulation cached, run one now (profile must exist)
    if not sim_row and profile_row:
        try:
            import json as _j
            profile_for_sim = {
                "agent": addr,
                "behavior_type": profile_row.get("behavior_type", "normal"),
                "risk_tolerance": profile_row.get("risk_tolerance", "medium"),
                "dependencies": _j.loads(profile_row["dependencies"]) if isinstance(profile_row.get("dependencies"), str) else (profile_row.get("dependencies") or []),
            }
            sim_result = _run_fresh_simulation(addr, profile_for_sim)
            sim_row = sim_result
        except Exception as e:
            logger.error(f"Fresh simulation failed for {addr}: {e}")

    if not profile_row and not sim_row:
        raise HTTPException(
            status_code=404,
            detail=f"No profile found for {addr}. Run the profiler first."
        )

    return _merge_profile_and_simulation(addr, profile_row, sim_row, cached)


@app.get("/wadjet/{address}/scenarios", tags=["Phase 2: Risk Intel"])
async def get_agent_scenarios(address: str):
    """Detailed scenario breakdowns for a specific agent."""
    addr = address.lower()
    db = _get_db()

    try:
        sim_row = db["get_simulation_result"](addr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not sim_row:
        raise HTTPException(
            status_code=404,
            detail=f"No simulation results for {addr}. Call /wadjet/{addr} first to trigger simulation."
        )

    import json as _j
    scenarios = sim_row.get("scenarios", [])
    if isinstance(scenarios, str):
        scenarios = _j.loads(scenarios)

    return {
        "address": addr,
        "resilience_score": sim_row.get("resilience_score"),
        "scenarios": scenarios,
        "simulation_runs": sim_row.get("simulation_runs"),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


async def _portfolio_risk_logic(agents: str):
    addresses = [a.strip().lower() for a in agents.split(",") if a.strip()]
    if not addresses:
        raise HTTPException(status_code=400, detail="No agent addresses provided")
    if len(addresses) > 50:
        raise HTTPException(status_code=400, detail="Max 50 agents per portfolio request")

    db = _get_db()
    agent_results = []
    import json as _j

    for addr in addresses:
        try:
            profile_row = db["get_agent_profile"](addr)
            sim_row = db["get_simulation_result"](addr)
            merged = _merge_profile_and_simulation(addr, profile_row, sim_row, bool(sim_row))
            agent_results.append(merged)
        except Exception as e:
            logger.warning(f"Portfolio: failed to get data for {addr}: {e}")
            agent_results.append({
                "address": addr,
                "error": str(e),
                "resilience_score": None,
            })

    resilience_scores = [
        r["resilience_score"] for r in agent_results
        if r.get("resilience_score") is not None
    ]

    avg_resilience = sum(resilience_scores) / len(resilience_scores) if resilience_scores else 0.0
    fragile_count = sum(1 for s in resilience_scores if s < 0.3)
    robust_count  = sum(1 for s in resilience_scores if s > 0.7)

    # Portfolio risk = inverse of avg resilience, amplified by fragile concentration
    portfolio_risk = 1.0 - avg_resilience
    if resilience_scores:
        fragile_ratio = fragile_count / len(resilience_scores)
        portfolio_risk = min(1.0, portfolio_risk * (1 + 0.5 * fragile_ratio))

    return {
        "total_agents": len(addresses),
        "avg_resilience": round(avg_resilience, 4),
        "fragile_count": fragile_count,
        "robust_count": robust_count,
        "portfolio_risk_score": round(portfolio_risk, 4),
        "agents": agent_results,
    }


@app.get("/wadjet/cascade/{address}", tags=["Phase 2: Risk Intel"])
async def get_cascade_map(address: str):
    """
    Cascade risk map for a specific agent.
    Shows: which dependencies put this agent at risk, and what addresses
    this agent's failure would cascade to.
    """
    addr = address.lower()
    db = _get_db()

    profile_row = None
    sim_row = None
    import json as _j

    try:
        profile_row = db["get_agent_profile"](addr)
        sim_row = db["get_simulation_result"](addr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not profile_row:
        raise HTTPException(status_code=404, detail=f"No profile for {addr}")

    deps = profile_row.get("dependencies", [])
    if isinstance(deps, str):
        try:
            deps = _j.loads(deps)
        except Exception:
            deps = []

    cascade_risk = []
    if sim_row:
        cascade_risk = sim_row.get("cascade_risk", [])
        if isinstance(cascade_risk, str):
            try:
                cascade_risk = _j.loads(cascade_risk)
            except Exception:
                cascade_risk = []

    # Graph edges for downstream cascade
    try:
        edges = db["get_graph_edges"](addr)
    except Exception:
        edges = []

    downstream = [
        {"address": e["to_address"], "weight": e.get("weight", 0)}
        for e in edges
        if e.get("from_address", "").lower() == addr
    ][:10]

    return {
        "address": addr,
        "behavior_type": profile_row.get("behavior_type", "unknown"),
        "resilience_score": (sim_row or {}).get("resilience_score"),
        "upstream_risks": cascade_risk,     # If these deps fail → this agent suffers
        "downstream_impact": downstream,    # If this agent fails → these addresses affected
        "dependencies": deps,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


# ─── Watchlist Endpoints ──────────────────────────────────────────────────────

@app.get("/watchlist", tags=["Watchlist"])
async def get_watchlist_endpoint(status: str = "active", limit: int = 200):
    """
    List all watchlist items sorted by severity (critical → high → medium → low).
    Use ?status=active (default) or ?status=resolved to filter.
    """
    try:
        from db.supabase_client import get_watchlist
        items = get_watchlist(status=status, limit=min(limit, 500))
        return {
            "total": len(items),
            "status_filter": status,
            "items": items,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB query failed: {e}")


@app.get("/watchlist/{token_address}", tags=["Watchlist"])
async def get_watchlist_item_endpoint(token_address: str):
    """
    Fetch a specific token's watchlist entry + its last 7 days of delta history.
    Returns 404 if the token is not on the watchlist.
    """
    addr = token_address.lower()
    try:
        from db.supabase_client import get_watchlist_item, get_last_two_snapshots
        import psycopg2
        import psycopg2.extras

        item = get_watchlist_item(addr)
        if not item:
            raise HTTPException(
                status_code=404,
                detail=f"Token {addr} is not on the watchlist."
            )

        # Fetch last 7 days of snapshots for delta history
        db_url = os.environ.get(
            "DATABASE_URL",
            os.environ["DATABASE_URL"]
        )
        conn = psycopg2.connect(db_url)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT snapshot_date,
                   top10_holder_pct, holder_count, lp_locked_pct,
                   creator_percent, owner_percent,
                   price_usd, liquidity_usd, volume_24h, market_cap,
                   trust_score, total_jobs, completion_rate
            FROM wadjet_daily_snapshots
            WHERE token_address = %s
            ORDER BY snapshot_date DESC
            LIMIT 7
        """, (addr,))
        snapshots = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()

        # Compute rolling deltas between consecutive days
        delta_history = []
        for i in range(len(snapshots) - 1):
            today = snapshots[i]
            yest  = snapshots[i + 1]

            def _rel(t, y):
                if y and y != 0:
                    return round((t - y) / abs(y), 4)
                return None

            delta_history.append({
                "date": str(today.get("snapshot_date")),
                "holder_concentration_delta": round(
                    (today.get("top10_holder_pct") or 0) - (yest.get("top10_holder_pct") or 0), 4
                ),
                "liquidity_delta":   _rel(today.get("liquidity_usd") or 0, yest.get("liquidity_usd") or 0),
                "volume_delta":      _rel(today.get("volume_24h") or 0, yest.get("volume_24h") or 0),
                "price_delta":       _rel(today.get("price_usd") or 0, yest.get("price_usd") or 0),
                "creator_pct_delta": round(
                    (today.get("creator_percent") or 0) - (yest.get("creator_percent") or 0), 6
                ),
                "holder_count_delta": (today.get("holder_count") or 0) - (yest.get("holder_count") or 0),
            })

        return {
            "token_address": addr,
            "watchlist_entry": item,
            "snapshot_history": snapshots,
            "delta_history": delta_history,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


# ─── Sentinel Endpoints ──────────────────────────────────────────────────────

# In-memory sentinel scan state (mirrors _cron_state pattern)
_sentinel_state: dict = {
    "stage1_running": False,
    "stage2_running": False,
    "last_stage1": None,
    "last_stage2": None,
}


def _run_sentinel_stage1_background():
    """Blocking wrapper for Stage 1 — runs in a thread."""
    import sys as _sys
    _sys.path.insert(0, str(BASE_DIR))
    try:
        import asyncio as _asyncio
        from scripts.sentinel import run_stage1_scan
        result = _asyncio.run(run_stage1_scan())
        _sentinel_state["stage1_running"] = False
        _sentinel_state["last_stage1"] = result
    except Exception as e:
        logger.error(f"Sentinel Stage 1 background failed: {e}", exc_info=True)
        _sentinel_state["stage1_running"] = False
        _sentinel_state["last_stage1"] = {
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }


def _run_sentinel_stage2_background():
    """Blocking wrapper for Stage 2 — runs in a thread."""
    import sys as _sys
    _sys.path.insert(0, str(BASE_DIR))
    try:
        import asyncio as _asyncio
        from scripts.sentinel import run_stage2_check
        result = _asyncio.run(run_stage2_check())
        _sentinel_state["stage2_running"] = False
        _sentinel_state["last_stage2"] = result
    except Exception as e:
        logger.error(f"Sentinel Stage 2 background failed: {e}", exc_info=True)
        _sentinel_state["stage2_running"] = False
        _sentinel_state["last_stage2"] = {
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }


@app.post("/sentinel/scan", tags=["Sentinel"])
async def sentinel_scan(
    background_tasks: BackgroundTasks,
    x_cron_api_key: Optional[str] = Header(None, alias="X-Cron-Api-Key"),
):
    """
    Trigger Stage 1 sentinel scan (all indexed tokens via GoPlus).

    Protected by X-Cron-Api-Key header.
    Recommended: schedule hourly via Railway cron.

    Returns immediately; scan runs in background.
    """
    _assert_cron_key(x_cron_api_key)

    if _sentinel_state["stage1_running"]:
        return {
            "status": "already_running",
            "message": "Stage 1 scan is already in progress.",
            "triggered_at": datetime.utcnow().isoformat() + "Z",
        }

    _sentinel_state["stage1_running"] = True
    background_tasks.add_task(_run_sentinel_stage1_background)

    return {
        "status": "triggered",
        "message": "Sentinel Stage 1 scan started (GoPlus hourly scan).",
        "triggered_at": datetime.utcnow().isoformat() + "Z",
        "poll_url": "/sentinel/status",
    }


@app.post("/sentinel/check-watchlist", tags=["Sentinel"])
async def sentinel_check_watchlist(
    background_tasks: BackgroundTasks,
    x_cron_api_key: Optional[str] = Header(None, alias="X-Cron-Api-Key"),
):
    """
    Trigger Stage 2 watchlist monitoring (Alchemy transfer tracking).

    Protected by X-Cron-Api-Key header.
    Recommended: schedule every 10 minutes via Railway cron.

    Returns immediately; check runs in background.
    """
    _assert_cron_key(x_cron_api_key)

    if _sentinel_state["stage2_running"]:
        return {
            "status": "already_running",
            "message": "Stage 2 watchlist check is already in progress.",
            "triggered_at": datetime.utcnow().isoformat() + "Z",
        }

    _sentinel_state["stage2_running"] = True
    background_tasks.add_task(_run_sentinel_stage2_background)

    return {
        "status": "triggered",
        "message": "Sentinel Stage 2 check started (watchlist monitoring).",
        "triggered_at": datetime.utcnow().isoformat() + "Z",
        "poll_url": "/sentinel/status",
    }


@app.get("/sentinel/alerts", tags=["Sentinel"])
async def get_sentinel_alerts(
    severity: Optional[str] = None,
    alert_type: Optional[str] = None,
    limit: int = 100,
):
    """
    List recent Sentinel alerts.

    Optional filters:
      - severity: 'info' | 'warning' | 'critical'
      - alert_type: 'watchlist_added' | 'sell_signal' | 'dump_pattern' | 'confirmed_rug'
      - limit: max results (default 100, max 500)
    """
    try:
        from db.supabase_client import get_alerts
        alerts = get_alerts(
            severity=severity,
            alert_type=alert_type,
            limit=min(limit, 500),
        )
        return {
            "total": len(alerts),
            "severity_filter": severity,
            "type_filter": alert_type,
            "alerts": alerts,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB query failed: {e}")


@app.get("/sentinel/alerts/{token_address}", tags=["Sentinel"])
async def get_token_alerts(token_address: str, limit: int = 50):
    """
    List all Sentinel alerts for a specific token address.
    """
    try:
        from db.supabase_client import get_alerts, get_watchlist_item
        alerts = get_alerts(
            token_address=token_address,
            limit=min(limit, 200),
        )
        watchlist_entry = get_watchlist_item(token_address)
        return {
            "token_address": token_address.lower(),
            "watchlist_entry": watchlist_entry,
            "total_alerts": len(alerts),
            "alerts": alerts,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB query failed: {e}")


@app.get("/sentinel/status", tags=["Sentinel"])
async def sentinel_status():
    """
    Sentinel status overview:
    - Last scan/check times
    - Watchlist active count
    - Alert counts by type/severity
    """
    try:
        from db.supabase_client import (
            get_alert_counts, get_last_stage1_scan, get_last_stage2_check,
            get_watchlist,
        )
        alert_counts = get_alert_counts()
        watchlist_active = get_watchlist(status="active", limit=500)
        watchlist_confirmed = get_watchlist(status="confirmed_rug", limit=500)

        return {
            "stage1_running": _sentinel_state["stage1_running"],
            "stage2_running": _sentinel_state["stage2_running"],
            "last_stage1_scan": get_last_stage1_scan(),
            "last_stage2_check": get_last_stage2_check(),
            "watchlist": {
                "active_count": len(watchlist_active),
                "confirmed_rug_count": len(watchlist_confirmed),
            },
            "alerts": alert_counts,
            "checked_at": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status query failed: {e}")


# ─── Cron Endpoints ───────────────────────────────────────────────────────────

# Simple in-memory state for the background job
_cron_state: dict = {"running": False, "last_result": None}

# API key for cron protection — set CRON_API_KEY env var on Railway
_CRON_API_KEY = os.environ.get("CRON_API_KEY", "wadjet-cron-secret")


def _assert_cron_key(x_cron_api_key: Optional[str]):
    if not x_cron_api_key or x_cron_api_key != _CRON_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-Cron-Api-Key header"
        )


def _run_cron_in_background():
    """Blocking wrapper — runs in a thread via BackgroundTasks."""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    try:
        from scripts.daily_cron import run_daily_cron
        result = run_daily_cron()
        _cron_state["running"] = False
        _cron_state["last_result"] = result
    except Exception as e:
        logger.error(f"Background cron failed: {e}", exc_info=True)
        _cron_state["running"] = False
        _cron_state["last_result"] = {
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }


@app.post("/cron/run-daily", tags=["Cron"])
async def run_daily_cron_endpoint(
    background_tasks: BackgroundTasks,
    x_cron_api_key: Optional[str] = Header(None, alias="X-Cron-Api-Key"),
):
    """
    Trigger the daily agent re-profiling job.

    Protected by X-Cron-Api-Key header (set CRON_API_KEY env var on Railway).

    Returns immediately with job status; profiling runs in the background.
    Use GET /cron/status to poll for completion.

    Recommended schedule: daily at 02:00 UTC.
    """
    _assert_cron_key(x_cron_api_key)

    if _cron_state["running"]:
        return {
            "status": "already_running",
            "message": "A cron job is already in progress. Check /cron/status.",
            "triggered_at": datetime.utcnow().isoformat() + "Z",
        }

    _cron_state["running"] = True
    background_tasks.add_task(_run_cron_in_background)

    return {
        "status": "triggered",
        "message": "Daily profiling job started in background.",
        "triggered_at": datetime.utcnow().isoformat() + "Z",
        "poll_url": "/cron/status",
    }


@app.post("/cron/auto-outcomes", tags=["Cron"])
async def trigger_auto_outcomes(
    background_tasks: BackgroundTasks,
    x_cron_api_key: Optional[str] = Header(None, alias="X-Cron-Api-Key"),
):
    """
    Trigger the auto-outcome reporter — closes the feedback loop for unreported queries.

    Scans query_logs WHERE outcome IS NULL AND older than 7 days, determines
    outcome (scam/failure/success/expired) via DexScreener + agent_scores, and
    persists results.

    Protected by X-Cron-Api-Key header.
    Returns immediately; reporter runs in background.
    """
    _assert_cron_key(x_cron_api_key)

    if _cron_state["running"]:
        return {
            "status": "already_running",
            "message": "Another cron job is already in progress. Check /cron/status.",
            "triggered_at": datetime.utcnow().isoformat() + "Z",
        }

    def _run_auto_outcomes_background():
        import sys as _sys
        _sys.path.insert(0, str(BASE_DIR))
        try:
            from scripts.auto_outcomes import run_auto_outcomes
            result = run_auto_outcomes()
            _cron_state["running"] = False
            _cron_state["last_result"] = {
                "job": "auto_outcomes",
                **result,
                "completed_at": datetime.utcnow().isoformat() + "Z",
            }
        except Exception as e:
            logger.error(f"Auto-outcome background job failed: {e}", exc_info=True)
            _cron_state["running"] = False
            _cron_state["last_result"] = {
                "job": "auto_outcomes",
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow().isoformat() + "Z",
            }

    _cron_state["running"] = True
    background_tasks.add_task(_run_auto_outcomes_background)

    return {
        "status": "triggered",
        "message": "Auto-outcome reporter started in background.",
        "triggered_at": datetime.utcnow().isoformat() + "Z",
        "poll_url": "/cron/status",
    }


@app.get("/cron/status", tags=["Cron"])
async def cron_status(
    x_cron_api_key: Optional[str] = Header(None, alias="X-Cron-Api-Key"),
):
    """
    Poll the current cron job status.
    Protected by X-Cron-Api-Key header.
    """
    _assert_cron_key(x_cron_api_key)
    return {
        "running": _cron_state["running"],
        "last_result": _cron_state["last_result"],
        "checked_at": datetime.utcnow().isoformat() + "Z",
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

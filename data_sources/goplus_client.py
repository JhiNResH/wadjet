"""
GoPlus Security API Client
--------------------------
Free-tier, no API key required for basic endpoints.

Endpoints used:
  Token security: GET /api/v1/token_security/{chain_id}?contract_addresses={addr}
  Address security: GET /api/v1/address_security/{addr}?chain_id={chain_id}

Base chain_id = 8453

Features:
  - Async HTTP via httpx
  - Rate limiting: max 5 req/s (token bucket)
  - In-memory cache with 24h TTL (optionally file-backed)
  - Graceful degradation on API failure (returns empty dict)
"""

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("wadjet.goplus")

# ─── Config ──────────────────────────────────────────────────────────────────
GOPLUS_BASE_URL = "https://api.gopluslabs.io/api/v1"
DEFAULT_CHAIN_ID = 8453          # Base mainnet
CACHE_TTL_SECONDS = 86_400       # 24 hours
MAX_RPS = 5                      # GoPlus free tier limit
REQUEST_TIMEOUT = 10.0           # seconds

# Optional: persist cache to disk so restarts keep it warm
CACHE_DIR = Path(__file__).parent.parent / "data" / "goplus_cache"


# ─── Rate limiter (token bucket) ─────────────────────────────────────────────
class _TokenBucket:
    def __init__(self, rate: float):
        self.rate = rate          # tokens per second
        self._tokens = rate
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self.rate, self._tokens + elapsed * self.rate)
            self._last = now
            if self._tokens < 1:
                wait = (1 - self._tokens) / self.rate
                await asyncio.sleep(wait)
                self._tokens = 0
            else:
                self._tokens -= 1


# ─── Cache ────────────────────────────────────────────────────────────────────
class _Cache:
    """Thread-safe in-memory cache with optional disk persistence."""

    def __init__(self, ttl: int = CACHE_TTL_SECONDS, disk: bool = True):
        self._ttl = ttl
        self._mem: dict[str, tuple[float, dict]] = {}  # key → (expires_at, data)
        self._disk = disk
        if disk:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        safe = hashlib.md5(key.encode()).hexdigest()
        return CACHE_DIR / f"{safe}.json"

    def get(self, key: str) -> Optional[dict]:
        # Memory first
        if key in self._mem:
            expires, data = self._mem[key]
            if time.time() < expires:
                return data
            del self._mem[key]

        # Disk fallback
        if self._disk:
            path = self._key_path(key)
            if path.exists():
                try:
                    payload = json.loads(path.read_text())
                    if time.time() < payload["expires_at"]:
                        self._mem[key] = (payload["expires_at"], payload["data"])
                        return payload["data"]
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
        return None

    def set(self, key: str, data: dict):
        expires = time.time() + self._ttl
        self._mem[key] = (expires, data)
        if self._disk:
            path = self._key_path(key)
            try:
                path.write_text(json.dumps({"expires_at": expires, "data": data}))
            except Exception as e:
                logger.debug(f"Cache disk write failed: {e}")


# ─── GoPlus Client ────────────────────────────────────────────────────────────
class GoPlusClient:
    """
    Async client for the GoPlus Security API.

    Usage:
        client = GoPlusClient()
        token_info = await client.check_token_security("0xabc...", chain_id=8453)
        addr_info  = await client.check_address_security("0xabc...")
    """

    def __init__(
        self,
        base_url: str = GOPLUS_BASE_URL,
        max_rps: float = MAX_RPS,
        cache_ttl: int = CACHE_TTL_SECONDS,
        disk_cache: bool = True,
    ):
        self._base_url = base_url.rstrip("/")
        self._bucket = _TokenBucket(max_rps)
        self._cache = _Cache(ttl=cache_ttl, disk=disk_cache)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "wadjet/1.0 (maiat-protocol)"},
                follow_redirects=True,
            )
        return self._client

    async def _get(self, url: str, params: dict) -> Optional[dict]:
        """Internal GET with rate limiting and error handling."""
        cache_key = url + json.dumps(params, sort_keys=True)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"GoPlus cache hit: {url}")
            return cached

        await self._bucket.acquire()
        client = await self._get_client()

        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != 1:
                logger.warning(f"GoPlus non-success code={data.get('code')} msg={data.get('message')} url={url}")
                return {}

            result = data.get("result") or {}
            self._cache.set(cache_key, result)
            return result

        except httpx.TimeoutException:
            logger.error(f"GoPlus request timed out: {url}")
        except httpx.HTTPStatusError as e:
            logger.error(f"GoPlus HTTP error {e.response.status_code}: {url}")
        except Exception as e:
            logger.error(f"GoPlus unexpected error: {e}")

        return {}

    # ─── Public methods ───────────────────────────────────────────────────────

    async def check_token_security(
        self,
        address: str,
        chain_id: int = DEFAULT_CHAIN_ID,
    ) -> dict:
        """
        Fetch token security data for a contract address.

        Returns a normalized dict with:
          - is_honeypot (bool)
          - is_blacklisted (bool)
          - is_proxy (bool)
          - owner_can_change_balance (bool)
          - is_mintable (bool)
          - buy_tax (float)
          - sell_tax (float)
          - holder_count (int)
          - lp_holder_count (int)
          - creator_address (str)
          - is_open_source (bool)
          - raw (dict) — full GoPlus response for the token
        """
        url = f"{self._base_url}/token_security/{chain_id}"
        address = address.lower()
        raw_result = await self._get(url, {"contract_addresses": address})

        if not raw_result:
            return self._empty_token_response(address)

        # GoPlus returns { "<address>": { ... } }
        token_data = raw_result.get(address) or {}
        if not token_data:
            # Try without '0x' prefix quirk
            for k, v in raw_result.items():
                if k.lower() == address:
                    token_data = v
                    break

        if not token_data:
            logger.debug(f"GoPlus: no token data for {address} on chain {chain_id}")
            return self._empty_token_response(address)

        def _bool(val) -> bool:
            return str(val).strip() in ("1", "true", "True")

        def _float(val, default=0.0) -> float:
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        return {
            "address": address,
            "chain_id": chain_id,
            "is_honeypot": _bool(token_data.get("is_honeypot")),
            "is_blacklisted": _bool(token_data.get("is_blacklisted")),
            "is_proxy": _bool(token_data.get("is_proxy")),
            "owner_can_change_balance": _bool(token_data.get("owner_change_balance")),
            "is_mintable": _bool(token_data.get("is_mintable")),
            "is_open_source": _bool(token_data.get("is_open_source")),
            "buy_tax": _float(token_data.get("buy_tax")),
            "sell_tax": _float(token_data.get("sell_tax")),
            "holder_count": int(token_data.get("holder_count") or 0),
            "lp_holder_count": int(token_data.get("lp_holder_count") or 0),
            "creator_address": token_data.get("creator_address", ""),
            "raw": token_data,
            "available": True,
        }

    async def check_address_security(
        self,
        address: str,
        chain_id: int = DEFAULT_CHAIN_ID,
    ) -> dict:
        """
        Fetch address security data (wallet-level flags).

        Returns a normalized dict with:
          - malicious_address (bool)
          - phishing_address (bool)
          - blacklist_doubt (bool)
          - honeypot_related (bool)
          - stealing_attack (bool)
          - fake_kyc (bool)
          - labels (list[str])
          - raw (dict)
        """
        url = f"{self._base_url}/address_security/{address.lower()}"
        raw_result = await self._get(url, {"chain_id": str(chain_id)})

        if not raw_result:
            return self._empty_address_response(address)

        def _bool(val) -> bool:
            return str(val).strip() in ("1", "true", "True")

        # Collect descriptive labels
        labels = []
        label_fields = [
            "malicious_address", "phishing_activities", "blacklist_doubt",
            "honeypot_related_address", "stealing_attack", "fake_kyc",
            "cybercrime", "money_laundering", "financial_crime",
            "darkweb_transactions", "reinit",
        ]
        for field in label_fields:
            if _bool(raw_result.get(field)):
                labels.append(field)

        return {
            "address": address.lower(),
            "chain_id": chain_id,
            "malicious_address": _bool(raw_result.get("malicious_address")),
            "phishing_address": _bool(raw_result.get("phishing_activities")),
            "blacklist_doubt": _bool(raw_result.get("blacklist_doubt")),
            "honeypot_related": _bool(raw_result.get("honeypot_related_address")),
            "stealing_attack": _bool(raw_result.get("stealing_attack")),
            "fake_kyc": _bool(raw_result.get("fake_kyc")),
            "cybercrime": _bool(raw_result.get("cybercrime")),
            "money_laundering": _bool(raw_result.get("money_laundering")),
            "labels": labels,
            "raw": raw_result,
            "available": True,
        }

    async def get_combined_risk(
        self,
        wallet_address: str,
        token_address: Optional[str] = None,
        chain_id: int = DEFAULT_CHAIN_ID,
    ) -> dict:
        """
        Fetch both address security and (optionally) token security.
        Returns a merged dict with all GoPlus flags.

        Used by the profiler to get a single batch of signals.
        """
        tasks = [self.check_address_security(wallet_address, chain_id=chain_id)]
        if token_address:
            tasks.append(self.check_token_security(token_address, chain_id=chain_id))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        addr_result = results[0] if not isinstance(results[0], Exception) else self._empty_address_response(wallet_address)
        token_result = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else {}

        return {
            "wallet": addr_result,
            "token": token_result,
        }

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_token_response(address: str) -> dict:
        return {
            "address": address,
            "chain_id": DEFAULT_CHAIN_ID,
            "is_honeypot": False,
            "is_blacklisted": False,
            "is_proxy": False,
            "owner_can_change_balance": False,
            "is_mintable": False,
            "is_open_source": False,
            "buy_tax": 0.0,
            "sell_tax": 0.0,
            "holder_count": 0,
            "lp_holder_count": 0,
            "creator_address": "",
            "raw": {},
            "available": False,
        }

    @staticmethod
    def _empty_address_response(address: str) -> dict:
        return {
            "address": address.lower(),
            "chain_id": DEFAULT_CHAIN_ID,
            "malicious_address": False,
            "phishing_address": False,
            "blacklist_doubt": False,
            "honeypot_related": False,
            "stealing_attack": False,
            "fake_kyc": False,
            "cybercrime": False,
            "money_laundering": False,
            "labels": [],
            "raw": {},
            "available": False,
        }

    def compute_goplus_score_delta(self, wallet: dict, token: dict) -> dict:
        """
        Convert GoPlus flags into rugScore adjustments and feature booleans.

        Returns:
          {
            "score_delta": int,           # additive adjustment to rugScore
            "goplus_malicious": bool,
            "goplus_phishing": bool,
            "goplus_honeypot": bool,
            "goplus_proxy_contract": bool,
            "goplus_owner_can_change_balance": bool,
            "goplus_flags": list[str],    # human-readable active flags
          }
        """
        score_delta = 0
        flags = []

        # Wallet-level flags
        is_malicious = wallet.get("malicious_address", False)
        is_phishing = wallet.get("phishing_address", False)
        is_honeypot_related = wallet.get("honeypot_related", False)

        if is_malicious:
            score_delta += 30
            flags.append("malicious_address")

        if is_phishing:
            score_delta += 25
            flags.append("phishing_address")

        if wallet.get("cybercrime"):
            score_delta += 20
            flags.append("cybercrime")

        if wallet.get("stealing_attack"):
            score_delta += 15
            flags.append("stealing_attack")

        if wallet.get("money_laundering"):
            score_delta += 15
            flags.append("money_laundering")

        # Token-level flags
        is_honeypot = token.get("is_honeypot", False)
        is_proxy = token.get("is_proxy", False)
        owner_change_balance = token.get("owner_can_change_balance", False)

        if is_honeypot or is_honeypot_related:
            score_delta += 40
            flags.append("honeypot")

        if owner_change_balance:
            score_delta += 20
            flags.append("owner_can_change_balance")

        if token.get("is_blacklisted"):
            score_delta += 25
            flags.append("token_blacklisted")

        # High tax = likely scam
        buy_tax = token.get("buy_tax", 0.0)
        sell_tax = token.get("sell_tax", 0.0)
        if sell_tax > 0.5 or buy_tax > 0.5:
            score_delta += 30
            flags.append(f"extreme_tax_sell={sell_tax:.0%}_buy={buy_tax:.0%}")
        elif sell_tax > 0.1 or buy_tax > 0.1:
            score_delta += 10
            flags.append(f"high_tax_sell={sell_tax:.0%}_buy={buy_tax:.0%}")

        return {
            "score_delta": min(100, score_delta),  # Cap at 100
            "goplus_malicious": is_malicious,
            "goplus_phishing": is_phishing,
            "goplus_honeypot": is_honeypot or is_honeypot_related,
            "goplus_proxy_contract": is_proxy,
            "goplus_owner_can_change_balance": owner_change_balance,
            "goplus_flags": flags,
            "goplus_available": wallet.get("available", False) or token.get("available", False),
        }


# ─── Singleton ───────────────────────────────────────────────────────────────
_goplus_client: Optional[GoPlusClient] = None


def get_goplus_client() -> GoPlusClient:
    """Return the shared GoPlusClient singleton."""
    global _goplus_client
    if _goplus_client is None:
        _goplus_client = GoPlusClient()
    return _goplus_client

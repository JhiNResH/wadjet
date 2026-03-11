#!/usr/bin/env python3
"""
scripts/chain_listener.py — Base On-Chain Event Listener

Ported from maiat-indexer/src/indexer.ts (WebSocket event watching)

Listens for:
  - EAS Attested events at 0x4200000000000000000000000000000000000021
  - ERC-8004 Registered events at 0x8004A169FB4a3325136EB29fA0ceB6D2e539a432

Uses websockets for WSS or web3.py HTTP fallback.
Auto-reconnects with exponential backoff (5s → 60s max).

Usage (standalone):
    python scripts/chain_listener.py

Environment:
    BASE_WSS_URL   — Base mainnet WebSocket URL (optional; HTTP fallback used if absent)
    BASE_RPC_URL   — Base mainnet HTTP RPC (default: https://mainnet.base.org)
    DATABASE_URL   — Supabase PostgreSQL connection string
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

# Bootstrap path so we can import wadjet modules
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PKG_ROOT)

logger = logging.getLogger("wadjet.chain_listener")

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_WSS_URL = os.environ.get("BASE_WSS_URL", "")
BASE_RPC_URL = os.environ.get("BASE_RPC_URL", "https://mainnet.base.org")

EAS_ADDRESS = "0x4200000000000000000000000000000000000021"
IDENTITY_REGISTRY = "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432"

# EAS: Attested(address indexed recipient, address indexed attester, bytes32 uid, bytes32 indexed schemaId)
EAS_ATTESTED_TOPIC = "0x8bf46bf4cfd674fa735a3d63ec1c9ad4153f033c290341f3a588b75685141b35"
# ERC-8004: Registered(uint256 indexed agentId, string agentURI, address indexed owner)
ERC8004_REGISTERED_TOPIC = "0xd6a85e1b8f76c2e25c3c48e72e3a24ddcc7d5f6b10aff62ce7a51d83e7a71aa9"

RECONNECT_MIN = 5    # seconds
RECONNECT_MAX = 60   # seconds

# ─── Event counters (shared state) ────────────────────────────────────────────

event_counts = {"eas": 0, "erc8004": 0}
start_time = datetime.now(timezone.utc)
_shutdown = False


# ─── DB helpers ───────────────────────────────────────────────────────────────

def log_alert(alert_type: str, token_address: str, wallet_address: str, detail: dict) -> None:
    """Log an on-chain event to wadjet_alerts table."""
    try:
        import psycopg2
        db_url = os.environ["DATABASE_URL"]
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS wadjet_alerts (
                id            SERIAL PRIMARY KEY,
                alert_type    TEXT NOT NULL,
                token_address TEXT,
                wallet_address TEXT,
                detail        JSONB DEFAULT '{}'::jsonb,
                severity      TEXT DEFAULT 'info',
                created_at    TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("""
            INSERT INTO wadjet_alerts (alert_type, token_address, wallet_address, detail, severity)
            VALUES (%s, %s, %s, %s::jsonb, %s)
        """, (alert_type, token_address.lower() if token_address else None,
              wallet_address.lower() if wallet_address else None,
              json.dumps(detail), "info"))

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.debug(f"Alert log failed (non-fatal): {e}")


def touch_agent_updated(wallet_address: str) -> None:
    """Update last_updated timestamp for a registered agent."""
    try:
        import psycopg2
        db_url = os.environ["DATABASE_URL"]
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("""
            UPDATE agent_scores
            SET updated_at = NOW()
            WHERE LOWER(wallet_address) = LOWER(%s)
        """, (wallet_address,))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.debug(f"touch_agent_updated failed (non-fatal): {e}")


# ─── Event handlers ───────────────────────────────────────────────────────────

def handle_eas_event(log: dict) -> None:
    """Process an EAS Attested event log."""
    event_counts["eas"] += 1
    topics = log.get("topics", [])
    recipient = ("0x" + topics[1][-40:]) if len(topics) > 1 else ""
    attester = ("0x" + topics[2][-40:]) if len(topics) > 2 else ""
    schema_id = topics[3] if len(topics) > 3 else ""
    uid = log.get("data", "")[:66]

    logger.info(
        f"[EAS] Attestation #{event_counts['eas']}: "
        f"recipient={recipient[:12]}… schema={schema_id[:12]}…"
    )
    log_alert(
        alert_type="eas_attestation",
        token_address="",
        wallet_address=recipient,
        detail={
            "recipient": recipient,
            "attester": attester,
            "schemaId": schema_id,
            "uid": uid,
            "blockNumber": log.get("blockNumber"),
            "txHash": log.get("transactionHash"),
        },
    )


def handle_erc8004_event(log: dict) -> None:
    """Process an ERC-8004 Registered event log."""
    event_counts["erc8004"] += 1
    topics = log.get("topics", [])
    owner = ("0x" + topics[2][-40:]) if len(topics) > 2 else ""
    agent_id_hex = topics[1] if len(topics) > 1 else "0x0"
    try:
        agent_id = int(agent_id_hex, 16)
    except Exception:
        agent_id = 0

    logger.info(
        f"[ERC-8004] New agent #{event_counts['erc8004']}: "
        f"agentId={agent_id} owner={owner[:12]}…"
    )

    log_alert(
        alert_type="erc8004_registered",
        token_address="",
        wallet_address=owner,
        detail={
            "agentId": agent_id,
            "owner": owner,
            "blockNumber": log.get("blockNumber"),
            "txHash": log.get("transactionHash"),
        },
    )

    if owner:
        touch_agent_updated(owner)


# ─── WebSocket listener ───────────────────────────────────────────────────────

async def _listen_wss(wss_url: str) -> None:
    """
    Subscribe to contract events via eth_subscribe over WebSocket.
    Raises on connection failure so caller can trigger reconnect.
    """
    try:
        import websockets  # type: ignore
    except ImportError:
        raise RuntimeError("websockets package not installed. Run: pip install websockets")

    logger.info(f"Connecting to WebSocket: {wss_url[:40]}…")

    async with websockets.connect(wss_url, ping_interval=30, ping_timeout=10) as ws:
        # Subscribe to EAS logs
        await ws.send(json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "method": "eth_subscribe",
            "params": ["logs", {
                "address": EAS_ADDRESS,
                "topics": [EAS_ATTESTED_TOPIC],
            }],
        }))
        eas_resp = json.loads(await ws.recv())
        eas_sub_id = eas_resp.get("result")
        logger.info(f"EAS subscription: {eas_sub_id}")

        # Subscribe to ERC-8004 logs
        await ws.send(json.dumps({
            "jsonrpc": "2.0", "id": 2,
            "method": "eth_subscribe",
            "params": ["logs", {
                "address": IDENTITY_REGISTRY,
                "topics": [ERC8004_REGISTERED_TOPIC],
            }],
        }))
        erc_resp = json.loads(await ws.recv())
        erc_sub_id = erc_resp.get("result")
        logger.info(f"ERC-8004 subscription: {erc_sub_id}")

        logger.info("✅ WebSocket listeners active — watching EAS + ERC-8004 events")

        async for message in ws:
            if _shutdown:
                break
            try:
                msg = json.loads(message)
                params = msg.get("params", {})
                result = params.get("result", {})
                sub_id = params.get("subscription")

                if sub_id == eas_sub_id:
                    handle_eas_event(result)
                elif sub_id == erc_sub_id:
                    handle_erc8004_event(result)
            except Exception as e:
                logger.warning(f"Message parse error: {e}")


# ─── HTTP polling fallback ────────────────────────────────────────────────────

async def _poll_http(rpc_url: str, poll_interval: int = 30) -> None:
    """
    Fallback HTTP polling when no WebSocket URL configured.
    Polls getLogs every poll_interval seconds.
    """
    import httpx

    logger.info(f"HTTP event polling fallback (every {poll_interval}s) via {rpc_url}")
    last_block = 0

    async with httpx.AsyncClient() as client:
        while not _shutdown:
            try:
                # Get current block
                resp = await client.post(rpc_url, json={
                    "jsonrpc": "2.0", "id": 1,
                    "method": "eth_blockNumber", "params": [],
                }, timeout=10.0)
                current_block = int(resp.json()["result"], 16)

                if last_block == 0:
                    last_block = current_block - 100  # start 100 blocks back

                if current_block <= last_block:
                    await asyncio.sleep(poll_interval)
                    continue

                from_hex = hex(last_block + 1)
                to_hex = hex(current_block)

                # Poll EAS
                eas_resp = await client.post(rpc_url, json={
                    "jsonrpc": "2.0", "id": 2,
                    "method": "eth_getLogs",
                    "params": [{
                        "address": EAS_ADDRESS,
                        "topics": [EAS_ATTESTED_TOPIC],
                        "fromBlock": from_hex,
                        "toBlock": to_hex,
                    }],
                }, timeout=15.0)
                for log in (eas_resp.json().get("result") or []):
                    handle_eas_event(log)

                # Poll ERC-8004
                erc_resp = await client.post(rpc_url, json={
                    "jsonrpc": "2.0", "id": 3,
                    "method": "eth_getLogs",
                    "params": [{
                        "address": IDENTITY_REGISTRY,
                        "topics": [ERC8004_REGISTERED_TOPIC],
                        "fromBlock": from_hex,
                        "toBlock": to_hex,
                    }],
                }, timeout=15.0)
                for log in (erc_resp.json().get("result") or []):
                    handle_erc8004_event(log)

                last_block = current_block

            except Exception as e:
                logger.warning(f"HTTP poll error: {e}")

            await asyncio.sleep(poll_interval)


# ─── Main listener loop with reconnect ────────────────────────────────────────

async def chain_listener_loop() -> None:
    """
    Persistent chain listener with exponential backoff reconnection.
    Falls back to HTTP polling if BASE_WSS_URL is not set.
    Designed to run as asyncio.create_task().
    """
    global _shutdown

    if not BASE_WSS_URL:
        logger.info("No BASE_WSS_URL — using HTTP polling fallback")
        await _poll_http(BASE_RPC_URL)
        return

    reconnect_delay = RECONNECT_MIN

    while not _shutdown:
        try:
            await _listen_wss(BASE_WSS_URL)
            reconnect_delay = RECONNECT_MIN  # Reset on clean exit
        except Exception as e:
            if _shutdown:
                break
            logger.warning(
                f"WebSocket disconnected: {e}. "
                f"Reconnecting in {reconnect_delay}s…"
            )
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, RECONNECT_MAX)

    logger.info("Chain listener loop exited")


def get_status() -> dict:
    """Return current listener status."""
    return {
        "event_counts": dict(event_counts),
        "uptime_seconds": round(
            (datetime.now(timezone.utc) - start_time).total_seconds(), 1
        ),
        "wss_enabled": bool(BASE_WSS_URL),
        "rpc_url": BASE_RPC_URL,
    }


# ─── Standalone entrypoint ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    async def _main():
        global _shutdown
        try:
            await chain_listener_loop()
        except KeyboardInterrupt:
            _shutdown = True
            logger.info("Interrupted — shutting down")

    asyncio.run(_main())

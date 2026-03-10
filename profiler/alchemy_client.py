"""
Alchemy API client for Base chain transaction history.
Uses the Alchemy Asset Transfers API.
"""

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger("wadjet.alchemy")

ALCHEMY_API_KEY = os.environ.get("ALCHEMY_API_KEY", "okgmVpKT-5iqER0g5yjyn")
ALCHEMY_BASE_URL = f"https://base-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"

SESSION = requests.Session()
SESSION.headers.update({"Content-Type": "application/json"})


def _rpc(method: str, params: list, retries: int = 3) -> Optional[dict]:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    for attempt in range(retries):
        try:
            resp = SESSION.post(ALCHEMY_BASE_URL, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                logger.warning(f"RPC error: {data['error']}")
                return None
            return data.get("result")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"RPC {method} failed after {retries} attempts: {e}")
    return None


def get_asset_transfers(
    address: str,
    direction: str = "both",
    max_count: int = 1000,
    from_block: str = "0x0",
) -> list[dict]:
    """
    Fetch ERC20 + ETH transfers for an address using alchemy_getAssetTransfers.
    direction: 'from' | 'to' | 'both'
    """
    all_transfers = []
    category = ["external", "erc20", "erc721", "erc1155"]

    directions = ["from", "to"] if direction == "both" else [direction]

    for dir_ in directions:
        page_key = None
        fetched = 0

        while fetched < max_count:
            params = {
                "category": category,
                "withMetadata": True,
                "excludeZeroValue": True,
                "maxCount": hex(min(1000, max_count - fetched)),
                "fromBlock": from_block,
            }
            if dir_ == "from":
                params["fromAddress"] = address
            else:
                params["toAddress"] = address

            if page_key:
                params["pageKey"] = page_key

            result = _rpc("alchemy_getAssetTransfers", [params])
            if not result:
                break

            transfers = result.get("transfers", [])
            all_transfers.extend(transfers)
            fetched += len(transfers)

            page_key = result.get("pageKey")
            if not page_key or not transfers:
                break

    logger.debug(f"Fetched {len(all_transfers)} transfers for {address}")
    return all_transfers


def get_balance(address: str) -> float:
    """Get ETH balance in ETH (not wei)."""
    result = _rpc("eth_getBalance", [address, "latest"])
    if result is None:
        return 0.0
    return int(result, 16) / 1e18


def get_transaction_count(address: str) -> int:
    """Total tx count (nonce)."""
    result = _rpc("eth_getTransactionCount", [address, "latest"])
    if result is None:
        return 0
    return int(result, 16)


def get_block_number() -> int:
    result = _rpc("eth_blockNumber", [])
    if result is None:
        return 0
    return int(result, 16)

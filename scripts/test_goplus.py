#!/usr/bin/env python3
"""
GoPlus API integration test.
Tests both known-malicious and known-clean addresses on Base mainnet.

Usage:
    cd packages/wadjet
    python scripts/test_goplus.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_sources.goplus_client import GoPlusClient

# ─── Test fixtures ────────────────────────────────────────────────────────────

# Known-clean: USDC on Base mainnet (trusted Circle-issued contract)
USDC_BASE = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"

# Known-clean wallet: Coinbase's public multisig (Base deployer)
COINBASE_DEPLOYER = "0x3304e22ddaa22bcdc5fca2269b418046ae7b566a"

# High-risk token: known honeypot on Base (example; may change over time)
# Using a low-liquidity meme token address as proxy test
TEST_SUSPICIOUS_TOKEN = "0x0000000000000000000000000000000000000001"  # Zero addr → expect empty

BASE_CHAIN_ID = 8453


# ─── Helpers ─────────────────────────────────────────────────────────────────

def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(label: str, data: dict):
    print(f"\n▶ {label}")
    # Print just the key fields, not the raw blob
    display = {k: v for k, v in data.items() if k != "raw"}
    print(json.dumps(display, indent=2, default=str))


# ─── Tests ────────────────────────────────────────────────────────────────────

async def test_token_security(client: GoPlusClient):
    print_section("Token Security: USDC on Base (should be clean)")
    result = await client.check_token_security(USDC_BASE, chain_id=BASE_CHAIN_ID)
    print_result("USDC", result)

    assert result["address"] == USDC_BASE.lower(), "Address mismatch"
    assert result["is_honeypot"] is False, f"USDC flagged as honeypot! {result}"

    # Sell tax check — USDC should have 0 tax
    assert result["sell_tax"] == 0.0 or not result["available"], \
        f"USDC has unexpected sell tax: {result['sell_tax']}"

    print("\n✅ USDC token security check passed")


async def test_address_security(client: GoPlusClient):
    print_section("Address Security: Coinbase deployer (should be clean)")
    result = await client.check_address_security(COINBASE_DEPLOYER, chain_id=BASE_CHAIN_ID)
    print_result("Coinbase deployer", result)

    assert result["address"] == COINBASE_DEPLOYER.lower(), "Address mismatch"
    assert result["malicious_address"] is False, "Coinbase deployer flagged as malicious!"
    assert result["phishing_address"] is False, "Coinbase deployer flagged as phishing!"

    print("\n✅ Coinbase deployer address security check passed")


async def test_combined_risk(client: GoPlusClient):
    print_section("Combined Risk: wallet + token")
    result = await client.get_combined_risk(
        wallet_address=COINBASE_DEPLOYER,
        token_address=USDC_BASE,
        chain_id=BASE_CHAIN_ID,
    )

    print_result("Wallet (Coinbase deployer)", result["wallet"])
    print_result("Token (USDC)", result["token"])

    score_info = client.compute_goplus_score_delta(result["wallet"], result["token"])
    print(f"\n▶ Score delta: +{score_info['score_delta']} rugScore")
    print(f"  Flags: {score_info['goplus_flags'] or '(none)'}")
    print(f"  goplus_honeypot: {score_info['goplus_honeypot']}")
    print(f"  goplus_malicious: {score_info['goplus_malicious']}")

    assert score_info["score_delta"] == 0, \
        f"Expected 0 score delta for clean combo, got {score_info['score_delta']}"

    print("\n✅ Combined risk check passed — score delta = 0 for clean pair")


async def test_cache(client: GoPlusClient):
    print_section("Cache Test: second call should use cache")
    import time

    t0 = time.monotonic()
    await client.check_token_security(USDC_BASE, chain_id=BASE_CHAIN_ID)
    first = time.monotonic() - t0

    t0 = time.monotonic()
    await client.check_token_security(USDC_BASE, chain_id=BASE_CHAIN_ID)
    second = time.monotonic() - t0

    print(f"  First call:  {first*1000:.1f}ms")
    print(f"  Second call: {second*1000:.1f}ms (cached)")

    assert second < first or second < 0.05, \
        f"Cache not working — second call ({second*1000:.1f}ms) ≥ first ({first*1000:.1f}ms)"

    print("\n✅ Cache is working correctly")


async def test_score_computation(client: GoPlusClient):
    print_section("Score Computation: simulated malicious flags")

    # Simulate what a malicious token would look like
    mock_wallet = {
        "malicious_address": True,
        "phishing_address": False,
        "cybercrime": False,
        "stealing_attack": False,
        "money_laundering": False,
        "available": True,
    }
    mock_token = {
        "is_honeypot": True,
        "is_blacklisted": False,
        "owner_can_change_balance": True,
        "buy_tax": 0.0,
        "sell_tax": 0.99,   # 99% sell tax
        "available": True,
    }

    score_info = client.compute_goplus_score_delta(mock_wallet, mock_token)
    print(json.dumps({k: v for k, v in score_info.items() if k != "raw"}, indent=2))

    # malicious(30) + honeypot(40) + owner_change_balance(20) + extreme_sell_tax(30) = 120 → capped 100
    assert score_info["score_delta"] == 100, f"Expected 100, got {score_info['score_delta']}"
    assert score_info["goplus_malicious"] is True
    assert score_info["goplus_honeypot"] is True
    assert score_info["goplus_owner_can_change_balance"] is True
    assert "honeypot" in score_info["goplus_flags"]
    assert "malicious_address" in score_info["goplus_flags"]

    print("\n✅ Score computation correct — capped at 100")


async def main():
    print("\n🔍 GoPlus API Integration Tests")
    print("   Chain: Base mainnet (8453)")

    client = GoPlusClient(disk_cache=False)  # No disk cache for test runs

    try:
        await test_token_security(client)
        await test_address_security(client)
        await test_combined_risk(client)
        await test_cache(client)
        await test_score_computation(client)

        print("\n" + "=" * 60)
        print("  ✅ ALL TESTS PASSED")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n❌ ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

"""
anchor_case.py

Computes a SHA-256 hash of a finalized case document and anchors it to the
CaseAnchor smart contract on Polygon Amoy (or Sepolia). Called by the backend
once a case is confirmed as a genuine threat (POST /api/cases/:id/confirm).

Uses web3.py directly — no Node.js / ethers.js bridge needed. One language,
one dependency, fewer moving parts to debug live during a demo.

Install:
    pip install web3 python-dotenv

.env (see .env.example):
    RPC_URL=https://rpc-amoy.polygon.technology
    BACKEND_WALLET_PRIVATE_KEY=0x...
    CASEANCHOR_CONTRACT_ADDRESS=0x...
"""

import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

RPC_URL = os.environ["RPC_URL"]
PRIVATE_KEY = os.environ["BACKEND_WALLET_PRIVATE_KEY"]
CONTRACT_ADDRESS = Web3.to_checksum_address(os.environ["CASEANCHOR_CONTRACT_ADDRESS"])

# Fields that must NEVER be included in the hash — they're written AFTER
# anchoring, so including them would make the hash different every time you
# tried to verify it (a moving target hashing itself).
EXCLUDED_FIELDS = {"blockchain_hash", "blockchain_tx_id", "anchored_at", "_id", "__v"}

ABI_PATH = Path(__file__).parent / "contract" / "CaseAnchor_abi.json"


def _load_abi() -> list:
    with open(ABI_PATH) as f:
        return json.load(f)


def _get_contract(w3: Web3):
    return w3.eth.contract(address=CONTRACT_ADDRESS, abi=_load_abi())


def compute_case_hash(case_document: dict) -> bytes:
    """
    Deterministically hash a case document. Same input -> same hash, every
    time, on any machine — that's the whole point of a verifiable fingerprint.

    - Drop fields that get written after anchoring (see EXCLUDED_FIELDS).
    - sort_keys=True so field order never changes the hash.
    - separators=(",", ":") to strip whitespace (json.dumps is not
      whitespace-stable across Python/Node otherwise).
    """
    clean = {k: v for k, v in case_document.items() if k not in EXCLUDED_FIELDS}
    canonical = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def anchor_case(case_id: str, case_document: dict) -> dict:
    """
    Hash the case and submit it to the contract. Returns the fields that get
    written back onto the case record: blockchain_hash, blockchain_tx_id,
    anchored_at.

    Raises on failure (network error, revert, out of gas, etc.) — the caller
    (orchestrator.js via subprocess, or your FastAPI route) should catch this
    and decide how to surface it; don't silently swallow a failed anchor.
    """
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        raise ConnectionError(f"Could not reach RPC endpoint: {RPC_URL}")

    account = w3.eth.account.from_key(PRIVATE_KEY)
    contract = _get_contract(w3)
    case_hash = compute_case_hash(case_document)

    if contract.functions.isAnchored(case_id).call():
        raise ValueError(f"Case {case_id} is already anchored on-chain")
    priority_fee = w3.to_wei(2, "gwei")
    max_fee = max(w3.eth.gas_price * 2, priority_fee * 2)

    txn = contract.functions.anchorCase(case_id, case_hash).build_transaction(
        {
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address),
            "gas": 150_000,
            "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": priority_fee,
            "chainId": w3.eth.chain_id,
        }
    )
    signed = account.sign_transaction(txn)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    if receipt.status != 1:
        raise RuntimeError(f"Transaction reverted: {tx_hash.hex()}")

    block = w3.eth.get_block(receipt.blockNumber)

    return {
        "blockchain_hash": "0x" + case_hash.hex(),
        "blockchain_tx_id": tx_hash.hex(),
        "anchored_at": block["timestamp"],  # unix timestamp, convert to ISO in Node if needed
    }


if __name__ == "__main__":
    # Quick manual smoke test — replace with a real finalized case document.
    demo_case = {
        "case_id": "CASE-DEMO-001",
        "detection": {"classification": "phishing", "confidence_score": 0.97},
        "forensics": {"spf_result": "fail", "spoofing_risk_score": 0.88},
        "enrichment": {"geolocation": {"country": "XX"}, "infrastructure_type": "hosting"},
        "overall_risk_score": 0.91,
    }
    result = anchor_case(demo_case["case_id"], demo_case)
    print(json.dumps(result, indent=2))
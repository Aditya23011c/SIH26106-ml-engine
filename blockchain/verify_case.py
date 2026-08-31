"""
verify_case.py

Re-hashes a case document and compares it against what's stored on-chain.
This IS the tamper-evidence check — the thing you'll actually demo to judges:
"here's an untouched case (MATCH), here's one I edited in Mongo by hand
(MISMATCH)."

Install:
    pip install web3 python-dotenv
"""

import json
import os

from dotenv import load_dotenv
from web3 import Web3

from blockchain.anchor_case import _get_contract, compute_case_hash

load_dotenv()

RPC_URL = os.environ["RPC_URL"]


def verify_case(case_id: str, case_document: dict) -> dict:
    """
    Returns:
        {
          "match": bool,
          "computed_hash": "0x...",
          "onchain_hash": "0x...",
          "onchain_timestamp": <unix ts or None if never anchored>,
        }
    """
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        raise ConnectionError(f"Could not reach RPC endpoint: {RPC_URL}")

    contract = _get_contract(w3)
    onchain_hash, onchain_ts, anchored_by = contract.functions.getCase(case_id).call()

    if onchain_ts == 0:
        return {
            "match": False,
            "computed_hash": "0x" + compute_case_hash(case_document).hex(),
            "onchain_hash": None,
            "onchain_timestamp": None,
            "note": f"Case {case_id} has never been anchored.",
        }

    computed_hash = compute_case_hash(case_document)
    onchain_hash_hex = "0x" + onchain_hash.hex()
    computed_hash_hex = "0x" + computed_hash.hex()

    return {
        "match": computed_hash == onchain_hash,
        "computed_hash": computed_hash_hex,
        "onchain_hash": onchain_hash_hex,
        "onchain_timestamp": onchain_ts,
        "anchored_by": anchored_by,
    }


if __name__ == "__main__":
    # Manual smoke test — swap in a real case_id + document fetched from Mongo.
    demo_case = {
        "case_id": "CASE-DEMO-001",
        "detection": {"classification": "phishing", "confidence_score": 0.97},
        "forensics": {"spf_result": "fail", "spoofing_risk_score": 0.88},
        "enrichment": {"geolocation": {"country": "XX"}, "infrastructure_type": "hosting"},
        "overall_risk_score": 0.91,
    }
    print(json.dumps(verify_case(demo_case["case_id"], demo_case), indent=2))
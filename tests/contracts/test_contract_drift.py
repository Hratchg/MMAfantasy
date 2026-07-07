"""Phase 25 PARTNER-V22-02 regression tripwire — direct-JSON-read.

Pins models/xgb_v2-contract.json invariants. Does NOT route through
save_contract_json (Phase 18/19/24 lineage — direct-read tripwires are
independent of the helper that wrote them; a bug in the helper that
writes wrong values would still be caught).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONTRACT_PATH = REPO_ROOT / "models" / "xgb_v2-contract.json"

# Pinned values — sourced from CONTEXT D-05 + AUDIT-01 baseline + Phase 24 gate output.
EXPECTED_MODEL_SHA = "0b0b40afc8ec41d87508745a9b5f40a46f7d86c054b1ab2acece03d319f6fecd"
EXPECTED_FEATURE_HASH = "402d59aed0edac88062f3c76e1c9d96b05fe168cda218ea5ee610058c32caead"


def test_contract_file_exists():
    assert CONTRACT_PATH.is_file(), f"Missing artifact: {CONTRACT_PATH}"


def test_contract_has_required_fields():
    contract = json.loads(CONTRACT_PATH.read_text())
    required = {
        "schema_version",
        "gate_contract_ref",
        "feature_columns_hash",
        "min_partner_version_supported",
        "deprecation_policy",
        "model_artifact_sha256",
        "created_at",
    }
    assert required.issubset(contract.keys()), f"Missing fields: {required - contract.keys()}"


def test_contract_pinned_values():
    contract = json.loads(CONTRACT_PATH.read_text())
    assert contract["schema_version"] == "1.0.0"
    assert contract["gate_contract_ref"] == ".planning/gate_contract_v2.2.json"
    assert contract["feature_columns_hash"] == EXPECTED_FEATURE_HASH
    assert contract["model_artifact_sha256"] == EXPECTED_MODEL_SHA
    assert contract["deprecation_policy"] == "N >= 2 minor versions"
    assert contract["min_partner_version_supported"] == "1.0.0"


def test_contract_created_at_is_iso_date():
    import datetime as dt

    contract = json.loads(CONTRACT_PATH.read_text())
    # Raises ValueError if not YYYY-MM-DD
    dt.date.fromisoformat(contract["created_at"])

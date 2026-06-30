"""Phase 24 regression tripwire — direct JSON read of gate_contract_v2.2.json.

Per Phase 18/19 lineage (tests/regression/test_gate_contract_phase{18,19}.py):
this tripwire does NOT route through load_gate_contract(). The JSON file IS
the binding source. If documentation drifts in PROJECT.md narrative, the
JSON cannot — this test enforces that.

Phase 24 operator decision: `v2.2_gate_breaks_floor_accept_truth`. All three
slices produced formula_acc < 0.70; the operator accepted the empirical truth
per D-18 (no post-measurement renegotiation). `accuracy_min` reflects the
un-floored mechanical formula output. The operator floor moves to v2.3+ as a
tighter empirical commitment after the data substrate fills.

Asserts:
  - .planning/gate_contract_v2.2.json exists post-Phase-24 close
  - version == "v2.2"
  - n_features == 90 (FEATURE_COLUMNS_V22)
  - base_features_set == "FEATURE_COLUMNS_V22"
  - formula_hash == v2.1 hash BYTE-IDENTICAL (D-01 binding; not renegotiated)
  - k_value == 1
  - cutoff_date == "2023-01-01"
  - n_seeds_observed == 10
  - All 3 slices present (most_recent_12mo / most_recent_24mo / random_15pct)
  - Per-slice accuracy_min reflects the un-floored mechanical output
    (operator decision `accept_truth`; floor moved to v2.3+); each slice
    matches the empirical values committed at Phase 24 close
  - operator_floor_applied == False on every slice (un-floored)
  - operator_decision block records `v2.2_gate_breaks_floor_accept_truth`
  - feature_columns_hash is a 64-char lowercase hex string (sha256 output)
  - bfo_backfill_committed_at is a non-empty ISO timestamp
  - models/xgb_v2.joblib SHA byte-identical to AUDIT-01 baseline (chain invariant)
  - .planning/gate_contract.json (v2.1) preserved unchanged
"""

from __future__ import annotations

import hashlib
import json
import pathlib

V22_CONTRACT_PATH = pathlib.Path(".planning/gate_contract_v2.2.json")
V21_CONTRACT_PATH = pathlib.Path(".planning/gate_contract.json")
XGB_V2_PATH = pathlib.Path("models/xgb_v2.joblib")

EXPECTED_FORMULA_HASH = "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"
EXPECTED_N_FEATURES = 90
EXPECTED_BASE_FEATURES_SET = "FEATURE_COLUMNS_V22"
EXPECTED_CUTOFF_DATE = "2023-01-01"
EXPECTED_K_VALUE = 1
EXPECTED_SLICES = ("most_recent_12mo", "most_recent_24mo", "random_15pct")
OPERATOR_ACCURACY_FLOOR = 0.70

# Per-slice empirical thresholds committed at Phase 24 close (un-floored;
# operator decision `accept_truth`).
EXPECTED_PER_SLICE = {
    "most_recent_12mo": {"brier_max": 0.2132, "accuracy_min": 0.6798},
    "most_recent_24mo": {"brier_max": 0.2164, "accuracy_min": 0.6587},
    "random_15pct": {"brier_max": 0.2172, "accuracy_min": 0.6648},
}

EXPECTED_OPERATOR_DECISION = "v2.2_gate_breaks_floor_accept_truth"

AUDIT_01_BASELINE_SHA = "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"


def _load_v22_contract() -> dict:
    assert V22_CONTRACT_PATH.exists(), (
        "gate_contract_v2.2.json missing — Phase 24 not closed; halt downstream work"
    )
    return json.loads(V22_CONTRACT_PATH.read_text(encoding="utf-8"))


def test_v22_contract_top_level_fields():
    """Top-level fields match the locked v2.2 spec."""
    c = _load_v22_contract()
    assert c["version"] == "v2.2", c["version"]
    assert c["n_features"] == EXPECTED_N_FEATURES, c["n_features"]
    assert c["base_features_set"] == EXPECTED_BASE_FEATURES_SET, c["base_features_set"]
    assert c["k_value"] == EXPECTED_K_VALUE, c["k_value"]
    assert c["cutoff_date"] == EXPECTED_CUTOFF_DATE, c["cutoff_date"]
    assert c["n_seeds_observed"] == 10, c["n_seeds_observed"]


def test_v22_formula_hash_byte_identical_to_v21():
    """formula_hash MUST match v2.1 byte-for-byte (CONTEXT D-01 / D-18 binding)."""
    c = _load_v22_contract()
    assert c["formula_hash"] == EXPECTED_FORMULA_HASH, (
        f"formula_hash drift: got {c['formula_hash']}, expected {EXPECTED_FORMULA_HASH}"
    )


def test_v22_per_slice_values_match_phase24_close():
    """Per-slice thresholds match the un-floored mechanical formula output
    committed at Phase 24 close (operator decision `accept_truth`).
    """
    c = _load_v22_contract()
    for slice_name in EXPECTED_SLICES:
        assert slice_name in c["per_slice"], slice_name
        s = c["per_slice"][slice_name]
        expected = EXPECTED_PER_SLICE[slice_name]
        assert s["brier_max"] == expected["brier_max"], (
            f"{slice_name}: brier_max drift — got {s['brier_max']}, "
            f"expected {expected['brier_max']}"
        )
        assert s["accuracy_min"] == expected["accuracy_min"], (
            f"{slice_name}: accuracy_min drift — got {s['accuracy_min']}, "
            f"expected {expected['accuracy_min']}"
        )
        assert 0.0 <= s["brier_max"] <= 1.0, (
            f"{slice_name}: brier_max {s['brier_max']} out of [0, 1]"
        )


def test_v22_operator_floor_not_applied_accept_truth():
    """Operator decision `accept_truth`: floor NOT applied; thresholds un-floored.

    Per CONTEXT D-06 + D-18: formula breached 0.70 on all 3 slices; operator
    branch `v2.2_gate_breaks_floor_accept_truth` accepts empirical truth and
    moves the operator floor to v2.3+. Per-slice `accuracy_min` MUST reflect
    the un-floored mechanical output (i.e., < 0.70).
    """
    c = _load_v22_contract()
    for slice_name in EXPECTED_SLICES:
        s = c["per_slice"][slice_name]
        assert s["operator_floor_applied"] is False, (
            f"{slice_name}: operator_floor_applied should be False under "
            f"`accept_truth` decision; got {s['operator_floor_applied']}"
        )
        assert s["accuracy_min"] < OPERATOR_ACCURACY_FLOOR, (
            f"{slice_name}: accuracy_min {s['accuracy_min']} should be "
            f"un-floored (< {OPERATOR_ACCURACY_FLOOR}) per operator "
            f"`accept_truth` decision"
        )


def test_v22_operator_decision_recorded():
    """operator_decision block records the materialized branch + rationale."""
    c = _load_v22_contract()
    od = c.get("operator_decision", {})
    assert od.get("decision") == EXPECTED_OPERATOR_DECISION, (
        f"operator_decision.decision drift: got {od.get('decision')!r}, "
        f"expected {EXPECTED_OPERATOR_DECISION!r}"
    )
    assert od.get("decided_at"), "operator_decision.decided_at missing"
    assert od.get("rationale"), "operator_decision.rationale missing"


def test_v22_feature_columns_hash_is_hex_64():
    """feature_columns_hash is a 64-char lowercase hex string (sha256 output)."""
    c = _load_v22_contract()
    fch = c.get("feature_columns_hash", "")
    assert isinstance(fch, str) and len(fch) == 64, f"feature_columns_hash malformed: {fch!r}"
    int(fch, 16)  # raises ValueError if not hex


def test_v22_bfo_backfill_committed_at_present():
    """bfo_backfill_committed_at is a non-empty ISO timestamp."""
    c = _load_v22_contract()
    ts = c.get("bfo_backfill_committed_at", "")
    assert isinstance(ts, str) and ts, f"bfo_backfill_committed_at malformed: {ts!r}"
    # ISO-8601 sanity: contains "T" between date and time
    assert "T" in ts, ts


def test_xgb_v2_sha_byte_identical_to_baseline():
    """models/xgb_v2.joblib byte-identity preserved (AUDIT-01 invariant, CONTEXT D-10)."""
    assert XGB_V2_PATH.exists(), f"missing {XGB_V2_PATH}"
    sha = hashlib.sha256(XGB_V2_PATH.read_bytes()).hexdigest()
    assert sha == AUDIT_01_BASELINE_SHA, (
        f"xgb_v2.joblib SHA drift: got {sha}, expected {AUDIT_01_BASELINE_SHA}. "
        "AUDIT-01 chain broken — investigate immediately."
    )


def test_v21_contract_unchanged():
    """v2.1 contract at .planning/gate_contract.json preserved (CONTEXT D-07)."""
    assert V21_CONTRACT_PATH.exists(), "v2.1 gate_contract.json missing — audit lineage broken"
    v21 = json.loads(V21_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert v21["version"] == "v2.1", v21["version"]
    assert v21["formula_hash"] == EXPECTED_FORMULA_HASH, (
        "v2.1 formula_hash drift — audit lineage broken"
    )
    assert v21["n_features"] == 72, v21["n_features"]
    assert v21["base_features_set"] == "FEATURE_COLUMNS_NO_NET", v21["base_features_set"]


def test_v22_contract_loads_via_loader():
    """Phase 26 will read this contract via load_gate_contract(version='v2.2').

    The direct-JSON tests above catch on-disk drift; this test catches the
    OTHER half of the drift surface: schema mismatch between the live
    GateContract / PerSliceThresholds dataclasses and the on-disk JSON.
    The two together close the tripwire loop (CR-02 lineage).

    Symmetric with `test_v22_contract_top_level_fields` but routed through
    the public loader so dataclass-schema drift is caught at CI.
    """
    from ufc_prediction.ml.gate_contract import load_gate_contract

    load_gate_contract.cache_clear()
    try:
        contract = load_gate_contract(version="v2.2")
    finally:
        # Bookend even on assertion failure — don't leak cached state.
        load_gate_contract.cache_clear()

    # Top-level
    assert contract.version == "v2.2", contract.version
    assert contract.n_features == EXPECTED_N_FEATURES, contract.n_features
    assert contract.base_features_set == EXPECTED_BASE_FEATURES_SET, contract.base_features_set
    assert contract.formula_hash == EXPECTED_FORMULA_HASH, contract.formula_hash
    assert contract.k_value == EXPECTED_K_VALUE, contract.k_value

    # Per-slice values match the on-disk constants
    for slice_name in EXPECTED_SLICES:
        ts = contract.per_slice[slice_name]
        expected = EXPECTED_PER_SLICE[slice_name]
        assert ts.brier_max == expected["brier_max"], (
            f"{slice_name}: brier_max via loader = {ts.brier_max}, expected {expected['brier_max']}"
        )
        assert ts.accuracy_min == expected["accuracy_min"], (
            f"{slice_name}: accuracy_min via loader = {ts.accuracy_min}, "
            f"expected {expected['accuracy_min']}"
        )
        # Per-slice operator metadata reachable via the loader
        assert ts.operator_floor_applied is False, (
            f"{slice_name}: operator_floor_applied via loader should be "
            f"False under accept_truth; got {ts.operator_floor_applied}"
        )
        assert ts.operator_decision == "accept_truth", (
            f"{slice_name}: operator_decision via loader = "
            f"{ts.operator_decision!r}, expected 'accept_truth'"
        )

    # Top-level operator_decision block reachable via the loader
    od = contract.operator_decision or {}
    assert od.get("decision") == EXPECTED_OPERATOR_DECISION, (
        f"top-level operator_decision.decision via loader = "
        f"{od.get('decision')!r}, expected {EXPECTED_OPERATOR_DECISION!r}"
    )

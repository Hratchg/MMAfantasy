"""Phase 31 regression tripwire — v2.3 gate contract + AUDIT-01 chain + Path A/D.

Per Phase 24 lineage (tests/regression/test_gate_contract_phase24.py): this
tripwire does NOT route through load_gate_contract() exclusively. The JSON file
IS the binding source for v2.3 (Path A); the HALT-AND-DECIDE artifact IS the
binding source for Path D. If documentation drifts in PROJECT.md narrative, the
on-disk artifacts cannot — this test enforces that.

Phase 31 operator pre-commit (CONTEXT D-01): Path A — reinstate 0.70 operator
floor. `accuracy_min = max(formula_output, 0.70)` per D-05(v2.2) carry-forward.
NO post-measurement renegotiation (D-18 binding). Three materialization paths:

  - **Path A** (any slice formula_output ≥ 0.70 across the board): contract
    emitted, operator_floor_applied=True per slice, END SHA artifact written.

  - **Path D** (any slice formula_output < 0.70): 31-HALT-AND-DECIDE.md emitted
    with three pre-templated outcome paths; contract NOT emitted; END SHA NOT
    written (Pitfall 5).

This file's tests split into three groups:

  1. **Path-independent** (run always): D-24 row present, v2.1/v2.2 preserved
     untouched, xgb_v2.joblib byte-identity, supported versions includes v2.3,
     loader dispatch wired.

  2. **Path-A conditional** (skip if HALT artifact present): contract shape,
     formula_hash locked, per-slice floor application, ingest_completed_at +
     bfo_backfill_committed_at + feature_columns_hash present, supersedes
     includes D-21, operator_decision block records Path A pre-commit, END SHA
     artifact present with content == AUDIT-01 baseline.

  3. **Path-D conditional** (skip if contract present): HALT frontmatter
     correct, all 3 outcome-path IDs present, body sections present, contract
     NOT emitted, END SHA NOT written.

Implements: GATE-V23-01 (contract shape), GATE-V23-02 (audit lineage),
GATE-V23-03 (Path A/D materialization assertions), GATE-V23-04 (D-24 binding).
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

# ── Path constants ───────────────────────────────────────────────────────

V23_CONTRACT_PATH = pathlib.Path(".planning/gate_contract_v2.3.json")
V22_CONTRACT_PATH = pathlib.Path(".planning/gate_contract_v2.2.json")
V21_CONTRACT_PATH = pathlib.Path(".planning/gate_contract.json")
XGB_V2_PATH = pathlib.Path("models/xgb_v2.joblib")
PROJECT_MD_PATH = pathlib.Path(".planning/PROJECT.md")
AUDIT_01_BASELINE_PATH = pathlib.Path(".planning/AUDIT-01-BASELINE-SHA.txt")
PHASE_31_DIR = pathlib.Path(
    ".planning/phases/31-gate-v23-re-derivation-on-populated-substrate"
)
HALT_PATH = PHASE_31_DIR / "31-HALT-AND-DECIDE.md"
END_SHA_PATH = PHASE_31_DIR / "31-XGB-V2-SHA-PHASE-31-END.txt"

# ── Expected constants (D-18 lineage) ────────────────────────────────────
#
# IN-05 fix: source the load-bearing constants (formula hash, AUDIT-01
# baseline SHA, operator floor) from the CLI module so per-test equality
# cascades from ONE source-of-truth. Local literal-value drift guards live
# in tests/unit/test_cli_gate_spike.py (test_cli_formula_hash_constant_*,
# test_cli_xgb_v2_sha_constant_*, test_cli_operator_floor_constant_*) —
# those assert the LITERAL values; downstream tests then reuse the imports
# for cascade comparisons.

from ufc_prediction.cli.predict import (  # noqa: E402
    EXPECTED_FORMULA_HASH,
    EXPECTED_XGB_V2_SHA as AUDIT_01_BASELINE_SHA,
    OPERATOR_FLOOR_V23 as OPERATOR_ACCURACY_FLOOR,
)

EXPECTED_N_FEATURES = 90
EXPECTED_BASE_FEATURES_SET = "FEATURE_COLUMNS_V22"
EXPECTED_CUTOFF_DATE = "2023-01-01"
EXPECTED_K_VALUE = 1
EXPECTED_SLICES = ("most_recent_12mo", "most_recent_24mo", "random_15pct")
EXPECTED_BFO_TS = "2026-05-15T15:06:49-07:00"

PATH_D_OUTCOME_IDS = (
    "v23_substrate_fills_but_gate_lt_floor_path_d_delay_ship",
    "v23_substrate_fills_but_gate_lt_floor_path_d_ship_v22_publicly",
    "v23_substrate_fills_but_gate_lt_floor_path_d_accept_lower_floor",
)


# ── Outcome-path fixture ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def outcome_path():
    """Returns 'path_a' / 'path_d' based on file presence.

    Skips the entire module's conditional tests if neither artifact exists
    (Phase 31 not yet run).
    """
    halt_exists = HALT_PATH.exists()
    contract_exists = V23_CONTRACT_PATH.exists()
    if halt_exists:
        return "path_d"
    if contract_exists:
        return "path_a"
    pytest.skip(
        "Phase 31 spike not yet run (neither contract nor HALT artifact present)"
    )


# ── Path-INDEPENDENT tests (always run) ──────────────────────────────────


def test_xgb_v2_byte_identity_baseline_preserved():
    """models/xgb_v2.joblib SHA-256 equals AUDIT-01 baseline (Pitfall 5 invariant).

    Guards: T-31-02-02 (xgb_v2 byte-identity tampering). The AUDIT-01 chain
    requires this hash stays stable end-to-end across Phase 31.

    WR-07 fix: when the model artifact is absent from the checkout
    (sandbox without LFS pull, fresh clone, CWD outside repo root) the
    tripwire skips rather than red-fails. A red-fail here mis-signals SHA
    drift that has not occurred; "skipped because the artifact is not in
    this environment" is the honest signal.
    """
    if not XGB_V2_PATH.exists():
        pytest.skip(
            f"{XGB_V2_PATH} not in checkout; AUDIT-01 byte-identity test n/a"
        )
    sha = hashlib.sha256(XGB_V2_PATH.read_bytes()).hexdigest()
    assert sha == AUDIT_01_BASELINE_SHA, (
        f"xgb_v2.joblib SHA drift: got {sha}, expected {AUDIT_01_BASELINE_SHA}. "
        "AUDIT-01 chain broken — investigate immediately."
    )


def test_v21_v22_preserved_untouched():
    """v2.1 + v2.2 contracts BYTE-IDENTICAL post-Phase-31 (GATE-V23-02 audit-lineage).

    Guards: T-31-02-03 (v2.1+v2.2 byte-identity tampering). Phase 31 emits a
    NEW v2.3 contract; v2.1 + v2.2 MUST stay untouched.
    """
    assert V21_CONTRACT_PATH.exists(), "v2.1 contract missing — audit lineage broken"
    assert V22_CONTRACT_PATH.exists(), "v2.2 contract missing — audit lineage broken"
    v21 = json.loads(V21_CONTRACT_PATH.read_text(encoding="utf-8"))
    v22 = json.loads(V22_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert v21["version"] == "v2.1", v21["version"]
    assert v22["version"] == "v2.2", v22["version"]
    # Both formula_hashes MUST equal D-18 lock (no renegotiation across versions).
    assert v21["formula_hash"] == EXPECTED_FORMULA_HASH, (
        "v2.1 formula_hash drift — audit lineage broken"
    )
    assert v22["formula_hash"] == EXPECTED_FORMULA_HASH, (
        "v2.2 formula_hash drift — audit lineage broken"
    )


def test_d24_row_present_in_project_md():
    """D-24(v2.3, GATE) row present in PROJECT.md with D-18 formula hash (GATE-V23-04).

    Guards: Pitfall 1 (SC4 binding — D-24 row precedes spike). Also asserts
    the row cites the locked D-18 formula hash.
    """
    assert PROJECT_MD_PATH.exists(), f"missing {PROJECT_MD_PATH}"
    body = PROJECT_MD_PATH.read_text(encoding="utf-8")
    assert "D-24(v2.3, GATE)" in body, "D-24(v2.3, GATE) row missing from PROJECT.md"
    assert EXPECTED_FORMULA_HASH in body, (
        f"D-18 formula hash {EXPECTED_FORMULA_HASH} missing from PROJECT.md"
    )


def test_supported_versions_includes_v23():
    """gate_contract.SUPPORTED_VERSIONS includes 'v2.3' (Plan 31-01 Task 2 product)."""
    from ufc_prediction.ml.gate_contract import SUPPORTED_VERSIONS

    assert "v2.3" in SUPPORTED_VERSIONS, (
        f"'v2.3' missing from SUPPORTED_VERSIONS={sorted(SUPPORTED_VERSIONS)}"
    )


def test_loader_dispatch_for_v23_wired():
    """V23_CONTRACT_PATH points at gate_contract_v2.3.json (Plan 31-01 Task 2 product)."""
    from ufc_prediction.ml.gate_contract import V23_CONTRACT_PATH

    assert V23_CONTRACT_PATH.name == "gate_contract_v2.3.json", V23_CONTRACT_PATH


# ── Path-A conditional tests (skip if HALT artifact present) ─────────────


def _load_v23_contract() -> dict:
    """Direct JSON read of v2.3 contract; raises if missing."""
    assert V23_CONTRACT_PATH.exists(), (
        f"{V23_CONTRACT_PATH} missing — Path A expected but contract not emitted"
    )
    return json.loads(V23_CONTRACT_PATH.read_text(encoding="utf-8"))


def test_v23_contract_top_level_fields(outcome_path):
    """v2.3 contract top-level fields match locked spec (GATE-V23-01)."""
    if outcome_path != "path_a":
        pytest.skip(f"Path A test; current outcome={outcome_path}")
    c = _load_v23_contract()
    assert c["version"] == "v2.3", c["version"]
    assert c["formula_hash"] == EXPECTED_FORMULA_HASH, (
        f"formula_hash drift: got {c['formula_hash']}, expected {EXPECTED_FORMULA_HASH}"
    )
    assert c["n_features"] == EXPECTED_N_FEATURES, c["n_features"]
    assert c["base_features_set"] == EXPECTED_BASE_FEATURES_SET, (
        c["base_features_set"]
    )
    assert c["k_value"] == EXPECTED_K_VALUE, c["k_value"]
    assert c["cutoff_date"] == EXPECTED_CUTOFF_DATE, c["cutoff_date"]
    assert c["n_seeds_observed"] == 10, c["n_seeds_observed"]


def test_v23_per_slice_floor_applied(outcome_path):
    """Every slice has accuracy_min >= 0.70 post-floor (Path A pre-commit, D-01)."""
    if outcome_path != "path_a":
        pytest.skip(f"Path A test; current outcome={outcome_path}")
    c = _load_v23_contract()
    for slice_name in EXPECTED_SLICES:
        s = c["per_slice"][slice_name]
        assert s["accuracy_min"] >= OPERATOR_ACCURACY_FLOOR, (
            f"{slice_name}: accuracy_min {s['accuracy_min']} < floor "
            f"{OPERATOR_ACCURACY_FLOOR} — Path A invariant violated"
        )
        # Per-slice metadata records the Path A semantics.
        assert s["operator_floor_value"] == OPERATOR_ACCURACY_FLOOR, (
            f"{slice_name}: operator_floor_value drift — "
            f"got {s['operator_floor_value']}"
        )


def test_v23_per_slice_keys_complete(outcome_path):
    """per_slice has exactly 3 keys: 12mo, 24mo, random_15pct (Phase 29 D-06 sizes)."""
    if outcome_path != "path_a":
        pytest.skip(f"Path A test; current outcome={outcome_path}")
    c = _load_v23_contract()
    keys = set(c["per_slice"].keys())
    assert keys == set(EXPECTED_SLICES), (
        f"per_slice keys drift: got {sorted(keys)}, expected {sorted(EXPECTED_SLICES)}"
    )


def test_v23_ingest_completed_at_present(outcome_path):
    """ingest_completed_at is non-empty ISO + distinct from bfo timestamp (Pitfall 7)."""
    if outcome_path != "path_a":
        pytest.skip(f"Path A test; current outcome={outcome_path}")
    c = _load_v23_contract()
    ingest_ts = c.get("ingest_completed_at", "")
    bfo_ts = c.get("bfo_backfill_committed_at", "")
    assert isinstance(ingest_ts, str) and ingest_ts, (
        f"ingest_completed_at malformed: {ingest_ts!r}"
    )
    # ISO-8601 sanity: contains "T" between date and time, plus timezone marker
    assert "T" in ingest_ts, ingest_ts
    assert ("+" in ingest_ts or "-" in ingest_ts[10:] or ingest_ts.endswith("Z")), (
        f"ingest_completed_at missing timezone offset: {ingest_ts!r}"
    )
    # MUST differ from bfo_backfill_committed_at — distinct phase commits (Pitfall 7).
    assert ingest_ts != bfo_ts, (
        f"ingest_completed_at == bfo_backfill_committed_at = {ingest_ts!r} — "
        "Pitfall 7 violation; the two should source from different phase dirs"
    )


def test_v23_feature_columns_hash_present(outcome_path):
    """feature_columns_hash is a 64-char lowercase hex string (Pitfall 3)."""
    if outcome_path != "path_a":
        pytest.skip(f"Path A test; current outcome={outcome_path}")
    c = _load_v23_contract()
    fch = c.get("feature_columns_hash", "")
    assert isinstance(fch, str) and len(fch) == 64, (
        f"feature_columns_hash malformed: {fch!r}"
    )
    int(fch, 16)  # raises ValueError if not hex


def test_v23_bfo_backfill_committed_at_carries_forward(outcome_path):
    """bfo_backfill_committed_at equals Phase 21 carry-forward verbatim."""
    if outcome_path != "path_a":
        pytest.skip(f"Path A test; current outcome={outcome_path}")
    c = _load_v23_contract()
    bfo_ts = c.get("bfo_backfill_committed_at", "")
    assert bfo_ts == EXPECTED_BFO_TS, (
        f"bfo_backfill_committed_at drift: got {bfo_ts!r}, expected {EXPECTED_BFO_TS!r} "
        "(Phase 21 carry-forward)"
    )


def test_v23_operator_decision_block_path_a(outcome_path):
    """Top-level operator_decision dict records Path A pre-commit (CONTEXT D-01)."""
    if outcome_path != "path_a":
        pytest.skip(f"Path A test; current outcome={outcome_path}")
    c = _load_v23_contract()
    od = c.get("operator_decision", {})
    assert isinstance(od, dict), f"operator_decision should be a dict; got {type(od)}"
    assert od.get("decision") == (
        "path_a_operator_floor_pre_committed_at_milestone_open"
    ), f"operator_decision.decision drift: got {od.get('decision')!r}"
    assert od.get("decided_at"), "operator_decision.decided_at missing"
    assert od.get("rationale"), "operator_decision.rationale missing"
    assert od.get("phase_32_implication"), (
        "operator_decision.phase_32_implication missing"
    )


def test_v23_supersedes_d21(outcome_path):
    """supersedes list includes 'D-21(v2.2, GATE)' (carry-forward chain)."""
    if outcome_path != "path_a":
        pytest.skip(f"Path A test; current outcome={outcome_path}")
    c = _load_v23_contract()
    supersedes = c.get("supersedes", [])
    assert "D-21(v2.2, GATE)" in supersedes, (
        f"supersedes missing D-21(v2.2, GATE); got {supersedes!r}"
    )


def test_v23_loader_returns_valid_contract(outcome_path):
    """load_gate_contract(version='v2.3') returns a valid GateContract (Pitfall 6)."""
    if outcome_path != "path_a":
        pytest.skip(f"Path A test; current outcome={outcome_path}")
    from ufc_prediction.ml.gate_contract import load_gate_contract

    load_gate_contract.cache_clear()
    try:
        contract = load_gate_contract(version="v2.3")
    finally:
        load_gate_contract.cache_clear()

    assert contract.version == "v2.3", contract.version
    assert contract.formula_hash == EXPECTED_FORMULA_HASH, contract.formula_hash
    assert contract.n_features == EXPECTED_N_FEATURES, contract.n_features
    assert contract.k_value == EXPECTED_K_VALUE, contract.k_value
    assert contract.ingest_completed_at, "ingest_completed_at empty via loader"
    # Per-slice metadata accessible via loader
    for slice_name in EXPECTED_SLICES:
        ts = contract.per_slice[slice_name]
        assert ts.accuracy_min >= OPERATOR_ACCURACY_FLOOR, (
            f"{slice_name}: accuracy_min {ts.accuracy_min} < {OPERATOR_ACCURACY_FLOOR} "
            "via loader"
        )


def test_phase_31_end_sha_artifact_present_path_a(outcome_path):
    """31-XGB-V2-SHA-PHASE-31-END.txt exists with AUDIT-01 baseline content (Path A only)."""
    if outcome_path != "path_a":
        pytest.skip(f"Path A test; current outcome={outcome_path}")
    assert END_SHA_PATH.exists(), (
        f"{END_SHA_PATH} missing — Path A should emit END SHA artifact "
        "(WR-04 carry-forward)"
    )
    content = END_SHA_PATH.read_text(encoding="utf-8").strip()
    assert content == AUDIT_01_BASELINE_SHA, (
        f"END SHA artifact content drift: got {content!r}, "
        f"expected {AUDIT_01_BASELINE_SHA!r}"
    )


# ── Path-D conditional tests (skip if contract present) ──────────────────


def test_halt_artifact_path_d_frontmatter(outcome_path):
    """31-HALT-AND-DECIDE.md frontmatter contains phase + artifact + operator_decision_required."""
    if outcome_path != "path_d":
        pytest.skip(f"Path D test; current outcome={outcome_path}")
    assert HALT_PATH.exists(), f"{HALT_PATH} missing — Path D requires HALT artifact"
    body = HALT_PATH.read_text(encoding="utf-8")
    assert "phase: 31-gate-v23-re-derivation-on-populated-substrate" in body
    assert "artifact: HALT-AND-DECIDE" in body
    assert "operator_decision_required: true" in body


def test_halt_artifact_path_d_outcome_path_ids(outcome_path):
    """31-HALT-AND-DECIDE.md lists all 3 pre-templated outcome-path IDs (Pattern 7)."""
    if outcome_path != "path_d":
        pytest.skip(f"Path D test; current outcome={outcome_path}")
    body = HALT_PATH.read_text(encoding="utf-8")
    for outcome_id in PATH_D_OUTCOME_IDS:
        assert outcome_id in body, (
            f"Outcome-path ID {outcome_id!r} missing from HALT artifact"
        )


def test_halt_artifact_path_d_body_sections(outcome_path):
    """31-HALT-AND-DECIDE.md contains required body sections (verbatim Phase 24 template)."""
    if outcome_path != "path_d":
        pytest.skip(f"Path D test; current outcome={outcome_path}")
    body = HALT_PATH.read_text(encoding="utf-8")
    assert "## Per-Slice Gap Diagnostic" in body
    assert "## Pre-Templated Outcome Paths" in body
    assert "## Operator Action Required" in body


def test_path_d_no_contract_emitted(outcome_path):
    """Path D blocks contract emit: gate_contract_v2.3.json MUST NOT exist."""
    if outcome_path != "path_d":
        pytest.skip(f"Path D test; current outcome={outcome_path}")
    assert not V23_CONTRACT_PATH.exists(), (
        f"{V23_CONTRACT_PATH} should NOT exist on Path D — operator-decision blocker"
    )


def test_path_d_no_phase_end_sha_artifact(outcome_path):
    """Path D blocks END SHA artifact emit (Pitfall 5; WR-04 carry-forward)."""
    if outcome_path != "path_d":
        pytest.skip(f"Path D test; current outcome={outcome_path}")
    assert not END_SHA_PATH.exists(), (
        f"{END_SHA_PATH} should NOT exist on Path D — Pitfall 5 violation"
    )

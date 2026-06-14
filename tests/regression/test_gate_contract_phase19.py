"""Phase 19 Wave-0 RED — assert gate_contract.json formula_hash + xgb_v2 SHA + n_features locked.

D-07(P19) carry-forward of Pitfall #16: any drift in gate_contract.json's formula_hash, xgb_v2.joblib's SHA-256 (baseline 6e7641...0a99), or xgb_v2_meta.json's n_features (=72) halts Phase 19. JSON+model are read DIRECTLY (no src plumbing) so RED stays decoupled from Wave 1 GREEN modules.
"""
import json
import pathlib

import pytest


def test_gate_contract_thresholds_are_v2_1_for_phase19():
    """D-06(P18): gate_contract.json must contain Phase-17-derived thresholds.

    Reads the JSON DIRECTLY (not via load_gate_contract()) so this test does
    not depend on src plumbing — Phase 17 verifier flagged a 4th-decimal
    narrative slip in PROJECT.md D-18 row; the JSON is the binding source.
    """
    contract_path = pathlib.Path(".planning/gate_contract.json")
    assert contract_path.exists(), \
        "gate_contract.json missing — Phase 17 not closed; halt Phase 19"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    # Sanity: top-level fields wired by Phase 17
    assert contract["version"] == "v2.1", \
        f"contract.version != v2.1; got {contract['version']!r}"
    assert contract["formula_hash"] == (
        "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"
    ), "Phase 17 operator-approved formula_hash drifted"

    # Per-slice thresholds — at least ONE of (brier_max < 0.215, accuracy_min > 0.67)
    # must hold per slice, otherwise the threshold equals v2.0's aspirational
    # default and Phase 17 has not actually run.
    for slice_name in ("most_recent_12mo", "most_recent_24mo", "random_15pct"):
        slice_data = contract["per_slice"][slice_name]
        v2_0_default_brier = 0.215
        v2_0_default_acc = 0.67
        is_phase_17_derived = (
            slice_data["brier_max"] < v2_0_default_brier
            or slice_data["accuracy_min"] > v2_0_default_acc
        )
        assert is_phase_17_derived, (
            f"{slice_name}: brier_max={slice_data['brier_max']}, "
            f"accuracy_min={slice_data['accuracy_min']} — both equal v2.0 "
            f"defaults; Phase 17 did not run; halt Phase 19"
        )


def test_xgb_v2_sha_baseline():
    """D-07(P19) part 2: xgb_v2.joblib SHA-256 must equal baseline (AUDIT-01 setup)."""
    import hashlib
    expected = "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
    model_path = pathlib.Path("models/xgb_v2.joblib")
    assert model_path.exists(), "models/xgb_v2.joblib missing — rollback path violated"
    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert actual == expected, (
        f"xgb_v2 SHA drift: got {actual} expected {expected}; "
        "xgb_v2 retraining FORBIDDEN per AUDIT-01"
    )


def test_xgb_v2_n_features():
    """D-07(P19) part 3: xgb_v2_meta.json n_features must equal 72 (Pitfall B / Phase 18 dispatch)."""
    meta_path = pathlib.Path("models/xgb_v2_meta.json")
    assert meta_path.exists(), "models/xgb_v2_meta.json missing"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["n_features"] == 72, (
        f"xgb_v2 n_features drift: got {meta['n_features']!r} expected 72; "
        "Phase 18 dispatch boundary violated"
    )

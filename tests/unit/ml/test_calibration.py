"""Phase 26 CALIB-V22-00 audit tests + sentinels for CALIB-V22-01/02 (Plan 26-03).

Per CONTEXT D-06: CALIB-V22-00 audit is the FIRST task of Phase 26 — surfaces
xgb_v2's existing calibration state so CALIB-V22-01 wires Platt as a REPLACEMENT
(not stacked, Pitfall #8 prevention).

Tests 1-2 PASS now (Plan 26-01 audit landing).
Tests 3-4 are SENTINELS for Plan 26-03 (CALIB-V22-01 + CALIB-V22-02) — they
remain @pytest.mark.skip with reason strings explicitly referencing the
activating plan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ufc_prediction.ml.calibration import audit_xgb_v2_calibration

CALIB_AUDIT_PATH = Path(
    ".planning/phases/26-forward-stepwise-candidate-promotion/CALIB_00_CURRENT_STATE.json"
)


@pytest.fixture
def fresh_audit_artifact(tmp_path):
    """Write the audit JSON into a tmp_path so tests don't write the canonical file repeatedly.

    The canonical artifact has already been written by Plan 26-01 Task 1 step C; these
    tests exercise the helper without clobbering that artifact.
    """
    out = tmp_path / "CALIB_00_CURRENT_STATE.json"
    audit = audit_xgb_v2_calibration(out_path=out)
    return audit, out


def test_audit_emits_artifact(fresh_audit_artifact):
    """test_audit_emits_artifact: helper writes a JSON file with the 8 required keys."""
    audit, out = fresh_audit_artifact
    assert out.exists(), f"audit did not write {out}"
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    required_keys = {
        "model_path",
        "model_class",
        "is_calibrated",
        "calibration_method",
        "n_calibrators",
        "double_calibration_risk",
        "audited_at",
        "decision",
    }
    assert required_keys <= set(on_disk.keys()), (
        f"missing keys: {required_keys - set(on_disk.keys())}"
    )
    # Returned dict mirrors on-disk content.
    assert on_disk == audit


def test_audit_detects_sigmoid_method(fresh_audit_artifact):
    """test_audit_detects_sigmoid_method: real xgb_v2 returns method='sigmoid' + risk=False."""
    audit, _ = fresh_audit_artifact
    assert audit["is_calibrated"] is True, audit
    assert audit["calibration_method"] == "sigmoid", audit
    assert audit["double_calibration_risk"] is False, audit
    assert audit["n_calibrators"] >= 1, audit


def test_canonical_artifact_on_disk_matches_expected():
    """The canonical artifact at .planning/phases/26-.../CALIB_00_CURRENT_STATE.json
    must reflect the expected post-audit state (Plan 26-01 must_have M1)."""
    assert CALIB_AUDIT_PATH.exists(), f"canonical audit missing at {CALIB_AUDIT_PATH}"
    on_disk = json.loads(CALIB_AUDIT_PATH.read_text(encoding="utf-8"))
    assert on_disk["calibration_method"] == "sigmoid", on_disk
    assert on_disk["is_calibrated"] is True, on_disk
    assert on_disk["double_calibration_risk"] is False, on_disk


def test_no_stacked_calibrators():
    """CALIB-V22-01 ACTIVATED — AST-parse trainer.py and count CalibratedClassifierCV
    invocations. Pitfall #8: there must be EXACTLY ONE (never stacked).
    """
    import ast

    src = Path("src/ufc_prediction/ml/trainer.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    ccv_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == "CalibratedClassifierCV")
            or (isinstance(n.func, ast.Attribute) and n.func.attr == "CalibratedClassifierCV")
        )
    ]
    assert len(ccv_calls) == 1, (
        f"expected exactly 1 CalibratedClassifierCV invocation in trainer.py; "
        f"got {len(ccv_calls)} — Pitfall #8 risk"
    )


def test_isotonic_conditional():
    """CALIB-V22-02 ACTIVATED — small-corpus dispatch returns 'sigmoid';
    large-corpus dispatch returns 'isotonic'. Threshold = ISOTONIC_THRESHOLD (1000)
    per D-14(v2.0).
    """
    from ufc_prediction.ml.trainer import (
        ISOTONIC_THRESHOLD,
        _pick_calibration_method,
    )

    # Boundary cases
    assert _pick_calibration_method(0) == "sigmoid"
    assert _pick_calibration_method(ISOTONIC_THRESHOLD - 1) == "sigmoid"
    assert _pick_calibration_method(ISOTONIC_THRESHOLD) == "isotonic"
    assert _pick_calibration_method(ISOTONIC_THRESHOLD + 1) == "isotonic"
    # Representative small + large corpus
    assert _pick_calibration_method(500) == "sigmoid"
    assert _pick_calibration_method(1500) == "isotonic"
    assert _pick_calibration_method(10_000) == "isotonic"


def test_calibration_method_explicit():
    """CALIB-V22-01 + CALIB-V22-02 — both method literals appear in trainer.py source
    (after stripping comments to avoid header-prose false positives).
    """
    src = Path("src/ufc_prediction/ml/trainer.py").read_text(encoding="utf-8")
    non_comment_lines = [line for line in src.splitlines() if not line.strip().startswith("#")]
    text = "\n".join(non_comment_lines)
    assert '"sigmoid"' in text or "'sigmoid'" in text, (
        "CALIB-V22-01 explicit sigmoid wiring missing in trainer.py (non-comment lines)"
    )
    assert '"isotonic"' in text or "'isotonic'" in text, (
        "CALIB-V22-02 isotonic conditional missing in trainer.py (non-comment lines)"
    )

"""Phase 26 META-V22-03 integration tests against the real spike artifacts.

Per Plan 26-02 Task 2 behaviors C1..C3:
- C1: meta_v2_candidate.{joblib,_meta.json} exist regardless of pass/fail.
- C2: meta_v2.joblib does NOT exist (promotion is Plan 26-04 only).
- C3: META_V22_SPIKE.json contains gate_verdict + brier_delta_vs_xgb_v2_baseline.

These tests run against the artifacts emitted by the actual spike run; they
do NOT re-run training (that would take ~minutes and require live DB access).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PHASE26_DIR = Path(".planning/phases/26-forward-stepwise-candidate-promotion")
SPIKE_PATH = PHASE26_DIR / "META_V22_SPIKE.json"
COEF_PATH = PHASE26_DIR / "META_V22_COEFFICIENT_STABILITY.json"
META_DIR = Path("models/meta")


pytestmark = pytest.mark.skipif(
    not SPIKE_PATH.exists(),
    reason="Phase 26 spike artifacts not present; run scripts/train_meta_v22.py first.",
)


def test_promotion_persists_candidate_either_way():
    """C1 — meta_v2_candidate.{joblib,_meta.json} exist after the spike,
    regardless of gate_verdict outcome (D-07(P15) hard-gate-then-save discipline)."""
    candidate_joblib = META_DIR / "meta_v2_candidate.joblib"
    candidate_meta = META_DIR / "meta_v2_candidate_meta.json"
    assert candidate_joblib.exists(), f"missing {candidate_joblib}"
    assert candidate_meta.exists(), f"missing {candidate_meta}"
    meta = json.loads(candidate_meta.read_text(encoding="utf-8"))
    assert meta["meta_version"] == "v2_candidate", meta["meta_version"]
    assert meta["meta_kind"] == "logistic", meta["meta_kind"]
    assert meta["phase"] == "26", meta["phase"]
    # candidate carries the ship_outcome PASS or FAIL marker.
    assert meta["metrics"]["ship_outcome"] in {"PASS", "FAIL"}


def test_meta_v2_joblib_not_promoted_yet():
    """C2 — `models/meta/meta_v2.joblib` does NOT exist after Plan 26-02.
    Promotion is Plan 26-04 only (D-10)."""
    meta_v2 = META_DIR / "meta_v2.joblib"
    assert not meta_v2.exists(), (
        f"meta_v2.joblib exists prematurely; promotion is Plan 26-04 only "
        f"(D-10 hard-gate-then-save discipline)"
    )


def test_spike_json_has_gate_verdict():
    """C3 — META_V22_SPIKE.json contains gate_verdict + brier_delta_vs_xgb_v2_baseline
    + per-slice keys + ship_outcome."""
    spike = json.loads(SPIKE_PATH.read_text(encoding="utf-8"))
    required_top_keys = {
        "gate_verdict",
        "brier_delta_vs_xgb_v2_baseline",
        "stepwise_clears_vs_xgb_v2",
        "ship_outcome",
        "median_per_slice",
        "xgb_v2_baseline_brier",
        "feature_set",
        "seeds",
        "feature_columns",
        "xgb_v2_sha256",
    }
    assert required_top_keys <= set(spike.keys()), (
        f"missing keys: {required_top_keys - set(spike.keys())}"
    )
    assert spike["feature_set"] == "v2.2"
    assert spike["ship_outcome"] in {"PASS", "FAIL"}
    # All 3 slices represented.
    for slc in ("most_recent_12mo", "most_recent_24mo", "random_15pct"):
        assert slc in spike["median_per_slice"]
        assert slc in spike["brier_delta_vs_xgb_v2_baseline"]
        assert slc in spike["xgb_v2_baseline_brier"]
    # AUDIT-01 SHA recorded.
    assert spike["xgb_v2_sha256"] == (
        "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
    )


def test_coefficient_stability_json_exists():
    """Side-test: META_V22_COEFFICIENT_STABILITY.json emitted (META-V22-04)."""
    assert COEF_PATH.exists(), f"missing {COEF_PATH}"
    report = json.loads(COEF_PATH.read_text(encoding="utf-8"))
    assert "per_feature_std" in report
    assert "per_feature_sign_agreement" in report
    assert report["n_seeds"] == 5

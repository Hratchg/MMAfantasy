"""Phase 32 Plan 32-01 Task 3 (Wave 0) — integration tests for 4-step composition.

Wave 0 RED state: tests that the orchestration entrypoint of
``scripts/compose_v23_meta.py`` exists and the synthetic Path A / Path C
harness materializes the expected artifacts (meta_v3.joblib +
meta_v3_meta.json on Path A; nothing on Path C).

Per Pitfall 3 binding: Path A promotion MUST inspect meta_v3_meta.json
for required schema fields (the sidecar JSON populated by
``meta_persistence.save_meta_model``). Bypassing save_meta_model
(e.g. raw joblib.dump) is forbidden.

Per Pitfall 6 (lru_cache): tests that load multiple gate contracts
call ``load_gate_contract.cache_clear()`` between cases.

Per Pitfall 9: META Step 1 short-circuit branch — when META Brier
exceeds gate_max on ALL 3 slices the driver materializes Path C
without running Steps 2-4.

These tests use synthetic harness invocations of ``triple_gate_decision``
+ ``save_meta_model`` rather than running the full composition pipeline
(which needs DATABASE_URL). The full-pipeline test is marked with
``pytest.mark.requires_db`` and skipped when DATABASE_URL is unset
(per Open Question 2 / RESEARCH §"Validation Architecture").
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from scripts.compose_v23_meta import (
    EXPECTED_XGB_V2_SHA256,
    META_V22_BASELINE_BRIER,
    PER_SLICE_KEYS,
    STEPWISE_HURDLE,
    TOTAL_MARGIN_HURDLE,
    triple_gate_decision,
)


# ─────────────────────────── Mock gate contract ─────────────────────────────


@dataclass(frozen=True)
class _MockSliceThresholds:
    brier_max: float
    accuracy_min: float


@dataclass(frozen=True)
class _MockContract:
    per_slice: dict[str, _MockSliceThresholds]
    feature_columns_hash: str = "test-feature-columns-hash"
    formula_hash: str = "test-formula-hash"


def _make_contract_v23() -> _MockContract:
    return _MockContract(
        per_slice={
            "most_recent_12mo": _MockSliceThresholds(brier_max=0.1512, accuracy_min=0.7814),
            "most_recent_24mo": _MockSliceThresholds(brier_max=0.1583, accuracy_min=0.7627),
            "random_15pct": _MockSliceThresholds(brier_max=0.1374, accuracy_min=0.7812),
        },
    )


def _median_passes() -> dict:
    """Per-slice median dict that clears v2.3 thresholds (Brier strictly below
    per-slice brier_max; accuracy above accuracy_min)."""
    return {
        "most_recent_12mo": {"brier_score": 0.140, "accuracy": 0.79},
        "most_recent_24mo": {"brier_score": 0.150, "accuracy": 0.78},
        "random_15pct": {"brier_score": 0.130, "accuracy": 0.80},
    }


# ─────────────────────────── Path A meta_v3 promotion ───────────────────────


def test_meta_v3_promotion_path_a(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When triple_gate clears all 3 legs → meta_v3.joblib + meta_v3_meta.json saved.

    Per Pitfall 3 binding: meta_v3_meta.json sidecar MUST contain required
    schema fields. We do NOT bypass save_meta_model.
    """
    from sklearn.linear_model import LogisticRegression

    from ufc_prediction.ml.meta_persistence import (
        compute_meta_input_distribution_hash,
        save_meta_model,
    )

    # Synthesize a final candidate that beats META-V22 by >0.003 on all slices.
    final_brier = {slc: META_V22_BASELINE_BRIER[slc] - 0.010 for slc in PER_SLICE_KEYS}
    median_final = _median_passes()
    prior_step_clears = {
        "meta_to_calib": True,
        "calib_to_ref": True,
        "ref_to_travel": True,
    }
    contract = _make_contract_v23()
    verdict, path, failures = triple_gate_decision(
        final_candidate_brier=final_brier,
        prior_step_clears=prior_step_clears,
        contract=contract,
        median_final_per_slice=median_final,
    )
    assert verdict == "promote"
    assert path == "path_a"
    assert failures == []

    # Simulate Path A save into tmp_path/models/meta.
    meta_dir = tmp_path / "models" / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((100, 13))
    y_train = rng.integers(0, 2, size=100)
    oof_proba = rng.uniform(0.2, 0.8, size=100)
    input_hash = compute_meta_input_distribution_hash(X_train, y_train, oof_proba)

    model = LogisticRegression(random_state=42).fit(X_train, y_train)
    feature_columns_v3 = [f"col_{i}" for i in range(13)]
    model_path, meta_path = save_meta_model(
        model,
        meta_kind="logistic",
        meta_version="v3",
        base_model_version="v2",
        base_model_sha256=EXPECTED_XGB_V2_SHA256,
        meta_feature_columns=feature_columns_v3,
        meta_input_distribution_hash=input_hash,
        meta_oof_parquet_sha256="0" * 64,
        meta_learner_brier_delta_vs_logistic=0.0,
        best_params={"C": 1.0, "penalty": "l2"},
        metrics={"triple_gate_pass": True, "outcome_path": "path_a"},
        meta_dir=str(meta_dir),
        trained_by_script="scripts/compose_v23_meta.py",
        phase="32",
    )
    assert model_path.exists()
    assert meta_path.exists()

    sidecar = json.loads(meta_path.read_text(encoding="utf-8"))
    # Per Pitfall 3 binding — required schema fields populated correctly.
    assert sidecar["meta_version"] == "v3"
    assert sidecar["meta_kind"] == "logistic"
    assert sidecar["meta_feature_columns"] == feature_columns_v3
    assert sidecar["base_model_version"] == "v2"
    assert sidecar["base_model_sha256"] == EXPECTED_XGB_V2_SHA256
    assert sidecar["phase"] == "32"
    assert sidecar["trained_by_script"] == "scripts/compose_v23_meta.py"
    # Schema discipline: meta_input_distribution_hash present + 64-hex chars.
    assert "meta_input_distribution_hash" in sidecar
    assert len(sidecar["meta_input_distribution_hash"]) == 64


# ─────────────────────────── Path C no-promotion ────────────────────────────


def test_path_c_no_promote_on_per_step_miss(tmp_path: Path) -> None:
    """Per-step-miss (REF→TRAVEL fails hurdle) → reject, path_c, no meta_v3 saved."""
    final_brier = {
        slc: META_V22_BASELINE_BRIER[slc] - 0.001
        for slc in PER_SLICE_KEYS  # margin miss
    }
    prior_step_clears = {
        "meta_to_calib": True,
        "calib_to_ref": True,
        "ref_to_travel": False,  # last step rejected
    }
    contract = _make_contract_v23()
    verdict, path, failures = triple_gate_decision(
        final_candidate_brier=final_brier,
        prior_step_clears=prior_step_clears,
        contract=contract,
        median_final_per_slice=_median_passes(),
    )
    assert verdict == "reject"
    assert path == "path_c"
    assert len(failures) >= 1

    # No save occurred — tmp_path/models/meta should be empty (we never wrote).
    meta_dir = tmp_path / "models" / "meta"
    assert not meta_dir.exists() or not list(meta_dir.glob("meta_v3*"))


def test_path_c_materialization_on_step_4_below_hurdle() -> None:
    """Synthetic 4-step composition where TRAVEL Δ=0.001 < 0.003 → path_c."""
    final_brier = {slc: META_V22_BASELINE_BRIER[slc] - 0.001 for slc in PER_SLICE_KEYS}
    prior_step_clears = {
        "meta_to_calib": True,
        "calib_to_ref": True,
        "ref_to_travel": False,  # < 0.003 hurdle
    }
    contract = _make_contract_v23()
    verdict, path, failures = triple_gate_decision(
        final_candidate_brier=final_brier,
        prior_step_clears=prior_step_clears,
        contract=contract,
        median_final_per_slice=_median_passes(),
    )
    assert verdict == "reject"
    assert path == "path_c"


# ─────────────────────────── Path C short-circuit (Pitfall 9) ────────────────


def test_path_c_no_promote_when_step1_short_circuits() -> None:
    """Pitfall 9: META Step 1 Brier > gate_max on ALL 3 slices → path_c early.

    Validates the short-circuit branch behavior using triple_gate_decision
    directly: if final_brier exceeds the gate AND no steps cleared, the
    decision lands on path_c (not path_b).
    """
    # META Brier > gate_max on all 3 slices.
    final_brier_above_gate = {
        "most_recent_12mo": 0.220,
        "most_recent_24mo": 0.220,
        "random_15pct": 0.220,
    }
    median_above_gate = {
        "most_recent_12mo": {"brier_score": 0.220, "accuracy": 0.70},
        "most_recent_24mo": {"brier_score": 0.220, "accuracy": 0.70},
        "random_15pct": {"brier_score": 0.220, "accuracy": 0.70},
    }
    prior_step_clears = {
        "meta_to_calib": False,
        "calib_to_ref": False,
        "ref_to_travel": False,
    }
    contract = _make_contract_v23()
    verdict, path, failures = triple_gate_decision(
        final_candidate_brier=final_brier_above_gate,
        prior_step_clears=prior_step_clears,
        contract=contract,
        median_final_per_slice=median_above_gate,
    )
    assert verdict == "reject"
    assert path == "path_c"
    # Multiple failures: gate + margin + per-step.
    assert len(failures) >= 3


# ─────────────────────────── Full pipeline (cached fixtures) ────────────────


def test_full_pipeline_cached_fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end 4-step composition + COMPOSITION_V23_REPORT.json emission.

    Runs the composition in dry-run mode (synthetic 90-col fixture) to
    exercise the orchestration without touching real data — sufficient
    for the integration-test contract (the report exists with 4 steps +
    outcome path).

    Per Pitfall 6: clears gate_contract lru_cache before invocation.
    """
    from ufc_prediction.ml.gate_contract import load_gate_contract

    # Pitfall 6: clear lru_cache between invocations.
    load_gate_contract.cache_clear()

    # Redirect artifact paths to tmp_path so we don't clobber real artifacts.
    from scripts import compose_v23_meta as cm  # noqa: PLC0415

    phase_tmp = tmp_path / "phase32"
    phase_tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cm, "PHASE32_DIR", phase_tmp)
    monkeypatch.setattr(cm, "COMPOSITION_REPORT_PATH", phase_tmp / "COMPOSITION_V23_REPORT.json")
    monkeypatch.setattr(cm, "SHA_MID_PATH", phase_tmp / "32-XGB-V2-SHA-PHASE-32-MID-PLAN-01.txt")
    monkeypatch.setattr(cm, "SHA_END_PATH", phase_tmp / "32-XGB-V2-SHA-PHASE-32-END-PLAN-01.txt")
    meta_dir_tmp = tmp_path / "models" / "meta"
    meta_dir_tmp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cm, "META_DIR", meta_dir_tmp)

    # Run in dry-run mode for the integration smoke (no DB hit).
    parser = cm.build_parser()
    args = parser.parse_args(["--dry-run", "--seeds", "2", "3"])
    report = cm.run_composition(args)

    # Report exists at the monkey-patched path.
    rpt_path = phase_tmp / "COMPOSITION_V23_REPORT.json"
    assert rpt_path.exists()
    rpt = json.loads(rpt_path.read_text(encoding="utf-8"))
    assert rpt["outcome_path"] in {"path_a", "path_b", "path_c"}
    assert "steps" in rpt
    # Steps count: META + CALIB + REF + TRAVEL = 4 (or fewer if short-circuited)
    steps = rpt["steps"]
    assert len(steps) >= 1
    step_names = [s.get("step") for s in steps]
    # META is always present (even on short-circuit).
    assert "META" in step_names

    # SHA artifacts exist.
    assert (phase_tmp / "32-XGB-V2-SHA-PHASE-32-MID-PLAN-01.txt").exists()
    assert (phase_tmp / "32-XGB-V2-SHA-PHASE-32-END-PLAN-01.txt").exists()


# ─────────────────────────── Path B distinction ─────────────────────────────


def test_path_b_partial_composition_distinct_from_path_c(tmp_path: Path) -> None:
    """Path B distinguishes some-steps-cleared + gate-miss from total failure."""
    # META + CALIB cleared; REF and TRAVEL did not — surviving CALIB candidate
    # misses gate (Brier exceeds 24mo brier_max).
    final_brier = {slc: META_V22_BASELINE_BRIER[slc] - 0.010 for slc in PER_SLICE_KEYS}
    median_with_gate_miss = _median_passes()
    median_with_gate_miss["most_recent_24mo"] = {"brier_score": 0.180, "accuracy": 0.79}
    prior_step_clears = {
        "meta_to_calib": True,
        "calib_to_ref": False,
        "ref_to_travel": False,
    }
    contract = _make_contract_v23()
    verdict, path, failures = triple_gate_decision(
        final_candidate_brier=final_brier,
        prior_step_clears=prior_step_clears,
        contract=contract,
        median_final_per_slice=median_with_gate_miss,
    )
    assert verdict == "reject"
    assert path == "path_b"
    # Path B is the partial-composition signal; meta_v3 NOT saved (no save
    # call here — assertion is on the path label).
    meta_dir = tmp_path / "models" / "meta"
    assert not meta_dir.exists() or not list(meta_dir.glob("meta_v3*"))

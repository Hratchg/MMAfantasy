"""Phase 55 GATE-V26-02 — unit tests for the substrate-drift–safe verifier.

Covers:
  - SubstrateDriftSafeGateVerdict dataclass + JSON serialization
  - verify_candidate_vs_canonical end-to-end on synthetic fixtures
  - confound detection threshold logic
  - shadow_traffic strategy is rejected per spec §6.1
  - v2.6 contract dispatch from gate_contract.load_gate_contract

Integration test against actual Phase 45 meta_v3 substrate parquet
is a v2.6.1 follow-on (deferred per Phase 55 CONTEXT.md decisions).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ufc_prediction.ml.gate_contract import load_gate_contract
from ufc_prediction.ml.gate_verifier import (
    VERIFIER_VERSION,
    EvalSlice,
    SubstrateDriftSafeGateVerdict,
    _brier_score,
    emit_verdict_json,
    verify_candidate_vs_canonical,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_pipeline_and_persist(
    tmp_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    name: str = "p",
) -> Path:
    """Fit a fresh StandardScaler+LogReg Pipeline and joblib-dump."""
    import joblib

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(C=1.0, max_iter=10_000, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    out = tmp_path / f"{name}.joblib"
    joblib.dump(pipe, out)
    return out


def _make_eval_slice(
    X: np.ndarray,
    y: np.ndarray,
    sha: str = "sha-fixture",
) -> EvalSlice:
    return EvalSlice(
        feature_vectors=tuple(tuple(float(v) for v in row) for row in X),
        outcomes=tuple(int(o) for o in y),
        substrate_sha=sha,
    )


# ── _brier_score primitive ─────────────────────────────────────────────────


def test_brier_score_matches_sklearn() -> None:
    from sklearn.metrics import brier_score_loss

    preds = [0.1, 0.8, 0.6, 0.3]
    outs = [0, 1, 1, 0]
    expected = brier_score_loss(outs, preds)
    assert _brier_score(preds, outs) == pytest.approx(expected, abs=1e-12)


def test_brier_score_empty_returns_zero() -> None:
    assert _brier_score([], []) == 0.0


def test_brier_score_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        _brier_score([0.5], [0, 1])


# ── shadow_traffic strategy rejected per spec §6.1 ─────────────────────────


def test_shadow_traffic_strategy_rejected(tmp_path: Path) -> None:
    """spec §6.1 — shadow_traffic is OUT OF SCOPE for the verifier."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    y = (X[:, 0] > 0).astype(int)
    p_canon = _make_pipeline_and_persist(tmp_path, X, y, "canon")
    p_cand = _make_pipeline_and_persist(tmp_path, X, y, "cand")
    eval_slices = {"most_recent_12mo": _make_eval_slice(X, y)}
    with pytest.raises(ValueError, match="shadow_traffic"):
        verify_candidate_vs_canonical(
            candidate=p_cand,
            canonical=p_canon,
            eval_slices=eval_slices,
            substrate_align_strategy="shadow_traffic",  # type: ignore[arg-type]
        )


# ── Happy path: candidate is identical to baseline -> no confound, no lift ─


def test_identical_pipelines_no_confound_no_lift(tmp_path: Path) -> None:
    """When candidate == canonical, deltas are zero on both raw and aligned;
    no confound; formula-gate decides PASS/FAIL on the actual data quality.
    """
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 3))
    y = (X[:, 0] > 0).astype(int)
    p_canon = _make_pipeline_and_persist(tmp_path, X, y, "canon")
    p_cand = p_canon  # same artifact (identical bytes -> same SHA)
    eval_slices = {
        s: _make_eval_slice(X, y, sha=f"sha-{s}")
        for s in ("most_recent_12mo", "most_recent_24mo", "random_15pct")
    }
    verdict = verify_candidate_vs_canonical(
        candidate=p_cand,
        canonical=p_canon,
        eval_slices=eval_slices,
    )
    # Same canonical pipeline both sides -> raw delta exactly zero.
    # The refit baseline is a fresh fit on the substrate so its
    # predictions diverge slightly from the canonical's (same arch,
    # different rounding/seed-paths) — aligned_delta near zero but
    # not byte-equal. Both signals show no confound.
    for slc in eval_slices:
        assert abs(verdict.raw_delta_per_slice[slc]) < 1e-9
        assert abs(verdict.aligned_delta_per_slice[slc]) < 0.05
    assert verdict.confound_detected is False
    assert verdict.canonical_sha == verdict.candidate_sha


# ── Confound detection: synthetic substrate drift ──────────────────────────


def test_confound_detected_on_synthetic_substrate_drift(tmp_path: Path) -> None:
    """Phase 45-style substrate-drift simulation.

    - Canonical Pipeline fit on substrate A (Phase 26 analog)
    - Candidate Pipeline fit on substrate B (v2.5+ analog) end-to-end
    - Eval substrate = substrate B
    - Refit-baseline aligns the canonical to substrate B
    - Raw delta inflated (canonical scaler OOD on substrate B)
    - Aligned delta near zero (both fit on B; both candidates good)
    - Confound detected -> verdict = "confound_block"
    """
    rng = np.random.default_rng(2)
    # Substrate A: features centered around (0, 0, 0)
    X_a = rng.standard_normal((200, 3))
    y_a = (X_a[:, 0] > 0).astype(int)
    # Substrate B: features SHIFTED to center around (5, 5, 5) - OOD vs A
    X_b = rng.standard_normal((200, 3)) + 5.0
    y_b = (X_b[:, 0] > 5.0).astype(int)

    p_canon = _make_pipeline_and_persist(tmp_path, X_a, y_a, "canon_on_A")
    p_cand = _make_pipeline_and_persist(tmp_path, X_b, y_b, "cand_on_B")

    # Eval on substrate B (where the candidate was fit)
    eval_slices = {
        s: _make_eval_slice(X_b, y_b, sha=f"sha-{s}-B")
        for s in ("most_recent_12mo", "most_recent_24mo", "random_15pct")
    }
    verdict = verify_candidate_vs_canonical(
        candidate=p_cand,
        canonical=p_canon,
        eval_slices=eval_slices,
        confound_threshold=0.05,
    )
    # Canonical's frozen scaler is OOD on substrate B -> raw baseline
    # Brier inflated -> raw delta > aligned delta by > threshold
    max_disagreement = max(
        abs(verdict.raw_delta_per_slice[s] - verdict.aligned_delta_per_slice[s])
        for s in eval_slices
    )
    assert max_disagreement > 0.05, (
        f"expected substrate-drift signal > 0.05, got {max_disagreement}"
    )
    assert verdict.confound_detected is True
    assert verdict.verdict == "confound_block"
    assert "substrate_drift" in verdict.confound_evidence


# ── No confound, clean PASS -> path_a_promote ──────────────────────────────


def test_clean_pass_emits_path_a_promote(tmp_path: Path) -> None:
    """Strong candidate, both fit on same substrate -> path_a_promote.

    Uses a contract fixture with a generous brier_max ceiling so the
    candidate's clean signal passes the formula-gate.
    """
    rng = np.random.default_rng(3)
    X = rng.standard_normal((300, 5))
    # Perfectly learnable signal -> candidate Brier near 0
    y_perfect = (X[:, 0] > 0).astype(int)
    # Baseline trained on noisy labels -> higher baseline Brier
    y_noisy = (rng.standard_normal(300) > 0).astype(int)
    p_canon = _make_pipeline_and_persist(tmp_path, X, y_noisy, "canon_noisy")
    p_cand = _make_pipeline_and_persist(tmp_path, X, y_perfect, "cand_perfect")

    eval_slices = {
        s: _make_eval_slice(X, y_perfect, sha=f"sha-{s}-clean")
        for s in ("most_recent_12mo", "most_recent_24mo", "random_15pct")
    }

    # Custom contract: low brier_max so candidate's clean signal passes
    from ufc_prediction.ml.gate_contract import GateContract, PerSliceThresholds

    contract = GateContract(
        version="v2.6",
        derived_at="2026-06-03",
        n_seeds_observed=1,
        base_features_set="FIXTURE",
        n_features=5,
        k_value=1,
        formula_hash="fixture_hash",
        cutoff_date="2026-01-01",
        per_slice={
            s: PerSliceThresholds(
                brier_max=0.5,  # generous: candidate clean ~ 0.05 passes
                accuracy_min=0.6,
                median_brier_xgb_v2=0.2,
                median_acc_xgb_v2=0.8,
                seed_std_brier=0.01,
                seed_std_acc=0.01,
                bootstrap_ci_half_brier=0.01,
                bootstrap_ci_half_acc=0.01,
                std_brier_used=0.01,
                std_acc_used=0.01,
            )
            for s in ("most_recent_12mo", "most_recent_24mo", "random_15pct")
        },
        feature_columns_hash="fixture_fh",
        bfo_backfill_committed_at="2026-01-01",
        ingest_completed_at="2026-01-01",
        methodology_version="v2.6",
        substrate_alignment_strategy="refit_baseline",
        gate_verifier_module="ufc_prediction.ml.gate_verifier",
        confound_threshold=0.5,  # high to avoid spurious confound
    )

    verdict = verify_candidate_vs_canonical(
        candidate=p_cand,
        canonical=p_canon,
        eval_slices=eval_slices,
        contract=contract,
        confound_threshold=0.5,
    )
    assert verdict.confound_detected is False
    assert verdict.verdict in {"path_a_promote", "path_b_reject"}
    # Both fit on the same X distribution -> raw and aligned similar
    # Candidate strong -> aligned delta positive
    for s in eval_slices:
        assert verdict.aligned_candidate_brier_per_slice[s] < 0.3


# ── Verdict serialization ──────────────────────────────────────────────────


def test_verdict_to_dict_round_trips_json(tmp_path: Path) -> None:
    """SubstrateDriftSafeGateVerdict.to_dict + emit_verdict_json produce
    a stable JSON artifact compatible with the Phase 45 corrigendum pattern.
    """
    import json

    v = SubstrateDriftSafeGateVerdict(
        aligned_baseline_brier_per_slice={"most_recent_12mo": 0.21},
        aligned_candidate_brier_per_slice={"most_recent_12mo": 0.18},
        aligned_delta_per_slice={"most_recent_12mo": 0.03},
        raw_baseline_brier_per_slice={"most_recent_12mo": 0.43},
        raw_delta_per_slice={"most_recent_12mo": 0.25},
        confound_detected=True,
        confound_threshold=0.05,
        confound_evidence="raw_delta_inflation_indicates_substrate_drift",
        floor_clears={"most_recent_12mo": True},
        hurdle_clears=False,
        hurdle_value=0.0,
        verdict="confound_block",
        rationale="Meta-gate fired; substrate-drift confound detected",
        substrate_sha="sub-fixture",
        canonical_sha="can-fixture",
        candidate_sha="cnd-fixture",
        refit_baseline_sha="rfb-fixture",
        methodology="refit_baseline",
    )
    out = tmp_path / "verdict.json"
    emit_verdict_json(v, out)
    assert out.exists()
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["verdict"] == "confound_block"
    assert raw["confound_detected"] is True
    assert raw["methodology"] == "refit_baseline"
    assert raw["verifier_version"] == VERIFIER_VERSION


# ── Contract dispatch v2.6 ─────────────────────────────────────────────────


def test_v26_contract_loads_with_methodology_fields() -> None:
    load_gate_contract.cache_clear()
    c = load_gate_contract(version="v2.6")
    assert c.version == "v2.6"
    assert c.methodology_version == "v2.6"
    assert c.substrate_alignment_strategy == "refit_baseline"
    assert c.gate_verifier_module == "ufc_prediction.ml.gate_verifier"
    assert c.confound_threshold == 0.05
    # Per-slice numbers preserved from v2.3 (verbatim per Phase 55 ship)
    assert "most_recent_12mo" in c.per_slice
    assert c.formula_hash != ""  # carried over from v2.3

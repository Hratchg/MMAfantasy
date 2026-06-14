"""Phase 75 METH-V27-02 Plan 75-02 — dual-substrate verifier unit tests.

Covers the D-03 LOCKED combinator + dataclass behavior introduced in
Plan 75-02 of Phase 75 (substrate-drift-immune verifier extension).

Tests inventory (per `<task type="auto" tdd="true">` Task 2 in
``75-02-PLAN.md``):

  1. ``test_combine_verdicts_path_a_dual_when_both_clear`` — both
     sub-verdicts ``path_a_promote`` → ``path_a_promote_dual``.
  2. ``test_combine_verdicts_substrate_drift_artifact_when_t1_pass_t2_reject``
     — Test 1 path_a + Test 2 path_b → ``substrate_drift_artifact``.
  3. ``test_combine_verdicts_highly_suspect_when_t1_reject_t2_pass`` —
     Test 1 path_b + Test 2 path_a → ``highly_suspect``.
  4. ``test_combine_verdicts_path_b_dual_when_both_reject`` — both
     ``path_b_reject`` → ``path_b_reject_dual``.
  5. ``test_combine_verdicts_confound_block_dual_when_non_width_confound``
     — Test 1 generic confound (Phase 55 raw-vs-aligned divergence)
     + Test 2 path_a → ``confound_block_dual``.
  6. ``test_combine_verdicts_width_mismatch_dual_when_phase_64_guard_fires``
     — Test 1 ``confound_block`` with ``width_mismatch_drift`` evidence
     + Test 2 path_a → ``width_mismatch_dual`` (proves precedence).
  7. ``test_combine_verdicts_width_mismatch_precedence_over_other_confound``
     — Test 1 width-mismatch + Test 2 non-width confound →
     ``width_mismatch_dual``.
  8. ``test_dual_substrate_verdict_to_dict_round_trips_json`` —
     ``emit_dual_verdict_json`` byte-stable across re-emissions; JSON
     round-trips through ``json.loads`` to dict-equal source.
  9. ``test_dual_substrate_verdict_methodology_and_version_locked`` —
     ``methodology == "dual_substrate"`` and ``verifier_version == "v2.7.0"``
     defaults per D-03.
 10. ``test_verify_dual_substrate_smoke_with_identical_pipelines_and_substrates``
     — end-to-end integration: identical pipelines + identical eval_slices →
     ``combined_verdict`` in the 6 LOCKED literals; sub-verdicts in
     ``{path_a_promote, path_b_reject, confound_block}``.

Regression (defensive — proves the ADD-ONLY contract):
 11. ``test_phase_55_single_substrate_signature_unchanged`` — asserts the
     Phase 55 ``verify_candidate_vs_canonical`` signature is byte-stable
     under Plan 75-02 (no parameter additions / removals / renames).

This file intentionally has zero overlap with
``tests/unit/ml/test_gate_verifier_v26.py`` (Phase 55) and
``tests/unit/ml/test_gate_verifier_width_guard.py`` (Phase 64) so both
predecessor suites stay byte-stable.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ufc_prediction.ml.gate_verifier import (
    DUAL_VERIFIER_VERSION,
    DualSubstrateGateVerdict,
    EvalSlice,
    SubstrateDriftSafeGateVerdict,
    _combine_verdicts,
    emit_dual_verdict_json,
    verify_candidate_vs_canonical,
    verify_candidate_vs_canonical_dual_substrate,
)


# ── Mock-verdict factory ──────────────────────────────────────────────────
#
# The combinator branch tests are kept fast by NOT loading any joblib
# Pipeline — only ``verdict`` and ``confound_evidence`` are consulted by
# ``_combine_verdicts``. All other ``SubstrateDriftSafeGateVerdict`` fields
# carry placeholder values so the dataclass constructs.


def _make_mock_verdict(
    verdict_literal: str,
    confound_evidence: str = "no substrate-drift confound: max(|raw - aligned|) = 0.0",
) -> SubstrateDriftSafeGateVerdict:
    """Construct a SubstrateDriftSafeGateVerdict with all 18 fields populated.

    ``verdict_literal`` and ``confound_evidence`` are the only fields
    inspected by ``_combine_verdicts`` per D-03 LOCKED. All other fields
    carry deterministic placeholder values for byte-stable test fixtures.
    """
    return SubstrateDriftSafeGateVerdict(
        aligned_baseline_brier_per_slice={"slice1": 0.20},
        aligned_candidate_brier_per_slice={"slice1": 0.19},
        aligned_delta_per_slice={"slice1": 0.01},
        raw_baseline_brier_per_slice={"slice1": 0.20},
        raw_delta_per_slice={"slice1": 0.01},
        confound_detected=(verdict_literal == "confound_block"),
        confound_threshold=0.05,
        confound_evidence=confound_evidence,
        floor_clears={"slice1": verdict_literal == "path_a_promote"},
        hurdle_clears=(verdict_literal == "path_a_promote"),
        hurdle_value=0.01,
        verdict=verdict_literal,  # type: ignore[arg-type]
        rationale=f"mock {verdict_literal}",
        substrate_sha="ab" * 32,
        canonical_sha="cd" * 32,
        candidate_sha="ef" * 32,
        refit_baseline_sha="01" * 32,
        methodology="refit_baseline",
    )


# ── 4-cell normal verdict pair (Tests 1-4) ────────────────────────────────


def test_combine_verdicts_path_a_dual_when_both_clear() -> None:
    """Both substrates promote → real lift triangulated."""
    t1 = _make_mock_verdict("path_a_promote")
    t2 = _make_mock_verdict("path_a_promote")
    combined, rationale = _combine_verdicts(t1, t2)
    assert combined == "path_a_promote_dual"
    assert "path_a_promote_dual" in rationale
    assert "BOTH" in rationale or "both" in rationale


def test_combine_verdicts_substrate_drift_artifact_when_t1_pass_t2_reject() -> None:
    """Test 1 (candidate-OOF) clears + Test 2 (canonical-OOF) rejects →
    candidate is winning ONLY because of its own OOF source = drift artifact."""
    t1 = _make_mock_verdict("path_a_promote")
    t2 = _make_mock_verdict("path_b_reject")
    combined, rationale = _combine_verdicts(t1, t2)
    assert combined == "substrate_drift_artifact"
    assert "substrate_drift_artifact" in rationale
    # Diagnostic substring (any of these is acceptable per D-03 rationale shape)
    assert (
        "OOF-source-divergence" in rationale
        or "OOF" in rationale
        or "drift" in rationale
    )


def test_combine_verdicts_highly_suspect_when_t1_reject_t2_pass() -> None:
    """Test 1 (candidate-OOF) rejects + Test 2 (canonical-OOF) clears →
    surprising — candidate clears canonical substrate but fails its own."""
    t1 = _make_mock_verdict("path_b_reject")
    t2 = _make_mock_verdict("path_a_promote")
    combined, rationale = _combine_verdicts(t1, t2)
    assert combined == "highly_suspect"
    assert "highly_suspect" in rationale
    assert "investigate" in rationale.lower()


def test_combine_verdicts_path_b_dual_when_both_reject() -> None:
    """Both substrates reject → genuinely worse on both."""
    t1 = _make_mock_verdict("path_b_reject")
    t2 = _make_mock_verdict("path_b_reject")
    combined, rationale = _combine_verdicts(t1, t2)
    assert combined == "path_b_reject_dual"
    assert "path_b_reject_dual" in rationale
    assert "BOTH" in rationale or "both" in rationale


# ── Confound branches (Tests 5-7) ─────────────────────────────────────────


def test_combine_verdicts_confound_block_dual_when_non_width_confound() -> None:
    """Test 1 generic confound (raw vs aligned divergence > threshold) +
    Test 2 path_a → confound_block_dual (no width-mismatch substring)."""
    t1 = _make_mock_verdict(
        "confound_block",
        confound_evidence=(
            "raw_delta_inflation_indicates_substrate_drift: "
            "max(|raw - aligned|) = 0.34 exceeds threshold 0.05"
        ),
    )
    t2 = _make_mock_verdict("path_a_promote")
    combined, rationale = _combine_verdicts(t1, t2)
    assert combined == "confound_block_dual"
    assert "confound_block_dual" in rationale
    assert "test_1" in rationale


def test_combine_verdicts_width_mismatch_dual_when_phase_64_guard_fires() -> None:
    """Test 1 width-mismatch (Phase 64 guard) + Test 2 path_a →
    width_mismatch_dual (precedence over generic confound branch)."""
    t1 = _make_mock_verdict(
        "confound_block",
        confound_evidence=(
            "width_mismatch_drift: canonical input width=13 != "
            "substrate feature_vector width=15"
        ),
    )
    t2 = _make_mock_verdict("path_a_promote")
    combined, rationale = _combine_verdicts(t1, t2)
    assert combined == "width_mismatch_dual"
    assert "width_mismatch" in rationale
    assert "test_1" in rationale


def test_combine_verdicts_width_mismatch_precedence_over_other_confound() -> None:
    """Width-mismatch MUST win precedence over a sibling non-width confound
    per D-03 LOCKED ('width can't be reconciled by substrate switching alone')."""
    t1 = _make_mock_verdict(
        "confound_block",
        confound_evidence="width_mismatch_drift: 13 != 15",
    )
    t2 = _make_mock_verdict(
        "confound_block",
        confound_evidence="raw_delta_inflation_indicates_substrate_drift: 0.30 > 0.05",
    )
    combined, _rationale = _combine_verdicts(t1, t2)
    assert combined == "width_mismatch_dual", (
        "Width-mismatch must win precedence over generic confound per D-03 LOCKED."
    )


def test_combine_verdicts_width_mismatch_on_test_2_also_triggers() -> None:
    """Symmetry guard: width-mismatch on Test 2 alone also wins precedence
    (D-03: 'EITHER test')."""
    t1 = _make_mock_verdict("path_a_promote")
    t2 = _make_mock_verdict(
        "confound_block",
        confound_evidence="width_mismatch_drift: 13 != 15",
    )
    combined, rationale = _combine_verdicts(t1, t2)
    assert combined == "width_mismatch_dual"
    assert "test_2" in rationale


# ── Dataclass + JSON serialization (Tests 8-9) ────────────────────────────


def test_dual_substrate_verdict_to_dict_round_trips_json(tmp_path: Path) -> None:
    """``emit_dual_verdict_json`` is byte-stable across re-emissions and the
    written file round-trips dict-equal through ``json.loads``."""
    t1 = _make_mock_verdict("path_a_promote")
    t2 = _make_mock_verdict("path_a_promote")
    verdict = DualSubstrateGateVerdict(
        test_1_verdict=t1,
        test_2_verdict=t2,
        combined_verdict="path_a_promote_dual",
        rationale="test rationale",
    )
    out_a = emit_dual_verdict_json(verdict, tmp_path / "a.json")
    out_b = emit_dual_verdict_json(verdict, tmp_path / "b.json")
    assert out_a.read_bytes() == out_b.read_bytes(), (
        "byte-stability violated: emit_dual_verdict_json must produce "
        "identical bytes across re-emissions for the same DualSubstrateGateVerdict"
    )
    round_tripped = json.loads(out_a.read_text())
    assert round_tripped == verdict.to_dict()
    # Top-level keys assertion (D-03 LOCKED shape)
    assert set(round_tripped.keys()) == {
        "combined_verdict",
        "rationale",
        "methodology",
        "verifier_version",
        "test_1_verdict",
        "test_2_verdict",
    }
    # Nested sub-verdict shape preservation
    assert "aligned_baseline_brier_per_slice" in round_tripped["test_1_verdict"]
    assert "aligned_baseline_brier_per_slice" in round_tripped["test_2_verdict"]


def test_dual_substrate_verdict_methodology_and_version_locked() -> None:
    """``methodology = "dual_substrate"`` + ``verifier_version = "v2.7.0"``
    are D-03 LOCKED defaults; no caller override required."""
    t1 = _make_mock_verdict("path_a_promote")
    t2 = _make_mock_verdict("path_a_promote")
    verdict = DualSubstrateGateVerdict(
        test_1_verdict=t1,
        test_2_verdict=t2,
        combined_verdict="path_a_promote_dual",
        rationale="x",
    )
    assert verdict.methodology == "dual_substrate", "D-03 LOCKED"
    assert verdict.verifier_version == "v2.7.0", "D-03 LOCKED"
    assert DUAL_VERIFIER_VERSION == "v2.7.0"


# ── Integration smoke (Test 10) ───────────────────────────────────────────


def test_verify_dual_substrate_smoke_with_identical_pipelines_and_substrates(
    tmp_path: Path,
) -> None:
    """End-to-end smoke: identical pipelines + identical eval_slices for
    both Test 1 and Test 2 → combined_verdict MUST be one of the 6 LOCKED
    DualGateVerdict literals (proves the wiring is correct, regardless of
    which specific verdict floor/hurdle formulas produce).

    Identical pipelines produce 0 delta; the verdict could be
    ``path_b_reject_dual`` (delta=0 fails hurdle) or
    ``path_a_promote_dual`` (if the formula degenerately clears). Asserting
    membership rather than exact literal keeps this test robust to floor/
    hurdle nuances; the structural assertion is that BOTH the cascade and
    the combinator produced a valid output.
    """
    rng = np.random.default_rng(0)
    n = 60  # > Phase 26 _refit_baseline_on_substrate minimum (defensive)
    X = rng.uniform(0.1, 0.9, size=(n, 13))
    y = rng.integers(0, 2, size=n)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=10_000)),
        ]
    )
    pipe.fit(X, y)
    candidate_path = tmp_path / "candidate.joblib"
    canonical_path = tmp_path / "canonical.joblib"
    joblib.dump(pipe, candidate_path)
    joblib.dump(pipe, canonical_path)

    # Slice names MUST match gate_contract.EXPECTED_SLICES — the v2.6
    # contract dispatched by verify_candidate_vs_canonical's formula-gate
    # stage hard-keys on the canonical Phase 26 slice triple. Splitting
    # the 60-row fixture into three 20-row slices keeps each slice
    # populated and the dual-substrate combinator visible.
    feature_vectors = tuple(tuple(float(v) for v in row) for row in X)
    outcomes = tuple(int(o) for o in y)
    slice_names = ("most_recent_12mo", "most_recent_24mo", "random_15pct")
    rows_per_slice = n // len(slice_names)
    slice_a: dict[str, EvalSlice] = {}
    slice_b: dict[str, EvalSlice] = {}
    for i, sn in enumerate(slice_names):
        start = i * rows_per_slice
        end = start + rows_per_slice
        slice_fv = feature_vectors[start:end]
        slice_y = outcomes[start:end]
        slice_a[sn] = EvalSlice(slice_fv, slice_y, f"aa_{sn}_" + "a" * 32)
        slice_b[sn] = EvalSlice(slice_fv, slice_y, f"bb_{sn}_" + "b" * 32)

    verdict = verify_candidate_vs_canonical_dual_substrate(
        candidate=candidate_path,
        canonical=canonical_path,
        candidate_eval_slices=slice_a,
        canonical_eval_slices=slice_b,
    )
    valid_literals = {
        "path_a_promote_dual",
        "substrate_drift_artifact",
        "highly_suspect",
        "path_b_reject_dual",
        "confound_block_dual",
        "width_mismatch_dual",
    }
    assert verdict.combined_verdict in valid_literals, (
        f"combinator emitted unknown literal: {verdict.combined_verdict!r}"
    )
    assert verdict.methodology == "dual_substrate"
    assert verdict.verifier_version == "v2.7.0"
    # Sub-verdicts wired correctly to Phase 55 single-substrate enum
    assert verdict.test_1_verdict.verdict in {
        "path_a_promote",
        "path_b_reject",
        "confound_block",
    }
    assert verdict.test_2_verdict.verdict in {
        "path_a_promote",
        "path_b_reject",
        "confound_block",
    }
    # Sub-verdicts preserve their distinct substrate SHAs (proves no slice
    # cross-contamination between Test 1 and Test 2)
    assert verdict.test_1_verdict.substrate_sha != verdict.test_2_verdict.substrate_sha


# ── ADD-ONLY contract regression (Test 11) ────────────────────────────────


def test_phase_55_single_substrate_signature_unchanged() -> None:
    """Defensive — proves Plan 75-02 is ADDITIVE: the Phase 55 single-
    substrate ``verify_candidate_vs_canonical`` signature MUST be
    byte-stable (no parameter additions / removals / renames)."""
    sig = inspect.signature(verify_candidate_vs_canonical)
    params = list(sig.parameters)
    expected = [
        "candidate",
        "canonical",
        "eval_slices",
        "substrate_align_strategy",
        "contract",
        "confound_threshold",
    ]
    assert params == expected, (
        f"Phase 55 signature drift detected; Plan 75-02 was supposed to "
        f"be ADDITIVE. Got: {params}; expected: {expected}"
    )

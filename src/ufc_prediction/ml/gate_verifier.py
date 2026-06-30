"""Phase 55 GATE-V26-02 — substrate-drift–safe gate verifier.

Implementation of the contract documented in
`.planning/gate_methodology_v2.6.md` §6. Detects substrate-drift confounds
automatically (e.g., the Phase 45 meta_v3 candidate's apparent +0.28-0.43
Brier lift that was a scaler-fit-on-stale-substrate artifact) so the
operator's manual Path B override 2026-06-03 is reproduced as an
automatic ``confound_block`` verdict.

Two stages, per spec §5.4:

  Meta-gate stage (always runs first):
    Is the measurement structurally valid? Substrate alignment via
    methodology (a) refit-baseline is the always-on answer. Detection of
    additional confounds (raw vs aligned delta disagreement >
    confound_threshold) sets `confound_detected: True`.

  Formula-gate stage (runs on substrate-aligned numbers):
    Per-slice Brier floor check + total-margin hurdle (>=0.003).
    Threshold values from gate_contract_v<methodology_version>.json.
    Logic unchanged from D-18 LOCKED — applied to aligned numbers only.

A candidate with ``confound_detected: True`` is NEVER promoted, regardless
of whether the formula-gate clears on the raw numbers (spec §5.4 closing
paragraph).

Phase 45 meta_v3 regression test (spec §6.4): the Phase 55 ship gate
requires the verifier to emit ``confound_block`` for the Phase 45
candidate. Unit tests use synthetic fixtures; the integration test
against the actual Phase 45 substrate parquet is a v2.6.1 follow-on
(deferred per Phase 55 CONTEXT.md decisions).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ufc_prediction.ml.gate_contract import GateContract

# Verifier version tag — embedded in every verdict for audit-trail
# reproducibility. Bump when verifier logic changes (NOT when the
# methodology spec is updated — that's `gate_methodology_v2.6.md` ownership).
VERIFIER_VERSION: str = "v2.6.0"

SubstrateAlignStrategy = Literal["refit_baseline", "dual_test_set"]
GateVerdict = Literal["path_a_promote", "path_b_reject", "confound_block"]


@dataclass(frozen=True)
class EvalSlice:
    """One evaluation slice for the gate verifier.

    Per spec §6.1 the verifier accepts ``dict[str, EvalSlice]`` keyed by
    slice name (e.g., 'most_recent_12mo', 'most_recent_24mo',
    'random_15pct'). Each slice carries the per-prediction inputs needed
    to compute Brier on both baseline and candidate Pipelines.
    """

    # Per-prediction Level-1 feature vectors (shape: n_predictions x n_l1_features).
    # Stored as nested tuples (immutable) so the dataclass stays frozen.
    feature_vectors: tuple[tuple[float, ...], ...]
    # Per-prediction ground-truth outcomes (1 = win, 0 = loss). Same n_predictions.
    outcomes: tuple[int, ...]
    # Substrate snapshot SHA — used for verdict.substrate_sha audit trail.
    substrate_sha: str


@dataclass(frozen=True)
class SubstrateDriftSafeGateVerdict:
    """Verdict structure per `.planning/gate_methodology_v2.6.md` §6.2.

    Substrate-aligned numbers are the v2.6 source-of-truth; raw numbers
    are preserved as audit-trail context (mirrors the Phase 45
    corrigendum pattern in immutable artifact form).
    """

    # Substrate-aligned baseline metrics (v2.6 source-of-truth)
    aligned_baseline_brier_per_slice: dict[str, float]
    aligned_candidate_brier_per_slice: dict[str, float]
    aligned_delta_per_slice: dict[str, float]

    # Raw (pre-alignment) numbers — preserved for audit trail
    raw_baseline_brier_per_slice: dict[str, float]
    raw_delta_per_slice: dict[str, float]

    # Meta-gate stage — substrate-drift confound detection
    confound_detected: bool
    confound_threshold: float
    confound_evidence: str

    # Formula-gate stage — D-18 LOCKED, applied to ALIGNED numbers
    floor_clears: dict[str, bool]
    hurdle_clears: bool
    hurdle_value: float

    # Final verdict
    verdict: GateVerdict
    rationale: str

    # Audit-trail metadata
    substrate_sha: str
    canonical_sha: str
    candidate_sha: str
    refit_baseline_sha: str | None
    methodology: SubstrateAlignStrategy
    verifier_version: str = field(default=VERIFIER_VERSION)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form for `gate_verdict_v26_<candidate>.json` output."""
        return {
            "aligned_baseline_brier_per_slice": dict(self.aligned_baseline_brier_per_slice),
            "aligned_candidate_brier_per_slice": dict(self.aligned_candidate_brier_per_slice),
            "aligned_delta_per_slice": dict(self.aligned_delta_per_slice),
            "raw_baseline_brier_per_slice": dict(self.raw_baseline_brier_per_slice),
            "raw_delta_per_slice": dict(self.raw_delta_per_slice),
            "confound_detected": self.confound_detected,
            "confound_threshold": self.confound_threshold,
            "confound_evidence": self.confound_evidence,
            "floor_clears": dict(self.floor_clears),
            "hurdle_clears": self.hurdle_clears,
            "hurdle_value": self.hurdle_value,
            "verdict": self.verdict,
            "rationale": self.rationale,
            "substrate_sha": self.substrate_sha,
            "canonical_sha": self.canonical_sha,
            "candidate_sha": self.candidate_sha,
            "refit_baseline_sha": self.refit_baseline_sha,
            "methodology": self.methodology,
            "verifier_version": self.verifier_version,
        }


# ── Internal computation helpers ──────────────────────────────────────────


def _brier_score(predictions: list[float], outcomes: list[int]) -> float:
    """Mean squared error between predicted probabilities and binary outcomes.

    Identical formula to ``sklearn.metrics.brier_score_loss`` for the
    binary case. Implemented inline so the verifier module has zero
    dependencies on the sklearn API surface (Phase 51 mypy-strict
    pilot keeps `sklearn.*` in the `ignore_missing_imports` overrides).
    """
    if len(predictions) != len(outcomes):
        raise ValueError(
            f"length mismatch: predictions={len(predictions)} outcomes={len(outcomes)}"
        )
    if not predictions:
        return 0.0
    total = 0.0
    for p, y in zip(predictions, outcomes, strict=True):
        total += (p - y) ** 2
    return total / len(predictions)


def _sha256_hex(path: Path) -> str:
    """SHA-256 hex digest of file at ``path``. Streaming-friendly."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _predict_proba_with_pipeline(
    pipeline: Any,
    feature_vectors: tuple[tuple[float, ...], ...],
) -> list[float]:
    """Run an sklearn-style Pipeline's predict_proba and return P(class=1)."""
    import numpy as np

    X = np.array(feature_vectors)
    proba = pipeline.predict_proba(X)
    # Pipeline returns shape (n, 2) for binary classifiers; P(class=1) is col 1.
    return [float(p) for p in proba[:, 1]]


def _refit_baseline_on_substrate(
    substrate_features: tuple[tuple[float, ...], ...],
    substrate_outcomes: tuple[int, ...],
) -> Any:
    """Refit the canonical META-V22 architecture on the supplied substrate.

    Per spec §3.1 methodology (a) — the v2.6 default. Produces a refit
    Pipeline directly comparable to the candidate (both fit on the same
    substrate distribution; no OOD scaler response).

    The architecture is HARDCODED to the Phase 23-26 META-V22 canonical
    spec (``StandardScaler + LogisticRegression(C=1.0)``) per
    ``.planning/gate_methodology_v2.6.md`` §3.1. The canonical pipeline
    object is intentionally NOT consulted; the architecture is pinned by
    the methodology spec, not derived from the loaded model. This keeps
    the refit reproducible regardless of (a) canonical Pipeline's
    internal step naming, (b) whether the canonical is a plain sklearn
    Pipeline or a ``MetaLearnerLogistic`` wrapper, or (c) any future
    canonical hyperparameter drift.

    WR-03 fix: prior signature took a dead ``canonical_pipeline`` first
    argument that the body never read; callers were misleadingly led to
    believe the architecture was reflected from the loaded canonical.
    Parameter removed; both in-file callers updated.

    Result is a NEW Pipeline instance; no input is mutated (audit-trail
    invariant).
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Mirror canonical Pipeline architecture exactly. Phase 23-26
    # META-V22 used (StandardScaler, LogisticRegression(C=1.0)) — pin
    # those hyperparameters explicitly so the refit is reproducible
    # regardless of canonical Pipeline's internal step naming.
    refit = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(C=1.0, max_iter=10_000, random_state=0)),
        ]
    )
    X = np.array(substrate_features)
    y = np.array(substrate_outcomes)
    refit.fit(X, y)
    return refit


# ── Verifier entry point ──────────────────────────────────────────────────


def verify_candidate_vs_canonical(
    candidate: Path,
    canonical: Path,
    eval_slices: dict[str, EvalSlice],
    substrate_align_strategy: SubstrateAlignStrategy = "refit_baseline",
    *,
    contract: GateContract | None = None,
    confound_threshold: float = 0.05,
) -> SubstrateDriftSafeGateVerdict:
    """Substrate-drift–safe gate verifier.

    Per `.planning/gate_methodology_v2.6.md` §6. Two-stage verdict:

      1. Meta-gate (always): align baseline to candidate substrate; detect
         confounds (raw vs aligned delta disagreement > confound_threshold).
         Confound -> `confound_block` regardless of formula-gate.
      2. Formula-gate (on aligned numbers): per-slice floor + total-margin
         hurdle. Logic unchanged from D-18 LOCKED.

    Args:
        candidate: Path to candidate Pipeline (.joblib).
        canonical: Path to canonical Pipeline (e.g., meta_v2.joblib).
        eval_slices: Per-slice evaluation data (see EvalSlice dataclass).
        substrate_align_strategy: 'refit_baseline' (default, operator-
            approved v2.6 standard) or 'dual_test_set' (sanity check).
            'shadow_traffic' is OUT OF SCOPE per spec §6.1.
        contract: Optional GateContract instance. Defaults to v2.6
            contract loaded via gate_contract.load_gate_contract.
        confound_threshold: Brier-delta threshold for confound detection
            (default 0.05). Overridable per gate_contract_v2.6.json.

    Returns:
        SubstrateDriftSafeGateVerdict per spec §6.2.

    Raises:
        ValueError: if `substrate_align_strategy == 'shadow_traffic'`
            (out of scope per spec §6.1).
    """
    if substrate_align_strategy == "shadow_traffic":  # type: ignore[comparison-overlap]
        raise ValueError(
            "shadow_traffic strategy is out of scope per "
            ".planning/gate_methodology_v2.6.md §6.1 — production deployment "
            "concern, not verifier-time concern. Use refit_baseline or "
            "dual_test_set."
        )

    # Defer joblib import to function-scope: keeps module import cheap
    # for tests that don't actually load Pipelines.
    import joblib

    canonical_pipeline = joblib.load(canonical)
    candidate_pipeline = joblib.load(candidate)
    canonical_sha = _sha256_hex(canonical)
    candidate_sha = _sha256_hex(candidate)

    # ── Phase 64 FEAT-V261-01 Plan 64-01 — width-mismatch guard ──────────
    # When the canonical Pipeline's ``n_features_in_`` differs from the
    # per-row width of the eval substrate's ``feature_vectors``, the raw
    # canonical ``predict_proba`` call below would crash with an opaque
    # sklearn ValueError. Per CONTEXT D-01 the cleanest semantic is to
    # auto-confound: a feature-widened candidate is itself the most
    # extreme form of substrate drift, so we skip the raw predict and
    # return ``verdict="confound_block"`` with both widths surfaced in
    # the rationale for operator debuggability. The refit_baseline path
    # still runs end-to-end on the candidate-width substrate so the
    # aligned numbers carry real signal for the audit trail.
    #
    # Width-matched cases fall through to the unchanged Phase 55 logic.
    substrate_sha = _aggregate_substrate_sha(eval_slices)
    width_mismatch = _detect_pipeline_width_mismatch(
        canonical_pipeline,
        eval_slices,
    )
    if width_mismatch is not None:
        canonical_width, expected_width = width_mismatch
        return _build_width_mismatch_verdict(
            canonical_pipeline=canonical_pipeline,
            candidate_pipeline=candidate_pipeline,
            eval_slices=eval_slices,
            canonical_width=canonical_width,
            expected_width=expected_width,
            confound_threshold=confound_threshold,
            substrate_sha=substrate_sha,
            canonical_sha=canonical_sha,
            candidate_sha=candidate_sha,
            substrate_align_strategy=substrate_align_strategy,
        )

    # ── Phase 75 METH-V27-02 Plan 75-04 — candidate-side width guard ─────
    # Phase 64's width-guard above catches CANONICAL-vs-substrate mismatch
    # (the v2.6.1 single-substrate failure mode: feature-widened candidate
    # produces a wider substrate that the canonical Pipeline cannot ingest).
    # In the v2.7 dual-substrate methodology, Test 2 inverts the polarity:
    # the canonical-aligned substrate (Plan 75-01, 13-wide) is fed to a
    # potentially WIDER candidate (e.g., Phase 64 TRAVEL at 15-wide). The
    # canonical-vs-substrate check passes (both 13-wide); the candidate's
    # inner PolynomialFeatures would then crash with an opaque sklearn
    # ValueError on its 15-wide expectation. Mirror the Phase 64 guard
    # semantics: detect the mismatch upfront and return a confound_block
    # verdict carrying ``width_mismatch`` in confound_evidence so the Plan
    # 75-02 dual-substrate combinator routes it to ``width_mismatch_dual``.
    candidate_width_mismatch = _detect_pipeline_width_mismatch(
        candidate_pipeline,
        eval_slices,
    )
    if candidate_width_mismatch is not None:
        cand_width, expected_width = candidate_width_mismatch
        return _build_candidate_width_mismatch_verdict(
            candidate_width=cand_width,
            expected_width=expected_width,
            eval_slices=eval_slices,
            confound_threshold=confound_threshold,
            substrate_sha=substrate_sha,
            canonical_sha=canonical_sha,
            candidate_sha=candidate_sha,
            substrate_align_strategy=substrate_align_strategy,
        )

    # Raw measurement: canonical Pipeline directly on the eval substrate.
    # This is the Phase 45-style measurement that introduces the
    # substrate-drift artifact when the canonical scaler is OOD on
    # the new substrate.
    raw_baseline_brier: dict[str, float] = {}
    raw_delta: dict[str, float] = {}
    aligned_baseline_brier: dict[str, float] = {}
    aligned_candidate_brier: dict[str, float] = {}
    aligned_delta: dict[str, float] = {}

    # Aligned measurement: refit baseline on substrate; measure both
    # canonical (raw) and refit (aligned) for the meta-gate stage.
    refit_baseline_sha: str | None = None
    if substrate_align_strategy == "refit_baseline":
        # Aggregate substrate across all slices for the refit driver.
        # Order is sorted by slice name for determinism.
        agg_features: list[tuple[float, ...]] = []
        agg_outcomes: list[int] = []
        for slice_name in sorted(eval_slices):
            sl = eval_slices[slice_name]
            agg_features.extend(sl.feature_vectors)
            agg_outcomes.extend(sl.outcomes)
        # WR-03: canonical_pipeline no longer passed; architecture is
        # pinned by methodology §3.1, not derived from the loaded model.
        refit_baseline = _refit_baseline_on_substrate(
            tuple(agg_features),
            tuple(agg_outcomes),
        )
        # Audit trail: SHA the refit's serialized form.
        refit_baseline_sha = _hash_refit_baseline(refit_baseline)
    else:
        # dual_test_set: aligned baseline = canonical applied to its own
        # training-time substrate (Phase 26 partition). For Phase 55 ship,
        # we accept the canonical's raw output as "aligned" under the
        # assumption the caller has supplied dual-test eval_slices that
        # are already substrate-stratified. v2.6.1 may tighten this.
        refit_baseline = canonical_pipeline

    candidate_predictions_full: list[float] = []
    candidate_outcomes_full: list[int] = []

    for slice_name, sl in eval_slices.items():
        # Raw canonical baseline
        raw_canon_preds = _predict_proba_with_pipeline(
            canonical_pipeline,
            sl.feature_vectors,
        )
        raw_baseline_brier[slice_name] = _brier_score(raw_canon_preds, list(sl.outcomes))

        # Aligned baseline (refit)
        aligned_baseline_preds = _predict_proba_with_pipeline(
            refit_baseline,
            sl.feature_vectors,
        )
        aligned_baseline_brier[slice_name] = _brier_score(
            aligned_baseline_preds,
            list(sl.outcomes),
        )

        # Candidate (always end-to-end on its own substrate)
        candidate_preds = _predict_proba_with_pipeline(
            candidate_pipeline,
            sl.feature_vectors,
        )
        cand_brier = _brier_score(candidate_preds, list(sl.outcomes))
        aligned_candidate_brier[slice_name] = cand_brier

        # Deltas — higher Brier = worse calibration, so baseline - candidate
        # is the "improvement" measurement.
        raw_delta[slice_name] = raw_baseline_brier[slice_name] - cand_brier
        aligned_delta[slice_name] = aligned_baseline_brier[slice_name] - cand_brier

        candidate_predictions_full.extend(candidate_preds)
        candidate_outcomes_full.extend(sl.outcomes)

    # Meta-gate stage: confound detection
    max_delta_disagreement = max(abs(raw_delta[s] - aligned_delta[s]) for s in eval_slices)
    confound_detected = max_delta_disagreement > confound_threshold
    if confound_detected:
        confound_evidence = (
            f"raw_delta_inflation_indicates_substrate_drift: "
            f"max(|raw - aligned|) = {max_delta_disagreement:.4f} "
            f"> threshold {confound_threshold:.4f}. See "
            f".planning/gate_methodology_v2.6.md §1 for failure-mode "
            f"taxonomy. Manual operator override pattern (Phase 45 Path B) "
            f"is no longer required — this verdict surfaces the confound "
            f"automatically."
        )
    else:
        confound_evidence = (
            f"no substrate-drift confound: "
            f"max(|raw - aligned|) = {max_delta_disagreement:.4f} "
            f"<= threshold {confound_threshold:.4f}"
        )

    # Formula-gate stage (applied to ALIGNED numbers)
    if contract is None:
        from ufc_prediction.ml.gate_contract import load_gate_contract

        contract = load_gate_contract(version="v2.6")

    floor_clears: dict[str, bool] = {}
    total_margin = 0.0
    for slice_name in eval_slices:
        brier_max = contract.per_slice[slice_name].brier_max
        cand_brier = aligned_candidate_brier[slice_name]
        floor_clears[slice_name] = cand_brier <= brier_max
        total_margin += aligned_delta[slice_name]

    hurdle_value = total_margin
    hurdle_clears = hurdle_value >= 0.003

    # Verdict logic per spec §6.3
    if confound_detected:
        verdict: GateVerdict = "confound_block"
        rationale = (
            "Meta-gate fired: substrate-drift confound detected. "
            "Promotion BLOCKED per gate_methodology_v2.6.md §5.4. "
            f"{confound_evidence}"
        )
    elif all(floor_clears.values()) and hurdle_clears:
        verdict = "path_a_promote"
        rationale = (
            "Clean PASS on substrate-aligned numbers. "
            f"All slices clear floor; total margin {hurdle_value:.4f} "
            ">= 0.003 hurdle."
        )
    else:
        verdict = "path_b_reject"
        failed_floors = [s for s, ok in floor_clears.items() if not ok]
        rationale = (
            "Formula-gate failed on substrate-aligned measurement. "
            f"Floors failed: {failed_floors or 'none'}; "
            f"total margin {hurdle_value:.4f} "
            f"{'>=' if hurdle_clears else '<'} 0.003 hurdle."
        )

    return SubstrateDriftSafeGateVerdict(
        aligned_baseline_brier_per_slice=aligned_baseline_brier,
        aligned_candidate_brier_per_slice=aligned_candidate_brier,
        aligned_delta_per_slice=aligned_delta,
        raw_baseline_brier_per_slice=raw_baseline_brier,
        raw_delta_per_slice=raw_delta,
        confound_detected=confound_detected,
        confound_threshold=confound_threshold,
        confound_evidence=confound_evidence,
        floor_clears=floor_clears,
        hurdle_clears=hurdle_clears,
        hurdle_value=hurdle_value,
        verdict=verdict,
        rationale=rationale,
        substrate_sha=substrate_sha,
        canonical_sha=canonical_sha,
        candidate_sha=candidate_sha,
        refit_baseline_sha=refit_baseline_sha,
        methodology=substrate_align_strategy,
    )


def _introspect_pipeline_width(pipeline: Any) -> int | None:
    """Return the input width of a fitted pipeline-like, or ``None``.

    Tries two introspection paths so the width guard works for both the
    Phase 55 synthetic test artifacts (raw sklearn Pipelines) AND the
    Phase 26 production canonical (``MetaLearnerLogistic`` wrapper that
    exposes its sklearn Pipeline via a ``pipeline`` attribute):

      1. Direct attribute ``pipeline.n_features_in_`` — set by sklearn
         after ``fit`` on any sklearn estimator / Pipeline.
      2. One-level indirection ``pipeline.pipeline.n_features_in_`` —
         the ``MetaLearnerLogistic`` convention from
         ``src/ufc_prediction/ml/meta_learner.py``. Returns the width
         of the inner sklearn Pipeline so the guard can fire BEFORE
         the inner ``PolynomialFeatures`` (or any first step) blows up
         on a width-mismatched input.

    Returns ``None`` for hand-rolled pipeline-likes that expose neither —
    caller treats that as "cannot detect mismatch; fall through to raw
    predict" rather than failing closed.

    Phase 64 Plan 64-03 — extracted from ``_detect_pipeline_width_mismatch``
    to close the e2e-integration gap: the real ``meta_v2.joblib`` is a
    ``MetaLearnerLogistic`` (no top-level ``n_features_in_``), so the
    Plan 64-01 guard's direct-attribute check missed the canonical and
    let the verifier crash inside ``PolynomialFeatures``.
    """
    direct = getattr(pipeline, "n_features_in_", None)
    if direct is not None:
        return int(direct)
    # MetaLearnerLogistic convention: outer wrapper holds an inner
    # sklearn Pipeline under the ``pipeline`` attribute.
    inner = getattr(pipeline, "pipeline", None)
    if inner is not None:
        inner_width = getattr(inner, "n_features_in_", None)
        if inner_width is not None:
            return int(inner_width)
    return None


def _detect_pipeline_width_mismatch(
    pipeline: Any,
    eval_slices: dict[str, EvalSlice],
) -> tuple[int, int] | None:
    """Detect width mismatch between a fitted Pipeline and the eval substrate.

    Phase 64 D-01 helper. Returns ``(canonical_width, expected_width)``
    when the Pipeline's input width differs from the per-row width of
    the FIRST slice's first ``feature_vector`` (Phase 64 locks the
    substrate-builder to a single width across slices via the locked
    ``feature_columns`` order, so the first row is a representative
    sample).

    Returns ``None`` when the widths match — caller falls through to
    the unchanged Phase 55 raw-predict path.

    Width introspection goes through ``_introspect_pipeline_width`` so
    the guard works for both raw sklearn Pipelines (Phase 55 test
    artifacts) and ``MetaLearnerLogistic`` wrappers (Phase 26 production
    canonical ``meta_v2.joblib``). See ``_introspect_pipeline_width``
    docstring for the introspection cascade.
    """
    if not eval_slices:
        return None
    first_slice = next(iter(eval_slices.values()))
    if not first_slice.feature_vectors:
        return None
    expected_width = len(first_slice.feature_vectors[0])
    canonical_width = _introspect_pipeline_width(pipeline)
    if canonical_width is None:
        return None
    if int(canonical_width) == int(expected_width):
        return None
    return int(canonical_width), int(expected_width)


def _build_width_mismatch_verdict(
    *,
    canonical_pipeline: Any,
    candidate_pipeline: Any,
    eval_slices: dict[str, EvalSlice],
    canonical_width: int,
    expected_width: int,
    confound_threshold: float,
    substrate_sha: str,
    canonical_sha: str,
    candidate_sha: str,
    substrate_align_strategy: SubstrateAlignStrategy,
) -> SubstrateDriftSafeGateVerdict:
    """Construct the D-01 ``confound_block`` verdict on width-mismatch.

    Skips the raw canonical ``predict_proba`` (which would crash) and
    populates ``raw_*_per_slice`` with ``None`` for every slice. Runs
    the refit_baseline path on the candidate-width substrate so the
    aligned numbers carry real signal — the refit Pipeline is trained
    at candidate width by definition, so it accepts the substrate.

    Both rationale and confound_evidence carry the literal substring
    ``"width_mismatch"`` plus both numeric widths for operator
    debuggability (e.g., ``"width_mismatch_drift: canonical=13,
    candidate=15"``).
    """
    # Refit baseline on the candidate-width substrate. This is the
    # SAME helper Phase 55 uses; it retrains StandardScaler+LR on
    # whatever width the substrate supplies, so no width error.
    agg_features: list[tuple[float, ...]] = []
    agg_outcomes: list[int] = []
    for slice_name in sorted(eval_slices):
        sl = eval_slices[slice_name]
        agg_features.extend(sl.feature_vectors)
        agg_outcomes.extend(sl.outcomes)
    # WR-03: canonical_pipeline no longer passed; architecture is pinned
    # by methodology §3.1. The width-mismatch confound path STILL relies
    # on the refit producing a candidate-width-compatible Pipeline, which
    # it does because the refit fits on the supplied substrate (which is
    # already at candidate width).
    refit_baseline = _refit_baseline_on_substrate(
        tuple(agg_features),
        tuple(agg_outcomes),
    )
    refit_baseline_sha = _hash_refit_baseline(refit_baseline)

    raw_baseline_brier: dict[str, float] = {}
    raw_delta: dict[str, float] = {}
    aligned_baseline_brier: dict[str, float] = {}
    aligned_candidate_brier: dict[str, float] = {}
    aligned_delta: dict[str, float] = {}
    floor_clears: dict[str, bool] = {}

    for slice_name, sl in eval_slices.items():
        # Raw canonical predict is SKIPPED — sentinel None per D-01.
        # Cast away the dataclass annotation; runtime accepts None.
        raw_baseline_brier[slice_name] = None  # type: ignore[assignment]
        raw_delta[slice_name] = None  # type: ignore[assignment]

        aligned_baseline_preds = _predict_proba_with_pipeline(
            refit_baseline,
            sl.feature_vectors,
        )
        aligned_baseline_brier[slice_name] = _brier_score(
            aligned_baseline_preds,
            list(sl.outcomes),
        )

        candidate_preds = _predict_proba_with_pipeline(
            candidate_pipeline,
            sl.feature_vectors,
        )
        cand_brier = _brier_score(candidate_preds, list(sl.outcomes))
        aligned_candidate_brier[slice_name] = cand_brier
        aligned_delta[slice_name] = aligned_baseline_brier[slice_name] - cand_brier
        # Width-mismatch confound -> no slice clears its floor (verdict
        # is forced to confound_block regardless, so this is cosmetic
        # for the formula-gate fields the dataclass requires).
        floor_clears[slice_name] = False

    # WR-01 fix: the canonical width is discovered by
    # ``_introspect_pipeline_width`` which handles both
    # ``Pipeline.n_features_in_`` (the plain-sklearn case) and
    # ``MetaLearnerLogistic.pipeline.n_features_in_`` (the production
    # ``meta_v2.joblib`` case). The prior message hardcoded the
    # ``canonical_pipeline.n_features_in_`` attribute path which does NOT
    # exist on ``MetaLearnerLogistic`` — operators who try to look it up
    # directly on ``meta_v2.joblib`` would get ``AttributeError``. Use a
    # neutral phrasing that names the introspection helper instead.
    confound_evidence = (
        f"width_mismatch_drift: canonical input width (via "
        f"_introspect_pipeline_width)={canonical_width} != substrate "
        f"feature_vector width={expected_width}. Raw canonical predict "
        f"skipped; aligned numbers from refit_baseline."
    )
    rationale = (
        f"confound_block: width_mismatch_drift "
        f"(canonical={canonical_width}, candidate={expected_width}). "
        f"See confound_evidence."
    )
    if substrate_align_strategy == "dual_test_set":
        # Dual-test was requested but the canonical-as-aligned baseline
        # would have crashed on the width mismatch; we forced refit
        # internally. Surface that in the audit trail.
        confound_evidence = (
            f"{confound_evidence} (dual_test_set strategy requested but "
            f"forced to refit_baseline due to width mismatch.)"
        )

    return SubstrateDriftSafeGateVerdict(
        aligned_baseline_brier_per_slice=aligned_baseline_brier,
        aligned_candidate_brier_per_slice=aligned_candidate_brier,
        aligned_delta_per_slice=aligned_delta,
        raw_baseline_brier_per_slice=raw_baseline_brier,
        raw_delta_per_slice=raw_delta,
        confound_detected=True,
        confound_threshold=confound_threshold,
        confound_evidence=confound_evidence,
        floor_clears=floor_clears,
        hurdle_clears=False,
        hurdle_value=0.0,
        verdict="confound_block",
        rationale=rationale,
        substrate_sha=substrate_sha,
        canonical_sha=canonical_sha,
        candidate_sha=candidate_sha,
        refit_baseline_sha=refit_baseline_sha,
        methodology=substrate_align_strategy,
    )


def _build_candidate_width_mismatch_verdict(
    *,
    candidate_width: int,
    expected_width: int,
    eval_slices: dict[str, EvalSlice],
    confound_threshold: float,
    substrate_sha: str,
    canonical_sha: str,
    candidate_sha: str,
    substrate_align_strategy: SubstrateAlignStrategy,
) -> SubstrateDriftSafeGateVerdict:
    """Construct ``confound_block`` verdict when CANDIDATE width != substrate width.

    Phase 75 METH-V27-02 Plan 75-04 — sibling to ``_build_width_mismatch_verdict``
    (which handles CANONICAL-vs-substrate mismatch, the Phase 64 v2.6.1 failure
    mode). This builder handles the inverse case introduced by the v2.7 dual-
    substrate methodology: Test 2 feeds a canonical-aligned (e.g., 13-wide)
    substrate into a wider candidate Pipeline. The candidate's inner
    ``PolynomialFeatures`` would crash; this verdict skips the candidate
    ``predict_proba`` entirely and surfaces both widths in the rationale.

    Skips BOTH the candidate predict (which would crash) AND the raw canonical
    predict (which works mechanically but is meaningless when the candidate
    can't even score) — populates ``aligned_candidate_brier_per_slice`` with
    None values along with the raw fields. The refit_baseline path still runs
    end-to-end on the supplied substrate so the aligned baseline numbers carry
    real signal for the audit trail (matching ``_build_width_mismatch_verdict``
    audit-trail discipline).

    Both rationale and confound_evidence carry the literal substring
    ``"width_mismatch"`` so the Plan 75-02 dual-substrate combinator routes
    the verdict to ``width_mismatch_dual`` (substring detection per Plan 75-02
    Decision #2).
    """
    # Refit baseline on the supplied substrate. The refit fits on whatever
    # width the substrate supplies, so no width error from this step.
    agg_features: list[tuple[float, ...]] = []
    agg_outcomes: list[int] = []
    for slice_name in sorted(eval_slices):
        sl = eval_slices[slice_name]
        agg_features.extend(sl.feature_vectors)
        agg_outcomes.extend(sl.outcomes)
    refit_baseline = _refit_baseline_on_substrate(
        tuple(agg_features),
        tuple(agg_outcomes),
    )
    refit_baseline_sha = _hash_refit_baseline(refit_baseline)

    raw_baseline_brier: dict[str, float] = {}
    raw_delta: dict[str, float] = {}
    aligned_baseline_brier: dict[str, float] = {}
    aligned_candidate_brier: dict[str, float] = {}
    aligned_delta: dict[str, float] = {}
    floor_clears: dict[str, bool] = {}

    for slice_name, sl in eval_slices.items():
        # Raw canonical predict is SKIPPED — sentinel None (matches
        # _build_width_mismatch_verdict discipline; raw measurement is
        # not meaningful when the candidate cannot score).
        raw_baseline_brier[slice_name] = None  # type: ignore[assignment]
        raw_delta[slice_name] = None  # type: ignore[assignment]

        aligned_baseline_preds = _predict_proba_with_pipeline(
            refit_baseline,
            sl.feature_vectors,
        )
        aligned_baseline_brier[slice_name] = _brier_score(
            aligned_baseline_preds,
            list(sl.outcomes),
        )

        # Candidate predict is SKIPPED — sentinel None per width-guard
        # invariant (the candidate's PolynomialFeatures would crash).
        aligned_candidate_brier[slice_name] = None  # type: ignore[assignment]
        aligned_delta[slice_name] = None  # type: ignore[assignment]
        # Width-mismatch confound -> no slice clears its floor (verdict is
        # forced to confound_block regardless; cosmetic for the formula-gate
        # dataclass fields).
        floor_clears[slice_name] = False

    confound_evidence = (
        f"width_mismatch_drift: candidate input width (via "
        f"_introspect_pipeline_width)={candidate_width} != substrate "
        f"feature_vector width={expected_width}. Candidate predict skipped; "
        f"raw canonical predict skipped; aligned baseline from refit on "
        f"substrate."
    )
    rationale = (
        f"confound_block: width_mismatch_drift on candidate side "
        f"(candidate={candidate_width}, substrate={expected_width}). "
        f"Phase 75 METH-V27-02 dual-substrate Test 2 guard: candidate "
        f"Pipeline cannot ingest the canonical-aligned substrate. See "
        f"confound_evidence."
    )
    if substrate_align_strategy == "dual_test_set":
        confound_evidence = (
            f"{confound_evidence} (dual_test_set strategy requested but "
            f"forced to refit_baseline due to candidate-width mismatch.)"
        )

    return SubstrateDriftSafeGateVerdict(
        aligned_baseline_brier_per_slice=aligned_baseline_brier,
        aligned_candidate_brier_per_slice=aligned_candidate_brier,
        aligned_delta_per_slice=aligned_delta,
        raw_baseline_brier_per_slice=raw_baseline_brier,
        raw_delta_per_slice=raw_delta,
        confound_detected=True,
        confound_threshold=confound_threshold,
        confound_evidence=confound_evidence,
        floor_clears=floor_clears,
        hurdle_clears=False,
        hurdle_value=0.0,
        verdict="confound_block",
        rationale=rationale,
        substrate_sha=substrate_sha,
        canonical_sha=canonical_sha,
        candidate_sha=candidate_sha,
        refit_baseline_sha=refit_baseline_sha,
        methodology=substrate_align_strategy,
    )


def _aggregate_substrate_sha(eval_slices: dict[str, EvalSlice]) -> str:
    """Stable SHA over per-slice substrate SHAs (sorted by slice name)."""
    h = hashlib.sha256()
    for slice_name in sorted(eval_slices):
        h.update(slice_name.encode("utf-8"))
        h.update(b"\0")
        h.update(eval_slices[slice_name].substrate_sha.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _hash_refit_baseline(refit_baseline: Any) -> str:
    """SHA of the refit baseline's serialized form (audit-trail invariant)."""
    import io

    import joblib

    buf = io.BytesIO()
    joblib.dump(refit_baseline, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


# ── JSON serialization for verdict artifacts ──────────────────────────────


def emit_verdict_json(
    verdict: SubstrateDriftSafeGateVerdict,
    out_path: Path,
) -> Path:
    """Write a verdict to `gate_verdict_v26_<candidate>.json`.

    Mirrors the Phase 45 Plan 45-04 immutable-output pattern: the JSON
    sidecar is byte-stable; if a human-readable corrigendum is needed
    later, it lives in a sibling .md file (NOT in this JSON).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(verdict.to_dict(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return out_path


# ── Phase 75 METH-V27-02 — dual-substrate methodology (v2.7) ─────────────
#
# Per ``.planning/gate_methodology_v2.7.md`` (Plan 75-03) §5 — dual-test
# substrate methodology. Wraps two single-substrate verifier calls + a
# combinator that triangulates "real lift" vs "substrate-drift artifact".
# Backward-compatible: the Phase 55 + Phase 64 single-substrate code
# path above is UNCHANGED by Plan 75-02 (ADD-ONLY contract; verifiable
# via `git diff` deletion count == 0).
# D-01 + D-03 LOCKED at Phase 75 CONTEXT 2026-06-07.


DUAL_VERIFIER_VERSION: str = "v2.7.0"

DualGateVerdict = Literal[
    "path_a_promote_dual",
    "substrate_drift_artifact",
    "highly_suspect",
    "path_b_reject_dual",
    "confound_block_dual",
    "width_mismatch_dual",
]


@dataclass(frozen=True)
class DualSubstrateGateVerdict:
    """Verdict structure per Phase 75 CONTEXT D-03 (LOCKED 2026-06-07).

    Wraps two ``SubstrateDriftSafeGateVerdict`` sub-verdicts (Test 1
    candidate-aligned substrate + Test 2 canonical-aligned substrate) and
    a combinator verdict literal that triangulates "real lift" vs
    "substrate-drift artifact" vs "highly suspect" vs "genuinely worse"
    vs "confound" vs "width-mismatch".

    Methodology source-of-truth: ``.planning/gate_methodology_v2.7.md``
    (Plan 75-03). Combinator logic source-of-truth: ``_combine_verdicts``.

    Backward compatibility invariant: this dataclass NEVER replaces
    ``SubstrateDriftSafeGateVerdict``; it wraps two instances of it. The
    Phase 55 single-substrate code path remains the source-of-truth for
    each sub-verdict's measurement; Plan 75-02 only adds a triangulation
    layer on top.
    """

    test_1_verdict: SubstrateDriftSafeGateVerdict
    test_2_verdict: SubstrateDriftSafeGateVerdict
    combined_verdict: DualGateVerdict
    rationale: str
    methodology: str = field(default="dual_substrate")
    verifier_version: str = field(default=DUAL_VERIFIER_VERSION)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form for ``gate_verdict_v27_<candidate>.json``.

        Nested sub-verdicts serialize via their own Phase 55 ``to_dict()``
        which is already deterministic (dict() copy + Phase 55 byte-stable
        ordering). Top-level key order is LOCKED per D-03:
        ``combined_verdict``, ``rationale``, ``methodology``,
        ``verifier_version``, ``test_1_verdict``, ``test_2_verdict``.
        """
        return {
            "combined_verdict": self.combined_verdict,
            "rationale": self.rationale,
            "methodology": self.methodology,
            "verifier_version": self.verifier_version,
            "test_1_verdict": self.test_1_verdict.to_dict(),
            "test_2_verdict": self.test_2_verdict.to_dict(),
        }


def _combine_verdicts(
    test_1: SubstrateDriftSafeGateVerdict,
    test_2: SubstrateDriftSafeGateVerdict,
) -> tuple[DualGateVerdict, str]:
    """Combinator per Phase 75 CONTEXT D-03 (LOCKED 2026-06-07).

    Precedence (highest to lowest):

      1. ``width_mismatch_dual`` — either test hit Phase 64 width-guard
         (``confound_evidence`` carries ``"width_mismatch"`` substring).
         Structurally unresolvable by substrate switching alone; requires
         either canonical retrain or candidate downsizing (v3.0+ scope).
      2. ``confound_block_dual`` — either test surfaced a non-width
         confound (Phase 55 raw-vs-aligned divergence > threshold).
         Methodology-level escalation required.
      3. The 4-cell normal verdict pair:

         | Test 1 | Test 2 | combined_verdict             |
         |--------|--------|------------------------------|
         | path_a | path_a | ``path_a_promote_dual``      |
         | path_a | path_b | ``substrate_drift_artifact`` |
         | path_b | path_a | ``highly_suspect``           |
         | path_b | path_b | ``path_b_reject_dual``       |

    Pure function — no I/O, no Pipeline loading. Returns
    ``(combined_verdict_literal, rationale_str)``. Inspects only
    ``test.verdict`` and ``test.confound_evidence`` of each input.
    """
    # Precedence 1: width-mismatch (Phase 64 case) — structurally
    # unresolvable by substrate switching alone. The Phase 64 width-guard
    # sets ``confound_evidence`` to a string containing
    # ``"width_mismatch_drift"`` (see ``_build_width_mismatch_verdict``).
    t1_is_width_mismatch = (
        test_1.verdict == "confound_block" and "width_mismatch" in test_1.confound_evidence
    )
    t2_is_width_mismatch = (
        test_2.verdict == "confound_block" and "width_mismatch" in test_2.confound_evidence
    )
    if t1_is_width_mismatch or t2_is_width_mismatch:
        which: list[str] = []
        if t1_is_width_mismatch:
            which.append("test_1 (candidate-aligned)")
        if t2_is_width_mismatch:
            which.append("test_2 (canonical-aligned)")
        return (
            "width_mismatch_dual",
            (
                f"width_mismatch_dual: {' + '.join(which)} hit the Phase 64 "
                f"width-guard. Width reconciliation requires either canonical "
                f"retrain (out of v2.7 scope) or candidate downsizing — "
                f"substrate switching alone cannot resolve. "
                f"test_1.verdict={test_1.verdict!r}, "
                f"test_2.verdict={test_2.verdict!r}."
            ),
        )

    # Precedence 2: non-width confound on either test. Width-mismatch is
    # already handled above, so any remaining confound_block is a generic
    # Phase 55 raw-vs-aligned divergence.
    if test_1.verdict == "confound_block" or test_2.verdict == "confound_block":
        confounding: list[str] = []
        if test_1.verdict == "confound_block":
            confounding.append(f"test_1 ({test_1.confound_evidence})")
        if test_2.verdict == "confound_block":
            confounding.append(f"test_2 ({test_2.confound_evidence})")
        return (
            "confound_block_dual",
            (
                f"confound_block_dual: substrate-drift confound detected "
                f"within {' + '.join(confounding)}. Methodology-level "
                f"escalation required — document evidence and consider "
                f"v2.8+ substrate-realignment (deferred from Phase 75)."
            ),
        )

    # Precedence 3: 4-cell normal verdict pair
    t1, t2 = test_1.verdict, test_2.verdict
    if t1 == "path_a_promote" and t2 == "path_a_promote":
        return (
            "path_a_promote_dual",
            (
                "path_a_promote_dual: clean PASS on BOTH candidate-aligned "
                "AND canonical-aligned substrates. Real lift triangulated; "
                "promotion approved per Phase 75 dual-substrate methodology."
            ),
        )
    if t1 == "path_a_promote" and t2 == "path_b_reject":
        return (
            "substrate_drift_artifact",
            (
                "substrate_drift_artifact: candidate wins on candidate-aligned "
                "substrate (Test 1 path_a) but loses on canonical-aligned "
                "substrate (Test 2 path_b). Lift is dominantly an OOF-source-"
                "divergence measurement artifact, NOT real candidate "
                "improvement. Rejected; documented as substrate drift."
            ),
        )
    if t1 == "path_b_reject" and t2 == "path_a_promote":
        return (
            "highly_suspect",
            (
                "highly_suspect: candidate clears canonical-aligned substrate "
                "(Test 2 path_a) but FAILS its own candidate-aligned substrate "
                "(Test 1 path_b). Surprising — investigate before any "
                "promotion. Likely candidate OOF artifact / training-pipeline "
                "regression. Rejected pending diagnostic."
            ),
        )
    # Remaining: (path_b, path_b)
    return (
        "path_b_reject_dual",
        (
            "path_b_reject_dual: candidate FAILS on BOTH candidate-aligned "
            "AND canonical-aligned substrates. Genuinely worse than canonical "
            "META-V22; no promotion. Documented as legitimate negative result."
        ),
    )


def verify_candidate_vs_canonical_dual_substrate(
    candidate: Path,
    canonical: Path,
    candidate_eval_slices: dict[str, EvalSlice],
    canonical_eval_slices: dict[str, EvalSlice],
    *,
    contract: GateContract | None = None,
    confound_threshold: float = 0.05,
) -> DualSubstrateGateVerdict:
    """Substrate-drift–immune dual-test gate verifier (Phase 75 METH-V27-02).

    Per ``.planning/gate_methodology_v2.7.md`` §5 (Plan 75-03). Runs the
    Phase 55 single-substrate verifier TWO TIMES with the same candidate/
    canonical pipelines but DIFFERENT substrates:

      Test 1: candidate-aligned substrate (col[0] = candidate's OOF; e.g.,
              ``xgb_v2_refv2_oof``). Same as v2.6.1 single-substrate
              measurement.

      Test 2: canonical-aligned substrate (col[0] = canonical training-time
              OOF; ``xgb_oof_prob`` from Phase 26
              ``oof_predictions_v22.parquet``). Built by
              ``scripts/build_canonical_substrate_v27.py`` (Plan 75-01).

    Both tests use ``refit_baseline`` strategy (D-01: ``shadow_traffic``
    structurally blocked by AUDIT-01; substrate-realignment deferred to
    v2.8+). The combinator (``_combine_verdicts``) triangulates the two
    sub-verdicts into one of six ``DualGateVerdict`` literals per D-03
    LOCKED table.

    Backward compatibility: this function is ADDITIVE. The Phase 55 +
    Phase 64 single-substrate ``verify_candidate_vs_canonical`` code path
    is UNCHANGED by Plan 75-02 (verifiable via
    ``git diff src/ufc_prediction/ml/gate_verifier.py`` deletion count
    == 0).

    Args:
        candidate: Path to candidate Pipeline (.joblib).
        canonical: Path to canonical Pipeline (e.g., ``meta_v2.joblib``).
        candidate_eval_slices: Test 1 substrate (candidate-aligned).
        canonical_eval_slices: Test 2 substrate (canonical-aligned; from
            Plan 75-01 builder).
        contract: Optional ``GateContract``; defaults to v2.6 contract
            (formula-gate thresholds are LOCKED at v2.6; methodology v2.7
            only changes the meta-gate measurement; D-18 LOCKED preserved).
        confound_threshold: passes through to BOTH sub-verifier calls.

    Returns:
        ``DualSubstrateGateVerdict`` per D-03.
    """
    test_1 = verify_candidate_vs_canonical(
        candidate=candidate,
        canonical=canonical,
        eval_slices=candidate_eval_slices,
        substrate_align_strategy="refit_baseline",
        contract=contract,
        confound_threshold=confound_threshold,
    )
    test_2 = verify_candidate_vs_canonical(
        candidate=candidate,
        canonical=canonical,
        eval_slices=canonical_eval_slices,
        substrate_align_strategy="refit_baseline",
        contract=contract,
        confound_threshold=confound_threshold,
    )
    combined_verdict, rationale = _combine_verdicts(test_1, test_2)
    return DualSubstrateGateVerdict(
        test_1_verdict=test_1,
        test_2_verdict=test_2,
        combined_verdict=combined_verdict,
        rationale=rationale,
    )


def emit_dual_verdict_json(
    verdict: DualSubstrateGateVerdict,
    out_path: Path,
) -> Path:
    """Write a dual-substrate verdict to ``gate_verdict_v27_<candidate>.json``.

    Byte-stable JSON contract mirroring Phase 55 ``emit_verdict_json``:
    ``json.dumps(verdict.to_dict(), indent=2, sort_keys=False) + "\\n"``.
    Phase 55 ``to_dict()`` is already deterministic, so nested sub-verdicts
    serialize byte-stably (verified by
    ``test_dual_substrate_verdict_to_dict_round_trips_json``).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(verdict.to_dict(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return out_path

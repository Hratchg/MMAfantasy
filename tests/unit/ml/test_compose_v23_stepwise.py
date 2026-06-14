"""Phase 32 Plan 32-01 Task 1 (Wave 0) — RED tests for per-step Brier hurdle.

These tests target `scripts/compose_v23_meta.py` (Task 2 deliverable).
The module is forked from `scripts/train_meta_v22.py::run_stepwise`
(lines 723-999) with META + CALIB steps prepended in front of REF + TRAVEL.

Wave 0 contract: tests must fail at IMPORT TIME (ModuleNotFoundError /
ImportError) until Task 2 creates the module and exports the required
public surface (`per_step_brier_clears`, `STEPWISE_HURDLE`,
`PER_SLICE_KEYS`).

Per D-13(v2.0) AF-10 binding: STEPWISE_HURDLE = 0.003 LOCKED; never tune
downward.

Per CONTEXT.md D-02 + D-03 binding: per-step ≥0.003 Brier hurdle is the
3rd leg of the triple-gate; ALL 3 slices must clear.
"""

from __future__ import annotations

import math

import pytest

# Wave 0 RED: this import MUST fail until Task 2 creates compose_v23_meta.
# Tests will not be skipped — they will error at import (pytest collects
# the failure, which is the RED signal).
from scripts.compose_v23_meta import (
    PER_SLICE_KEYS,
    STEPWISE_HURDLE,
    per_step_brier_clears,
)


# ───────────────────────────── Per-slice constants ─────────────────────────────

EXPECTED_PER_SLICE_KEYS: tuple[str, ...] = (
    "most_recent_12mo",
    "most_recent_24mo",
    "random_15pct",
)


def test_stepwise_hurdle_locked_at_0003() -> None:
    """D-13(v2.0) AF-10 binding — STEPWISE_HURDLE LOCKED at 0.003."""
    assert STEPWISE_HURDLE == 0.003


def test_per_slice_keys_match_canonical_tuple() -> None:
    """PER_SLICE_KEYS must match the 3-slice canonical convention."""
    assert tuple(PER_SLICE_KEYS) == EXPECTED_PER_SLICE_KEYS


# ───────────────────────── per_step_brier_clears (Δ ≥ 0.003) ────────────────────

def test_per_step_brier_hurdle_clears() -> None:
    """All 3 slices clear with Δ=0.004 (>= 0.003 hurdle) → (True, [])."""
    baseline = {slc: 0.180 for slc in PER_SLICE_KEYS}
    candidate = {slc: 0.176 for slc in PER_SLICE_KEYS}
    clears, failures = per_step_brier_clears(candidate, baseline)
    assert clears is True
    assert failures == []


def test_per_step_brier_hurdle_rejects_lt_003() -> None:
    """Candidate Δ=0.002 on random_15pct → rejects with failure listing slice + Δ."""
    baseline = {slc: 0.180 for slc in PER_SLICE_KEYS}
    candidate = {
        "most_recent_12mo": 0.176,  # Δ=0.004 ✓
        "most_recent_24mo": 0.176,  # Δ=0.004 ✓
        "random_15pct": 0.178,      # Δ=0.002 ✗
    }
    clears, failures = per_step_brier_clears(candidate, baseline)
    assert clears is False
    assert any("random_15pct" in f for f in failures)
    assert any("0.0020" in f for f in failures)
    assert any("0.003" in f for f in failures)


def test_per_step_brier_hurdle_rejects_one_slice_failure() -> None:
    """Clears 12mo + 24mo but misses random_15pct → rejects (D-13: ALL 3 slices)."""
    baseline = {slc: 0.180 for slc in PER_SLICE_KEYS}
    candidate = {
        "most_recent_12mo": 0.170,  # Δ=0.010 ✓
        "most_recent_24mo": 0.170,  # Δ=0.010 ✓
        "random_15pct": 0.180,      # Δ=0.000 ✗
    }
    clears, failures = per_step_brier_clears(candidate, baseline)
    assert clears is False
    assert len(failures) >= 1
    # Only the failing slice should appear
    assert any("random_15pct" in f for f in failures)
    assert not any("most_recent_12mo" in f for f in failures)


def test_per_step_brier_hurdle_exact_003_clears() -> None:
    """Δ exactly == 0.003 satisfies the >= hurdle."""
    baseline = {slc: 0.180 for slc in PER_SLICE_KEYS}
    candidate = {slc: 0.177 for slc in PER_SLICE_KEYS}  # Δ=0.003 exactly
    clears, failures = per_step_brier_clears(candidate, baseline)
    assert clears is True, f"Δ=0.003 should clear (>= hurdle), got failures={failures}"


def test_per_step_brier_hurdle_rejects_nan() -> None:
    """NaN candidate Brier on any slice → reject explicitly (Pitfall 6 binding)."""
    baseline = {slc: 0.180 for slc in PER_SLICE_KEYS}
    candidate = {
        "most_recent_12mo": 0.176,
        "most_recent_24mo": 0.176,
        "random_15pct": float("nan"),  # degenerate (TRAVEL 0-row case)
    }
    clears, failures = per_step_brier_clears(candidate, baseline)
    assert clears is False
    assert any("nan" in f.lower() or "NaN" in f for f in failures)


def test_per_step_brier_hurdle_negative_delta() -> None:
    """Candidate WORSE than baseline → rejects (no degradation allowed)."""
    baseline = {slc: 0.180 for slc in PER_SLICE_KEYS}
    candidate = {slc: 0.190 for slc in PER_SLICE_KEYS}  # Δ=-0.010 (worse)
    clears, failures = per_step_brier_clears(candidate, baseline)
    assert clears is False
    assert len(failures) == 3  # all 3 slices fail


# ──────────────────────── Forward-stepwise pruning discipline ───────────────────

def test_rejected_step_baseline_unchanged() -> None:
    """If step N is rejected, step N+1 composes on PRIOR baseline (not rejected step).

    This is the canonical forward-stepwise pruning discipline per D-13(v2.0).
    """
    prior_baseline = {slc: 0.180 for slc in PER_SLICE_KEYS}
    # Step N rejected (Δ=0.001 < 0.003)
    step_n_brier = {slc: 0.179 for slc in PER_SLICE_KEYS}
    clears_n, _ = per_step_brier_clears(step_n_brier, prior_baseline)
    assert clears_n is False

    # Step N+1 must compose on prior_baseline, NOT step_n_brier.
    # The driver picks: next_baseline = step_n_brier if clears_n else prior_baseline.
    next_baseline = step_n_brier if clears_n else prior_baseline
    assert next_baseline == prior_baseline

    # Confirm a Δ=0.004 step N+1 over prior_baseline clears but would NOT
    # clear over step_n_brier (Δ=0.179 - 0.176 = 0.003 exactly — boundary
    # case demonstrating discipline matters).
    step_n_plus_1_brier = {slc: 0.176 for slc in PER_SLICE_KEYS}
    clears_n_plus_1, _ = per_step_brier_clears(step_n_plus_1_brier, next_baseline)
    assert clears_n_plus_1 is True

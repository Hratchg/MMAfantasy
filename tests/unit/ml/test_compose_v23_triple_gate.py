"""Phase 32 Plan 32-01 Task 1 (Wave 0) — RED tests for triple-gate AND-logic.

These tests target `scripts/compose_v23_meta.py::triple_gate_decision`
(Task 2 deliverable).

Per CONTEXT.md D-03 (operator-locked) — triple-gate hard-AND for
meta_v3 promotion:
  1. Gate clearance: candidate Brier <= per_slice.brier_max AND
     candidate accuracy >= per_slice.accuracy_min on all 3 slices
     (gate_contract_v2.3.json)
  2. Total-margin hurdle: candidate Brier <= META-V22 Brier - 0.003
     on all 3 slices
  3. Per-step hurdle: each composition step (META→CALIB, CALIB→REF,
     REF→TRAVEL) clears >= 0.003 Brier improvement over prior step.

Verdict surface:
  triple_gate_decision(...) -> tuple[str, str, list[str]]
  where verdict in {"promote", "reject"} and path in
  {"path_a", "path_b", "path_c"}.

  Path A: all 3 legs clear → promote
  Path B: SOME steps cleared but surviving candidate misses gate → reject
  Path C: total failure (gate-miss OR margin-miss OR per-step-miss
          OR NaN) → reject
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from scripts.compose_v23_meta import (
    META_V22_BASELINE_BRIER,
    PER_SLICE_KEYS,
    TOTAL_MARGIN_HURDLE,
    triple_gate_decision,
)


# ───────────────────────────── Mock gate contract ─────────────────────────────


@dataclass(frozen=True)
class _MockSliceThresholds:
    brier_max: float
    accuracy_min: float


@dataclass(frozen=True)
class _MockContract:
    per_slice: dict[str, _MockSliceThresholds]


def _make_contract_v23() -> _MockContract:
    """Minimal mock of GateContract honoring gate_contract_v2.3.json thresholds."""
    return _MockContract(
        per_slice={
            "most_recent_12mo": _MockSliceThresholds(brier_max=0.1512, accuracy_min=0.7814),
            "most_recent_24mo": _MockSliceThresholds(brier_max=0.1583, accuracy_min=0.7627),
            "random_15pct": _MockSliceThresholds(brier_max=0.1374, accuracy_min=0.7812),
        },
    )


def _all_steps_clear() -> dict[str, bool]:
    return {
        "meta_to_calib": True,
        "calib_to_ref": True,
        "ref_to_travel": True,
    }


def _median_clearly_passes() -> dict[str, dict[str, float]]:
    """Per-slice median dict that clears all v2.3 thresholds by wide margin."""
    return {
        "most_recent_12mo": {"brier_score": 0.140, "accuracy": 0.79},
        "most_recent_24mo": {"brier_score": 0.150, "accuracy": 0.78},
        "random_15pct": {"brier_score": 0.130, "accuracy": 0.80},
    }


def _final_brier_beats_meta_v22() -> dict[str, float]:
    """Final candidate Brier per slice, beats META-V22 by >0.003 on all 3."""
    return {slc: META_V22_BASELINE_BRIER[slc] - 0.010 for slc in PER_SLICE_KEYS}


# ───────────────────────────── Total-margin constant ─────────────────────────


def test_total_margin_hurdle_locked_at_0003() -> None:
    """TOTAL_MARGIN_HURDLE LOCKED at 0.003 per D-03 leg 2."""
    assert TOTAL_MARGIN_HURDLE == 0.003


def test_meta_v22_baseline_per_slice_present() -> None:
    """META_V22_BASELINE_BRIER must carry all 3 slice keys."""
    for slc in PER_SLICE_KEYS:
        assert slc in META_V22_BASELINE_BRIER
        assert isinstance(META_V22_BASELINE_BRIER[slc], float)
    # Phase 73 (DEBT-V261-01): values reconciled with current META_V22_SPIKE.json::median_per_slice.
    # The stale 0.213/0.213/0.187 values were Phase 26 OOF; current file is v2.6 close baseline.
    assert abs(META_V22_BASELINE_BRIER["most_recent_12mo"] - 0.15465991222123496) < 1e-9
    assert abs(META_V22_BASELINE_BRIER["most_recent_24mo"] - 0.15935635076986582) < 1e-9
    assert abs(META_V22_BASELINE_BRIER["random_15pct"] - 0.14167626892373050) < 1e-9


# ───────────────────────────── Path A (all 3 legs clear) ─────────────────────


def test_triple_gate_all_three_clear_promotes() -> None:
    """All 3 legs clear → ('promote', 'path_a', [])."""
    contract = _make_contract_v23()
    verdict, path, failures = triple_gate_decision(
        final_candidate_brier=_final_brier_beats_meta_v22(),
        prior_step_clears=_all_steps_clear(),
        contract=contract,
        median_final_per_slice=_median_clearly_passes(),
    )
    assert verdict == "promote"
    assert path == "path_a"
    assert failures == []


# ───────────────────────────── Path C (any-leg-miss) ─────────────────────────


def test_triple_gate_rejects_on_gate_miss() -> None:
    """Gate-miss (Brier > brier_max on a slice) → ('reject', 'path_c', [<reason>])."""
    contract = _make_contract_v23()
    median_bad = _median_clearly_passes()
    median_bad["most_recent_12mo"] = {"brier_score": 0.200, "accuracy": 0.79}  # Brier > 0.1512
    verdict, path, failures = triple_gate_decision(
        final_candidate_brier=_final_brier_beats_meta_v22(),
        prior_step_clears=_all_steps_clear(),
        contract=contract,
        median_final_per_slice=median_bad,
    )
    assert verdict == "reject"
    assert path == "path_c"
    assert len(failures) >= 1
    assert any("brier" in f.lower() or "most_recent_12mo" in f for f in failures)


def test_triple_gate_rejects_on_margin_miss() -> None:
    """Total-margin-miss (Δ < 0.003 vs META-V22) → ('reject', 'path_c', [<reason>])."""
    contract = _make_contract_v23()
    median_passes_gate = _median_clearly_passes()
    # Brier that beats gate but only beats META-V22 by 0.001 (< 0.003 hurdle).
    final_brier_thin = {slc: META_V22_BASELINE_BRIER[slc] - 0.001 for slc in PER_SLICE_KEYS}
    verdict, path, failures = triple_gate_decision(
        final_candidate_brier=final_brier_thin,
        prior_step_clears=_all_steps_clear(),
        contract=contract,
        median_final_per_slice=median_passes_gate,
    )
    assert verdict == "reject"
    assert path == "path_c"
    assert any("margin" in f.lower() or "META-V22" in f or "0.003" in f for f in failures)


def test_triple_gate_rejects_on_per_step_miss_all_zero() -> None:
    """Per-step-miss with NO steps cleared → ('reject', 'path_c', [...]).

    When all prior transitions failed (zero steps cleared), the verdict
    is Path C — total failure, no partial composition signal.
    """
    contract = _make_contract_v23()
    step_clears = {
        "meta_to_calib": False,
        "calib_to_ref": False,
        "ref_to_travel": False,
    }
    verdict, path, failures = triple_gate_decision(
        final_candidate_brier=_final_brier_beats_meta_v22(),
        prior_step_clears=step_clears,
        contract=contract,
        median_final_per_slice=_median_clearly_passes(),
    )
    assert verdict == "reject"
    assert path == "path_c"
    assert any("per-step" in f.lower() or "transition" in f.lower() for f in failures)


def test_triple_gate_partial_step_miss_is_path_b() -> None:
    """SOME (not all) steps cleared + gate+margin pass + per-step fails → Path B.

    This is the canonical D-07 Path B signature: "some steps clear,
    some don't; surviving META + CALIB candidate clears gate but
    REF/TRAVEL don't add value". The candidate would PASS gate +
    total-margin but the per-step ALL-must-clear discipline (D-13)
    blocks promotion.
    """
    contract = _make_contract_v23()
    step_clears = _all_steps_clear()
    step_clears["ref_to_travel"] = False  # only last transition failed
    verdict, path, failures = triple_gate_decision(
        final_candidate_brier=_final_brier_beats_meta_v22(),
        prior_step_clears=step_clears,
        contract=contract,
        median_final_per_slice=_median_clearly_passes(),
    )
    assert verdict == "reject"
    assert path == "path_b"  # partial composition
    assert any("ref_to_travel" in f or "per-step" in f.lower() for f in failures)


def test_triple_gate_rejects_nan() -> None:
    """NaN final Brier on any slice → reject explicitly (Pitfall 6 binding)."""
    contract = _make_contract_v23()
    final_brier_nan = _final_brier_beats_meta_v22()
    final_brier_nan["random_15pct"] = float("nan")
    median_bad = _median_clearly_passes()
    median_bad["random_15pct"] = {"brier_score": float("nan"), "accuracy": 0.80}
    verdict, path, failures = triple_gate_decision(
        final_candidate_brier=final_brier_nan,
        prior_step_clears=_all_steps_clear(),
        contract=contract,
        median_final_per_slice=median_bad,
    )
    assert verdict == "reject"
    assert path == "path_c"
    assert any("nan" in f.lower() or "NaN" in f for f in failures)


# ───────────────────────────── Path B (partial composition) ───────────────────


def test_triple_gate_path_b_distinction() -> None:
    """Some-but-not-all-steps cleared + surviving candidate misses gate → path_b.

    Distinguishes partial composition (Path B — META + CALIB cleared, REF or
    TRAVEL did not add value) from total failure (Path C — gate-miss or
    margin-miss or all-step-miss).
    """
    contract = _make_contract_v23()
    # META→CALIB cleared; CALIB→REF rejected (step pruning kicked in);
    # REF→TRAVEL also rejected → some steps cleared but not all.
    step_clears = {
        "meta_to_calib": True,
        "calib_to_ref": False,
        "ref_to_travel": False,
    }
    # Surviving candidate (just META + CALIB) misses gate on 24mo slice.
    median_partial = _median_clearly_passes()
    median_partial["most_recent_24mo"] = {"brier_score": 0.200, "accuracy": 0.79}
    verdict, path, failures = triple_gate_decision(
        final_candidate_brier=_final_brier_beats_meta_v22(),
        prior_step_clears=step_clears,
        contract=contract,
        median_final_per_slice=median_partial,
    )
    # Partial composition with gate miss → Path B distinguishes from Path C
    assert verdict == "reject"
    assert path == "path_b"


# ───────────────────────────── Parametrized any-leg rejection ─────────────────


@pytest.mark.parametrize(
    "fail_leg",
    ["gate", "margin", "per_step", "nan"],
)
def test_triple_gate_rejects_any_leg_miss(fail_leg: str) -> None:
    """ANY-leg failure forces non-promotion (D-03 hard-AND).

    Parametrized rejection of each leg in isolation (gate-miss, margin-miss,
    per-step-miss, NaN). The verdict must be `reject` regardless of which
    leg fails.
    """
    contract = _make_contract_v23()
    final_brier = _final_brier_beats_meta_v22()
    step_clears = _all_steps_clear()
    median = _median_clearly_passes()

    if fail_leg == "gate":
        median["most_recent_12mo"]["brier_score"] = 0.200  # > 0.1512
    elif fail_leg == "margin":
        final_brier = {slc: META_V22_BASELINE_BRIER[slc] - 0.001 for slc in PER_SLICE_KEYS}
    elif fail_leg == "per_step":
        # With per_step miss + gate-pass + margin-pass → Path B (partial
        # composition). To force Path C on per-step miss, ALL transitions
        # must fail (no partial-composition signal). Use a single
        # transition fail here — verdict still 'reject' (the parametrized
        # contract).
        step_clears["calib_to_ref"] = False
    elif fail_leg == "nan":
        final_brier["random_15pct"] = float("nan")
        median["random_15pct"]["brier_score"] = float("nan")

    verdict, path, failures = triple_gate_decision(
        final_candidate_brier=final_brier,
        prior_step_clears=step_clears,
        contract=contract,
        median_final_per_slice=median,
    )
    assert verdict == "reject"
    assert path in {"path_b", "path_c"}
    assert len(failures) >= 1

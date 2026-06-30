"""Phase 42 Plan 42-02 Task 1 (Wave 1) — RED tests for compose_v25_travel.

Targets ``scripts/compose_v25_travel.py`` (Task 2 deliverable). Forks
``scripts/compose_v23_meta.py`` (READ-ONLY per AUDIT-01 fork-not-mutate):
baseline = META-V22 + CALIB (v2.3 canonical Path A stack); candidate =
META-V22 + CALIB + TRAVEL-V25 (2 new Level-1 cols ``travel_distance_km`` +
``tz_shift_hours`` from Plan 42-01).

D-18 binding gate (CONTEXT-locked; formula_hash
``7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a``):
  - **Floor**: candidate Brier ≤ baseline Brier on ALL 3 slices AND
    candidate accuracy ≥ 0.70 on ALL 3 slices.
  - **Hurdle**: candidate Brier improvement ≥ 0.003 over baseline on
    MAJORITY (≥2/3) of slices.

NB: the 0.70 accuracy floor is the COARSE CONTEXT D-18 lock and is
DELIBERATELY looser than the per-slice gate_contract_v2.3.json
accuracy_min values (0.7814 / 0.7627 / 0.7812). Per CONTEXT
§Recomposition the locked gate is 0.70 verbatim — Plan 42-02 tests THIS
threshold; the report ALSO carries the gate_contract per-slice numbers
for operator transparency (consumed by Plan 42-03).

Wave 1 contract: tests MUST fail at IMPORT TIME (ModuleNotFoundError /
ImportError) until Task 2 creates ``scripts/compose_v25_travel.py`` and
exports ``floor_clears``, ``hurdle_clears``, ``determine_path``,
``build_report`` plus the 4 LOCKED constants
(``FLOOR_ACCURACY_MIN``, ``HURDLE_BRIER_MIN``,
``HURDLE_MAJORITY_THRESHOLD``, ``PER_SLICE_KEYS``).
"""

from __future__ import annotations

import json
import math

import pytest

# Wave 1 RED: this import MUST fail until Task 2 creates compose_v25_travel.
# Tests will not be skipped — pytest collects the ImportError and reports
# RED at collection time.
from scripts.compose_v25_travel import (
    FLOOR_ACCURACY_MIN,
    FORMULA_HASH,
    HURDLE_BRIER_MIN,
    HURDLE_MAJORITY_THRESHOLD,
    PER_SLICE_KEYS,
    SUBSTRATE_VERSION,
    build_report,
    determine_path,
    floor_clears,
    hurdle_clears,
)


# ────────────────── Locked-constant invariants ───────────────────────


def test_floor_accuracy_min_locked_at_070() -> None:
    """CONTEXT D-18 floor accuracy threshold LOCKED at 0.70 verbatim."""
    assert FLOOR_ACCURACY_MIN == 0.70


def test_hurdle_brier_min_locked_at_0003() -> None:
    """CONTEXT D-18 + D-13(v2.0) Brier hurdle LOCKED at 0.003."""
    assert HURDLE_BRIER_MIN == 0.003


def test_hurdle_majority_threshold_locked_at_2_of_3() -> None:
    """CONTEXT D-18 majority threshold LOCKED at 2 (≥2 of 3 slices)."""
    assert HURDLE_MAJORITY_THRESHOLD == 2


def test_per_slice_keys_match_canonical_tuple() -> None:
    """PER_SLICE_KEYS must match the 3-slice canonical convention."""
    assert tuple(PER_SLICE_KEYS) == (
        "most_recent_12mo",
        "most_recent_24mo",
        "random_15pct",
    )


def test_substrate_version_is_v25() -> None:
    """SUBSTRATE_VERSION = 'v2.5' (Phase 42 + Plan 42-01 TRAVEL primitives)."""
    assert SUBSTRATE_VERSION == "v2.5"


def test_formula_hash_d18_locked() -> None:
    """D-18 formula hash LOCKED — NO post-measurement renegotiation."""
    assert FORMULA_HASH == "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"


# ───────────────────────── TestFloorClears ───────────────────────────


class TestFloorClears:
    """D-18 Floor: candidate Brier ≤ baseline AND candidate acc ≥ 0.70 on ALL slices."""

    def test_floor_clears_all_three_slices(self) -> None:
        """Symmetric Brier improvement + acc ≥ 0.70 on all slices → (True, [])."""
        baseline = {slc: 0.180 for slc in PER_SLICE_KEYS}
        candidate_brier = {slc: 0.176 for slc in PER_SLICE_KEYS}
        candidate_accuracy = {
            "most_recent_12mo": 0.72,
            "most_recent_24mo": 0.71,
            "random_15pct": 0.72,
        }
        ok, failures = floor_clears(baseline, candidate_brier, candidate_accuracy)
        assert ok is True
        assert failures == []

    def test_floor_fails_on_brier_regression_any_slice(self) -> None:
        """Candidate Brier WORSE on ANY slice → (False, [slice msg])."""
        baseline = {slc: 0.180 for slc in PER_SLICE_KEYS}
        candidate_brier = {
            "most_recent_12mo": 0.176,
            "most_recent_24mo": 0.176,
            "random_15pct": 0.181,  # regression
        }
        candidate_accuracy = {slc: 0.72 for slc in PER_SLICE_KEYS}
        ok, failures = floor_clears(baseline, candidate_brier, candidate_accuracy)
        assert ok is False
        assert any("random_15pct" in f and "brier" in f for f in failures)
        assert any("0.1810" in f and "0.1800" in f for f in failures)

    def test_floor_fails_on_accuracy_below_70(self) -> None:
        """Candidate acc < 0.70 on ANY slice → (False, [slice msg])."""
        baseline = {slc: 0.180 for slc in PER_SLICE_KEYS}
        candidate_brier = {slc: 0.176 for slc in PER_SLICE_KEYS}
        candidate_accuracy = {
            "most_recent_12mo": 0.72,
            "most_recent_24mo": 0.69,  # < 0.70
            "random_15pct": 0.72,
        }
        ok, failures = floor_clears(baseline, candidate_brier, candidate_accuracy)
        assert ok is False
        assert any("most_recent_24mo" in f and "accuracy" in f and "0.6900" in f for f in failures)
        assert any("0.70" in f for f in failures)

    def test_floor_fails_on_both_brier_regression_AND_low_acc(self) -> None:
        """Both Brier regression AND low acc → both failure msgs present."""
        baseline = {slc: 0.180 for slc in PER_SLICE_KEYS}
        candidate_brier = {
            "most_recent_12mo": 0.176,
            "most_recent_24mo": 0.176,
            "random_15pct": 0.181,  # regression
        }
        candidate_accuracy = {
            "most_recent_12mo": 0.72,
            "most_recent_24mo": 0.65,  # low
            "random_15pct": 0.72,
        }
        ok, failures = floor_clears(baseline, candidate_brier, candidate_accuracy)
        assert ok is False
        # Both kinds of failures should be present.
        assert any("brier" in f for f in failures)
        assert any("accuracy" in f for f in failures)
        # Specifically the two failing slices.
        assert any("random_15pct" in f and "brier" in f for f in failures)
        assert any("most_recent_24mo" in f and "accuracy" in f for f in failures)

    def test_floor_equality_at_baseline_is_clear(self) -> None:
        """Candidate Brier == baseline (delta = 0.0) passes the floor (≤)."""
        baseline = {slc: 0.180 for slc in PER_SLICE_KEYS}
        candidate_brier = {slc: 0.180 for slc in PER_SLICE_KEYS}
        candidate_accuracy = {slc: 0.70 for slc in PER_SLICE_KEYS}  # exactly 0.70
        ok, failures = floor_clears(baseline, candidate_brier, candidate_accuracy)
        assert ok is True
        assert failures == []


# ───────────────────────── TestHurdleClears ───────────────────────────


class TestHurdleClears:
    """D-18 Hurdle: ≥0.003 Brier improvement on ≥2/3 slices."""

    def test_hurdle_clears_all_three_slices(self) -> None:
        """All 3 slices clear by Δ=0.005 → majority trivially clears."""
        baseline_brier = {slc: 0.180 for slc in PER_SLICE_KEYS}
        candidate_brier = {slc: 0.175 for slc in PER_SLICE_KEYS}
        ok, msgs = hurdle_clears(baseline_brier, candidate_brier)
        assert ok is True
        # Per-slice messages should be present (3 entries).
        assert len(msgs) == 3

    def test_hurdle_clears_majority_two_of_three(self) -> None:
        """2/3 clear; 1/3 fails → majority (≥2) clears → (True, ...)."""
        baseline_brier = {slc: 0.180 for slc in PER_SLICE_KEYS}
        candidate_brier = {
            "most_recent_12mo": 0.175,  # Δ=0.005 ✓
            "most_recent_24mo": 0.176,  # Δ=0.004 ✓
            "random_15pct": 0.179,  # Δ=0.001 ✗
        }
        ok, msgs = hurdle_clears(baseline_brier, candidate_brier)
        assert ok is True

    def test_hurdle_fails_only_one_clears(self) -> None:
        """1/3 clears; 2/3 fail → majority FAILS → (False, ...)."""
        baseline_brier = {slc: 0.180 for slc in PER_SLICE_KEYS}
        candidate_brier = {
            "most_recent_12mo": 0.175,  # Δ=0.005 ✓
            "most_recent_24mo": 0.179,  # Δ=0.001 ✗
            "random_15pct": 0.179,  # Δ=0.001 ✗
        }
        ok, msgs = hurdle_clears(baseline_brier, candidate_brier)
        assert ok is False

    def test_hurdle_fails_all_below(self) -> None:
        """All 3 below hurdle → (False, ...)."""
        baseline_brier = {slc: 0.180 for slc in PER_SLICE_KEYS}
        candidate_brier = {slc: 0.179 for slc in PER_SLICE_KEYS}  # Δ=0.001 each
        ok, msgs = hurdle_clears(baseline_brier, candidate_brier)
        assert ok is False

    def test_hurdle_exact_003_counts_as_clear(self) -> None:
        """Δ = 0.003 (exact equality) counts as clearing (≥, not >)."""
        baseline_brier = {slc: 0.180 for slc in PER_SLICE_KEYS}
        candidate_brier = {slc: 0.177 for slc in PER_SLICE_KEYS}  # Δ=0.003
        ok, msgs = hurdle_clears(baseline_brier, candidate_brier)
        assert ok is True

    def test_hurdle_rejects_nan_delta(self) -> None:
        """NaN candidate Brier → that slice does NOT count as cleared."""
        baseline_brier = {slc: 0.180 for slc in PER_SLICE_KEYS}
        candidate_brier = {
            "most_recent_12mo": float("nan"),
            "most_recent_24mo": float("nan"),
            "random_15pct": 0.175,  # Δ=0.005 ✓
        }
        # Only 1 slice clears → majority fails.
        ok, msgs = hurdle_clears(baseline_brier, candidate_brier)
        assert ok is False

    def test_hurdle_nan_baseline_also_fails_that_slice(self) -> None:
        """NaN baseline → that slice does NOT count toward majority."""
        baseline_brier = {
            "most_recent_12mo": float("nan"),
            "most_recent_24mo": 0.180,
            "random_15pct": 0.180,
        }
        candidate_brier = {slc: 0.175 for slc in PER_SLICE_KEYS}  # Δ=0.005 each (if non-NaN)
        ok, msgs = hurdle_clears(baseline_brier, candidate_brier)
        # 2 valid slices both clear → majority passes.
        assert ok is True

    def test_hurdle_nan_baseline_breaks_majority(self) -> None:
        """NaN baseline on 2 slices + only 1 clearing slice → majority FAILS."""
        baseline_brier = {
            "most_recent_12mo": float("nan"),
            "most_recent_24mo": float("nan"),
            "random_15pct": 0.180,
        }
        candidate_brier = {slc: 0.175 for slc in PER_SLICE_KEYS}
        ok, msgs = hurdle_clears(baseline_brier, candidate_brier)
        assert ok is False


# ─────────────────────── TestPathDetermination ───────────────────────


class TestPathDetermination:
    """Path A eligibility vs Path B inevitability (mutually exclusive)."""

    def test_path_a_eligible_when_floor_AND_hurdle_both_clear(self) -> None:
        """Both gates clear → path_a_eligible=True, path_b_inevitable=False."""
        floor_result = (True, [])
        hurdle_result = (True, ["msg1", "msg2", "msg3"])
        path = determine_path(floor_result, hurdle_result)
        assert path["path_a_eligible"] is True
        assert path["path_b_inevitable"] is False

    def test_path_b_inevitable_when_floor_fails(self) -> None:
        """Floor fails (any reason) → path_b_inevitable=True."""
        floor_result = (False, ["random_15pct: brier 0.1810 > baseline 0.1800"])
        hurdle_result = (True, ["msg1", "msg2", "msg3"])
        path = determine_path(floor_result, hurdle_result)
        assert path["path_a_eligible"] is False
        assert path["path_b_inevitable"] is True

    def test_path_b_inevitable_when_hurdle_fails(self) -> None:
        """Floor clears but hurdle fails → path_b_inevitable=True."""
        floor_result = (True, [])
        hurdle_result = (False, ["msg1", "msg2", "msg3"])
        path = determine_path(floor_result, hurdle_result)
        assert path["path_a_eligible"] is False
        assert path["path_b_inevitable"] is True

    def test_path_b_inevitable_when_both_fail(self) -> None:
        """Both fail → path_b_inevitable=True."""
        floor_result = (False, ["x"])
        hurdle_result = (False, ["y"])
        path = determine_path(floor_result, hurdle_result)
        assert path["path_a_eligible"] is False
        assert path["path_b_inevitable"] is True

    @pytest.mark.parametrize(
        "floor_pass,hurdle_pass",
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_both_eligible_and_inevitable_are_mutually_exclusive(
        self,
        floor_pass: bool,
        hurdle_pass: bool,
    ) -> None:
        """For ANY input: XOR(path_a_eligible, path_b_inevitable) == True."""
        floor_result = (floor_pass, [] if floor_pass else ["x"])
        hurdle_result = (hurdle_pass, [] if hurdle_pass else ["y"])
        path = determine_path(floor_result, hurdle_result)
        assert bool(path["path_a_eligible"]) != bool(path["path_b_inevitable"]), (
            f"XOR broken: {path}"
        )


# ─────────────────────────── TestReportShape ──────────────────────────


def _synthetic_baseline_per_slice() -> dict[str, dict[str, float]]:
    return {
        "most_recent_12mo": {"brier_score": 0.180, "accuracy": 0.72, "n": 459},
        "most_recent_24mo": {"brier_score": 0.180, "accuracy": 0.71, "n": 967},
        "random_15pct": {"brier_score": 0.180, "accuracy": 0.72, "n": 160},
    }


def _synthetic_candidate_per_slice() -> dict[str, dict[str, float]]:
    return {
        "most_recent_12mo": {"brier_score": 0.175, "accuracy": 0.73, "n": 459},
        "most_recent_24mo": {"brier_score": 0.176, "accuracy": 0.72, "n": 967},
        "random_15pct": {"brier_score": 0.177, "accuracy": 0.72, "n": 160},
    }


class TestReportShape:
    """build_report produces the report shape Plan 42-03 consumes."""

    def test_report_shape_complete(self) -> None:
        """All required top-level keys present + per-slice has 3 entries."""
        report = build_report(
            baseline=_synthetic_baseline_per_slice(),
            candidate=_synthetic_candidate_per_slice(),
            gate_contract_ref=".planning/gate_contract_v2.3.json",
            xgb_sha="6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099",
            meta_sha="77076d3b2eed79797c355195f0f76156582b4c2f9b16df923c06ae2c855f9196",
        )
        # Required top-level keys.
        for key in (
            "baseline",
            "candidate",
            "per_slice",
            "floor_clears_all_three",
            "hurdle_clears_majority",
            "path_a_eligible",
            "path_b_inevitable",
            "gate_contract_ref",
            "formula_hash",
            "produced_at",
            "substrate_version",
            "xgb_v2_sha256",
            "meta_v2_sha256",
        ):
            assert key in report, f"missing report key: {key!r}"
        # per_slice has all 3 entries.
        assert set(report["per_slice"].keys()) == set(PER_SLICE_KEYS)
        # Each per-slice entry carries brier + accuracy + delta_brier +
        # floor_clears + hurdle_clears.
        for slc in PER_SLICE_KEYS:
            entry = report["per_slice"][slc]
            for k in (
                "baseline_brier",
                "candidate_brier",
                "baseline_accuracy",
                "candidate_accuracy",
                "delta_brier",
                "floor_clears",
                "hurdle_clears",
            ):
                assert k in entry, f"per_slice[{slc!r}] missing key {k!r}"

    def test_report_serializable_json(self) -> None:
        """Report dict round-trips through json.dumps without coercion errors."""
        report = build_report(
            baseline=_synthetic_baseline_per_slice(),
            candidate=_synthetic_candidate_per_slice(),
            gate_contract_ref=".planning/gate_contract_v2.3.json",
            xgb_sha="abc" * 21 + "a",
            meta_sha="def" * 21 + "a",
        )
        s = json.dumps(report)
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["substrate_version"] == "v2.5"
        # Booleans round-trip correctly (not numpy.bool_).
        assert isinstance(parsed["floor_clears_all_three"], bool)
        assert isinstance(parsed["hurdle_clears_majority"], bool)
        assert isinstance(parsed["path_a_eligible"], bool)
        assert isinstance(parsed["path_b_inevitable"], bool)

    def test_report_path_a_eligible_xor_inevitable(self) -> None:
        """For any baseline/candidate: XOR(path_a_eligible, path_b_inevitable)."""
        # Use baseline-vs-candidate that clears both gates.
        report = build_report(
            baseline=_synthetic_baseline_per_slice(),
            candidate=_synthetic_candidate_per_slice(),
            gate_contract_ref=".planning/gate_contract_v2.3.json",
            xgb_sha="x",
            meta_sha="y",
        )
        assert bool(report["path_a_eligible"]) != bool(report["path_b_inevitable"])

    def test_report_delta_brier_is_signed(self) -> None:
        """delta_brier = baseline - candidate (positive = candidate improves)."""
        baseline = _synthetic_baseline_per_slice()
        candidate = _synthetic_candidate_per_slice()
        report = build_report(
            baseline=baseline,
            candidate=candidate,
            gate_contract_ref=".planning/gate_contract_v2.3.json",
            xgb_sha="x",
            meta_sha="y",
        )
        for slc in PER_SLICE_KEYS:
            expected = baseline[slc]["brier_score"] - candidate[slc]["brier_score"]
            actual = report["per_slice"][slc]["delta_brier"]
            assert math.isclose(actual, expected, abs_tol=1e-12), (
                f"delta_brier mismatch on {slc}: actual={actual} expected={expected}"
            )

    def test_report_formula_hash_d18_locked(self) -> None:
        """report.formula_hash matches D-18 locked hash."""
        report = build_report(
            baseline=_synthetic_baseline_per_slice(),
            candidate=_synthetic_candidate_per_slice(),
            gate_contract_ref=".planning/gate_contract_v2.3.json",
            xgb_sha="x",
            meta_sha="y",
        )
        assert (
            report["formula_hash"]
            == "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"
        )

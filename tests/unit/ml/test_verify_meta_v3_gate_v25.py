"""Plan 45-04 — D-18 LOCKED gate decision function tests (META3-V25-03).

Tests cover the 4 binding behaviors for ``scripts/verify_meta_v3_gate_v25.py``:

1. ``floor_clears_all_three`` — True iff candidate Brier ≤ baseline Brier on ALL
   3 slices AND candidate accuracy ≥ 0.70 on ALL 3 slices.
2. ``hurdle_clears_majority`` — True iff candidate Brier improvement ≥ 0.003 on
   ≥ majority (≥2/3) of slices.
3. ``path_determination`` — returns "path_a" iff floor AND hurdle; "path_b"
   otherwise. Ensures Path A XOR Path B invariant.
4. ``gate_formula_hash`` — returns the D-18 LOCKED formula hash verbatim
   ``7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a``;
   sentinel test guards against post-measurement renegotiation.

Tests use pure-functional synthetic per_slice dicts — no model load, no DB,
no sklearn dependency. Fast (<1s).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VERIFY_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "verify_meta_v3_gate_v25.py"

PER_SLICE_KEYS = ("most_recent_12mo", "most_recent_24mo", "random_15pct")

# D-18 LOCKED formula hash (PROJECT.md cross-cutting invariant #3 —
# gate_contract_v2.3.json::formula_hash).
EXPECTED_FORMULA_HASH = (
    "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"
)


@pytest.fixture(scope="module")
def verify_module():
    """importlib-load the script as a module without invoking main()."""
    if not VERIFY_SCRIPT_PATH.exists():
        pytest.fail(
            f"verify_meta_v3_gate_v25.py missing at {VERIFY_SCRIPT_PATH}",
        )
    spec = importlib.util.spec_from_file_location(
        "verify_meta_v3_gate_v25", VERIFY_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        pytest.fail("Could not load spec for verify_meta_v3_gate_v25.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_meta_v3_gate_v25"] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Helpers ────────────────────────────────────────────────────────────────


def _mk_slice_metrics(brier: float, acc: float) -> dict:
    """Build a per-slice metric dict shaped like evaluate_per_slice output."""
    return {"brier_score": brier, "accuracy": acc}


def _mk_per_slice(brier_per_slice: dict, acc_per_slice: dict) -> dict:
    """Build full per_slice dict for all 3 keys."""
    return {
        slc: _mk_slice_metrics(brier_per_slice[slc], acc_per_slice[slc])
        for slc in PER_SLICE_KEYS
    }


# ─── Test 1: floor_clears_all_three ──────────────────────────────────────────


def test_floor_requires_all_three_slices(verify_module):
    """Floor: candidate Brier ≤ baseline Brier on ALL 3 slices AND
    candidate accuracy ≥ 0.70 on ALL 3 slices.
    """
    # Baseline (META-V22 apples-to-apples).
    baseline = _mk_per_slice(
        {"most_recent_12mo": 0.18, "most_recent_24mo": 0.20,
         "random_15pct": 0.17},
        {"most_recent_12mo": 0.75, "most_recent_24mo": 0.74,
         "random_15pct": 0.76},
    )
    # Candidate clears floor on all 3 (lower brier + acc ≥ 0.70).
    candidate_pass = _mk_per_slice(
        {"most_recent_12mo": 0.10, "most_recent_24mo": 0.11,
         "random_15pct": 0.09},
        {"most_recent_12mo": 0.85, "most_recent_24mo": 0.84,
         "random_15pct": 0.88},
    )
    assert verify_module.floor_clears_all_three(
        candidate_pass, baseline,
    ) is True

    # Candidate FAILS floor on one slice (brier > baseline on 24mo).
    candidate_brier_fail = _mk_per_slice(
        {"most_recent_12mo": 0.10, "most_recent_24mo": 0.25,
         "random_15pct": 0.09},
        {"most_recent_12mo": 0.85, "most_recent_24mo": 0.84,
         "random_15pct": 0.88},
    )
    assert verify_module.floor_clears_all_three(
        candidate_brier_fail, baseline,
    ) is False

    # Candidate FAILS floor on one slice (acc < 0.70 on random_15pct).
    candidate_acc_fail = _mk_per_slice(
        {"most_recent_12mo": 0.10, "most_recent_24mo": 0.11,
         "random_15pct": 0.09},
        {"most_recent_12mo": 0.85, "most_recent_24mo": 0.84,
         "random_15pct": 0.69},
    )
    assert verify_module.floor_clears_all_three(
        candidate_acc_fail, baseline,
    ) is False

    # Edge: candidate brier == baseline brier (must still pass floor on Brier).
    candidate_brier_equal = _mk_per_slice(
        {"most_recent_12mo": 0.18, "most_recent_24mo": 0.20,
         "random_15pct": 0.17},
        {"most_recent_12mo": 0.85, "most_recent_24mo": 0.84,
         "random_15pct": 0.88},
    )
    assert verify_module.floor_clears_all_three(
        candidate_brier_equal, baseline,
    ) is True

    # Edge: candidate accuracy == 0.70 exactly (must pass).
    candidate_acc_edge = _mk_per_slice(
        {"most_recent_12mo": 0.10, "most_recent_24mo": 0.11,
         "random_15pct": 0.09},
        {"most_recent_12mo": 0.70, "most_recent_24mo": 0.70,
         "random_15pct": 0.70},
    )
    assert verify_module.floor_clears_all_three(
        candidate_acc_edge, baseline,
    ) is True


# ─── Test 2: hurdle_clears_majority ──────────────────────────────────────────


def test_hurdle_requires_majority(verify_module):
    """Hurdle: candidate Brier improvement (baseline − candidate) ≥ 0.003 on
    ≥ majority (≥2/3) of slices.
    """
    baseline = _mk_per_slice(
        {"most_recent_12mo": 0.18, "most_recent_24mo": 0.20,
         "random_15pct": 0.17},
        {"most_recent_12mo": 0.75, "most_recent_24mo": 0.75,
         "random_15pct": 0.75},
    )
    # All 3 slices clear ≥0.003 improvement → True.
    candidate_3of3 = _mk_per_slice(
        {"most_recent_12mo": 0.10, "most_recent_24mo": 0.11,
         "random_15pct": 0.09},
        {"most_recent_12mo": 0.85, "most_recent_24mo": 0.84,
         "random_15pct": 0.88},
    )
    assert verify_module.hurdle_clears_majority(
        candidate_3of3, baseline,
    ) is True

    # 2 of 3 slices clear → True (majority threshold met).
    candidate_2of3 = _mk_per_slice(
        {"most_recent_12mo": 0.10, "most_recent_24mo": 0.11,
         "random_15pct": 0.169},  # delta = 0.001 < 0.003
        {"most_recent_12mo": 0.85, "most_recent_24mo": 0.84,
         "random_15pct": 0.88},
    )
    assert verify_module.hurdle_clears_majority(
        candidate_2of3, baseline,
    ) is True

    # 1 of 3 slices clears → False (below majority).
    candidate_1of3 = _mk_per_slice(
        {"most_recent_12mo": 0.10, "most_recent_24mo": 0.199,
         "random_15pct": 0.169},
        {"most_recent_12mo": 0.85, "most_recent_24mo": 0.84,
         "random_15pct": 0.88},
    )
    assert verify_module.hurdle_clears_majority(
        candidate_1of3, baseline,
    ) is False

    # Exactly delta == 0.003 (boundary — must clear, since ≥0.003).
    candidate_boundary = _mk_per_slice(
        {"most_recent_12mo": 0.177, "most_recent_24mo": 0.197,
         "random_15pct": 0.167},
        {"most_recent_12mo": 0.85, "most_recent_24mo": 0.84,
         "random_15pct": 0.88},
    )
    assert verify_module.hurdle_clears_majority(
        candidate_boundary, baseline,
    ) is True

    # No slice clears → False.
    candidate_0of3 = _mk_per_slice(
        {"most_recent_12mo": 0.179, "most_recent_24mo": 0.199,
         "random_15pct": 0.169},
        {"most_recent_12mo": 0.85, "most_recent_24mo": 0.84,
         "random_15pct": 0.88},
    )
    assert verify_module.hurdle_clears_majority(
        candidate_0of3, baseline,
    ) is False


# ─── Test 3: path_determination + XOR invariant ──────────────────────────────


def test_path_a_xor_path_b(verify_module):
    """``path_determination(floor, hurdle)`` returns "path_a" iff floor AND
    hurdle; "path_b" otherwise. Ensures the XOR invariant for
    (path_a_eligible, path_b_inevitable).
    """
    # path_a only when BOTH floor AND hurdle clear.
    assert verify_module.path_determination(True, True) == "path_a"
    assert verify_module.path_determination(False, True) == "path_b"
    assert verify_module.path_determination(True, False) == "path_b"
    assert verify_module.path_determination(False, False) == "path_b"

    # XOR invariant — verdict + booleans are consistent.
    for floor_pass, hurdle_pass in [(True, True), (False, True),
                                     (True, False), (False, False)]:
        verdict = verify_module.path_determination(floor_pass, hurdle_pass)
        path_a_eligible = verdict == "path_a"
        path_b_inevitable = not path_a_eligible
        # XOR: exactly one of the two booleans must be True.
        assert path_a_eligible != path_b_inevitable, (
            f"XOR invariant broken at "
            f"floor={floor_pass} hurdle={hurdle_pass}: "
            f"path_a_eligible={path_a_eligible} "
            f"path_b_inevitable={path_b_inevitable}"
        )


# ─── Test 4: formula_hash sentinel (D-18 LOCKED) ─────────────────────────────


def test_formula_hash_locked(verify_module):
    """``gate_formula_hash()`` returns the D-18 LOCKED formula hash verbatim
    from ``load_gate_contract(version='v2.3').formula_hash``.

    This sentinel test guards against post-measurement renegotiation. If
    .planning/gate_contract_v2.3.json::formula_hash drifts from the locked
    value, this test MUST fail — and the operator MUST audit the change
    (PROJECT.md cross-cutting invariant #3).
    """
    actual = verify_module.gate_formula_hash()
    assert actual == EXPECTED_FORMULA_HASH, (
        f"D-18 LOCKED formula hash drift detected! "
        f"got={actual!r} expected={EXPECTED_FORMULA_HASH!r}. "
        f"PROJECT.md cross-cutting invariant #3 binds this value — "
        f"no post-measurement renegotiation. Audit gate_contract_v2.3.json."
    )

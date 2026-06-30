"""Unit tests for scripts/verify_travel_oof_v25.py — TRAVEL OOF artifact verification.

Tests the pure-functional verdict logic that closes the Phase 42 operator caveat:
was the +0.249 Brier delta a real signal or a runtime-regenerated-OOF vs
training-time-OOF source-divergence artifact?

All tests use synthetic per-slice dicts (no DB / no model load). Pure-stdlib +
pytest; do NOT depend on real Phase 26 / 32 parquets being present.

RED-GREEN cycle:
    1. Write tests (this file) -> NameError on classify_verdict (RED).
    2. Implement scripts/verify_travel_oof_v25.py -> tests GREEN.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_travel_oof_v25.py"


def _load_script_module():
    """Load scripts/verify_travel_oof_v25.py as a module without packaging.

    Pure import to keep tests independent of sys.path tweaks.
    """
    spec = importlib.util.spec_from_file_location("verify_travel_oof_v25", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load spec for {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_travel_oof_v25"] = module
    spec.loader.exec_module(module)
    return module


# Synthetic GateContract stand-in. Mirrors the only attributes classify_verdict
# is allowed to touch (per the plan interface). Keeps tests free of real
# gate_contract_v2.3.json loading + lru_cache state.
def _gate_stub(
    hurdle_brier_delta: float = 0.003,
    floor_acc_threshold: float = 0.70,
) -> SimpleNamespace:
    return SimpleNamespace(
        hurdle_brier_delta=hurdle_brier_delta,
        floor_acc_threshold=floor_acc_threshold,
        hurdle_majority=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Behavior 1: artifact verdict — re-measurement evaporates the phase-42 delta
# ─────────────────────────────────────────────────────────────────────────────


def test_classify_artifact_when_re_measurement_evaporates_delta():
    """|re_measured| < 0.05 on >=2/3 slices AND phase_42 > 0.20 -> artifact."""
    mod = _load_script_module()
    re_measured = {
        "most_recent_12mo": 0.001,
        "most_recent_24mo": 0.001,
        "random_15pct": 0.002,
    }
    phase_42 = {
        "most_recent_12mo": 0.2492,
        "most_recent_24mo": 0.2492,
        "random_15pct": 0.2405,
    }
    floor_clears = {
        "most_recent_12mo": True,
        "most_recent_24mo": True,
        "random_15pct": True,
    }
    verdict = mod.classify_verdict(
        re_measured_per_slice_delta_brier=re_measured,
        phase_42_per_slice_delta_brier=phase_42,
        per_slice_floor_clears=floor_clears,
        gate=_gate_stub(),
    )
    assert verdict == "artifact"


# ─────────────────────────────────────────────────────────────────────────────
# Behavior 2: real verdict — re-measurement survives, floor clears
# ─────────────────────────────────────────────────────────────────────────────


def test_classify_real_when_re_measurement_survives_and_floor_clears():
    """All slices clear floor AND >=2/3 slices have delta >= 0.003 -> real."""
    mod = _load_script_module()
    re_measured = {
        "most_recent_12mo": 0.20,
        "most_recent_24mo": 0.20,
        "random_15pct": 0.18,
    }
    phase_42 = {
        "most_recent_12mo": 0.2492,
        "most_recent_24mo": 0.2492,
        "random_15pct": 0.2405,
    }
    floor_clears = {
        "most_recent_12mo": True,
        "most_recent_24mo": True,
        "random_15pct": True,
    }
    verdict = mod.classify_verdict(
        re_measured_per_slice_delta_brier=re_measured,
        phase_42_per_slice_delta_brier=phase_42,
        per_slice_floor_clears=floor_clears,
        gate=_gate_stub(),
    )
    assert verdict == "real"


# ─────────────────────────────────────────────────────────────────────────────
# Behavior 3: real_but_floor_misses — delta survives but floor breaks on a slice
# ─────────────────────────────────────────────────────────────────────────────


def test_classify_real_but_floor_misses_when_floor_breaks_on_any_slice():
    """Delta >= 0.003 on majority but floor fails on any slice -> no promote."""
    mod = _load_script_module()
    re_measured = {
        "most_recent_12mo": 0.20,
        "most_recent_24mo": 0.20,
        "random_15pct": 0.18,
    }
    phase_42 = {
        "most_recent_12mo": 0.2492,
        "most_recent_24mo": 0.2492,
        "random_15pct": 0.2405,
    }
    floor_clears = {
        "most_recent_12mo": True,
        "most_recent_24mo": False,  # floor failure on one slice
        "random_15pct": True,
    }
    verdict = mod.classify_verdict(
        re_measured_per_slice_delta_brier=re_measured,
        phase_42_per_slice_delta_brier=phase_42,
        per_slice_floor_clears=floor_clears,
        gate=_gate_stub(),
    )
    assert verdict == "real_but_floor_misses"


# ─────────────────────────────────────────────────────────────────────────────
# Behavior 4: artifact_explanation quotes BOTH source deltas side-by-side
# ─────────────────────────────────────────────────────────────────────────────


def test_oof_source_divergence_explanation_quotes_both_deltas():
    """artifact_explanation() string contains both phase_42 (+0.249) and
    re-measurement delta numbers so the audit trail is reproducible."""
    mod = _load_script_module()
    phase_42 = {
        "most_recent_12mo": 0.2492,
        "most_recent_24mo": 0.2492,
        "random_15pct": 0.2405,
    }
    re_measured = {
        "most_recent_12mo": 0.001,
        "most_recent_24mo": 0.001,
        "random_15pct": 0.002,
    }
    explanation = mod.artifact_explanation(
        phase_42_deltas=phase_42,
        re_measured_deltas=re_measured,
    )
    # Both source delta numbers MUST appear in the string for partner audit trail.
    assert "0.2492" in explanation or "0.249" in explanation
    assert "0.001" in explanation
    assert "training-time" in explanation.lower()
    assert "runtime" in explanation.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Behavior 5: artifact verdict -> conservative path locked downstream
# ─────────────────────────────────────────────────────────────────────────────


def test_downstream_implication_artifact_locks_conservative_path():
    """verdict == 'artifact' -> downstream_implication reads 'conservative
    path locked: TRAVEL cols 75-80 ONLY in Wave 2 meta_v3 input space'."""
    mod = _load_script_module()
    impl = mod.downstream_implication(verdict="artifact")
    assert "conservative path locked" in impl
    assert "TRAVEL cols 75-80" in impl
    assert "Wave 2" in impl


# ─────────────────────────────────────────────────────────────────────────────
# Behavior 6: real verdict -> v2.6 backlog entry + conservative path stays
# ─────────────────────────────────────────────────────────────────────────────


def test_downstream_implication_real_flags_v26_backlog_but_keeps_conservative():
    """verdict == 'real' -> downstream_implication mentions v2.6 META-V24 backlog
    entry AND still notes Wave 2 stays on conservative path (Phase 45
    scope-locked per D-CONTEXT §TRAVEL Inclusion Strategy)."""
    mod = _load_script_module()
    impl = mod.downstream_implication(verdict="real")
    assert "v2.6" in impl
    assert "META-V24" in impl
    assert "backlog" in impl
    assert "conservative path" in impl
    assert "Phase 45 scope-locked" in impl


# ─────────────────────────────────────────────────────────────────────────────
# Extra: real_but_floor_misses also documents same conservative path
# ─────────────────────────────────────────────────────────────────────────────


def test_downstream_implication_real_but_floor_misses_keeps_conservative():
    mod = _load_script_module()
    impl = mod.downstream_implication(verdict="real_but_floor_misses")
    assert "conservative path" in impl
    assert "floor" in impl.lower()

"""Phase 31 / GATE-V23-01,02,03 — v2.3 contract output shape tests (Wave 0 RED).

Exercises the ``_emit_v23_contract`` helper in
``src/ufc_prediction/cli/predict.py`` directly with synthetic
per-slice thresholds + aggregated variance dicts. Verifies:

  - Top-level fields (version=v2.3, formula_hash D-18-locked,
    n_seeds_observed, n_features=90, base_features_set, k_value,
    cutoff_date, ingest_completed_at, bfo_backfill_committed_at,
    feature_columns_hash present).
  - Per-slice operator_floor_applied=True when pre_floor_accuracy >= 0.70.
  - supersedes list includes "D-21(v2.2, GATE)".
  - JSON byte format: indent=2 + trailing newline (audit convention).
  - formula_hash equals D-18 binding constant (drift guard).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _synthetic_per_slice_above_floor():
    """PerSliceThresholds dict where all slices have accuracy_min >= 0.70 (Path A)."""
    from ufc_prediction.ml.gate_contract import PerSliceThresholds

    base = dict(
        brier_max=0.21,
        accuracy_min=0.72,  # >= floor 0.70
        median_brier_xgb_v2=0.20,
        median_acc_xgb_v2=0.71,
        seed_std_brier=0.003,
        seed_std_acc=0.005,
        bootstrap_ci_half_brier=0.004,
        bootstrap_ci_half_acc=0.006,
        std_brier_used=0.004,
        std_acc_used=0.006,
        operator_floor_applied=True,
        operator_floor_value=0.70,
        operator_decision="path_a_floor_applied",
    )
    return {
        "most_recent_12mo": PerSliceThresholds(**base),
        "most_recent_24mo": PerSliceThresholds(**base),
        "random_15pct": PerSliceThresholds(**base),
    }


def test_emit_v23_contract_shape(tmp_path):
    """Emitted JSON top-level fields satisfy v2.3 contract schema."""
    from ufc_prediction.cli.predict import _emit_v23_contract

    contract_path = tmp_path / "gate_contract_v2.3.json"
    _emit_v23_contract(
        _synthetic_per_slice_above_floor(),
        seed_list=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        contract_path=contract_path,
    )

    assert contract_path.exists()
    c = json.loads(contract_path.read_text(encoding="utf-8"))
    assert c["version"] == "v2.3"
    assert c["n_seeds_observed"] == 10
    assert c["n_features"] == 90
    assert c["base_features_set"] == "FEATURE_COLUMNS_V22"
    assert c["k_value"] == 1
    assert c["cutoff_date"] == "2023-01-01"
    assert c["formula_hash"] == (
        "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"
    )
    assert c["ingest_completed_at"]  # non-empty
    assert c["bfo_backfill_committed_at"]  # non-empty
    assert c["feature_columns_hash"]  # non-empty (sha256 hex)
    assert len(c["feature_columns_hash"]) == 64


def test_emit_v23_contract_formula_hash_is_locked_constant(tmp_path):
    """formula_hash field equals D-18 binding hash; no drift allowed."""
    from ufc_prediction.cli.predict import _emit_v23_contract

    contract_path = tmp_path / "gate_contract_v2.3.json"
    _emit_v23_contract(
        _synthetic_per_slice_above_floor(),
        seed_list=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        contract_path=contract_path,
    )
    c = json.loads(contract_path.read_text(encoding="utf-8"))
    assert c["formula_hash"] == (
        "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"
    )


def test_emit_v23_contract_per_slice_floor_applied_when_above_floor(tmp_path):
    """When pre_floor accuracy_min >= 0.70, operator_floor_applied=True per slice."""
    from ufc_prediction.cli.predict import _emit_v23_contract

    contract_path = tmp_path / "gate_contract_v2.3.json"
    _emit_v23_contract(
        _synthetic_per_slice_above_floor(),
        seed_list=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        contract_path=contract_path,
    )
    c = json.loads(contract_path.read_text(encoding="utf-8"))
    for slc in ("most_recent_12mo", "most_recent_24mo", "random_15pct"):
        ts = c["per_slice"][slc]
        assert ts["accuracy_min"] >= 0.70
        assert ts["operator_floor_applied"] is True
        assert ts["operator_floor_value"] == 0.70


def test_emit_v23_contract_supersedes_includes_d21(tmp_path):
    """supersedes list includes 'D-21(v2.2, GATE)' (lineage chain)."""
    from ufc_prediction.cli.predict import _emit_v23_contract

    contract_path = tmp_path / "gate_contract_v2.3.json"
    _emit_v23_contract(
        _synthetic_per_slice_above_floor(),
        seed_list=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        contract_path=contract_path,
    )
    c = json.loads(contract_path.read_text(encoding="utf-8"))
    assert "D-21(v2.2, GATE)" in c["supersedes"]


def test_emit_v23_contract_json_indent_and_trailing_newline(tmp_path):
    """Audit format convention: indent=2 + trailing newline."""
    from ufc_prediction.cli.predict import _emit_v23_contract

    contract_path = tmp_path / "gate_contract_v2.3.json"
    _emit_v23_contract(
        _synthetic_per_slice_above_floor(),
        seed_list=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        contract_path=contract_path,
    )
    raw = contract_path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    # Indent-2 = the JSON is multi-line with 2-space indents on nested values.
    # Heuristic: top-level keys appear on their own indented lines.
    assert '\n  "version": "v2.3"' in raw

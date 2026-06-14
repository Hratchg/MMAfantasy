"""Tests for gate_contract module — versioned promotion-gate loader.

Per CONTEXT.md D-04/D-05/D-08(P17): GateContract is a frozen dataclass
loaded from .planning/gate_contract.json with lru_cache(maxsize=1).
GateContractError fail-closes on missing/malformed/out-of-range input.

Pitfall C (lru_cache discipline): every test calls
load_gate_contract.cache_clear() BEFORE writing the contract so cached
state from a prior test cannot leak into this one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _valid_contract_dict() -> dict:
    """Canonical 9-required-field + 3-slice-nested contract dict per D-08(P17).

    Each slice has all 10 PerSliceThresholds fields populated with values
    inside the validation range (brier_max in [0,1], accuracy_min in [0.5,1]).
    """
    slice_payload = {
        "brier_max": 0.21,
        "accuracy_min": 0.68,
        "median_brier_xgb_v2": 0.22,
        "median_acc_xgb_v2": 0.67,
        "seed_std_brier": 0.005,
        "seed_std_acc": 0.008,
        "bootstrap_ci_half_brier": 0.004,
        "bootstrap_ci_half_acc": 0.006,
        "std_brier_used": 0.005,
        "std_acc_used": 0.008,
    }
    return {
        "version": "v2.1",
        "derived_at": "2026-05-04",
        "n_seeds_observed": 10,
        "base_features_set": "FEATURE_COLUMNS_NO_NET",
        "n_features": 72,  # IN-04: matches real v2.1 contract on disk
        "k_value": 1,
        "formula_hash": "0" * 64,
        "cutoff_date": "2023-01-01",
        "per_slice": {
            "most_recent_12mo": dict(slice_payload),
            "most_recent_24mo": dict(slice_payload),
            "random_15pct": dict(slice_payload),
        },
        "secondary_metrics_observed": {},
        "supersedes": ["D-13(P16)", "D-17(v2.0)"],
        "notes": "test fixture",
    }


def _write_contract(tmp_path: Path, payload: dict) -> Path:
    """Serialize payload to tmp_path/gate_contract.json and return the path."""
    p = tmp_path / "gate_contract.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _valid_v22_contract_dict() -> dict:
    """Canonical v2.2 contract dict per CONTEXT D-07.

    Differs from _valid_contract_dict():
      - version: "v2.2"
      - base_features_set: "FEATURE_COLUMNS_V22"
      - n_features: 90
      - feature_columns_hash: 64-char hex placeholder
      - bfo_backfill_committed_at: ISO timestamp placeholder
      - All other fields schema-identical (per D-07).
    """
    payload = _valid_contract_dict()
    payload["version"] = "v2.2"
    payload["base_features_set"] = "FEATURE_COLUMNS_V22"
    payload["n_features"] = 90
    payload["feature_columns_hash"] = "0" * 64
    payload["bfo_backfill_committed_at"] = "2026-05-15T11:14:55-07:00"
    return payload


def _write_v22_contract(tmp_path: Path, payload: dict) -> Path:
    """Serialize payload to tmp_path/gate_contract_v2.2.json and return the path."""
    p = tmp_path / "gate_contract_v2.2.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


class TestLoadGateContract:
    """Tests for src/ufc_prediction/ml/gate_contract.py:load_gate_contract().

    Pitfall C: every test clears the lru_cache BEFORE writing the contract
    so stale cached parses cannot leak across tests.
    """

    def test_load_valid(self, tmp_path):
        """A schema-valid contract loads into a GateContract with all fields populated."""
        from ufc_prediction.ml.gate_contract import load_gate_contract

        load_gate_contract.cache_clear()
        p = _write_contract(tmp_path, _valid_contract_dict())

        contract = load_gate_contract(p)

        assert contract.version == "v2.1"
        assert contract.n_features == 72
        assert contract.n_seeds_observed == 10
        assert contract.base_features_set == "FEATURE_COLUMNS_NO_NET"
        assert contract.k_value == 1
        assert contract.cutoff_date == "2023-01-01"
        assert "most_recent_12mo" in contract.per_slice
        assert "most_recent_24mo" in contract.per_slice
        assert "random_15pct" in contract.per_slice
        assert contract.per_slice["most_recent_12mo"].brier_max == pytest.approx(0.21)
        assert contract.per_slice["most_recent_12mo"].accuracy_min == pytest.approx(0.68)

        load_gate_contract.cache_clear()

    def test_load_missing_field_raises(self, tmp_path):
        """Dropping a required top-level field raises GateContractError with 'missing'."""
        from ufc_prediction.ml.gate_contract import GateContractError, load_gate_contract

        load_gate_contract.cache_clear()
        payload = _valid_contract_dict()
        del payload["formula_hash"]
        p = _write_contract(tmp_path, payload)

        with pytest.raises(GateContractError, match="missing"):
            load_gate_contract(p)

        load_gate_contract.cache_clear()

    def test_load_bad_version_raises(self, tmp_path):
        """An unsupported version raises GateContractError mentioning 'version'."""
        from ufc_prediction.ml.gate_contract import GateContractError, load_gate_contract

        load_gate_contract.cache_clear()
        payload = _valid_contract_dict()
        payload["version"] = "v3.0"
        p = _write_contract(tmp_path, payload)

        with pytest.raises(GateContractError, match="version"):
            load_gate_contract(p)

        load_gate_contract.cache_clear()

    def test_load_missing_slice_raises(self, tmp_path):
        """Dropping an EXPECTED_SLICES key raises GateContractError mentioning the slice."""
        from ufc_prediction.ml.gate_contract import GateContractError, load_gate_contract

        load_gate_contract.cache_clear()
        payload = _valid_contract_dict()
        del payload["per_slice"]["random_15pct"]
        p = _write_contract(tmp_path, payload)

        with pytest.raises(GateContractError, match="random_15pct"):
            load_gate_contract(p)

        load_gate_contract.cache_clear()

    def test_load_is_cached(self, tmp_path):
        """Calling load_gate_contract twice with the same path returns the same instance."""
        from ufc_prediction.ml.gate_contract import load_gate_contract

        load_gate_contract.cache_clear()
        p = _write_contract(tmp_path, _valid_contract_dict())

        c1 = load_gate_contract(p)
        c2 = load_gate_contract(p)

        assert c1 is c2

        load_gate_contract.cache_clear()


class TestV22Dispatch:
    """Tests for v2.2 version kwarg dispatch in load_gate_contract.

    Pitfall C (lru_cache discipline): every test clears the lru_cache BOTH
    before writing the contract AND after assertions, so cached state from
    a prior test cannot leak.
    """

    def test_load_v22_via_explicit_path(self, tmp_path):
        """A schema-valid v2.2 contract loads with all fields including v2.2-only fields."""
        from ufc_prediction.ml.gate_contract import load_gate_contract

        load_gate_contract.cache_clear()
        p = _write_v22_contract(tmp_path, _valid_v22_contract_dict())

        contract = load_gate_contract(p)

        assert contract.version == "v2.2"
        assert contract.base_features_set == "FEATURE_COLUMNS_V22"
        assert contract.n_features == 90
        assert contract.feature_columns_hash == "0" * 64
        assert contract.bfo_backfill_committed_at == "2026-05-15T11:14:55-07:00"
        load_gate_contract.cache_clear()

    def test_load_v21_default_back_compat(self, tmp_path):
        """load_gate_contract() with no kwargs still parses a v2.1 contract from default."""
        from ufc_prediction.ml.gate_contract import load_gate_contract

        load_gate_contract.cache_clear()
        p = _write_contract(tmp_path, _valid_contract_dict())

        # Pass path explicitly to bypass DEFAULT_CONTRACT_PATH; default version stays v2.1.
        contract = load_gate_contract(p)

        assert contract.version == "v2.1"
        assert contract.feature_columns_hash == ""  # v2.1 back-compat default
        assert contract.bfo_backfill_committed_at == ""
        load_gate_contract.cache_clear()

    def test_load_unknown_version_raises(self):
        """load_gate_contract(version='v3.0') raises ValueError with 'unknown version'."""
        from ufc_prediction.ml.gate_contract import load_gate_contract

        load_gate_contract.cache_clear()
        with pytest.raises(ValueError, match="unknown version"):
            load_gate_contract(version="v3.0")
        load_gate_contract.cache_clear()

    def test_load_v22_missing_feature_columns_hash_raises(self, tmp_path):
        """A v2.2 contract missing feature_columns_hash raises GateContractError."""
        from ufc_prediction.ml.gate_contract import GateContractError, load_gate_contract

        load_gate_contract.cache_clear()
        payload = _valid_v22_contract_dict()
        payload["feature_columns_hash"] = ""  # explicit empty
        p = _write_v22_contract(tmp_path, payload)

        with pytest.raises(GateContractError, match="feature_columns_hash"):
            load_gate_contract(p)
        load_gate_contract.cache_clear()

    def test_load_v22_missing_bfo_backfill_committed_at_raises(self, tmp_path):
        """A v2.2 contract missing bfo_backfill_committed_at raises GateContractError."""
        from ufc_prediction.ml.gate_contract import GateContractError, load_gate_contract

        load_gate_contract.cache_clear()
        payload = _valid_v22_contract_dict()
        payload["bfo_backfill_committed_at"] = ""
        p = _write_v22_contract(tmp_path, payload)

        with pytest.raises(GateContractError, match="bfo_backfill_committed_at"):
            load_gate_contract(p)
        load_gate_contract.cache_clear()

    def test_load_v22_dispatch_resolves_to_v22_path(self, tmp_path, monkeypatch):
        """load_gate_contract(version='v2.2') resolves to V22_CONTRACT_PATH."""
        from ufc_prediction.ml import gate_contract as gc_mod
        from ufc_prediction.ml.gate_contract import load_gate_contract

        load_gate_contract.cache_clear()
        v22_path = tmp_path / "gate_contract_v2.2.json"
        v22_path.write_text(json.dumps(_valid_v22_contract_dict()), encoding="utf-8")
        monkeypatch.setattr(gc_mod, "V22_CONTRACT_PATH", v22_path)
        load_gate_contract.cache_clear()  # bookend after monkey-patch

        contract = load_gate_contract(version="v2.2")

        assert contract.version == "v2.2"
        assert contract.n_features == 90
        load_gate_contract.cache_clear()

"""Phase 31 / GATE-V23-02 — gate_contract v2.3 dispatch unit tests (Wave 0 RED).

Mirrors the v2.2 dispatch test pattern at ``tests/ml/test_gate_contract.py``
(``TestV22Dispatch``); extends ``SUPPORTED_VERSIONS`` to include ``v2.3``
and adds the ``ingest_completed_at`` required-field validation per
CONTEXT D-02 (preconditions plan).

Tests follow the Pitfall 6 (lru_cache discipline) bookend pattern:
every test calls ``load_gate_contract.cache_clear()`` BOTH before any
file write AND after assertions, so cached parses cannot leak across tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _valid_v23_contract_dict() -> dict:
    """Canonical v2.3 contract dict per CONTEXT D-02 / D-05.

    Differs from the v2.2 baseline:
      - version: "v2.3"
      - base_features_set: "FEATURE_COLUMNS_V22"  (substrate fill, not schema change)
      - n_features: 90
      - feature_columns_hash: 64-char hex placeholder
      - bfo_backfill_committed_at: ISO timestamp placeholder (carry-forward from v2.2)
      - ingest_completed_at: ISO timestamp placeholder (NEW v2.3 required field)
    """
    slice_payload = {
        "brier_max": 0.21,
        "accuracy_min": 0.70,
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
        "version": "v2.3",
        "derived_at": "2026-05-22",
        "n_seeds_observed": 10,
        "base_features_set": "FEATURE_COLUMNS_V22",
        "n_features": 90,
        "k_value": 1,
        "formula_hash": (
            "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"
        ),
        "cutoff_date": "2023-01-01",
        "per_slice": {
            "most_recent_12mo": dict(slice_payload),
            "most_recent_24mo": dict(slice_payload),
            "random_15pct": dict(slice_payload),
        },
        "secondary_metrics_observed": {},
        "supersedes": ["D-21(v2.2, GATE)", "v2.2-gate"],
        "notes": "test fixture (v2.3)",
        "feature_columns_hash": "0" * 64,
        "bfo_backfill_committed_at": "2026-05-15T15:06:49-07:00",
        "ingest_completed_at": "2026-05-20T01:59:08-07:00",
    }


def _write_v23_contract(tmp_path: Path, payload: dict) -> Path:
    """Serialize payload to tmp_path/gate_contract_v2.3.json and return path."""
    p = tmp_path / "gate_contract_v2.3.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


class TestV23Dispatch:
    """Tests for v2.3 version kwarg dispatch in load_gate_contract.

    Wave 0 RED tests — fail before the v2.3 dispatch is added in this task;
    GREEN after the gate_contract.py delta lands.
    """

    def test_load_v23(self, tmp_path, monkeypatch):
        """load_gate_contract(version='v2.3') with V23 file MISSING raises GateContractError.

        Post-Plan-31-02 the real `.planning/gate_contract_v2.3.json` exists on
        disk, so the canary asserts the loader's file-missing path by
        redirecting V23_CONTRACT_PATH to a non-existent tmp_path. The DISPATCH
        must resolve cleanly to V23_CONTRACT_PATH (not raise ValueError); only
        the missing FILE should trigger GateContractError. This tests that
        v2.3 is registered in SUPPORTED_VERSIONS independently of whether
        Plan 31-02 has materialized the contract.
        """
        from ufc_prediction.ml import gate_contract as gc_module
        from ufc_prediction.ml.gate_contract import (
            GateContractError,
            load_gate_contract,
        )

        missing_path = tmp_path / "gate_contract_v2.3.json"
        assert not missing_path.exists()
        monkeypatch.setattr(gc_module, "V23_CONTRACT_PATH", missing_path)

        load_gate_contract.cache_clear()
        with pytest.raises(GateContractError):
            load_gate_contract(version="v2.3")
        load_gate_contract.cache_clear()

    def test_load_v23_validates_ingest_completed_at_required(self, tmp_path):
        """A v2.3 contract missing ingest_completed_at raises GateContractError."""
        from ufc_prediction.ml.gate_contract import (
            GateContractError,
            load_gate_contract,
        )

        load_gate_contract.cache_clear()
        payload = _valid_v23_contract_dict()
        payload["ingest_completed_at"] = ""  # explicit empty
        p = _write_v23_contract(tmp_path, payload)

        with pytest.raises(GateContractError, match="ingest_completed_at"):
            load_gate_contract(p)
        load_gate_contract.cache_clear()

    def test_load_v21_v22_preserved(self):
        """v2.1 + v2.2 dispatch still resolve against the real on-disk contracts.

        Audit-lineage regression: GATE-V23-02 binds v2.1 + v2.2 contracts to
        PRESERVED untouched on-disk state; the loader MUST keep parsing them.
        """
        from ufc_prediction.ml.gate_contract import load_gate_contract

        load_gate_contract.cache_clear()
        c21 = load_gate_contract(version="v2.1")
        assert c21.version == "v2.1"
        # D-18 binding: formula hash preserved
        assert (
            c21.formula_hash
            == "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"
        )

        load_gate_contract.cache_clear()
        c22 = load_gate_contract(version="v2.2")
        assert c22.version == "v2.2"
        assert (
            c22.formula_hash
            == "7d221b4ac21e550c3341db32c2bcec0de0bee5b87c5b9ec498163b81dd7ed20a"
        )
        load_gate_contract.cache_clear()

    def test_supported_versions_includes_v23(self):
        """v2.3 is registered in the SUPPORTED_VERSIONS frozenset."""
        from ufc_prediction.ml.gate_contract import SUPPORTED_VERSIONS

        assert "v2.3" in SUPPORTED_VERSIONS
        # v2.1 + v2.2 stay registered (audit lineage)
        assert "v2.1" in SUPPORTED_VERSIONS
        assert "v2.2" in SUPPORTED_VERSIONS

    def test_cache_maxsize_is_at_least_3(self):
        """lru_cache(maxsize) >= 3 so v2.1+v2.2+v2.3 can coexist (Pitfall 6)."""
        from ufc_prediction.ml.gate_contract import load_gate_contract

        info = load_gate_contract.cache_info()
        assert info.maxsize is not None
        assert info.maxsize >= 3

    def test_load_v23_via_explicit_path_succeeds(self, tmp_path):
        """A schema-valid v2.3 contract loads with all fields including v2.3-only field."""
        from ufc_prediction.ml.gate_contract import load_gate_contract

        load_gate_contract.cache_clear()
        p = _write_v23_contract(tmp_path, _valid_v23_contract_dict())

        contract = load_gate_contract(p)

        assert contract.version == "v2.3"
        assert contract.base_features_set == "FEATURE_COLUMNS_V22"
        assert contract.n_features == 90
        assert contract.feature_columns_hash == "0" * 64
        assert contract.bfo_backfill_committed_at == "2026-05-15T15:06:49-07:00"
        assert contract.ingest_completed_at == "2026-05-20T01:59:08-07:00"
        load_gate_contract.cache_clear()

    def test_load_v23_dispatch_resolves_to_v23_path(self, tmp_path, monkeypatch):
        """load_gate_contract(version='v2.3') resolves to V23_CONTRACT_PATH."""
        from ufc_prediction.ml import gate_contract as gc_mod
        from ufc_prediction.ml.gate_contract import load_gate_contract

        load_gate_contract.cache_clear()
        v23_path = tmp_path / "gate_contract_v2.3.json"
        v23_path.write_text(
            json.dumps(_valid_v23_contract_dict()), encoding="utf-8"
        )
        monkeypatch.setattr(gc_mod, "V23_CONTRACT_PATH", v23_path)
        load_gate_contract.cache_clear()  # bookend after monkeypatch

        contract = load_gate_contract(version="v2.3")

        assert contract.version == "v2.3"
        assert contract.ingest_completed_at == "2026-05-20T01:59:08-07:00"
        load_gate_contract.cache_clear()

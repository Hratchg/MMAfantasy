"""FALLBACK-V24-01 — Artifact integrity tests for xgb_v2_no_odds triad.

Plan 34-02 Task 3 — locks the contract of the three artifacts shipped by
scripts/train_xgb_v2_no_odds.py. Run as a unit test; no DB / no training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"

FALLBACK_JOBLIB = MODELS_DIR / "xgb_v2_no_odds.joblib"
FALLBACK_META = MODELS_DIR / "xgb_v2_no_odds_meta.json"
FALLBACK_CONTRACT = MODELS_DIR / "xgb_v2_no_odds-contract.json"

CANONICAL_XGB_V2_SHA = "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"


def test_artifact_triad_exists() -> None:
    """All 3 files exist with non-zero size."""
    for p in (FALLBACK_JOBLIB, FALLBACK_META, FALLBACK_CONTRACT):
        assert p.exists(), f"missing artifact: {p}"
        assert p.stat().st_size > 0, f"zero-byte artifact: {p}"


def test_model_n_features_is_67() -> None:
    """The persisted CalibratedClassifierCV reports n_features_in_ == 67."""
    model = joblib.load(FALLBACK_JOBLIB)
    assert hasattr(model, "n_features_in_"), "no n_features_in_ attribute"
    assert model.n_features_in_ == 67, model.n_features_in_


def test_meta_json_schema() -> None:
    """meta JSON has required keys with the expected types."""
    meta = json.loads(FALLBACK_META.read_text())
    assert meta["model_name"] == "xgb_v2_no_odds"
    assert meta["n_features"] == 67
    assert isinstance(meta["feature_columns"], list)
    assert len(meta["feature_columns"]) == 67
    assert meta["parent_model"] == "xgb_v2"
    assert meta["parent_model_sha256"] == CANONICAL_XGB_V2_SHA
    assert isinstance(meta["best_params"], dict)
    assert isinstance(meta["per_slice_metrics"], dict)
    assert set(meta["per_slice_metrics"].keys()) == {
        "most_recent_12mo",
        "most_recent_24mo",
        "random_15pct",
    }
    # Dropped columns assertion (revision: must list the 5 odds cols sorted)
    assert set(meta["dropped_columns"]) == {
        "opening_prob_diff",
        "closing_prob_diff",
        "line_movement_diff",
        "sharp_money_signal",
        "odds_elo_divergence",
    }
    assert meta["train_seed"] == 42


def test_contract_promoted() -> None:
    """Contract is marked promoted and references the v2.3 gate contract."""
    con = json.loads(FALLBACK_CONTRACT.read_text())
    assert con["candidate_or_promoted"] == "promoted"
    assert con["gate_contract_ref"].endswith("gate_contract_v2.3.json")
    assert con["model_name"] == "xgb_v2_no_odds"
    assert con["parent_model"] == "xgb_v2"
    # model_artifact_sha256 must match the on-disk .joblib SHA
    on_disk_sha = hashlib.sha256(FALLBACK_JOBLIB.read_bytes()).hexdigest()
    assert con["model_artifact_sha256"] == on_disk_sha


def test_rejects_72col_input() -> None:
    """Feeding a 72-col vector raises (catches accidental wrong-feature-set bug)."""
    model = joblib.load(FALLBACK_JOBLIB)
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        model.predict_proba(np.zeros((1, 72)))


def test_parent_sha_matches_canonical() -> None:
    """meta JSON's parent_model_sha256 == canonical xgb_v2 hex (AUDIT-01 anchor)."""
    meta = json.loads(FALLBACK_META.read_text())
    assert meta["parent_model_sha256"] == CANONICAL_XGB_V2_SHA

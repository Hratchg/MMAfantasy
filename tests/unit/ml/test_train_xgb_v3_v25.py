"""Unit tests for scripts/train_xgb_v3_v25.py (Phase 45 META3-V25-01).

Pure-functional invariant tests; no DB load, no model training.

Per Plan 45-02 Task 1 <behavior>:
  - Test 1: load_xgb_v2_training_config() returns dict with cutoff_date,
    feature_columns (== FEATURE_COLUMNS_NO_NET), best_params (verbatim
    from models/xgb_v2_meta.json).
  - Test 2: assert_apples_to_apples(config) raises if cutoff_date,
    feature_columns, or best_params keys diverge from xgb_v2's.
  - Test 3: build_seed_list() returns [42, 43, 44, 45, 46] exactly.
  - Test 4: debutant_elo_is_seeded(fight_record) — for a synthetic debutant
    record with elo_overall_pre = 1500.0, returns False; with seeded value
    (e.g., 1487.3), returns True. For non-debutants (n_ufc_fights > 0),
    returns True unconditionally (no-op).

Loads test target via importlib (matches Phase 45-01 verify_travel_oof_v25
pattern) so the script doesn't need to be a packaged module.
"""
from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_xgb_v3_v25.py"
XGB_V2_META_PATH = PROJECT_ROOT / "models" / "xgb_v2_meta.json"


def _load_module():
    """Importlib-load scripts/train_xgb_v3_v25.py as a module."""
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Script not found: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("train_xgb_v3_v25", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─── Test 1: load_xgb_v2_training_config ──────────────────────────────────

def test_load_xgb_v2_training_config_returns_cutoff_features_params():
    """load_xgb_v2_training_config() returns dict with cutoff_date,
    feature_columns (FEATURE_COLUMNS_NO_NET 72 cols), best_params verbatim
    from models/xgb_v2_meta.json."""
    from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET

    module = _load_module()
    config = module.load_xgb_v2_training_config()

    # Cutoff date: parsed to date(2023, 1, 1).
    assert config["cutoff_date"] == date(2023, 1, 1), (
        f"cutoff_date mismatch: {config['cutoff_date']}"
    )

    # Feature columns: FEATURE_COLUMNS_NO_NET verbatim (72 cols).
    assert config["feature_columns"] == list(FEATURE_COLUMNS_NO_NET), (
        "feature_columns drift from FEATURE_COLUMNS_NO_NET"
    )
    assert len(config["feature_columns"]) == 72

    # best_params: byte-identical to xgb_v2_meta.json::best_params.
    xgb_v2_meta = json.loads(XGB_V2_META_PATH.read_text(encoding="utf-8"))
    assert config["best_params"] == xgb_v2_meta["best_params"], (
        "best_params drift from xgb_v2_meta.json"
    )


# ─── Test 2: assert_apples_to_apples ──────────────────────────────────────

def test_assert_apples_to_apples_accepts_valid_config():
    """assert_apples_to_apples(config) returns None for a valid config."""
    module = _load_module()
    config = module.load_xgb_v2_training_config()
    # Should not raise.
    module.assert_apples_to_apples(config)


def test_assert_apples_to_apples_raises_on_cutoff_drift():
    """Mutated cutoff_date triggers AssertionError."""
    module = _load_module()
    config = module.load_xgb_v2_training_config()
    bad = dict(config)
    bad["cutoff_date"] = date(2024, 1, 1)  # drift
    with pytest.raises(AssertionError, match="cutoff"):
        module.assert_apples_to_apples(bad)


def test_assert_apples_to_apples_raises_on_feature_columns_drift():
    """Mutated feature_columns triggers AssertionError."""
    module = _load_module()
    config = module.load_xgb_v2_training_config()
    bad = dict(config)
    bad["feature_columns"] = config["feature_columns"][:-1]  # drop one col
    with pytest.raises(AssertionError, match="feature_columns"):
        module.assert_apples_to_apples(bad)


def test_assert_apples_to_apples_raises_on_best_params_keyset_drift():
    """Mutated best_params keyset triggers AssertionError."""
    module = _load_module()
    config = module.load_xgb_v2_training_config()
    bad = dict(config)
    bad_bp = dict(config["best_params"])
    bad_bp["bogus_extra_key"] = 1.234
    bad["best_params"] = bad_bp
    with pytest.raises(AssertionError, match="best_params"):
        module.assert_apples_to_apples(bad)


# ─── Test 3: build_seed_list ──────────────────────────────────────────────

def test_build_seed_list_returns_42_43_44_45_46():
    """build_seed_list() returns [42, 43, 44, 45, 46] exactly (matches
    xgb_v2 5-seed harness per D-CONTEXT)."""
    module = _load_module()
    seeds = module.build_seed_list()
    assert seeds == [42, 43, 44, 45, 46]


# ─── Test 4: debutant_elo_is_seeded ───────────────────────────────────────

def test_debutant_with_default_1500_elo_is_not_seeded():
    """For a debutant fighter (n_ufc_fights=0), if elo_overall_pre == 1500.0,
    return False (default value, not seeded from Sherdog)."""
    module = _load_module()
    fight_record = {
        "n_ufc_fights": 0,
        "elo_overall_pre": 1500.0,
    }
    assert module.debutant_elo_is_seeded(fight_record) is False


def test_debutant_with_seeded_elo_is_seeded():
    """For a debutant fighter with elo_overall_pre != 1500.0 (e.g., Phase
    43 backfilled value from pre-UFC record), return True."""
    module = _load_module()
    fight_record = {
        "n_ufc_fights": 0,
        "elo_overall_pre": 1487.3,
    }
    assert module.debutant_elo_is_seeded(fight_record) is True


def test_non_debutant_elo_check_is_noop():
    """For non-debutants (n_ufc_fights > 0), seeded check is a no-op:
    return True unconditionally."""
    module = _load_module()
    fight_record_default_elo = {
        "n_ufc_fights": 5,
        "elo_overall_pre": 1500.0,
    }
    fight_record_seeded_elo = {
        "n_ufc_fights": 10,
        "elo_overall_pre": 1623.7,
    }
    assert module.debutant_elo_is_seeded(fight_record_default_elo) is True
    assert module.debutant_elo_is_seeded(fight_record_seeded_elo) is True

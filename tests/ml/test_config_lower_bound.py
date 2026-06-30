"""RED tests for MLConfig.train_lower_bound_date (Phase 15.1 D-05/D-06/D-07)."""

from __future__ import annotations

import dataclasses

import pytest

from ufc_prediction.ml.config import MLConfig


def test_field_added_with_default_none() -> None:
    cfg = MLConfig()
    assert cfg.train_lower_bound_date is None


def test_field_accepts_iso_date_string() -> None:
    cfg = MLConfig(train_lower_bound_date="2014-01-01")
    assert cfg.train_lower_bound_date == "2014-01-01"


def test_field_is_immutable() -> None:
    cfg = MLConfig(train_lower_bound_date="2014-01-01")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.train_lower_bound_date = "2015-01-01"  # type: ignore[misc]


def test_existing_fields_unchanged() -> None:
    cfg = MLConfig()
    assert cfg.cutoff_date == "2023-01-01"
    assert cfg.n_optuna_trials == 50
    assert cfg.cv_splits == 5
    assert cfg.random_seed == 42
    assert cfg.model_dir == "models"

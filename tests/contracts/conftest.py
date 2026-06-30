"""Shared fixtures for Phase 25 contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = REPO_ROOT / "src" / "ufc_prediction" / "contracts" / "predictor.schema.v1.0.0.json"


@pytest.fixture(scope="module")
def schema_validator() -> Draft202012Validator:
    """Validator for the COMMITTED predictor.schema.v1.0.0.json file."""
    schema = json.loads(SCHEMA_PATH.read_text())
    return Draft202012Validator(schema)


@pytest.fixture(scope="session")
def xgb_v2_only_mock() -> dict:
    """Current production response — meta skipped (no meta artifact)."""
    return {
        "schema_version": "1.0.0",
        "win_probability": 0.6234,
        "fighter_a": "Khabib Nurmagomedov",
        "fighter_b": "Conor McGregor",
        "event_date": "2018-10-06",
        "base_prob": 0.6234,
        "meta_prob": None,
        "meta_learner_version": None,
        "meta_skipped_reason": "no_meta_artifact",
    }


@pytest.fixture(scope="session")
def meta_v22_active_mock() -> dict:
    """Forward-compat scenario per CONTEXT D-09 (Phase 25 lock-time mock).

    This is the LOCKED contract that Phase 26 MUST emit field-shapes
    compatible with. If Phase 26 changes the META response shape and
    THIS dict fails to validate, that is the regression Position 5a exists
    to surface.
    """
    return {
        "schema_version": "1.0.0",
        "win_probability": 0.7234,
        "fighter_a": "Islam Makhachev",
        "fighter_b": "Charles Oliveira",
        "event_date": "2026-08-15",
        "base_prob": 0.6512,
        "meta_prob": 0.7234,
        "meta_learner_version": "v22.1",
        "meta_skipped_reason": None,
    }

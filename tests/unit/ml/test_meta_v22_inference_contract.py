"""Regression guard for review CRITICAL #1 (+ coverage gap #2).

The promoted META-V22 stacker (`models/meta/meta_v2.joblib`) is a 13-feature
model. Before the fix, `predictor.predict()` fed it a 3-wide vector
`[prob_a, elo_prob_a, closing_prob_diff]`, which raised
`ValueError: X has 3 features, but ... expecting 13` on every odds-present
prediction — and CI stayed green because the only real-artifact meta test was
`@pytest.mark.slow` (excluded from CI) while the unit tests mocked the model.

These tests are DB-free and NON-slow so CI actually runs them. They pin the
13-vs-3 contract at the artifact + assembler boundary that the bug violated.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from ufc_prediction.ml.config import FEATURE_COLUMNS_V22
from ufc_prediction.ml.meta_features_v22 import (
    META_V22_FEATURE_COLUMNS,
    build_meta_features_v22,
)

META_PATH = Path("models/meta/meta_v2.joblib")

pytestmark = pytest.mark.skipif(
    not META_PATH.exists(), reason="promoted meta_v2.joblib not present in this checkout"
)


def _finite_x_v22() -> np.ndarray:
    """A finite (1, 90) FEATURE_COLUMNS_V22 row (values don't matter, only shape)."""
    return (np.arange(len(FEATURE_COLUMNS_V22), dtype=float) * 0.01 + 0.1).reshape(1, -1)


def test_meta_v22_declares_13_level1_columns() -> None:
    assert len(META_V22_FEATURE_COLUMNS) == 13


def test_build_meta_features_v22_produces_13_wide() -> None:
    meta_input = build_meta_features_v22(np.array([0.6]), np.array([0.55]), _finite_x_v22())
    assert meta_input.shape == (1, 13)
    assert np.isfinite(meta_input).all()


def test_promoted_meta_accepts_13_wide_input() -> None:
    """The fix: the 13-wide assembled vector yields a valid probability."""
    meta = joblib.load(META_PATH)
    meta_input = build_meta_features_v22(np.array([0.6]), np.array([0.55]), _finite_x_v22())
    proba = meta.predict_proba(meta_input)
    assert proba.shape == (1, 2)
    assert 0.0 <= float(proba[0, 1]) <= 1.0


def test_promoted_meta_rejects_3_wide_input() -> None:
    """The bug: the old 3-wide input raises. Documents what predictor.py:384 did."""
    meta = joblib.load(META_PATH)
    with pytest.raises(ValueError, match=r"3 features|expecting 13"):
        meta.predict_proba(np.zeros((1, 3)))

"""Phase 19 META-01 Wave-0 RED — meta_learner.py implements MetaLearnerLogistic.

Per CONTEXT.md D-02(P19): minimal 3-feature set + PolynomialFeatures interactions.
Pipeline shape: PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                 → StandardScaler → LogisticRegression(C=1.0, penalty='l2', solver='lbfgs').
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

meta_learner = pytest.importorskip("ufc_prediction.ml.meta_learner")


@pytest.fixture
def synthetic_meta_inputs():
    rng = np.random.default_rng(42)
    n = 200
    X_meta = rng.uniform(0.1, 0.9, size=(n, 3))
    y = rng.integers(0, 2, size=n)
    return X_meta, y


def test_pipeline_shape():
    """D-02(P19): named_steps keys == ['poly', 'scaler', 'clf']."""
    m = meta_learner.MetaLearnerLogistic()
    assert list(m.pipeline.named_steps.keys()) == ["poly", "scaler", "clf"]


def test_polynomial_features_expansion(synthetic_meta_inputs):
    """D-02(P19): 3 inputs → 6 columns after fit (interaction_only=True, no bias)."""
    X_meta, y = synthetic_meta_inputs
    m = meta_learner.MetaLearnerLogistic().fit(X_meta, y)
    assert m.pipeline.named_steps["poly"].n_output_features_ == 6


def test_fit_predict_shape(synthetic_meta_inputs):
    """sklearn contract: predict_proba shape == (n, 2)."""
    X_meta, y = synthetic_meta_inputs
    m = meta_learner.MetaLearnerLogistic().fit(X_meta, y)
    proba = m.predict_proba(X_meta)
    assert proba.shape == (X_meta.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_nan_drop_warns(caplog, synthetic_meta_inputs):
    """NaN tolerance: drops NaN rows; warns when drop rate > 10%."""
    X_meta, y = synthetic_meta_inputs
    # Inject NaN into 30% of rows
    nan_mask = np.zeros(len(X_meta), dtype=bool)
    nan_mask[: int(0.3 * len(X_meta))] = True
    X_meta_nan = X_meta.copy()
    X_meta_nan[nan_mask, 0] = np.nan
    m = meta_learner.MetaLearnerLogistic()
    with caplog.at_level(logging.WARNING, logger="ufc_prediction.ml.meta_learner"):
        m.fit(X_meta_nan, y)
    assert any("dropping" in rec.message for rec in caplog.records), (
        "Expected log warning about NaN drop"
    )


def test_meta_feature_columns_constant():
    """D-02(P19): META_FEATURE_COLUMNS literal."""
    assert meta_learner.META_FEATURE_COLUMNS == [
        "xgb_oof_prob",
        "elo_prob",
        "closing_prob_diff",
    ]


def test_build_meta_features_helper():
    """build_meta_features stacks (n,3) array."""
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.4, 0.5, 0.6])
    c = np.array([0.7, 0.8, 0.9])
    out = meta_learner.build_meta_features(a, b, c)
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out[:, 0], a)
    np.testing.assert_array_equal(out[:, 1], b)
    np.testing.assert_array_equal(out[:, 2], c)

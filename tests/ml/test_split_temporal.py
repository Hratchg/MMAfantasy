"""RED tests for split_temporal 3-bucket extension (Phase 15.1 D-05/D-06/D-07).

Per Pitfall 8 (CONTEXT regression budget), kept in a NEW file separate from
tests/ml/test_feature_matrix.py to avoid the 9 pre-existing elo_momentum_diff
fixture failures. Tests use NumPy literals (no fixture dependency).
"""

from __future__ import annotations

from datetime import date

import numpy as np

from ufc_prediction.ml.feature_matrix import split_temporal


def _make_inputs():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 1, 0, 1, 0])
    dates = np.array(
        [date(2010, 1, 1), date(2015, 1, 1), date(2020, 1, 1), date(2023, 6, 1), date(2024, 6, 1)],
        dtype=object,
    )
    return X, y, dates


def test_default_unchanged_when_train_lower_none() -> None:
    X, y, dates = _make_inputs()
    X_tr, X_te, y_tr, y_te = split_temporal(X, y, dates, cutoff_date=date(2023, 1, 1))
    # All 5 rows partitioned: 3 train (pre-2023), 2 test (>= 2023)
    assert X_tr.shape[0] == 3
    assert X_te.shape[0] == 2
    assert len(X_tr) + len(X_te) == len(X)


def test_train_lower_drops_pre_lower_rows() -> None:
    X, y, dates = _make_inputs()
    X_tr, X_te, y_tr, y_te = split_temporal(
        X,
        y,
        dates,
        cutoff_date=date(2023, 1, 1),
        train_lower=date(2014, 1, 1),
    )
    # 2010 DROPPED; 2015+2020 -> train; 2023+2024 -> test
    assert X_tr.shape[0] == 2
    assert X_te.shape[0] == 2
    # Pitfall 2 invariant: dropped rows are NEITHER in train NOR test
    assert len(X_tr) + len(X_te) < len(X)


def test_test_fold_unchanged_by_train_lower() -> None:
    """Pitfall 5: train_lower must ONLY mask train; test fold unchanged."""
    X, y, dates = _make_inputs()
    _, X_te_no_lower, _, _ = split_temporal(X, y, dates, cutoff_date=date(2023, 1, 1))
    _, X_te_with_lower, _, _ = split_temporal(
        X,
        y,
        dates,
        cutoff_date=date(2023, 1, 1),
        train_lower=date(2014, 1, 1),
    )
    assert X_te_no_lower.shape == X_te_with_lower.shape


def test_train_lower_equals_cutoff_drops_all_train() -> None:
    X, y, dates = _make_inputs()
    X_tr, X_te, _, _ = split_temporal(
        X,
        y,
        dates,
        cutoff_date=date(2023, 1, 1),
        train_lower=date(2023, 1, 1),
    )
    assert X_tr.shape[0] == 0
    assert X_te.shape[0] == 2  # 2023 + 2024


def test_train_lower_after_cutoff_yields_empty_train() -> None:
    X, y, dates = _make_inputs()
    X_tr, _, _, _ = split_temporal(
        X,
        y,
        dates,
        cutoff_date=date(2023, 1, 1),
        train_lower=date(2024, 1, 1),
    )
    assert X_tr.shape[0] == 0

"""Unit tests for `_reevaluate_v1_baseline` (Phase 15.1, D-11(P15.1)).

Per CONTEXT D-11(P15.1): xgb_v1 is re-evaluated on the existing test fold to
provide an anchor for the Path B relaxed-gate derivation (D-09(P15.1)/D-10(P15.1)).
The helper is READ-ONLY: it MUST never call save_model — D-09(P15) byte-identity
of xgb_v1 is enforced at the test level here, in addition to the SHA audits in
the Plan-02 RUN-LOG.

Module-path mock pattern (Phase 15-02 established / memory observation 712):
patch `ufc_prediction.cli.predict.<symbol>` (NOT `ufc_prediction.ml.persistence.*`
or `ufc_prediction.ml.evaluator.*`). Patches must shadow the names as the SUT
(`predict.py`) sees them after its own `from ... import ...`.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from ufc_prediction.cli.predict import _reevaluate_v1_baseline  # NEW in Task 3


X_test_dummy = np.array([[1.0] * 72] * 5)  # 5 fights x 72 features (FEATURE_COLUMNS)
y_test_dummy = np.array([0, 1, 0, 1, 0])


# ── Test 1: returns metrics dict with required keys ─────────────────────────


@patch("ufc_prediction.cli.predict.evaluate_model")
@patch("ufc_prediction.cli.predict.load_model")
@patch("ufc_prediction.cli.predict.save_model")
def test_reevaluate_v1_returns_metrics_dict(
    mock_save_model: MagicMock,
    mock_load_model: MagicMock,
    mock_evaluate_model: MagicMock,
) -> None:
    """The helper returns a dict with at least 'brier_score' and 'accuracy' keys."""
    mock_load_model.return_value = MagicMock(name="v1_model")
    mock_evaluate_model.return_value = {
        "brier_score": 0.2302,
        "accuracy": 0.6190,
        "auc": 0.6624,
    }

    result = _reevaluate_v1_baseline(X_test_dummy, y_test_dummy)

    assert isinstance(result, dict)
    assert "brier_score" in result
    assert "accuracy" in result
    assert isinstance(result["brier_score"], float)
    assert isinstance(result["accuracy"], float)


# ── Test 2: D-09(P15) invariant — never calls save_model ────────────────────


@patch("ufc_prediction.cli.predict.evaluate_model")
@patch("ufc_prediction.cli.predict.load_model")
@patch("ufc_prediction.cli.predict.save_model")
def test_reevaluate_v1_never_calls_save_model(
    mock_save_model: MagicMock,
    mock_load_model: MagicMock,
    mock_evaluate_model: MagicMock,
) -> None:
    """D-09(P15): re-evaluation MUST NOT touch xgb_v1 on disk."""
    mock_load_model.return_value = MagicMock(name="v1_model")
    mock_evaluate_model.return_value = {"brier_score": 0.23, "accuracy": 0.62}

    _reevaluate_v1_baseline(X_test_dummy, y_test_dummy)

    assert mock_save_model.call_count == 0, (
        "save_model was called during _reevaluate_v1_baseline — "
        "D-09(P15) byte-identity invariant violated"
    )


# ── Test 3: only loads v1, never v2 ─────────────────────────────────────────


@patch("ufc_prediction.cli.predict.evaluate_model")
@patch("ufc_prediction.cli.predict.load_model")
@patch("ufc_prediction.cli.predict.save_model")
def test_reevaluate_v1_loads_v1_only(
    mock_save_model: MagicMock,
    mock_load_model: MagicMock,
    mock_evaluate_model: MagicMock,
) -> None:
    """The helper loads xgb_v1 specifically — never v2 (which doesn't anchor anything)."""
    mock_load_model.return_value = MagicMock(name="v1_model")
    mock_evaluate_model.return_value = {"brier_score": 0.23, "accuracy": 0.62}

    _reevaluate_v1_baseline(X_test_dummy, y_test_dummy)

    assert mock_load_model.call_count >= 1, "load_model was never called"

    saw_v1 = False
    for call in mock_load_model.call_args_list:
        args, kwargs = call.args, call.kwargs
        # version may be positional (args[1]) or kwarg ("version")
        version = None
        if len(args) >= 2:
            version = args[1]
        if "version" in kwargs:
            version = kwargs["version"]
        assert version != "v2", (
            f"_reevaluate_v1_baseline called load_model with version='v2': {call}"
        )
        if version == "v1":
            saw_v1 = True

    assert saw_v1, (
        "_reevaluate_v1_baseline never called load_model with version='v1' — "
        "D-11(P15.1) anchor measurement requires v1 specifically"
    )


# ── Test 4: passes v1 model to evaluate_model ───────────────────────────────


@patch("ufc_prediction.cli.predict.evaluate_model")
@patch("ufc_prediction.cli.predict.load_model")
@patch("ufc_prediction.cli.predict.save_model")
def test_reevaluate_v1_calls_evaluate_model_with_v1_model(
    mock_save_model: MagicMock,
    mock_load_model: MagicMock,
    mock_evaluate_model: MagicMock,
) -> None:
    """The model passed to evaluate_model is exactly the one returned by load_model."""
    sentinel_model = MagicMock(name="v1_model_sentinel")
    mock_load_model.return_value = sentinel_model
    mock_evaluate_model.return_value = {"brier_score": 0.23, "accuracy": 0.62}

    _reevaluate_v1_baseline(X_test_dummy, y_test_dummy)

    assert mock_evaluate_model.call_count >= 1, "evaluate_model was never called"
    first_arg = mock_evaluate_model.call_args.args[0]
    assert first_arg is sentinel_model, (
        f"evaluate_model first positional arg ({first_arg!r}) is not the "
        f"v1 model returned by load_model ({sentinel_model!r})"
    )

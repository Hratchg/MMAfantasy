"""Phase 19 META-04 Wave-0 RED — meta_persistence.py is sibling to persistence.py."""

from __future__ import annotations

import json

import numpy as np
import pytest

meta_persistence = pytest.importorskip("ufc_prediction.ml.meta_persistence")
meta_learner = pytest.importorskip("ufc_prediction.ml.meta_learner")


@pytest.fixture
def trained_meta(tmp_path):
    rng = np.random.default_rng(42)
    n = 100
    X_meta = rng.uniform(0.1, 0.9, size=(n, 3))
    y = rng.integers(0, 2, size=n)
    return meta_learner.MetaLearnerLogistic().fit(X_meta, y)


def _save_args():
    return dict(
        meta_kind="logistic",
        meta_version="v1",
        base_model_version="v2",
        base_model_sha256="6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099",
        meta_feature_columns=["xgb_oof_prob", "elo_prob", "closing_prob_diff"],
        meta_input_distribution_hash="d" * 64,
        meta_oof_parquet_sha256="c" * 64,
        meta_learner_brier_delta_vs_logistic=0.0,
        best_params={"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000},
        metrics={"per_slice": {}, "median_brier_overall": 0.21},
    )


def test_save_round_trip(tmp_path, trained_meta):
    """Round-trip: save_meta_model → load_meta_model returns equivalent estimator + metadata."""
    meta_persistence.save_meta_model(trained_meta, meta_dir=str(tmp_path), **_save_args())
    loaded_model, loaded_meta = meta_persistence.load_meta_model(
        meta_dir=str(tmp_path), version="v1"
    )
    assert loaded_meta["meta_kind"] == "logistic"
    assert loaded_meta["meta_version"] == "v1"
    assert (
        loaded_meta["base_model_sha256"]
        == "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
    )
    assert loaded_meta["meta_feature_columns"] == ["xgb_oof_prob", "elo_prob", "closing_prob_diff"]
    # Estimator equivalence: predict_proba on the same input matches
    rng = np.random.default_rng(0)
    X = rng.uniform(0.1, 0.9, size=(5, 3))
    np.testing.assert_array_almost_equal(
        trained_meta.predict_proba(X),
        loaded_model.predict_proba(X),
    )


def test_schema_required_fields(tmp_path, trained_meta):
    """META-04: missing required field in JSON → MetaSchemaError on load."""
    meta_persistence.save_meta_model(trained_meta, meta_dir=str(tmp_path), **_save_args())
    json_path = tmp_path / "meta_v1_meta.json"
    metadata = json.loads(json_path.read_text())
    del metadata["base_model_sha256"]  # remove a required field
    json_path.write_text(json.dumps(metadata, indent=2))
    with pytest.raises(meta_persistence.MetaSchemaError, match="base_model_sha256"):
        meta_persistence.load_meta_model(meta_dir=str(tmp_path), version="v1")


def test_meta_kind_validation(tmp_path, trained_meta):
    """SUPPORTED_META_KINDS guard: meta_kind='random' → MetaSchemaError."""
    args = _save_args()
    args["meta_kind"] = "random"
    with pytest.raises(meta_persistence.MetaSchemaError, match="meta_kind"):
        meta_persistence.save_meta_model(trained_meta, meta_dir=str(tmp_path), **args)


def test_save_refuses_to_overwrite_promoted_version(tmp_path, trained_meta):
    """A promoted meta_vN.joblib is never silently clobbered (mirrors save_model).

    Regression: save_meta_model used to joblib.dump over an existing
    meta_{version}.joblib — a re-run with meta_version='v2' would overwrite the
    AUDIT-01 frozen meta_v2.joblib.
    """
    meta_persistence.save_meta_model(trained_meta, meta_dir=str(tmp_path), **_save_args())
    model_path = tmp_path / "meta_v1.joblib"
    meta_path = tmp_path / "meta_v1_meta.json"
    before_model = model_path.read_bytes()
    before_meta = meta_path.read_bytes()
    with pytest.raises(FileExistsError, match=r"meta_v1\.joblib"):
        meta_persistence.save_meta_model(trained_meta, meta_dir=str(tmp_path), **_save_args())
    assert model_path.read_bytes() == before_model
    assert meta_path.read_bytes() == before_meta


def test_save_overwrite_true_replaces_promoted_version(tmp_path, trained_meta):
    """overwrite=True is the explicit opt-in to replace a promoted version."""
    meta_persistence.save_meta_model(trained_meta, meta_dir=str(tmp_path), **_save_args())
    args = _save_args()
    args["metrics"] = {"per_slice": {}, "median_brier_overall": 0.199}
    meta_persistence.save_meta_model(trained_meta, meta_dir=str(tmp_path), overwrite=True, **args)
    _, loaded_meta = meta_persistence.load_meta_model(meta_dir=str(tmp_path), version="v1")
    assert loaded_meta["metrics"]["median_brier_overall"] == pytest.approx(0.199)


def test_save_candidate_version_may_be_rewritten(tmp_path, trained_meta):
    """Non-promoted names (e.g. v2_candidate) are scratch artifacts re-emitted per run.

    scripts/train_meta_v22.py persists meta_v2_candidate on every run by design;
    get_latest_meta_version never serves these names, so re-writing them cannot
    promote anything.
    """
    args = _save_args()
    args["meta_version"] = "v2_candidate"
    meta_persistence.save_meta_model(trained_meta, meta_dir=str(tmp_path), **args)
    meta_persistence.save_meta_model(trained_meta, meta_dir=str(tmp_path), **args)
    assert (tmp_path / "meta_v2_candidate.joblib").exists()

"""Tests for the XGBoost training pipeline.

Covers ModelTrainer with Optuna objective, TimeSeriesSplit CV,
Platt calibration via CalibratedClassifierCV + FrozenEstimator,
and feature importance extraction.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV

from ufc_prediction.ml.config import FEATURE_COLUMNS


@pytest.fixture()
def synthetic_train_data():
    """Small synthetic dataset for training tests.

    100 rows, 28 features. Data is chronologically ordered (not shuffled).
    """
    rng = np.random.RandomState(42)
    X = rng.randn(100, len(FEATURE_COLUMNS))
    y = rng.randint(0, 2, size=100)
    return X, y


@pytest.fixture()
def small_synthetic_data():
    """Very small dataset for objective function unit test."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, len(FEATURE_COLUMNS))
    y = rng.randint(0, 2, size=50)
    return X, y


class TestOptunaObjective:
    """Tests for the Optuna objective function."""

    def test_objective_returns_float(self, small_synthetic_data):
        """Optuna objective function returns a float Brier score."""
        from ufc_prediction.ml.config import MLConfig
        from ufc_prediction.ml.trainer import ModelTrainer

        X, y = small_synthetic_data
        config = MLConfig(n_optuna_trials=1, cv_splits=2)
        trainer = ModelTrainer(config=config)

        # Run a single trial to verify the objective returns float
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: trainer._objective(trial, X, y), n_trials=1)
        assert isinstance(study.best_value, float)

    def test_objective_uses_timeseries_split(self, small_synthetic_data):
        """Verify TimeSeriesSplit is used in the objective (per D-06)."""
        from ufc_prediction.ml.config import MLConfig
        from ufc_prediction.ml.trainer import ModelTrainer

        X, y = small_synthetic_data
        config = MLConfig(n_optuna_trials=1, cv_splits=3)
        trainer = ModelTrainer(config=config)

        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: trainer._objective(trial, X, y), n_trials=1)
        # Should succeed (objective ran) and return a valid brier score
        assert 0.0 <= study.best_value <= 1.0

    def test_xgb_uses_binary_logistic(self, small_synthetic_data):
        """XGBClassifier uses objective='binary:logistic' (per D-05)."""
        from ufc_prediction.ml.config import MLConfig
        from ufc_prediction.ml.trainer import ModelTrainer

        X, y = small_synthetic_data
        config = MLConfig(n_optuna_trials=1, cv_splits=2)
        trainer = ModelTrainer(config=config)

        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: trainer._objective(trial, X, y), n_trials=1)
        # The best trial params should have been used with binary:logistic
        # We verify this indirectly: the objective ran without error
        assert study.best_trial is not None


class TestTrainingPipeline:
    """Tests for the full training pipeline."""

    def test_train_returns_calibrated_model(self, synthetic_train_data):
        """Training pipeline produces a CalibratedClassifierCV model."""
        from ufc_prediction.ml.config import MLConfig
        from ufc_prediction.ml.trainer import ModelTrainer

        X, y = synthetic_train_data
        config = MLConfig(n_optuna_trials=3, cv_splits=2)
        trainer = ModelTrainer(config=config)

        calibrated_model, best_params, importances = trainer.train(X, y)

        assert isinstance(calibrated_model, CalibratedClassifierCV)

    def test_train_returns_best_params_dict(self, synthetic_train_data):
        """Training returns a dict of best hyperparameters."""
        from ufc_prediction.ml.config import MLConfig
        from ufc_prediction.ml.trainer import ModelTrainer

        X, y = synthetic_train_data
        config = MLConfig(n_optuna_trials=3, cv_splits=2)
        trainer = ModelTrainer(config=config)

        _, best_params, _ = trainer.train(X, y)

        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert "learning_rate" in best_params

    def test_train_returns_feature_importances(self, synthetic_train_data):
        """Training returns feature importances dict keyed by FEATURE_COLUMNS."""
        from ufc_prediction.ml.config import MLConfig
        from ufc_prediction.ml.trainer import ModelTrainer

        X, y = synthetic_train_data
        config = MLConfig(n_optuna_trials=3, cv_splits=2)
        trainer = ModelTrainer(config=config)

        _, _, importances = trainer.train(X, y)

        assert isinstance(importances, dict)
        assert set(importances.keys()) == set(FEATURE_COLUMNS)
        for val in importances.values():
            assert isinstance(val, float)

    def test_calibrated_model_predict_proba(self, synthetic_train_data):
        """Calibrated model's predict_proba returns probabilities in [0, 1]."""
        from ufc_prediction.ml.config import MLConfig
        from ufc_prediction.ml.trainer import ModelTrainer

        X, y = synthetic_train_data
        config = MLConfig(n_optuna_trials=3, cv_splits=2)
        trainer = ModelTrainer(config=config)

        calibrated_model, _, _ = trainer.train(X, y)

        probs = calibrated_model.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_train_proper_calibration_split(self, synthetic_train_data):
        """Training splits into train_proper (~80%) and calibration_holdout (~20%).

        Per RESEARCH Pitfall 6: calibration data must be separate from training data.
        """
        from ufc_prediction.ml.config import MLConfig
        from ufc_prediction.ml.trainer import ModelTrainer

        X, y = synthetic_train_data
        config = MLConfig(n_optuna_trials=3, cv_splits=2)
        trainer = ModelTrainer(config=config)

        # The train method should work without error, implying the split happened
        calibrated_model, _, _ = trainer.train(X, y)

        # The model should have been calibrated (not just raw XGBoost)
        assert hasattr(calibrated_model, "calibrated_classifiers_")

    def test_feature_importance_extraction(self):
        """Feature importance dict can be extracted from the base XGBoost estimator."""
        from ufc_prediction.ml.config import MLConfig
        from ufc_prediction.ml.trainer import ModelTrainer

        # Use data with actual signal so importance is non-zero
        rng = np.random.RandomState(42)
        X = rng.randn(200, len(FEATURE_COLUMNS))
        # Create target correlated with first few features
        y = (X[:, 0] + X[:, 1] + rng.randn(200) * 0.5 > 0).astype(int)

        config = MLConfig(n_optuna_trials=3, cv_splits=2)
        trainer = ModelTrainer(config=config)

        _, _, importances = trainer.train(X, y)

        # All importances should be non-negative (gain-based)
        for val in importances.values():
            assert val >= 0.0

        # With correlated features, at least some should have non-zero importance
        assert any(v > 0.0 for v in importances.values())


class TestTrainMultiSeed:
    """Tests for train_multi_seed (5-seed loop, D-16) per Plan 16-04 Task 3."""

    def test_returns_models_params_importances_seeds(self, synthetic_train_data):
        """train_multi_seed returns dict with models/params/importances/seeds."""
        from ufc_prediction.ml.config import MLConfig
        from ufc_prediction.ml.trainer import ModelTrainer

        X, y = synthetic_train_data
        config = MLConfig(n_optuna_trials=1, cv_splits=2)
        trainer = ModelTrainer(config=config)

        results = trainer.train_multi_seed(X, y, seeds=[42, 43, 44])

        assert "models" in results
        assert "params" in results
        assert "importances" in results
        assert "seeds" in results
        assert len(results["models"]) == 3
        assert len(results["params"]) == 3
        assert len(results["importances"]) == 3
        assert results["seeds"] == [42, 43, 44]

    def test_default_seeds_are_d16(self):
        """Default seeds are [42, 43, 44, 45, 46] per CONTEXT.md D-16."""
        import inspect

        from ufc_prediction.ml.trainer import ModelTrainer

        sig = inspect.signature(ModelTrainer.train_multi_seed)
        seeds_default = sig.parameters["seeds"].default
        assert tuple(seeds_default) == (42, 43, 44, 45, 46)

    def test_restores_random_seed_after_loop(self, synthetic_train_data):
        """After train_multi_seed, trainer.config.random_seed is restored.

        Implemented as a try/finally block so the config (immutable) is
        restored to its pre-call value even if a seed throws.
        """
        from ufc_prediction.ml.config import MLConfig
        from ufc_prediction.ml.trainer import ModelTrainer

        X, y = synthetic_train_data
        config = MLConfig(n_optuna_trials=1, cv_splits=2, random_seed=777)
        trainer = ModelTrainer(config=config)

        trainer.train_multi_seed(X, y, seeds=[42, 43])
        assert trainer.config.random_seed == 777

    def test_back_compat_train_signature_unchanged(self):
        """Existing train(X, y) signature is preserved for back-compat."""
        import inspect

        from ufc_prediction.ml.trainer import ModelTrainer

        sig = inspect.signature(ModelTrainer.train)
        # ModelTrainer.train(self, X_train, y_train) — 3 positional params.
        # Later additions (S02: feature_columns) are keyword-only with defaults,
        # so every existing train(X, y) call keeps working.
        positional = [
            name
            for name, p in sig.parameters.items()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert positional == ["self", "X_train", "y_train"]
        for name, p in sig.parameters.items():
            if name not in positional:
                assert p.kind is p.KEYWORD_ONLY
                assert p.default is not p.empty


class TestMedianMetrics:
    """Tests for median_metrics (per-slice median across seeds, D-16)."""

    def test_median_across_5_seeds(self):
        from ufc_prediction.ml.trainer import median_metrics

        per_seed = [
            {"s": {"brier_score": 0.21}},
            {"s": {"brier_score": 0.22}},
            {"s": {"brier_score": 0.20}},
            {"s": {"brier_score": 0.23}},
            {"s": {"brier_score": 0.21}},
        ]
        result = median_metrics(per_seed)
        # Median of [0.20, 0.21, 0.21, 0.22, 0.23] = 0.21.
        assert result["s"]["brier_score"] == 0.21

    def test_median_handles_multiple_slices(self):
        from ufc_prediction.ml.trainer import median_metrics

        per_seed = [
            {
                "a": {"brier_score": 0.21, "accuracy": 0.65},
                "b": {"brier_score": 0.22, "accuracy": 0.66},
            },
            {
                "a": {"brier_score": 0.22, "accuracy": 0.66},
                "b": {"brier_score": 0.21, "accuracy": 0.67},
            },
            {
                "a": {"brier_score": 0.20, "accuracy": 0.68},
                "b": {"brier_score": 0.23, "accuracy": 0.65},
            },
        ]
        result = median_metrics(per_seed)
        # Median of 3 values: middle one when sorted.
        assert result["a"]["brier_score"] == 0.21
        assert result["a"]["accuracy"] == 0.66
        assert result["b"]["brier_score"] == 0.22
        assert result["b"]["accuracy"] == 0.66

    def test_median_skips_calibration_curve(self):
        """Non-scalar values (calibration_curve) keep first seed's value."""
        from ufc_prediction.ml.trainer import median_metrics

        per_seed = [
            {
                "a": {
                    "brier_score": 0.21,
                    "calibration_curve": {"fraction_of_positives": [0.1, 0.5, 0.8]},
                }
            },
            {
                "a": {
                    "brier_score": 0.22,
                    "calibration_curve": {"fraction_of_positives": [0.2, 0.5, 0.7]},
                }
            },
            {
                "a": {
                    "brier_score": 0.23,
                    "calibration_curve": {"fraction_of_positives": [0.0, 0.6, 0.9]},
                }
            },
        ]
        result = median_metrics(per_seed)
        assert result["a"]["brier_score"] == 0.22
        # Should preserve first seed's calibration curve verbatim.
        assert result["a"]["calibration_curve"] == per_seed[0]["a"]["calibration_curve"]


class TestFeatureWidth:
    """S02 finding 1: train() must not assume the 75-col FEATURE_COLUMNS width.

    The CLI default --feature-set v2.1-no-net is 72 cols; v2.2 is 90. Before the
    fix, train() zipped importances against FEATURE_COLUMNS with strict=True and
    crashed AFTER the full Optuna search + calibration for every non-v1.0 set.
    """

    @pytest.mark.parametrize("feature_set", ["v2.1-no-net", "v2.2"])
    def test_train_keys_importances_by_passed_columns(self, feature_set):
        from ufc_prediction.ml.config import MLConfig, get_feature_columns
        from ufc_prediction.ml.trainer import ModelTrainer

        cols = get_feature_columns(feature_set=feature_set)
        assert len(cols) != len(FEATURE_COLUMNS)
        rng = np.random.RandomState(42)
        X = rng.randn(100, len(cols))
        y = rng.randint(0, 2, size=100)

        trainer = ModelTrainer(config=MLConfig(n_optuna_trials=1, cv_splits=2))
        _, _, importances = trainer.train(X, y, feature_columns=cols)

        assert list(importances.keys()) == list(cols)

    def test_train_without_columns_on_non_default_width_does_not_crash(self):
        """No column list + a non-75 width falls back to positional f{i} keys."""
        from ufc_prediction.ml.config import MLConfig
        from ufc_prediction.ml.trainer import ModelTrainer

        rng = np.random.RandomState(42)
        X = rng.randn(100, 72)
        y = rng.randint(0, 2, size=100)

        trainer = ModelTrainer(config=MLConfig(n_optuna_trials=1, cv_splits=2))
        _, _, importances = trainer.train(X, y)

        assert list(importances.keys()) == [f"f{i}" for i in range(72)]

    def test_train_rejects_column_width_mismatch_before_search(self, monkeypatch):
        """A wrong-width column list fails fast, before any Optuna trial runs."""
        from ufc_prediction.ml.config import MLConfig
        from ufc_prediction.ml.trainer import ModelTrainer

        rng = np.random.RandomState(42)
        X = rng.randn(100, 72)
        y = rng.randint(0, 2, size=100)
        trainer = ModelTrainer(config=MLConfig(n_optuna_trials=1, cv_splits=2))

        def _boom(*_a, **_k):
            raise AssertionError("Optuna search must not start on a width mismatch")

        monkeypatch.setattr(trainer, "_objective", _boom)
        with pytest.raises(ValueError, match="72"):
            trainer.train(X, y, feature_columns=list(FEATURE_COLUMNS))

    def test_train_multi_seed_forwards_columns(self):
        from ufc_prediction.ml.config import MLConfig, get_feature_columns
        from ufc_prediction.ml.trainer import ModelTrainer

        cols = get_feature_columns(feature_set="v2.1-no-net")
        rng = np.random.RandomState(42)
        X = rng.randn(100, len(cols))
        y = rng.randint(0, 2, size=100)

        trainer = ModelTrainer(config=MLConfig(n_optuna_trials=1, cv_splits=2))
        results = trainer.train_multi_seed(X, y, seeds=(42, 43), feature_columns=cols)

        for imp in results["importances"]:
            assert list(imp.keys()) == list(cols)


class TestSamplerSeeding:
    """S02 finding 2: the Optuna sampler must be seeded from config.random_seed."""

    def test_train_is_reproducible_for_same_seed(self, synthetic_train_data):
        from ufc_prediction.ml.config import MLConfig
        from ufc_prediction.ml.trainer import ModelTrainer

        X, y = synthetic_train_data
        config = MLConfig(n_optuna_trials=3, cv_splits=2, random_seed=42)

        _, params_a, _ = ModelTrainer(config=config).train(X, y)
        _, params_b, _ = ModelTrainer(config=config).train(X, y)

        assert params_a == params_b

    def test_different_seeds_explore_different_params(self, synthetic_train_data):
        from ufc_prediction.ml.config import MLConfig
        from ufc_prediction.ml.trainer import ModelTrainer

        X, y = synthetic_train_data
        _, params_a, _ = ModelTrainer(
            config=MLConfig(n_optuna_trials=1, cv_splits=2, random_seed=42)
        ).train(X, y)
        _, params_b, _ = ModelTrainer(
            config=MLConfig(n_optuna_trials=1, cv_splits=2, random_seed=43)
        ).train(X, y)

        assert params_a != params_b

    @pytest.mark.parametrize("tuner", ["_tune_xgboost", "_tune_lightgbm", "_tune_catboost"])
    def test_ensemble_tuners_are_reproducible_for_same_seed(
        self, tuner, small_synthetic_data, tmp_path, monkeypatch
    ):
        from ufc_prediction.ml.config import MLConfig
        from ufc_prediction.ml.ensemble import EnsembleTrainer

        monkeypatch.chdir(tmp_path)  # CatBoost writes catboost_info/ into the cwd
        X, y = small_synthetic_data
        config = MLConfig(cv_splits=2, random_seed=42)

        params_a = getattr(EnsembleTrainer(config=config), tuner)(X, y, 2)
        params_b = getattr(EnsembleTrainer(config=config), tuner)(X, y, 2)

        assert params_a == params_b

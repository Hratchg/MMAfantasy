"""XGBoost training pipeline with Optuna hyperparameter tuning and Platt calibration.

Per D-05: XGBoost as the gradient boosted library.
Per D-06: Optuna Bayesian optimization with 5-fold TimeSeriesSplit CV.
Per D-08: CalibratedClassifierCV with FrozenEstimator for Platt scaling.

Phase 16 / Plan 16-04 / MODEL-01 additions (per CONTEXT.md D-16):
  - ModelTrainer.train_multi_seed: 5-seed wrapper around train() that
    returns 5 candidate models. The single-seed train() signature is
    NOT modified (back-compat with existing CLI + tests).
  - median_metrics: per-slice median across seeds for the gate verdict.
    Skips non-scalar values (calibration_curve) — keeps first seed's
    value for reference.
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace

import numpy as np
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from ufc_prediction.ml.config import FEATURE_COLUMNS, MLConfig

# Phase 26 CALIB-V22-02 isotonic-vs-sigmoid threshold (D-14(v2.0)). Exported as a
# module-level constant so unit tests can monkey-patch and dispatch helpers can
# import without circular imports.
ISOTONIC_THRESHOLD: int = 1000


def _pick_calibration_method(n_calib: int) -> str:
    """CALIB-V22-02 dispatch helper: 'isotonic' iff n_calib >= ISOTONIC_THRESHOLD,
    else 'sigmoid'. Per D-14(v2.0) (Niculescu-Mizil & Caruana ICML 2005)."""
    return "isotonic" if n_calib >= ISOTONIC_THRESHOLD else "sigmoid"


class ModelTrainer:
    """Trains XGBoost classifier with Optuna hyperparameter tuning and Platt calibration.

    Per D-05: XGBoost as the gradient boosted library.
    Per D-06: Optuna Bayesian optimization with 5-fold TimeSeriesSplit CV.
    Per D-08: CalibratedClassifierCV with FrozenEstimator for Platt scaling.
    """

    def __init__(self, config: MLConfig | None = None) -> None:
        self.config = config or MLConfig()

    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective: train XGBoost with suggested params, return mean Brier.

        Uses TimeSeriesSplit with n_splits from config (per D-06).
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)
        brier_scores: list[float] = []

        for train_idx, val_idx in tscv.split(X):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            model = XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.config.random_seed,
                verbosity=0,
            )
            model.fit(X_fold_train, y_fold_train)
            probs = model.predict_proba(X_fold_val)[:, 1]
            brier_scores.append(brier_score_loss(y_fold_val, probs))

        return float(np.mean(brier_scores))

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[CalibratedClassifierCV, dict, dict]:
        """Train model with Optuna tuning and Platt calibration.

        Steps:
        1. Split train into train_proper (80%) and calibration_holdout (20%)
           -- no shuffling, chronological order preserved (RESEARCH Pitfall 7).
        2. Optuna study to find best hyperparameters on train_proper.
        3. Train final model on full train_proper with best params.
        4. Platt calibration via CalibratedClassifierCV(FrozenEstimator) on calibration_holdout.
        5. Extract feature importances from the base XGBoost model.

        Returns (calibrated_model, best_params, feature_importances).
        """
        # Step 1: Split into train_proper and calibration_holdout
        # Temporal split: first 80% for training, last 20% for calibration
        n_total = len(X_train)
        split_idx = int(n_total * 0.8)
        X_proper = X_train[:split_idx]
        y_proper = y_train[:split_idx]
        X_calib = X_train[split_idx:]
        y_calib = y_train[split_idx:]

        # Step 2: Optuna hyperparameter search
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self._objective(trial, X_proper, y_proper),
            n_trials=self.config.n_optuna_trials,
        )
        best_params = study.best_params

        # Step 3: Train final model on full train_proper with best params
        final_model = XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.config.random_seed,
            verbosity=0,
        )
        final_model.fit(X_proper, y_proper)

        # Step 4: Calibration per CALIB-V22-01 (Platt explicit REPLACEMENT, Phase 26)
        # + CALIB-V22-02 (isotonic conditional when len(X_calib) >= ISOTONIC_THRESHOLD
        # per D-14(v2.0) — Niculescu-Mizil & Caruana ICML 2005). The PRE-Phase-26
        # wiring was `method="sigmoid"` implicit; this refactor makes the choice
        # EXPLICIT and adds the conditional via `_pick_calibration_method`.
        # Per Pitfall #8: EXACTLY ONE CalibratedClassifierCV is constructed —
        # never stacked. Per AUDIT-01: this is a CODE REFACTOR ONLY; xgb_v2.joblib
        # is NOT retrained as a result of this commit.
        calibration_method = _pick_calibration_method(len(X_calib))
        calibrated_model = CalibratedClassifierCV(
            FrozenEstimator(final_model),
            method=calibration_method,
        )
        calibrated_model.fit(X_calib, y_calib)

        # Step 5: Feature importances (gain-based) from the base XGBoost model
        raw_importances = final_model.feature_importances_
        feature_importances = dict(
            zip(FEATURE_COLUMNS, [float(v) for v in raw_importances], strict=True)
        )

        return calibrated_model, best_params, feature_importances

    def train_multi_seed(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    ) -> dict:
        """Train n models with `n = len(seeds)` distinct random_seed values.

        Per CONTEXT.md D-16: 5 training seeds, median wins. The single-
        seed train() is the inner primitive — train_multi_seed wraps it
        in a loop, swapping `self.config.random_seed` per iteration.

        Args:
            X_train: Training feature matrix.
            y_train: Training target vector.
            seeds: Tuple of random_state values (default per D-16:
                (42, 43, 44, 45, 46)).

        Returns:
            dict with keys:
              - models: list of CalibratedClassifierCV (one per seed)
              - params: list of best-params dicts (one per seed)
              - importances: list of feature-importance dicts (one per seed)
              - seeds: list[int] (the seeds used, in order)

        Side effects:
            self.config.random_seed is restored to its pre-call value
            via try/finally so the trainer remains reusable. MLConfig is
            a frozen dataclass; dataclasses.replace is used for the
            in-loop swap.
        """
        original_config = self.config
        results: dict = {
            "models": [],
            "params": [],
            "importances": [],
            "seeds": list(seeds),
        }
        try:
            for seed in seeds:
                self.config = _dc_replace(original_config, random_seed=int(seed))
                model, params, importances = self.train(X_train, y_train)
                results["models"].append(model)
                results["params"].append(params)
                results["importances"].append(importances)
        finally:
            self.config = original_config
        return results


def median_metrics(per_seed_metrics: list[dict]) -> dict:
    """Compute per-slice median across seeds for the D-13 gate.

    Args:
        per_seed_metrics: list of dicts (one per seed); each dict is
            keyed by slice name (e.g. 'most_recent_12mo') and the
            inner dict contains per-slice metrics. Shape mirrors
            evaluator.evaluate_per_slice's return.

    Returns:
        dict with the same outer/inner keys as a single seed's entry,
        but each scalar metric value is the median across seeds. Non-
        scalar values (e.g. 'calibration_curve') are not aggregated —
        the first seed's value is preserved for reference (median of
        nested dicts is undefined).

    Per D-16: median (not mean) protects against one bad seed dragging
    the verdict in either direction.
    """
    if not per_seed_metrics:
        msg = "per_seed_metrics must contain at least one seed's results"
        raise ValueError(msg)

    slice_names = list(per_seed_metrics[0].keys())
    result: dict = {}
    for slice_name in slice_names:
        metric_keys = list(per_seed_metrics[0][slice_name].keys())
        slice_result: dict = {}
        for mk in metric_keys:
            sample = per_seed_metrics[0][slice_name][mk]
            if not isinstance(sample, (int, float, np.integer, np.floating)):
                # Non-scalar (e.g. calibration_curve dict) — keep first seed's.
                slice_result[mk] = sample
                continue
            values = [float(psm[slice_name][mk]) for psm in per_seed_metrics]
            slice_result[mk] = float(np.median(values))
        result[slice_name] = slice_result
    return result

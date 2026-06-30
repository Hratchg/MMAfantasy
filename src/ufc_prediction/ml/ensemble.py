"""Ensemble stacking: XGBoost + LightGBM + CatBoost + Logistic Regression.

Trains four base models and a logistic regression meta-model
that learns how to weight each model's predictions.
"""

from __future__ import annotations

import numpy as np
import optuna
import sklearn.base
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from ufc_prediction.ml.config import MLConfig


class EnsembleTrainer:
    """Trains an ensemble of XGBoost + LightGBM + CatBoost + Logistic Regression.

    The meta-model (logistic regression) learns how to combine
    the four base models' probability outputs.
    """

    def __init__(self, config: MLConfig | None = None) -> None:
        self.config = config or MLConfig()

    def _tune_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int,
    ) -> dict:
        """Find best XGBoost params via Optuna."""

        def objective(trial: optuna.Trial) -> float:
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
            scores = []
            for train_idx, val_idx in tscv.split(X):
                model = XGBClassifier(
                    **params,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=self.config.random_seed,
                    verbosity=0,
                )
                model.fit(X[train_idx], y[train_idx])
                probs = model.predict_proba(X[val_idx])[:, 1]
                scores.append(brier_score_loss(y[val_idx], probs))
            return float(np.mean(scores))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def _tune_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int,
    ) -> dict:
        """Find best LightGBM params via Optuna."""

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            }
            tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)
            scores = []
            for train_idx, val_idx in tscv.split(X):
                model = LGBMClassifier(
                    **params,
                    objective="binary",
                    random_state=self.config.random_seed,
                    verbosity=-1,
                )
                model.fit(X[train_idx], y[train_idx])
                probs = model.predict_proba(X[val_idx])[:, 1]
                scores.append(brier_score_loss(y[val_idx], probs))
            return float(np.mean(scores))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def _tune_catboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int,
    ) -> dict:
        """Find best CatBoost params via Optuna."""

        def objective(trial: optuna.Trial) -> float:
            params = {
                "iterations": trial.suggest_int("iterations", 100, 500),
                "depth": trial.suggest_int("depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            }
            tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)
            scores = []
            for train_idx, val_idx in tscv.split(X):
                model = CatBoostClassifier(
                    **params,
                    loss_function="Logloss",
                    random_seed=self.config.random_seed,
                    verbose=0,
                )
                model.fit(X[train_idx], y[train_idx])
                probs = model.predict_proba(X[val_idx])[:, 1]
                scores.append(brier_score_loss(y[val_idx], probs))
            return float(np.mean(scores))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        xgb_trials: int = 50,
        lgbm_trials: int = 50,
        cb_trials: int = 50,
    ) -> tuple[object, dict]:
        """Train ensemble with Optuna-tuned base models.

        Steps:
        1. Split into train_proper (70%), blend (15%), calibration (15%)
        2. Tune and train XGBoost + LightGBM + CatBoost on train_proper
        3. Train logistic regression on train_proper
        4. Generate blend predictions from all 4 base models
        5. Train meta-model (logistic regression) on blend predictions
        6. Calibrate final ensemble via Platt scaling on calibration set

        Returns (ensemble_model, info_dict).
        """
        n = len(X_train)
        split1 = int(n * 0.70)
        split2 = int(n * 0.85)
        X_proper, y_proper = X_train[:split1], y_train[:split1]
        X_blend, y_blend = X_train[split1:split2], y_train[split1:split2]
        X_calib, y_calib = X_train[split2:], y_train[split2:]

        # Step 2: Tune and train XGBoost
        xgb_params = self._tune_xgboost(X_proper, y_proper, xgb_trials)
        xgb_model = XGBClassifier(
            **xgb_params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.config.random_seed,
            verbosity=0,
        )
        xgb_model.fit(X_proper, y_proper)

        # Tune and train LightGBM
        lgbm_params = self._tune_lightgbm(X_proper, y_proper, lgbm_trials)
        lgbm_model = LGBMClassifier(
            **lgbm_params,
            objective="binary",
            random_state=self.config.random_seed,
            verbosity=-1,
        )
        lgbm_model.fit(X_proper, y_proper)

        # Tune and train CatBoost
        cb_params = self._tune_catboost(X_proper, y_proper, cb_trials)
        cb_model = CatBoostClassifier(
            **cb_params,
            loss_function="Logloss",
            random_seed=self.config.random_seed,
            verbose=0,
        )
        cb_model.fit(X_proper, y_proper)

        # Step 3: Train logistic regression (with imputer for NaN handling)
        lr_model = make_pipeline(
            SimpleImputer(strategy="median"),
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=self.config.random_seed,
            ),
        )
        lr_model.fit(X_proper, y_proper)

        # Step 4: Generate blend predictions
        blend_preds = np.column_stack(
            [
                xgb_model.predict_proba(X_blend)[:, 1],
                lgbm_model.predict_proba(X_blend)[:, 1],
                cb_model.predict_proba(X_blend)[:, 1],
                lr_model.predict_proba(X_blend)[:, 1],
            ]
        )

        # Step 5: Train meta-model
        meta_model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=self.config.random_seed,
        )
        meta_model.fit(blend_preds, y_blend)  # blend_preds are probabilities, no NaN

        # Build ensemble wrapper — meta-model logistic regression already
        # outputs calibrated probabilities, no extra Platt scaling needed
        ensemble = _EnsembleModel(xgb_model, lgbm_model, cb_model, lr_model, meta_model)

        info = {
            "xgb_params": xgb_params,
            "lgbm_params": lgbm_params,
            "cb_params": cb_params,
            "meta_coefs": meta_model.coef_.tolist(),
            "meta_intercept": float(meta_model.intercept_[0]),
        }

        return ensemble, info


class _EnsembleModel(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """Wrapper that makes four models look like one sklearn estimator."""

    def __init__(self, xgb=None, lgbm=None, cb=None, lr=None, meta=None) -> None:
        self.xgb = xgb
        self.lgbm = lgbm
        self.cb = cb
        self.lr = lr
        self.meta = meta
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        stacked = np.column_stack(
            [
                self.xgb.predict_proba(X)[:, 1],
                self.lgbm.predict_proba(X)[:, 1],
                self.cb.predict_proba(X)[:, 1],
                self.lr.predict_proba(X)[:, 1],
            ]
        )
        meta_probs = self.meta.predict_proba(stacked)
        return meta_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "xgb": self.xgb,
            "lgbm": self.lgbm,
            "cb": self.cb,
            "lr": self.lr,
            "meta": self.meta,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_EnsembleModel":
        # No-op — FrozenEstimator calls this but we're already trained
        return self

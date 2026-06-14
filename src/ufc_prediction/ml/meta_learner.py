"""Meta-learner blender (META-01 mandatory ship).

Per CONTEXT.md D-02(P19): MINIMAL 3-feature set [xgb_oof_prob, elo_prob,
closing_prob_diff]. PolynomialFeatures(degree=2, interaction_only=True,
include_bias=False) generates 3 pairwise interactions → 6 columns total →
~7 fitted params after intercept.

Pipeline shape:
    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        in:  (n, 3)   columns: [xgb_oof_prob, elo_prob, closing_prob_diff]
        out: (n, 6)   columns: [a, b, c, a*b, a*c, b*c]
    StandardScaler() — zero-mean, unit-variance per column post-interaction
    LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000)

Per RESEARCH.md F3 risks: elo_prob MUST be as-of fight-date Elo, NOT
post-fight Elo. Use EloEngine.expected_win_probability(fa, fb,
as_of_date=fight.event_date).
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

logger = logging.getLogger(__name__)

META_FEATURE_COLUMNS: list[str] = ["xgb_oof_prob", "elo_prob", "closing_prob_diff"]
NAN_DROP_WARN_FRACTION: float = 0.10  # warn when >10% rows dropped


def build_meta_features(
    xgb_proba: np.ndarray,
    elo_proba: np.ndarray,
    closing_prob_diff: np.ndarray,
) -> np.ndarray:
    """Stack the 3 Level-1 features into shape (n, 3). NaN-safe (preserves NaNs)."""
    return np.column_stack([xgb_proba, elo_proba, closing_prob_diff])


class MetaLearnerLogistic:
    """sklearn Pipeline wrapping PolynomialFeatures × StandardScaler × LogisticRegression.

    Per D-02(P19): minimal 3-feature set + interaction expansion.
    """

    def __init__(self, *, random_state: int = 42, C: float = 1.0):
        self.pipeline: Pipeline = Pipeline([
            ("poly", PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False,
            )),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C, penalty="l2", solver="lbfgs", max_iter=1000,
                random_state=random_state,
            )),
        ])
        self.random_state: int = random_state
        self.C: float = C

    def fit(self, X_meta: np.ndarray, y: np.ndarray) -> "MetaLearnerLogistic":
        """Fit pipeline; drop rows with NaN in any Level-1 feature.

        Warns (logger.warning) when NaN drop rate exceeds NAN_DROP_WARN_FRACTION (10%).
        """
        mask = ~np.isnan(X_meta).any(axis=1)
        n_dropped = int((~mask).sum())
        if n_dropped / max(len(mask), 1) > NAN_DROP_WARN_FRACTION:
            logger.warning(
                "MetaLearnerLogistic: dropping %d NaN rows (%.1f%% of total)",
                n_dropped, 100 * n_dropped / max(len(mask), 1),
            )
        self.pipeline.fit(X_meta[mask], y[mask])
        return self

    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        result: np.ndarray = self.pipeline.predict_proba(X_meta)
        return result

    def predict(self, X_meta: np.ndarray) -> np.ndarray:
        result: np.ndarray = self.pipeline.predict(X_meta)
        return result

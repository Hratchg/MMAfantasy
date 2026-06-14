"""Bayesian shrinkage regularization for fighter features.

Implements the same shrinkage pattern as EloEngine._compute_shrinkage (Phase 3):
fighters with fewer than min_fights have their feature values pulled toward
the league-wide mean. Per D-12, D-13.
"""

from __future__ import annotations


def apply_shrinkage(
    value: float,
    league_mean: float,
    fight_count: int,
    min_fights: int = 5,
) -> float:
    """Apply Bayesian shrinkage to a single feature value.

    Per D-12: factor = min(fight_count / min_fights, 1.0).
    Result = league_mean + (value - league_mean) * factor.

    Args:
        value: The raw feature value to shrink.
        league_mean: League-wide mean for this feature.
        fight_count: Number of fights the fighter has.
        min_fights: Minimum fights for full weight (default 5).

    Returns:
        The shrunk feature value.
    """
    factor = min(fight_count / min_fights, 1.0)
    return league_mean + (value - league_mean) * factor


def apply_shrinkage_to_features(
    features: dict[str, float | None],
    league_means: dict[str, float],
    fight_count: int,
    min_fights: int = 5,
) -> dict[str, float | None]:
    """Apply Bayesian shrinkage to all numeric features in a dict.

    For each key in features:
    - If value is None, keep None.
    - If value is not numeric (e.g., string), keep as-is.
    - If key exists in league_means, apply shrinkage.
    - Otherwise keep original value.

    Per D-13, regularization is applied at feature computation time.

    Args:
        features: Dict of feature_name -> value (may contain None or strings).
        league_means: Dict of feature_name -> league-wide mean value.
        fight_count: Number of fights the fighter has.
        min_fights: Minimum fights for full weight (default 5).

    Returns:
        New dict with shrunk values where applicable.
    """
    result: dict[str, float | None] = {}
    for key, value in features.items():
        if value is None or not isinstance(value, (int, float)):
            result[key] = value
        elif key in league_means:
            result[key] = apply_shrinkage(
                value=float(value),
                league_mean=league_means[key],
                fight_count=fight_count,
                min_fights=min_fights,
            )
        else:
            result[key] = value
    return result

"""Offensive-vs-defensive matrices for striking and grappling.

Computes structured matchup data showing each fighter's offensive stats
against the opponent's defensive stats, using EWMA values with career
average fallback (D-05, D-07).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StrikingMatrix:
    """Striking matchup matrix: A's offense vs B's defense and vice versa.

    Fields represent each fighter's key striking metrics.
    """

    a_sig_str_per_min: float | None
    a_striking_accuracy: float | None
    a_strike_defense: float | None
    b_sig_str_per_min: float | None
    b_striking_accuracy: float | None
    b_strike_defense: float | None


@dataclass
class GrapplingMatrix:
    """Grappling matchup matrix: A's offense vs B's defense and vice versa.

    Fields represent each fighter's key grappling metrics plus control time.
    """

    a_td_rate: float | None
    a_td_accuracy: float | None
    a_td_defense: float | None
    b_td_rate: float | None
    b_td_accuracy: float | None
    b_td_defense: float | None
    a_ctrl_time: float | None
    b_ctrl_time: float | None


def _get_with_fallback(
    features: dict[str, float | None] | None,
    ewma_key: str,
    career_key: str,
) -> float | None:
    """Return EWMA value if present, else career value, else None.

    If features dict is None, returns None.
    """
    if features is None:
        return None
    value = features.get(ewma_key)
    if value is not None:
        return value
    return features.get(career_key)


def compute_striking_matrix(
    features_a: dict[str, float | None] | None,
    features_b: dict[str, float | None] | None,
    accuracy_a: float | None = None,
    accuracy_b: float | None = None,
) -> StrikingMatrix:
    """Compute striking matchup matrix from fighter features.

    Args:
        features_a: Fighter A's latest computed features dict, or None.
        features_b: Fighter B's latest computed features dict, or None.
        accuracy_a: Fighter A's striking accuracy from round stats query.
        accuracy_b: Fighter B's striking accuracy from round stats query.

    Returns:
        StrikingMatrix with each fighter's offensive and defensive metrics.
        Uses EWMA with career fallback per D-05.
    """
    return StrikingMatrix(
        a_sig_str_per_min=_get_with_fallback(
            features_a, "sig_str_per_minute_ewma", "sig_str_per_minute"
        ),
        a_striking_accuracy=accuracy_a,
        a_strike_defense=_get_with_fallback(
            features_a, "strike_defense_ewma", "strike_defense"
        ),
        b_sig_str_per_min=_get_with_fallback(
            features_b, "sig_str_per_minute_ewma", "sig_str_per_minute"
        ),
        b_striking_accuracy=accuracy_b,
        b_strike_defense=_get_with_fallback(
            features_b, "strike_defense_ewma", "strike_defense"
        ),
    )


def compute_grappling_matrix(
    features_a: dict[str, float | None] | None,
    features_b: dict[str, float | None] | None,
) -> GrapplingMatrix:
    """Compute grappling matchup matrix from fighter features.

    Args:
        features_a: Fighter A's latest computed features dict, or None.
        features_b: Fighter B's latest computed features dict, or None.

    Returns:
        GrapplingMatrix with each fighter's offensive and defensive metrics.
        Uses EWMA with career fallback per D-07.
    """
    return GrapplingMatrix(
        a_td_rate=_get_with_fallback(features_a, "td_rate_ewma", "td_rate"),
        a_td_accuracy=_get_with_fallback(features_a, "td_accuracy_ewma", "td_accuracy"),
        a_td_defense=_get_with_fallback(features_a, "td_defense_ewma", "td_defense"),
        b_td_rate=_get_with_fallback(features_b, "td_rate_ewma", "td_rate"),
        b_td_accuracy=_get_with_fallback(features_b, "td_accuracy_ewma", "td_accuracy"),
        b_td_defense=_get_with_fallback(features_b, "td_defense_ewma", "td_defense"),
        a_ctrl_time=_get_with_fallback(
            features_a, "ctrl_time_per_fight_ewma", "ctrl_time_per_fight"
        ),
        b_ctrl_time=_get_with_fallback(
            features_b, "ctrl_time_per_fight_ewma", "ctrl_time_per_fight"
        ),
    )

"""Physical matchup differentials computation.

Computes height, reach, leg reach, and age differentials between
two fighters. Returns None for any differential where data is missing
for either fighter (D-02: no imputation).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol


class _FighterLike(Protocol):
    """Structural type for objects with the physical attributes used here."""

    height_inches: float | None
    reach_inches: float | None
    leg_reach_inches: float | None
    date_of_birth: date | None


@dataclass
class PhysicalDifferentials:
    """Physical attribute differentials between Fighter A and Fighter B.

    Positive values mean A has more (taller, longer reach, older).
    None means data was missing for at least one fighter (D-02).
    """

    height_diff: float | None
    reach_diff: float | None
    leg_reach_diff: float | None
    age_diff: float | None
    fighter_a_age: float | None
    fighter_b_age: float | None


def _compute_diff(a_val: float | None, b_val: float | None) -> float | None:
    """Return A - B if both values present, else None."""
    if a_val is not None and b_val is not None:
        return a_val - b_val
    return None


def _compute_age(dob: date | None, reference_date: date) -> float | None:
    """Compute age in years from DOB and reference date."""
    if dob is None:
        return None
    return (reference_date - dob).days / 365.25


def compute_physical_differentials(
    fighter_a: _FighterLike,
    fighter_b: _FighterLike,
    reference_date: date | None = None,
) -> PhysicalDifferentials:
    """Compute physical attribute differentials between two fighters.

    Args:
        fighter_a: Fighter-like object with height_inches, reach_inches,
            leg_reach_inches, date_of_birth attributes.
        fighter_b: Fighter-like object with same attributes.
        reference_date: Date to compute ages from. Defaults to today.

    Returns:
        PhysicalDifferentials with A - B differences. Positive = A has more.
        None for any field where either fighter's data is missing (D-02).
    """
    ref_date = reference_date or date.today()

    height_diff = _compute_diff(fighter_a.height_inches, fighter_b.height_inches)
    reach_diff = _compute_diff(fighter_a.reach_inches, fighter_b.reach_inches)
    leg_reach_diff = _compute_diff(fighter_a.leg_reach_inches, fighter_b.leg_reach_inches)

    fighter_a_age = _compute_age(fighter_a.date_of_birth, ref_date)
    fighter_b_age = _compute_age(fighter_b.date_of_birth, ref_date)
    age_diff = _compute_diff(fighter_a_age, fighter_b_age)

    return PhysicalDifferentials(
        height_diff=height_diff,
        reach_diff=reach_diff,
        leg_reach_diff=leg_reach_diff,
        age_diff=age_diff,
        fighter_a_age=fighter_a_age,
        fighter_b_age=fighter_b_age,
    )

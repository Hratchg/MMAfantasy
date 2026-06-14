"""Unit tests for physical differentials computation."""

from __future__ import annotations

from datetime import date

from ufc_prediction.matchup.physical import (
    PhysicalDifferentials,
    compute_physical_differentials,
)


class TestPhysicalDifferentials:
    """Tests for compute_physical_differentials."""

    def test_both_fighters_all_data(self, make_fighter):
        """Both fighters have all physical data -> returns correct diffs (A - B)."""
        fighter_a = make_fighter(
            height_inches=73.0,
            reach_inches=75.0,
            leg_reach_inches=43.0,
            date_of_birth=date(1990, 6, 15),
        )
        fighter_b = make_fighter(
            height_inches=70.0,
            reach_inches=72.0,
            leg_reach_inches=40.0,
            date_of_birth=date(1988, 3, 10),
        )

        result = compute_physical_differentials(
            fighter_a, fighter_b, reference_date=date(2025, 1, 1)
        )

        assert isinstance(result, PhysicalDifferentials)
        assert result.height_diff == 3.0
        assert result.reach_diff == 3.0
        assert result.leg_reach_diff == 3.0
        # A is younger than B, so age_diff should be negative
        assert result.age_diff is not None
        assert result.age_diff < 0

    def test_reach_diff_specific_values(self, make_fighter):
        """Fighter A has reach=72, Fighter B has reach=69 -> reach_diff = 3.0."""
        fighter_a = make_fighter(reach_inches=72.0)
        fighter_b = make_fighter(reach_inches=69.0)

        result = compute_physical_differentials(fighter_a, fighter_b)

        assert result.reach_diff == 3.0

    def test_age_diff_a_younger(self, make_fighter):
        """Fighter A DOB 1990-01-01, Fighter B DOB 1985-06-15 -> age_diff negative."""
        fighter_a = make_fighter(date_of_birth=date(1990, 1, 1))
        fighter_b = make_fighter(date_of_birth=date(1985, 6, 15))

        result = compute_physical_differentials(
            fighter_a, fighter_b, reference_date=date(2025, 1, 1)
        )

        # A is younger, so age_diff (A_age - B_age) should be negative
        assert result.age_diff is not None
        assert result.age_diff < 0
        # A age ~ 35.0, B age ~ 39.5
        assert result.fighter_a_age is not None
        assert result.fighter_b_age is not None
        assert result.fighter_a_age < result.fighter_b_age

    def test_one_fighter_missing_reach(self, make_fighter):
        """One fighter missing reach -> reach_diff is None (per D-02)."""
        fighter_a = make_fighter(reach_inches=72.0)
        fighter_b = make_fighter(reach_inches=None)

        result = compute_physical_differentials(fighter_a, fighter_b)

        assert result.reach_diff is None

    def test_both_fighters_missing_dob(self, make_fighter):
        """Both fighters missing DOB -> age_diff is None."""
        fighter_a = make_fighter(date_of_birth=None)
        fighter_b = make_fighter(date_of_birth=None)

        result = compute_physical_differentials(fighter_a, fighter_b)

        assert result.age_diff is None
        assert result.fighter_a_age is None
        assert result.fighter_b_age is None

    def test_custom_reference_date(self, make_fighter):
        """Custom reference_date affects individual ages but not DOB-based diff."""
        fighter_a = make_fighter(date_of_birth=date(1990, 1, 1))
        fighter_b = make_fighter(date_of_birth=date(1985, 1, 1))

        result_early = compute_physical_differentials(
            fighter_a, fighter_b, reference_date=date(2020, 1, 1)
        )
        result_late = compute_physical_differentials(
            fighter_a, fighter_b, reference_date=date(2025, 1, 1)
        )

        # Individual ages differ between the two reference dates
        assert result_early.fighter_a_age is not None
        assert result_late.fighter_a_age is not None
        assert result_late.fighter_a_age > result_early.fighter_a_age

        # But age_diff between the two is the same (both aged 5 years)
        assert result_early.age_diff is not None
        assert result_late.age_diff is not None
        assert abs(result_early.age_diff - result_late.age_diff) < 0.01

    def test_one_fighter_missing_height(self, make_fighter):
        """One fighter missing height -> height_diff is None."""
        fighter_a = make_fighter(height_inches=None)
        fighter_b = make_fighter(height_inches=70.0)

        result = compute_physical_differentials(fighter_a, fighter_b)

        assert result.height_diff is None

    def test_one_fighter_missing_leg_reach(self, make_fighter):
        """One fighter missing leg_reach -> leg_reach_diff is None."""
        fighter_a = make_fighter(leg_reach_inches=42.0)
        fighter_b = make_fighter(leg_reach_inches=None)

        result = compute_physical_differentials(fighter_a, fighter_b)

        assert result.leg_reach_diff is None

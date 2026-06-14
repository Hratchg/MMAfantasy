"""Tests for Kaggle CSV parsing utilities."""

from datetime import date

import pytest

from ufc_prediction.data.parsers import (
    cm_to_inches,
    parse_control_time,
    parse_dob,
    parse_fight_date,
    parse_height_to_inches,
    parse_num_rounds,
    parse_percentage,
    parse_reach_to_inches,
    parse_weight_class,
    parse_x_of_y,
)

# ── parse_x_of_y ─────────────────────────────────────────────────────────────


class TestParseXOfY:
    def test_normal(self):
        assert parse_x_of_y("69 of 101") == (69, 101)

    def test_zeros(self):
        assert parse_x_of_y("0 of 0") == (0, 0)

    def test_empty_string(self):
        assert parse_x_of_y("") == (None, None)

    def test_dashes(self):
        assert parse_x_of_y("---") == (None, None)

    def test_double_dash(self):
        assert parse_x_of_y("--") == (None, None)

    def test_none_input(self):
        assert parse_x_of_y(None) == (None, None)

    def test_single_digit(self):
        assert parse_x_of_y("1 of 3") == (1, 3)

    def test_large_numbers(self):
        assert parse_x_of_y("150 of 300") == (150, 300)


# ── parse_control_time ────────────────────────────────────────────────────────


class TestParseControlTime:
    def test_normal(self):
        assert parse_control_time("2:15") == 135

    def test_zero(self):
        assert parse_control_time("0:00") == 0

    def test_empty_string(self):
        assert parse_control_time("") is None

    def test_dashes(self):
        assert parse_control_time("---") is None

    def test_none_input(self):
        assert parse_control_time(None) is None

    def test_one_minute(self):
        assert parse_control_time("1:00") == 60

    def test_seconds_only(self):
        assert parse_control_time("0:45") == 45

    def test_large_time(self):
        assert parse_control_time("10:30") == 630


# ── parse_percentage ──────────────────────────────────────────────────────────


class TestParsePercentage:
    def test_normal(self):
        assert parse_percentage("65%") == pytest.approx(0.65)

    def test_zero(self):
        assert parse_percentage("0%") == pytest.approx(0.0)

    def test_hundred(self):
        assert parse_percentage("100%") == pytest.approx(1.0)

    def test_empty_string(self):
        assert parse_percentage("") is None

    def test_dashes(self):
        assert parse_percentage("---") is None

    def test_none_input(self):
        assert parse_percentage(None) is None


# ── parse_height_to_inches ───────────────────────────────────────────────────


class TestParseHeightToInches:
    def test_five_ten(self):
        assert parse_height_to_inches("5' 10\"") == pytest.approx(70.0)

    def test_six_two(self):
        assert parse_height_to_inches("6' 2\"") == pytest.approx(74.0)

    def test_double_dash(self):
        assert parse_height_to_inches("--") is None

    def test_empty_string(self):
        assert parse_height_to_inches("") is None

    def test_none_input(self):
        assert parse_height_to_inches(None) is None

    def test_no_space(self):
        assert parse_height_to_inches("5'10\"") == pytest.approx(70.0)

    def test_six_feet_even(self):
        assert parse_height_to_inches("6' 0\"") == pytest.approx(72.0)


# ── parse_reach_to_inches ────────────────────────────────────────────────────


class TestParseReachToInches:
    def test_with_quote(self):
        assert parse_reach_to_inches('72"') == pytest.approx(72.0)

    def test_decimal(self):
        assert parse_reach_to_inches("72.0") == pytest.approx(72.0)

    def test_empty_string(self):
        assert parse_reach_to_inches("") is None

    def test_dashes(self):
        assert parse_reach_to_inches("--") is None

    def test_none_input(self):
        assert parse_reach_to_inches(None) is None

    def test_decimal_value(self):
        assert parse_reach_to_inches("70.5") == pytest.approx(70.5)


# ── parse_weight_class ───────────────────────────────────────────────────────


class TestParseWeightClass:
    def test_title_bout(self):
        assert parse_weight_class("UFC Heavyweight Title Bout") == ("Heavyweight", True)

    def test_womens(self):
        assert parse_weight_class("Women's Strawweight Bout") == ("Women's Strawweight", False)

    def test_regular(self):
        assert parse_weight_class("Middleweight Bout") == ("Middleweight", False)

    def test_catch_weight(self):
        assert parse_weight_class("Catch Weight Bout") == ("Catch Weight", False)

    def test_light_heavyweight(self):
        assert parse_weight_class("Light Heavyweight Bout") == ("Light Heavyweight", False)

    def test_womens_title(self):
        assert parse_weight_class("UFC Women's Bantamweight Title Bout") == (
            "Women's Bantamweight",
            True,
        )

    def test_flyweight(self):
        assert parse_weight_class("Flyweight Bout") == ("Flyweight", False)

    def test_welterweight(self):
        assert parse_weight_class("Welterweight Bout") == ("Welterweight", False)

    def test_featherweight_title(self):
        assert parse_weight_class("UFC Featherweight Title Bout") == ("Featherweight", True)


# ── parse_fight_date ─────────────────────────────────────────────────────────


class TestParseFightDate:
    def test_long_month_format(self):
        assert parse_fight_date("March 28, 2026") == date(2026, 3, 28)

    def test_iso_format(self):
        assert parse_fight_date("2026-03-28") == date(2026, 3, 28)

    def test_empty_string(self):
        assert parse_fight_date("") is None

    def test_none_input(self):
        assert parse_fight_date(None) is None

    def test_short_month_format(self):
        assert parse_fight_date("Jan 01, 2020") == date(2020, 1, 1)


# ── parse_dob ────────────────────────────────────────────────────────────────


class TestParseDob:
    def test_short_month(self):
        assert parse_dob("Jul 13, 1978") == date(1978, 7, 13)

    def test_empty_string(self):
        assert parse_dob("") is None

    def test_none_input(self):
        assert parse_dob(None) is None

    def test_long_month(self):
        assert parse_dob("January 28, 1986") == date(1986, 1, 28)

    def test_iso_format(self):
        assert parse_dob("1990-05-15") == date(1990, 5, 15)


# ── parse_num_rounds ─────────────────────────────────────────────────────────


class TestParseNumRounds:
    def test_three_rounds(self):
        assert parse_num_rounds("3 Rnd (5-5-5)") == 3

    def test_five_rounds(self):
        assert parse_num_rounds("5 Rnd (5-5-5-5-5)") == 5

    def test_no_time_limit(self):
        assert parse_num_rounds("No Time Limit") == 3

    def test_empty_string(self):
        assert parse_num_rounds("") == 3

    def test_none_input(self):
        assert parse_num_rounds(None) == 3


# ── cm_to_inches ─────────────────────────────────────────────────────────────


class TestCmToInches:
    def test_standard(self):
        assert cm_to_inches(180.0) == pytest.approx(70.87, abs=0.01)

    def test_zero(self):
        assert cm_to_inches(0.0) == pytest.approx(0.0)

    def test_152(self):
        assert cm_to_inches(152.4) == pytest.approx(60.0, abs=0.01)

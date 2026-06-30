"""RED tests for `ufc predict coverage` subcommand (Phase 15.1 D-06)."""

from __future__ import annotations

import inspect
from datetime import date
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ufc_prediction.cli.main import app
from ufc_prediction.cli.predict import predict_coverage

runner = CliRunner()


@patch("ufc_prediction.cli.predict.SessionLocal")
@patch("ufc_prediction.cli.predict.load_fight_records")
@patch("ufc_prediction.cli.predict.load_fight_odds")
def test_per_year_table_renders_with_70pct_marker(
    mock_odds: MagicMock,
    mock_fr: MagicMock,
    mock_sl: MagicMock,
) -> None:
    mock_fr.return_value = [
        {"fight_id": f"f{i}", "event_date": date(2015, 1, 1)} for i in range(100)
    ]
    # 70/100 fights have odds = 70% coverage
    mock_odds.return_value = {("fighter_x", f"f{i}"): {} for i in range(70)}
    result = runner.invoke(app, ["predict", "coverage"])
    assert result.exit_code == 0, result.output
    assert "70.0%" in result.output
    assert "2015" in result.output


@patch("ufc_prediction.cli.predict.SessionLocal")
@patch("ufc_prediction.cli.predict.load_fight_records")
@patch("ufc_prediction.cli.predict.load_fight_odds")
def test_no_year_meets_threshold_warns(
    mock_odds: MagicMock,
    mock_fr: MagicMock,
    mock_sl: MagicMock,
) -> None:
    mock_fr.return_value = [
        {"fight_id": f"f{i}", "event_date": date(2010, 1, 1)} for i in range(100)
    ]
    mock_odds.return_value = {}
    result = runner.invoke(app, ["predict", "coverage"])
    assert result.exit_code == 0, result.output
    assert "No year reaches" in result.output


@patch("ufc_prediction.cli.predict.SessionLocal")
@patch("ufc_prediction.cli.predict.load_fight_records")
@patch("ufc_prediction.cli.predict.load_fight_odds")
def test_threshold_flag_overrides_default(
    mock_odds: MagicMock,
    mock_fr: MagicMock,
    mock_sl: MagicMock,
) -> None:
    mock_fr.return_value = [
        {"fight_id": f"f{i}", "event_date": date(2018, 1, 1)} for i in range(100)
    ]
    mock_odds.return_value = {("fighter_x", f"f{i}"): {} for i in range(60)}
    result = runner.invoke(app, ["predict", "coverage", "--threshold", "0.50"])
    assert result.exit_code == 0, result.output
    assert "60.0%" in result.output


def test_default_threshold_is_70pct() -> None:
    sig = inspect.signature(predict_coverage)
    threshold_param = sig.parameters["threshold"]
    # Typer.OptionInfo wraps the literal default; .default is OptionInfo,
    # .default.default is the actual literal (per test_predict_train.py:147-152)
    assert threshold_param.default.default == 0.70


@patch("ufc_prediction.cli.predict.SessionLocal")
@patch("ufc_prediction.cli.predict.load_fight_records")
@patch("ufc_prediction.cli.predict.load_fight_odds")
def test_read_only_no_db_mutation(
    mock_odds: MagicMock,
    mock_fr: MagicMock,
    mock_sl: MagicMock,
) -> None:
    mock_fr.return_value = []
    mock_odds.return_value = {}
    mock_session = MagicMock()
    mock_sl.return_value = mock_session
    runner.invoke(app, ["predict", "coverage"])
    mock_session.add.assert_not_called()
    mock_session.commit.assert_not_called()
    mock_session.flush.assert_not_called()


@patch("ufc_prediction.cli.predict.SessionLocal")
@patch("ufc_prediction.cli.predict.load_fight_records")
@patch("ufc_prediction.cli.predict.load_fight_odds")
def test_load_fight_records_called_once(
    mock_odds: MagicMock,
    mock_fr: MagicMock,
    mock_sl: MagicMock,
) -> None:
    mock_fr.return_value = [{"fight_id": "f1", "event_date": date(2015, 1, 1)}]
    mock_odds.return_value = {}
    runner.invoke(app, ["predict", "coverage"])
    assert mock_fr.call_count == 1


@patch("ufc_prediction.cli.predict.SessionLocal")
@patch("ufc_prediction.cli.predict.load_fight_records")
@patch("ufc_prediction.cli.predict.load_fight_odds")
def test_collision_fighter_id_not_counted_as_fight_id(
    mock_odds: MagicMock,
    mock_fr: MagicMock,
    mock_sl: MagicMock,
) -> None:
    """Regression: fighter_id=5 must NOT be mistaken for fight_id=5.

    Bug: predict.py:380 + :536 previously destructured fight_odds.keys()
    as (fid, _fighter_id), but the actual key shape is
    (fighter_id, fight_id). The fix swaps to (_fighter_id, fid).

    This test fixture deliberately constructs a collision:
      - fight_id=5 exists in 2024 with NO odds
      - fighter_id=5 has odds for a DIFFERENT fight (fight_id=999) in 2020
    Under the buggy destructure, the set-of-fight-ids-with-odds would
    contain {5, 999} (capturing fighter_id=5 as if it were a fight_id),
    and 2024 would falsely report 1/1 = 100% coverage.
    Under the correct destructure, the set is {999}, and 2024 correctly
    reports 0/1 = 0% coverage.
    """
    # 2024 fight: fight_id=5, no odds in fixture
    # 2020 fight: fight_id=999, has odds (keyed by fighter_id=5 -> collision trap)
    mock_fr.return_value = [
        {"fight_id": 5, "event_date": date(2024, 6, 1)},
        {"fight_id": 999, "event_date": date(2020, 6, 1)},
    ]
    mock_odds.return_value = {
        # Correct key shape: (fighter_id, fight_id)
        # fighter_id=5 happens to numerically equal the OTHER fight's fight_id.
        (5, 999): {"opening_implied_prob": 0.5, "closing_implied_prob": 0.5},
    }
    result = runner.invoke(app, ["predict", "coverage", "--threshold", "0.50"])
    assert result.exit_code == 0, result.output
    # 2024: 0 of 1 fights have odds -> 0.0%
    # 2020: 1 of 1 fights have odds -> 100.0%
    # If bug present: 2024 would show 100.0% (collision false positive)
    assert "0.0%" in result.output, (
        "2024 must report 0% -- fighter_id=5 collision must NOT be "
        f"counted as fight_id=5 odds coverage. Output:\n{result.output}"
    )
    assert "100.0%" in result.output, (
        f"2020 must report 100% -- fight_id=999 has real odds. Output:\n{result.output}"
    )


@patch("ufc_prediction.cli.predict.SessionLocal")
@patch("ufc_prediction.cli.predict.load_fight_records")
@patch("ufc_prediction.cli.predict.load_fight_odds")
def test_coverage_matches_direct_fight_id_count(
    mock_odds: MagicMock,
    mock_fr: MagicMock,
    mock_sl: MagicMock,
) -> None:
    """Semantic correctness: rendered per-year coverage equals K/N where
    K = distinct fight_ids in fight_odds for fights in that year and
    N = total fights in that year.

    Two fighters per fight (matches real load_fight_odds output where
    each fight contributes 2 keys, one per fighter POV).
    """
    # 2023: 4 fights total, 2 distinct fight_ids have odds (each from 2 fighters)
    mock_fr.return_value = [
        {"fight_id": fid, "event_date": date(2023, 3, 1)} for fid in [101, 102, 103, 104]
    ]
    # Two-fighter-per-fight pattern: fights 101 and 102 covered, 103/104 not
    mock_odds.return_value = {
        (201, 101): {"opening_implied_prob": 0.5, "closing_implied_prob": 0.5},
        (202, 101): {"opening_implied_prob": 0.5, "closing_implied_prob": 0.5},
        (203, 102): {"opening_implied_prob": 0.5, "closing_implied_prob": 0.5},
        (204, 102): {"opening_implied_prob": 0.5, "closing_implied_prob": 0.5},
    }
    result = runner.invoke(app, ["predict", "coverage", "--threshold", "0.99"])
    assert result.exit_code == 0, result.output
    # 2023: 2 of 4 fights have odds -> 50.0%
    assert "50.0%" in result.output, (
        f"Expected 50.0% coverage for 2023 (2/4 fights). Output:\n{result.output}"
    )

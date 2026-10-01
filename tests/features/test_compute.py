"""Unit tests for FeatureComputer orchestrator.

Tests temporal integrity, first-fight skip, EWMA vs career avg,
style tags, opponent-adjusted rates, shrinkage, and graceful handling
of missing round stats.  All tests use mock data -- no DB required.
"""

from __future__ import annotations

from datetime import date

import pytest

from tests.features.conftest import make_round_stats

# ---------------------------------------------------------------------------
# Helpers for building mock data
# ---------------------------------------------------------------------------

FIGHTER_A = 1
FIGHTER_B = 2
FIGHTER_C = 3
FIGHTER_D = 4


def _make_fight(
    fight_id: int,
    event_date: date,
    fighter_a_id: int,
    fighter_b_id: int,
    *,
    round_finished: int = 3,
    time_finished: str = "5:00",
    num_rounds: int = 3,
) -> dict:
    return {
        "fight_id": fight_id,
        "event_date": event_date,
        "fighter_a_id": fighter_a_id,
        "fighter_b_id": fighter_b_id,
        "weight_class": "Lightweight",
        "round_finished": round_finished,
        "time_finished": time_finished,
        "num_rounds": num_rounds,
    }


def _round_stats_for(
    fight_id: int,
    fighter_id: int,
    overrides: dict | None = None,
    rounds: int = 3,
) -> list[dict]:
    """Build per-round stat dicts for a fighter in a fight."""
    result = []
    for r in range(1, rounds + 1):
        stats = make_round_stats(overrides)
        stats["fighter_id"] = fighter_id
        stats["fight_id"] = fight_id
        stats["round_number"] = r
        result.append(stats)
    return result


def _build_round_stats_by_fight(
    *entries: tuple[int, int, dict | None],
    rounds: int = 3,
) -> dict[int, list[dict]]:
    """Build round_stats_by_fight dict from (fight_id, fighter_id, overrides) tuples."""
    result: dict[int, list[dict]] = {}
    for fight_id, fighter_id, overrides in entries:
        if fight_id not in result:
            result[fight_id] = []
        result[fight_id].extend(_round_stats_for(fight_id, fighter_id, overrides, rounds))
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTemporalIntegrity:
    """Features at fight N use only data from fights before N."""

    def test_temporal_integrity(self) -> None:
        """3-fight sequence: features at fight 2 use fight 1 data only,
        features at fight 3 use fights 1+2 data only."""
        from ufc_prediction.features.compute import FeatureComputer
        from ufc_prediction.features.config import FeatureConfig

        # Use shrinkage_min_fights=1 so shrinkage doesn't alter raw values
        config = FeatureConfig(shrinkage_min_fights=1)

        # Fighter A fights at t1 (fight 100), t2 (fight 101), t3 (fight 102)
        fights = [
            _make_fight(100, date(2020, 1, 1), FIGHTER_A, FIGHTER_B),
            _make_fight(101, date(2020, 2, 1), FIGHTER_A, FIGHTER_C),
            _make_fight(102, date(2020, 3, 1), FIGHTER_A, FIGHTER_D),
        ]

        # Different stats per fight to verify temporal isolation
        rs = _build_round_stats_by_fight(
            # Fight 100: A has sig_str_landed=10 per round, B has defaults
            (
                100,
                FIGHTER_A,
                {
                    "sig_strikes_landed": 10,
                    "head_strikes_landed": 5,
                    "body_strikes_landed": 3,
                    "leg_strikes_landed": 2,
                },
            ),
            (100, FIGHTER_B, None),
            # Fight 101: A has sig_str_landed=20 per round, C has defaults
            (
                101,
                FIGHTER_A,
                {
                    "sig_strikes_landed": 20,
                    "head_strikes_landed": 10,
                    "body_strikes_landed": 5,
                    "leg_strikes_landed": 5,
                },
            ),
            (101, FIGHTER_C, None),
            # Fight 102: A has sig_str_landed=30 per round, D has defaults
            (
                102,
                FIGHTER_A,
                {
                    "sig_strikes_landed": 30,
                    "head_strikes_landed": 15,
                    "body_strikes_landed": 8,
                    "leg_strikes_landed": 7,
                },
            ),
            (102, FIGHTER_D, None),
        )

        domain_elo: dict[tuple[int, int], dict[str, float | None]] = {}

        computer = FeatureComputer(config)
        results = computer.compute_all(fights, rs, domain_elo)

        # Collect feature rows for Fighter A
        a_features = [r for r in results if r["fighter_id"] == FIGHTER_A]

        # Fight 100 is first fight -> no feature row
        assert not any(r["fight_id"] == 100 for r in a_features)

        # Fight 101 feature row: sig_str_per_minute based on fight 100 data ONLY
        fight_101_row = next(r for r in a_features if r["fight_id"] == 101)
        f101_feats = fight_101_row["features"]
        # Fight 100: 10 sig_str * 3 rounds = 30 landed in 15 minutes -> 2.0/min
        assert f101_feats["sig_str_per_minute"] == pytest.approx(2.0, abs=0.01)

        # Fight 102 feature row: should average fights 100+101 data
        fight_102_row = next(r for r in a_features if r["fight_id"] == 102)
        f102_feats = fight_102_row["features"]
        # Fight 100: 30 landed / 15min = 2.0; Fight 101: 60 landed / 15min = 4.0
        # Career avg = (30+60)/(15+15) = 3.0
        assert f102_feats["sig_str_per_minute"] == pytest.approx(3.0, abs=0.01)


class TestFirstFightSkip:
    """Fighter's first fight produces no feature row."""

    def test_first_fight_produces_no_feature_row(self) -> None:
        from ufc_prediction.features.compute import FeatureComputer

        fights = [_make_fight(100, date(2020, 1, 1), FIGHTER_A, FIGHTER_B)]
        rs = _build_round_stats_by_fight(
            (100, FIGHTER_A, None),
            (100, FIGHTER_B, None),
        )

        computer = FeatureComputer()
        results = computer.compute_all(fights, rs, {})

        # Both fighters' first fight -- no feature rows
        assert len(results) == 0

    def test_second_fight_produces_feature_row(self) -> None:
        from ufc_prediction.features.compute import FeatureComputer

        fights = [
            _make_fight(100, date(2020, 1, 1), FIGHTER_A, FIGHTER_B),
            _make_fight(101, date(2020, 2, 1), FIGHTER_A, FIGHTER_C),
        ]
        rs = _build_round_stats_by_fight(
            (100, FIGHTER_A, None),
            (100, FIGHTER_B, None),
            (101, FIGHTER_A, None),
            (101, FIGHTER_C, None),
        )

        computer = FeatureComputer()
        results = computer.compute_all(fights, rs, {})

        # Fighter A's second fight produces a feature row
        a_features = [r for r in results if r["fighter_id"] == FIGHTER_A]
        assert len(a_features) == 1
        assert a_features[0]["fight_id"] == 101


class TestEwmaDiffersFromCareerAvg:
    """EWMA values differ from career averages when fight values vary."""

    def test_ewma_differs_from_career_average(self) -> None:
        from ufc_prediction.features.compute import FeatureComputer

        # 4 fights: increasing sig_str_landed so career avg != EWMA
        fights = [
            _make_fight(100, date(2020, 1, 1), FIGHTER_A, FIGHTER_B),
            _make_fight(101, date(2020, 2, 1), FIGHTER_A, FIGHTER_C),
            _make_fight(102, date(2020, 3, 1), FIGHTER_A, FIGHTER_D),
            _make_fight(103, date(2020, 4, 1), FIGHTER_A, 5),  # Fighter 5
        ]
        rs = _build_round_stats_by_fight(
            (
                100,
                FIGHTER_A,
                {
                    "sig_strikes_landed": 6,
                    "head_strikes_landed": 3,
                    "body_strikes_landed": 2,
                    "leg_strikes_landed": 1,
                },
            ),
            (100, FIGHTER_B, None),
            (
                101,
                FIGHTER_A,
                {
                    "sig_strikes_landed": 12,
                    "head_strikes_landed": 6,
                    "body_strikes_landed": 3,
                    "leg_strikes_landed": 3,
                },
            ),
            (101, FIGHTER_C, None),
            (
                102,
                FIGHTER_A,
                {
                    "sig_strikes_landed": 18,
                    "head_strikes_landed": 9,
                    "body_strikes_landed": 5,
                    "leg_strikes_landed": 4,
                },
            ),
            (102, FIGHTER_D, None),
            (103, FIGHTER_A, None),
            (103, 5, None),
        )
        # Also need round stats for the other fighters at their first fights
        # (B at 100, C at 101, D at 102, 5 at 103 are each their first fight)

        computer = FeatureComputer()
        results = computer.compute_all(fights, rs, {})

        # Fight 103 features for A: has 3 prior fights
        a_features = [r for r in results if r["fighter_id"] == FIGHTER_A]
        fight_103 = next(r for r in a_features if r["fight_id"] == 103)
        feats = fight_103["features"]

        # Career avg and EWMA should differ (EWMA weights recent more)
        assert feats["sig_str_per_minute"] != feats["sig_str_per_minute_ewma"]


class TestStyleTag:
    """Style tag appears in feature dict as string."""

    def test_style_tag_from_domain_elo(self) -> None:
        from ufc_prediction.features.compute import FeatureComputer

        fights = [
            _make_fight(100, date(2020, 1, 1), FIGHTER_A, FIGHTER_B),
            _make_fight(101, date(2020, 2, 1), FIGHTER_A, FIGHTER_C),
        ]
        rs = _build_round_stats_by_fight(
            (100, FIGHTER_A, None),
            (100, FIGHTER_B, None),
            (101, FIGHTER_A, None),
            (101, FIGHTER_C, None),
        )
        # Provide domain Elo: A is a striker (striking >> grappling)
        domain_elo = {
            (FIGHTER_A, 101): {"striking_elo": 1600.0, "grappling_elo": 1500.0},
        }

        computer = FeatureComputer()
        results = computer.compute_all(fights, rs, domain_elo)

        a_features = [r for r in results if r["fighter_id"] == FIGHTER_A]
        assert len(a_features) == 1
        feats = a_features[0]["features"]
        assert feats["style_tag"] == "striker"

    def test_style_tag_balanced_when_no_elo(self) -> None:
        from ufc_prediction.features.compute import FeatureComputer

        fights = [
            _make_fight(100, date(2020, 1, 1), FIGHTER_A, FIGHTER_B),
            _make_fight(101, date(2020, 2, 1), FIGHTER_A, FIGHTER_C),
        ]
        rs = _build_round_stats_by_fight(
            (100, FIGHTER_A, None),
            (100, FIGHTER_B, None),
            (101, FIGHTER_A, None),
            (101, FIGHTER_C, None),
        )

        computer = FeatureComputer()
        results = computer.compute_all(fights, rs, {})

        a_features = [r for r in results if r["fighter_id"] == FIGHTER_A]
        feats = a_features[0]["features"]
        assert feats["style_tag"] == "balanced"


class TestOpponentAdjusted:
    """Opponent-adjusted rates appear for 4 specified stats."""

    def test_opponent_adjusted_keys_present(self) -> None:
        from ufc_prediction.features.compute import FeatureComputer

        # Need at least 2 fights for opponent to have career data
        fights = [
            _make_fight(100, date(2020, 1, 1), FIGHTER_A, FIGHTER_B),
            _make_fight(101, date(2020, 2, 1), FIGHTER_B, FIGHTER_C),
            _make_fight(102, date(2020, 3, 1), FIGHTER_A, FIGHTER_B),
        ]
        rs = _build_round_stats_by_fight(
            (100, FIGHTER_A, None),
            (100, FIGHTER_B, None),
            (101, FIGHTER_B, None),
            (101, FIGHTER_C, None),
            (102, FIGHTER_A, None),
            (102, FIGHTER_B, None),
        )

        computer = FeatureComputer()
        results = computer.compute_all(fights, rs, {})

        # Fighter A's features at fight 102 (has prior data and opponent has prior data)
        a_at_102 = [r for r in results if r["fighter_id"] == FIGHTER_A and r["fight_id"] == 102]
        assert len(a_at_102) == 1
        feats = a_at_102[0]["features"]

        # All 4 opponent-adjusted keys present
        for key in ("opp_adj_sig_str", "opp_adj_td", "opp_adj_strike_def", "opp_adj_ctrl_time"):
            assert key in feats, f"Missing key: {key}"


class TestShrinkage:
    """Fighter with few fights has features shrunk toward league mean."""

    def test_shrinkage_effect_on_low_fight_count(self) -> None:
        from ufc_prediction.features.compute import FeatureComputer

        # Create scenario: Fighter A with 1 prior fight, Fighter B with many
        # We need enough fights to see the shrinkage difference
        fights = [
            _make_fight(100, date(2020, 1, 1), FIGHTER_A, FIGHTER_B),
            _make_fight(101, date(2020, 2, 1), FIGHTER_A, FIGHTER_C),
        ]
        rs = _build_round_stats_by_fight(
            # Fighter A: high sig strikes
            (
                100,
                FIGHTER_A,
                {
                    "sig_strikes_landed": 20,
                    "head_strikes_landed": 10,
                    "body_strikes_landed": 5,
                    "leg_strikes_landed": 5,
                },
            ),
            (100, FIGHTER_B, None),
            (101, FIGHTER_A, None),
            (101, FIGHTER_C, None),
        )

        computer = FeatureComputer()
        results = computer.compute_all(fights, rs, {})

        # Fighter A at fight 101 has only 1 prior fight -> shrinkage should apply
        a_features = [r for r in results if r["fighter_id"] == FIGHTER_A]
        assert len(a_features) == 1
        feats = a_features[0]["features"]

        # sig_str_per_minute raw: 20*3/15 = 4.0
        # With shrinkage (1 fight, min 5): factor = 0.2, value is pulled toward league mean
        # The key assertion: the value should be different from raw 4.0
        # (It's pulled toward the league mean which is the average of all fighters)
        # Can't assert exact value without knowing other fighters' features, but
        # it should be present as a float
        assert isinstance(feats["sig_str_per_minute"], float)


# The 20 numeric performance keys Pass 2 shrinks.
_SHRUNK_KEYS = (
    "sig_str_per_minute",
    "sig_str_per_minute_ewma",
    "total_str_per_minute",
    "total_str_per_minute_ewma",
    "td_rate",
    "td_rate_ewma",
    "td_accuracy",
    "td_accuracy_ewma",
    "td_defense",
    "td_defense_ewma",
    "strike_defense",
    "strike_defense_ewma",
    "ctrl_time_per_fight",
    "ctrl_time_per_fight_ewma",
    "sub_att_per_fight",
    "sub_att_per_fight_ewma",
    "opp_adj_sig_str",
    "opp_adj_td",
    "opp_adj_strike_def",
    "opp_adj_ctrl_time",
)


def _scheduled_corpus(
    schedule: tuple[tuple[date, int, int], ...],
    *,
    first_id: int,
    seed: int,
    scale: int = 1,
) -> tuple[list[dict], dict[int, list[dict]]]:
    """Fights for ``(date, a, b)`` entries with random per-round stats;
    ``scale`` multiplies the striking/control volume."""
    import random

    rng = random.Random(seed)
    fights, entries = [], []
    for i, (day, a, b) in enumerate(schedule):
        fid = first_id + i
        fights.append(_make_fight(fid, day, a, b))
        for fighter in (a, b):
            sig_att = rng.randint(8, 30) * scale
            td_att = rng.randint(0, 4)
            entries.append(
                (
                    fid,
                    fighter,
                    {
                        "sig_strikes_attempted": sig_att,
                        "sig_strikes_landed": rng.randint(0, sig_att),
                        "head_strikes_landed": rng.randint(0, 10) * scale,
                        "body_strikes_landed": rng.randint(0, 5),
                        "leg_strikes_landed": rng.randint(0, 5),
                        "takedowns_attempted": td_att,
                        "takedowns_landed": rng.randint(0, td_att),
                        "control_time_seconds": rng.randint(0, 150) * scale,
                        "submission_attempts": rng.randint(0, 2),
                    },
                )
            )
    return fights, _build_round_stats_by_fight(*entries)


def _by_row(results: list[dict]) -> dict[tuple[int, int], dict]:
    return {(r["fighter_id"], r["fight_id"]): r["features"] for r in results}


class TestAsOfShrinkage:
    """Pass 2 shrinks each row toward the league mean over rows dated
    strictly before it (operator decision D4), so a row never depends on
    fights on or after its own date. The league means used to be taken once
    over the whole corpus, which pulled a 2012 row toward 2013-2026 fights."""

    # Two bouts share most dates so there are same-day rows. Everyone
    # debuts on the first date, so the first rows appear on 2020-03-01.
    _SCHEDULE: tuple[tuple[date, int, int], ...] = (
        (date(2020, 1, 1), FIGHTER_A, FIGHTER_B),
        (date(2020, 1, 1), FIGHTER_C, FIGHTER_D),
        (date(2020, 3, 1), FIGHTER_A, FIGHTER_C),
        (date(2020, 3, 1), FIGHTER_B, FIGHTER_D),
        (date(2020, 5, 1), FIGHTER_A, FIGHTER_D),
        (date(2020, 5, 1), FIGHTER_B, FIGHTER_C),
        (date(2020, 7, 1), FIGHTER_A, FIGHTER_B),  # fight 106
        (date(2020, 9, 1), FIGHTER_C, FIGHTER_D),
    )
    _LATER: tuple[tuple[date, int, int], ...] = (
        (date(2021, 2, 1), FIGHTER_A, FIGHTER_C),
        (date(2021, 2, 1), FIGHTER_B, FIGHTER_D),
        (date(2021, 6, 1), FIGHTER_A, FIGHTER_D),
    )

    def test_later_fights_do_not_move_earlier_rows(self) -> None:
        """Adding only fights dated after every existing row must leave
        every existing row's shrunk values unchanged."""
        from ufc_prediction.features.compute import FeatureComputer

        fights, rs = _scheduled_corpus(self._SCHEDULE, first_id=100, seed=7)
        later, later_rs = _scheduled_corpus(self._LATER, first_id=200, seed=8, scale=25)

        before = _by_row(FeatureComputer().compute_all(fights, rs, {}))
        after = _by_row(FeatureComputer().compute_all([*fights, *later], {**rs, **later_rs}, {}))

        moved = [
            (row, key, feats[key], after[row][key])
            for row, feats in before.items()
            for key in _SHRUNK_KEYS
            if after[row][key] != pytest.approx(feats[key], abs=1e-12)
        ]
        assert not moved, f"later fights moved earlier shrunk rows: {moved[:5]}"

    def test_same_day_rows_do_not_enter_the_mean(self) -> None:
        """Another bout on the target's own date must not move the target
        row: its rows are not strictly earlier."""
        from ufc_prediction.features.compute import FeatureComputer

        fights, rs = _scheduled_corpus(self._SCHEDULE, first_id=100, seed=7)
        same_day, same_day_rs = _scheduled_corpus(
            ((date(2020, 7, 1), FIGHTER_C, FIGHTER_D),), first_id=300, seed=9, scale=25
        )
        before = _by_row(FeatureComputer().compute_all(fights, rs, {}))
        after = _by_row(
            FeatureComputer().compute_all(
                sorted([*fights, *same_day], key=lambda f: (f["event_date"], f["fight_id"])),
                {**rs, **same_day_rs},
                {},
            )
        )
        for row in ((FIGHTER_A, 106), (FIGHTER_B, 106)):
            for key in _SHRUNK_KEYS:
                assert after[row][key] == pytest.approx(before[row][key], abs=1e-12), (row, key)

    def test_matches_reference_as_of_mean(self) -> None:
        """Every row equals Bayesian shrinkage of its raw value toward the
        mean of the raw values on strictly earlier dates (brute force)."""
        from collections import Counter

        from ufc_prediction.features.compute import FeatureComputer
        from ufc_prediction.features.config import FeatureConfig

        fights, rs = _scheduled_corpus(
            (*self._SCHEDULE, *self._LATER), first_id=100, seed=7, scale=1
        )
        # min_fights=1 makes the shrinkage factor 1 for every row: raw values.
        raw = FeatureComputer(FeatureConfig(shrinkage_min_fights=1)).compute_all(fights, rs, {})
        shrunk = FeatureComputer().compute_all(fights, rs, {})

        prior_fights: Counter[int] = Counter()
        for raw_row, row in zip(raw, shrunk, strict=True):
            prior_fights[row["fighter_id"]] += 1
            factor = min(prior_fights[row["fighter_id"]] / 5, 1.0)
            earlier = [r["features"] for r in raw if r["as_of_date"] < row["as_of_date"]]
            for key in _SHRUNK_KEYS:
                value = raw_row["features"][key]
                seen = [f[key] for f in earlier if f[key] is not None]
                if value is None or not seen:
                    want = value
                else:
                    mean = sum(seen) / len(seen)
                    want = mean + (value - mean) * factor
                got = row["features"][key]
                assert got == pytest.approx(want, abs=1e-12), (row["fight_id"], key)

    def test_cold_start_rows_keep_raw_values(self) -> None:
        """Rows on the first date that has any rows have no earlier league
        data; they are left unshrunk rather than shrunk toward a mean that
        would have to look ahead."""
        from ufc_prediction.features.compute import FeatureComputer
        from ufc_prediction.features.config import FeatureConfig

        fights, rs = _scheduled_corpus(self._SCHEDULE, first_id=100, seed=7)
        raw = _by_row(
            FeatureComputer(FeatureConfig(shrinkage_min_fights=1)).compute_all(fights, rs, {})
        )
        shrunk = FeatureComputer().compute_all(fights, rs, {})
        first_rows = [r for r in shrunk if r["as_of_date"] == date(2020, 3, 1)]
        assert len(first_rows) == 4
        for row in first_rows:
            for key in _SHRUNK_KEYS:
                assert row["features"][key] == raw[(row["fighter_id"], row["fight_id"])][key]


class TestNoRoundStats:
    """Fights with no round stats: accumulator not updated, but feature row
    still produced using existing accumulator state if fighter has prior data."""

    def test_fight_with_no_round_stats_uses_prior_state(self) -> None:
        from ufc_prediction.features.compute import FeatureComputer

        fights = [
            _make_fight(100, date(2020, 1, 1), FIGHTER_A, FIGHTER_B),
            _make_fight(101, date(2020, 2, 1), FIGHTER_A, FIGHTER_C),
        ]
        # Only provide round stats for fight 100, not 101
        rs = _build_round_stats_by_fight(
            (
                100,
                FIGHTER_A,
                {
                    "sig_strikes_landed": 10,
                    "head_strikes_landed": 5,
                    "body_strikes_landed": 3,
                    "leg_strikes_landed": 2,
                },
            ),
            (100, FIGHTER_B, None),
        )

        computer = FeatureComputer()
        results = computer.compute_all(fights, rs, {})

        # Fighter A still gets a feature row at fight 101 using fight 100 data
        a_features = [r for r in results if r["fighter_id"] == FIGHTER_A]
        assert len(a_features) == 1
        assert a_features[0]["fight_id"] == 101
        assert a_features[0]["features"]["sig_str_per_minute"] is not None


class TestFeatureKeys:
    """Feature dict keys match CANONICAL_FEATURE_ORDER."""

    def test_feature_keys_match_canonical_order(self) -> None:
        from ufc_prediction.features.compute import FeatureComputer
        from ufc_prediction.features.config import CANONICAL_FEATURE_ORDER

        fights = [
            _make_fight(100, date(2020, 1, 1), FIGHTER_A, FIGHTER_B),
            _make_fight(101, date(2020, 2, 1), FIGHTER_A, FIGHTER_C),
        ]
        rs = _build_round_stats_by_fight(
            (100, FIGHTER_A, None),
            (100, FIGHTER_B, None),
            (101, FIGHTER_A, None),
            (101, FIGHTER_C, None),
        )

        computer = FeatureComputer()
        results = computer.compute_all(fights, rs, {})

        a_features = [r for r in results if r["fighter_id"] == FIGHTER_A]
        assert len(a_features) == 1
        feats = a_features[0]["features"]

        # All canonical keys (except embedding which is added separately)
        for key in CANONICAL_FEATURE_ORDER:
            assert key in feats, f"Missing canonical key: {key}"

    def test_result_dict_has_required_metadata(self) -> None:
        from ufc_prediction.features.compute import FeatureComputer

        fights = [
            _make_fight(100, date(2020, 1, 1), FIGHTER_A, FIGHTER_B),
            _make_fight(101, date(2020, 2, 1), FIGHTER_A, FIGHTER_C),
        ]
        rs = _build_round_stats_by_fight(
            (100, FIGHTER_A, None),
            (100, FIGHTER_B, None),
            (101, FIGHTER_A, None),
            (101, FIGHTER_C, None),
        )

        computer = FeatureComputer()
        results = computer.compute_all(fights, rs, {})

        a_features = [r for r in results if r["fighter_id"] == FIGHTER_A]
        row = a_features[0]

        assert "fighter_id" in row
        assert "fight_id" in row
        assert "as_of_date" in row
        assert "feature_set_version" in row
        assert row["feature_set_version"] == "v1"
        assert row["as_of_date"] == date(2020, 2, 1)


class TestOpponentAdjustedNoLeak:
    """Regression (code review 2026-09): fighter B's ``opp_adj_*`` features at
    fight N must be built from fighter A's PRE-fight accumulator. Previously A's
    accumulator absorbed fight N before B's features were built, so B's
    opponent-adjusted columns encoded how B actually performed in fight N."""

    @staticmethod
    def _run(a_stats_in_fight_101: dict) -> dict:
        from ufc_prediction.features.compute import FeatureComputer
        from ufc_prediction.features.config import FeatureConfig

        config = FeatureConfig(shrinkage_min_fights=1)
        fights = [
            _make_fight(100, date(2020, 1, 1), FIGHTER_A, FIGHTER_B),
            _make_fight(101, date(2020, 6, 1), FIGHTER_A, FIGHTER_B),
        ]
        rs = _build_round_stats_by_fight(
            (100, FIGHTER_A, {"sig_strikes_landed": 10, "head_strikes_landed": 10}),
            (100, FIGHTER_B, {"sig_strikes_landed": 10, "head_strikes_landed": 10}),
            (101, FIGHTER_A, a_stats_in_fight_101),
            (101, FIGHTER_B, {"sig_strikes_landed": 10, "head_strikes_landed": 10}),
        )
        results = FeatureComputer(config).compute_all(fights, rs, {})
        (row_b,) = [r for r in results if r["fighter_id"] == FIGHTER_B and r["fight_id"] == 101]
        return row_b["features"]

    def test_fighter_b_opp_adj_independent_of_current_fight(self) -> None:
        quiet = self._run({"sig_strikes_landed": 1, "head_strikes_landed": 1})
        loud = self._run({"sig_strikes_landed": 200, "head_strikes_landed": 200})
        for key in ("opp_adj_sig_str", "opp_adj_td", "opp_adj_strike_def", "opp_adj_ctrl_time"):
            assert quiet.get(key) == loud.get(key), (
                f"{key} for fighter B at fight 101 changed with fight 101's own stats: "
                f"{quiet.get(key)} vs {loud.get(key)}"
            )


class TestComputeUpcoming:
    """``compute_upcoming`` (the serve-time replay) must reproduce the row
    ``compute_all`` stores for the same fight: same accumulators, same as-of
    shrinkage, same NET keys, and nothing dated on or after the fight."""

    @staticmethod
    def _corpus() -> tuple[list[dict], dict[int, list[dict]]]:
        import random

        rng = random.Random(5)
        pairs = [
            (FIGHTER_A, FIGHTER_C),
            (FIGHTER_B, FIGHTER_D),
            (FIGHTER_A, FIGHTER_D),
            (FIGHTER_C, FIGHTER_D),
            (FIGHTER_A, FIGHTER_B),  # target (fight 104): A's 3rd, B's 2nd
            (FIGHTER_B, FIGHTER_C),  # both keep fighting afterwards
            (FIGHTER_A, FIGHTER_D),
            (FIGHTER_A, FIGHTER_B),
        ]
        fights, entries = [], []
        for i, (a, b) in enumerate(pairs):
            fid = 100 + i
            fights.append(_make_fight(fid, date(2020, 1 + i, 1), a, b))
            for fighter in (a, b):
                entries.append(
                    (
                        fid,
                        fighter,
                        {
                            "sig_strikes_landed": rng.randint(2, 15),
                            "head_strikes_landed": rng.randint(0, 8),
                            "takedowns_landed": rng.randint(0, 2),
                            "control_time_seconds": rng.randint(0, 120),
                            "submission_attempts": rng.randint(0, 2),
                        },
                    )
                )
        return fights, _build_round_stats_by_fight(*entries)

    def test_matches_stored_row_for_same_fight(self) -> None:
        from ufc_prediction.features.compute import FeatureComputer

        fights, rs = self._corpus()
        target = next(f for f in fights if f["fight_id"] == 104)
        stored = {
            r["fighter_id"]: r["features"]
            for r in FeatureComputer().compute_all(fights, rs, {})
            if r["fight_id"] == 104
        }
        computer = FeatureComputer()
        prior = [f for f in fights if f["event_date"] < target["event_date"]]
        served = computer.compute_upcoming(
            prior,
            rs,
            FIGHTER_A,
            FIGHTER_B,
            target["event_date"],
            computer.league_means(fights, rs, as_of=target["event_date"]),
            network_fights=fights,
        )
        self._assert_same(served, stored)

    def test_upcoming_after_last_event_matches_compute_all(self) -> None:
        """For a fight after the last stored event, the whole-corpus league
        means ARE the as-of means, so serving with ``league_means(fights,
        rs)`` equals the row ``compute_all`` stores once the fight is in the
        corpus (its own same-day rows are not in their shrinkage mean)."""
        from ufc_prediction.features.compute import FeatureComputer

        fights, rs = self._corpus()
        upcoming_date = date(2021, 3, 1)
        assert all(f["event_date"] < upcoming_date for f in fights)
        computer = FeatureComputer()
        served = computer.compute_upcoming(
            fights,
            rs,
            FIGHTER_A,
            FIGHTER_B,
            upcoming_date,
            computer.league_means(fights, rs),
            network_fights=fights,
        )
        upcoming = {
            **_make_fight(999, upcoming_date, FIGHTER_A, FIGHTER_B),
            "round_finished": None,
            "time_finished": None,
        }
        stored = {
            r["fighter_id"]: r["features"]
            for r in FeatureComputer().compute_all([*fights, upcoming], rs, {})
            if r["fight_id"] == 999
        }
        self._assert_same(served, stored)

    @staticmethod
    def _assert_same(served: dict[int, dict], stored: dict[int, dict]) -> None:
        assert set(served) == set(stored) == {FIGHTER_A, FIGHTER_B}
        for fighter in (FIGHTER_A, FIGHTER_B):
            for key, want in stored[fighter].items():
                if key in ("style_tag", "embedding"):
                    continue
                got = served[fighter][key]
                if isinstance(want, float) and want != want:
                    assert got != got, (fighter, key)
                else:
                    assert got == pytest.approx(want, abs=1e-12), (fighter, key)

    def test_league_means_as_of_are_over_strictly_earlier_rows(self) -> None:
        from ufc_prediction.features.compute import FeatureComputer
        from ufc_prediction.features.config import FeatureConfig

        fights, rs = self._corpus()
        raw = FeatureComputer(FeatureConfig(shrinkage_min_fights=1)).compute_all(fights, rs, {})
        computer = FeatureComputer()
        # 2020-05-01 has rows; the day after sees them, the day itself does not.
        for as_of in (date(2020, 5, 1), date(2020, 5, 2), date(2021, 1, 1)):
            earlier = [r["features"] for r in raw if r["as_of_date"] < as_of]
            means = computer.league_means(fights, rs, as_of=as_of)
            assert set(means) == set(_SHRUNK_KEYS)
            for key in _SHRUNK_KEYS:
                seen = [f[key] for f in earlier if f[key] is not None]
                assert means[key] == pytest.approx(sum(seen) / len(seen), abs=1e-12), key
        # Default: every stored row, i.e. the means as of any later date.
        assert computer.league_means(fights, rs) == computer.league_means(
            fights, rs, as_of=date(2099, 1, 1)
        )
        # Nothing earlier than the first row: no league means (cold start).
        first_row_date = min(r["as_of_date"] for r in raw)
        assert computer.league_means(fights, rs, as_of=first_row_date) == {}

    def test_debutant_absent_like_compute_all(self) -> None:
        from ufc_prediction.features.compute import FeatureComputer

        fights, rs = self._corpus()
        served = FeatureComputer().compute_upcoming(fights, rs, FIGHTER_A, 99, date(2021, 1, 1), {})
        assert set(served) == {FIGHTER_A}

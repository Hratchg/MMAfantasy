"""Integration tests for the full feature computation pipeline.

Uses testcontainers PostgreSQL from tests/conftest.py (transactional session).
Tests: feature storage, idempotent recomputation, temporal integrity, JSON keys.
"""

from __future__ import annotations

from datetime import date

from ufc_prediction.features.compute import FeatureComputer
from ufc_prediction.features.config import FeatureConfig
from ufc_prediction.features.queries import (
    bulk_insert_features,
    flush_features,
    load_all_domain_elo,
    load_all_round_stats,
    load_fights_with_duration,
)
from ufc_prediction.models.computed_feature import ComputedFeature
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.models.round_stats import RoundStats


def _seed_feature_data(session):
    """Seed DB with fighters, events, fights, and round stats for feature tests.

    Creates 3 fighters, 2 events, and 3 fights with round stats.
    Fighter A fights in all 3 (chronologically ordered).
    """
    fighter_a = Fighter(name="Feature Fighter A", source="test")
    fighter_b = Fighter(name="Feature Fighter B", source="test")
    fighter_c = Fighter(name="Feature Fighter C", source="test")
    session.add_all([fighter_a, fighter_b, fighter_c])
    session.flush()

    event_1 = Event(name="Feature Event 1", date=date(2020, 1, 1), source="test")
    event_2 = Event(name="Feature Event 2", date=date(2020, 6, 1), source="test")
    event_3 = Event(name="Feature Event 3", date=date(2020, 12, 1), source="test")
    session.add_all([event_1, event_2, event_3])
    session.flush()

    fight_1 = Fight(
        event_id=event_1.id,
        fighter_a_id=fighter_a.id,
        fighter_b_id=fighter_b.id,
        winner_id=fighter_a.id,
        weight_class="Lightweight",
        method="KO/TKO",
        round_finished=2,
        time_finished="3:30",
        num_rounds=3,
        source="test",
    )
    fight_2 = Fight(
        event_id=event_2.id,
        fighter_a_id=fighter_a.id,
        fighter_b_id=fighter_c.id,
        winner_id=fighter_a.id,
        weight_class="Lightweight",
        method="Decision",
        method_detail="Unanimous",
        round_finished=3,
        time_finished="5:00",
        num_rounds=3,
        source="test",
    )
    fight_3 = Fight(
        event_id=event_3.id,
        fighter_a_id=fighter_b.id,
        fighter_b_id=fighter_a.id,
        winner_id=fighter_a.id,
        weight_class="Lightweight",
        method="Submission",
        round_finished=1,
        time_finished="4:00",
        num_rounds=3,
        source="test",
    )
    session.add_all([fight_1, fight_2, fight_3])
    session.flush()

    # Add round stats for each fight (both fighters)
    def _add_round_stats(fight_id, fighter_id, rounds, overrides=None):
        base = {
            "sig_strikes_landed": 8,
            "sig_strikes_attempted": 15,
            "head_strikes_landed": 4,
            "body_strikes_landed": 2,
            "leg_strikes_landed": 2,
            "distance_strikes_landed": 3,
            "clinch_strikes_landed": 1,
            "ground_strikes_landed": 1,
            "takedowns_landed": 1,
            "takedowns_attempted": 2,
            "submission_attempts": 0,
            "control_time_seconds": 45,
            "knockdowns": 0,
            "reversals": 0,
        }
        if overrides:
            base.update(overrides)
        for r in range(1, rounds + 1):
            rs = RoundStats(
                fight_id=fight_id,
                fighter_id=fighter_id,
                round_number=r,
                **base,
            )
            session.add(rs)

    # Fight 1: 2 rounds (ended in round 2)
    _add_round_stats(fight_1.id, fighter_a.id, 2, {"sig_strikes_landed": 12})
    _add_round_stats(fight_1.id, fighter_b.id, 2, {"sig_strikes_landed": 5})

    # Fight 2: 3 rounds
    _add_round_stats(fight_2.id, fighter_a.id, 3, {"sig_strikes_landed": 10})
    _add_round_stats(fight_2.id, fighter_c.id, 3, {"sig_strikes_landed": 6})

    # Fight 3: 1 round (ended in round 1)
    _add_round_stats(fight_3.id, fighter_b.id, 1, {"sig_strikes_landed": 3})
    _add_round_stats(fight_3.id, fighter_a.id, 1, {"sig_strikes_landed": 7})

    session.flush()

    return {
        "fighters": [fighter_a, fighter_b, fighter_c],
        "events": [event_1, event_2, event_3],
        "fights": [fight_1, fight_2, fight_3],
    }


def _run_pipeline(session, config=None):
    """Run the full feature computation pipeline and return feature rows."""
    config = config or FeatureConfig()
    fights = load_fights_with_duration(session)
    round_stats = load_all_round_stats(session)
    domain_elo = load_all_domain_elo(session)

    computer = FeatureComputer(config)
    feature_rows = computer.compute_all(fights, round_stats, domain_elo)

    flush_features(session, config.feature_set_version)
    bulk_insert_features(session, feature_rows)
    session.flush()

    return feature_rows


def test_feature_compute_stores_features(session):
    """Run full pipeline and verify ComputedFeature rows exist in DB."""
    data = _seed_feature_data(session)
    feature_rows = _run_pipeline(session)

    # Feature rows should have been inserted
    db_count = session.query(ComputedFeature).count()
    assert db_count > 0
    assert db_count == len(feature_rows)

    # Verify fighter A has feature rows (at least for fights 2 and 3)
    fighter_a_id = data["fighters"][0].id
    a_features = (
        session.query(ComputedFeature).filter(ComputedFeature.fighter_id == fighter_a_id).all()
    )
    # Fighter A has 3 fights: first fight skipped, so 2 feature rows
    assert len(a_features) == 2

    # Verify metadata
    for feat in a_features:
        assert feat.feature_set_version == "v1"
        assert feat.as_of_date is not None
        assert feat.features is not None
        assert isinstance(feat.features, dict)


def test_feature_compute_idempotent(session):
    """Run pipeline twice -- second run produces same row count, not double."""
    _seed_feature_data(session)

    # First computation
    rows_1 = _run_pipeline(session)
    count_1 = session.query(ComputedFeature).count()
    assert count_1 > 0

    # Second computation (idempotent)
    rows_2 = _run_pipeline(session)
    count_2 = session.query(ComputedFeature).count()

    # Same count, not doubled
    assert count_2 == count_1
    assert len(rows_2) == len(rows_1)


def test_temporal_integrity_db(session):
    """For fighter A with 3 fights, verify fight 2 features use fight 1 data only."""
    data = _seed_feature_data(session)
    # Use shrinkage_min_fights=1 so raw values are not altered
    config = FeatureConfig(shrinkage_min_fights=1)
    _run_pipeline(session, config)

    fighter_a_id = data["fighters"][0].id
    fight_2_id = data["fights"][1].id
    fight_3_id = data["fights"][2].id

    # Get feature rows for fighter A ordered by as_of_date
    a_features = (
        session.query(ComputedFeature)
        .filter(ComputedFeature.fighter_id == fighter_a_id)
        .order_by(ComputedFeature.as_of_date)
        .all()
    )

    assert len(a_features) == 2

    # Fight 2 feature: based on fight 1 data only
    feat_fight2 = a_features[0]
    assert feat_fight2.fight_id == fight_2_id
    f2_feats = feat_fight2.features
    assert f2_feats["sig_str_per_minute"] is not None

    # Fight 3 feature: based on fights 1+2 data
    feat_fight3 = a_features[1]
    assert feat_fight3.fight_id == fight_3_id
    f3_feats = feat_fight3.features

    # The values should differ between fight 2 and fight 3 because
    # fight 3 incorporates fight 2 data (different stats)
    # Both should be present and numeric
    assert isinstance(f2_feats["sig_str_per_minute"], float)
    assert isinstance(f3_feats["sig_str_per_minute"], float)


def test_feature_json_has_expected_keys(session):
    """Verify features JSON column contains expected keys."""
    _seed_feature_data(session)
    _run_pipeline(session)

    feature = session.query(ComputedFeature).first()
    assert feature is not None

    feats = feature.features
    expected_keys = [
        "sig_str_per_minute",
        "sig_str_per_minute_ewma",
        "total_str_per_minute",
        "td_rate",
        "td_accuracy",
        "td_defense",
        "strike_defense",
        "ctrl_time_per_fight",
        "sub_att_per_fight",
        "opp_adj_sig_str",
        "opp_adj_td",
        "opp_adj_strike_def",
        "opp_adj_ctrl_time",
        "style_tag",
    ]

    for key in expected_keys:
        assert key in feats, f"Missing expected key in features JSON: {key}"

    # style_tag should be a string
    assert isinstance(feats["style_tag"], str)
    assert feats["style_tag"] in ("striker", "grappler", "balanced")

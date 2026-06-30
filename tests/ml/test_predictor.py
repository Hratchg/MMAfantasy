"""Unit + integration tests for ModelPredictor helpers — covers DEDUP-01, DEDUP-02.

Phase 16-02 D-12 update: the legacy 28-feat NaN-pad ``_build_feature_vector``
has been deleted. The original ``TestBuildFeatureVector`` block (3 tests)
is replaced with three new behaviours covered separately:

- ``inference_features.build`` cardinality + odds parity is in
  ``tests/unit/ml/test_inference_features.py`` (LIVE-02).
- ``predict()`` integration with ``fetch_matchup_odds`` +
  ``inference_features.build`` + the D-11 JSONL trace line is in
  ``tests/integration/test_live_odds_acceptance.py`` (Pitfall #4).

This file retains the dedup tests (``TestPreferCanonical``,
``TestResolveFighter``) because they still exercise predictor.py
internals. The ``_build_feature_vector`` import is gone with the function.
"""

from __future__ import annotations

from datetime import date

import pytest

from ufc_prediction.ml.predictor import _resolve_fighter
from ufc_prediction.models.elo_snapshot import EloSnapshot
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter


def _import_prefer_canonical():
    """Lazy import for the symbol Plan 02 will introduce.

    Today this raises ImportError (RED). Plan 02 turns it green.
    """
    from ufc_prediction.ml.predictor import _prefer_canonical

    return _prefer_canonical


def _make_minimal_fighter(name: str, source: str = "test") -> Fighter:
    """Build a Fighter row with all the physical attrs the feature vector reads."""
    return Fighter(
        name=name,
        source=source,
        height_inches=70.0,
        reach_inches=72.0,
        leg_reach_inches=40.0,
        stance="Orthodox",
        date_of_birth=date(1990, 1, 1),
    )


def _seed_overall_snapshot(
    session,
    fighter: Fighter,
    fight: Fight,
    elo_after_shrinkage: float = 1500.0,
) -> EloSnapshot:
    """Create the minimum 'overall' EloSnapshot for ``_get_latest_elo`` to return a real value."""
    snap = EloSnapshot(
        fighter_id=fighter.id,
        fight_id=fight.id,
        division="Lightweight",
        elo_type="overall",
        elo_before=1500.0,
        elo_after=elo_after_shrinkage,
        elo_after_shrinkage=elo_after_shrinkage,
        k_factor_used=40.0,
        fight_date=date(2024, 1, 1),
    )
    session.add(snap)
    session.flush()
    return snap


def _seed_two_fighters_with_elo(
    session,
    name_a: str = "Fighter A",
    name_b: str = "Fighter B",
) -> tuple[Fighter, Fighter]:
    """Insert two fighters + an Event + a Fight + 'overall' snapshots for both.

    Returns the (fighter_a, fighter_b) pair already flushed (IDs populated).
    """
    fighter_a = _make_minimal_fighter(name_a)
    fighter_b = _make_minimal_fighter(name_b)
    session.add_all([fighter_a, fighter_b])
    session.flush()

    event = Event(name="Wave 0 Test Event", date=date(2024, 1, 1), source="test")
    session.add(event)
    session.flush()

    fight = Fight(
        event_id=event.id,
        fighter_a_id=fighter_a.id,
        fighter_b_id=fighter_b.id,
        winner_id=fighter_a.id,
        weight_class="Lightweight",
        source="test",
    )
    session.add(fight)
    session.flush()

    _seed_overall_snapshot(session, fighter_a, fight, elo_after_shrinkage=1520.0)
    _seed_overall_snapshot(session, fighter_b, fight, elo_after_shrinkage=1480.0)
    return fighter_a, fighter_b


class TestPreferCanonical:
    """DEDUP-01: pure-function priority selection over a list of Fighter rows."""

    def test_prefer_canonical_prefers_ufcstats_over_kaggle(self):
        _prefer_canonical = _import_prefer_canonical()
        results = [
            Fighter(id=2, name="Joe", source="kaggle-rajeevw"),
            Fighter(id=1, name="Joe", source="ufcstats"),
        ]
        assert _prefer_canonical(results).source == "ufcstats"

    def test_prefer_canonical_kaggle_only_returns_kaggle(self):
        # D-03: kaggle-only fighters must still resolve, not raise.
        _prefer_canonical = _import_prefer_canonical()
        results = [Fighter(id=1, name="Joe", source="kaggle-rajeevw")]
        assert _prefer_canonical(results).source == "kaggle-rajeevw"

    def test_prefer_canonical_unknown_source_sorts_last(self):
        _prefer_canonical = _import_prefer_canonical()
        results = [
            Fighter(id=1, name="Joe", source="ufcstats"),
            Fighter(id=2, name="Joe", source="some-future-source"),
        ]
        assert _prefer_canonical(results).source == "ufcstats"


class TestResolveFighter:
    """DEDUP-01: ``_resolve_fighter`` must collapse same-name multi-source rows."""

    def test_resolve_fighter_duplicate_exact_returns_ufcstats(self, session):
        ufc = _make_minimal_fighter("Joe Smith", source="ufcstats")
        kag = _make_minimal_fighter("Joe Smith", source="kaggle-rajeevw")
        session.add_all([ufc, kag])
        session.flush()

        resolved = _resolve_fighter(session, "Joe Smith")
        assert resolved.source == "ufcstats"

    def test_resolve_fighter_kaggle_only_resolves_to_kaggle(self, session):
        # D-03: a fighter with only a kaggle row must still resolve.
        kag = _make_minimal_fighter("Old Fighter", source="kaggle-rajeevw")
        session.add(kag)
        session.flush()

        resolved = _resolve_fighter(session, "Old Fighter")
        assert resolved.source == "kaggle-rajeevw"

    def test_resolve_fighter_genuine_ambiguity_still_raises(self, session):
        # Two different real people both substring-match "Silva".
        a = _make_minimal_fighter("Anderson Silva", source="ufcstats")
        b = _make_minimal_fighter("Antonio Silva", source="ufcstats")
        session.add_all([a, b])
        session.flush()

        with pytest.raises(ValueError, match="Multiple fighters match"):
            _resolve_fighter(session, "Silva")

    def test_resolve_fighter_substring_matches_same_person_dedups(self, session):
        """Regression: 'Cormier' (substring) for Daniel Cormier across two
        ingest sources must resolve to the canonical UFCStats row."""
        ufc = _make_minimal_fighter("Daniel Cormier", source="ufcstats")
        kag = _make_minimal_fighter("Daniel Cormier", source="kaggle-rajeevw")
        session.add_all([ufc, kag])
        session.flush()

        resolved = _resolve_fighter(session, "Cormier")
        assert resolved.name == "Daniel Cormier"
        assert resolved.source == "ufcstats"

    def test_resolve_fighter_substring_matches_two_real_people_raises(self, session):
        """Regression: 'Topuria' substring matches Ilia AND Aleksandre — two
        real people, each with a UFCStats + Kaggle pair. After per-name
        dedup, two distinct people remain, so we must still raise."""
        ilia_ufc = _make_minimal_fighter("Ilia Topuria", source="ufcstats")
        ilia_kag = _make_minimal_fighter("Ilia Topuria", source="kaggle-rajeevw")
        alex_ufc = _make_minimal_fighter("Aleksandre Topuria", source="ufcstats")
        alex_kag = _make_minimal_fighter("Aleksandre Topuria", source="kaggle-mdabbert")
        session.add_all([ilia_ufc, ilia_kag, alex_ufc, alex_kag])
        session.flush()

        with pytest.raises(ValueError, match="Multiple fighters match"):
            _resolve_fighter(session, "Topuria")


# Phase 16-02 D-12: TestBuildFeatureVector removed. The 3 tests that
# asserted ``_build_feature_vector(session, A, B).shape == (1, 70)``
# (28-feat NaN-pad behaviour) are superseded by:
#
#   * tests/unit/ml/test_inference_features.py (LIVE-02 — full 72-feature
#     vector, NaN-tolerance, 5-feature odds parity with feature_matrix.py).
#   * tests/integration/test_live_odds_acceptance.py (Pitfall #4 — predict
#     path end-to-end cold + warm cache + JSONL trace).
#
# Per CONTEXT.md D-12: no fallback wrapper, no deprecation. Trust the
# new path. Removing the dead-symbol imports is part of the cleanup.

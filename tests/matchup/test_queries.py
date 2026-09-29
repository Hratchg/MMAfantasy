"""DB-backed tests for matchup.queries."""

from __future__ import annotations

from datetime import date

from ufc_prediction.matchup.queries import get_style_matchup_counts
from ufc_prediction.models.computed_feature import ComputedFeature
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter


def _seed_bout(session, source: str, striker_wins: bool) -> None:
    """One striker-vs-grappler bout ingested under ``source``."""
    striker = Fighter(name="Style Striker", source=source)
    grappler = Fighter(name="Style Grappler", source=source)
    session.add_all([striker, grappler])
    session.flush()
    event = Event(name=f"Card ({source})", date=date(2020, 1, 1), source=source)
    session.add(event)
    session.flush()
    fight = Fight(
        event_id=event.id,
        fighter_a_id=striker.id,
        fighter_b_id=grappler.id,
        winner_id=striker.id if striker_wins else grappler.id,
        weight_class="Lightweight",
        source=source,
    )
    session.add(fight)
    session.flush()
    for fighter, tag in ((striker, "striker"), (grappler, "grappler")):
        session.add(
            ComputedFeature(
                fighter_id=fighter.id,
                fight_id=fight.id,
                as_of_date=event.date,
                feature_set_version="v1",
                features={"style_tag": tag},
            )
        )
    session.flush()


def test_style_counts_ignore_kaggle_duplicates_of_the_same_bout(session):
    """Finding 7: Kaggle copies of a ufcstats bout are not counted again."""
    _seed_bout(session, "ufcstats", striker_wins=True)
    _seed_bout(session, "kaggle-rajeevw", striker_wins=True)
    _seed_bout(session, "kaggle-mdabbert", striker_wins=True)

    counts = get_style_matchup_counts(session)

    assert counts == {("grappler", "striker"): {"a_wins": 0, "b_wins": 1, "total": 1}}

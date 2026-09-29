"""Tests for scripts/backfill_ufcstats_odds_from_kaggle_twins.py.

Before PR #18 the BFO ingester matched fighters across every source, so odds
for ~245 bouts attached to the kaggle twin fight instead of the ufcstats one
(the only lineage training/serving read). The backfill copies those rows onto
the ufcstats fight, re-keying each kaggle fighter_id to the ufcstats fighter
with the same canonical name. No network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy.orm import Session

from tests.scripts.twin_seed import add_fight, add_odds
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fight_odds import FightOdds
from ufc_prediction.models.fighter import Fighter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "backfill_ufcstats_odds_from_kaggle_twins.py"

D = date(2019, 3, 2)


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "backfill_ufcstats_odds_from_kaggle_twins", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["backfill_ufcstats_odds_from_kaggle_twins"] = m
    spec.loader.exec_module(m)
    return m


def _odds(session: Session, fight_id: int) -> dict[int, tuple[int | None, float | None]]:
    rows = session.query(FightOdds).filter(FightOdds.fight_id == fight_id).all()
    return {r.fighter_id: (r.opening_ml, r.closing_implied_prob) for r in rows}


def _pair(session: Session, fight: Fight, a_ml: int, b_ml: int) -> None:
    add_odds(session, fight, fight.fighter_a_id, opening_ml=a_ml, closing_implied_prob=0.6)
    add_odds(session, fight, fight.fighter_b_id, opening_ml=b_ml, closing_implied_prob=0.4)


def test_orphan_odds_are_copied_by_canonical_name(mod: ModuleType, session: Session) -> None:
    # ufcstats: winner first; kaggle: red first and a spelling variant.
    ufc = add_fight(session, "ufcstats", D, "Georges St-Pierre", "B.J. Penn")
    kag = add_fight(session, "kaggle-rajeevw", D, "BJ Penn", "Georges Saint Pierre")
    add_odds(session, kag, kag.fighter_a_id, opening_ml=150, closing_implied_prob=0.38)  # Penn
    add_odds(session, kag, kag.fighter_b_id, opening_ml=-170, closing_implied_prob=0.62)  # GSP

    report = mod.backfill(session, apply=False)

    assert report.counts["backfillable_fights"] == 1
    assert report.counts["rows_planned"] == 2
    assert report.backfilled == [
        {"fight_id": ufc.id, "kaggle_fight_id": kag.id, "kaggle_source": "kaggle-rajeevw"}
    ]
    assert _odds(session, ufc.id) == {}  # report-only by default

    report = mod.backfill(session, apply=True)
    assert report.counts["rows_inserted"] == 2
    session.expire_all()
    assert _odds(session, ufc.id) == {
        ufc.fighter_a_id: (-170, 0.62),  # GSP
        ufc.fighter_b_id: (150, 0.38),  # Penn
    }
    copied = session.query(FightOdds).filter(FightOdds.fight_id == ufc.id).first()
    assert copied is not None
    assert copied.source == "bestfightodds"

    # Idempotent: the fight is no longer an orphan.
    again = mod.backfill(session, apply=True)
    assert again.counts.get("rows_inserted", 0) == 0
    assert again.backfilled == []


def test_fight_with_odds_and_fights_without_usable_twin_are_left_alone(
    mod: ModuleType, session: Session
) -> None:
    has_odds = add_fight(session, "ufcstats", D, "Has Odds", "Already")
    _pair(session, has_odds, -200, 170)
    twin = add_fight(session, "kaggle-rajeevw", D, "Has Odds", "Already")
    _pair(session, twin, -999, 999)
    no_twin = add_fight(session, "ufcstats", D, "No Twin", "Anywhere")
    bare = add_fight(session, "ufcstats", D, "Bare Twin", "No Odds")
    add_fight(session, "kaggle-rajeevw", D, "Bare Twin", "No Odds")

    report = mod.backfill(session, apply=True)

    assert report.backfilled == []
    assert report.counts["no_twin"] == 1
    assert report.counts["twin_without_odds"] == 1
    session.expire_all()
    assert _odds(session, has_odds.id)[has_odds.fighter_a_id] == (-200, 0.6)
    assert _odds(session, no_twin.id) == {}
    assert _odds(session, bare.id) == {}


def test_refuses_two_same_source_twins(mod: ModuleType, session: Session) -> None:
    ufc = add_fight(session, "ufcstats", D, "Dup A", "Dup B")
    for ml in (-150, -155):
        k = add_fight(session, "kaggle-rajeevw", D, "Dup A", "Dup B")
        _pair(session, k, ml, 130)

    report = mod.backfill(session, apply=True)

    assert report.refusals == [{"fight_id": ufc.id, "reason": "ambiguous_twin"}]
    assert report.counts["refused:ambiguous_twin"] == 1
    assert _odds(session, ufc.id) == {}


def test_cross_source_twins_must_agree(mod: ModuleType, session: Session) -> None:
    agree = add_fight(session, "ufcstats", D, "Same A", "Same B")
    for src in ("kaggle-rajeevw", "kaggle-mdabbert"):
        _pair(session, add_fight(session, src, D, "Same A", "Same B"), -140, 120)
    clash = add_fight(session, "ufcstats", D, "Clash A", "Clash B")
    _pair(session, add_fight(session, "kaggle-rajeevw", D, "Clash A", "Clash B"), -140, 120)
    _pair(session, add_fight(session, "kaggle-mdabbert", D, "Clash B", "Clash A"), -300, 250)

    report = mod.backfill(session, apply=True)

    assert [b["fight_id"] for b in report.backfilled] == [agree.id]
    assert report.backfilled[0]["kaggle_source"] == "kaggle-rajeevw"
    assert report.refusals == [{"fight_id": clash.id, "reason": "conflicting_twin_odds"}]
    session.expire_all()
    assert len(_odds(session, agree.id)) == 2
    assert _odds(session, clash.id) == {}


def test_refuses_odds_row_for_a_fighter_outside_the_twin(mod: ModuleType, session: Session) -> None:
    ufc = add_fight(session, "ufcstats", D, "Out A", "Out B")
    kag = add_fight(session, "kaggle-rajeevw", D, "Out A", "Out B")
    stranger = Fighter(name="Stranger", source="kaggle-rajeevw")
    session.add(stranger)
    session.flush()
    add_odds(session, kag, kag.fighter_a_id, opening_ml=-110)
    add_odds(session, kag, stranger.id, opening_ml=-110)

    report = mod.backfill(session, apply=True)

    assert report.refusals == [{"fight_id": ufc.id, "reason": "odds_fighter_not_in_twin"}]
    assert _odds(session, ufc.id) == {}


def test_insert_is_on_conflict_do_nothing(mod: ModuleType, session: Session) -> None:
    ufc = add_fight(session, "ufcstats", D, "Conf A", "Conf B")
    row = {
        "fight_id": ufc.id,
        "fighter_id": ufc.fighter_a_id,
        "opening_ml": -120,
        "closing_range_min_ml": None,
        "closing_range_max_ml": None,
        "opening_implied_prob": None,
        "closing_implied_prob": None,
        "source": "bestfightodds",
    }
    assert mod.insert_odds(session, [row]) == 1
    assert mod.insert_odds(session, [{**row, "opening_ml": 999}]) == 0
    session.expire_all()
    assert _odds(session, ufc.id) == {ufc.fighter_a_id: (-120, None)}


def test_report_json_shape(mod: ModuleType, session: Session) -> None:
    ufc = add_fight(session, "ufcstats", D, "Json A", "Json B")
    _pair(session, add_fight(session, "kaggle-rajeevw", D, "Json A", "Json B"), -130, 110)
    summary = mod.backfill(session, apply=False).as_dict(applied=False)
    assert json.loads(json.dumps(summary))["backfilled"][0]["fight_id"] == ufc.id
    assert summary["applied"] is False

"""Tests for scripts/split_merged_fighter.py.

Live case (S23): ufcstats ``fighters.id = 6121`` ("Bruno Silva", flyweight,
UFCStats id 294aa73dbf37d281) also carries 11 Middleweight bouts of a
different UFC fighter with the same name (UFCStats id 12ebd7d157e91701). The
script moves exactly the listed fights (plus their round_stats / fight_odds
rows, and optionally the Sherdog seed) onto a ufcstats row for the other
fighter. These tests reproduce that shape on a disposable Postgres.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, date, datetime
from itertools import count
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from ufc_prediction.db.base import Base
from ufc_prediction.models.computed_feature import ComputedFeature
from ufc_prediction.models.debutant_seed_input import DebutantSeedInput
from ufc_prediction.models.elo_snapshot import EloSnapshot
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fight_odds import FightOdds
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.models.round_stats import RoundStats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "split_merged_fighter.py"

FLY_ID = "294aa73dbf37d281"
MW_ID = "12ebd7d157e91701"
SHERDOG = "https://www.sherdog.com/fighter/Bruno-Silva-66304"

_seq = count(1)


@pytest.fixture(autouse=True)
def _schema(engine: Engine) -> None:
    """The full ORM schema (idempotent: earlier migration tests may have reset it)."""
    Base.metadata.create_all(engine)


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    spec = importlib.util.spec_from_file_location("split_merged_fighter", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["split_merged_fighter"] = m
    spec.loader.exec_module(m)
    return m


# ── seeding ──────────────────────────────────────────────────────────────────


def _fighter(session: Session, name: str, source: str, **kw: Any) -> Fighter:
    f = Fighter(name=name, source=source, **kw)
    session.add(f)
    session.flush()
    return f


def _fight(
    session: Session,
    source: str,
    a: Fighter,
    b: Fighter,
    weight_class: str,
    *,
    winner: Fighter | None,
    d: date,
) -> Fight:
    ev = Event(name=f"{source} card #{next(_seq)}", date=d, source=source)
    session.add(ev)
    session.flush()
    slug = f"{next(_seq):016x}"
    fight = Fight(
        event_id=ev.id,
        fighter_a_id=a.id,
        fighter_b_id=b.id,
        winner_id=winner.id if winner is not None else None,
        weight_class=weight_class,
        source=source,
        source_url=f"http://ufcstats.com/fight-details/{slug}" if source == "ufcstats" else None,
    )
    session.add(fight)
    session.flush()
    for fighter in (a, b):
        for rnd in (0, 1):
            session.add(
                RoundStats(
                    fight_id=fight.id,
                    fighter_id=fighter.id,
                    round_number=rnd,
                    sig_strikes_landed=fighter.id * 10 + rnd,
                )
            )
    session.flush()
    return fight


def _odds(session: Session, fight: Fight, fighter_id: int, ml: int) -> None:
    session.add(FightOdds(fight_id=fight.id, fighter_id=fighter_id, opening_ml=ml))
    session.flush()


class World:
    """The S23 shape in miniature."""

    def __init__(self, session: Session) -> None:
        self.merged = _fighter(
            session,
            "Bruno Silva",
            "ufcstats",
            source_id=FLY_ID,
            nickname="Bulldog",
            height_inches=64.0,
            reach_inches=65.0,
            stance="Orthodox",
            date_of_birth=date(1990, 3, 16),
            sherdog_url=SHERDOG,
            pre_ufc_record={"win_pct": 0.76, "total_fights": 25},
        )
        self.twin = _fighter(
            session,
            "Bruno Silva",
            "kaggle-rajeevw",
            height_inches=72.0,
            reach_inches=65.0,
            stance="Orthodox",
            date_of_birth=date(1989, 7, 13),
            sherdog_url=SHERDOG,
            pre_ufc_record={"win_pct": 0.76, "total_fights": 25},
        )
        opp = [
            _fighter(session, f"Opponent {i}", "ufcstats", source_id=f"{i:016x}") for i in range(5)
        ]
        m = self.merged
        # Middleweight bouts that belong to the OTHER Bruno Silva (to move):
        self.mw_win_a = _fight(
            session, "ufcstats", m, opp[0], "Middleweight", winner=m, d=date(2021, 6, 19)
        )
        self.mw_loss_b = _fight(
            session, "ufcstats", opp[1], m, "Middleweight", winner=opp[1], d=date(2022, 8, 13)
        )
        self.mw_nc_b = _fight(
            session, "ufcstats", opp[2], m, "Middleweight", winner=None, d=date(2023, 6, 24)
        )
        # Flyweight bouts of the real 6121 (to keep):
        self.fly_win = _fight(
            session, "ufcstats", m, opp[3], "Flyweight", winner=m, d=date(2021, 3, 20)
        )
        self.fly_loss = _fight(
            session, "ufcstats", opp[4], m, "Flyweight", winner=opp[4], d=date(2025, 6, 7)
        )
        self.move = [self.mw_win_a, self.mw_loss_b, self.mw_nc_b]
        self.keep = [self.fly_win, self.fly_loss]
        # Odds: a moved fight and a kept fight both carry a pair.
        _odds(session, self.mw_loss_b, m.id, 150)
        _odds(session, self.mw_loss_b, opp[1].id, -170)
        _odds(session, self.fly_loss, m.id, 300)
        _odds(session, self.fly_loss, opp[4].id, -400)
        # Kaggle twin fights (never touched).
        kopp = _fighter(session, "Opponent 0", "kaggle-rajeevw")
        self.kaggle_fight = _fight(
            session,
            "kaggle-rajeevw",
            self.twin,
            kopp,
            "Middleweight",
            winner=self.twin,
            d=date(2021, 6, 19),
        )
        _odds(session, self.kaggle_fight, self.twin.id, 120)
        # Sherdog seed (the record is the middleweight's).
        session.add(
            DebutantSeedInput(
                fighter_id=m.id,
                sherdog_url=SHERDOG,
                n_pre_ufc_fights=25,
                win_rate=0.76,
                org_tier="regional",
                scraped_at=datetime(2026, 7, 4, tzinfo=UTC),
            )
        )
        # Derived rows (must NOT be edited — recomputed by elo/features compute).
        for f in self.move + self.keep:
            session.add(
                EloSnapshot(
                    fighter_id=m.id,
                    fight_id=f.id,
                    division=f.weight_class,
                    elo_type="overall",
                    elo_before=1500.0,
                    elo_after=1510.0,
                    elo_after_shrinkage=1509.0,
                    k_factor_used=32.0,
                    fight_date=date(2021, 1, 1),
                )
            )
            session.add(
                ComputedFeature(
                    fighter_id=m.id,
                    fight_id=f.id,
                    as_of_date=date(2021, 1, 1),
                    feature_set_version="v2",
                    features={"x": 1},
                )
            )
        session.flush()

    def spec(self, mod: ModuleType, **kw: Any) -> Any:
        base: dict[str, Any] = {
            "from_fighter_id": self.merged.id,
            "to_source_id": MW_ID,
            "fight_ids": tuple(f.id for f in self.move),
        }
        base.update(kw)
        return mod.SplitSpec(**base)


@pytest.fixture()
def world(session: Session) -> World:
    return World(session)


_TABLES: dict[str, str] = {
    "fighters": "id",
    "fights": "id",
    "round_stats": "id",
    "fight_odds": "fight_id, fighter_id",
    "debutant_seed_inputs": "fighter_id",
    "elo_snapshots": "id",
    "computed_features": "id",
    "fighter_aliases": "id",
}


def _snapshot(session: Session, where: dict[str, str] | None = None) -> dict[str, list[Any]]:
    session.expire_all()
    out: dict[str, list[Any]] = {}
    for table, order in _TABLES.items():
        clause = f" WHERE {where[table]}" if where and table in where else ""
        out[table] = [
            tuple(r)
            for r in session.execute(text(f"SELECT * FROM {table}{clause} ORDER BY {order}"))
        ]
    return out


def _kaggle_snapshot(session: Session) -> dict[str, list[Any]]:
    return _snapshot(
        session,
        {
            "fighters": "source LIKE 'kaggle-%'",
            "fights": "source LIKE 'kaggle-%'",
            "round_stats": "fight_id IN (SELECT id FROM fights WHERE source LIKE 'kaggle-%')",
            "fight_odds": "fight_id IN (SELECT id FROM fights WHERE source LIKE 'kaggle-%')",
            "debutant_seed_inputs": "fighter_id IN (SELECT id FROM fighters WHERE source LIKE 'kaggle-%')",
            "elo_snapshots": "fight_id IN (SELECT id FROM fights WHERE source LIKE 'kaggle-%')",
            "computed_features": "fight_id IN (SELECT id FROM fights WHERE source LIKE 'kaggle-%')",
            "fighter_aliases": "source LIKE 'kaggle-%'",
        },
    )


def _new_fighter(session: Session) -> Fighter:
    session.expire_all()
    return (
        session.query(Fighter)
        .filter(Fighter.source == "ufcstats", Fighter.source_id == MW_ID)
        .one()
    )


def _sides(session: Session, fight: Fight) -> tuple[int, int, int | None]:
    session.expire_all()
    f = session.get(Fight, fight.id)
    assert f is not None
    return (f.fighter_a_id, f.fighter_b_id, f.winner_id)


def _rs_fighters(session: Session, fight: Fight) -> list[tuple[int, int]]:
    rows = session.query(RoundStats).filter(RoundStats.fight_id == fight.id).all()
    return sorted((r.fighter_id, r.round_number) for r in rows)


def _odds_map(session: Session, fight: Fight) -> dict[int, int | None]:
    rows = session.query(FightOdds).filter(FightOdds.fight_id == fight.id).all()
    return {r.fighter_id: r.opening_ml for r in rows}


# ── tests ────────────────────────────────────────────────────────────────────


def test_split_moves_exactly_the_given_fights(
    mod: ModuleType, session: Session, world: World
) -> None:
    m = world.merged.id
    before_keep = {
        f.id: (_sides(session, f), _rs_fighters(session, f), _odds_map(session, f))
        for f in world.keep
    }
    opp_rs = {f.id: [x for x in _rs_fighters(session, f) if x[0] != m] for f in world.move}

    report = mod.split(session, world.spec(mod), apply=True)

    assert report.errors == []
    new = _new_fighter(session)
    assert new.id != m
    assert (new.name, new.source, new.source_id) == ("Bruno Silva", "ufcstats", MW_ID)
    assert report.to_fighter["id"] == new.id
    assert report.to_fighter["action"] == "create"

    a, b, w = _sides(session, world.mw_win_a)
    assert (a, w) == (new.id, new.id) and b != m
    a, b, w = _sides(session, world.mw_loss_b)
    assert b == new.id and a != m and w == a
    a, b, w = _sides(session, world.mw_nc_b)
    assert b == new.id and w is None

    for f in world.move:
        rs = _rs_fighters(session, f)
        assert (new.id, 0) in rs and (new.id, 1) in rs
        assert all(fid != m for fid, _ in rs)
        assert [x for x in rs if x[0] != new.id] == opp_rs[f.id]  # opponent rows untouched
    assert _odds_map(session, world.mw_loss_b) == {new.id: 150, world.mw_loss_b.fighter_a_id: -170}

    # The flyweight's own fights are untouched.
    for f in world.keep:
        assert (_sides(session, f), _rs_fighters(session, f), _odds_map(session, f)) == before_keep[
            f.id
        ]

    assert report.counts["fights"] == 3
    assert report.counts["round_stats"] == 6
    assert report.counts["fight_odds"] == 1
    assert report.written["fights"] == 3
    assert report.written["round_stats"] == 6
    assert report.written["fight_odds"] == 1


def test_profile_comes_from_twin_row_with_overrides(
    mod: ModuleType, session: Session, world: World
) -> None:
    spec = world.spec(
        mod,
        profile_from_fighter_id=world.twin.id,
        profile_overrides={"reach_inches": 74.0, "nickname": "Blindado"},
    )
    report = mod.split(session, spec, apply=True)

    assert report.errors == []
    new = _new_fighter(session)
    assert new.height_inches == 72.0
    assert new.reach_inches == 74.0
    assert new.stance == "Orthodox"
    assert new.date_of_birth == date(1989, 7, 13)
    assert new.nickname == "Blindado"
    assert report.profile["source"] == f"fighters.id={world.twin.id} (kaggle-rajeevw)"
    assert report.profile["overrides"] == {"reach_inches": 74.0, "nickname": "Blindado"}
    # The flyweight row keeps its own profile.
    session.refresh(world.merged)
    assert (world.merged.height_inches, world.merged.nickname) == (64.0, "Bulldog")


def test_parse_overrides(mod: ModuleType) -> None:
    assert mod.parse_overrides(
        ["reach_inches=74", "nickname=Blindado", "date_of_birth=1989-07-13"]
    ) == {
        "reach_inches": 74.0,
        "nickname": "Blindado",
        "date_of_birth": date(1989, 7, 13),
    }
    with pytest.raises(ValueError, match="source_id"):
        mod.parse_overrides(["source_id=abc"])
    with pytest.raises(ValueError, match="FIELD=VALUE"):
        mod.parse_overrides(["reach_inches"])


def test_move_sherdog_moves_the_seed_and_sherdog_columns(
    mod: ModuleType, session: Session, world: World
) -> None:
    report = mod.split(session, world.spec(mod, move_sherdog=True), apply=True)

    assert report.errors == []
    new = _new_fighter(session)
    session.refresh(world.merged)
    seed_ids = [s.fighter_id for s in session.query(DebutantSeedInput).all()]
    assert seed_ids == [new.id]
    assert new.sherdog_url == SHERDOG
    assert new.pre_ufc_record == {"win_pct": 0.76, "total_fights": 25}
    assert world.merged.sherdog_url is None
    # SQL NULL, not JSON 'null' (readers filter with IS NULL / IS NOT NULL).
    assert session.execute(
        text("SELECT pre_ufc_record IS NULL FROM fighters WHERE id = :i"), {"i": world.merged.id}
    ).scalar_one()
    assert report.counts["debutant_seed_inputs"] == 1
    assert report.written["debutant_seed_inputs"] == 1
    # The kaggle twin keeps its (identical) Sherdog link: kaggle rows are never written.
    session.refresh(world.twin)
    assert world.twin.sherdog_url == SHERDOG


def test_without_move_sherdog_the_seed_stays_and_is_reported(
    mod: ModuleType, session: Session, world: World
) -> None:
    report = mod.split(session, world.spec(mod), apply=True)

    assert [s.fighter_id for s in session.query(DebutantSeedInput).all()] == [world.merged.id]
    assert report.sherdog["action"] == "left_on_source"
    assert report.sherdog["seed_sherdog_url"] == SHERDOG
    assert _new_fighter(session).sherdog_url is None


def test_never_touches_kaggle_rows(mod: ModuleType, session: Session, world: World) -> None:
    before = _kaggle_snapshot(session)
    spec = world.spec(
        mod,
        profile_from_fighter_id=world.twin.id,
        profile_overrides={"reach_inches": 74.0},
        move_sherdog=True,
    )
    assert mod.split(session, spec, apply=True).errors == []
    assert _kaggle_snapshot(session) == before


def test_refuses_kaggle_fight_or_kaggle_source_fighter(
    mod: ModuleType, session: Session, world: World
) -> None:
    before = _snapshot(session)

    r1 = mod.split(session, world.spec(mod, fight_ids=(world.kaggle_fight.id,)), apply=True)
    r2 = mod.split(
        session,
        world.spec(mod, from_fighter_id=world.twin.id, fight_ids=(world.kaggle_fight.id,)),
        apply=True,
    )

    assert any("not a ufcstats fight" in e for e in r1.errors)
    assert any("not a ufcstats fighter" in e for e in r2.errors)
    assert _snapshot(session) == before


def test_refuses_fight_that_is_not_on_the_source_fighter(
    mod: ModuleType, session: Session, world: World
) -> None:
    stranger_a = _fighter(session, "Someone", "ufcstats", source_id="aaaaaaaaaaaaaaaa")
    stranger_b = _fighter(session, "Else", "ufcstats", source_id="bbbbbbbbbbbbbbbb")
    other = _fight(
        session,
        "ufcstats",
        stranger_a,
        stranger_b,
        "Middleweight",
        winner=stranger_a,
        d=date(2022, 1, 1),
    )
    before = _snapshot(session)

    report = mod.split(
        session, world.spec(mod, fight_ids=(world.mw_win_a.id, other.id)), apply=True
    )

    assert any(f"fight {other.id}" in e for e in report.errors)
    assert report.written == {}
    assert _snapshot(session) == before


def test_split_is_idempotent(mod: ModuleType, session: Session, world: World) -> None:
    spec = world.spec(
        mod,
        profile_from_fighter_id=world.twin.id,
        profile_overrides={"reach_inches": 74.0, "nickname": "Blindado"},
        move_sherdog=True,
    )
    first = mod.split(session, spec, apply=True)
    assert first.errors == []
    after_first = _snapshot(session)

    second = mod.split(session, spec, apply=True)

    assert second.errors == []
    assert second.to_fighter["action"] == "reuse"
    assert sorted(second.already_moved) == sorted(f.id for f in world.move)
    assert second.counts["fights"] == 0
    assert second.counts["round_stats"] == 0
    assert second.counts["fight_odds"] == 0
    assert second.counts["debutant_seed_inputs"] == 0
    assert second.profile["fill"] == {}
    assert sum(second.written.values()) == 0
    assert _snapshot(session) == after_first
    assert session.query(Fighter).filter(Fighter.source_id == MW_ID).count() == 1


def test_dry_run_writes_nothing(mod: ModuleType, session: Session, world: World) -> None:
    before = _snapshot(session)
    spec = world.spec(
        mod,
        profile_from_fighter_id=world.twin.id,
        profile_overrides={"reach_inches": 74.0},
        move_sherdog=True,
    )

    report = mod.split(session, spec, apply=False)

    assert report.errors == []
    assert report.applied is False
    assert report.to_fighter == {
        "id": None,
        "action": "create",
        "name": "Bruno Silva",
        "source_id": MW_ID,
    }
    assert report.counts["fights"] == 3
    assert report.counts["fight_columns"] == 4  # a/b on 3 fights + 1 winner
    assert report.counts["round_stats"] == 6
    assert report.counts["fight_odds"] == 1
    assert report.counts["debutant_seed_inputs"] == 1
    assert report.profile["fill"]["height_inches"] == 72.0
    assert report.written == {}
    assert not session.new and not session.dirty
    assert _snapshot(session) == before


def test_derived_tables_are_reported_not_edited(
    mod: ModuleType, session: Session, world: World
) -> None:
    before = _snapshot(session)

    report = mod.split(session, world.spec(mod), apply=True)

    assert report.derived_stale == {"elo_snapshots": 3, "computed_features": 3}
    after = _snapshot(session)
    assert after["elo_snapshots"] == before["elo_snapshots"]
    assert after["computed_features"] == before["computed_features"]


def test_reuses_an_existing_target_row_and_fills_only_null_fields(
    mod: ModuleType, session: Session, world: World
) -> None:
    existing = _fighter(session, "Bruno Silva", "ufcstats", source_id=MW_ID, reach_inches=74.0)
    spec = world.spec(mod, profile_from_fighter_id=world.twin.id)

    report = mod.split(session, spec, apply=True)

    assert report.errors == []
    assert report.to_fighter == {
        "id": existing.id,
        "action": "reuse",
        "name": "Bruno Silva",
        "source_id": MW_ID,
    }
    assert report.profile["fill"] == {
        "height_inches": 72.0,
        "stance": "Orthodox",
        "date_of_birth": date(1989, 7, 13),
    }
    session.refresh(existing)
    assert existing.reach_inches == 74.0  # not clobbered by the twin's 65
    assert existing.height_inches == 72.0
    assert _sides(session, world.mw_win_a)[0] == existing.id


def test_refuses_fight_odds_conflict(mod: ModuleType, session: Session, world: World) -> None:
    existing = _fighter(session, "Bruno Silva", "ufcstats", source_id=MW_ID)
    _odds(session, world.mw_loss_b, existing.id, 999)
    before = _snapshot(session)

    report = mod.split(session, world.spec(mod), apply=True)

    assert any("fight_odds conflict" in e for e in report.errors)
    assert _snapshot(session) == before


def test_page_check_requires_the_target_id_on_each_fight_page(
    mod: ModuleType, session: Session, world: World, tmp_path: Path
) -> None:
    def write_page(fight: Fight, *ids: str) -> None:
        assert fight.source_url is not None
        slug = fight.source_url.rstrip("/").rsplit("/", 1)[-1]
        links = "".join(f'<a href="http://ufcstats.com/fighter-details/{i}">x</a>' for i in ids)
        (tmp_path / f"{slug}.html").write_text(f"<html>{links}</html>", encoding="utf-8")

    write_page(world.mw_win_a, MW_ID, "0000000000000000")
    write_page(world.mw_loss_b, "0000000000000001", MW_ID)
    write_page(world.mw_nc_b, FLY_ID, "0000000000000002")  # wrong fighter on the page

    bad = mod.split(session, world.spec(mod, page_cache_dir=tmp_path), apply=False)
    assert any(f"fight {world.mw_nc_b.id}" in e and "page" in e for e in bad.errors)

    ok = mod.split(
        session,
        world.spec(mod, page_cache_dir=tmp_path, fight_ids=(world.mw_win_a.id, world.mw_loss_b.id)),
        apply=False,
    )
    assert ok.errors == []
    assert ok.pages_checked == 2

    missing = mod.split(
        session,
        world.spec(mod, page_cache_dir=tmp_path / "nope", fight_ids=(world.mw_win_a.id,)),
        apply=False,
    )
    assert any("no cached page" in e for e in missing.errors)

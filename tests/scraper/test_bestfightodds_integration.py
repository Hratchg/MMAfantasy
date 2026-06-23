"""DB integration tests for the Phase 13 Plan 02 BestFightOdds pipeline.

Scope:
- Task 1: FightOdds model + fight_odds table schema (via Alembic migration).
- Task 2: BFOOddsIngester CSV -> validate -> match -> devig -> upsert.

All DB-level tests run against the testcontainers PostgreSQL session
fixture (``session``) from ``tests/conftest.py``. The fixture creates
tables via ``Base.metadata.create_all`` so the ``fight_odds`` table is
registered by importing ``FightOdds`` via ``ufc_prediction.models``.

The downgrade-drops-table test additionally invokes Alembic programmatically
on its own engine/database, then upgrades back to head for isolation.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fight_odds import FightOdds
from ufc_prediction.models.fighter import Fighter

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ── Seed helpers ────────────────────────────────────────────────────────────


def _seed_basic_fighters(session: Session) -> dict[str, Fighter]:
    """Insert a deterministic set of fighters matching the Plan 01 CSV
    fixtures' fighters_names rows.

    Returns a name -> Fighter map for assertion use.
    """
    fighters = {
        "jones": Fighter(name="Jon Jones", source="test"),
        "gus": Fighter(name="Alexander Gustafsson", source="test"),
        "khabib": Fighter(name="Khabib Nurmagomedov", source="test"),
        "mcgregor": Fighter(name="Conor McGregor", source="test"),
        "adesanya": Fighter(name="Israel Adesanya", source="test"),
        "whittaker": Fighter(name="Robert Whittaker", source="test"),
    }
    session.add_all(list(fighters.values()))
    session.flush()
    return fighters


def _seed_events_and_fights(
    session: Session, fighters: dict[str, Fighter]
) -> dict[str, Fight]:
    """Create events + fights linking the seeded fighters.

    Fight 1: Jones vs Gustafsson (UFC 165, 2013)
    Fight 2: Khabib vs McGregor  (UFC 229, 2018)
    Fight 3: Adesanya vs Whittaker (UFC 243, 2019)
    """
    ev1 = Event(name="UFC 165", date=date(2013, 9, 21), source="test")
    ev2 = Event(name="UFC 229", date=date(2018, 10, 6), source="test")
    ev3 = Event(name="UFC 243", date=date(2019, 10, 5), source="test")
    session.add_all([ev1, ev2, ev3])
    session.flush()

    fight_jg = Fight(
        event_id=ev1.id,
        fighter_a_id=fighters["jones"].id,
        fighter_b_id=fighters["gus"].id,
        weight_class="Light Heavyweight",
        source="test",
    )
    fight_km = Fight(
        event_id=ev2.id,
        fighter_a_id=fighters["khabib"].id,
        fighter_b_id=fighters["mcgregor"].id,
        weight_class="Lightweight",
        source="test",
    )
    fight_aw = Fight(
        event_id=ev3.id,
        fighter_a_id=fighters["adesanya"].id,
        fighter_b_id=fighters["whittaker"].id,
        weight_class="Middleweight",
        source="test",
    )
    session.add_all([fight_jg, fight_km, fight_aw])
    session.flush()
    return {"jg": fight_jg, "km": fight_km, "aw": fight_aw}


# ── Task 1: Schema/model smoke tests ────────────────────────────────────────


class TestFightOddsTable:
    """Verify the fight_odds table exists with the correct schema."""

    def test_fight_odds_table_exists_after_migration(self, session: Session) -> None:
        result = session.execute(text("SELECT to_regclass('fight_odds')")).scalar()
        assert result == "fight_odds"

    def test_fight_odds_has_composite_pk(self, session: Session) -> None:
        insp = inspect(session.bind)
        pk = insp.get_pk_constraint("fight_odds")
        assert set(pk["constrained_columns"]) == {"fight_id", "fighter_id"}
        assert pk["name"] == "pk_fight_odds"

    def test_fight_odds_has_foreign_keys(self, session: Session) -> None:
        insp = inspect(session.bind)
        fks = insp.get_foreign_keys("fight_odds")
        fk_by_col = {fk["constrained_columns"][0]: fk for fk in fks}
        assert "fight_id" in fk_by_col
        assert fk_by_col["fight_id"]["referred_table"] == "fights"
        assert fk_by_col["fight_id"]["referred_columns"] == ["id"]
        assert "fighter_id" in fk_by_col
        assert fk_by_col["fighter_id"]["referred_table"] == "fighters"
        assert fk_by_col["fighter_id"]["referred_columns"] == ["id"]


class TestFightOddsModel:
    """Round-trip + constraint tests for the FightOdds model."""

    def test_fight_odds_insert_and_query(self, session: Session) -> None:
        fighters = _seed_basic_fighters(session)
        fights = _seed_events_and_fights(session, fighters)

        session.add(
            FightOdds(
                fight_id=fights["jg"].id,
                fighter_id=fighters["jones"].id,
                opening_ml=-200,
                closing_range_min_ml=-205,
                closing_range_max_ml=-195,
                opening_implied_prob=0.6429,
                closing_implied_prob=0.6530,
            )
        )
        session.flush()

        row = (
            session.query(FightOdds)
            .filter_by(fight_id=fights["jg"].id, fighter_id=fighters["jones"].id)
            .one()
        )
        assert row.opening_ml == -200
        assert row.closing_range_min_ml == -205
        assert row.closing_range_max_ml == -195
        assert row.opening_implied_prob == pytest.approx(0.6429)
        assert row.closing_implied_prob == pytest.approx(0.6530)
        assert row.source == "bestfightodds"
        assert isinstance(row.scraped_at, datetime)

    def test_fight_odds_composite_pk_prevents_duplicate(
        self, session: Session
    ) -> None:
        fighters = _seed_basic_fighters(session)
        fights = _seed_events_and_fights(session, fighters)

        session.add(
            FightOdds(
                fight_id=fights["jg"].id,
                fighter_id=fighters["jones"].id,
                opening_ml=-200,
            )
        )
        session.flush()

        session.add(
            FightOdds(
                fight_id=fights["jg"].id,
                fighter_id=fighters["jones"].id,
                opening_ml=-210,
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()

    def test_fight_odds_allows_null_odds_columns(self, session: Session) -> None:
        fighters = _seed_basic_fighters(session)
        fights = _seed_events_and_fights(session, fighters)

        # Pre-2007 missing-data case (D-07): all ML columns None.
        session.add(
            FightOdds(
                fight_id=fights["aw"].id,
                fighter_id=fighters["adesanya"].id,
                opening_ml=None,
                closing_range_min_ml=None,
                closing_range_max_ml=None,
                opening_implied_prob=None,
                closing_implied_prob=None,
            )
        )
        session.flush()

        row = (
            session.query(FightOdds)
            .filter_by(
                fight_id=fights["aw"].id, fighter_id=fighters["adesanya"].id
            )
            .one()
        )
        assert row.opening_ml is None
        assert row.closing_range_min_ml is None
        assert row.closing_range_max_ml is None
        assert row.opening_implied_prob is None
        assert row.closing_implied_prob is None

    def test_fight_odds_scraped_at_defaults_to_now(self, session: Session) -> None:
        fighters = _seed_basic_fighters(session)
        fights = _seed_events_and_fights(session, fighters)

        # Do NOT specify scraped_at — let the server default fill it.
        session.execute(
            text(
                "INSERT INTO fight_odds (fight_id, fighter_id) "
                "VALUES (:fid, :figid)"
            ),
            {"fid": fights["jg"].id, "figid": fighters["jones"].id},
        )
        session.flush()
        row = session.execute(
            text(
                "SELECT scraped_at FROM fight_odds WHERE fight_id = :fid "
                "AND fighter_id = :figid"
            ),
            {"fid": fights["jg"].id, "figid": fighters["jones"].id},
        ).first()
        assert row is not None
        assert row[0] is not None
        # Allow a 60-second tolerance — the test session may run before/after
        # the server clock by a small margin.
        now_utc = datetime.now(UTC).replace(tzinfo=None)
        delta = abs(row[0] - now_utc)
        assert delta < timedelta(seconds=60)

    def test_fight_odds_source_defaults_to_bestfightodds(
        self, session: Session
    ) -> None:
        fighters = _seed_basic_fighters(session)
        fights = _seed_events_and_fights(session, fighters)

        # Use raw SQL so SQLAlchemy doesn't send its Python-side default.
        session.execute(
            text(
                "INSERT INTO fight_odds (fight_id, fighter_id) "
                "VALUES (:fid, :figid)"
            ),
            {"fid": fights["jg"].id, "figid": fighters["jones"].id},
        )
        session.flush()
        row = session.execute(
            text(
                "SELECT source FROM fight_odds WHERE fight_id = :fid "
                "AND fighter_id = :figid"
            ),
            {"fid": fights["jg"].id, "figid": fighters["jones"].id},
        ).first()
        assert row is not None
        assert row[0] == "bestfightodds"


class TestFightOddsMigration:
    """Alembic upgrade/downgrade roundtrip on a standalone container."""

    def test_downgrade_drops_table(self) -> None:
        """Invoke Alembic programmatically on a dedicated container."""
        from alembic import command
        from alembic.config import Config
        from sqlalchemy import create_engine
        from testcontainers.postgres import PostgresContainer

        project_root = Path(__file__).resolve().parents[2]
        alembic_ini = project_root / "alembic.ini"
        assert alembic_ini.exists(), alembic_ini

        with PostgresContainer("postgres:18", driver="psycopg") as pg:
            url = pg.get_connection_url()
            cfg = Config(str(alembic_ini))
            cfg.set_main_option("script_location", str(project_root / "migrations"))
            cfg.set_main_option("sqlalchemy.url", url)

            command.upgrade(cfg, "head")
            engine = create_engine(url)
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT to_regclass('fight_odds')")
                ).scalar()
            assert result == "fight_odds"

            command.downgrade(cfg, "-1")
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT to_regclass('fight_odds')")
                ).scalar()
            assert result is None

            # Upgrade back to head for cleanliness.
            command.upgrade(cfg, "head")
            engine.dispose()


# ── Task 2: BFOOddsIngester ─────────────────────────────────────────────────


class TestBFOOddsIngester:
    """End-to-end tests for BFOOddsIngester.ingest_all() against the
    testcontainers DB + Plan 01 fixture CSVs."""

    def test_ingester_reads_fixture_csv(self, session: Session) -> None:
        from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester

        fighters = _seed_basic_fighters(session)
        _seed_events_and_fights(session, fighters)
        session.flush()

        summary = BFOOddsIngester(session, FIXTURES_DIR).ingest_all()

        # The fixture has 3 BFO fights (one with all-None odds).
        assert summary.bfo_fights_scanned == 3
        # All 6 fighters match; all 3 fights resolve to DB fights.
        assert summary.fighters_matched == 6
        assert summary.fighters_unmatched == 0
        assert summary.fights_matched == 3
        assert summary.rows_upserted == 6  # 3 fights * 2 fighters

        # Spot-check the Jones row.
        jones_row = (
            session.query(FightOdds)
            .filter_by(fighter_id=fighters["jones"].id)
            .one()
        )
        assert jones_row.opening_ml == -200
        assert jones_row.opening_implied_prob == pytest.approx(0.6429, abs=1e-3)

    def test_ingester_computes_devig_per_fight(self, session: Session) -> None:
        from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester

        fighters = _seed_basic_fighters(session)
        _seed_events_and_fights(session, fighters)
        session.flush()

        BFOOddsIngester(session, FIXTURES_DIR).ingest_all()

        jones = (
            session.query(FightOdds)
            .filter_by(fighter_id=fighters["jones"].id)
            .one()
        )
        gus = (
            session.query(FightOdds)
            .filter_by(fighter_id=fighters["gus"].id)
            .one()
        )
        # D-02 normalization: the two opening probs must sum to exactly 1.0.
        assert jones.opening_implied_prob is not None
        assert gus.opening_implied_prob is not None
        total = jones.opening_implied_prob + gus.opening_implied_prob
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_ingester_computes_closing_consensus(self, session: Session) -> None:
        from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester

        fighters = _seed_basic_fighters(session)
        _seed_events_and_fights(session, fighters)
        session.flush()

        BFOOddsIngester(session, FIXTURES_DIR).ingest_all()

        jones = (
            session.query(FightOdds)
            .filter_by(fighter_id=fighters["jones"].id)
            .one()
        )
        gus = (
            session.query(FightOdds)
            .filter_by(fighter_id=fighters["gus"].id)
            .one()
        )
        assert jones.closing_implied_prob is not None
        assert gus.closing_implied_prob is not None
        total = jones.closing_implied_prob + gus.closing_implied_prob
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_ingester_preserves_raw_ml(self, session: Session) -> None:
        from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester

        fighters = _seed_basic_fighters(session)
        _seed_events_and_fights(session, fighters)
        session.flush()

        BFOOddsIngester(session, FIXTURES_DIR).ingest_all()

        jones = (
            session.query(FightOdds)
            .filter_by(fighter_id=fighters["jones"].id)
            .one()
        )
        # D-03: the raw American moneylines must be stored as-is.
        assert jones.opening_ml == -200
        assert jones.closing_range_min_ml == -205
        assert jones.closing_range_max_ml == -195

    def test_ingester_handles_missing_odds(self, session: Session) -> None:
        from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester

        fighters = _seed_basic_fighters(session)
        _seed_events_and_fights(session, fighters)
        session.flush()

        BFOOddsIngester(session, FIXTURES_DIR).ingest_all()

        adesanya = (
            session.query(FightOdds)
            .filter_by(fighter_id=fighters["adesanya"].id)
            .one()
        )
        whittaker = (
            session.query(FightOdds)
            .filter_by(fighter_id=fighters["whittaker"].id)
            .one()
        )
        # D-07: missing odds flow through as SQL NULL — never 0 / NaN.
        for row in (adesanya, whittaker):
            assert row.opening_ml is None
            assert row.closing_range_min_ml is None
            assert row.closing_range_max_ml is None
            assert row.opening_implied_prob is None
            assert row.closing_implied_prob is None

    def test_ingester_is_idempotent(self, session: Session) -> None:
        from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester

        fighters = _seed_basic_fighters(session)
        _seed_events_and_fights(session, fighters)
        session.flush()

        first = BFOOddsIngester(session, FIXTURES_DIR).ingest_all()
        count_after_first = session.query(FightOdds).count()

        second = BFOOddsIngester(session, FIXTURES_DIR).ingest_all()
        count_after_second = session.query(FightOdds).count()

        # Row count must not grow on re-run — upsert is idempotent.
        assert count_after_first == count_after_second
        assert first.rows_upserted == second.rows_upserted

    def test_ingester_reports_unmatched_fighters(
        self, session: Session
    ) -> None:
        from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester

        # Seed only the Jones/Gus pair; others in the fixture have no DB
        # counterpart so match_bfo_name should return None (threshold=80).
        jones = Fighter(name="Jon Jones", source="test")
        gus = Fighter(name="Alexander Gustafsson", source="test")
        session.add_all([jones, gus])
        session.flush()

        ev = Event(name="UFC 165", date=date(2013, 9, 21), source="test")
        session.add(ev)
        session.flush()
        session.add(
            Fight(
                event_id=ev.id,
                fighter_a_id=jones.id,
                fighter_b_id=gus.id,
                weight_class="Light Heavyweight",
                source="test",
            )
        )
        session.flush()

        summary = BFOOddsIngester(session, FIXTURES_DIR).ingest_all()

        # 4 unmatched (khabib, mcgregor, adesanya, whittaker)
        assert summary.fighters_unmatched >= 1
        # Only Jones/Gus fight should have odds rows.
        total_rows = session.query(FightOdds).count()
        assert total_rows == 2

    def test_ingester_uses_latest_fight_for_fighter_pair(
        self, session: Session
    ) -> None:
        """Pitfall 1 disambiguation: when multiple DB fights share a pair,
        prefer the most recent. The fixture has no date-window data so we
        verify the deterministic ORDER BY Event.date DESC fallback."""
        from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester

        fighters = _seed_basic_fighters(session)
        _seed_events_and_fights(session, fighters)

        # Add a second Khabib vs McGregor fight ~180 days earlier than the
        # seeded UFC 229. The ingester must pick UFC 229 (later date).
        early_event = Event(
            name="Fictional Early",
            date=date(2018, 4, 10),
            source="test",
        )
        session.add(early_event)
        session.flush()
        early_fight = Fight(
            event_id=early_event.id,
            fighter_a_id=fighters["khabib"].id,
            fighter_b_id=fighters["mcgregor"].id,
            weight_class="Lightweight",
            source="test",
        )
        session.add(early_fight)
        session.flush()

        BFOOddsIngester(session, FIXTURES_DIR).ingest_all()

        # Khabib's odds row must be attached to the LATER fight.
        khabib_row = (
            session.query(FightOdds)
            .filter_by(fighter_id=fighters["khabib"].id)
            .one()
        )
        later_fight = (
            session.query(Fight)
            .join(Event, Fight.event_id == Event.id)
            .filter(Fight.fighter_a_id == fighters["khabib"].id)
            .filter(Fight.fighter_b_id == fighters["mcgregor"].id)
            .order_by(Event.date.desc())
            .first()
        )
        assert khabib_row.fight_id == later_fight.id

    def test_ingester_reports_coverage_pct(self, session: Session) -> None:
        from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester, IngestSummary

        fighters = _seed_basic_fighters(session)
        _seed_events_and_fights(session, fighters)
        session.flush()

        summary = BFOOddsIngester(session, FIXTURES_DIR).ingest_all()

        assert isinstance(summary, IngestSummary)
        assert 0.0 <= summary.coverage_pct <= 1.0
        expected = summary.fights_matched / summary.bfo_fights_scanned
        assert summary.coverage_pct == pytest.approx(expected)

    def test_coverage_pct_zero_when_no_fights_scanned(self) -> None:
        """IngestSummary is a plain dataclass — its coverage property must
        handle the empty-input case without ZeroDivisionError."""
        from ufc_prediction.scraper.bfo_ingest import IngestSummary

        empty = IngestSummary()
        assert empty.coverage_pct == 0.0

    # ── 999.4: closest-date matching in _resolve_fight ────────────────────

    def test_resolve_fight_uses_event_date_to_disambiguate_rematch(
        self, session: Session
    ) -> None:
        """Per backlog 999.4, _resolve_fight must use the BFO event_date
        to pick between candidate rematches — not always the most recent.

        Seed: a single fighter pair with two real bouts (e.g. Cormier vs
        Jones I in 2015-01-03 and II in 2017-07-29). Resolving a BFO row
        whose event_date is 2015-01-03 must return the 2015 fight.id;
        2017-07-29 must return the 2017 fight.id.
        """
        from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester

        fa = Fighter(name="Daniel Cormier", source="test")
        fb = Fighter(name="Jon Jones", source="test")
        session.add_all([fa, fb])
        session.flush()

        ev_i = Event(name="UFC 182", date=date(2015, 1, 3), source="test")
        ev_ii = Event(name="UFC 214", date=date(2017, 7, 29), source="test")
        session.add_all([ev_i, ev_ii])
        session.flush()

        fight_i = Fight(
            event_id=ev_i.id,
            fighter_a_id=fa.id,
            fighter_b_id=fb.id,
            weight_class="Light Heavyweight",
            source="test",
        )
        fight_ii = Fight(
            event_id=ev_ii.id,
            fighter_a_id=fa.id,
            fighter_b_id=fb.id,
            weight_class="Light Heavyweight",
            source="test",
        )
        session.add_all([fight_i, fight_ii])
        session.flush()

        ingester = BFOOddsIngester(
            session, FIXTURES_DIR, date_window_days=14
        )

        # The 2015 BFO event_date must resolve to fight_i, NOT the
        # always-most-recent fight_ii (which was the H1 bug).
        assert ingester._resolve_fight(
            fa.id, fb.id, date(2015, 1, 3)
        ) == fight_i.id
        assert ingester._resolve_fight(
            fa.id, fb.id, date(2017, 7, 29)
        ) == fight_ii.id

    def test_resolve_fight_within_date_window(
        self, session: Session
    ) -> None:
        """A BFO event_date within ``date_window_days`` of the DB Event.date
        still resolves to that fight. Boundary: 14 days off.
        """
        from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester

        fa = Fighter(name="Conor McGregor", source="test")
        fb = Fighter(name="Nate Diaz", source="test")
        session.add_all([fa, fb])
        session.flush()

        ev = Event(name="UFC 202", date=date(2017, 7, 29), source="test")
        session.add(ev)
        session.flush()

        fight = Fight(
            event_id=ev.id,
            fighter_a_id=fa.id,
            fighter_b_id=fb.id,
            weight_class="Welterweight",
            source="test",
        )
        session.add(fight)
        session.flush()

        ingester = BFOOddsIngester(
            session, FIXTURES_DIR, date_window_days=14
        )

        # 14 days earlier — at the boundary — must still match.
        assert ingester._resolve_fight(
            fa.id, fb.id, date(2017, 7, 15)
        ) == fight.id

    def test_resolve_fight_outside_date_window_returns_none(
        self, session: Session
    ) -> None:
        """A BFO event_date outside ``date_window_days`` returns None —
        we refuse to attach odds to a fight that's not actually the right
        one (better to drop than mis-attribute, per the bug we're fixing).
        """
        from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester

        fa = Fighter(name="Max Holloway", source="test")
        fb = Fighter(name="Alexander Volkanovski", source="test")
        session.add_all([fa, fb])
        session.flush()

        ev = Event(name="UFC 245", date=date(2017, 7, 29), source="test")
        session.add(ev)
        session.flush()

        fight = Fight(
            event_id=ev.id,
            fighter_a_id=fa.id,
            fighter_b_id=fb.id,
            weight_class="Featherweight",
            source="test",
        )
        session.add(fight)
        session.flush()

        ingester = BFOOddsIngester(
            session, FIXTURES_DIR, date_window_days=14
        )

        # ~58 days off — well outside the 14-day window.
        assert (
            ingester._resolve_fight(fa.id, fb.id, date(2017, 6, 1)) is None
        )

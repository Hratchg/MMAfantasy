"""Integration tests for Phase 22 REF-V22-01 ingest wire-through.

Covers: parse_fight_detail.referee -> upsert_referee -> events.referee_id population.
Uses the session fixture from tests/conftest.py (transactional rollback per test).
"""

from __future__ import annotations

from collections import Counter
from datetime import date

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from ufc_prediction.data.upsert import upsert_event, upsert_referee
from ufc_prediction.models import Event, Referee


@pytest.fixture
def sample_event(session: Session) -> Event:
    """Insert a minimal Event row for per-event referee_id smoke."""
    event = upsert_event(
        session,
        name="UFC Test Phase22",
        event_date=date(2026, 5, 16),
        location="Test Arena",
        source="ufcstats",
        source_url="http://example.com/event/test-phase22",
    )
    session.flush()
    return event


class TestUpsertReferee:
    def test_upsert_creates_new_referee(self, session: Session) -> None:
        ref_id = upsert_referee(session, raw_name="Herb Dean")
        assert ref_id is not None
        ref = session.get(Referee, ref_id)
        assert ref.name == "Herb Dean"
        assert ref.normalized_name == "herb-dean"
        assert ref.alias_list == ["Herb Dean"]

    def test_upsert_returns_existing_for_alias(self, session: Session) -> None:
        first_id = upsert_referee(session, raw_name="Herb Dean")
        # "Herbert Dean" -> normalized "herbert-dean" -> REFEREE_ALIASES -> "herb-dean"
        second_id = upsert_referee(session, raw_name="Herbert Dean")
        assert first_id == second_id
        ref = session.get(Referee, first_id)
        assert "Herbert Dean" in (ref.alias_list or [])
        assert "Herb Dean" in (ref.alias_list or [])

    def test_upsert_handles_unicode_fold(self, session: Session) -> None:
        ref_id = upsert_referee(session, raw_name="Mike Beltrán")
        ref = session.get(Referee, ref_id)
        assert ref.normalized_name == "mike-beltran"

    def test_upsert_returns_none_for_none(self, session: Session) -> None:
        assert upsert_referee(session, raw_name=None) is None

    def test_upsert_returns_none_for_empty_string(self, session: Session) -> None:
        assert upsert_referee(session, raw_name="") is None

    def test_upsert_returns_none_for_whitespace_only(self, session: Session) -> None:
        assert upsert_referee(session, raw_name="   ") is None

    def test_alias_dedup(self, session: Session) -> None:
        # Re-ingesting the same raw name should NOT duplicate in alias_list
        upsert_referee(session, raw_name="Marc Goddard")
        upsert_referee(session, raw_name="Marc Goddard")
        ref = session.execute(
            select(Referee).where(Referee.normalized_name == "marc-goddard")
        ).scalar_one()
        assert ref.alias_list == ["Marc Goddard"]  # no duplicate


class TestIngestRefereePersistence:
    """Per-event aggregation smoke: most-frequent raw -> upsert -> events.referee_id."""

    def test_homogeneous_referee_picks_canonical(
        self, session: Session, sample_event: Event
    ) -> None:
        # 3 fights all with "Herb Dean" -> events.referee_id resolves to herb-dean
        raw = ["Herb Dean", "Herb Dean", "Herb Dean"]
        most = Counter(raw).most_common(1)[0][0]
        sample_event.referee_id = upsert_referee(session, raw_name=most)
        session.flush()
        assert sample_event.referee_id is not None
        ref = session.get(Referee, sample_event.referee_id)
        assert ref.normalized_name == "herb-dean"

    def test_mixed_referees_picks_most_frequent(
        self, session: Session, sample_event: Event
    ) -> None:
        # 2 Herb Dean + 1 Marc Goddard -> most_common == Herb Dean
        raw = ["Herb Dean", "Herb Dean", "Marc Goddard"]
        most = Counter(raw).most_common(1)[0][0]
        sample_event.referee_id = upsert_referee(session, raw_name=most)
        session.flush()
        assert sample_event.referee_id is not None
        ref = session.get(Referee, sample_event.referee_id)
        assert ref.normalized_name == "herb-dean"

    def test_all_none_referees_keeps_event_referee_id_null(
        self, session: Session, sample_event: Event
    ) -> None:
        # All None -> filter yields empty list -> skip upsert -> stays NULL
        raw_filtered = [r for r in [None, None, None] if r]
        if raw_filtered:
            sample_event.referee_id = upsert_referee(
                session, raw_name=Counter(raw_filtered).most_common(1)[0][0]
            )
        session.flush()
        assert sample_event.referee_id is None

    def test_event_referee_id_fk_persisted(self, session: Session, sample_event: Event) -> None:
        # Verify the FK actually round-trips through SQL select
        sample_event.referee_id = upsert_referee(session, raw_name="John McCarthy")
        session.flush()
        refetched = session.execute(select(Event).where(Event.id == sample_event.id)).scalar_one()
        assert refetched.referee_id == sample_event.referee_id
        ref = session.get(Referee, refetched.referee_id)
        assert ref.normalized_name == "john-mccarthy"


class TestIngestRefereeIdempotency:
    """CR-01 regression: re-ingest must NOT overwrite an already-resolved referee_id.

    Simulates the per-event aggregation block in ingest._scrape_event (the
    ``if event_referee_raw_names:`` guard at ingest.py:460-466). Verifies the
    fixed branch only assigns ``db_event.referee_id`` when it is currently
    ``None``, mirroring the "non-None-only" overwrite policy of upsert_event /
    upsert_fight / upsert_fighter.
    """

    @staticmethod
    def _apply_aggregation(session: Session, db_event: Event, raw_names: list[str]) -> None:
        """Mirror of the fixed branch in ingest._scrape_event (CR-01)."""
        if raw_names:
            most_frequent_referee = Counter(raw_names).most_common(1)[0][0]
            new_ref_id = upsert_referee(session, raw_name=most_frequent_referee)
            if db_event.referee_id is None and new_ref_id is not None:
                db_event.referee_id = new_ref_id

    def test_reingest_does_not_overwrite_already_resolved_referee_id(
        self, session: Session, sample_event: Event
    ) -> None:
        # First ingest: 3 Herb Dean -> events.referee_id = herb-dean
        self._apply_aggregation(session, sample_event, ["Herb Dean", "Herb Dean", "Herb Dean"])
        session.flush()
        first_ref_id = sample_event.referee_id
        assert first_ref_id is not None
        first_ref = session.get(Referee, first_ref_id)
        assert first_ref.normalized_name == "herb-dean"

        # Partial re-ingest: only 1 fight came back, and it now reports a
        # DIFFERENT referee (HTML drift, or a network glitch dropped the
        # other two pages). The previously-correct herb-dean FK MUST be
        # preserved (idempotency).
        self._apply_aggregation(session, sample_event, ["Marc Goddard"])
        session.flush()
        assert sample_event.referee_id == first_ref_id  # NOT overwritten
        still_ref = session.get(Referee, sample_event.referee_id)
        assert still_ref.normalized_name == "herb-dean"

    def test_first_ingest_with_null_referee_id_does_assign(
        self, session: Session, sample_event: Event
    ) -> None:
        # Sanity: starting from NULL, aggregation MUST assign the FK.
        assert sample_event.referee_id is None
        self._apply_aggregation(session, sample_event, ["John McCarthy"])
        session.flush()
        assert sample_event.referee_id is not None
        ref = session.get(Referee, sample_event.referee_id)
        assert ref.normalized_name == "john-mccarthy"

    def test_reingest_with_empty_raw_names_does_not_clear_referee_id(
        self, session: Session, sample_event: Event
    ) -> None:
        # First ingest resolves referee.
        self._apply_aggregation(session, sample_event, ["Herb Dean"])
        session.flush()
        first_ref_id = sample_event.referee_id
        assert first_ref_id is not None

        # Re-ingest with NO parsed referees (all fights had referee=None
        # after parse): the if-guard skips entirely, FK stays put.
        self._apply_aggregation(session, sample_event, [])
        session.flush()
        assert sample_event.referee_id == first_ref_id


class TestIngestVenueIdempotency:
    """CR-01 regression (Phase 28): re-ingest must NOT overwrite an already-resolved venue_id.

    Mirrors TestIngestRefereeIdempotency. The "ingest aggregation" for venues
    is the assign_venue_id_if_null helper from venue_match.py, which encodes
    the same set-once policy as ingest.py:459-474 for referee_id (Phase 22 CR-01).
    """

    @staticmethod
    def _build_tiny_lookup() -> tuple[dict[str, int], list[tuple[int, str]], dict[str, int]]:
        """Build a minimal in-memory venue_lookup + names_raw + named-vid map.

        Returns (venue_lookup, venue_names_raw, named_vids) where named_vids
        maps a friendly key ("vegas", "tokyo") to its venue_id.
        """
        # Pre-populate Venue rows in DB at test time (via fixture), then
        # build the same lookup the production code uses (post-Plan-28-01
        # data/venues.csv).
        named: dict[str, int] = {"vegas": 1001, "tokyo": 1002}
        lookup: dict[str, int] = {
            "las vegas, nevada, usa": named["vegas"],
            "tokyo, japan": named["tokyo"],
        }
        names_raw: list[tuple[int, str]] = [
            (named["vegas"], "Las Vegas, Nevada, USA"),
            (named["tokyo"], "Tokyo, Japan"),
        ]
        return lookup, names_raw, named

    @pytest.fixture
    def venue_setup(
        self, session: Session
    ) -> tuple[dict[str, int], list[tuple[int, str]], dict[str, int]]:
        """Persist real Venue rows that match the in-memory lookup."""
        from ufc_prediction.models.venue import Venue

        lookup, names_raw, named = self._build_tiny_lookup()
        session.add_all(
            [
                Venue(
                    id=named["vegas"],
                    name="Las Vegas, Nevada, USA",
                    city="Las Vegas",
                    state="Nevada",
                    country="USA",
                    lat=36.1674,
                    lon=-115.1484,
                    timezone_iana="America/Los_Angeles",
                ),
                Venue(
                    id=named["tokyo"],
                    name="Tokyo, Japan",
                    city="Tokyo",
                    state=None,
                    country="Japan",
                    lat=35.6762,
                    lon=139.6503,
                    timezone_iana="Asia/Tokyo",
                ),
            ]
        )
        session.flush()
        return lookup, names_raw, named

    def test_first_ingest_with_null_venue_id_does_assign(
        self,
        session: Session,
        sample_event: Event,
        venue_setup: tuple[dict[str, int], list[tuple[int, str]], dict[str, int]],
    ) -> None:
        from ufc_prediction.scraper.venue_match import assign_venue_id_if_null

        lookup, names_raw, named = venue_setup
        assert sample_event.venue_id is None
        vid, kind = assign_venue_id_if_null(
            session,
            event_id=sample_event.id,
            location="Las Vegas, Nevada, USA",
            venue_lookup=lookup,
            venue_names_raw=names_raw,
        )
        session.flush()
        assert vid == named["vegas"]
        assert kind == "exact"
        assert sample_event.venue_id == named["vegas"]

    def test_reingest_does_not_overwrite_already_resolved_venue_id(
        self,
        session: Session,
        sample_event: Event,
        venue_setup: tuple[dict[str, int], list[tuple[int, str]], dict[str, int]],
    ) -> None:
        from ufc_prediction.scraper.venue_match import assign_venue_id_if_null

        lookup, names_raw, named = venue_setup
        # First ingest: resolve to Vegas.
        assign_venue_id_if_null(
            session,
            event_id=sample_event.id,
            location="Las Vegas, Nevada, USA",
            venue_lookup=lookup,
            venue_names_raw=names_raw,
        )
        session.flush()
        assert sample_event.venue_id == named["vegas"]

        # Partial re-ingest with a DIFFERENT location string that would
        # match Tokyo. The previously-correct Vegas FK MUST be preserved
        # (CR-01 guarantee carried forward from referee to venue).
        vid, kind = assign_venue_id_if_null(
            session,
            event_id=sample_event.id,
            location="Tokyo, Japan",
            venue_lookup=lookup,
            venue_names_raw=names_raw,
        )
        session.flush()
        assert vid == named["vegas"]  # returns existing, NOT the new match
        assert kind == "preserved"
        assert sample_event.venue_id == named["vegas"]  # NOT overwritten

    def test_reingest_with_no_match_does_not_clear_venue_id(
        self,
        session: Session,
        sample_event: Event,
        venue_setup: tuple[dict[str, int], list[tuple[int, str]], dict[str, int]],
    ) -> None:
        from ufc_prediction.scraper.venue_match import assign_venue_id_if_null

        lookup, names_raw, named = venue_setup
        assign_venue_id_if_null(
            session,
            event_id=sample_event.id,
            location="Las Vegas, Nevada, USA",
            venue_lookup=lookup,
            venue_names_raw=names_raw,
        )
        session.flush()
        first_vid = sample_event.venue_id
        assert first_vid is not None

        # Re-ingest with an unmatchable string: assign_venue_id_if_null
        # returns (None, None) but the EXISTING venue_id MUST be preserved.
        vid, kind = assign_venue_id_if_null(
            session,
            event_id=sample_event.id,
            location="Mars Crater, Olympus Mons",
            venue_lookup=lookup,
            venue_names_raw=names_raw,
        )
        session.flush()
        # NULL match result; helper should NOT touch the event since it can't.
        # (No new venue, no overwrite — venue_id stays as-was.)
        assert sample_event.venue_id == first_vid

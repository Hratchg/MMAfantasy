"""Integration tests for Kaggle data ingestion pipeline (DATA-01, DATA-02, DATA-03, DATA-04).

Uses testcontainers PostgreSQL for isolated database testing with transactional
rollback per test. Tests verify that CSV data flows correctly through Pydantic
validation, upsert logic, and into SQLAlchemy models.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from ufc_prediction.data.ingest_mdabbert import ingest_mdabbert
from ufc_prediction.data.ingest_rajeevw import ingest_rajeevw_fighters, ingest_rajeevw_fights
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.models.round_stats import RoundStats

# ── Rajeevw fighter ingestion ────────────────────────────────────────────────


def test_ingest_rajeevw_fighters_creates_records(
    session: Session,
    rajeevw_fighters_csv: Path,
) -> None:
    """ingest_rajeevw_fighters creates Fighter records from sample CSV data.

    Verifies correct parsing of height, reach, stance, and DOB.
    Also verifies that the test fighter with '--' values has NULL fields per D-04.
    """
    result = ingest_rajeevw_fighters(rajeevw_fighters_csv, session)

    assert result.accepted == 3
    assert result.rejected == 0

    fighters = session.query(Fighter).all()
    assert len(fighters) == 3

    # Verify Jon Jones profile
    jones = session.query(Fighter).filter(Fighter.name == "Jon Jones").first()
    assert jones is not None
    assert jones.height_inches == 76.0  # 6'4"
    assert jones.reach_inches == 84.5
    assert jones.stance == "Orthodox"
    assert jones.date_of_birth == date(1987, 7, 19)
    assert jones.source == "kaggle-rajeevw"

    # Verify Test Fighter with missing data (D-04 nullable)
    test_fighter = session.query(Fighter).filter(Fighter.name == "Test Fighter").first()
    assert test_fighter is not None
    assert test_fighter.height_inches is None
    assert test_fighter.reach_inches is None
    assert test_fighter.stance is None  # empty string parsed as None


def test_ingest_rajeevw_fighters_idempotent(
    session: Session,
    rajeevw_fighters_csv: Path,
) -> None:
    """Running ingest_rajeevw_fighters twice produces same fighter count (D-03 upsert).

    Second run should report updates, not new inserts.
    """
    result1 = ingest_rajeevw_fighters(rajeevw_fighters_csv, session)
    assert result1.accepted == 3

    result2 = ingest_rajeevw_fighters(rajeevw_fighters_csv, session)
    assert result2.accepted == 0
    assert result2.updated == 3

    # Total fighter count should still be 3
    fighters = session.query(Fighter).all()
    assert len(fighters) == 3


# ── Rajeevw fight ingestion ──────────────────────────────────────────────────


def test_ingest_rajeevw_fights_creates_records(
    session: Session,
    rajeevw_fighters_csv: Path,
    rajeevw_fights_csv: Path,
) -> None:
    """ingest_rajeevw_fights creates Event, Fight, and RoundStats records.

    Verifies correct parsing of fight metadata: weight_class, method,
    round_finished per DATA-03.
    """
    # Ingest fighters first (needed for foreign keys)
    ingest_rajeevw_fighters(rajeevw_fighters_csv, session)

    result = ingest_rajeevw_fights(rajeevw_fights_csv, session)
    assert result.accepted == 2
    assert result.rejected == 0

    # Verify events created
    events = session.query(Event).all()
    assert len(events) == 2

    # Verify fights created
    fights = session.query(Fight).all()
    assert len(fights) == 2

    # Verify fight metadata
    fight1 = fights[0]
    assert fight1.weight_class == "Middleweight"
    assert fight1.method == "Decision"
    assert fight1.method_detail == "Unanimous"
    assert fight1.round_finished == 3
    assert fight1.num_rounds == 3

    # Verify RoundStats created (fight-level aggregates)
    stats = session.query(RoundStats).all()
    assert len(stats) >= 2  # At least 2 (one per fighter per fight)


def test_round_stats_are_round_zero(
    session: Session,
    rajeevw_fighters_csv: Path,
    rajeevw_fights_csv: Path,
) -> None:
    """After fight ingestion, RoundStats records have round_number=0.

    Fight-level aggregates stored as round_number=0 per DATA-04.
    """
    ingest_rajeevw_fighters(rajeevw_fighters_csv, session)
    ingest_rajeevw_fights(rajeevw_fights_csv, session)

    stats = session.query(RoundStats).all()
    assert len(stats) > 0
    for stat in stats:
        assert stat.round_number == 0, f"Expected round_number=0 but got {stat.round_number}"


def test_round_stats_have_correct_values(
    session: Session,
    rajeevw_fighters_csv: Path,
    rajeevw_fights_csv: Path,
) -> None:
    """RoundStats contain correct parsed stat values from CSV."""
    ingest_rajeevw_fighters(rajeevw_fighters_csv, session)
    ingest_rajeevw_fights(rajeevw_fights_csv, session)

    # Find Jones's stats for the first fight
    jones = session.query(Fighter).filter(Fighter.name == "Jon Jones").first()
    fight = session.query(Fight).filter(Fight.fighter_a_id == jones.id).first()
    stats = (
        session.query(RoundStats)
        .filter(RoundStats.fight_id == fight.id, RoundStats.fighter_id == jones.id)
        .first()
    )
    assert stats is not None
    assert stats.sig_strikes_landed == 69
    assert stats.sig_strikes_attempted == 101
    assert stats.takedowns_landed == 2
    assert stats.takedowns_attempted == 5
    assert stats.knockdowns == 2


# ── Malformed row handling ───────────────────────────────────────────────────


def test_ingest_skips_malformed_rows(
    session: Session,
    rajeevw_fighters_csv: Path,
    rajeevw_fights_csv_with_bad_row: Path,
) -> None:
    """CSV with one malformed row: rejected > 0, other rows still ingested (D-06)."""
    ingest_rajeevw_fighters(rajeevw_fighters_csv, session)

    result = ingest_rajeevw_fights(rajeevw_fights_csv_with_bad_row, session)
    assert result.rejected >= 1, "Expected at least 1 rejected row"
    assert result.accepted >= 1, "Expected at least 1 accepted row"

    # Verify the valid fight was still ingested
    fights = session.query(Fight).all()
    assert len(fights) >= 1


# ── Mdabbert ingestion ───────────────────────────────────────────────────────


def test_ingest_mdabbert_creates_new_fighters(
    session: Session,
    mdabbert_csv: Path,
) -> None:
    """ingest_mdabbert creates new Fighter for names not in rajeevw.

    New fighter should have source='kaggle-mdabbert'.
    """
    ingest_mdabbert(mdabbert_csv, session)

    # "New Fighter" should be created with mdabbert source
    new_fighter = session.query(Fighter).filter(Fighter.name == "New Fighter").first()
    assert new_fighter is not None
    assert new_fighter.source == "kaggle-mdabbert"
    assert new_fighter.height_inches is not None  # from cm conversion
    assert new_fighter.reach_inches is not None  # from cm conversion
    assert new_fighter.stance == "Southpaw"


def test_ingest_mdabbert_supplements_existing(
    session: Session,
    rajeevw_fighters_csv: Path,
    mdabbert_csv: Path,
) -> None:
    """Mdabbert supplements existing rajeevw fighter profiles (fills NULL fields only).

    Ingests rajeevw first (Test Fighter with all NULL fields), then mdabbert
    should fill in the missing fields without overwriting existing rajeevw data.
    """
    # Ingest rajeevw fighters first
    ingest_rajeevw_fighters(rajeevw_fighters_csv, session)

    # Verify Jon Jones has all data from rajeevw
    jones_before = (
        session.query(Fighter)
        .filter(Fighter.name == "Jon Jones", Fighter.source == "kaggle-rajeevw")
        .first()
    )
    assert jones_before is not None
    original_height = jones_before.height_inches
    original_reach = jones_before.reach_inches

    # Ingest mdabbert (should supplement, not overwrite)
    ingest_mdabbert(mdabbert_csv, session)

    # Jones's height/reach should remain unchanged (rajeevw data preserved)
    jones_after = (
        session.query(Fighter)
        .filter(Fighter.name == "Jon Jones", Fighter.source == "kaggle-rajeevw")
        .first()
    )
    assert jones_after.height_inches == original_height
    assert jones_after.reach_inches == original_reach


def test_ingest_mdabbert_skips_existing_fights(
    session: Session,
    rajeevw_fighters_csv: Path,
    rajeevw_fights_csv: Path,
    mdabbert_csv: Path,
) -> None:
    """Full pipeline: rajeevw then mdabbert. No duplicate fights created.

    The mdabbert CSV has a fight on 2020-03-28 between Jon Jones and Amanda Nunes
    which already exists from rajeevw. It should be detected as a duplicate.
    """
    # Full rajeevw pipeline
    ingest_rajeevw_fighters(rajeevw_fighters_csv, session)
    ingest_rajeevw_fights(rajeevw_fights_csv, session)

    rajeevw_fight_count = session.query(Fight).count()

    # Mdabbert supplement
    mdabbert_result = ingest_mdabbert(mdabbert_csv, session)

    # Only 1 new fight should be added (the 2023 fight with New Fighter)
    final_fight_count = session.query(Fight).count()
    new_fights = final_fight_count - rajeevw_fight_count
    assert new_fights == 1, f"Expected 1 new fight from mdabbert, got {new_fights}"
    assert mdabbert_result.accepted == 1
    assert mdabbert_result.updated >= 1  # The duplicate was detected


# ── Per-row isolation: one bad row must not discard earlier rows in its batch ──


def test_rajeevw_bad_row_does_not_roll_back_earlier_rows_in_batch(
    session: Session,
    rajeevw_fights_csv: Path,
) -> None:
    """A row rejected mid-batch (here: unparseable Fight_type -> FightRow
    ValidationError) used to trigger a session-wide rollback, silently dropping
    every earlier uncommitted row of the batch while still counting it accepted."""
    lines = rajeevw_fights_csv.read_text(encoding="utf-8").splitlines()
    bad = lines[2].replace("Middleweight Bout", "Garbage Bout")
    rajeevw_fights_csv.write_text("\n".join([lines[0], lines[1], bad]), encoding="utf-8")

    result = ingest_rajeevw_fights(rajeevw_fights_csv, session)

    assert result.rejected == 1
    assert result.accepted == 1
    assert session.query(Fight).count() == 1
    assert session.query(Event).filter(Event.name == "UFC 248").count() == 1


def test_rajeevw_db_error_rolls_back_only_the_failing_row(
    session: Session,
    rajeevw_fights_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A DB-stage failure after the Fight insert (round-stats upsert) must undo
    only that row's writes and must not count it as accepted."""
    import ufc_prediction.data.ingest_rajeevw as rj

    real_upsert = rj._upsert_round_stats
    calls = {"n": 0}

    def flaky_upsert(*args: object, **kwargs: object) -> None:
        calls["n"] += 1
        if calls["n"] == 3:  # first stats call of the second fight
            raise RuntimeError("simulated DB error")
        real_upsert(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(rj, "_upsert_round_stats", flaky_upsert)

    result = ingest_rajeevw_fights(rajeevw_fights_csv, session)

    assert result.rejected == 1
    assert result.accepted == 1
    fights = session.query(Fight).all()
    assert len(fights) == 1
    assert session.query(RoundStats).filter(RoundStats.fight_id == fights[0].id).count() == 2
    assert session.query(Event).filter(Event.name == "UFC 263").count() == 0


def test_mdabbert_bad_row_does_not_roll_back_earlier_rows_in_batch(
    session: Session,
    mdabbert_csv: Path,
) -> None:
    lines = mdabbert_csv.read_text(encoding="utf-8").splitlines()
    bad = lines[1].replace(",Middleweight,", ",Garbage,")
    # Good (new) fight first, then the invalid row.
    mdabbert_csv.write_text("\n".join([lines[0], lines[2], bad]), encoding="utf-8")

    result = ingest_mdabbert(mdabbert_csv, session)

    assert result.rejected == 1
    assert result.accepted == 1
    assert session.query(Fight).count() == 1
    assert session.query(Fighter).filter(Fighter.name == "New Fighter").count() == 1

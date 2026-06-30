"""Per-migration integration tests for Phase 22.

Complements tests/test_migrations.py (full-chain upgrade/downgrade smoke) by
exercising each migration in isolation: reset DB → upgrade to Phase 21 baseline
(4c9cb5ced391) → upgrade to head → assert schema → optionally downgrade -1 and
assert clean reversal.

Phase 22 migrations covered here:
- referees migration (Task 2 of Plan 22-02, revision 11e7e94d0370)
- venues migration (Task 4 of Plan 22-03, revision 59981c08e056)
- model_runs migration (Task 5 of Plan 22-03; revision determined at land time)

Banned imports per Pitfall #1 / Finding 11: nothing under ``ufc_prediction.ml.*``.
"""

from __future__ import annotations

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


@pytest.fixture
def alembic_cfg(postgres_container):
    """Alembic config wired to the postgres_container fixture."""
    url = postgres_container.get_connection_url()
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container) -> None:
    """Reset DB then upgrade to the pre-Phase-22 head (4c9cb5ced391)."""
    url = postgres_container.get_connection_url()
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
        # Drop all tables for clean state
        conn.execute(
            text(
                "DO $$ DECLARE r RECORD; BEGIN "
                "FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname='public') LOOP "
                "EXECUTE 'DROP TABLE IF EXISTS public.' || quote_ident(r.tablename) || ' CASCADE'; "
                "END LOOP; END $$;"
            )
        )
    engine.dispose()
    command.upgrade(alembic_cfg, "4c9cb5ced391")


class TestRefereesMigration:
    """Per-migration tests for revision 11e7e94d0370 (Plan 22-02 Task 2)."""

    # Pin the upgrade target to the referees revision (NOT "head") so this
    # per-migration test stays isolated to Plan 22-02's surface even after
    # Plan 22-03 advances the head with venues + model_runs migrations.

    def test_upgrade_creates_referees_table(self, alembic_cfg, postgres_container) -> None:
        _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container)
        # Pre-state: no referees table
        engine = create_engine(postgres_container.get_connection_url())
        i_before = inspect(engine)
        assert "referees" not in i_before.get_table_names()
        # Upgrade to the referees revision specifically (not chain head)
        command.upgrade(alembic_cfg, "11e7e94d0370")
        i_after = inspect(engine)
        assert "referees" in i_after.get_table_names()
        engine.dispose()

    def test_upgrade_adds_events_referee_id_fk(self, alembic_cfg, postgres_container) -> None:
        _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container)
        command.upgrade(alembic_cfg, "11e7e94d0370")
        engine = create_engine(postgres_container.get_connection_url())
        i = inspect(engine)
        events_cols = {c["name"] for c in i.get_columns("events")}
        assert "referee_id" in events_cols
        events_fks = {fk["name"] for fk in i.get_foreign_keys("events")}
        assert "fk_events_referee_id_referees" in events_fks
        engine.dispose()

    def test_referees_unique_constraint(self, alembic_cfg, postgres_container) -> None:
        _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container)
        command.upgrade(alembic_cfg, "11e7e94d0370")
        engine = create_engine(postgres_container.get_connection_url())
        i = inspect(engine)
        uqs = {uq["name"] for uq in i.get_unique_constraints("referees")}
        assert "uq_referees_normalized_name" in uqs
        engine.dispose()

    def test_referees_referee_id_nullable(self, alembic_cfg, postgres_container) -> None:
        # Pitfall #3: referee_id must be NULLABLE so existing 5,799-row events
        # table doesn't fail IntegrityError on the migration.
        _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container)
        command.upgrade(alembic_cfg, "11e7e94d0370")
        engine = create_engine(postgres_container.get_connection_url())
        i = inspect(engine)
        ref_id_col = next(c for c in i.get_columns("events") if c["name"] == "referee_id")
        assert ref_id_col["nullable"] is True
        engine.dispose()

    def test_downgrade_reverses_cleanly(self, alembic_cfg, postgres_container) -> None:
        _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container)
        # Upgrade only to referees, not chain head — keeps downgrade -1
        # scoped to the referees migration.
        command.upgrade(alembic_cfg, "11e7e94d0370")
        # Now downgrade -1 — should remove referees table + events.referee_id
        command.downgrade(alembic_cfg, "-1")
        engine = create_engine(postgres_container.get_connection_url())
        i = inspect(engine)
        assert "referees" not in i.get_table_names()
        events_cols = {c["name"] for c in i.get_columns("events")}
        assert "referee_id" not in events_cols
        engine.dispose()


class TestVenuesMigration:
    """Per-migration tests for revision 59981c08e056 (Plan 22-03 Task 4).

    Covers: venues table creation, bulk_insert from data/venues.csv,
    events.venue_id NULLABLE FK addition, op.f() naming convention, and
    clean downgrade reversal.

    Note (Task 3 PARTIAL operator-PROCEED): data/venues.csv ships 52 rows
    (~30% of 173 distinct location strings; ~10% by event count). Schema
    is forward-compatible — future operator can INSERT additional rows
    without re-running the migration.
    """

    def test_upgrade_creates_venues_table(self, alembic_cfg, postgres_container) -> None:
        _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container)
        engine = create_engine(postgres_container.get_connection_url())
        i_before = inspect(engine)
        assert "venues" not in i_before.get_table_names()
        command.upgrade(alembic_cfg, "59981c08e056")
        i_after = inspect(engine)
        assert "venues" in i_after.get_table_names()
        engine.dispose()

    def test_venues_csv_row_count_matches_db(self, alembic_cfg, postgres_container) -> None:
        import csv as _csv

        _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container)
        command.upgrade(alembic_cfg, "59981c08e056")
        engine = create_engine(postgres_container.get_connection_url())
        with engine.connect() as c:
            db_count = c.execute(text("SELECT COUNT(*) FROM venues")).scalar()
        with open("data/venues.csv", encoding="utf-8") as f:
            csv_count = sum(1 for _ in _csv.DictReader(f))
        assert db_count == csv_count, (
            f"venues bulk_insert row drift: db={db_count}, csv={csv_count}"
        )
        engine.dispose()

    def test_events_venue_id_fk_nullable(self, alembic_cfg, postgres_container) -> None:
        # Pitfall #3: venue_id must be NULLABLE so existing 5,799-row events
        # table doesn't fail IntegrityError on the migration.
        _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container)
        command.upgrade(alembic_cfg, "59981c08e056")
        engine = create_engine(postgres_container.get_connection_url())
        i = inspect(engine)
        venue_id_col = next(c for c in i.get_columns("events") if c["name"] == "venue_id")
        assert venue_id_col["nullable"] is True
        fks = {fk["name"] for fk in i.get_foreign_keys("events")}
        assert "fk_events_venue_id_venues" in fks
        engine.dispose()

    def test_venues_primary_key_op_f_naming(self, alembic_cfg, postgres_container) -> None:
        # Pitfall #2 / Finding 4: op.f() naming convention applied.
        _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container)
        command.upgrade(alembic_cfg, "59981c08e056")
        engine = create_engine(postgres_container.get_connection_url())
        i = inspect(engine)
        pk = i.get_pk_constraint("venues")
        assert pk["name"] == "pk_venues"
        engine.dispose()

    def test_venues_downgrade_reverses_cleanly(self, alembic_cfg, postgres_container) -> None:
        _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container)
        command.upgrade(alembic_cfg, "59981c08e056")
        # Downgrade -1 — should remove venues table + events.venue_id column + FK
        command.downgrade(alembic_cfg, "-1")
        engine = create_engine(postgres_container.get_connection_url())
        i = inspect(engine)
        assert "venues" not in i.get_table_names()
        events_cols = {c["name"] for c in i.get_columns("events")}
        assert "venue_id" not in events_cols
        engine.dispose()


class TestModelRunsMigration:
    """Per-migration tests for revision e09bc46ad044 (Plan 22-03 Task 5).

    Covers: model_runs table creation, composite + single-col indexes, JSONB
    type for metadata_json, op.f() naming for pk, and clean downgrade reversal.

    Note: Phase 22 ships SCHEMA ONLY for model_runs — no bulk_insert; Phase 26
    CALIB-V22-01 will write rows via ml/persistence.py extension.
    """

    def test_upgrade_creates_model_runs_table(self, alembic_cfg, postgres_container) -> None:
        _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container)
        engine = create_engine(postgres_container.get_connection_url())
        i_before = inspect(engine)
        assert "model_runs" not in i_before.get_table_names()
        command.upgrade(alembic_cfg, "e09bc46ad044")
        i_after = inspect(engine)
        assert "model_runs" in i_after.get_table_names()
        engine.dispose()

    def test_model_runs_indexes(self, alembic_cfg, postgres_container) -> None:
        _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container)
        command.upgrade(alembic_cfg, "e09bc46ad044")
        engine = create_engine(postgres_container.get_connection_url())
        i = inspect(engine)
        idx_names = {idx["name"] for idx in i.get_indexes("model_runs")}
        assert "ix_model_runs_model_name_model_version" in idx_names
        assert "ix_model_runs_trained_at" in idx_names
        engine.dispose()

    def test_model_runs_metadata_json_jsonb(self, alembic_cfg, postgres_container) -> None:
        _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container)
        command.upgrade(alembic_cfg, "e09bc46ad044")
        engine = create_engine(postgres_container.get_connection_url())
        with engine.connect() as c:
            result = c.execute(
                text(
                    "SELECT data_type FROM information_schema.columns "
                    "WHERE table_name='model_runs' AND column_name='metadata_json'"
                )
            ).scalar()
        assert result == "jsonb"
        engine.dispose()

    def test_model_runs_primary_key_op_f_naming(self, alembic_cfg, postgres_container) -> None:
        _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container)
        command.upgrade(alembic_cfg, "e09bc46ad044")
        engine = create_engine(postgres_container.get_connection_url())
        i = inspect(engine)
        pk = i.get_pk_constraint("model_runs")
        assert pk["name"] == "pk_model_runs"
        engine.dispose()

    def test_model_runs_cutoff_date_is_date_type(self, alembic_cfg, postgres_container) -> None:
        # Finding 12: sa.Date for cutoff_date matches Event.date precedent.
        _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container)
        command.upgrade(alembic_cfg, "e09bc46ad044")
        engine = create_engine(postgres_container.get_connection_url())
        with engine.connect() as c:
            result = c.execute(
                text(
                    "SELECT data_type FROM information_schema.columns "
                    "WHERE table_name='model_runs' AND column_name='cutoff_date'"
                )
            ).scalar()
        assert result == "date"
        engine.dispose()

    def test_model_runs_downgrade_reverses_cleanly(self, alembic_cfg, postgres_container) -> None:
        _drop_all_then_to_phase21_baseline(alembic_cfg, postgres_container)
        command.upgrade(alembic_cfg, "e09bc46ad044")
        # Downgrade -1 — should remove model_runs table + its 2 indexes
        command.downgrade(alembic_cfg, "-1")
        engine = create_engine(postgres_container.get_connection_url())
        i = inspect(engine)
        assert "model_runs" not in i.get_table_names()
        engine.dispose()

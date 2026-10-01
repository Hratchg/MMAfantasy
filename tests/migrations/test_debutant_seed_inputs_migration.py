"""Migration 9fc47fdd75b6: the ``debutant_seed_inputs`` table (Sherdog seeds in the DB).

Exercises the migration in isolation against testcontainers Postgres: reset →
upgrade to the previous head (e09bc46ad044) → upgrade one step → assert the
schema, constraints and ORM parity → downgrade one step → assert a clean
reversal. The module leaves the database at ``head`` so later tests that rely
on the full schema still find every table.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from ufc_prediction.db.base import Base

_PREV = "e09bc46ad044"
_REV = "9fc47fdd75b6"
_TABLE = "debutant_seed_inputs"


@pytest.fixture
def alembic_cfg(postgres_container: Any) -> Config:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", postgres_container.get_connection_url())
    return cfg


@pytest.fixture
def pg(postgres_container: Any, alembic_cfg: Config) -> Iterator[Engine]:
    """Empty DB migrated to the revision before this one; left at head after."""
    engine = create_engine(postgres_container.get_connection_url())
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
        conn.execute(
            text(
                "DO $$ DECLARE r RECORD; BEGIN "
                "FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname='public') LOOP "
                "EXECUTE 'DROP TABLE IF EXISTS public.' || quote_ident(r.tablename) || ' CASCADE'; "
                "END LOOP; END $$;"
            )
        )
    command.upgrade(alembic_cfg, _PREV)
    try:
        yield engine
    finally:
        command.upgrade(alembic_cfg, "head")
        engine.dispose()


def _insert_fighter(engine: Engine, fighter_id: int) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO fighters (id, name, source, created_at, updated_at) "
                "VALUES (:id, 'Seed Test', 'ufcstats', now(), now())"
            ),
            {"id": fighter_id},
        )


def _insert_seed(engine: Engine, **overrides: Any) -> None:
    row: dict[str, Any] = {
        "fighter_id": 1,
        "sherdog_url": "https://www.sherdog.com/fighter/Seed-Test-1",
        "n_pre_ufc_fights": 10,
        "win_rate": 0.8,
        "org_tier": "major",
        "scraped_at": "2026-07-03T23:38:16.314549+00:00",
    }
    row.update(overrides)
    with engine.begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO {_TABLE} "
                "(fighter_id, sherdog_url, n_pre_ufc_fights, win_rate, org_tier, scraped_at) "
                "VALUES (:fighter_id, :sherdog_url, :n_pre_ufc_fights, :win_rate, :org_tier, "
                ":scraped_at)"
            ),
            row,
        )


def test_upgrade_creates_table_with_seed_inputs_and_provenance(
    pg: Engine, alembic_cfg: Config
) -> None:
    assert _TABLE not in inspect(pg).get_table_names()

    command.upgrade(alembic_cfg, _REV)

    insp = inspect(pg)
    assert _TABLE in insp.get_table_names()
    cols = {c["name"]: c for c in insp.get_columns(_TABLE)}
    assert set(cols) == {
        "fighter_id",
        "sherdog_url",
        "n_pre_ufc_fights",
        "wins",
        "losses",
        "draws",
        "nc_dq",
        "win_rate",
        "kos",
        "submissions",
        "decisions",
        "last_organization",
        "org_tier",
        "scraped_at",
    }
    # Seed inputs and provenance are mandatory; the descriptive counts are not.
    for required in (
        "fighter_id",
        "n_pre_ufc_fights",
        "win_rate",
        "org_tier",
        "sherdog_url",
        "scraped_at",
    ):
        assert cols[required]["nullable"] is False, required
    for optional in ("wins", "losses", "kos", "last_organization"):
        assert cols[optional]["nullable"] is True, optional

    pk = insp.get_pk_constraint(_TABLE)
    assert pk["constrained_columns"] == ["fighter_id"]
    assert pk["name"] == "pk_debutant_seed_inputs"
    (fk,) = insp.get_foreign_keys(_TABLE)
    assert fk["referred_table"] == "fighters"
    assert fk["constrained_columns"] == ["fighter_id"]
    assert fk["options"].get("ondelete") == "CASCADE"

    with pg.connect() as conn:
        types = dict(
            conn.execute(
                text(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_name = :t"
                ),
                {"t": _TABLE},
            ).all()
        )
    # float8 round-trips a Python float exactly, so seeds read back bit-identical.
    assert types["win_rate"] == "double precision"
    assert types["scraped_at"] == "timestamp with time zone"


@pytest.mark.parametrize(
    ("overrides", "constraint"),
    [
        ({"org_tier": "elite"}, "ck_debutant_seed_inputs_org_tier"),
        ({"win_rate": 1.01}, "ck_debutant_seed_inputs_win_rate"),
        ({"win_rate": -0.01}, "ck_debutant_seed_inputs_win_rate"),
        ({"n_pre_ufc_fights": -1}, "ck_debutant_seed_inputs_n_pre_ufc_fights"),
    ],
)
def test_check_constraints_reject_invalid_seed_inputs(
    pg: Engine, alembic_cfg: Config, overrides: dict[str, Any], constraint: str
) -> None:
    command.upgrade(alembic_cfg, _REV)
    _insert_fighter(pg, 1)

    with pytest.raises(IntegrityError, match=constraint):
        _insert_seed(pg, **overrides)


def test_foreign_key_rejects_unknown_fighter(pg: Engine, alembic_cfg: Config) -> None:
    command.upgrade(alembic_cfg, _REV)

    with pytest.raises(IntegrityError, match="fk_debutant_seed_inputs_fighter_id_fighters"):
        _insert_seed(pg, fighter_id=424242)


def test_valid_row_inserts_and_every_tier_is_accepted(pg: Engine, alembic_cfg: Config) -> None:
    command.upgrade(alembic_cfg, _REV)
    for fid, tier in enumerate(("major", "regional", "local", "none"), start=1):
        _insert_fighter(pg, fid)
        _insert_seed(pg, fighter_id=fid, org_tier=tier)

    with pg.connect() as conn:
        assert conn.execute(text(f"SELECT count(*) FROM {_TABLE}")).scalar() == 4


def test_migration_matches_the_orm_model(pg: Engine, alembic_cfg: Config) -> None:
    """Autogenerate finds no drift for this table between the migration and the model."""
    from alembic.autogenerate import compare_metadata

    command.upgrade(alembic_cfg, _REV)
    with pg.connect() as conn:
        diffs = compare_metadata(MigrationContext.configure(conn), Base.metadata)

    def _touches_table(diff: Any) -> bool:
        return _TABLE in repr(diff)

    assert [d for d in diffs if _touches_table(d)] == []


def test_downgrade_drops_the_table(pg: Engine, alembic_cfg: Config) -> None:
    command.upgrade(alembic_cfg, _REV)
    _insert_fighter(pg, 1)
    _insert_seed(pg)

    command.downgrade(alembic_cfg, "-1")

    insp = inspect(pg)
    assert _TABLE not in insp.get_table_names()
    assert "fighters" in insp.get_table_names()
    with pg.connect() as conn:
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == _PREV

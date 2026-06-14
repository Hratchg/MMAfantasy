from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from ufc_prediction.db.base import Base


def test_alembic_upgrade_downgrade(postgres_container):
    """Alembic upgrade head creates all 7 tables, downgrade base removes them."""
    url = postgres_container.get_connection_url()
    engine = create_engine(url)

    # Drop all tables first (engine fixture may have created them via create_all)
    Base.metadata.drop_all(engine)
    # Also drop alembic_version if it exists
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))

    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", url)

    # Upgrade to head
    command.upgrade(alembic_cfg, "head")

    # Verify all 7 tables exist
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            )
        )
        tables = [row[0] for row in result]

    expected_tables = [
        "computed_features",
        "elo_snapshots",
        "events",
        "fighter_aliases",
        "fighters",
        "fights",
        "round_stats",
        # Phase 22 additions:
        "model_runs",
        "referees",
        "venues",
    ]
    for table in expected_tables:
        assert table in tables, f"Table '{table}' not found after upgrade"

    # Downgrade to base
    command.downgrade(alembic_cfg, "base")

    # Verify tables are gone (only alembic_version may remain)
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' "
                "AND table_name != 'alembic_version' "
                "ORDER BY table_name"
            )
        )
        remaining_tables = [row[0] for row in result]

    assert len(remaining_tables) == 0, f"Tables remain after downgrade: {remaining_tables}"

    # Re-create tables so other tests continue to work if run after this
    Base.metadata.create_all(engine)
    engine.dispose()

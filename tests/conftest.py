import sys
from pathlib import Path

# Ensure the local src/ directory takes precedence over installed packages.
# This is needed when running tests from a git worktree whose source tree
# may differ from the editable-installed copy (e.g. new modules like export.py).
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.postgres import PostgresContainer

from ufc_prediction.db.base import Base
from ufc_prediction.models import *  # noqa: F403 -- register all models


@pytest.fixture(scope="session")
def postgres_container():
    """Start a PostgreSQL 16 container for the entire test session."""
    with PostgresContainer("postgres:16", driver="psycopg") as pg:
        yield pg


@pytest.fixture(scope="session")
def engine(postgres_container):
    """Create SQLAlchemy engine connected to test container."""
    url = postgres_container.get_connection_url()
    eng = create_engine(url, echo=True)
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def session(engine) -> Session:
    """Provide a transactional session that rolls back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    sess = sessionmaker(bind=connection)()
    yield sess
    sess.close()
    # Transaction may already be invalidated if an IntegrityError was raised
    if transaction.is_active:
        transaction.rollback()
    connection.close()

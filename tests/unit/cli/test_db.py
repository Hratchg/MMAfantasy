from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ufc_prediction.cli.db import CANONICAL_TABLES, DEFAULT_DUMP_PATH, db_app

runner = CliRunner()


def test_canonical_tables_count():
    assert len(CANONICAL_TABLES) == 12
    assert CANONICAL_TABLES[0] == "events"
    assert CANONICAL_TABLES[-1] == "alembic_version"


def test_default_dump_path():
    assert str(DEFAULT_DUMP_PATH) == "data/seed/ufc_corpus_v30.dump"


def test_seed_help_advertises_flags():
    result = runner.invoke(db_app, ["seed", "--help"])
    assert result.exit_code == 0
    assert "--from" in result.stdout
    assert "--force" in result.stdout
    assert "--no-migrate" in result.stdout


def test_status_help_succeeds():
    result = runner.invoke(db_app, ["status", "--help"])
    assert result.exit_code == 0


@patch("ufc_prediction.cli.db.settings")
@patch.dict("os.environ", {}, clear=True)
def test_preflight_1_database_url_missing(mock_settings):
    mock_settings.database_url = ""
    result = runner.invoke(db_app, ["seed"])
    assert result.exit_code == 1
    assert "Pre-flight 1/5" in result.stdout


@patch.dict("os.environ", {"DATABASE_URL": "postgres://u:p@localhost:5433/x"})
@patch("ufc_prediction.cli.db.psycopg.connect", side_effect=Exception("unreachable"))
def test_preflight_2_postgres_unreachable(mock_connect):
    result = runner.invoke(db_app, ["seed"])
    assert result.exit_code == 1
    assert "Pre-flight 2/5" in result.stdout


@patch.dict("os.environ", {"DATABASE_URL": "postgres://u:p@localhost:5433/x"})
@patch("ufc_prediction.cli.db.psycopg.connect")
@patch("ufc_prediction.cli.db.shutil.which", return_value=None)
def test_preflight_3_pg_restore_not_on_path(mock_which, mock_connect):
    result = runner.invoke(db_app, ["seed"])
    assert result.exit_code == 1
    assert "Pre-flight 3/5" in result.stdout


@patch.dict("os.environ", {"DATABASE_URL": "postgres://u:p@localhost:5433/x"})
@patch("ufc_prediction.cli.db.psycopg.connect")
@patch("ufc_prediction.cli.db.shutil.which", return_value="/usr/bin/pg_restore")
def test_preflight_4_dump_missing(mock_which, mock_connect, tmp_path):
    missing = tmp_path / "does-not-exist.dump"
    result = runner.invoke(db_app, ["seed", "--from", str(missing)])
    assert result.exit_code == 1
    assert "Pre-flight 4/5" in result.stdout


@patch.dict("os.environ", {"DATABASE_URL": "postgres://u:p@localhost:5433/x"})
@patch("ufc_prediction.cli.db.psycopg.connect")
@patch("ufc_prediction.cli.db.shutil.which", return_value="/usr/bin/pg_restore")
@patch("ufc_prediction.cli.db.SessionLocal")
def test_preflight_5_target_not_empty_without_force(
    mock_session, mock_which, mock_connect, tmp_path
):
    dump = tmp_path / "x.dump"
    dump.write_bytes(b"PGDMP")
    mock_session.return_value.__enter__.return_value.execute.return_value.scalar.return_value = 999
    result = runner.invoke(db_app, ["seed", "--from", str(dump)])
    assert result.exit_code == 1
    assert "Pre-flight 5/5" in result.stdout


@patch.dict("os.environ", {"DATABASE_URL": "postgres://u:p@localhost:5433/x"})
@patch("ufc_prediction.cli.db.psycopg.connect")
@patch("ufc_prediction.cli.db.shutil.which", return_value="/usr/bin/pg_restore")
@patch("ufc_prediction.cli.db.SessionLocal")
@patch("ufc_prediction.cli.db.subprocess.run")
@patch("ufc_prediction.cli.db._row_counts", return_value={t: 1 for t in CANONICAL_TABLES})
@patch("ufc_prediction.cli.db._print_row_table")
@patch("ufc_prediction.cli.db._alembic_stamp_head")
@patch("ufc_prediction.cli.db._predictor_sanity_check")
def test_seed_skips_alembic_when_no_migrate(
    mock_sanity,
    mock_stamp,
    mock_print,
    mock_counts,
    mock_run,
    mock_session,
    mock_which,
    mock_connect,
    tmp_path,
):
    dump = tmp_path / "x.dump"
    dump.write_bytes(b"PGDMP")
    mock_session.return_value.__enter__.return_value.execute.return_value.scalar.return_value = 0
    mock_run.return_value = MagicMock(returncode=0, stderr="")
    result = runner.invoke(db_app, ["seed", "--from", str(dump), "--no-migrate"])
    assert result.exit_code == 0
    mock_stamp.assert_not_called()
    mock_sanity.assert_called_once()


@patch.dict("os.environ", {"DATABASE_URL": "postgres://u:p@localhost:5433/x"})
@patch("ufc_prediction.cli.db.psycopg.connect")
@patch("ufc_prediction.cli.db.shutil.which", return_value="/usr/bin/pg_restore")
@patch("ufc_prediction.cli.db.SessionLocal")
@patch("ufc_prediction.cli.db.subprocess.run")
@patch("ufc_prediction.cli.db._row_counts", return_value={t: 1 for t in CANONICAL_TABLES})
@patch("ufc_prediction.cli.db._print_row_table")
@patch("ufc_prediction.cli.db._alembic_stamp_head")
@patch("ufc_prediction.cli.db._predictor_sanity_check")
def test_seed_runs_alembic_by_default(
    mock_sanity,
    mock_stamp,
    mock_print,
    mock_counts,
    mock_run,
    mock_session,
    mock_which,
    mock_connect,
    tmp_path,
):
    dump = tmp_path / "x.dump"
    dump.write_bytes(b"PGDMP")
    mock_session.return_value.__enter__.return_value.execute.return_value.scalar.return_value = 0
    mock_run.return_value = MagicMock(returncode=0, stderr="")
    result = runner.invoke(db_app, ["seed", "--from", str(dump)])
    assert result.exit_code == 0
    mock_stamp.assert_called_once()
    mock_sanity.assert_called_once()


@patch.dict("os.environ", {"DATABASE_URL": "postgres://u:p@localhost:5433/x"})
@patch("ufc_prediction.cli.db.psycopg.connect")
@patch("ufc_prediction.cli.db.shutil.which", return_value="/usr/bin/pg_restore")
@patch("ufc_prediction.cli.db.SessionLocal")
@patch("ufc_prediction.cli.db.subprocess.run")
def test_seed_surfaces_pg_restore_failure(
    mock_run,
    mock_session,
    mock_which,
    mock_connect,
    tmp_path,
):
    dump = tmp_path / "x.dump"
    dump.write_bytes(b"PGDMP")
    mock_session.return_value.__enter__.return_value.execute.return_value.scalar.return_value = 0
    mock_run.return_value = MagicMock(returncode=1, stderr="pg_restore: error: corrupt")
    result = runner.invoke(db_app, ["seed", "--from", str(dump)])
    assert result.exit_code == 1
    assert "pg_restore FAILED" in result.stdout


@patch.dict("os.environ", {"DATABASE_URL": "postgres://u:p@localhost:5433/x"})
@patch("ufc_prediction.cli.db.psycopg.connect")
@patch("ufc_prediction.cli.db.shutil.which", return_value="/usr/bin/pg_restore")
@patch("ufc_prediction.cli.db.SessionLocal")
@patch("ufc_prediction.cli.db.subprocess.run")
@patch("ufc_prediction.cli.db._row_counts", return_value={t: 1 for t in CANONICAL_TABLES})
@patch("ufc_prediction.cli.db._print_row_table")
@patch("ufc_prediction.cli.db._alembic_stamp_head")
def test_seed_exit_2_on_predictor_failure(
    mock_stamp,
    mock_print,
    mock_counts,
    mock_run,
    mock_session,
    mock_which,
    mock_connect,
    tmp_path,
):
    """Predictor sanity check failure exits 2 (distinct from DB exit 1)."""
    dump = tmp_path / "x.dump"
    dump.write_bytes(b"PGDMP")
    mock_session.return_value.__enter__.return_value.execute.return_value.scalar.return_value = 0
    mock_run.return_value = MagicMock(returncode=0, stderr="")
    with patch(
        "ufc_prediction.ml.predictor.ModelPredictor",
        side_effect=RuntimeError("xgb_v3 default broken"),
    ):
        result = runner.invoke(db_app, ["seed", "--from", str(dump)])
    assert result.exit_code == 2
    assert "Predictor sanity check FAILED" in result.stdout
    assert "KNOWN_ISSUES" in result.stdout

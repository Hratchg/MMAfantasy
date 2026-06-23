"""Docker-gated round-trip integration test for `ufc db seed`.

Scaffolded in Plan 88-02; first live execution happens in Plan 88-03.
Spins up an ephemeral postgres:18-alpine container on a disposable host
port, runs `ufc db seed` against it, asserts per-table row counts match
PROVENANCE.md goldens, then tears the container down.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

import pytest


def _ephemeral_port() -> int:
    """Ask the OS for a free TCP port.

    REG-V30-02 fix: prior hardcoded port 55555 failed to bind the userland
    Docker proxy on some macOS Docker Desktop installs. The OS-allocated
    port is reliable across Docker Desktop, colima, and Linux.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("docker") is None,
        reason="docker not available — integration test skipped",
    ),
]

EXPECTED_ROW_COUNTS = {
    "elo_snapshots": 89_988,
    "round_stats": 68_960,
    "computed_features": 28_624,
    "fight_odds": 25_632,
    "fights": 16_902,
    "fighters": 6_820,
    "events": 1_872,
    "fighter_aliases": 399,
    "venues": 174,
    "referees": 39,
    "alembic_version": 1,
    "model_runs": 0,
}

DUMP_PATH = Path("data/seed/ufc_corpus_v30.dump")


def _wait_for_pg(
    host: str, port: int, user: str, password: str, db: str, timeout_s: int = 30
) -> None:
    import psycopg

    url = f"postgres://{user}:{password}@{host}:{port}/{db}"
    deadline = time.time() + timeout_s
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            conn = psycopg.connect(url, connect_timeout=2)
            conn.close()
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(0.5)
    raise RuntimeError(
        f"postgres at {host}:{port} not reachable within {timeout_s}s: {last_exc}"
    )


@pytest.mark.skipif(
    not DUMP_PATH.exists(),
    reason=f"{DUMP_PATH} missing — Plan 88-01 must run first",
)
def test_round_trip_seed_against_disposable_postgres():
    """Round-trip: spin up postgres → ufc db seed → assert golden row counts."""
    container = f"ufc-seed-test-{uuid.uuid4().hex[:8]}"
    port = _ephemeral_port()
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                container,
                "-e",
                "POSTGRES_USER=ufc",
                "-e",
                "POSTGRES_PASSWORD=ufc",
                "-e",
                "POSTGRES_DB=ufc_prediction",
                "-p",
                f"{port}:5432",
                "postgres:18-alpine",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        _wait_for_pg("localhost", port, "ufc", "ufc", "ufc_prediction")

        env = {
            **os.environ,
            # Canonical scheme: SQLAlchemy's create_engine (used by the seed
            # command's empty-target check) rejects a bare `postgres://` URL.
            "DATABASE_URL": (
                f"postgresql+psycopg://ufc:ufc@localhost:{port}/ufc_prediction"
            ),
        }
        result = subprocess.run(
            [
                "uv",
                "run",
                "ufc",
                "db",
                "seed",
                "--from",
                str(DUMP_PATH),
                "--no-migrate",
                "--force",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"ufc db seed failed:\n{result.stdout}\n{result.stderr}"
        )

        import psycopg

        from ufc_prediction.cli.db import _normalize_for_psycopg

        # psycopg.connect can't parse the SQLAlchemy `+psycopg` driver tag, so
        # strip it the same way the seed command does.
        conn = psycopg.connect(_normalize_for_psycopg(env["DATABASE_URL"]))
        try:
            with conn.cursor() as cur:
                for table, expected in EXPECTED_ROW_COUNTS.items():
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    got = cur.fetchone()[0]
                    assert got == expected, (
                        f"{table}: expected {expected}, got {got}"
                    )
        finally:
            conn.close()
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container], capture_output=True, text=True
        )

"""Integration-tier regression for BFO disambiguation anomaly (BFO-V25-03).

This is the DB-touching companion to ``test_bfo_disambiguation_v25_unit.py``.
The unit tier freezes the matcher contract in pure Python; this tier verifies
the same fix survives end-to-end through a real Postgres schema with real
SQLAlchemy ORM round-tripping — catching regressions that pure-Python mocks
cannot (FK constraint violations, ORM identity-map quirks, transaction
boundaries, etc.).

Phase 37 TEMPORAL-V24-02 pattern reuse:
- testcontainers PostgresContainer (session-scoped, via ``tests/conftest.py``).
- Transactional rollback per test (``session`` fixture wraps each test in a
  connection-level transaction; teardown rolls back).
- Skips gracefully when Docker daemon is unavailable (CI without Docker still
  GREEN; full integration suite GREEN only with Docker on).

Plan 41-04 BFO-V25-03 integration tier contract:
- Seed event + fighters + fight (from ``integration_anomaly_row.json``).
- Construct ``BFOFighterName`` rows mirroring ``data/bfo/fighters_names.csv``
  (canonical ``database_id`` populated for both fighters in the anomaly row).
- Invoke ``BFOOddsIngester._match_fighters`` against the live Postgres session.
- Assert the resolved ``bfo_id -> DB Fighter.id`` map matches the expected
  canonical resolution from the fixture's ``expected_resolutions`` block.
- Assert the canonical-path counter (``IngestSummary.fighters_matched_canonical``)
  incremented — proving the canonical-substrate-first contract is honored.
- Assert transactional rollback isolation (the inserted rows do not leak
  across tests).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

# Phase-level integration marker: matches Phase 37 TEMPORAL-V24-02 convention
# and the CONTEXT decision "CI runs unit always; integration via `pytest -m
# integration` (skipped if Docker daemon unavailable; matches Phase 35/36
# conventions)".
pytestmark = pytest.mark.integration

FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "bfo_disambiguation_v25"
    / "integration_anomaly_row.json"
)


# ── Docker-availability gate (skip-gracefully per plan T1 Sub-step 1d) ─────


def _docker_available() -> bool:
    """Return True iff the Docker daemon is reachable.

    The testcontainers PostgresContainer fixture requires Docker. Many CI
    environments (and gsd-execute worktrees, per the 2026-05-20 observation
    in MEMORY: "Docker Not Available in Execute Worktree — testcontainers
    Integration Tests Fail With DockerException") do not expose Docker.
    In those environments we SKIP every test in this module rather than
    letting testcontainers raise DockerException mid-test (which would
    surface as ERROR, not SKIPPED).
    """
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


# Module-level skip gate (Phase 41 CONTEXT decision: "CI runs unit always;
# integration via `pytest -m integration` (skipped if Docker daemon
# unavailable; matches Phase 35/36 conventions)"). This MUST fire before
# pytest resolves the session-scoped postgres_container fixture from
# tests/conftest.py — otherwise testcontainers would raise DockerException
# during fixture setup, surfacing as ERROR (not SKIPPED). Module-level skip
# at import time guarantees the collection still succeeds but every test
# in the module is skipped before any session-scoped fixture is touched.
_DOCKER_OK = _docker_available()
if not _DOCKER_OK:  # pragma: no cover - environment-dependent
    pytest.skip(
        "Docker daemon unavailable — BFO-V25-03 integration tier "
        "requires testcontainers Postgres (per Phase 37 TEMPORAL-V24-02 "
        "pattern). Run with Docker on to exercise the real-DB path.",
        allow_module_level=True,
    )


# ── Fixture loader ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fixture_data() -> dict[str, Any]:
    """Load the integration anomaly + control fixture once per module."""
    with FIXTURE_PATH.open() as fh:
        return json.load(fh)


# ── Seed helpers ───────────────────────────────────────────────────────────


def _seed_event_fighters_fight(
    session: Any,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Insert event + fighters + fight rows for a fixture row.

    Returns a dict carrying the persisted Event/Fighter/Fight ids the test
    can use for assertions + downstream BFO source construction. Fighter
    rows are inserted with explicit ``id`` columns from the fixture's
    ``fixture_db_id`` so the resolved canonical_id from the BFO
    ``database_id`` column lands on a real DB row.
    """
    from ufc_prediction.models.event import Event
    from ufc_prediction.models.fight import Fight
    from ufc_prediction.models.fighter import Fighter

    event = Event(
        name=row["event"]["name"],
        date=date.fromisoformat(row["event"]["event_date"]),
        source=row["event"]["source"],
    )
    session.add(event)
    session.flush()

    # Explicit-id Fighter insert so the BFO database_id column resolves to
    # a real row. Production fighters get auto-assigned ids; in this
    # fixture we control the ids so we can assert equality.
    fighters: dict[int, Fighter] = {}
    for f_spec in row["fighters"]:
        f = Fighter(
            id=f_spec["fixture_db_id"],
            name=f_spec["canonical_name"],
            source=f_spec["source"],
        )
        session.add(f)
        fighters[f_spec["fixture_db_id"]] = f
        # Add near-name homonyms with auto-assigned ids — these populate the
        # candidate pool the fuzzy matcher would walk if the canonical path
        # were bypassed. The integration test must NOT resolve to these ids.
        for homonym_name in f_spec["near_name_homonyms"]:
            session.add(
                Fighter(
                    name=homonym_name,
                    source=f_spec["source"],
                )
            )
    session.flush()

    fight = Fight(
        event_id=event.id,
        fighter_a_id=row["fight"]["fixture_db_fighter_a_id"],
        fighter_b_id=row["fight"]["fixture_db_fighter_b_id"],
        weight_class=row["fight"]["weight_class"],
        source=row["fight"]["source"],
    )
    session.add(fight)
    session.flush()

    return {
        "event_id": event.id,
        "fight_id": fight.id,
        "fighter_a_id": row["fight"]["fixture_db_fighter_a_id"],
        "fighter_b_id": row["fight"]["fixture_db_fighter_b_id"],
    }


def _build_bfo_names_map(row: dict[str, Any]) -> dict[str, list[Any]]:
    """Construct the ``bfo_names`` dict that ``_match_fighters`` consumes.

    Mirrors the shape ``BFOOddsIngester._load_fighter_names`` returns:
    ``{bfo_fighter_id: [BFOFighterName, ...]}``. Each entry carries the
    canonical ``database_id`` from the fixture so the post-fix matcher
    short-circuits the fuzzy path.
    """
    from ufc_prediction.scraper.bfo_models import BFOFighterName

    bfo_a = row["bfo_source"]["bfo_fighter_a"]
    bfo_b = row["bfo_source"]["bfo_fighter_b"]
    return {
        bfo_a["bfo_fighter_id"]: [
            BFOFighterName(
                fighter_id=bfo_a["bfo_fighter_id"],
                database=bfo_a["database"],
                name=bfo_a["bfo_name"],
                database_id=bfo_a["database_id"],
            )
        ],
        bfo_b["bfo_fighter_id"]: [
            BFOFighterName(
                fighter_id=bfo_b["bfo_fighter_id"],
                database=bfo_b["database"],
                name=bfo_b["bfo_name"],
                database_id=bfo_b["database_id"],
            )
        ],
    }


# ── Tests ──────────────────────────────────────────────────────────────────


def test_anomaly_row_resolves_post_patch(
    fixture_data: dict[str, Any], session: Any, tmp_path: Path
) -> None:
    """Integration tier: a deliberately-anomalous row (sourced from the
    Plan 41-01 §2.6 cascade Stage 2->3 drop sample) seeded into real
    Postgres resolves to the expected canonical Fighter.id via the
    patched ``_match_fighters`` (canonical ``database_id`` path),
    NOT via the fuzzy matcher walking the candidate pool.

    This is the BFO-V25-03 integration tier's primary assertion —
    closes the second half of REQ BFO-V25-03 (unit tier was closed in
    Plan 41-02).
    """
    from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester, IngestSummary

    anomaly = fixture_data["anomaly_row"]
    expected = fixture_data["expected_resolutions"]

    # Seed event + fighters + fight into the testcontainers Postgres.
    seeded = _seed_event_fighters_fight(session, anomaly)
    assert seeded["fighter_a_id"] == expected["anomaly_fighter_a_id"]
    assert seeded["fighter_b_id"] == expected["anomaly_fighter_b_id"]

    # Construct the BFO names map (canonical database_id populated for both
    # fighters — this is the substrate the post-fix matcher must honor).
    bfo_names = _build_bfo_names_map(anomaly)

    # Run the patched _match_fighters against the live Postgres session.
    ingester = BFOOddsIngester(session=session, data_folder=tmp_path)
    summary = IngestSummary()
    bfo_to_db = ingester._match_fighters(bfo_names, summary)

    # Primary assertion: BFO ids resolved to expected canonical DB ids.
    bfo_a_id = anomaly["bfo_source"]["bfo_fighter_a"]["bfo_fighter_id"]
    bfo_b_id = anomaly["bfo_source"]["bfo_fighter_b"]["bfo_fighter_id"]
    assert bfo_to_db[bfo_a_id] == expected["anomaly_fighter_a_id"], (
        f"Anomaly BFO id {bfo_a_id!r} resolved to "
        f"{bfo_to_db.get(bfo_a_id)!r}; expected "
        f"{expected['anomaly_fighter_a_id']} (Niko Price canonical id)."
    )
    assert bfo_to_db[bfo_b_id] == expected["anomaly_fighter_b_id"], (
        f"Anomaly BFO id {bfo_b_id!r} resolved to "
        f"{bfo_to_db.get(bfo_b_id)!r}; expected "
        f"{expected['anomaly_fighter_b_id']} (James Vick canonical id)."
    )

    # Canonical-path observability: BOTH rows must take the canonical path.
    # If either had fallen through to fuzzy, the candidate pool's
    # near-name homonyms (Nico Price / Nick Price / James Vic) could have
    # mis-attributed — Plan 41-01 §2.6 cascade Stage 2->3 silent drop.
    assert summary.fighters_matched_canonical == 2, (
        f"Expected 2 canonical-path matches; got "
        f"{summary.fighters_matched_canonical}. Both BFO rows carry a "
        f"canonical database_id, so neither should have invoked the fuzzy "
        f"matcher (Plan 41-02 BFO-V25-02 contract)."
    )
    assert summary.fighters_matched_fuzzy == 0, (
        f"Expected 0 fuzzy-path matches on the anomaly row; got "
        f"{summary.fighters_matched_fuzzy}. A non-zero fuzzy count here "
        f"indicates the canonical-substrate-first contract is being "
        f"bypassed — Plan 41-02 fix regressed."
    )
    assert summary.fighters_unmatched == 0


def test_control_row_no_regression(
    fixture_data: dict[str, Any], session: Any, tmp_path: Path
) -> None:
    """No-regression integration: a non-anomaly-year (2022) control row
    seeded into real Postgres resolves identically to its pre-patch
    expected value. The healthy post-2021 cohort must not regress.
    """
    from ufc_prediction.scraper.bfo_ingest import BFOOddsIngester, IngestSummary

    control = fixture_data["control_row"]
    expected = fixture_data["expected_resolutions"]

    seeded = _seed_event_fighters_fight(session, control)
    assert seeded["fighter_a_id"] == expected["control_fighter_a_id"]
    assert seeded["fighter_b_id"] == expected["control_fighter_b_id"]

    bfo_names = _build_bfo_names_map(control)

    ingester = BFOOddsIngester(session=session, data_folder=tmp_path)
    summary = IngestSummary()
    bfo_to_db = ingester._match_fighters(bfo_names, summary)

    bfo_a_id = control["bfo_source"]["bfo_fighter_a"]["bfo_fighter_id"]
    bfo_b_id = control["bfo_source"]["bfo_fighter_b"]["bfo_fighter_id"]
    assert bfo_to_db[bfo_a_id] == expected["control_fighter_a_id"], (
        f"Control row REGRESSED: BFO id {bfo_a_id!r} resolved to "
        f"{bfo_to_db.get(bfo_a_id)!r}; expected "
        f"{expected['control_fighter_a_id']}."
    )
    assert bfo_to_db[bfo_b_id] == expected["control_fighter_b_id"], (
        f"Control row REGRESSED: BFO id {bfo_b_id!r} resolved to "
        f"{bfo_to_db.get(bfo_b_id)!r}; expected "
        f"{expected['control_fighter_b_id']}."
    )
    assert summary.fighters_matched_canonical == 2
    assert summary.fighters_matched_fuzzy == 0


def test_rollback_isolation(
    fixture_data: dict[str, Any], session: Any
) -> None:
    """Verify transactional rollback isolates the integration tests.

    Plan 41-04 threat model T-41-04-03 ("testcontainers Postgres state
    leak between tests") mitigation: the ``session`` fixture in
    ``tests/conftest.py`` wraps each test in a connection-level
    transaction and rolls back on teardown. This test seeds an event +
    fighter into the session — then asserts that no fighters from the
    ANOMALY row's fixture id space leak into this transaction (proving
    the prior ``test_anomaly_row_resolves_post_patch`` did not commit).
    """
    from sqlalchemy import select

    from ufc_prediction.models.fighter import Fighter

    anomaly = fixture_data["anomaly_row"]
    anomaly_fixture_ids = [
        f["fixture_db_id"] for f in anomaly["fighters"]
    ]

    # Before this test seeds anything: the anomaly fixture's explicit-id
    # Fighter rows must NOT exist in this session (proves prior test's
    # rollback). If they DO exist, transactional isolation has failed and
    # subsequent integration tests would see cross-test contamination.
    stmt = select(Fighter.id).where(Fighter.id.in_(anomaly_fixture_ids))
    leaked_ids = list(session.execute(stmt).scalars().all())
    assert leaked_ids == [], (
        f"Transactional rollback isolation FAILED: anomaly fixture "
        f"Fighter ids {leaked_ids} leaked from a prior test into this "
        f"session. The conftest.py session fixture's rollback semantics "
        f"are not working — investigate before extending coverage."
    )

    # Seed a sentinel fighter inside this test; teardown should roll it
    # back. We do not need to verify the rollback from inside the test
    # (the session fixture's teardown handles that and the next test in
    # the module re-checks isolation).
    sentinel = Fighter(
        name="bfo-v25-03-rollback-sentinel",
        source="test-integration-rollback",
    )
    session.add(sentinel)
    session.flush()
    assert sentinel.id is not None

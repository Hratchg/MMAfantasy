"""DEBUT-V25-03 integration test — seeded debutant first-fight Elo.

Plan 43-03 Task 3 — Real-Postgres (testcontainers) integration tier for the
seed-aware EloEngine. Verifies that the in-memory EloEngine path (the same
code path compute_elo uses in CLI) produces the seeded elo_before for a
debutant's first fight when wired against a live Postgres-backed Fighter/
Event/Fight schema.

Pattern reuses Phase 41 BFO-V25-03 integration shape:
- ``pytestmark = pytest.mark.integration`` marker on the module.
- Module-level Docker daemon skip-gate (CI without Docker -> SKIPPED, not
  ERROR). This MUST fire before pytest resolves the session-scoped
  ``postgres_container`` fixture from tests/conftest.py.
- ``session`` fixture from ``tests/conftest.py`` provides transactional
  rollback per test (Phase 37 TEMPORAL-V24-02 pattern).

NOTE: This integration tier exercises the FULL stack — fighter/event/fight
rows seeded into a live Postgres + the EloEngine consuming them — but the
Elo math itself runs in-Python (the engine is database-free; this test
verifies the integration boundary between seeded CSV + Postgres-backed
FightRecord + Elo replay).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest


# ── Docker-availability gate (per Phase 41 CONTEXT) ───────────────────────


def _docker_available() -> bool:
    """Return True iff the Docker daemon is reachable.

    The testcontainers PostgresContainer fixture requires Docker. CI
    environments / gsd-execute worktrees without Docker exposure must SKIP
    cleanly rather than letting testcontainers raise DockerException
    (which surfaces as ERROR, not SKIPPED). The skip is applied via
    ``pytestmark`` so collection still succeeds — the plan's verify command
    asserts on collected-count, and operators get visibility into which
    tests would have run had Docker been available.
    """
    try:
        import docker  # type: ignore[import-untyped]

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


_DOCKER_OK = _docker_available()

# Phase-level integration marker — matches Phase 37/41 convention.
# Combined with a skipif so tests are COLLECTED (visible to plan-verify)
# but SKIPPED when Docker is unreachable.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _DOCKER_OK,
        reason=(
            "Docker daemon unavailable -- DEBUT-V25-03 integration tier "
            "requires testcontainers Postgres (per Phase 37 TEMPORAL-V24-02 "
            "pattern). Run with Docker on to exercise the real-DB path."
        ),
    ),
]


# ── Seed helpers ───────────────────────────────────────────────────────────


def _write_seeds_csv(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write a minimal pre_ufc_records.csv with the locked 14-column schema.

    Only the seed-loading-required columns are populated with meaningful
    values; the rest receive empty strings (load_seeds enforces only the
    required-columns subset per the Plan 43-02 contract).
    """
    csv_path = tmp_path / "pre_ufc_records.csv"
    header = [
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
    ]
    lines = [",".join(header)]
    for r in rows:
        line = ",".join(str(r.get(col, "")) for col in header)
        lines.append(line)
    csv_path.write_text("\n".join(lines) + "\n")
    return csv_path


def _seed_fighters_event_fight(
    session: Any,
    *,
    fighter_a_id: int,
    fighter_b_id: int,
    event_date: date,
    winner_id: int | None,
    weight_class: str = "Lightweight",
    method: str = "Decision",
    method_detail: str = "Unanimous",
) -> dict[str, int]:
    """Insert minimal Fighter + Event + Fight rows for the integration test."""
    from ufc_prediction.models.event import Event
    from ufc_prediction.models.fight import Fight
    from ufc_prediction.models.fighter import Fighter

    fa = Fighter(id=fighter_a_id, name=f"Test Fighter A {fighter_a_id}", source="test-debut-v25-03")
    fb = Fighter(id=fighter_b_id, name=f"Test Fighter B {fighter_b_id}", source="test-debut-v25-03")
    session.add_all([fa, fb])
    session.flush()

    event = Event(
        name="DEBUT-V25-03 Integration Test Event",
        date=event_date,
        source="test-debut-v25-03",
    )
    session.add(event)
    session.flush()

    fight = Fight(
        event_id=event.id,
        fighter_a_id=fighter_a_id,
        fighter_b_id=fighter_b_id,
        winner_id=winner_id,
        weight_class=weight_class,
        method=method,
        method_detail=method_detail,
        source="test-debut-v25-03",
    )
    session.add(fight)
    session.flush()

    return {
        "fight_id": fight.id,
        "event_id": event.id,
        "fighter_a_id": fighter_a_id,
        "fighter_b_id": fighter_b_id,
    }


# ── Tests ──────────────────────────────────────────────────────────────────


def test_seeded_debutant_first_fight_elo_before(
    session: Any,
    tmp_path: Path,
) -> None:
    """A seeded debutant's first-fight SnapshotRecord has elo_before == seed.

    Closes the primary half of REQ DEBUT-V25-03 — the seed flows from the
    CSV through ``load_seeds`` -> ``EloEngine(seeds=...)`` -> first-encounter
    dispatch -> emitted SnapshotRecord.elo_before, end-to-end. Postgres is in
    the loop via the Fighter/Event/Fight rows that feed FightRecord.
    """
    from ufc_prediction.elo.config import EloConfig
    from ufc_prediction.elo.engine import EloEngine, FightRecord
    from ufc_prediction.elo.seed import load_seeds

    # Seed CSV: fighter A=1 has a major-tier 100% win-rate 20-fight record
    # -> derive_seed yields 1625.0 (per Plan 43-02 derive_seed corner-case
    # table). Fighter B=2 has NO seed row.
    csv = _write_seeds_csv(
        tmp_path,
        [
            {
                "fighter_id": 1,
                "n_pre_ufc_fights": 20,
                "wins": 20,
                "losses": 0,
                "draws": 0,
                "nc_dq": 0,
                "win_rate": 1.0,
                "kos": 10,
                "submissions": 5,
                "decisions": 5,
                "last_organization": "Bellator 290",
                "org_tier": "major",
                "scraped_at": "2026-06-02T00:00:00+00:00",
            },
        ],
    )
    seeds = load_seeds(csv)
    assert seeds == {1: 1625.0}, f"Pre-condition: seeds dict mismatch -> {seeds}"

    # Seed the Postgres-backed schema (proves the integration boundary works
    # — the test would catch ORM/FK/transaction quirks even though the Elo
    # math is in-Python).
    seeded = _seed_fighters_event_fight(
        session,
        fighter_a_id=1,
        fighter_b_id=2,
        event_date=date(2020, 1, 1),
        winner_id=1,
    )

    # Build the FightRecord (compute_elo's load_fights_chronological returns
    # this shape; for the integration tier we construct one directly).
    fight = FightRecord(
        fight_id=seeded["fight_id"],
        event_date=date(2020, 1, 1),
        fighter_a_id=seeded["fighter_a_id"],
        fighter_b_id=seeded["fighter_b_id"],
        winner_id=seeded["fighter_a_id"],
        weight_class="Lightweight",
        method="Decision",
        method_detail="Unanimous",
    )

    engine = EloEngine(EloConfig(), seeds=seeds)
    snaps = engine.compute_all([fight])

    assert len(snaps) == 2, f"Expected 2 snapshots; got {len(snaps)}"
    snap_a = next(s for s in snaps if s.fighter_id == 1)
    snap_b = next(s for s in snaps if s.fighter_id == 2)

    # PRIMARY ASSERTION: seeded fighter A's first-fight elo_before == seed.
    assert snap_a.elo_before == 1625.0, (
        f"Seeded debutant A first-fight elo_before={snap_a.elo_before!r}; "
        f"expected seed value 1625.0 (load_seeds + EloEngine seed dispatch)."
    )
    # Unseeded B falls back to config.initial_rating (1500.0).
    assert snap_b.elo_before == 1500.0, (
        f"Unseeded debutant B first-fight elo_before={snap_b.elo_before!r}; "
        f"expected config.initial_rating 1500.0 (fallback path)."
    )


def test_unseeded_debutant_falls_back_to_1500(
    session: Any,
    tmp_path: Path,
) -> None:
    """Empty seeds CSV -> all fighters fall back to config.initial_rating.

    This is the graceful-degradation contract from Plan 43-02 (load_seeds
    returns {} when CSV is missing/empty) combined with the empty-seeds-is-
    no-op invariant from Plan 43-03 Task 1 Test 3 — wired end-to-end through
    real Postgres rows.
    """
    from ufc_prediction.elo.config import EloConfig
    from ufc_prediction.elo.engine import EloEngine, FightRecord
    from ufc_prediction.elo.seed import load_seeds

    # Empty CSV (header only).
    csv = _write_seeds_csv(tmp_path, [])
    seeds = load_seeds(csv)
    assert seeds == {}, f"Empty CSV must produce empty seeds dict; got {seeds}"

    seeded = _seed_fighters_event_fight(
        session,
        fighter_a_id=10,
        fighter_b_id=11,
        event_date=date(2021, 6, 1),
        winner_id=10,
    )

    fight = FightRecord(
        fight_id=seeded["fight_id"],
        event_date=date(2021, 6, 1),
        fighter_a_id=seeded["fighter_a_id"],
        fighter_b_id=seeded["fighter_b_id"],
        winner_id=seeded["fighter_a_id"],
        weight_class="Lightweight",
        method="Decision",
        method_detail="Unanimous",
    )

    engine = EloEngine(EloConfig(), seeds=seeds)
    snaps = engine.compute_all([fight])
    assert len(snaps) == 2
    for snap in snaps:
        assert snap.elo_before == 1500.0, (
            f"Empty-seeds fallback FAILED: fighter {snap.fighter_id} got "
            f"elo_before={snap.elo_before!r}; expected 1500.0."
        )


def test_seeded_first_fight_uses_expected_win_probability_from_seed(
    session: Any,
    tmp_path: Path,
) -> None:
    """Two seeded fighters meet -> win-probability + delta use the seeds.

    Proves the seed propagates into the actual Elo math (not just the
    elo_before field). Asserts numerically against the formula
    ``E(A) = 1 / (1 + 10^((R_B - R_A) / 400))`` using the seed values.
    """
    from ufc_prediction.elo.config import EloConfig
    from ufc_prediction.elo.engine import EloEngine, FightRecord
    from ufc_prediction.elo.seed import load_seeds

    # Two seeded fighters: A has major + 100% + 20-fight (seed=1625), B has
    # regional + 50% + 0-fight (seed=1500).
    csv = _write_seeds_csv(
        tmp_path,
        [
            {
                "fighter_id": 100,
                "n_pre_ufc_fights": 20,
                "wins": 20,
                "losses": 0,
                "draws": 0,
                "nc_dq": 0,
                "win_rate": 1.0,
                "kos": 10,
                "submissions": 5,
                "decisions": 5,
                "last_organization": "Bellator 290",
                "org_tier": "major",
                "scraped_at": "2026-06-02T00:00:00+00:00",
            },
            {
                "fighter_id": 200,
                "n_pre_ufc_fights": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "nc_dq": 0,
                "win_rate": 0.5,
                "kos": 0,
                "submissions": 0,
                "decisions": 0,
                "last_organization": "LFA",
                "org_tier": "regional",
                "scraped_at": "2026-06-02T00:00:00+00:00",
            },
        ],
    )
    seeds = load_seeds(csv)
    assert seeds == {100: 1625.0, 200: 1500.0}, f"Pre-condition: seeds dict mismatch -> {seeds}"

    seeded = _seed_fighters_event_fight(
        session,
        fighter_a_id=100,
        fighter_b_id=200,
        event_date=date(2022, 1, 1),
        winner_id=100,
    )

    fight = FightRecord(
        fight_id=seeded["fight_id"],
        event_date=date(2022, 1, 1),
        fighter_a_id=seeded["fighter_a_id"],
        fighter_b_id=seeded["fighter_b_id"],
        winner_id=seeded["fighter_a_id"],
        weight_class="Lightweight",
        method="Decision",
        method_detail="Unanimous",
    )

    engine = EloEngine(EloConfig(), seeds=seeds)

    # The expected win probability for A (1625) over B (1500), computed via
    # the engine's own formula. Both inputs come from the seed values, NOT
    # from config.initial_rating.
    expected_p_a = engine.expected_win_probability(1625.0, 1500.0)
    # Sanity bound: A is the higher-seeded fighter, so P(A wins) > 0.5.
    assert 0.5 < expected_p_a < 1.0, (
        f"expected_win_probability sanity failed: P(A wins)={expected_p_a!r}"
    )

    snaps = engine.compute_all([fight])
    snap_a = next(s for s in snaps if s.fighter_id == 100)
    snap_b = next(s for s in snaps if s.fighter_id == 200)

    assert snap_a.elo_before == 1625.0
    assert snap_b.elo_before == 1500.0

    # Delta math: both fighters at fight_count=0 -> K_initial=40.0.
    # MOV unanimous decision = 1.0. A wins -> delta_a = 40 * 1.0 * (1 - e_a).
    expected_delta_a = 40.0 * 1.0 * (1.0 - expected_p_a)
    expected_delta_b = 40.0 * 1.0 * (0.0 - (1.0 - expected_p_a))
    assert snap_a.elo_after == pytest.approx(1625.0 + expected_delta_a), (
        f"snap_a.elo_after={snap_a.elo_after!r}; expected "
        f"{1625.0 + expected_delta_a!r} (from seed-derived expected_p_a)."
    )
    assert snap_b.elo_after == pytest.approx(1500.0 + expected_delta_b), (
        f"snap_b.elo_after={snap_b.elo_after!r}; expected "
        f"{1500.0 + expected_delta_b!r} (from seed-derived expected_p_a)."
    )

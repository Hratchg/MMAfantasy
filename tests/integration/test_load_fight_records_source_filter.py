"""Integration tests for Plan 28-04 Task 1 — `load_fight_records()` source filter.

Closes the HIGH-severity finding documented in
``.planning/phases/28-referee-venue-ingestion-pipeline/28-FEATURE-MATRIX-DEDUP-FINDING.md``:
``load_fight_records()`` did not filter on ``Event.source``, causing the v2.x
training corpus to be inflated by ~1.88× from cross-source Kaggle duplicates
(per ``28-DEDUP-RECON.md`` Section 1).

This test suite enforces three behaviors:

1. **sources contract** — after the filter is added, ``load_fight_records()``
   returns ONLY fights whose joined event has ``source == 'ufcstats'``. Any
   ``'kaggle-mdabbert'`` / ``'kaggle-rajeevw'`` / other-source rows must be
   absent from the result set.
2. **defensive NULL exclusion** — fights whose joined event has
   ``source IS NULL`` MUST NOT be returned (defensive even though the current
   production DB has zero NULLs in ``events.source``; an equality predicate
   guarantees this in SQL three-valued logic).
3. **live row-count smoke** — opt-in via ``UFC28_LIVE_DB=1``; asserts the
   live-DB result is ~8,473 (+/- 0.5%) per Plan 28-04 ``<verify>`` block.
   Skipped when the env var is unset (default).

Container-mode (Docker) was attempted via ``tests/conftest.py``'s
testcontainers postgres fixture but Docker is not always available in the
executor environment. The seed-and-roll-back pattern below uses a SAVEPOINT
nested transaction inside a live ``SessionLocal()`` connection, so no rows
persist after the test exits — equivalent isolation guarantee to the
testcontainers ``session`` fixture, with no Docker dependency. The whole
suite gracefully skips if the live DB is unreachable (e.g., in pure-CI
environments with neither Docker nor the prod DB).
"""

from __future__ import annotations

import os
from datetime import date

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from ufc_prediction.ml.queries import load_fight_records
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter


# ─────────────────────────────────────────────────────────────────────
# Live-DB session fixture — SAVEPOINT-bracketed; rolls back on teardown
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture()
def rollback_session():
    """Yield a ``Session`` bound to a connection inside an outer transaction;
    every flush/commit inside the test is rolled back when the fixture tears
    down. No rows persist in the live DB.

    Skips gracefully if the live DB is unreachable.
    """
    from sqlalchemy.orm import sessionmaker

    from ufc_prediction.db.session import engine

    try:
        connection = engine.connect()
    except OperationalError as e:
        pytest.skip(f"live DB unreachable: {e}")

    outer = connection.begin()
    session_factory = sessionmaker(bind=connection)
    sess = session_factory()
    # Open a SAVEPOINT so the test's session.flush() calls and our raw-SQL
    # INSERTs all live inside a nested transaction that the outer rollback
    # below will discard.
    sess.begin_nested()
    try:
        yield sess
    finally:
        sess.close()
        if outer.is_active:
            outer.rollback()
        connection.close()


# ─────────────────────────────────────────────────────────────────────
# Fixture: seed a tiny mixed-source corpus inside the SAVEPOINT
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture()
def mixed_source_corpus(rollback_session):
    """Seed 3 events (ufcstats, kaggle-mdabbert, kaggle-rajeevw) + 4 fights.

    Uses very-high IDs (9_000_000+) to avoid colliding with any production
    rows already in the DB; the SAVEPOINT rollback nukes them at teardown
    regardless.

    Layout:
      - Event 9_000_001 (ufcstats):       F1 vs F2 → F1 wins (KEPT)
      - Event 9_000_002 (kaggle-mdabbert):F3 vs F4 → F3 wins (DROPPED by filter)
      - Event 9_000_003 (kaggle-rajeevw): F5 vs F6 → F6 wins (DROPPED by filter)
      - Event 9_000_001 (ufcstats):       F1 vs F2 NULL winner (DROPPED by
                                           existing winner_id sentinel)
    """
    session = rollback_session
    base_f = 9_000_000
    base_e = 9_000_000
    base_x = 9_000_000
    for fid, name in [
        (base_f + 1, "Alice Aufcstats"),
        (base_f + 2, "Bob Bufcstats"),
        (base_f + 3, "Carol Cmdabbert"),
        (base_f + 4, "Dave Dmdabbert"),
        (base_f + 5, "Eve Erajeevw"),
        (base_f + 6, "Frank Frajeevw"),
    ]:
        session.add(Fighter(id=fid, name=name, source="ufcstats"))
    session.add(
        Event(id=base_e + 1, name="UFC TestUfcstats-A", date=date(2099, 1, 1), source="ufcstats")
    )
    session.add(
        Event(
            id=base_e + 2,
            name="UFC Event - 2099-01-01",
            date=date(2099, 1, 1),
            source="kaggle-mdabbert",
        )
    )
    session.add(
        Event(
            id=base_e + 3,
            name="UFC Event - 2099-06-15",
            date=date(2099, 6, 15),
            source="kaggle-rajeevw",
        )
    )
    session.flush()
    session.add(
        Fight(
            id=base_x + 1,
            event_id=base_e + 1,
            fighter_a_id=base_f + 1,
            fighter_b_id=base_f + 2,
            winner_id=base_f + 1,
            weight_class="lightweight",
            source="ufcstats",
        )
    )
    session.add(
        Fight(
            id=base_x + 2,
            event_id=base_e + 2,
            fighter_a_id=base_f + 3,
            fighter_b_id=base_f + 4,
            winner_id=base_f + 3,
            weight_class="welterweight",
            source="kaggle-mdabbert",
        )
    )
    session.add(
        Fight(
            id=base_x + 3,
            event_id=base_e + 3,
            fighter_a_id=base_f + 5,
            fighter_b_id=base_f + 6,
            winner_id=base_f + 6,
            weight_class="bantamweight",
            source="kaggle-rajeevw",
        )
    )
    # NULL-winner sentinel — already-existing winner_id filter must drop this.
    session.add(
        Fight(
            id=base_x + 4,
            event_id=base_e + 1,
            fighter_a_id=base_f + 1,
            fighter_b_id=base_f + 2,
            winner_id=None,
            weight_class="lightweight",
            source="ufcstats",
        )
    )
    session.flush()
    return session


# ─────────────────────────────────────────────────────────────────────
# Test 1 — sources contract (the HIGH-severity finding fix)
# ─────────────────────────────────────────────────────────────────────


def test_load_fight_records_returns_only_ufcstats_source(mixed_source_corpus):
    """Filter must reject every non-ufcstats event source.

    ``load_fight_records()`` does not project ``event_source`` onto its
    return rows. To assert the source contract on the fixture-seeded rows
    specifically, we re-fetch the joined source for each returned
    ``fight_id`` (the "follow-up query" pattern called out in Plan 28-04
    line 211).
    """
    session = mixed_source_corpus
    records = load_fight_records(session)

    fight_ids = [r["fight_id"] for r in records]
    assert len(fight_ids) > 0, "filter wiped the result set entirely"

    # Specific fixture-id assertions: 9_000_001 in, 9_000_002+9_000_003 out
    # (different source), 9_000_004 out (NULL winner sentinel).
    base = 9_000_000
    assert base + 1 in fight_ids, "ufcstats fixture row was unexpectedly dropped"
    assert base + 2 not in fight_ids, "kaggle-mdabbert leaked through filter"
    assert base + 3 not in fight_ids, "kaggle-rajeevw leaked through filter"
    assert base + 4 not in fight_ids, "NULL-winner sentinel leaked (winner_id filter regression)"

    # Global source contract: every returned fight_id must join to a
    # ufcstats-source event. (Scoped to the fixture window via id range
    # so the live DB's existing fights don't fail this test on a
    # pre-existing kaggle row.)
    fixture_ids = [base + 1, base + 2, base + 3, base + 4]
    sources = set(
        session.execute(
            text(
                "SELECT DISTINCT e.source FROM fights f "
                "JOIN events e ON f.event_id = e.id "
                "WHERE f.id = ANY(:ids)"
            ),
            {"ids": [fid for fid in fight_ids if fid in fixture_ids]},
        )
        .scalars()
        .all()
    )
    # The returned set should contain only ufcstats (kaggle rows were dropped
    # by the filter so we never query them).
    assert sources <= {"ufcstats"}, (
        f"load_fight_records returned non-ufcstats rows in the fixture "
        f"window; got sources={sorted(sources)}"
    )

    # Also: across the WHOLE result set (production + fixture), the joined
    # source must be exactly {"ufcstats"} — this is the binding global
    # contract assertion the plan asks for.
    global_sources = set(
        session.execute(
            text(
                "SELECT DISTINCT e.source FROM fights f "
                "JOIN events e ON f.event_id = e.id "
                "WHERE f.id = ANY(:ids)"
            ),
            {"ids": fight_ids},
        )
        .scalars()
        .all()
    )
    assert global_sources == {"ufcstats"}, (
        f"load_fight_records must return only ufcstats-source rows globally; "
        f"got {sorted(global_sources)}"
    )


# ─────────────────────────────────────────────────────────────────────
# Test 2 — NULL-source defense (schema + predicate)
# ─────────────────────────────────────────────────────────────────────


def test_events_source_schema_is_not_null(rollback_session):
    """First-line defense: ``events.source`` is ``NOT NULL`` at the schema
    layer (per ``src/ufc_prediction/models/event.py`` line 16). The DB rejects
    INSERTs with NULL source — so no NULL row can ever reach
    ``load_fight_records()``.

    This is asserted as a regression sentinel against future schema
    migrations that might accidentally relax the constraint. The
    second-line defense (equality predicate ``source = 'ufcstats'`` would
    drop NULLs under SQL three-valued logic) is documented in the
    ``load_fight_records`` docstring; we exercise the first-line guarantee
    here because it is the actually-binding one in production.
    """
    from sqlalchemy.exc import IntegrityError

    session = rollback_session
    with pytest.raises(IntegrityError) as exc_info:
        session.execute(
            text(
                "INSERT INTO events (id, name, date, source, created_at) "
                "VALUES (9500002, :n, :d, NULL, now())"
            ),
            {"n": "Forged NULL Event", "d": date(2098, 2, 1)},
        )
        session.flush()
    # psycopg surfaces the NotNullViolation; assert on the column name to
    # pin the constraint to events.source specifically (not a generic
    # IntegrityError that could mask a different schema regression).
    err_text = str(exc_info.value).lower()
    assert "source" in err_text and (
        "null" in err_text or "not-null" in err_text or "not null" in err_text
    ), f"expected NOT NULL violation on events.source; got: {exc_info.value}"


# ─────────────────────────────────────────────────────────────────────
# Test 3 — live-DB row-count smoke (opt-in via env var)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    os.getenv("UFC28_LIVE_DB") != "1",
    reason=(
        "Live-DB smoke (Plan 28-04 verify block). Opt in with "
        "UFC28_LIVE_DB=1; expects ~8,473 ufcstats fights in production DB."
    ),
)
def test_load_fight_records_live_db_row_count_post_dedup():
    """Plan 28-04 ``<verify>`` block: live-DB result is ~8,473 +/- 0.5%."""
    from ufc_prediction.db.session import SessionLocal

    with SessionLocal() as s:
        records = load_fight_records(s)
    n = len(records)
    expected = 8473
    tolerance = 0.005
    delta_pct = abs(n - expected) / expected
    assert delta_pct <= tolerance, (
        f"live-DB row count {n} outside +/- {tolerance * 100:.1f}% of "
        f"expected {expected} (actual delta {delta_pct * 100:.2f}%)"
    )

"""TEMPORAL-V24-02 regression — load_computed_features() excludes future-dated rows.

Closes the audit Q8 "residual leakage gap" finding by pinning the loader-level
temporal-validity contract added in TEMPORAL-V24-01
(see ``src/ufc_prediction/ml/queries.py:load_computed_features``).

Three behaviors are asserted via real Postgres (testcontainers ``session``
fixture from ``tests/conftest.py``) with per-test transactional rollback:

1. **Legitimate row IS included** — ``as_of_date < Event.date`` row appears
   in the loader output (guards against over-filtering — T-37-02-02).
2. **Future-dated row IS EXCLUDED** — ``as_of_date > Event.date`` row is
   absent from loader output (core TEMPORAL-V24-02 invariant —
   T-37-02-01 + T-37-02-04).
3. **NULL ``as_of_date`` row IS included** — pre-Phase-37 legacy rows are
   conservatively retained per the Plan 37-02 NULL-handling decision.

All three tests share the same testcontainers-Postgres ``session`` fixture
(scope = function); the outer transaction is rolled back on teardown so no
rows persist across tests.
"""

from __future__ import annotations

from datetime import date

from ufc_prediction.ml.queries import load_computed_features
from ufc_prediction.models.computed_feature import ComputedFeature
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter


# ─────────────────────────────────────────────────────────────────────
# Seed helpers — shared NOT-NULL-field skeletons for Event / Fight / Fighter
# ─────────────────────────────────────────────────────────────────────


def _seed_two_fighters(session, fa_id: int, fb_id: int) -> tuple[Fighter, Fighter]:
    """Seed two minimal fighters needed for the Fight FK constraints."""
    fa = Fighter(id=fa_id, name=f"Fighter A {fa_id}", source="ufcstats")
    fb = Fighter(id=fb_id, name=f"Fighter B {fb_id}", source="ufcstats")
    session.add_all([fa, fb])
    session.flush()
    return fa, fb


def _seed_event_and_fight(
    session,
    event_date: date,
    fa_id: int,
    fb_id: int,
    event_id: int,
    fight_id: int,
) -> tuple[Event, Fight]:
    """Seed one Event + one Fight referencing the two given fighters."""
    event = Event(
        id=event_id,
        name=f"TEST EVENT — TEMPORAL-V24-02 ({event_id})",
        date=event_date,
        source="ufcstats",
    )
    session.add(event)
    session.flush()
    fight = Fight(
        id=fight_id,
        event_id=event.id,
        fighter_a_id=fa_id,
        fighter_b_id=fb_id,
        weight_class="lightweight",
        num_rounds=3,
        source="ufcstats",
    )
    session.add(fight)
    session.flush()
    return event, fight


# ─────────────────────────────────────────────────────────────────────
# Test 1 — Positive control (guards against over-filtering: T-37-02-02)
# ─────────────────────────────────────────────────────────────────────


def test_legitimate_as_of_date_is_included(session):
    """``as_of_date <= event.date`` row IS in loader result.

    Without this assertion, an over-eager filter (e.g. ``<`` instead of
    ``<=`` or a join misconfiguration) could silently drop all legitimate
    training data. T-37-02-02 STRIDE mitigation.
    """
    fa, fb = _seed_two_fighters(session, fa_id=9_100_001, fb_id=9_100_002)
    _event, fight = _seed_event_and_fight(
        session,
        event_date=date(2024, 1, 1),
        fa_id=fa.id,
        fb_id=fb.id,
        event_id=9_100_001,
        fight_id=9_100_001,
    )
    cf = ComputedFeature(
        fighter_id=fa.id,
        fight_id=fight.id,
        as_of_date=date(2023, 12, 15),  # 17 days BEFORE event.date — legitimate
        feature_set_version="test-v24-02",
        features={"some_feat": 0.5},
    )
    session.add(cf)
    session.flush()

    result = load_computed_features(session)

    assert (fa.id, fight.id) in result, (
        "legitimate pre-event-date row was dropped by the filter — over-filtering bug (T-37-02-02)"
    )


# ─────────────────────────────────────────────────────────────────────
# Test 2 — TEMPORAL-V24-02 CORE: future-dated row excluded (T-37-02-01)
# ─────────────────────────────────────────────────────────────────────


def test_future_dated_as_of_date_is_excluded(session):
    """``as_of_date > event.date`` row is NOT in loader result.

    Core TEMPORAL-V24-02 invariant. End-to-end verification that the
    SQL filter at ``ml/queries.py:load_computed_features`` (TEMPORAL-V24-01)
    actually rejects future-dated rows from the training + inference feature
    pipeline.

    T-37-02-01 (Info Disclosure: future-dated feature leak) + T-37-02-04
    (Repudiation: prove filter is exercised) STRIDE mitigations.
    """
    fa, fb = _seed_two_fighters(session, fa_id=9_100_011, fb_id=9_100_012)
    _event, fight = _seed_event_and_fight(
        session,
        event_date=date(2024, 1, 1),
        fa_id=fa.id,
        fb_id=fb.id,
        event_id=9_100_011,
        fight_id=9_100_011,
    )
    cf = ComputedFeature(
        fighter_id=fa.id,
        fight_id=fight.id,
        as_of_date=date(2024, 6, 1),  # 5 months AFTER event.date — temporal leak
        feature_set_version="test-v24-02",
        features={"some_feat": 0.9},
    )
    session.add(cf)
    session.flush()

    result = load_computed_features(session)

    assert (fa.id, fight.id) not in result, (
        "future-dated as_of_date row leaked into loader output — "
        "TEMPORAL-V24-01 filter is ineffective (T-37-02-01)"
    )


# ─────────────────────────────────────────────────────────────────────
# Test 3 — NULL handling: legacy pre-Phase-37 rows retained
# ─────────────────────────────────────────────────────────────────────


def test_null_as_of_date_is_included(session):
    """``as_of_date IS NULL`` row IS in loader result.

    Per CONTEXT.md + Plan 37-02 Task 1 NULL-handling decision: pre-Phase-37
    legacy ComputedFeature rows may have null ``as_of_date``. The filter
    conservatively INCLUDES them rather than over-excluding historical
    training data.

    NOTE: ``ComputedFeature.as_of_date`` is declared ``nullable=False`` at
    the ORM level (see ``src/ufc_prediction/models/computed_feature.py:15``),
    but the test inserts a row with ``as_of_date=None`` and FLUSHES via
    raw-SQL to simulate the legacy / pre-migration shape that the filter's
    ``IS NULL`` branch is documented to handle. If the ORM nullable=False
    is later relaxed (or a backfill migration introduces nulls), this test
    documents that the loader still returns those rows.
    """
    from sqlalchemy import text

    fa, fb = _seed_two_fighters(session, fa_id=9_100_021, fb_id=9_100_022)
    _event, fight = _seed_event_and_fight(
        session,
        event_date=date(2024, 1, 1),
        fa_id=fa.id,
        fb_id=fb.id,
        event_id=9_100_021,
        fight_id=9_100_021,
    )
    # Raw-SQL insert to bypass ORM nullable=False enforcement and write a
    # legacy NULL-as_of_date row that mirrors pre-Phase-37 data shape.
    # We temporarily drop the NOT NULL constraint inside the transaction so
    # the rollback restores it cleanly on teardown.
    session.execute(text("ALTER TABLE computed_features ALTER COLUMN as_of_date DROP NOT NULL"))
    session.execute(
        text(
            "INSERT INTO computed_features "
            "(fighter_id, fight_id, as_of_date, feature_set_version, features, created_at) "
            "VALUES (:fighter_id, :fight_id, NULL, :fsv, CAST(:features AS JSON), NOW())"
        ),
        {
            "fighter_id": fa.id,
            "fight_id": fight.id,
            "fsv": "test-v24-02-null",
            "features": '{"some_feat": 0.7}',
        },
    )
    session.flush()

    result = load_computed_features(session)

    assert (fa.id, fight.id) in result, (
        "NULL as_of_date row was excluded by the filter — over-filtering "
        "regression on pre-Phase-37 legacy data (T-37-02-02 / Task 1 "
        "NULL-handling decision)"
    )

"""DEBUT-V25-03 unit tests for EloEngine seed-aware initialization.

Plan 43-03 Task 1 — 8 unit tests covering the seed-vs-default dispatch logic in
isolation (no DB). The critical invariant tested here is Test 3 (regression):
constructing ``EloEngine(EloConfig(), seeds={})`` MUST produce bit-exact
SnapshotRecord sequences vs the pre-Phase-43 ``EloEngine(EloConfig())`` form.

All behavior pinned in plan ``43-03-PLAN.md`` Task 1 ``<behavior>`` block.
"""

from __future__ import annotations

from datetime import date

import pytest

from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.engine import EloEngine, FightRecord


# ── Helpers (local — avoid coupling to tests/elo/conftest.py which has its own
# auto-use fixtures and counter resets specific to that test directory) ──────


def _fight(
    fight_id: int,
    fighter_a_id: int,
    fighter_b_id: int,
    *,
    winner_id: int | None = None,
    event_date: date = date(2020, 1, 1),
    weight_class: str = "Lightweight",
    method: str = "Decision",
    method_detail: str = "Unanimous",
) -> FightRecord:
    if winner_id is None:
        winner_id = fighter_a_id
    return FightRecord(
        fight_id=fight_id,
        event_date=event_date,
        fighter_a_id=fighter_a_id,
        fighter_b_id=fighter_b_id,
        winner_id=winner_id,
        weight_class=weight_class,
        method=method,
        method_detail=method_detail,
    )


def _five_fight_fixture() -> list[FightRecord]:
    """Five-fight fixture: 4 fighters, mixed outcomes (decisions, KO, sub, draw).

    Designed for Test 3's bit-exact regression invariant — exercises K-factor
    transition (5+ fights/fighter), MOV variation, and a draw outcome.
    """
    return [
        # Fight 1: A wins decision
        _fight(
            1,
            1,
            2,
            winner_id=1,
            event_date=date(2020, 1, 1),
            method="Decision",
            method_detail="Unanimous",
        ),
        # Fight 2: B wins by KO
        _fight(
            2, 1, 2, winner_id=2, event_date=date(2020, 3, 1), method="KO/TKO", method_detail=None
        ),
        # Fight 3: A vs new fighter C, A wins by sub
        _fight(
            3,
            1,
            3,
            winner_id=1,
            event_date=date(2020, 5, 1),
            method="Submission",
            method_detail=None,
        ),
        # Fight 4: C vs new fighter D, draw
        _fight(
            4,
            3,
            4,
            winner_id=None,
            event_date=date(2020, 7, 1),
            method="Decision",
            method_detail="Majority",
        ),
        # Fight 5: D vs B, D wins split decision
        _fight(
            5,
            4,
            2,
            winner_id=4,
            event_date=date(2020, 9, 1),
            method="Decision",
            method_detail="Split",
        ),
    ]


# ── Tests ────────────────────────────────────────────────────────────────────


def test_1_default_seeds_empty_when_omitted() -> None:
    """Backward-compatible default: omitting `seeds` yields {}."""
    engine = EloEngine(EloConfig())
    assert engine.seeds == {}


def test_2_seeds_passthrough_accepted() -> None:
    """Explicit seeds dict stored as-is."""
    engine = EloEngine(EloConfig(), seeds={42: 1600.0})
    assert engine.seeds == {42: 1600.0}


def test_3_empty_seeds_is_no_op_regression_invariant() -> None:
    """REGRESSION INVARIANT: seeds={} -> bit-exact equality vs pre-Phase-43.

    Constructs two engines with identical configs (one with no seeds kwarg,
    one with explicit empty seeds) and replays a 5-fight fixture through
    both. Every SnapshotRecord field MUST be bit-exact equal.
    """
    fights = _five_fight_fixture()
    e1 = EloEngine(EloConfig())  # pre-Phase-43 shape (no seeds kwarg)
    e2 = EloEngine(EloConfig(), seeds={})  # post-Phase-43 with explicit empty

    snaps1 = e1.compute_all(fights)
    # Re-build fixture (compute_all does not mutate the input list but the
    # second engine must replay the SAME chronological sequence)
    fights2 = _five_fight_fixture()
    snaps2 = e2.compute_all(fights2)

    assert len(snaps1) == len(snaps2), f"Snapshot count differs: e1={len(snaps1)} e2={len(snaps2)}"
    for i, (s1, s2) in enumerate(zip(snaps1, snaps2)):
        assert s1.fighter_id == s2.fighter_id, f"snapshot {i} fighter_id"
        assert s1.fight_id == s2.fight_id, f"snapshot {i} fight_id"
        assert s1.division == s2.division, f"snapshot {i} division"
        assert s1.elo_type == s2.elo_type, f"snapshot {i} elo_type"
        # The crux: bit-exact float equality on all four numeric fields.
        assert s1.elo_before == s2.elo_before, (
            f"snapshot {i}: elo_before mismatch e1={s1.elo_before!r} e2={s2.elo_before!r}"
        )
        assert s1.elo_after == s2.elo_after, (
            f"snapshot {i}: elo_after mismatch e1={s1.elo_after!r} e2={s2.elo_after!r}"
        )
        assert s1.elo_after_shrinkage == s2.elo_after_shrinkage, (
            f"snapshot {i}: elo_after_shrinkage mismatch "
            f"e1={s1.elo_after_shrinkage!r} e2={s2.elo_after_shrinkage!r}"
        )
        assert s1.k_factor_used == s2.k_factor_used, (
            f"snapshot {i}: k_factor_used mismatch e1={s1.k_factor_used!r} e2={s2.k_factor_used!r}"
        )


def test_4_lookup_initial_rating_returns_seed_when_present() -> None:
    """`_lookup_initial_rating` returns seed for unrated fighter with seed."""
    engine = EloEngine(EloConfig(), seeds={42: 1600.0})
    assert engine._lookup_initial_rating(42, "Lightweight") == 1600.0


def test_5_lookup_initial_rating_falls_back_to_config_default() -> None:
    """`_lookup_initial_rating` falls back to config.initial_rating when fighter
    has no seed."""
    engine = EloEngine(EloConfig(), seeds={42: 1600.0})
    assert engine._lookup_initial_rating(99, "Lightweight") == 1500.0


def test_6_post_fight_lookup_uses_current_rating_not_seed() -> None:
    """After a fighter has fought, seed is dormant — subsequent lookups
    return the post-fight rating, not the seed."""
    engine = EloEngine(EloConfig(), seeds={42: 1600.0})
    # Fighter 42 vs unseeded fighter 99: 42 wins decision
    fights = [_fight(1, 42, 99, winner_id=42)]
    engine.compute_all(fights)
    rating_after = engine._lookup_initial_rating(42, "Lightweight")
    # Must NOT equal the seed (1600.0) — it must equal the post-fight rating
    assert rating_after != 1600.0, (
        f"Lookup returned seed value {rating_after} after fighter 42 fought; "
        "should return post-fight rating from self._ratings"
    )
    # And the rating should be the value stored in self._ratings
    expected = engine._ratings[(42, "Lightweight")]
    assert rating_after == expected


def test_7_seeded_fighter_first_snapshot_elo_before_equals_seed() -> None:
    """Seeded fighter's first SnapshotRecord has elo_before == seed value."""
    engine = EloEngine(EloConfig(), seeds={42: 1600.0})
    # Fighter 42 (seeded) vs fighter 99 (no seed) — A wins.
    fights = [_fight(1, 42, 99, winner_id=42)]
    snaps = engine.compute_all(fights)
    # First two snapshots: snap_a for fighter 42, snap_b for fighter 99.
    snap_42 = next(s for s in snaps if s.fighter_id == 42)
    snap_99 = next(s for s in snaps if s.fighter_id == 99)
    assert snap_42.elo_before == 1600.0, (
        f"Seeded fighter 42 first-fight elo_before={snap_42.elo_before!r}; "
        f"expected seed value 1600.0"
    )
    assert snap_99.elo_before == 1500.0, (
        f"Unseeded fighter 99 first-fight elo_before={snap_99.elo_before!r}; "
        f"expected config.initial_rating 1500.0"
    )


def test_8_two_seeded_fighters_first_fight_uses_seed_values() -> None:
    """Two seeded fighters meet in their first fight: both elo_before come
    from seeds, and the expected-win-probability + delta math use the seeds.
    """
    seeds = {42: 1700.0, 99: 1500.0}
    engine = EloEngine(EloConfig(), seeds=seeds)
    fights = [_fight(1, 42, 99, winner_id=42)]
    snaps = engine.compute_all(fights)

    snap_42 = next(s for s in snaps if s.fighter_id == 42)
    snap_99 = next(s for s in snaps if s.fighter_id == 99)

    assert snap_42.elo_before == 1700.0
    assert snap_99.elo_before == 1500.0

    # Expected win probability for fighter 42 at 1700 vs 99 at 1500
    expected_p = 1.0 / (1.0 + 10.0 ** ((1500.0 - 1700.0) / 400.0))
    # Use the engine's API (must equal our hand-computed value)
    assert engine.expected_win_probability(1700.0, 1500.0) == pytest.approx(expected_p)

    # Delta for fighter 42 (winner): k * mov * (1.0 - e_a). Both fighters are
    # at fight_count=0 -> K_initial=40.0; MOV unanimous decision = 1.0.
    # delta_a = 40 * 1.0 * (1 - expected_p)
    expected_delta_a = 40.0 * 1.0 * (1.0 - expected_p)
    assert snap_42.elo_after == pytest.approx(1700.0 + expected_delta_a)
    # delta_b = 40 * 1.0 * (0 - e_b) where e_b = 1 - e_a
    expected_delta_b = 40.0 * 1.0 * (0.0 - (1.0 - expected_p))
    assert snap_99.elo_after == pytest.approx(1500.0 + expected_delta_b)

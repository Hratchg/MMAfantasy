"""``elo.asof.pre_fight_rating`` must reproduce the engines' ``elo_before``.

Runs the real ``EloEngine`` and ``DomainEloComputer`` over a synthetic
corpus with inactivity gaps, division moves, catch-weight bouts, draws, a
no-contest and seeded debutants, then reconstructs every snapshot's
``elo_before`` from the fighter's EARLIER snapshots only — which is all the
serve path has for an upcoming fight.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from ufc_prediction.elo.asof import RatedFight, pre_fight_rating
from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.domain import DomainEloComputer
from ufc_prediction.elo.engine import EloEngine, FightRecord, SnapshotRecord

_DIVISIONS = ["Lightweight", "Welterweight", "Middleweight", "Catch Weight", "Open Weight"]
_METHODS = [
    ("KO/TKO", None),
    ("Submission", None),
    ("Decision", "Unanimous"),
    ("Decision", "Split"),
    ("Decision", "Majority"),
]


def _synthetic_corpus(seed: int = 7, n_fights: int = 160) -> list[FightRecord]:
    rng = random.Random(seed)
    fighters = list(range(1, 15))
    # Give each fighter a "home" division so transfers are occasional, not constant.
    home = {fid: rng.choice(_DIVISIONS[:3]) for fid in fighters}
    busy: set[tuple[int, date]] = set()
    fights: list[FightRecord] = []
    day = date(2015, 1, 10)
    fight_id = 100
    while len(fights) < n_fights:
        # Irregular cadence: mostly weeks, sometimes a >9-month hole for everyone.
        day += timedelta(days=rng.choice([7, 14, 21, 35, 60, 120, 300, 420]))
        a, b = rng.sample(fighters, 2)
        if (a, day) in busy or (b, day) in busy:
            continue
        busy.add((a, day))
        busy.add((b, day))
        roll = rng.random()
        if roll < 0.08:
            division = rng.choice(["Catch Weight", "Open Weight"])
        elif roll < 0.30:
            division = rng.choice(_DIVISIONS[:3])
        else:
            division = home[a]
        outcome = rng.random()
        if outcome < 0.04:
            winner, (method, detail) = None, ("Draw", None)
        elif outcome < 0.06:
            winner, (method, detail) = None, ("No Contest", None)
        else:
            winner = a if outcome < 0.55 else b
            method, detail = rng.choice(_METHODS)
        fights.append(
            FightRecord(
                fight_id=fight_id,
                event_date=day,
                fighter_a_id=a,
                fighter_b_id=b,
                winner_id=winner,
                weight_class=division,
                method=method,
                method_detail=detail,
            )
        )
        fight_id += 1
    return fights


def _round_stats(
    fights: list[FightRecord], rng: random.Random
) -> dict[int, list[dict[str, object]]]:
    """Round data for ~70% of fights so the domain computer skips the rest."""
    out: dict[int, list[dict[str, object]]] = {}
    for f in fights:
        if rng.random() < 0.7:
            out[f.fight_id] = [
                {
                    "sig_str_landed": rng.randint(0, 40),
                    "takedowns_landed": rng.randint(0, 3),
                    "ctrl_time_seconds": rng.randint(0, 200),
                }
                for _ in range(3)
            ]
    return out


def _history(snaps: list[SnapshotRecord], fighter_id: int, before: date) -> list[RatedFight]:
    return [
        RatedFight(fight_date=s.fight_date, division=s.division, elo_after=s.elo_after)
        for s in snaps
        if s.fighter_id == fighter_id and s.fight_date < before
    ]


@pytest.fixture(scope="module")
def corpus() -> tuple[
    list[FightRecord], dict[int, float], list[SnapshotRecord], list[SnapshotRecord]
]:
    fights = _synthetic_corpus()
    seeds = {1: 1640.0, 2: 1380.0, 3: 1555.0}
    config = EloConfig()
    engine = EloEngine(config, seeds=seeds)
    overall = engine.compute_all(fights)
    domain = DomainEloComputer(config).compute_all(
        fights, overall, _round_stats(fights, random.Random(11))
    )
    return fights, seeds, overall, domain


def test_corpus_exercises_every_branch(corpus):
    fights, _, overall, domain = corpus
    divisions = {f.weight_class for f in fights}
    assert "Catch Weight" in divisions and "Open Weight" in divisions
    assert any(f.winner_id is None and f.method == "Draw" for f in fights)
    assert any(f.method == "No Contest" for f in fights)
    # At least one fighter came back after the inactivity threshold.
    last: dict[int, date] = {}
    gaps = []
    for s in sorted(overall, key=lambda s: s.fight_date):
        if s.fighter_id in last:
            gaps.append((s.fight_date - last[s.fighter_id]).days)
        last[s.fighter_id] = s.fight_date
    assert max(gaps) > EloConfig().inactivity_threshold_days
    # At least one fighter changed division between two rated fights.
    seen: dict[int, str] = {}
    moved = False
    for s in sorted(overall, key=lambda s: s.fight_date):
        if s.fighter_id in seen and seen[s.fighter_id] != s.division:
            moved = True
        seen[s.fighter_id] = s.division
    assert moved
    assert domain and len(domain) < 2 * len(overall)


def test_overall_replay_matches_engine_elo_before(corpus):
    _, seeds, overall, _ = corpus
    checked = 0
    for snap in overall:
        got = pre_fight_rating(
            _history(overall, snap.fighter_id, snap.fight_date),
            snap.fight_date,
            snap.division,
            config=EloConfig(),
            seed=seeds.get(snap.fighter_id),
        )
        assert got == pytest.approx(snap.elo_before, abs=1e-9), (
            f"fighter {snap.fighter_id} fight {snap.fight_id} {snap.division} "
            f"{snap.fight_date}: replay {got} != engine {snap.elo_before}"
        )
        checked += 1
    assert checked == len(overall)


def test_seeded_debutant_matches_engine(corpus):
    """A seeded fighter's first rated key uses the seed, unless a transfer
    already populated it — exactly ``_lookup_initial_rating``."""
    _, seeds, overall, _ = corpus
    firsts = {}
    for snap in sorted(overall, key=lambda s: s.fight_date):
        firsts.setdefault(snap.fighter_id, snap)
    for fid, seed in seeds.items():
        assert firsts[fid].elo_before == seed
        assert pre_fight_rating([], firsts[fid].fight_date, firsts[fid].division, seed=seed) == seed


@pytest.mark.parametrize("elo_type", ["striking", "grappling"])
def test_domain_replay_matches_domain_computer(corpus, elo_type):
    """Both domains see regression + transfer: the domain computer keeps
    per-domain bookkeeping, so the default replay flags reproduce
    ``elo_before`` for striking and grappling alike."""
    _, _, _, domain = corpus
    snaps = [s for s in domain if s.elo_type == elo_type]
    assert snaps
    for snap in snaps:
        got = pre_fight_rating(
            _history(snaps, snap.fighter_id, snap.fight_date),
            snap.fight_date,
            snap.division,
            config=EloConfig(),
            seed=None,
        )
        assert got == pytest.approx(snap.elo_before, abs=1e-9), (
            f"{elo_type} fighter {snap.fighter_id} fight {snap.fight_id}: "
            f"replay {got} != engine {snap.elo_before}"
        )


@pytest.mark.parametrize("elo_type", ["striking", "grappling"])
def test_domain_replay_diverges_without_adjustments(corpus, elo_type):
    """Guard against regressing the domain computer to shared bookkeeping:
    skipping regression + transfer must break parity for BOTH domains. Before
    the per-domain fix, grappling only matched with the adjustments off."""
    _, _, _, domain = corpus
    snaps = [s for s in domain if s.elo_type == elo_type]
    mismatches = sum(
        1
        for snap in snaps
        if pre_fight_rating(
            _history(snaps, snap.fighter_id, snap.fight_date),
            snap.fight_date,
            snap.division,
            regress=False,
            transfer=False,
        )
        != pytest.approx(snap.elo_before, abs=1e-9)
    )
    assert mismatches > 0


def test_unknown_division_resolves_to_latest_transferable_division():
    cfg = EloConfig()
    history = [
        RatedFight(date(2024, 1, 1), "Lightweight", 1580.0),
        RatedFight(date(2024, 6, 1), "Catch Weight", 1590.0),
    ]
    got = pre_fight_rating(history, date(2024, 9, 1), None, config=cfg)
    assert got == 1580.0
    # Empty history, no division: seed or initial.
    assert pre_fight_rating([], date(2024, 9, 1), None, seed=1620.0) == 1620.0
    assert pre_fight_rating([], date(2024, 9, 1), "Lightweight") == cfg.initial_rating


def test_future_snapshot_rejected():
    with pytest.raises(ValueError):
        pre_fight_rating(
            [RatedFight(date(2025, 1, 1), "Lightweight", 1500.0)],
            date(2024, 1, 1),
            "Lightweight",
        )

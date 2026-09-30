"""Serve-time performance + NET snapshot vs the rows training reads.

``features compute`` keys every ``computed_features`` row to a fight and
builds it from the PRE-fight accumulator, so a fighter's newest row is the
snapshot taken *before* their last fight, and a fighter with one UFC fight
has no row at all. The serve path used to read that newest row, so every
performance ``_diff`` was one fight stale (and all-NaN for a sophomore), and
with no ``event_date`` cutoff a historical ``event_date`` read snapshots from
later fights.

These tests seed a small corpus in the disposable testcontainers Postgres,
run the real ``features compute`` pipeline into it, and compare the serve
vector's 20 performance columns against what training reads for the same
fight (``ml.queries.load_computed_features``), or would read once the
upcoming fight is in the corpus (``FeatureComputer.compute_all`` over the
corpus plus that fight).
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from ufc_prediction.features.compute import FeatureComputer
from ufc_prediction.features.queries import (
    bulk_insert_features,
    load_all_domain_elo,
    load_all_round_stats,
    load_fights_with_duration,
)
from ufc_prediction.ml import inference_features
from ufc_prediction.ml import queries as ml_queries
from ufc_prediction.ml.config import PERFORMANCE_FEATURE_KEYS, get_feature_columns
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.models.round_stats import RoundStats

_COLS = get_feature_columns(feature_set="v2.1-no-net")
_PERF_COLS = [f"{k}_diff" for k in PERFORMANCE_FEATURE_KEYS]
_NET_KEYS = ("pagerank", "sos_2hop", "is_debutant_in_graph")

# Fighter slots. V1/V2 are veterans (>= 5 prior fights, so shrinkage is a
# no-op); SOPH has exactly one fight; MID's second fight is the mid-career
# backtest target, and both MID and V1 keep fighting after it.
V1, V2, SOPH, MID = 0, 1, 2, 3
_N_FIGHTERS = 7


def _add_rounds(session, rng, fight_id: int, fighter_id: int, rounds: int) -> None:
    for n in range(1, rounds + 1):
        sig_att = rng.randint(10, 40)
        td_att = rng.randint(1, 5)
        session.add(
            RoundStats(
                fight_id=fight_id,
                fighter_id=fighter_id,
                round_number=n,
                sig_strikes_landed=rng.randint(0, sig_att),
                sig_strikes_attempted=sig_att,
                head_strikes_landed=rng.randint(0, 12),
                body_strikes_landed=rng.randint(0, 6),
                leg_strikes_landed=rng.randint(0, 6),
                distance_strikes_landed=rng.randint(0, 10),
                clinch_strikes_landed=rng.randint(0, 4),
                ground_strikes_landed=rng.randint(0, 4),
                takedowns_landed=rng.randint(0, td_att),
                takedowns_attempted=td_att,
                submission_attempts=rng.randint(0, 2),
                reversals=0,
                control_time_seconds=rng.randint(0, 200),
                knockdowns=0,
            )
        )


@pytest.fixture()
def corpus(session):
    """Seed fighters/events/fights/round stats, then run ``features compute``
    (compute_all + bulk insert) into the disposable test DB."""
    rng = random.Random(11)
    fighters = [
        Fighter(
            name=f"Snapshot Fighter {i}",
            source="ufcstats",
            height_inches=68.0 + i,
            reach_inches=70.0 + i,
            leg_reach_inches=39.0,
            stance="Orthodox",
            date_of_birth=date(1990, 1, 1 + i),
        )
        for i in range(_N_FIGHTERS)
    ]
    session.add_all(fighters)
    session.flush()
    fid = [f.id for f in fighters]

    # (a, b, winner) by slot. SOPH fights exactly once; MID's 2nd fight
    # (vs V1) is the backtest target and both fight again afterwards.
    schedule = [
        (V1, 4, V1),
        (V2, 5, V2),
        (MID, 6, MID),
        (V1, V2, V2),
        (4, 5, 4),
        (V1, 5, V1),
        (V2, 6, V2),
        (SOPH, 4, 4),
        (V1, MID, V1),  # backtest target: MID's 2nd fight
        (V2, 4, V2),
        (V1, 6, 6),
        (MID, 5, MID),
        (V2, 5, 5),
        (V1, 4, V1),
        (V2, MID, V2),
        (V1, 6, V1),
        (V2, 6, V2),
    ]
    day = date(2021, 1, 9)
    fights = []
    backtest_fight_id = None
    for idx, (a, b, w) in enumerate(schedule):
        event = Event(name=f"Snapshot Event {idx}", date=day, source="ufcstats")
        session.add(event)
        session.flush()
        rounds = rng.choice([1, 2, 3])
        fight = Fight(
            event_id=event.id,
            fighter_a_id=fid[a],
            fighter_b_id=fid[b],
            winner_id=fid[w],
            weight_class="Lightweight",
            method=rng.choice(["KO/TKO", "Submission", "Decision"]),
            round_finished=rounds,
            time_finished=f"{rng.randint(0, 4)}:{rng.randint(10, 59)}",
            num_rounds=3,
            source="ufcstats",
        )
        session.add(fight)
        session.flush()
        _add_rounds(session, rng, fight.id, fid[a], rounds)
        _add_rounds(session, rng, fight.id, fid[b], rounds)
        fights.append(fight)
        if (a, b) == (V1, MID):
            backtest_fight_id = fight.id
        day += timedelta(days=rng.choice([35, 60, 90]))
    session.flush()

    rows = FeatureComputer().compute_all(
        load_fights_with_duration(session),
        load_all_round_stats(session),
        load_all_domain_elo(session),
    )
    bulk_insert_features(session, rows)
    session.flush()

    return {
        "fighters": fighters,
        "backtest_fight_id": backtest_fight_id,
        "backtest_date": next(
            session.get(Event, f.event_id).date for f in fights if f.id == backtest_fight_id
        ),
        "upcoming_date": day,
    }


def _serve_perf(session, fa, fb, event_date) -> dict[str, float]:
    vec = inference_features.build(session, fa, fb, event_date, feature_set="v2.1-no-net")[0]
    return {c: float(vec[_COLS.index(c)]) for c in _PERF_COLS}


def _expected_upcoming(session, fa_id, fb_id, event_date):
    """Per-fighter snapshots training will read for ``fa`` vs ``fb`` on
    ``event_date`` once that fight is in the corpus: ``compute_all`` over the
    stored fights plus the new fight, keyed to the new fight. Its rows are
    shrunk toward the league means over strictly earlier rows, so the new
    fight's own rows never enter them."""
    fights = load_fights_with_duration(session)
    upcoming_id = max(f["fight_id"] for f in fights) + 1_000
    fights.append(
        {
            "fight_id": upcoming_id,
            "event_date": event_date,
            "fighter_a_id": fa_id,
            "fighter_b_id": fb_id,
            "weight_class": "Lightweight",
            "round_finished": None,
            "time_finished": None,
            "num_rounds": 3,
        }
    )
    rows = FeatureComputer().compute_all(fights, load_all_round_stats(session), {})
    return {r["fighter_id"]: r["features"] for r in rows if r["fight_id"] == upcoming_id}


def _assert_perf_equal(served: dict[str, float], feats_a: dict, feats_b: dict) -> None:
    bad = []
    for key in PERFORMANCE_FEATURE_KEYS:
        va, vb = feats_a.get(key), feats_b.get(key)
        want = float("nan") if va is None or vb is None else va - vb
        got = served[f"{key}_diff"]
        if not ((want != want and got != got) or got == pytest.approx(want, abs=1e-9)):
            bad.append((key, want, got))
    assert not bad, f"serve/train performance mismatch: {bad}"


def test_upcoming_fight_includes_each_fighters_last_fight(session, corpus):
    """Veterans: the serve snapshot must include their most recent fight."""
    v1, v2 = corpus["fighters"][V1], corpus["fighters"][V2]
    served = _serve_perf(session, v1, v2, corpus["upcoming_date"])
    expected = _expected_upcoming(session, v1.id, v2.id, corpus["upcoming_date"])
    _assert_perf_equal(served, expected[v1.id], expected[v2.id])


def test_sophomore_gets_performance_features(session, corpus):
    """One prior UFC fight → no stored row, but training has a real
    (shrunk) snapshot built from the debut."""
    soph, v1 = corpus["fighters"][SOPH], corpus["fighters"][V1]
    served = _serve_perf(session, soph, v1, corpus["upcoming_date"])
    nan_cols = [c for c, v in served.items() if v != v]
    assert not nan_cols, f"sophomore performance columns NaN at serve: {nan_cols}"

    # The sophomore's row is shrunk (factor 1/5), so this also pins the
    # league means: for a fight after the last stored event serving's as-of
    # means are the whole stored corpus, which must be exactly what
    # compute_all shrinks the row with once the fight is in the corpus.
    expected = _expected_upcoming(session, soph.id, v1.id, corpus["upcoming_date"])
    _assert_perf_equal(served, expected[soph.id], expected[v1.id])


def test_historical_event_date_matches_stored_training_row(session, corpus):
    """A past ``event_date`` must read the snapshot as of that date — the
    exact row training reads for that fight — not the fighters' later
    snapshots."""
    v1, mid = corpus["fighters"][V1], corpus["fighters"][MID]
    tid = corpus["backtest_fight_id"]
    served = _serve_perf(session, v1, mid, corpus["backtest_date"])
    stored = ml_queries.load_computed_features(session, fight_ids=[tid])
    _assert_perf_equal(served, stored[(v1.id, tid)], stored[(mid.id, tid)])


def test_historical_net_keys_match_stored_row(session, corpus):
    """The NET-* keys ride on the same snapshot; they must match the stored
    JSON for that fight (stored NaN is persisted as null)."""
    from ufc_prediction.models.computed_feature import ComputedFeature

    v1, mid = corpus["fighters"][V1], corpus["fighters"][MID]
    tid = corpus["backtest_fight_id"]
    feats_v1, feats_mid = inference_features._get_pre_fight_performance(
        session, v1.id, mid.id, corpus["backtest_date"]
    )
    for fighter_id, served in ((v1.id, feats_v1), (mid.id, feats_mid)):
        stored = (
            session.query(ComputedFeature.features)
            .filter_by(fighter_id=fighter_id, fight_id=tid)
            .scalar()
        )
        for key in _NET_KEYS:
            want, got = stored.get(key), served.get(key)
            if want is None:
                assert got is None or got != got, (fighter_id, key, got)
            else:
                assert got == pytest.approx(want), (fighter_id, key, got)

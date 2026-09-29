"""Serve-side career replay must equal the training assembler, column for column.

Before this replay existed, 37 of the 72 base columns (career, context,
pace, layoff, rolling windows, rematch, pre-UFC) were NaN at predict time
while never NaN in training. The test builds one synthetic corpus, assembles
the training row for its final fight with ``FeatureMatrixAssembler``, then
builds the serve vector for the same matchup from the same substrate and
compares every column with NaN-equality.
"""

from __future__ import annotations

import hashlib
import random
from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

from ufc_prediction.ml import inference_features
from ufc_prediction.ml.config import PERFORMANCE_FEATURE_KEYS, get_feature_columns
from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler

_COLS = get_feature_columns(feature_set="v2.1-no-net")
_DIVISIONS = ["Lightweight", "Welterweight", "Middleweight"]


def _no_swap_id(start: int) -> int:
    """First id >= start that the assembler's md5 coin flip does NOT swap."""
    fid = start
    while int(hashlib.md5(str(fid).encode()).hexdigest(), 16) % 2 == 0:
        fid += 1
    return fid


def _corpus(seed: int = 3):
    rng = random.Random(seed)
    fighters = list(range(1, 9))
    records = []
    day = date(2019, 3, 1)
    fight_id = 500
    # Force a prior meeting of 1 vs 2 (rematch) and a long layoff for 2.
    scripted = [(1, 2, 1), (3, 4, 4), (1, 5, 1), (2, 6, 6), (7, 8, 7), (1, 3, 3)]
    for a, b, w in scripted:
        records.append(
            {
                "fight_id": fight_id,
                "event_date": day,
                "fighter_a_id": a,
                "fighter_b_id": b,
                "winner_id": w,
                "weight_class": rng.choice(_DIVISIONS),
                "method": rng.choice(["KO/TKO", "Submission", "Decision"]),
                "is_title_fight": False,
                "num_rounds": 3,
            }
        )
        fight_id += 1
        day += timedelta(days=rng.choice([30, 45, 90, 400]))
    for _ in range(24):
        a, b = rng.sample(fighters, 2)
        records.append(
            {
                "fight_id": fight_id,
                "event_date": day,
                "fighter_a_id": a,
                "fighter_b_id": b,
                "winner_id": rng.choice([a, b]),
                "weight_class": rng.choice(_DIVISIONS),
                "method": rng.choice(["KO/TKO", "Submission", "Decision", "TKO", "SUB"]),
                "is_title_fight": rng.random() < 0.1,
                "num_rounds": rng.choice([3, 5]),
            }
        )
        fight_id += 1
        day += timedelta(days=rng.choice([14, 30, 45, 90]))

    target_id = _no_swap_id(fight_id)
    target_date = day + timedelta(days=50)
    target = {
        "fight_id": target_id,
        "event_date": target_date,
        "fighter_a_id": 1,
        "fighter_b_id": 2,
        "winner_id": 1,
        "weight_class": "Welterweight",
        "method": "Decision",
        "is_title_fight": True,
        "num_rounds": 5,
    }
    all_records = [*records, target]

    elo_features = {}
    computed = {}
    round_stats = {}
    for r in all_records:
        for fid in (r["fighter_a_id"], r["fighter_b_id"]):
            key = (fid, r["fight_id"])
            elo_features[key] = {
                "elo_overall": 1500.0 + rng.uniform(-120, 120),
                "elo_striking": 1500.0 + rng.uniform(-120, 120),
                "elo_grappling": 1500.0 + rng.uniform(-120, 120),
            }
            computed[key] = {
                k: (None if rng.random() < 0.1 else round(rng.uniform(0, 5), 3))
                for k in PERFORMANCE_FEATURE_KEYS
            }
            if rng.random() < 0.75:
                round_stats[key] = [
                    {
                        "round_number": n,
                        "sig_str_landed": rng.randint(0, 30),
                        "td_landed": rng.randint(0, 3),
                    }
                    for n in range(1, 4)
                ]
    physicals = {
        fid: {
            "height_inches": 66.0 + fid,
            "reach_inches": 68.0 + fid,
            "leg_reach_inches": 38.0 + fid * 0.5,
            "stance": "Southpaw" if fid % 3 == 0 else "Orthodox",
            "date_of_birth": date(1988 + fid % 6, 5, 1 + fid),
        }
        for fid in fighters
    }
    medians = {
        d: {"height_inches": 70.0, "reach_inches": 72.0, "leg_reach_inches": 40.0}
        for d in _DIVISIONS
    }
    pre_ufc = {1: {"win_pct": 0.8}, 2: {"win_pct": 0.6}, 3: {"win_pct": 0.5}}
    return all_records, target, elo_features, computed, round_stats, physicals, medians, pre_ufc


def _stub_fighter(fid: int, physicals: dict) -> MagicMock:
    p = physicals[fid]
    return MagicMock(
        id=fid,
        name=f"Fighter-{fid}",
        height_inches=p["height_inches"],
        reach_inches=p["reach_inches"],
        leg_reach_inches=p["leg_reach_inches"],
        stance=p["stance"],
        date_of_birth=p["date_of_birth"],
    )


def _serve_inputs(all_records, target, elo_features, computed, round_stats, pre_ufc):
    """What ``_load_career_inputs`` would return for the target matchup."""
    a, b, d = target["fighter_a_id"], target["fighter_b_id"], target["event_date"]
    prior = [
        r
        for r in all_records
        if r["event_date"] < d
        and (
            a in (r["fighter_a_id"], r["fighter_b_id"])
            or b in (r["fighter_a_id"], r["fighter_b_id"])
        )
    ]
    ids = {r["fight_id"] for r in prior}
    return inference_features.CareerInputs(
        fight_records=prior,
        elo_features={k: v for k, v in elo_features.items() if k[1] in ids},
        computed_features={k: v for k, v in computed.items() if k[1] in ids},
        round_stats={k: v for k, v in round_stats.items() if k[1] in ids},
        pre_ufc_records={fid: pre_ufc[fid] for fid in (a, b) if fid in pre_ufc},
    )


@pytest.fixture
def parity_setup(monkeypatch):
    all_records, target, elo_features, computed, round_stats, physicals, medians, pre_ufc = (
        _corpus()
    )
    X, _y, _dates = FeatureMatrixAssembler().assemble(
        all_records,
        elo_features,
        computed,
        physicals,
        medians,
        round_stats=round_stats,
        pre_ufc_records=pre_ufc,
        feature_set="v2.1-no-net",
    )
    train_row = X[-1]
    tid = target["fight_id"]

    monkeypatch.setattr(
        inference_features,
        "_load_career_inputs",
        lambda s, fa, fb, d: _serve_inputs(
            all_records, target, elo_features, computed, round_stats, pre_ufc
        ),
    )
    monkeypatch.setattr(
        inference_features,
        "_get_latest_elo",
        lambda s, fid, et, *a, **kw: elo_features[(fid, tid)][f"elo_{et}"],
    )
    # The performance block is stubbed with the target fight's own rows;
    # its serve/train parity against real FeatureComputer rows is covered by
    # tests/integration/test_serve_performance_snapshot.py.
    monkeypatch.setattr(
        inference_features,
        "_get_pre_fight_performance",
        lambda s, fa, fb, d: (dict(computed[(fa, tid)]), dict(computed[(fb, tid)])),
    )
    monkeypatch.setattr(inference_features, "_get_cached_odds", lambda *a: (None, None))
    return train_row, target, physicals


def test_corpus_has_history_for_both_fighters(parity_setup):
    train_row, _, _ = parity_setup
    non_nan = {c for c, v in zip(_COLS, train_row, strict=True) if v == v}
    for col in (
        "ufc_fight_count_diff",
        "days_since_last_fight_diff",
        "career_win_pct_diff",
        "pace_decay_strikes_diff",
        "sig_str_per_min_last3_diff",
        "log_days_since_last_fight_diff",
        "pre_ufc_win_pct_diff",
    ):
        assert col in non_nan, f"training row has NaN {col}; fixture too thin"
    assert train_row[_COLS.index("is_rematch")] == 1.0
    assert train_row[_COLS.index("first_fight_winner_diff")] == 1.0
    assert train_row[_COLS.index("num_rounds")] == 5.0
    assert train_row[_COLS.index("is_title_fight")] == 1.0
    assert train_row[_COLS.index("weight_class_ordinal")] == 6.0


def test_serve_vector_equals_training_row(parity_setup):
    train_row, target, physicals = parity_setup
    vec = inference_features.build(
        MagicMock(),
        _stub_fighter(1, physicals),
        _stub_fighter(2, physicals),
        target["event_date"],
        feature_set="v2.1-no-net",
        weight_class=target["weight_class"],
        num_rounds=target["num_rounds"],
        is_title_fight=target["is_title_fight"],
    )[0]
    assert vec.shape == train_row.shape
    bad = [
        (c, t, s)
        for c, t, s in zip(_COLS, train_row, vec, strict=True)
        if not ((t != t and s != s) or t == pytest.approx(s, abs=1e-12))
    ]
    assert not bad, f"train/serve mismatch: {bad}"


def test_previously_dead_columns_are_now_populated(parity_setup):
    """The 37 columns the old serve path never wrote."""
    _, target, physicals = parity_setup
    vec = inference_features.build(
        MagicMock(),
        _stub_fighter(1, physicals),
        _stub_fighter(2, physicals),
        target["event_date"],
        feature_set="v2.1-no-net",
        weight_class=target["weight_class"],
        num_rounds=target["num_rounds"],
        is_title_fight=target["is_title_fight"],
    )[0]
    for col in (
        "win_streak_diff",
        "ufc_fight_count_diff",
        "days_since_last_fight_diff",
        "is_title_fight",
        "num_rounds",
        "weight_class_ordinal",
        "pace_decay_strikes_diff",
        "is_comeback_diff",
        "td_rate_last3_diff",
        "is_rematch",
        "first_fight_winner_diff",
        "pre_ufc_win_pct_diff",
    ):
        assert not np.isnan(vec[_COLS.index(col)]), col


def test_weight_class_defaults_to_fighter_a_latest_division(parity_setup):
    """No weight_class passed → A's most recent division; the Elo replay
    receives the same division."""
    _, target, physicals = parity_setup
    seen = {}

    def spy(session, fid, et, as_of=None, division=None):
        seen[(fid, et)] = (as_of, division)
        return 1500.0

    inference_features._get_latest_elo = spy  # fixture already patched; override
    vec = inference_features.build(
        MagicMock(),
        _stub_fighter(1, physicals),
        _stub_fighter(2, physicals),
        target["event_date"],
        feature_set="v2.1-no-net",
    )[0]
    assert all(v[0] == target["event_date"] for v in seen.values())
    divisions = {v[1] for v in seen.values()}
    assert len(divisions) == 1
    division = divisions.pop()
    assert division in _DIVISIONS
    assert vec[_COLS.index("weight_class_ordinal")] == float(
        inference_features._WEIGHT_CLASS_ORDINAL[division]
    )
    assert vec[_COLS.index("num_rounds")] == 3.0
    assert vec[_COLS.index("is_title_fight")] == 0.0


def test_two_debutants_match_training_semantics(monkeypatch):
    monkeypatch.setattr(
        inference_features, "_load_career_inputs", lambda *a: inference_features.CareerInputs()
    )
    monkeypatch.setattr(inference_features, "_get_latest_elo", lambda *a, **kw: 1500.0)
    monkeypatch.setattr(inference_features, "_get_pre_fight_performance", lambda *a: ({}, {}))
    monkeypatch.setattr(inference_features, "_get_cached_odds", lambda *a: (None, None))
    fa = MagicMock(
        id=1,
        height_inches=70.0,
        reach_inches=72.0,
        leg_reach_inches=40.0,
        stance="Orthodox",
        date_of_birth=date(1990, 1, 1),
    )
    fb = MagicMock(
        id=2,
        height_inches=70.0,
        reach_inches=72.0,
        leg_reach_inches=40.0,
        stance="Orthodox",
        date_of_birth=date(1990, 1, 1),
    )
    vec = inference_features.build(
        MagicMock(), fa, fb, date(2026, 6, 14), feature_set="v2.1-no-net"
    )[0]
    assert vec[_COLS.index("is_debut_diff")] == 0.0
    assert vec[_COLS.index("ufc_fight_count_diff")] == 0.0
    assert vec[_COLS.index("career_win_pct_diff")] == 0.0
    assert np.isnan(vec[_COLS.index("days_since_last_fight_diff")])
    assert np.isnan(vec[_COLS.index("fights_per_year_diff")])
    assert vec[_COLS.index("weight_class_ordinal")] == 5.0
    assert vec[_COLS.index("is_rematch")] == 0.0


@pytest.mark.parametrize("division", [*inference_features._WEIGHT_CLASS_ORDINAL, "Catch Weight"])
def test_weight_class_ordinal_matches_assembler(division):
    """The ordinal map is duplicated from the frozen assembler; pin it."""
    record = {
        "fight_id": _no_swap_id(1),
        "event_date": date(2024, 1, 1),
        "fighter_a_id": 1,
        "fighter_b_id": 2,
        "winner_id": 1,
        "weight_class": division,
        "method": "Decision",
        "is_title_fight": False,
        "num_rounds": 3,
    }
    X, _, _ = FeatureMatrixAssembler().assemble([record], {}, {}, {}, {}, feature_set="v2.1-no-net")
    expected = inference_features._WEIGHT_CLASS_ORDINAL.get(
        division, inference_features._WEIGHT_CLASS_ORDINAL_DEFAULT
    )
    assert X[0, _COLS.index("weight_class_ordinal")] == float(expected)

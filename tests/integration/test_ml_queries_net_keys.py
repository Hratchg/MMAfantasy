"""``ml.queries.load_computed_features`` carries the NET-* keys to training.

``features compute`` persists ``pagerank`` / ``sos_2hop`` /
``is_debutant_in_graph`` in each computed-feature JSON blob (JSON ``null`` for
the NaN debutant case), and ``FeatureMatrixAssembler`` reads them for the
75-col ``v1.0`` feature set. The loader used to project only the 20
performance keys, so every NET column came out NaN regardless of the stored
values. These tests pin the loader contract and prove the 72-col
``v2.1-no-net`` matrix is unaffected by the extra keys.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from ufc_prediction.ml import queries
from ufc_prediction.ml.config import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_NO_NET,
    PERFORMANCE_FEATURE_KEYS,
)
from ufc_prediction.ml.feature_matrix import FeatureMatrixAssembler
from ufc_prediction.models.computed_feature import ComputedFeature
from ufc_prediction.models.event import Event
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter

pytestmark = pytest.mark.integration

_NET_KEYS = ("pagerank", "sos_2hop", "is_debutant_in_graph")


def _perf(value: float) -> dict[str, float]:
    return {k: value for k in PERFORMANCE_FEATURE_KEYS}


@pytest.fixture
def net_corpus(session):
    fa = Fighter(name="Net Loader A", source="ufcstats")
    fb = Fighter(name="Net Loader B", source="ufcstats")
    fc = Fighter(name="Net Loader C", source="ufcstats")
    session.add_all([fa, fb, fc])
    session.flush()
    event = Event(name="UFC Net Loader", date=date(2023, 5, 6), source="ufcstats")
    session.add(event)
    session.flush()
    both = Fight(
        event_id=event.id,
        fighter_a_id=fa.id,
        fighter_b_id=fb.id,
        winner_id=fa.id,
        weight_class="Lightweight",
        method="Decision",
        source="ufcstats",
    )
    debut = Fight(
        event_id=event.id,
        fighter_a_id=fa.id,
        fighter_b_id=fc.id,
        winner_id=fc.id,
        weight_class="Lightweight",
        method="Decision",
        source="ufcstats",
    )
    session.add_all([both, debut])
    session.flush()
    rows = [
        # Both sides in the graph: finite NET values.
        (
            fa,
            both,
            {**_perf(1.0), "pagerank": 0.004, "sos_2hop": 0.003, "is_debutant_in_graph": 0.0},
        ),
        (
            fb,
            both,
            {**_perf(2.0), "pagerank": 0.001, "sos_2hop": 0.002, "is_debutant_in_graph": 0.0},
        ),
        (
            fa,
            debut,
            {**_perf(1.0), "pagerank": 0.004, "sos_2hop": 0.003, "is_debutant_in_graph": 0.0},
        ),
        # Graph debutant: persisted as JSON null (features/queries.py NaN->null).
        (
            fc,
            debut,
            {**_perf(3.0), "pagerank": None, "sos_2hop": None, "is_debutant_in_graph": 1.0},
        ),
    ]
    session.add_all(
        [
            ComputedFeature(
                fighter_id=fighter.id,
                fight_id=fight.id,
                as_of_date=date(2023, 5, 6),
                feature_set_version="test-net-keys",
                features=feats,
            )
            for fighter, fight, feats in rows
        ]
    )
    session.flush()
    return {"fighters": (fa, fb, fc), "fights": (both, debut)}


def test_loader_returns_net_keys(session, net_corpus):
    fa, fb, fc = net_corpus["fighters"]
    both, debut = net_corpus["fights"]
    loaded = queries.load_computed_features(session, fight_ids=[both.id, debut.id])

    assert loaded[(fa.id, both.id)]["pagerank"] == pytest.approx(0.004)
    assert loaded[(fb.id, both.id)]["sos_2hop"] == pytest.approx(0.002)
    assert loaded[(fb.id, both.id)]["is_debutant_in_graph"] == 0.0
    # JSON null (NaN debutant) comes back as NaN — never None, which the
    # assembler's NET block cannot subtract.
    debutant = loaded[(fc.id, debut.id)]
    assert math.isnan(debutant["pagerank"])
    assert math.isnan(debutant["sos_2hop"])
    assert debutant["is_debutant_in_graph"] == 1.0
    # Performance keys are unchanged.
    assert debutant["sig_str_per_minute"] == 3.0


def test_legacy_row_without_net_keys_yields_nan(session, net_corpus):
    fa, _fb, _fc = net_corpus["fighters"]
    both, _debut = net_corpus["fights"]
    session.query(ComputedFeature).filter_by(fighter_id=fa.id, fight_id=both.id).update(
        {"features": _perf(1.0)}
    )
    session.flush()
    loaded = queries.load_computed_features(session, fight_ids=[both.id])
    assert all(math.isnan(loaded[(fa.id, both.id)][k]) for k in _NET_KEYS)


def _assemble(session, computed, feature_set):
    records = [
        r for r in queries.load_fight_records(session) if r["fight_id"] in {f for _, f in computed}
    ]
    X, _y, _d = FeatureMatrixAssembler().assemble(
        records,
        queries.load_elo_features(session, fight_ids=[r["fight_id"] for r in records]),
        computed,
        queries.load_fighter_physicals(session),
        {},
        feature_set=feature_set,
    )
    return records, X


def test_net_columns_reach_the_75_col_matrix(session, net_corpus):
    both, debut = net_corpus["fights"]
    computed = queries.load_computed_features(session, fight_ids=[both.id, debut.id])
    records, X = _assemble(session, computed, "v1.0")
    assert X.shape[1] == len(FEATURE_COLUMNS) == 75

    row = {r["fight_id"]: i for i, r in enumerate(records)}[both.id]
    net = X[row, FEATURE_COLUMNS.index("pagerank_diff") :]
    assert np.isfinite(net).all(), net
    # The assembler's deterministic A/B swap only flips the sign.
    assert abs(net[0]) == pytest.approx(0.003)
    assert abs(net[1]) == pytest.approx(0.001)
    assert net[2] == 0.0

    debut_row = {r["fight_id"]: i for i, r in enumerate(records)}[debut.id]
    debut_net = X[debut_row, FEATURE_COLUMNS.index("pagerank_diff") :]
    assert np.isnan(debut_net[0]) and np.isnan(debut_net[1])
    assert abs(debut_net[2]) == 1.0


def test_72_col_matrix_unaffected_by_net_keys(session, net_corpus):
    both, debut = net_corpus["fights"]
    computed = queries.load_computed_features(session, fight_ids=[both.id, debut.id])
    perf_only = {
        key: {k: v for k, v in feats.items() if k in PERFORMANCE_FEATURE_KEYS}
        for key, feats in computed.items()
    }
    _r1, with_net = _assemble(session, computed, "v2.1-no-net")
    _r2, without_net = _assemble(session, perf_only, "v2.1-no-net")
    assert with_net.shape[1] == len(FEATURE_COLUMNS_NO_NET) == 72
    assert with_net.tobytes() == without_net.tobytes()

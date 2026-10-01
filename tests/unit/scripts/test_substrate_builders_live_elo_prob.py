"""Substrate builders use the real as-of-fight Elo probability in live mode.

``scripts/build_{net,ref,travel}_substrate_v261.py`` and
``scripts/build_canonical_substrate_v27.py`` filled col[1] (``elo_prob``) from
a seeded ``uniform(0.2, 0.8)`` RNG in BOTH source modes, so every live gate
substrate carried pure noise in a baseline column. Live mode must derive it
from the assembled row's own ``elo_overall_diff`` (the row's fighter A vs B,
after the assembler's deterministic A/B swap — the same orientation as the
row's outcome). Synthetic mode keeps its seeded RNG so the DB-free fixtures
stay byte-stable.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_canonical_substrate_v27 as canonical_builder
import build_net_substrate_v261 as net_builder
import build_ref_substrate_v261 as ref_builder
import build_travel_substrate_v261 as travel_builder
import compose_v25_travel

from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.engine import EloEngine
from ufc_prediction.ml.config import FEATURE_COLUMNS_V22
from ufc_prediction.ml.meta_features_v22 import elo_prob_from_v22_matrix

N_ROWS = 60
ELO_DIFF_IDX = FEATURE_COLUMNS_V22.index("elo_overall_diff")


def _ratings() -> tuple[np.ndarray, np.ndarray]:
    """Per-row pre-fight overall Elo for the row's fighter A and B, ordered so
    the A-minus-B diff is strictly increasing."""
    rng = np.random.default_rng(99)
    rating_b = rng.uniform(1300.0, 1700.0, size=N_ROWS)
    rating_a = rating_b + np.linspace(-450.0, 450.0, N_ROWS) + rng.uniform(0, 1e-3, N_ROWS)
    return rating_a, rating_b


def _fake_live_matrix() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(123)
    X = rng.standard_normal((N_ROWS, 92))
    rating_a, rating_b = _ratings()
    # Same arithmetic as FeatureMatrixAssembler: elo_a - elo_b.
    X[:, ELO_DIFF_IDX] = rating_a - rating_b
    y = (X[:, ELO_DIFF_IDX] > 0).astype(int)
    start = date(2025, 7, 1)
    dates = np.array([start + timedelta(days=5 * i) for i in range(N_ROWS)])
    records = [
        {
            "fight_id": 900_000 + i,
            "event_date": dates[i],
            "fighter_a_id": 2 * i,
            "fighter_b_id": 2 * i + 1,
            "venue_country": "USA",
        }
        for i in range(N_ROWS)
    ]
    return X, y, dates, records


def _expected_elo_prob() -> np.ndarray:
    """What the per-fight driver helpers compute from the two ratings."""
    engine = EloEngine(EloConfig())
    rating_a, rating_b = _ratings()
    return np.array(
        [
            engine.expected_win_probability(float(a), float(b))
            for a, b in zip(rating_a, rating_b, strict=True)
        ]
    )


@pytest.fixture
def fake_live(monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    X, y, dates, records = _fake_live_matrix()
    monkeypatch.setattr(
        compose_v25_travel,
        "_load_assembled_data_v25_travel",
        lambda *a, **k: (X.copy(), y.copy(), dates.copy(), [dict(r) for r in records]),
    )
    monkeypatch.setattr(net_builder, "_load_xgb_v2_netd_oof_map", lambda: {})
    monkeypatch.setattr(ref_builder, "_load_xgb_v2_refv2_oof_map", lambda: {})
    monkeypatch.setattr(canonical_builder, "_load_canonical_oof_map", lambda *a, **k: {})
    return X


def _build(builder: Any, source: str) -> np.ndarray:
    return np.asarray(builder.build_eval_matrix(source=source)[0])


BUILDERS = [
    pytest.param(net_builder, id="net"),
    pytest.param(ref_builder, id="ref"),
    pytest.param(travel_builder, id="travel"),
    pytest.param(canonical_builder, id="canonical_v27"),
]


def test_helper_matches_per_fight_elo_engine_bit_for_bit() -> None:
    X, _y, _d, _r = _fake_live_matrix()
    got = elo_prob_from_v22_matrix(X[:, :90])
    assert got.tobytes() == _expected_elo_prob().tobytes()


@pytest.mark.parametrize("builder", BUILDERS)
def test_live_mode_elo_prob_is_real(builder: Any, fake_live: np.ndarray) -> None:
    X_out = _build(builder, "live")
    np.testing.assert_array_equal(X_out[:, 1], _expected_elo_prob())
    # Orientation: a higher Elo diff for the row's fighter A means P(A) > 0.5.
    assert np.all(np.diff(X_out[:, 1]) > 0)


def test_travel_live_mode_uses_the_canonical_oof(
    monkeypatch: pytest.MonkeyPatch, fake_live: np.ndarray
) -> None:
    """S18: the travel builder also filled col[0] (``xgb_oof_prob``) from its
    seeded RNG in live mode. Live mode now takes the canonical Phase 26 OOF,
    the same source and fallback rule as ``build_canonical_substrate_v27``.
    Rows the OOF covers with a finite value use it. Every other row keeps the
    seeded draw, so the draw order and synthetic mode stay unchanged."""
    ids = [900_000 + i for i in range(N_ROWS)]
    oof = {ids[i]: 0.01 * i for i in range(0, N_ROWS, 2)}  # every other row covered
    oof[ids[4]] = float("nan")  # the canonical parquet carries NaN rows
    monkeypatch.setattr(canonical_builder, "_load_canonical_oof_map", lambda *a, **k: oof)

    X_out = _build(travel_builder, "live")

    seeded = np.random.default_rng(travel_builder.RANDOM_15PCT_SEED).uniform(
        0.05, 0.95, size=N_ROWS
    )
    expected = np.array(
        [
            oof[fid] if fid in oof and not np.isnan(oof[fid]) else seeded[i]
            for i, fid in enumerate(ids)
        ]
    )
    np.testing.assert_array_equal(X_out[:, 0], expected)


@pytest.mark.parametrize("builder", BUILDERS)
def test_synthetic_mode_keeps_seeded_rng(builder: Any, fake_live: np.ndarray) -> None:
    first = _build(builder, "synthetic")
    second = _build(builder, "synthetic")
    assert first[:, 1].tobytes() == second[:, 1].tobytes()
    assert np.all((first[:, 1] >= 0.2) & (first[:, 1] <= 0.8))

"""S18 / operator decision D5: meta ``elo_prob`` shares the assembled row's orientation.

``FeatureMatrixAssembler`` swaps fighter A/B on about half the fights (an
md5(fight_id) coin flip) and labels each row from the swapped A. On ufcstats
the record's ``fighter_a`` is always the winner, so the label is exactly
``1 - swap``. The per-fight ``_compute_elo_prob_for_fight`` helpers keyed
``elo_prob`` on the record's unswapped ``fighter_a``. On every swapped row
``elo_prob`` therefore pointed the opposite way from the row's
``elo_overall_diff``. The META-V22 Level-1 set carries both, so the
disagreement between them encodes the label. On the live substrate a logistic
fit on [elo_prob, elo_overall_diff, product] reached AUC 0.9993, against
0.6345 for elo_overall_diff alone.

Every meta driver must take ``elo_prob`` from the assembled row
(``meta_features_v22.elo_prob_from_v22_matrix``). Each test runs one driver's
live (non-dry-run) Level-1 path. The input corpus comes from the real
assembler, and in it the record's fighter A always wins. The test captures the
Level-1 rows the driver builds and checks two things:

- orientation: on every row, elo_prob > 0.5 <=> elo_overall_diff > 0, and
  elo_prob equals the Elo expectation of the row's own fighters' pre-fight
  ratings
- no label recovery: a logistic fit on [elo_prob, elo_overall_diff, product]
  separates the label no better than elo_overall_diff alone

The DB is faked so the run is hermetic. The heavy base-model steps (OOF
generation, the transient eval XGB) are stubbed; they do not touch elo_prob.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.engine import EloEngine
from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET, FEATURE_COLUMNS_V22
from ufc_prediction.ml.meta_features_v22 import (
    META_V22_FEATURE_COLUMNS,
    elo_prob_from_v22_matrix,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"

N_FIGHTS = 600
FIGHT_ID_BASE = 50_000
SUBSTRATE_START = date(2020, 1, 1)
SUBSTRATE_END = date(2025, 12, 31)
CUTOFF = date(2023, 1, 1)

ELO_DIFF_IDX = FEATURE_COLUMNS_V22.index("elo_overall_diff")
META_ELO_PROB_IDX = META_V22_FEATURE_COLUMNS.index("elo_prob")
META_ELO_DIFF_IDX = META_V22_FEATURE_COLUMNS.index("elo_overall_diff")
CLOSING_IDX_72 = FEATURE_COLUMNS_NO_NET.index("closing_prob_diff")

# In-sample AUC that a fit on [elo_prob, diff, product] may gain over diff
# alone when elo_prob is a monotone function of diff. That gain is fit noise
# only; it is at most 0.016 on the 100-row slices here. The leaky
# record-oriented elo_prob gains ~0.35 on this corpus.
AUC_SLACK = 0.05


class _StopRunError(Exception):
    """Raised by a spy once the driver's Level-1 rows have been captured."""


# ── Corpus: the real assembler, fighter A always wins ───────────────────────


@dataclass(frozen=True)
class _Corpus:
    records: list[dict[str, Any]]
    elo_features: dict[tuple[int, int], dict[str, float]]
    physicals: dict[int, dict[str, Any]]
    rating_a: np.ndarray  # record fighter_a pre-fight overall Elo
    rating_b: np.ndarray
    X_v22: np.ndarray
    X_72: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    label_by_diff: dict[float, int]
    oracle_by_diff: dict[float, float]


def _swap(fight_id: int) -> bool:
    # FeatureMatrixAssembler's A/B swap rule.
    return int(hashlib.md5(str(fight_id).encode()).hexdigest(), 16) % 2 == 0


@functools.lru_cache(maxsize=1)
def _corpus() -> _Corpus:
    from ufc_prediction.ml.feature_matrix import (
        FeatureMatrixAssembler,
        compute_division_medians,
    )

    rng = np.random.default_rng(7)
    span = (SUBSTRATE_END - SUBSTRATE_START).days
    records: list[dict[str, Any]] = []
    elo: dict[tuple[int, int], dict[str, float]] = {}
    # The winner carries a modest pre-fight edge, so elo_overall_diff alone
    # is mildly predictive (AUC ~0.63, close to the live substrate).
    rating_a = 1500.0 + rng.normal(40.0, 120.0, size=N_FIGHTS)
    rating_b = 1500.0 + rng.normal(0.0, 120.0, size=N_FIGHTS)
    for i in range(N_FIGHTS):
        fight_id = FIGHT_ID_BASE + i
        a_id, b_id = 2 * i + 1, 2 * i + 2
        records.append(
            {
                "fight_id": fight_id,
                "event_id": 70_000 + i // 10,
                "event_date": SUBSTRATE_START + timedelta(days=round(i * span / (N_FIGHTS - 1))),
                "fighter_a_id": a_id,
                "fighter_b_id": b_id,
                "winner_id": a_id,  # ufcstats: the record's fighter A always won
                "weight_class": "Lightweight",
                "referee_id": None,
                "method": "Decision - Unanimous",
            }
        )
        for fid, rating in ((a_id, rating_a[i]), (b_id, rating_b[i])):
            elo[(fid, fight_id)] = {
                "elo_overall": float(rating),
                "elo_striking": 1500.0,
                "elo_grappling": 1500.0,
            }
    physicals = {
        fid: {
            "height_inches": 70.0,
            "reach_inches": 72.0,
            "leg_reach_inches": 40.0,
            "stance": "Orthodox",
            "date_of_birth": date(1990, 1, 1),
        }
        for rec in records
        for fid in (rec["fighter_a_id"], rec["fighter_b_id"])
    }
    medians = compute_division_medians(physicals, records, CUTOFF)
    assembler = FeatureMatrixAssembler()
    X_v22, y, dates = assembler.assemble(records, elo, {}, physicals, medians, feature_set="v2.2")
    X_72, y_72, _ = assembler.assemble(
        records, elo, {}, physicals, medians, feature_set="v2.1-no-net"
    )
    assert X_v22.shape == (N_FIGHTS, 90) and X_72.shape == (N_FIGHTS, 72)
    assert np.array_equal(y, y_72)

    engine = EloEngine(EloConfig())
    diffs = X_v22[:, ELO_DIFF_IDX]
    label_by_diff: dict[float, int] = {}
    oracle_by_diff: dict[float, float] = {}
    for i in range(N_FIGHTS):
        # Row A is the record's fighter A exactly when the row is labelled a win.
        ra, rb = (rating_a[i], rating_b[i]) if y[i] == 1 else (rating_b[i], rating_a[i])
        label_by_diff[float(diffs[i])] = int(y[i])
        oracle_by_diff[float(diffs[i])] = engine.expected_win_probability(float(ra), float(rb))
    assert len(label_by_diff) == N_FIGHTS, "elo_overall_diff must identify the row"
    return _Corpus(
        records=records,
        elo_features=elo,
        physicals=physicals,
        rating_a=rating_a,
        rating_b=rating_b,
        X_v22=X_v22,
        X_72=X_72,
        y=np.asarray(y),
        dates=np.asarray(dates),
        label_by_diff=label_by_diff,
        oracle_by_diff=oracle_by_diff,
    )


def _assembled_v22() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    c = _corpus()
    return c.X_v22.copy(), c.y.copy(), c.dates.copy(), [dict(r) for r in c.records]


# ── Assertions ──────────────────────────────────────────────────────────────


def _fit_auc(columns: list[np.ndarray], labels: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Z = np.column_stack(columns)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(Z, labels)
    return float(roc_auc_score(labels, model.predict_proba(Z)[:, 1]))


def _assert_row_oriented(
    elo_prob: np.ndarray,
    elo_diff: np.ndarray,
    where: str,
    labels: np.ndarray | None = None,
) -> None:
    c = _corpus()
    elo_prob = np.asarray(elo_prob, dtype=float)
    elo_diff = np.asarray(elo_diff, dtype=float)
    assert len(elo_prob) == len(elo_diff) >= 50, f"{where}: too few rows captured"
    if labels is None:
        labels = np.array([c.label_by_diff[float(d)] for d in elo_diff])
    assert 0 < labels.mean() < 1, f"{where}: single-class rows"

    n_disagree = int(((elo_prob > 0.5) != (elo_diff > 0)).sum())
    assert n_disagree == 0, (
        f"{where}: elo_prob points against the row's elo_overall_diff on "
        f"{n_disagree}/{len(elo_prob)} rows"
    )
    expected = np.array([c.oracle_by_diff[float(d)] for d in elo_diff])
    np.testing.assert_allclose(elo_prob, expected, rtol=0, atol=1e-12, err_msg=where)

    auc_diff = _fit_auc([elo_diff], labels)
    auc_all = _fit_auc([elo_prob, elo_diff, elo_prob * elo_diff], labels)
    assert auc_all <= auc_diff + AUC_SLACK, (
        f"{where}: [elo_prob, diff, product] AUC {auc_all:.4f} vs diff-only {auc_diff:.4f}"
    )


# ── Fakes / spies ───────────────────────────────────────────────────────────


def _import_script(name: str) -> ModuleType:
    # Resolve at call time: other tests pop / replace script modules in
    # sys.modules, so a module-level import could hold a stale object that
    # the code under test no longer sees.
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    return importlib.import_module(name)


class _FakeSession:
    def close(self) -> None:
        pass


class _ConstEstimator:
    def fit(self, X: np.ndarray, y: np.ndarray) -> _ConstEstimator:
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])


class _FixedDate(date):
    @classmethod
    def today(cls) -> date:  # type: ignore[override]
        return SUBSTRATE_END


def _stub_oof(X: np.ndarray, *_a: Any, **_k: Any) -> tuple[np.ndarray, dict]:
    # Also keeps the OOF parquet caches (relative .planning paths) untouched.
    return np.full(len(X), 0.5), {}


@pytest.fixture
def fake_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the corpus's pre-fight Elo where a driver still reads it from the DB."""
    c = _corpus()
    monkeypatch.setattr("ufc_prediction.db.session.SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr("ufc_prediction.ml.queries.load_elo_features", lambda _s: c.elo_features)


@pytest.fixture
def stub_base_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ufc_prediction.ml.oof.generate_oof_predictions", _stub_oof)
    monkeypatch.setattr(
        "ufc_prediction.ml.oof._make_oof_estimator", lambda seed=42: _ConstEstimator()
    )


@pytest.fixture
def level1_rows(monkeypatch: pytest.MonkeyPatch) -> list[tuple[np.ndarray, np.ndarray]]:
    """Capture (elo_prob, elo_overall_diff) from each META-V22 Level-1 build.

    The drivers build the meta_train set and then the meta_eval set. The spy
    stops the run after the second build, before any fit or artifact write.
    """
    from ufc_prediction.ml import meta_features_v22

    real = meta_features_v22.build_meta_features_v22
    seen: list[tuple[np.ndarray, np.ndarray]] = []

    def spy(xgb_oof_prob: np.ndarray, elo_prob: np.ndarray, X_v22: np.ndarray) -> np.ndarray:
        out = real(xgb_oof_prob, elo_prob, X_v22)
        seen.append((np.asarray(elo_prob, dtype=float).copy(), X_v22[:, ELO_DIFF_IDX].copy()))
        if len(seen) == 2:
            raise _StopRunError
        return out

    monkeypatch.setattr("ufc_prediction.ml.meta_features_v22.build_meta_features_v22", spy)
    return seen


def _assert_level1_rows(seen: list[tuple[np.ndarray, np.ndarray]], where: str) -> None:
    assert len(seen) == 2, f"{where}: expected the meta_train and meta_eval builds"
    for (elo_prob, diff), part in zip(seen, ("meta_train", "meta_eval"), strict=True):
        _assert_row_oriented(elo_prob, diff, f"{where} {part}")


def _patch_train_meta_v22(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    tm = _import_script("train_meta_v22")
    monkeypatch.setattr(tm, "assert_phase26_invariants", lambda: None)
    monkeypatch.setattr(tm, "_load_assembled_data_v22", _assembled_v22)
    monkeypatch.setattr(
        tm,
        "_build_meta_eval_xgb_probs",
        lambda _est, _Xt, _yt, X_eval: np.full(len(X_eval), 0.5),
    )
    return tm


# ── The corpus has teeth ────────────────────────────────────────────────────


def test_corpus_label_is_the_swap_and_record_oriented_elo_prob_reveals_it() -> None:
    c = _corpus()
    swap = np.array([_swap(r["fight_id"]) for r in c.records])
    assert 0.3 < swap.mean() < 0.7
    assert np.array_equal(c.y, 1 - swap.astype(int))

    engine = EloEngine(EloConfig())
    record_oriented = np.array(
        [
            engine.expected_win_probability(float(a), float(b))
            for a, b in zip(c.rating_a, c.rating_b, strict=True)
        ]
    )
    diff = c.X_v22[:, ELO_DIFF_IDX]
    assert _fit_auc([diff], c.y) < 0.75
    assert _fit_auc([record_oriented, diff, record_oriented * diff], c.y) > 0.95


def test_shared_helper_is_row_oriented() -> None:
    c = _corpus()
    diff = c.X_v22[:, ELO_DIFF_IDX]
    _assert_row_oriented(elo_prob_from_v22_matrix(c.X_v22), diff, "v2.2 matrix", c.y)
    # The 72-col v2.1-no-net view is the v2.2 prefix; train_meta_v1 passes it.
    _assert_row_oriented(elo_prob_from_v22_matrix(c.X_72), diff, "72-col view", c.y)


# ── train_meta_v22 (META_V22_SPIKE.json / meta_v2 lineage) ───────────────────


@pytest.mark.usefixtures("fake_db", "stub_base_models")
def test_train_meta_v22_spike_live_elo_prob_is_row_oriented(
    monkeypatch: pytest.MonkeyPatch, level1_rows: list[tuple[np.ndarray, np.ndarray]]
) -> None:
    tm = _patch_train_meta_v22(monkeypatch)
    with pytest.raises(_StopRunError):
        tm.main(["--seeds", "42"])
    _assert_level1_rows(level1_rows, "train_meta_v22 spike")


@pytest.mark.usefixtures("fake_db", "stub_base_models")
def test_train_meta_v22_stepwise_live_elo_prob_is_row_oriented(
    monkeypatch: pytest.MonkeyPatch,
    level1_rows: list[tuple[np.ndarray, np.ndarray]],
    tmp_path: Path,
) -> None:
    tm = _patch_train_meta_v22(monkeypatch)
    slices = ("most_recent_12mo", "most_recent_24mo", "random_15pct")
    spike_json = tmp_path / "META_V22_SPIKE.json"
    spike_json.write_text(
        json.dumps(
            {
                "stepwise_clears_vs_xgb_v2": False,
                "xgb_v2_baseline_brier": dict.fromkeys(slices, 0.2),
                "median_per_slice": {s: {"brier_score": 0.2} for s in slices},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(tm, "SPIKE_JSON_PATH", spike_json)
    with pytest.raises(_StopRunError):
        tm.main(["--mode", "stepwise", "--seeds", "42"])
    _assert_level1_rows(level1_rows, "train_meta_v22 stepwise")


# ── spike_noise_floor_v23 (variance harness / gate-spike contract) ──────────


@pytest.mark.usefixtures("fake_db", "stub_base_models")
def test_spike_v23_live_loader_elo_prob_is_row_oriented(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_train_meta_v22(monkeypatch)
    spike = _import_script("spike_noise_floor_v23")
    X_train, _y_train, X_eval, _y_eval, _dates = spike._load_meta_train_eval_matrices(dry_run=False)
    for X, part in ((X_train, "meta_train"), (X_eval, "meta_eval")):
        _assert_row_oriented(
            X[:, META_ELO_PROB_IDX], X[:, META_ELO_DIFF_IDX], f"spike_noise_floor_v23 {part}"
        )


# ── compose_v23_meta / compose_v25_travel ────────────────────────────────────


@pytest.mark.usefixtures("fake_db", "stub_base_models")
def test_compose_v23_live_elo_prob_is_row_oriented(
    monkeypatch: pytest.MonkeyPatch,
    level1_rows: list[tuple[np.ndarray, np.ndarray]],
    tmp_path: Path,
) -> None:
    # compose_v23_meta cross-checks its pinned baseline against a CWD-relative
    # META_V22_SPIKE.json at import time; run from an empty directory.
    monkeypatch.chdir(tmp_path)
    compose = _import_script("compose_v23_meta")
    monkeypatch.setattr(compose, "_read_xgb_v2_sha", lambda: compose.EXPECTED_XGB_V2_SHA256)
    monkeypatch.setattr(compose, "_load_assembled_data_v22", _assembled_v22)
    args = compose.build_parser().parse_args(["--seeds", "42"])
    with pytest.raises(_StopRunError):
        compose.run_composition(args)
    _assert_level1_rows(level1_rows, "compose_v23_meta")


@pytest.mark.usefixtures("fake_db", "stub_base_models")
def test_compose_v25_travel_live_elo_prob_is_row_oriented(
    monkeypatch: pytest.MonkeyPatch, level1_rows: list[tuple[np.ndarray, np.ndarray]]
) -> None:
    compose = _import_script("compose_v25_travel")

    def assembled_v25() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        X_v22, y, dates, records = _assembled_v22()
        return np.column_stack([X_v22, np.zeros((len(X_v22), 2))]), y, dates, records

    monkeypatch.setattr(compose, "_assert_canonical_shas", lambda: ("0" * 64, "0" * 64))
    monkeypatch.setattr(compose, "_load_assembled_data_v25_travel", assembled_v25)
    monkeypatch.setattr(compose, "date", _FixedDate)  # its split still reads date.today()
    args = compose.build_parser().parse_args(["--seeds", "42"])
    with pytest.raises(_StopRunError):
        compose.run_composition(args)
    _assert_level1_rows(level1_rows, "compose_v25_travel")


# ── train_meta_v1 (META-01, 3-col Level-1) ──────────────────────────────────


@pytest.mark.usefixtures("fake_db", "stub_base_models")
def test_train_meta_v1_live_elo_prob_is_row_oriented(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _corpus()
    tm1 = _import_script("train_meta_v1")
    # The 3-col Level-1 set has no elo_overall_diff column; tag each row
    # through closing_prob_diff (NaN in this odds-free corpus) instead.
    X_tagged = c.X_72.copy()
    X_tagged[:, CLOSING_IDX_72] = np.arange(N_FIGHTS, dtype=float)
    monkeypatch.setattr(tm1, "assert_phase19_invariants", lambda: None)
    monkeypatch.setattr(
        tm1,
        "_load_assembled_data",
        lambda: (X_tagged.copy(), c.y.copy(), c.dates.copy(), [dict(r) for r in c.records]),
    )
    monkeypatch.setattr(
        tm1,
        "_build_meta_eval_xgb_probs",
        lambda _est, _Xt, _yt, X_eval: np.full(len(X_eval), 0.5),
    )
    monkeypatch.setattr(tm1, "date", _FixedDate)  # its split still reads date.today()

    from ufc_prediction.ml import meta_learner

    real = meta_learner.build_meta_features
    seen: list[tuple[np.ndarray, np.ndarray]] = []

    def spy(xgb_prob: np.ndarray, elo_prob: np.ndarray, closing: np.ndarray) -> np.ndarray:
        out = real(xgb_prob, elo_prob, closing)
        rows = np.asarray(closing).astype(int)
        seen.append((np.asarray(elo_prob, dtype=float).copy(), c.X_72[rows, ELO_DIFF_IDX]))
        if len(seen) == 2:
            raise _StopRunError
        return out

    monkeypatch.setattr("ufc_prediction.ml.meta_learner.build_meta_features", spy)
    with pytest.raises(_StopRunError):
        tm1.main(["--seeds", "42"])
    _assert_level1_rows(seen, "train_meta_v1")


# ── Sibling / candidate meta trainers (13-col live builders) ────────────────


def _oof_df(value: float = 0.5) -> Any:
    import pandas as pd

    ids = [r["fight_id"] for r in _corpus().records]
    return pd.DataFrame({"fight_id": ids, "oof_prob": [value] * len(ids)})


@pytest.mark.usefixtures("fake_db")
@pytest.mark.parametrize(
    ("script", "kwarg"),
    [
        ("train_meta_v2_refv2", "xgb_refv2_oof_df"),
        ("train_meta_v2_netd", "xgb_netd_oof_df"),
    ],
)
def test_candidate_live_13col_elo_prob_is_row_oriented(
    monkeypatch: pytest.MonkeyPatch, script: str, kwarg: str
) -> None:
    _patch_train_meta_v22(monkeypatch)
    mod = _import_script(script)
    X_13, y = mod._build_live_13col_matrix(**{kwarg: _oof_df()})
    assert len(X_13) == N_FIGHTS
    _assert_row_oriented(X_13[:, META_ELO_PROB_IDX], X_13[:, META_ELO_DIFF_IDX], script, y)


@pytest.mark.usefixtures("fake_db")
def test_refit_v2_6_live_13col_elo_prob_is_row_oriented(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_train_meta_v22(monkeypatch)
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "_s18_refit_meta_v22_v2_6", SCRIPTS_DIR / "refit_meta_v22_v2.6.py"
    )
    assert spec is not None and spec.loader is not None
    refit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refit)

    oof_path = tmp_path / "data" / "intermediate" / "xgb_v2_oof.parquet"
    oof_path.parent.mkdir(parents=True)
    _oof_df().to_parquet(oof_path)
    monkeypatch.setattr(refit, "PROJECT_ROOT", tmp_path)

    X_13, y = refit._build_live_13col_matrix()
    assert len(X_13) == N_FIGHTS
    _assert_row_oriented(
        X_13[:, META_ELO_PROB_IDX], X_13[:, META_ELO_DIFF_IDX], "refit_meta_v22_v2.6", y
    )


# ── meta_v3 (v2.5) trainer + gate verifier ──────────────────────────────────


@pytest.mark.usefixtures("fake_db")
def test_train_meta_v3_v25_level1_elo_prob_is_row_oriented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _corpus()
    for name, value in (
        ("load_fight_records", [dict(r) for r in c.records]),
        ("load_computed_features", {}),
        ("load_fighter_physicals", c.physicals),
        ("load_round_stats_for_ml", {}),
        ("load_pre_ufc_records", {}),
        ("load_fight_odds", {}),
    ):
        monkeypatch.setattr(f"ufc_prediction.ml.queries.{name}", lambda _s, _v=value: _v)
    mod = _import_script("train_meta_v3_v25")
    df = mod._load_level1_substrate_from_db()
    assert len(df) == N_FIGHTS
    _assert_row_oriented(
        df["elo_prob"].to_numpy(),
        df["elo_overall_diff"].to_numpy(),
        "train_meta_v3_v25",
        df["y"].to_numpy(),
    )


def test_verify_meta_v3_gate_level1_elo_prob_is_row_oriented() -> None:
    X_v22, y, _dates, records = _assembled_v22()
    mod = _import_script("verify_meta_v3_gate_v25")
    df = mod._build_level1_df(X_v22, y, records)
    _assert_row_oriented(
        df["elo_prob"].to_numpy(),
        df["elo_overall_diff"].to_numpy(),
        "verify_meta_v3_gate_v25",
        df["y"].to_numpy(),
    )

"""S18 / operator decision D2: the meta gate paths anchor their dates to the substrate.

PR #27 anchored the non-protected gate paths to
``evaluator.gate_reference_date(fight_dates)``, the latest event in the
substrate. The meta-gate producers still cut the three-way split and the
12mo / 24mo slices from ``date.today()``:
- ``scripts/train_meta_v22.py``, which writes ``META_V22_SPIKE.json`` and
  from it the ``META_V22_BASELINE_BRIER`` anchors
- ``scripts/spike_noise_floor_v23.py``, the variance harness
- the in-process re-run of that harness in ``ufc predict gate-spike``
- ``scripts/compose_v23_meta.py``

So the same substrate produced different gate numbers on different days.

Each test feeds a substrate whose latest event (``SUBSTRATE_END``) is well in
the past. It then asserts that every split and slice anchor the path uses is
that date, not the wall clock. The heavy base-model steps (OOF generation, the
transient eval XGB) are stubbed. They do not take a date.
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"

# Long before any plausible wall-clock "today", but late enough that the
# 2023-01-01 base cutoff still leaves a non-empty meta_train partition under
# both the 365d and the 730d meta_eval windows.
SUBSTRATE_END = date(2025, 12, 31)
SUBSTRATE_START = date(2020, 1, 1)


class _StopRunError(Exception):
    """Raised by a spy once the anchor under test has been observed."""


def _import_script(name: str) -> ModuleType:
    # Resolve at call time: other tests pop / replace script modules in
    # sys.modules, so a module-level import could hold a stale object that
    # the code under test no longer sees.
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    return importlib.import_module(name)


def _substrate_dates(n: int) -> np.ndarray:
    span = (SUBSTRATE_END - SUBSTRATE_START).days
    dates = [SUBSTRATE_START + timedelta(days=round(i * span / (n - 1))) for i in range(n)]
    # Shuffle so the anchor has to be the max, not the last row.
    order = np.random.default_rng(3).permutation(n)
    return np.array([dates[i] for i in order], dtype=object)


def _synthetic_v22(n: int = 600):
    """Same shape as ``train_meta_v22._build_synthetic_data_v22`` (90 cols),
    but every event date is on or before ``SUBSTRATE_END``."""
    rng = np.random.default_rng(42)
    X_v22 = rng.standard_normal((n, 90))
    y = rng.integers(0, 2, size=n)
    dates = _substrate_dates(n)
    return X_v22, y, dates, list(range(n)), date(2023, 1, 1), date.today()


class _ConstEstimator:
    """Stand-in for the transient base XGB (no date involved)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> _ConstEstimator:
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])


def _stub_oof(X: np.ndarray, *_a: Any, **_k: Any) -> tuple[np.ndarray, dict]:
    # Also keeps the OOF parquet cache (a relative .planning path) untouched.
    return np.full(len(X), 0.5), {}


@pytest.fixture
def anchors(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Record the ``today`` every split / slice call receives.

    ``stop_at_slice`` (default True) raises ``_StopRunError`` on the first
    ``evaluate_per_slice`` call, so a driver run ends before it writes any
    artifact.
    """
    from ufc_prediction.ml import evaluator, oof

    seen: dict[str, list[Any]] = {"split": [], "slice": [], "stop_at_slice": [True]}
    real_split = oof.make_three_way_split
    real_eval = evaluator.evaluate_per_slice

    def split_spy(*args: Any, **kwargs: Any) -> Any:
        seen["split"].append(kwargs.get("today"))
        return real_split(*args, **kwargs)

    def eval_spy(*args: Any, **kwargs: Any) -> Any:
        seen["slice"].append(kwargs.get("today"))
        if seen["stop_at_slice"][0]:
            raise _StopRunError
        return real_eval(*args, **kwargs)

    monkeypatch.setattr("ufc_prediction.ml.oof.make_three_way_split", split_spy)
    monkeypatch.setattr("ufc_prediction.ml.oof.generate_oof_predictions", _stub_oof)
    monkeypatch.setattr("ufc_prediction.ml.evaluator.evaluate_per_slice", eval_spy)
    return seen


def _patch_train_meta_v22(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    tm = _import_script("train_meta_v22")
    monkeypatch.setattr(tm, "assert_phase26_invariants", lambda: None)
    monkeypatch.setattr(tm, "_build_synthetic_data_v22", _synthetic_v22)
    monkeypatch.setattr(
        tm,
        "_build_meta_eval_xgb_probs",
        lambda _est, _Xt, _yt, X_eval: np.full(len(X_eval), 0.5),
    )
    return tm


# ── train_meta_v22 (META_V22_SPIKE.json producer) ─────────────────────────


def test_train_meta_v22_spike_anchors_split_and_slices_to_substrate(
    monkeypatch: pytest.MonkeyPatch, anchors: dict[str, list[Any]]
) -> None:
    tm = _patch_train_meta_v22(monkeypatch)
    with pytest.raises(_StopRunError):
        tm.main(["--dry-run", "--seeds", "42"])
    assert anchors["split"] == [SUBSTRATE_END]
    assert anchors["slice"] == [SUBSTRATE_END]


def test_train_meta_v22_stepwise_anchors_split_and_slices_to_substrate(
    monkeypatch: pytest.MonkeyPatch,
    anchors: dict[str, list[Any]],
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
        tm.main(["--mode", "stepwise", "--dry-run", "--seeds", "42"])
    assert anchors["split"] == [SUBSTRATE_END]
    assert anchors["slice"] == [SUBSTRATE_END]


# ── spike_noise_floor_v23 (variance harness) ──────────────────────────────


def test_spike_v23_loader_anchors_split_to_substrate(
    monkeypatch: pytest.MonkeyPatch, anchors: dict[str, list[Any]]
) -> None:
    _patch_train_meta_v22(monkeypatch)
    spike = _import_script("spike_noise_floor_v23")
    out = spike._load_meta_train_eval_matrices(dry_run=True)
    assert anchors["split"] == [SUBSTRATE_END]
    assert max(out[4]) == SUBSTRATE_END  # fight_dates_eval ends at the anchor


def _fake_eval_matrices() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(0)
    n_train, n_eval = 120, 90
    X_train = rng.standard_normal((n_train, 13))
    y_train = (rng.random(n_train) < 0.5).astype(int)
    X_eval = rng.standard_normal((n_eval, 13))
    y_eval = (rng.random(n_eval) < 0.5).astype(int)
    dates_eval = _substrate_dates(400)[:n_eval]
    dates_eval[17] = SUBSTRATE_END  # make sure the latest event is present
    return X_train, y_train, X_eval, y_eval, dates_eval


def _harness_spies(
    monkeypatch: pytest.MonkeyPatch, spike: ModuleType, seen: dict[str, list[Any]]
) -> None:
    """Spy the harness calls both spike_v23.main and the CLI re-run make."""
    fake = _fake_eval_matrices()
    per_seed = {
        42: {
            s: {"brier_score": 0.2, "accuracy": 0.6, "auc_roc": 0.6}
            for s in ("most_recent_12mo", "most_recent_24mo", "random_15pct")
        }
    }

    def no_bootstrap_spy(*_a: Any, seeds: Any, today: Any = None, **_k: Any) -> Any:
        seen["harness"].append(today)
        return per_seed

    def multi_seed_spy(*_a: Any, today: Any = None, **_k: Any) -> Any:
        seen["harness"].append(today)
        return per_seed

    def aggregate_spy(*_a: Any, today: Any = None, **_k: Any) -> Any:
        seen["aggregate"].append(today)
        raise _StopRunError

    monkeypatch.setattr(spike, "_load_meta_train_eval_matrices", lambda *, dry_run: fake)
    monkeypatch.setattr(spike, "_no_bootstrap_metrics", no_bootstrap_spy)
    monkeypatch.setattr(spike, "_assert_xgb_v2_sha", lambda label: spike.EXPECTED_XGB_V2_SHA)
    monkeypatch.setattr("ufc_prediction.ml.variance.multi_seed_metrics", multi_seed_spy)
    monkeypatch.setattr("ufc_prediction.ml.variance.assert_distinct_seed_brier", lambda _ps: None)
    monkeypatch.setattr("ufc_prediction.ml.variance.aggregate_variance", aggregate_spy)


@pytest.mark.parametrize("bootstrap", [False, True], ids=["no_bootstrap", "bootstrap"])
def test_spike_v23_main_anchors_every_seed_and_ci_to_latest_eval_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bootstrap: bool
) -> None:
    spike = _import_script("spike_noise_floor_v23")
    seen: dict[str, list[Any]] = {"harness": [], "aggregate": []}
    _harness_spies(monkeypatch, spike, seen)
    argv = [
        "--seeds",
        "42",
        "--report-path",
        str(tmp_path / "r.md"),
        "--halt-path",
        str(tmp_path / "h.md"),
        "--sha-end-path",
        str(tmp_path / "s.txt"),
    ]
    if not bootstrap:
        argv.append("--no-bootstrap")
    with pytest.raises(_StopRunError):
        spike.main(argv)
    assert seen["harness"] == [SUBSTRATE_END]
    assert seen["aggregate"] == [SUBSTRATE_END]


@pytest.mark.parametrize("bootstrap", [False, True], ids=["no_bootstrap", "bootstrap"])
def test_gate_spike_in_process_rerun_uses_the_spike_anchor(
    monkeypatch: pytest.MonkeyPatch, bootstrap: bool
) -> None:
    """WR-01 parity: the contract values `ufc predict gate-spike` derives in
    process must be cut on the same slices as the spike's report."""
    from ufc_prediction.cli import predict as cli_predict

    spike = _import_script("spike_noise_floor_v23")
    seen: dict[str, list[Any]] = {"harness": [], "aggregate": []}
    _harness_spies(monkeypatch, spike, seen)
    with pytest.raises(_StopRunError):
        cli_predict._compute_v23_variance_dict([42], bootstrap=bootstrap)
    assert seen["harness"] == [SUBSTRATE_END]
    assert seen["aggregate"] == [SUBSTRATE_END]


# ── compose_v23_meta (4-step composition gate) ────────────────────────────


def test_compose_v23_anchors_split_and_every_step_to_substrate(
    monkeypatch: pytest.MonkeyPatch,
    anchors: dict[str, list[Any]],
    tmp_path: Path,
) -> None:
    # compose_v23_meta cross-checks its pinned META_V22_BASELINE_BRIER against
    # a CWD-relative META_V22_SPIKE.json at import time; run from an empty
    # directory so a stale local spike file cannot abort this date test, and
    # so the dry-run's relative artifact paths land in tmp_path.
    monkeypatch.chdir(tmp_path)
    compose = _import_script("compose_v23_meta")
    monkeypatch.setattr(compose, "_read_xgb_v2_sha", lambda: compose.EXPECTED_XGB_V2_SHA256)
    monkeypatch.setattr(compose, "_build_synthetic_data_v22", _synthetic_v22)
    monkeypatch.setattr("ufc_prediction.ml.oof._make_oof_estimator", lambda seed: _ConstEstimator())
    monkeypatch.setattr(compose, "_emit_report_and_end_sha", lambda report: None)
    anchors["stop_at_slice"][0] = False  # run every step; record each anchor

    args = compose.build_parser().parse_args(["--dry-run", "--seeds", "42"])
    compose.run_composition(args)

    assert anchors["split"] == [SUBSTRATE_END]
    assert anchors["slice"], "no evaluate_per_slice call was observed"
    assert set(anchors["slice"]) == {SUBSTRATE_END}

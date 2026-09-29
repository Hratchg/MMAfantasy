"""S11 finding 2 — gate metric paths must be anchorable to a fixed reference date.

``evaluate_per_slice`` / ``bootstrap_per_slice_ci`` accept ``today`` but the
multi-seed gate harness (``variance.multi_seed_metrics`` /
``variance.aggregate_variance``) had no way to pass it, so every gate run
cut its 12mo / 24mo slices relative to the wall clock: the same substrate
produced different gate numbers on different days, and a run crossing
midnight evaluated different seeds on different slice masks.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pytest

from ufc_prediction.ml import evaluator, variance
from ufc_prediction.ml.evaluator import gate_reference_date
from ufc_prediction.ml.meta_learner import MetaLearnerLogistic

SUBSTRATE_END = date(2021, 3, 1)  # long before any plausible wall-clock "today"


def _eval_set(n: int = 240, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.standard_normal((n, 4))
    y = (rng.random(n) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    days_back = rng.randint(0, 3 * 365, size=n)
    days_back[0] = 0  # the latest fight is exactly SUBSTRATE_END
    dates = np.array([SUBSTRATE_END - timedelta(days=int(k)) for k in days_back])
    return X, y, dates


def _fit(X: np.ndarray, y: np.ndarray, seed: int) -> MetaLearnerLogistic:
    return MetaLearnerLogistic(random_state=seed).fit(X, y)


def test_gate_reference_date_is_latest_event_date() -> None:
    _X, _y, dates = _eval_set()
    assert gate_reference_date(dates) == SUBSTRATE_END
    assert gate_reference_date(list(dates)) == SUBSTRATE_END


def test_gate_reference_date_empty_means_no_anchor() -> None:
    """Empty eval set: every window slice is empty, so keep the callee default."""
    assert gate_reference_date(np.array([])) is None


def test_multi_seed_metrics_honours_reference_date() -> None:
    """With an explicit anchor the slices match evaluate_per_slice(today=anchor).

    Without threading, the 12mo / 24mo windows are cut from the wall clock
    and (for a 2021 substrate) are empty, so the numbers cannot match.
    """
    X, y, dates = _eval_set()
    anchor = gate_reference_date(dates)
    per_seed = variance.multi_seed_metrics(
        X, y, X, y, dates, seeds=[42, 43], fit_fn=_fit, today=anchor
    )
    for seed in (42, 43):
        Xb, yb = variance.bootstrap_resample(X, y, seed=seed)
        expected = evaluator.evaluate_per_slice(_fit(Xb, yb, seed), X, y, dates, today=anchor)
        for slc in evaluator.PER_SLICE_KEYS:
            assert per_seed[seed][slc]["brier_score"] == expected[slc]["brier_score"]


def test_aggregate_variance_threads_reference_date(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    real = variance.bootstrap_per_slice_ci

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        kwargs["n_resamples"] = 99
        return real(*args, **kwargs)

    monkeypatch.setattr(variance, "bootstrap_per_slice_ci", spy)
    X, y, dates = _eval_set()
    anchor = gate_reference_date(dates)
    per_seed = variance.multi_seed_metrics(
        X, y, X, y, dates, seeds=[42, 43], fit_fn=_fit, today=anchor
    )
    variance.aggregate_variance(
        per_seed,
        representative_model=_fit(X, y, 42),
        X_eval=X,
        y_eval=y,
        fight_dates_eval=dates,
        today=anchor,
    )
    assert seen["today"] == anchor


def test_multi_seed_metrics_resolves_wall_clock_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default (no anchor) path: every seed must share ONE slice anchor.

    Simulates a run that crosses midnight: each ``date.today()`` call
    returns the next day.
    """
    calls = {"n": 0}

    class _TickingDate(date):
        @classmethod
        def today(cls) -> _TickingDate:
            calls["n"] += 1
            return cls(2021, 3, 1) + timedelta(days=calls["n"])  # type: ignore[return-value]

    seen: list[date | None] = []

    def spy(model: Any, X: Any, y: Any, fd: Any, *, today: date | None = None, **kw: Any) -> Any:
        seen.append(today)
        return {slc: {"brier_score": float(len(seen))} for slc in evaluator.PER_SLICE_KEYS}

    monkeypatch.setattr(variance, "date", _TickingDate)
    monkeypatch.setattr(variance, "evaluate_per_slice", spy)
    X, y, dates = _eval_set()
    variance.multi_seed_metrics(X, y, X, y, dates, seeds=[1, 2, 3], fit_fn=_fit)
    assert len(seen) == 3
    assert seen[0] is not None
    assert len(set(seen)) == 1, seen

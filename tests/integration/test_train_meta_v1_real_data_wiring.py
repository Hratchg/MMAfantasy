"""Phase 19 / Plan 19-02 Task 1a — mock-based integration tests for the
``scripts/train_meta_v1.py`` real-data wiring (BLOCKER #2 split).

Per OQ-5 lock (Plan 19-01) and the Plan 19-02 Task 1a action contract, all
3 tests use ``unittest.mock.patch`` to substitute:

  - the script's data loader (``_load_assembled_data``) — returns synthetic
    600-row (X, y, fight_dates, fight_records) suitable for the three-way
    split with non-empty meta_train + meta_eval partitions;
  - the script's Elo-prob lookup (``_compute_elo_prob_for_fight``) — returns
    a deterministic value so tests do not require a live ``EloEngine`` or
    DB session;
  - ``save_meta_model`` — captured as a MagicMock so test 3 can inspect the
    full kwargs schema.

NO DB. NO live BFO HTTP. NO xgb_v2 retraining (the OOF generation uses the
in-memory base estimator built from ``models/xgb_v2_meta.json::best_params``;
``models/xgb_v2.joblib`` is only read for SHA verification).

Tests follow the Plan 19-01 mock-style integration test pattern established
by ``tests/integration/test_predict_matchup_meta.py``.
"""
from __future__ import annotations

import datetime as _dt
import importlib.util
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Importorskip pattern (Plan 19-01 Wave-0 RED discipline carry-forward) —
# this test file ships RED if the script is not yet importable, then flips
# to GREEN once Task 1a's wiring lands.
spec = importlib.util.spec_from_file_location(
    "train_meta_v1",
    Path(__file__).resolve().parents[2] / "scripts" / "train_meta_v1.py",
)
if spec is None or spec.loader is None:
    pytest.skip(
        "scripts/train_meta_v1.py not importable as module",
        allow_module_level=True,
    )
train_meta_v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_meta_v1)

pytestmark = [pytest.mark.integration]


# ── Synthetic data helpers ──────────────────────────────────────────────────


def _build_synthetic_real_data(
    n: int = 600,
    n_features: int = 72,
    seed: int = 42,
):
    """Synthesize a (X, y, fight_dates, fight_records) tuple shaped like the
    real ``_load_assembled_data`` output.

    Partitions for the three-way split (per D-01(P19) + RESEARCH §"Three-Way
    Split Partition Strategy"):
      - 200 rows pre-cutoff (event_date < 2023-01-01) — base_train
      - 200 rows post-cutoff but > 365d ago — meta_train
      - 200 rows last 365 days — meta_eval

    The closing_prob_diff column (FEATURE_COLUMNS_NO_NET index for that
    column) is filled with small finite values so META-01 doesn't drop them
    on NaN. Other columns are gaussian noise.

    fight_records is a list[dict] with the minimum fields the wiring
    references: fight_id, event_date, fighter_a_id, fighter_b_id.
    """
    from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    # Fill closing_prob_diff with small finite values (avoid NaN drop in META).
    closing_idx = FEATURE_COLUMNS_NO_NET.index("closing_prob_diff")
    X[:, closing_idx] = rng.uniform(-0.3, 0.3, size=n)
    # Synthesize a target slightly correlated with closing_prob_diff so the
    # logistic baseline has some signal (avoids a perfectly random meta_eval).
    y = (X[:, closing_idx] + rng.normal(0, 0.5, size=n) > 0).astype(np.int32)

    today = date.today()
    base_cutoff = date(2023, 1, 1)
    eval_start = today - timedelta(days=365)

    dates: list[date] = []
    fight_records: list[dict] = []
    for i in range(n):
        if i < n // 3:
            # Pre-cutoff
            d = date(2020 + (i % 3), 6, 1 + (i % 28))
        elif i < 2 * n // 3:
            # Post-cutoff but > 365d ago
            d = base_cutoff + _dt.timedelta(
                days=int(rng.integers(0, max(1, (eval_start - base_cutoff).days - 1)))
            )
        else:
            # Last 365 days
            d = eval_start + _dt.timedelta(
                days=int(rng.integers(0, 364))
            )
        dates.append(d)
        fight_records.append({
            "fight_id": 100_000 + i,
            "event_date": d,
            "fighter_a_id": 10_000 + (i * 2),
            "fighter_b_id": 10_000 + (i * 2) + 1,
            "weight_class": "lightweight",
            "winner_id": (10_000 + (i * 2)) if y[i] == 1 else (10_000 + (i * 2) + 1),
        })
    fight_dates = np.array(dates, dtype=object)
    return X, y, fight_dates, fight_records


def _deterministic_elo_prob(*args, **kwargs) -> float:
    """Stub — returns 0.5 (no-information Elo)."""
    return 0.5


# ── Common patcher fixture ──────────────────────────────────────────────────


@pytest.fixture
def patched_main_environment(tmp_path, monkeypatch):
    """Patch the data loader, Elo-prob lookup, and save_meta_model so
    ``train_meta_v1.main()`` runs end-to-end without DB / live odds.

    Re-routes the OOF cache + META artifact + report paths to ``tmp_path``
    so each test starts with a clean slate and tests don't pollute the
    git-tracked ``.planning/phases/19-meta-learner/`` tree.
    """
    X, y, fight_dates, fight_records = _build_synthetic_real_data()

    # Re-route persistence paths to tmp_path so we don't pollute the repo.
    oof_cache = tmp_path / "oof_predictions.parquet"
    meta_dir = tmp_path / "models" / "meta"
    meta_dir.mkdir(parents=True)
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    report_path = tmp_path / "19-META-LEARNER-REPORT.md"

    monkeypatch.setattr(train_meta_v1, "META_OOF_PARQUET_PATH", oof_cache)
    monkeypatch.setattr(train_meta_v1, "META_DIR", meta_dir)
    monkeypatch.setattr(train_meta_v1, "META_ARCHIVE_DIR", archive_dir)
    monkeypatch.setattr(train_meta_v1, "META_LEARNER_REPORT_PATH", report_path)

    save_meta_mock = MagicMock(
        return_value=(meta_dir / "meta_v1.joblib", meta_dir / "meta_v1_meta.json"),
    )

    with patch.object(
        train_meta_v1, "_load_assembled_data",
        return_value=(X, y, fight_dates, fight_records),
    ), patch.object(
        train_meta_v1, "_compute_elo_prob_for_fight",
        side_effect=_deterministic_elo_prob,
    ), patch.object(
        train_meta_v1, "save_meta_model", save_meta_mock,
    ):
        yield {
            "save_meta_mock": save_meta_mock,
            "oof_cache": oof_cache,
            "meta_dir": meta_dir,
            "archive_dir": archive_dir,
            "report_path": report_path,
            "X": X,
            "y": y,
            "fight_records": fight_records,
        }


# ── The 3 integration tests ────────────────────────────────────────────────


def test_real_data_loop_runs_for_5_seeds(patched_main_environment):
    """train_meta_v1.main() runs the full real-data pipeline across 5 seeds
    without raising — exits 0 (gate-pass) or 1 (gate-fail), NEVER raises.

    Asserts the run COMPLETES (Task 1a behavior), not that it ships.
    """
    rc = train_meta_v1.main(
        ["--seeds", "42", "43", "44", "45", "46"],
    )
    assert rc in (0, 1), (
        f"main() should exit 0 or 1, got {rc!r}"
    )
    # Per-seed metrics must have been collected for all 5 seeds.
    per_seed = train_meta_v1.LAST_PER_SEED_RESULTS
    assert isinstance(per_seed, dict), (
        f"LAST_PER_SEED_RESULTS not populated: {per_seed!r}"
    )
    assert set(per_seed.keys()) == {42, 43, 44, 45, 46}, (
        f"per-seed keys {sorted(per_seed.keys())} != {{42..46}}"
    )


def test_per_seed_metrics_for_all_3_slices(patched_main_environment):
    """After main() returns, each seed's results dict must contain entries
    for all 3 slices (most_recent_12mo, most_recent_24mo, random_15pct),
    each with brier_score + accuracy keys.
    """
    rc = train_meta_v1.main(
        ["--seeds", "42", "43", "44", "45", "46"],
    )
    assert rc in (0, 1)
    per_seed = train_meta_v1.LAST_PER_SEED_RESULTS
    expected_slices = {"most_recent_12mo", "most_recent_24mo", "random_15pct"}
    for seed, results in per_seed.items():
        assert set(results.keys()) >= expected_slices, (
            f"seed={seed} slices {sorted(results.keys())} missing some of "
            f"{sorted(expected_slices)}"
        )
        for slice_name in expected_slices:
            slice_metrics = results[slice_name]
            assert "brier_score" in slice_metrics, (
                f"seed={seed} slice={slice_name} missing brier_score"
            )
            assert "accuracy" in slice_metrics, (
                f"seed={seed} slice={slice_name} missing accuracy"
            )


def test_save_meta_model_called_with_full_schema(patched_main_environment, monkeypatch):
    """When the gate clears, save_meta_model is called exactly once with the
    full META-04 + OQ-4 schema kwargs.

    To force the gate to clear regardless of synthetic-data variance, we
    monkeypatch ``gate_verdict`` to always return (True, []).
    """
    from ufc_prediction.ml import meta_learner as _meta_learner_mod

    # Force the gate to PASS so save_meta_model is invoked.
    monkeypatch.setattr(
        train_meta_v1, "gate_verdict",
        lambda median, contract=None: (True, []),
    )

    rc = train_meta_v1.main(
        ["--seeds", "42", "43", "44", "45", "46"],
    )
    assert rc == 0, f"forced-pass run should exit 0, got {rc!r}"

    save_mock = patched_main_environment["save_meta_mock"]
    assert save_mock.call_count == 1, (
        f"save_meta_model called {save_mock.call_count} times; expected 1 "
        f"(hard-gate-then-save discipline per D-07(P15))"
    )
    kwargs = save_mock.call_args.kwargs
    # Per Task 1a action Step 2(e) — full kwargs schema.
    assert kwargs["meta_kind"] == "logistic"
    assert kwargs["meta_version"] == "v1"
    assert kwargs["base_model_version"] == "v2"
    assert kwargs["base_model_sha256"] == train_meta_v1.EXPECTED_XGB_V2_SHA256
    assert kwargs["meta_feature_columns"] == _meta_learner_mod.META_FEATURE_COLUMNS
    # Distribution + parquet hashes are sha256 hex (length 64).
    assert isinstance(kwargs["meta_input_distribution_hash"], str)
    assert len(kwargs["meta_input_distribution_hash"]) == 64
    assert isinstance(kwargs["meta_oof_parquet_sha256"], str)
    assert len(kwargs["meta_oof_parquet_sha256"]) == 64
    # Fixed META-01 baseline delta.
    assert kwargs["meta_learner_brier_delta_vs_logistic"] == 0.0
    # best_params is the LR hyperparams dict.
    bp = kwargs["best_params"]
    assert isinstance(bp, dict)
    for required_param in ("C", "penalty", "solver", "max_iter"):
        assert required_param in bp, (
            f"best_params missing {required_param!r}: keys={sorted(bp.keys())}"
        )
    # metrics dict shape per save_meta_model contract.
    metrics = kwargs["metrics"]
    assert "per_slice" in metrics
    assert "median_brier_overall" in metrics

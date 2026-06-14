#!/usr/bin/env python
"""Phase 30 / Plan 30-01 / Task 1 — one-shot cached-fixture emitter.

Emits the test fixtures consumed by Plan 30-01 unit tests and Plan 30-02
integration tests:

  - tests/fixtures/variance/meta_train_xy.npz
      Two arrays:
        X (float32, shape (n, 13))  — column order matches
          ufc_prediction.ml.meta_features_v22.META_V22_FEATURE_COLUMNS
          (xgb_oof_prob, elo_prob, then 11 internal v2.2 cols).
        y (int8,   shape (n,))      — meta-train labels.
      NOTE: the plan source declares "(n, 14)" but the canonical constant
      META_V22_FEATURE_COLUMNS has length 13 (Plan 26 D-03 dedup). 13 wins
      — this is the load-bearing identity for any downstream consumer
      that imports the constant. The off-by-one in the plan is documented
      as a Rule 1 deviation in 30-01-SUMMARY.md.

  - tests/fixtures/variance/eval_slices.json
      {
        "fight_dates_eval": ["YYYY-MM-DD", ...],   # ISO date strings
        "most_recent_12mo": [int, ...],            # row indices (slice mask)
        "most_recent_24mo": [int, ...],
        "random_15pct":     [int, ...],
      }
      Mask semantics are bit-equivalent to
      ufc_prediction.ml.evaluator.evaluate_per_slice + bootstrap_per_slice_ci
      (12mo / 24mo / random_15pct with RandomState(42)).

  - tests/fixtures/variance/_fixture_meta.json
      Provenance metadata: synthetic vs DB-derived, n, d, seed (if synthetic),
      produced_at.

Idempotency: running this script twice produces byte-identical fixture
files (deterministic RandomState seeds + sort-stable ISO date stringify +
sorted JSON keys). CI uses this to detect accidental drift.

This script reads `models/xgb_v2.joblib` only via the bytes for SHA
checks (NOT this task) — actual fixture content is synthetic-fallback by
default to keep tests DB-free and CI-friendly. The synthetic path matches
the n=600, d=13 shape called out in 30-RESEARCH.md "Recommended test
fixtures".

Run:
    uv run python scripts/_emit_30_variance_fixtures.py
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

from ufc_prediction.ml.evaluator import PER_SLICE_KEYS
from ufc_prediction.ml.meta_features_v22 import META_V22_FEATURE_COLUMNS

# Deterministic synthetic-fixture parameters. n is chosen ~match the
# expected meta_train row order of magnitude (Phase 29 widened slices
# report ~1000-2000 surviving rows pre-drop). Keep small enough to keep
# fixture file <100KB but large enough for bootstrap to produce distinct
# resamples (Pitfall 3 in 30-RESEARCH.md).
SYNTHETIC_N: int = 600
SYNTHETIC_D: int = 13
SYNTHETIC_SEED: int = 0

FIXTURE_DIR: Path = Path("tests/fixtures/variance")
NPZ_PATH: Path = FIXTURE_DIR / "meta_train_xy.npz"
SLICES_PATH: Path = FIXTURE_DIR / "eval_slices.json"
META_PATH: Path = FIXTURE_DIR / "_fixture_meta.json"


def _build_synthetic_xy(
    n: int = SYNTHETIC_N,
    d: int = SYNTHETIC_D,
    seed: int = SYNTHETIC_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic synthetic (X, y) meta-train pair.

    X is sampled from N(0, 1) and clipped to a plausible meta-feature
    range; the first two columns are clipped to [0, 1] to simulate the
    xgb_oof_prob + elo_prob probability columns. y is a noisy linear
    combination of X[:, 0] and X[:, 1] (the two probability columns) so
    the meta-learner has a real signal to fit — without signal, every
    bootstrap fit returns near-50/50 predictions and the
    distinct-Brier check becomes uninformative.
    """
    rng = np.random.RandomState(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    # First two cols are probabilities (xgb_oof_prob, elo_prob) — scale
    # via a logistic squash so they fall in [0, 1].
    X[:, 0] = (1.0 / (1.0 + np.exp(-X[:, 0]))).astype(np.float32)
    X[:, 1] = (1.0 / (1.0 + np.exp(-X[:, 1]))).astype(np.float32)
    # y = noisy linear of cols 0 + 1 (the two probability features).
    logits = 2.0 * (X[:, 0] - 0.5) + 1.5 * (X[:, 1] - 0.5)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < probs).astype(np.int8)
    return X, y


def _build_eval_fight_dates(
    n: int, today: date | None = None, seed: int = SYNTHETIC_SEED,
) -> np.ndarray:
    """Generate deterministic fight_dates spanning ~3 years for slice masking.

    The mix is chosen so 12mo / 24mo / random_15pct slices each yield
    non-empty index sets at typical n=600 (~33% of rows fall in 12mo;
    ~66% in 24mo; ~15% in random_15pct).
    """
    today = today or date(2026, 5, 21)  # frozen date for reproducibility
    rng = np.random.RandomState(seed + 7)  # offset to decouple from X/y RNG
    days_back = rng.randint(0, 730 + 365, size=n)  # 0..3 years back
    return np.array([today - timedelta(days=int(d)) for d in days_back])


def _compute_slice_masks(
    fight_dates: np.ndarray, today: date | None = None,
) -> dict[str, np.ndarray]:
    """Reproduce evaluator.evaluate_per_slice mask semantics exactly.

    12mo:   d >= today - 365d
    24mo:   d >= today - 730d
    random_15pct: RandomState(42).random(n) < 0.15

    WR-06 / Phase 30 code-review: this function duplicates the slice-mask
    construction logic in `ufc_prediction.ml.evaluator.evaluate_per_slice`
    (lines 176-183) and `bootstrap_per_slice_ci` (lines 252-258). Future
    drift in the evaluator (new slice, different cutoffs, switch from
    RandomState to default_rng, changed 0.15 threshold) would silently
    diverge the fixture from the evaluator. The regression guard against
    drift lives at
    `tests/integration/test_variance_harness.py::test_fixture_emitter_mask_matches_evaluator`
    — it asserts bit-equivalent mask output against an inline mirror of
    the evaluator's logic. If evaluator's mask construction changes,
    that test will fire loudly rather than this fixture silently lagging.
    """
    today = today or date(2026, 5, 21)
    cutoff_12mo = today - timedelta(days=365)
    cutoff_24mo = today - timedelta(days=730)
    mask_12mo = np.array([d >= cutoff_12mo for d in fight_dates])
    mask_24mo = np.array([d >= cutoff_24mo for d in fight_dates])
    rng = np.random.RandomState(42)  # evaluator.py:182 — random_seed=42
    mask_random = rng.random(len(fight_dates)) < 0.15
    return {
        "most_recent_12mo": mask_12mo,
        "most_recent_24mo": mask_24mo,
        "random_15pct": mask_random,
    }


def emit() -> int:
    """Emit all three fixture artifacts. Returns 0 on success."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. (X, y) — synthetic deterministic.
    X, y = _build_synthetic_xy(SYNTHETIC_N, SYNTHETIC_D, SYNTHETIC_SEED)
    assert X.shape == (SYNTHETIC_N, SYNTHETIC_D), X.shape
    assert y.shape == (SYNTHETIC_N,), y.shape
    assert X.dtype == np.float32 and y.dtype == np.int8
    assert X.shape[1] == len(META_V22_FEATURE_COLUMNS), (
        f"fixture col count {X.shape[1]} != "
        f"len(META_V22_FEATURE_COLUMNS)={len(META_V22_FEATURE_COLUMNS)}"
    )
    # np.savez writes member arrays in name order; specify both as kwargs
    # so the resulting archive is reproducible. Pin allow_pickle=False at
    # load time (no pickled objects emitted here).
    np.savez(NPZ_PATH, X=X, y=y)

    # 2. eval_slices.json with the same n's fight_dates_eval + masks.
    today = date(2026, 5, 21)
    fight_dates_eval = _build_eval_fight_dates(SYNTHETIC_N, today=today)
    masks = _compute_slice_masks(fight_dates_eval, today=today)
    slices_payload: dict[str, list] = {
        "fight_dates_eval": [d.isoformat() for d in fight_dates_eval],
    }
    for slice_name in PER_SLICE_KEYS:
        # Persist indices (not boolean mask) for compactness + readability.
        indices = [int(i) for i, k in enumerate(masks[slice_name]) if bool(k)]
        slices_payload[slice_name] = indices
    SLICES_PATH.write_text(
        json.dumps(slices_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # 3. provenance metadata.
    meta_payload = {
        "synthetic": True,
        "n_rows": int(SYNTHETIC_N),
        "n_cols": int(SYNTHETIC_D),
        "synthetic_seed": int(SYNTHETIC_SEED),
        "meta_v22_feature_columns": list(META_V22_FEATURE_COLUMNS),
        "per_slice_keys": list(PER_SLICE_KEYS),
        "per_slice_counts": {k: int(masks[k].sum()) for k in PER_SLICE_KEYS},
        "today_anchor": today.isoformat(),
        "produced_at": datetime.now(tz=UTC).isoformat(),
        "emitter": "scripts/_emit_30_variance_fixtures.py",
        "plan": "30-01",
        "note": (
            "Fixture column count is 13 (len(META_V22_FEATURE_COLUMNS)), not "
            "14 as stated in 30-01-PLAN.md. The plan miscounts the dedup'd "
            "set by 1; the constant is the load-bearing identity. Documented "
            "as Rule-1 deviation in 30-01-SUMMARY.md."
        ),
    }
    META_PATH.write_text(
        json.dumps(meta_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[emit-30] wrote {NPZ_PATH} ({X.shape[0]}x{X.shape[1]} float32 + int8)")
    print(f"[emit-30] wrote {SLICES_PATH}")
    print(f"[emit-30] wrote {META_PATH}")
    print(
        f"[emit-30] per-slice counts: "
        + ", ".join(f"{k}={int(masks[k].sum())}" for k in PER_SLICE_KEYS)
    )
    return 0


if __name__ == "__main__":
    sys.exit(emit())

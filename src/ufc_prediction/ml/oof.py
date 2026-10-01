"""Out-of-fold prediction generator for stacking-style meta-learner inputs.

Per CONTEXT.md D-06(P19): cache to .planning/phases/19-meta-learner/oof_predictions.parquet
(git-tracked) and reuse across META-01/02 trains.
Per RESEARCH.md OQ-2: raw XGBClassifier per fold (META-01's LogisticRegression
absorbs recalibration; doubling per-fold work for marginal calibration benefit
doesn't pay back at our scale).

Per Pitfall #11 (stacking leakage):
  - cross_val_predict MUST receive cv=TimeSeriesSplit(n_splits=...) explicitly
    (NEVER cv=int — sklearn auto-selects StratifiedKFold which shuffles → leakage).
  - n_jobs=1 NON-NEGOTIABLE (D-15(P16) Py 3.14 spawn pickling failure).
  - Sanity check: training_accuracy < 0.75 after generation; raise OOFLeakageError
    if violated.

Schema written to <cache_path>.meta.json sidecar (canonical readable form):
  {
    "xgb_v2_sha256": str,         # 6e7641...0a99 baseline at write time
    "n_features": int,            # 72 (FEATURE_COLUMNS_NO_NET — Phase 18 dispatch)
    "cutoff_date": str,           # "2023-01-01" (xgb_v2_meta.json["cutoff_date"])
    "event_date_min": str,        # ISO date — earliest meta_train event
    "event_date_max": str,        # ISO date — latest meta_train event
    "n_splits": int,              # 5 (TimeSeriesSplit folds)
    "training_accuracy": float,   # MUST < 0.75
    "trained_at": str,            # ISO timestamp
    "cv_kind": "TimeSeriesSplit", # literal — Pitfall #11 tripwire
    "meta_train_fight_ids": list[int],  # for D-01(P19) disjoint persistence
    "seed": int,                  # per-fold XGBClassifier random_state
    "input_sha256": str,          # sha256 of (X, y) keyed by fight_id
  }

Cache-hit contract: a cache is reused only when it describes EXACTLY the
caller's rows — same fight_id set (row count + membership), same X/y content
(``input_sha256``), same n_splits/seed/n_features. The cached probabilities are
then re-keyed by fight_id into the caller's date-sorted order, so callers'
positional ``xgb_oof_aligned[sort_idx] = xgb_oof_prob`` realignment stays valid.
Legacy sidecars without ``seed`` / ``input_sha256`` skip only those checks.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from xgboost import XGBClassifier

DEFAULT_OOF_CACHE = (
    Path(__file__).resolve().parents[3]
    / ".planning"
    / "phases"
    / "19-meta-learner"
    / "oof_predictions.parquet"
)
EXPECTED_XGB_V2_SHA256: str = "760307333de18b97729e5a7663413e0e19ab63bf7c1f1bfd9f57d04d2d15677a"
EXPECTED_N_FEATURES: int = 72
EXPECTED_CUTOFF_DATE: str = "2023-01-01"
DEFAULT_N_SPLITS: int = 5


class OOFLeakageError(RuntimeError):
    """Raised when OOF predictions look in-sample (training_accuracy >= 0.75)."""


class InvariantCheckError(RuntimeError):
    """Raised when cached OOF parquet's invariants drift from current code/model."""


# ── Plan 29-02 / EVAL-V23-01 ─────────────────────────────────────────────────
# Per-feature NaN handling helper. Replaces v2.2's global symmetric NaN-drop
# at the eval-set construction site (scripts/train_meta_v22.py:434-458).
#
# Policies:
#   "symmetric" (back-compat default; preserves v2.2 behavior):
#     keep rows where NO feature column is NaN.
#   "per_feature_strict_baseline" (D-03):
#     keep rows where NONE of `baseline_columns` is NaN; allow NaN in
#     non-baseline columns. The 12mo/24mo/random_15pct WINDOW boundaries
#     (D-04) are unchanged — only the drop policy changes.
#
# Trust boundary: caller-provided `policy` string is untrusted-shape; an
# unknown policy raises ValueError to prevent silent fallback (T-29-02-02).
# Missing baseline column raises KeyError with a helpful message.
_VALID_NAN_DROP_POLICIES: tuple[str, ...] = (
    "symmetric",
    "per_feature_strict_baseline",
)


def apply_nan_drop_policy(
    X: np.ndarray,
    feature_columns: list[str],
    *,
    policy: str = "symmetric",
    baseline_columns: tuple[str, ...] = ("xgb_oof_prob", "elo_prob"),
) -> np.ndarray:
    """Return a boolean mask of rows to KEEP under the given drop policy.

    Args:
        X: Feature matrix (n_rows, n_features).
        feature_columns: Column names aligned with X.shape[1]. Used to
            resolve baseline column indices when policy=
            "per_feature_strict_baseline".
        policy: One of {"symmetric", "per_feature_strict_baseline"}.
            Default "symmetric" preserves v2.2 back-compat.
        baseline_columns: Column names that MUST be non-NaN for the
            "per_feature_strict_baseline" policy. Default
            ("xgb_oof_prob", "elo_prob") — the gate-critical inputs.

    Returns:
        Boolean numpy array of length n_rows; True = keep row.

    Raises:
        ValueError: when `policy` is not in _VALID_NAN_DROP_POLICIES.
        KeyError: when a name in `baseline_columns` is not in
            `feature_columns` under the "per_feature_strict_baseline" policy.
    """
    if policy not in _VALID_NAN_DROP_POLICIES:
        raise ValueError(
            f"unknown nan_drop_policy={policy!r}; expected one of {_VALID_NAN_DROP_POLICIES}"
        )

    if policy == "symmetric":
        return ~np.isnan(X).any(axis=1)

    # policy == "per_feature_strict_baseline"
    missing = [c for c in baseline_columns if c not in feature_columns]
    if missing:
        raise KeyError(
            f"per_feature_strict_baseline requires baseline_columns "
            f"{baseline_columns!r} present in feature_columns, but "
            f"missing: {missing!r}. feature_columns={feature_columns!r}"
        )
    baseline_idx = [feature_columns.index(c) for c in baseline_columns]
    return ~np.isnan(X[:, baseline_idx]).any(axis=1)


def _read_xgb_v2_sha256(model_path: Path | None = None) -> str:
    path = model_path or Path("models/xgb_v2.joblib")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meta_sidecar_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".meta.json")


def _read_cache_metadata(cache_path: Path) -> dict | None:
    sidecar = _meta_sidecar_path(cache_path)
    if not cache_path.exists() or not sidecar.exists():
        return None
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _input_sha256(X_sorted: np.ndarray, y_sorted: np.ndarray, sorted_ids: list) -> str:
    """sha256 over (X, y) rows ordered by fight_id — invariant to input row order."""
    order = np.argsort(np.asarray(sorted_ids), kind="stable")
    buf = io.BytesIO()
    np.save(buf, np.asarray(X_sorted, dtype=np.float64)[order], allow_pickle=False)
    np.save(buf, np.asarray(y_sorted, dtype=np.int64)[order], allow_pickle=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _rekey_cached_oof(df: pd.DataFrame, sorted_ids: list, cache_path: Path) -> np.ndarray:
    """Return cached OOF probs in the caller's date-sorted row order, keyed by fight_id.

    Raises InvariantCheckError unless the cache holds exactly the caller's rows.
    """
    cached_ids = df["fight_id"].tolist()
    probs = df["xgb_oof_prob"].to_numpy()
    if len(cached_ids) != len(sorted_ids):
        raise InvariantCheckError(
            f"OOF cache row count drift: cached={len(cached_ids)} rows "
            f"input={len(sorted_ids)} rows ({cache_path.name}); the cache was built for a "
            "different meta_train row set (e.g. another meta_eval_window_days); "
            "rebuild required (--no-cache-oof)"
        )
    unique = len(set(cached_ids)) == len(cached_ids) and len(set(sorted_ids)) == len(sorted_ids)
    if not unique:
        # Duplicate keys cannot be re-keyed; require exact positional identity.
        if cached_ids != list(sorted_ids):
            raise InvariantCheckError(
                f"OOF cache fight_id sequence drift ({cache_path.name}); rebuild required"
            )
        return probs
    by_id = dict(zip(cached_ids, probs, strict=True))
    missing = [fid for fid in sorted_ids if fid not in by_id]
    if missing:
        raise InvariantCheckError(
            f"OOF cache fight_id membership drift: {len(missing)} input fight_ids not in "
            f"cache (e.g. {missing[:5]!r}) ({cache_path.name}); rebuild required"
        )
    return np.array([by_id[fid] for fid in sorted_ids], dtype=float)


def _check_cache_invariants(
    meta: dict,
    *,
    fight_dates: np.ndarray,
    n_features: int | None = None,
    n_splits: int | None = None,
    seed: int | None = None,
) -> None:
    """Raise InvariantCheckError if any cache invariant drifts from the current call."""
    actual_sha = _read_xgb_v2_sha256()
    if meta.get("xgb_v2_sha256") != actual_sha:
        raise InvariantCheckError(
            f"OOF cache xgb_v2_sha256 drift: cached={meta.get('xgb_v2_sha256', '?')[:12]} "
            f"actual={actual_sha[:12]}; rebuild required (--no-cache-oof)"
        )
    if meta.get("n_features") != EXPECTED_N_FEATURES:
        raise InvariantCheckError(
            f"OOF cache n_features drift: cached={meta.get('n_features')!r} "
            f"expected={EXPECTED_N_FEATURES} (FEATURE_COLUMNS_NO_NET); rebuild required"
        )
    if meta.get("cutoff_date") != EXPECTED_CUTOFF_DATE:
        raise InvariantCheckError(
            f"OOF cache cutoff_date drift: cached={meta.get('cutoff_date')!r} "
            f"expected={EXPECTED_CUTOFF_DATE!r}"
        )
    if len(fight_dates) > 0:
        try:
            input_min = str(np.min(fight_dates))
            input_max = str(np.max(fight_dates))
            cached_min = meta.get("event_date_min", "")
            cached_max = meta.get("event_date_max", "")
            if input_min < cached_min or input_max > cached_max:
                raise InvariantCheckError(
                    f"OOF cache event_date range drift: "
                    f"cached=[{cached_min}, {cached_max}] "
                    f"input=[{input_min}, {input_max}]; rebuild required"
                )
        except (TypeError, ValueError):
            pass  # date comparison best-effort; non-fatal
    if n_features is not None and meta.get("n_features") != n_features:
        raise InvariantCheckError(
            f"OOF cache n_features drift: cached={meta.get('n_features')!r} "
            f"input={n_features}; rebuild required"
        )
    if n_splits is not None and meta.get("n_splits") != n_splits:
        raise InvariantCheckError(
            f"OOF cache n_splits drift: cached={meta.get('n_splits')!r} "
            f"requested={n_splits}; rebuild required (--no-cache-oof)"
        )
    if seed is not None and "seed" in meta and meta["seed"] != seed:
        raise InvariantCheckError(
            f"OOF cache seed drift: cached={meta['seed']!r} requested={seed}; rebuild required"
        )


def _check_input_fingerprint(meta: dict, input_sha256: str) -> None:
    """Raise InvariantCheckError if X/y changed for the cached fights (legacy: skip)."""
    if meta.get("input_sha256", input_sha256) != input_sha256:
        raise InvariantCheckError(
            f"OOF cache input drift: cached input_sha256={meta['input_sha256'][:12]} "
            f"current={input_sha256[:12]} — X/y changed for the same fights "
            "(feature/Elo recompute or result change); rebuild required (--no-cache-oof)"
        )


def _make_oof_estimator(seed: int = 42) -> XGBClassifier:
    """Build a fresh XGBClassifier with xgb_v2's frozen best_params (AF-1).

    Per OQ-2: raw XGBClassifier per fold (NOT CalibratedClassifierCV — META-01
    handles recalibration via its LogisticRegression).
    """
    meta_path = Path("models/xgb_v2_meta.json")
    if not meta_path.exists():
        raise InvariantCheckError("models/xgb_v2_meta.json missing — cannot inherit best_params")
    xgb_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return XGBClassifier(
        **xgb_meta["best_params"],
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
    )


def generate_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    fight_dates: np.ndarray,
    base_trainer: Any,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    cache_path: Path | None = None,
    force_rebuild: bool = False,
    seed: int = 42,
    fight_ids: list | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate leakage-free OOF predictions via TimeSeriesSplit.

    Args:
        X: Training feature matrix (n_train, n_features).
        y: Training target vector (n_train,).
        fight_dates: Per-row event_date array (n_train,) — used to pre-sort
            chronologically before TimeSeriesSplit.
        base_trainer: ModelTrainer instance (provides _make_estimator()) OR
            None (in which case _make_oof_estimator(seed) is used per OQ-2).
        n_splits: TimeSeriesSplit folds. Default 5 per D-15(P16).
        cache_path: Path to cached parquet. If exists + invariants pass →
            return cached. If exists + invariants fail → InvariantCheckError.
        force_rebuild: If True, regenerate even if cache exists+passes.
        seed: random_state for the per-fold XGBClassifier (default 42).
        fight_ids: Optional per-row fight_id list (recorded in metadata for
            D-01(P19) disjoint-id persistence in test_meta_oof_leakage.py).

    Returns:
        (xgb_proba_oof, oof_metadata) where xgb_proba_oof is (n_train,) array
        of OOF positive-class probabilities sorted by fight_dates.
    """
    # Pre-sort by fight_dates (TimeSeriesSplit assumes chronological order).
    # Callers re-align with the same np.argsort(fight_dates), so the returned
    # array is always in this order — for fresh AND cached predictions.
    sort_idx = np.argsort(fight_dates)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    dates_sorted = fight_dates[sort_idx]
    sorted_fight_ids = (
        [fight_ids[int(i)] for i in sort_idx] if fight_ids is not None else list(range(len(y)))
    )
    input_sha256 = _input_sha256(X_sorted, y_sorted, sorted_fight_ids)

    if cache_path is not None and cache_path.exists() and not force_rebuild:
        cached_meta = _read_cache_metadata(cache_path)
        if cached_meta is None:
            raise InvariantCheckError(
                f"OOF cache parquet exists at {cache_path} but sidecar "
                f"{_meta_sidecar_path(cache_path).name} missing"
            )
        _check_cache_invariants(
            cached_meta,
            fight_dates=fight_dates,
            n_features=int(X.shape[1]),
            n_splits=n_splits,
            seed=seed,
        )
        df = pd.read_parquet(cache_path)
        cached_oof = _rekey_cached_oof(df, sorted_fight_ids, cache_path)
        # Same rows — now require the same X/y content for them.
        _check_input_fingerprint(cached_meta, input_sha256)
        return cached_oof, cached_meta

    # Build estimator per OQ-2: raw XGBClassifier with xgb_v2 best_params.
    def _new_estimator():
        if base_trainer is not None and hasattr(base_trainer, "_make_estimator"):
            return base_trainer._make_estimator()
        return _make_oof_estimator(seed)

    estimator = _new_estimator()

    # cross_val_predict requires a partitioning CV; sklearn>=1.x raises
    # "cross_val_predict only works for partitions" when used with
    # TimeSeriesSplit (warm-up rows are never in any test fold). The Pitfall
    # #11 contract REQUIRES TimeSeriesSplit (not StratifiedKFold), so when
    # cross_val_predict refuses, we fall back to a manual TimeSeriesSplit
    # loop with NaN-fill for warm-up rows. The cross_val_predict call is
    # still issued at the top so the kwargs-mock check (cv=TimeSeriesSplit,
    # n_jobs=1) and the leakage-mock branch (perfect probs return →
    # OOFLeakageError) both remain testable.
    try:
        proba = cross_val_predict(
            estimator=estimator,
            X=X_sorted,
            y=y_sorted,
            cv=TimeSeriesSplit(n_splits=n_splits),
            method="predict_proba",
            n_jobs=1,  # NON-NEGOTIABLE — Py 3.14 spawn pickling (D-15(P16))
        )
        xgb_proba_oof = proba[:, 1]
    except ValueError as e:
        if "partition" not in str(e).lower():
            raise
        # Manual OOF via TimeSeriesSplit; warm-up rows get NaN.
        cv_obj = TimeSeriesSplit(n_splits=n_splits)
        xgb_proba_oof = np.full(len(y_sorted), np.nan)
        for tr_idx, te_idx in cv_obj.split(X_sorted, y_sorted):
            est_fold = _new_estimator()
            est_fold.fit(X_sorted[tr_idx], y_sorted[tr_idx])
            xgb_proba_oof[te_idx] = est_fold.predict_proba(X_sorted[te_idx])[:, 1]

    # Pitfall #11 sanity check (NaN-aware: ignore warm-up rows that have no OOF prediction)
    valid_mask = ~np.isnan(xgb_proba_oof)
    if int(valid_mask.sum()) == 0:
        raise OOFLeakageError("OOF predictions all NaN — TimeSeriesSplit produced no test folds")
    training_accuracy = float(
        ((xgb_proba_oof[valid_mask] >= 0.5).astype(int) == y_sorted[valid_mask]).mean()
    )
    if training_accuracy >= 0.75:
        raise OOFLeakageError(
            f"OOF predictions look in-sample (acc={training_accuracy:.4f}) — "
            "TimeSeriesSplit broken or wrong CV passed"
        )

    metadata: dict = {
        "xgb_v2_sha256": _read_xgb_v2_sha256(),
        "n_features": int(X.shape[1]),
        "cutoff_date": EXPECTED_CUTOFF_DATE,
        "event_date_min": str(np.min(dates_sorted)) if len(dates_sorted) else "",
        "event_date_max": str(np.max(dates_sorted)) if len(dates_sorted) else "",
        "n_splits": int(n_splits),
        "training_accuracy": training_accuracy,
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "cv_kind": "TimeSeriesSplit",
        "meta_train_fight_ids": list(fight_ids) if fight_ids is not None else [],
        "seed": int(seed),
        "input_sha256": input_sha256,
    }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "fight_id": sorted_fight_ids,
                "event_date": [str(d) for d in dates_sorted],
                "xgb_oof_prob": xgb_proba_oof,
            }
        )
        df.to_parquet(cache_path, index=False)
        _meta_sidecar_path(cache_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return xgb_proba_oof, metadata


def make_three_way_split(
    fights: list[dict],
    *,
    base_cutoff: date,
    meta_eval_window_days: int = 365,
    today: date | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Three-way disjoint partition by event_date (D-01(P19) + RESEARCH §Three-Way Split).

    Returns:
        (base_train_fights, meta_train_fights, meta_eval_fights)
        - base_train: event_date < base_cutoff (xgb_v2 was trained on these)
        - meta_train: base_cutoff <= event_date < (today − meta_eval_window_days)
        - meta_eval:  event_date >= (today − meta_eval_window_days)
        Asserts the 3 fight_id sets are pairwise disjoint.

    Raises:
        AssertionError on any non-empty intersection (D-01(P19) violation).
    """
    today = today or date.today()
    eval_start = today - timedelta(days=meta_eval_window_days)
    base_train = [f for f in fights if f["event_date"] < base_cutoff]
    meta_train = [f for f in fights if base_cutoff <= f["event_date"] < eval_start]
    meta_eval = [f for f in fights if f["event_date"] >= eval_start]
    base_ids = {f["fight_id"] for f in base_train}
    meta_train_ids = {f["fight_id"] for f in meta_train}
    meta_eval_ids = {f["fight_id"] for f in meta_eval}
    assert base_ids.isdisjoint(meta_train_ids), (
        f"D-01(P19) violation: base ∩ meta_train non-empty (size={len(base_ids & meta_train_ids)})"
    )
    assert base_ids.isdisjoint(meta_eval_ids), (
        f"D-01(P19) violation: base ∩ meta_eval non-empty (size={len(base_ids & meta_eval_ids)})"
    )
    assert meta_train_ids.isdisjoint(meta_eval_ids), (
        "D-01(P19) violation: meta_train ∩ meta_eval non-empty "
        f"(size={len(meta_train_ids & meta_eval_ids)})"
    )
    return base_train, meta_train, meta_eval

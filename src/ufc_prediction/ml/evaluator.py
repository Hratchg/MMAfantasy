"""Model evaluation with 4 metrics per D-09.

Reports Brier score, AUC-ROC, accuracy, and calibration curve data.

Phase 16 / Plan 16-04 / MODEL-01 additions (per CONTEXT.md D-13 + D-16):
  - evaluate_per_slice: 3-slice harness wrapping evaluate_model for the
    most-recent-12mo, most-recent-24mo, and random-15%-temporal slices.

Phase 17 / Plan 17-01 (per CONTEXT.md D-04/D-06(P17)):
  - gate_verdict: applies the v2.1 promotion gate via
    .planning/gate_contract.json (loaded lazily through
    gate_contract.load_gate_contract()). Hardcoded v2.0 module-level
    threshold constants have been removed — thresholds are per-slice
    and live inside the contract.
  - bootstrap_per_slice_ci: BCa 68% CI half-widths on per-fight Brier
    and accuracy arrays per slice (uses scipy.stats.bootstrap; degenerate
    inputs return NaN gracefully per Pitfall B).

evaluate_model is intentionally NOT modified — it is consumed by the
existing `tests/ml/test_evaluator.py` and the `ufc predict eval` CLI
(Gotcha 8 / 16-PATTERNS.md).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

if TYPE_CHECKING:
    from ufc_prediction.ml.gate_contract import GateContract


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate model with 4 metrics per D-09.

    Args:
        model: Trained model with predict_proba and predict methods.
        X_test: Test feature matrix.
        y_test: Test target vector.
        n_bins: Number of bins for calibration curve.

    Returns:
        Dict with:
            brier_score: float
            auc_roc: float
            accuracy: float
            calibration_curve: dict with fraction_of_positives and mean_predicted_value
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    brier = float(brier_score_loss(y_test, probs))
    auc = float(roc_auc_score(y_test, probs))
    acc = float(accuracy_score(y_test, preds))

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test,
        probs,
        n_bins=n_bins,
        strategy="uniform",
    )

    return {
        "brier_score": brier,
        "auc_roc": auc,
        "accuracy": acc,
        "calibration_curve": {
            "fraction_of_positives": fraction_of_positives.tolist(),
            "mean_predicted_value": mean_predicted_value.tolist(),
        },
    }


def format_evaluation_report(
    metrics: dict[str, Any],
    feature_importances: dict[str, Any],
) -> str:
    """Format evaluation results as a Rich-compatible string.

    Args:
        metrics: Evaluation metrics from evaluate_model.
        feature_importances: Feature importance dict from training.

    Returns:
        Formatted string suitable for Rich console output.
    """
    lines: list[str] = []

    lines.append("=== Model Evaluation Report ===\n")

    # Metrics summary
    lines.append("Metrics:")
    lines.append(f"  Brier Score:  {metrics['brier_score']:.4f}")
    lines.append(f"  AUC-ROC:      {metrics['auc_roc']:.4f}")
    lines.append(f"  Accuracy:     {metrics['accuracy']:.4f}")
    lines.append("")

    # Top 10 feature importances
    sorted_features = sorted(
        feature_importances.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    lines.append("Top 10 Feature Importances (Gain):")
    for name, score in sorted_features[:10]:
        lines.append(f"  {name:<40s} {score:.4f}")
    lines.append("")

    # Calibration curve comparison
    if "calibration_curve" in metrics:
        cal = metrics["calibration_curve"]
        lines.append("Calibration Curve (Predicted vs Observed):")
        lines.append(f"  {'Predicted':>12s}  {'Observed':>12s}")
        for pred, obs in zip(
            cal["mean_predicted_value"],
            cal["fraction_of_positives"],
            strict=True,
        ):
            lines.append(f"  {pred:>12.4f}  {obs:>12.4f}")

    return "\n".join(lines)


# ── Phase 16 / Plan 16-04 / MODEL-01 ───────────────────────────────────
# 3-slice gate harness. Per D-04(P17, v2.1): per-slice thresholds load
# from .planning/gate_contract.json via gate_contract.load_gate_contract().
# evaluate_model is the per-slice metric primitive — DO NOT modify.

PER_SLICE_KEYS: tuple[str, str, str] = (
    "most_recent_12mo",
    "most_recent_24mo",
    "random_15pct",
)


def evaluate_per_slice(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fight_dates: np.ndarray,
    *,
    today: date | None = None,
    random_seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """3-slice evaluation harness per CONTEXT.md D-13.

    Slices (order in returned dict matches PER_SLICE_KEYS):
      - most_recent_12mo: fights with event_date >= today − 365 days.
      - most_recent_24mo: fights with event_date >= today − 730 days.
      - random_15pct:     deterministic 15% random sample (seeded RNG).

    Args:
        model: Trained model with predict_proba and predict methods.
        X_test: Test feature matrix (n_test, n_features).
        y_test: Test target vector (n_test,).
        fight_dates: ndarray of `datetime.date` (or any object with date
            comparison semantics) of length n_test.
        today: Reference date for the 12mo / 24mo windows. Defaults to
            `date.today()` when None.
        random_seed: Seed for the random_15pct slice (deterministic per
            CONTEXT.md `<plan_specific_notes>`).

    Returns:
        dict with keys 'most_recent_12mo', 'most_recent_24mo',
        'random_15pct'; each value is the same shape as evaluate_model's
        return (brier_score, auc_roc, accuracy, calibration_curve).

    Note: this function delegates per-slice metric computation to
    evaluate_model. evaluate_model itself is unchanged (Gotcha 8 /
    16-PATTERNS.md back-compat).
    """
    today = today or date.today()
    cutoff_12mo = today - timedelta(days=365)
    cutoff_24mo = today - timedelta(days=730)

    mask_12mo = np.array([d >= cutoff_12mo for d in fight_dates])
    mask_24mo = np.array([d >= cutoff_24mo for d in fight_dates])
    rng = np.random.RandomState(random_seed)
    mask_random = rng.random(len(fight_dates)) < 0.15

    return {
        "most_recent_12mo": evaluate_model(
            model,
            X_test[mask_12mo],
            y_test[mask_12mo],
        ),
        "most_recent_24mo": evaluate_model(
            model,
            X_test[mask_24mo],
            y_test[mask_24mo],
        ),
        "random_15pct": evaluate_model(
            model,
            X_test[mask_random],
            y_test[mask_random],
        ),
    }


def _per_fight_brier_and_accuracy(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-fight (Brier, accuracy_indicator) arrays.

    per_fight_brier[i] = (p_i - y_i)^2  (mean = standard Brier score)
    per_fight_acc[i]   = 1 if (p_i >= 0.5) == y_i else 0
    """
    p = model.predict_proba(X)[:, 1]
    per_fight_brier = (p - y) ** 2
    per_fight_acc = ((p >= 0.5).astype(int) == y).astype(int)
    return per_fight_brier, per_fight_acc


def bootstrap_per_slice_ci(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fight_dates: np.ndarray,
    *,
    today: date | None = None,
    confidence_level: float = 0.68,  # one-sigma per D-05(P17)
    n_resamples: int = 9999,
    rng_seed: int = 42,
    random_seed_for_slice: int = 42,
) -> dict[str, dict[str, float]]:
    """BCa bootstrap CI half-widths for per-slice Brier / Accuracy per D-06(P17).

    Mask construction is bit-equivalent to evaluate_per_slice (same 12mo / 24mo
    cutoffs and same random_seed_for_slice masking convention) so per-slice CI
    widths align with per-slice metrics.

    Args:
        model: Trained model with predict_proba.
        X_test: Test feature matrix.
        y_test: Test target vector.
        fight_dates: ndarray of date-comparable values; length n_test.
        today: Reference date for the 12mo/24mo windows. Defaults to date.today().
        confidence_level: BCa CI width (default 0.68 = one-sigma per D-05(P17)).
        n_resamples: BCa resample count (default 9999 per scipy convention).
        rng_seed: Seed for the bootstrap RNG (deterministic).
        random_seed_for_slice: Seed for the random_15pct slice RNG (matches
            evaluate_per_slice's random_seed default).

    Returns:
        {slice_name: {"brier_ci_half": float, "acc_ci_half": float}}
        for each PER_SLICE_KEYS entry. NaN values mean BCa was degenerate
        on that slice (rare; happens if all per-fight values are identical
        — Pitfall B).
    """
    # scipy is a moderately-heavy import; loaded lazily so non-spike code
    # paths (and the rest of evaluator.py) don't pay the import cost.
    from scipy.stats import bootstrap

    today = today or date.today()
    cutoff_12mo = today - timedelta(days=365)
    cutoff_24mo = today - timedelta(days=730)
    mask_12mo = np.array([d >= cutoff_12mo for d in fight_dates])
    mask_24mo = np.array([d >= cutoff_24mo for d in fight_dates])
    rng_slice = np.random.RandomState(random_seed_for_slice)
    mask_random = rng_slice.random(len(fight_dates)) < 0.15

    results: dict[str, dict[str, float]] = {}
    for slice_name, mask in (
        ("most_recent_12mo", mask_12mo),
        ("most_recent_24mo", mask_24mo),
        ("random_15pct", mask_random),
    ):
        per_brier, per_acc = _per_fight_brier_and_accuracy(
            model,
            X_test[mask],
            y_test[mask],
        )
        rng = np.random.default_rng(rng_seed)
        try:
            ci_brier = bootstrap(
                (per_brier,),
                np.mean,
                method="BCa",
                n_resamples=n_resamples,
                rng=rng,
                confidence_level=confidence_level,
            ).confidence_interval
            half_brier = (ci_brier.high - ci_brier.low) / 2
        except Exception:
            half_brier = float("nan")
        rng = np.random.default_rng(rng_seed)
        try:
            ci_acc = bootstrap(
                (per_acc.astype(float),),
                np.mean,
                method="BCa",
                n_resamples=n_resamples,
                rng=rng,
                confidence_level=confidence_level,
            ).confidence_interval
            half_acc = (ci_acc.high - ci_acc.low) / 2
        except Exception:
            half_acc = float("nan")
        results[slice_name] = {
            "brier_ci_half": float(half_brier),
            "acc_ci_half": float(half_acc),
        }
    return results


def gate_verdict(
    per_slice_median: dict[str, dict[str, Any]],
    contract: GateContract | None = None,
) -> tuple[bool, list[str]]:
    """Apply the v2.1 per-slice promotion gate per CONTEXT.md D-04(P17).

    Args:
        per_slice_median: dict keyed by slice name (PER_SLICE_KEYS); each
            value contains 'brier_score' and 'accuracy'.
        contract: Optional GateContract. If None, lazy-loads from
            .planning/gate_contract.json via load_gate_contract() (cached).

    Returns:
        (passed, failed_slices_or_criteria) — same shape as v2.0.

    Per D-15 + D-17 carry-forward: gate-fail behavior is HALT at gsd-checkpoint.
    """
    # Lazy import per RESEARCH.md Anti-Patterns (no module-level fs I/O).
    from ufc_prediction.ml.gate_contract import load_gate_contract

    contract = contract or load_gate_contract()
    failed: list[str] = []
    for slice_name in PER_SLICE_KEYS:
        metrics = per_slice_median[slice_name]
        brier = float(metrics["brier_score"])
        acc = float(metrics["accuracy"])
        thresholds = contract.per_slice[slice_name]
        if brier > thresholds.brier_max:
            failed.append(f"{slice_name}: brier_score {brier:.4f} > {thresholds.brier_max}")
        if acc < thresholds.accuracy_min:
            failed.append(f"{slice_name}: accuracy {acc:.4f} < {thresholds.accuracy_min}")
    return (len(failed) == 0, failed)

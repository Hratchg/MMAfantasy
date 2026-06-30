"""META-V22-04: per-feature std + direction stability across seeds.

Per CONTEXT D-04 + RESEARCH §"Coefficient Stability Report" (lines 501-545):
validate that the rich Level-1 set isn't noise-fitting by reporting per-feature
std of the post-PolynomialFeatures-expanded coefficients across 5 seeds plus
per-feature sign agreement (fraction of seeds agreeing with the median sign).

Net-new per Phase 26 — Phase 19 spike skipped this on ship-fail.

Exports
-------
- coefficient_stability_report(per_seed_meta, feature_names) -> dict
- write_coefficient_stability_json(report, out_path) -> None
- format_coefficient_stability_markdown(report) -> str
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ufc_prediction.ml.meta_learner import MetaLearnerLogistic


def coefficient_stability_report(
    per_seed_meta: dict[int, MetaLearnerLogistic],
    feature_names: list[str],
) -> dict[str, Any]:
    """Per-feature std + per-feature direction stability across seeds.

    Args:
        per_seed_meta: dict mapping seed (int) → fitted MetaLearnerLogistic instance.
            Each instance's pipeline.named_steps["clf"].coef_ is read.
        feature_names: post-PolynomialFeatures-expanded feature names (length must
            equal coef_[0].size; caller supplies via
            ``pipeline.named_steps["poly"].get_feature_names_out(...)``).

    Returns:
        {
          "feature_names": [...],
          "per_feature_std": np.ndarray (n_features,),
          "per_feature_sign_agreement": np.ndarray (n_features,) in [0, 1],
          "median_abs_coef": np.ndarray (n_features,),
          "n_seeds": int,
        }
    """
    coefs = np.array(
        [m.pipeline.named_steps["clf"].coef_[0] for m in per_seed_meta.values()]
    )  # shape (n_seeds, n_features)
    per_feature_std = coefs.std(axis=0)
    median_sign = np.sign(np.median(coefs, axis=0))
    per_feature_sign_agreement = (np.sign(coefs) == median_sign).mean(axis=0)
    return {
        "feature_names": list(feature_names),
        "per_feature_std": per_feature_std,
        "per_feature_sign_agreement": per_feature_sign_agreement,
        "median_abs_coef": np.abs(np.median(coefs, axis=0)),
        "n_seeds": len(per_seed_meta),
    }


def _convert_ndarray(obj: Any) -> Any:
    """Recursively convert np.ndarray → list for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_ndarray(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_ndarray(item) for item in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def write_coefficient_stability_json(report: dict[str, Any], out_path: Path) -> None:
    """Serialize the report (with np.ndarray → list) and write to ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_convert_ndarray(report), indent=2),
        encoding="utf-8",
    )


def format_coefficient_stability_markdown(report: dict[str, Any]) -> str:
    """Return a markdown table with one header row + N data rows (N = n_features).

    Columns: feature_name | std | sign_agreement | median_abs_coef
    """
    lines: list[str] = []
    lines.append("| feature_name | std | sign_agreement | median_abs_coef |")
    lines.append("|---|---|---|---|")
    names = list(report["feature_names"])
    std_arr = np.asarray(report["per_feature_std"])
    sign_arr = np.asarray(report["per_feature_sign_agreement"])
    median_arr = np.asarray(report["median_abs_coef"])
    for i, name in enumerate(names):
        lines.append(
            f"| {name} | {float(std_arr[i]):.4f} | "
            f"{float(sign_arr[i]):.3f} | {float(median_arr[i]):.4f} |"
        )
    return "\n".join(lines)

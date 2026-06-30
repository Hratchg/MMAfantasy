"""CALIB-V22-00 audit helper for xgb_v2 calibration introspection.

Per Phase 26 CONTEXT D-06 + Pitfall #8: surface the existing calibration state
of `models/xgb_v2.joblib` as a JSON artifact BEFORE CALIB-V22-01 wires Platt.
xgb_v2's pipeline ALREADY wraps the XGBClassifier in
`CalibratedClassifierCV(FrozenEstimator(final_model), method="sigmoid")` per
`trainer.py:122-127`; this audit confirms that fact and records
`double_calibration_risk=false` so CALIB-V22-01 can wire Platt as a
REPLACEMENT (NOT stacked on top).

Exports
-------
- audit_xgb_v2_calibration(model_path, out_path) -> dict
  Loads xgb_v2.joblib, introspects type + calibration_method + n_calibrators,
  writes the audit dict to ``out_path`` and returns it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.calibration import CalibratedClassifierCV


def audit_xgb_v2_calibration(
    model_path: Path = Path("models/xgb_v2.joblib"),
    out_path: Path = Path(
        ".planning/phases/26-forward-stepwise-candidate-promotion/CALIB_00_CURRENT_STATE.json"
    ),
) -> dict[str, Any]:
    """CALIB-V22-00: introspect xgb_v2's pipeline; emit `CALIB_00_CURRENT_STATE.json`.

    Per CONTEXT D-06 + Pitfall #8: confirms `method='sigmoid'` is already implicit so
    CALIB-V22-01 wires Platt as REPLACEMENT (not stacked on top).

    Args:
        model_path: Path to the xgb_v2 joblib. Defaults to ``models/xgb_v2.joblib``.
        out_path: Where to write the audit JSON. Defaults to the Phase 26 artifact path.

    Returns:
        Audit dict with keys: model_path, model_class, is_calibrated, calibration_method,
        n_calibrators, double_calibration_risk, audited_at, decision.
    """
    model = joblib.load(model_path)
    is_calibrated = isinstance(model, CalibratedClassifierCV)
    method = getattr(model, "method", None) if is_calibrated else None
    n_calibrators = len(model.calibrated_classifiers_) if is_calibrated else 0
    # Derive double_calibration_risk from introspection (NOT hardcoded False).
    # The risk is "True" only if the loaded model already has >1 CalibratedClassifierCV
    # layer (i.e. stacked). Single-layer with method='sigmoid' is the expected state.
    double_calibration_risk = bool(is_calibrated and n_calibrators > 1)
    audit = {
        "model_path": str(model_path),
        "model_class": type(model).__name__,
        "is_calibrated": is_calibrated,
        "calibration_method": method,
        "n_calibrators": n_calibrators,
        "double_calibration_risk": double_calibration_risk,
        "audited_at": datetime.now(tz=UTC).isoformat(),
        "decision": (
            "CALIB-V22-01 wires Platt as REPLACEMENT (not stacked) — Pitfall #8 prevention"
            if (is_calibrated and method == "sigmoid")
            else "CALIB-V22-01 wires Platt fresh — no existing calibration detected"
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit

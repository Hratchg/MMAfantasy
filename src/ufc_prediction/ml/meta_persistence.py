"""Meta-learner persistence layer: sibling to persistence.py.

Per CONTEXT.md D-09(P15) byte-identity carry-forward + Pitfall #15:
extending persistence.save_model with optional kwargs would silently break
v2.0 model loaders (the feature_columns length check in predictor.py). NEW
SIBLING module, separate JSON schema, separate directory (`models/meta/`).

Schema (META-04 + discretionary meta_oof_parquet_sha256 per OQ-4):
    meta_version: "v1" | "v2"
    meta_kind: "logistic" | "nn"
    meta_feature_columns: ["xgb_oof_prob", "elo_prob", "closing_prob_diff"]
    meta_input_distribution_hash: sha256 of np.save(BytesIO, [X, y, oof])
    meta_oof_parquet_sha256: sha256 of cached OOF parquet (auditor traceability)
    base_model_version: "v2"
    base_model_sha256: 6e7641...0a99
    meta_learner_brier_delta_vs_logistic: 0.0 (META-01) | Δ vs META-01 (META-02)
    best_params: {"C": 1.0, ...}
    metrics: {"per_slice": {...}, "median_brier_overall": ..., "monotonicity_score": ...}
    trained_at: ISO-8601 timestamp
    trained_by_script: "scripts/train_meta_v1.py" | "scripts/train_meta_v2.py"
    phase: "19"
    sklearn_version: str
    python_version: str
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import sklearn

DEFAULT_META_DIR: str = "models/meta"

# Phase 26 (Plan 26-04): version-discovery helper for predictor.py LIVE-03 boundary.
# Matches PROMOTED meta_v{N}.joblib files ONLY (NOT meta_v{N}_candidate.joblib).
# Per D-10 (CONTEXT.md): '_candidate' artifacts are ignored — only files committing
# to the promoted-vN naming convention count as discoverable versions.
_META_VERSION_RE: re.Pattern[str] = re.compile(r"^meta_v(\d+)\.joblib$")


def get_latest_meta_version(meta_dir: str = DEFAULT_META_DIR) -> str | None:
    """Return the highest 'vN' meta version present in `meta_dir`, or None if absent.

    Phase 26 Plan 26-04 (RESEARCH §Runtime State Inventory Pitfall #5): replaces the
    hardcoded `meta_v1.joblib` lookup in `predictor.py` so the predictor auto-picks
    whichever meta version is promoted (v1 or v2 today, vN tomorrow).

    Per D-10 (CONTEXT.md): `_candidate` artifacts are IGNORED — only promoted vN
    files count. Examples:
        - dir=[]                                          -> None
        - dir=[meta_v1.joblib]                            -> "v1"
        - dir=[meta_v1.joblib, meta_v2.joblib]            -> "v2"
        - dir=[meta_v1.joblib, meta_v2_candidate.joblib]  -> "v1"  (candidate ignored)

    Args:
        meta_dir: Path to the meta directory. Defaults to `DEFAULT_META_DIR`.

    Returns:
        Highest-numbered promoted version string (e.g., "v2") or None if directory
        is empty / missing / contains only candidate artifacts.
    """
    p = Path(meta_dir)
    if not p.exists():
        return None
    versions: list[int] = []
    for entry in p.iterdir():
        m = _META_VERSION_RE.match(entry.name)
        if m:
            versions.append(int(m.group(1)))
    if not versions:
        return None
    return f"v{max(versions)}"


SUPPORTED_META_KINDS: frozenset[str] = frozenset({"logistic", "nn"})
SCHEMA_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "meta_version",
        "meta_kind",
        "meta_feature_columns",
        "meta_input_distribution_hash",
        "base_model_version",
        "base_model_sha256",
        "meta_learner_brier_delta_vs_logistic",
    }
)


class MetaSchemaError(ValueError):
    """Raised when meta_v{X}_meta.json is missing required fields or contains invalid values."""


def _convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization.

    Verbatim copy of persistence._convert_numpy_types to keep meta_persistence.py
    self-contained (Pitfall #15 spirit — no cross-module coupling).
    """
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def compute_meta_input_distribution_hash(
    X_meta_train: np.ndarray,
    y_meta_train: np.ndarray,
    oof_proba: np.ndarray,
) -> str:
    """Compute stable sha256 over the meta-train input distribution (OQ-4).

    Uses np.save into BytesIO so the hash is invariant across pyarrow / parquet
    writer versions. The 3 arrays are saved in deterministic order.
    """
    buf = io.BytesIO()
    np.save(buf, np.asarray(X_meta_train), allow_pickle=False)
    np.save(buf, np.asarray(y_meta_train), allow_pickle=False)
    np.save(buf, np.asarray(oof_proba), allow_pickle=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def save_meta_model(
    meta_model,
    *,
    meta_kind: Literal["logistic", "nn"],
    meta_version: str,
    base_model_version: str,
    base_model_sha256: str,
    meta_feature_columns: list[str],
    meta_input_distribution_hash: str,
    meta_oof_parquet_sha256: str,
    meta_learner_brier_delta_vs_logistic: float,
    best_params: dict,
    metrics: dict,
    meta_dir: str = DEFAULT_META_DIR,
    trained_by_script: str = "scripts/train_meta_v1.py",
    phase: str = "19",
) -> tuple[Path, Path]:
    """Save meta model + meta JSON.

    Creates:
        meta_dir/meta_{meta_version}.joblib
        meta_dir/meta_{meta_version}_meta.json

    Validates meta_kind ∈ SUPPORTED_META_KINDS at save time (fail-fast).

    Returns:
        (model_path, meta_path)

    Raises:
        MetaSchemaError if meta_kind is invalid.
    """
    if meta_kind not in SUPPORTED_META_KINDS:
        raise MetaSchemaError(f"meta_kind={meta_kind!r} not in {sorted(SUPPORTED_META_KINDS)}")

    dir_path = Path(meta_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    model_path = dir_path / f"meta_{meta_version}.joblib"
    meta_path = dir_path / f"meta_{meta_version}_meta.json"
    joblib.dump(meta_model, model_path)

    metadata: dict = {
        "meta_version": meta_version,
        "meta_kind": meta_kind,
        "meta_feature_columns": list(meta_feature_columns),
        "meta_input_distribution_hash": meta_input_distribution_hash,
        "meta_oof_parquet_sha256": meta_oof_parquet_sha256,
        "base_model_version": base_model_version,
        "base_model_sha256": base_model_sha256,
        "meta_learner_brier_delta_vs_logistic": float(meta_learner_brier_delta_vs_logistic),
        "best_params": _convert_numpy_types(best_params),
        "metrics": _convert_numpy_types(metrics),
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "trained_by_script": trained_by_script,
        "phase": phase,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, meta_path


def _validate_schema(metadata: dict) -> None:
    """Raise MetaSchemaError if metadata violates schema."""
    missing = SCHEMA_REQUIRED_FIELDS - set(metadata.keys())
    if missing:
        raise MetaSchemaError(f"meta JSON missing required fields: {sorted(missing)}")
    if metadata["meta_kind"] not in SUPPORTED_META_KINDS:
        raise MetaSchemaError(
            f"meta_kind={metadata['meta_kind']!r} not in {sorted(SUPPORTED_META_KINDS)}"
        )


def load_meta_model(
    meta_dir: str = DEFAULT_META_DIR,
    version: str = "v1",
) -> tuple[object, dict]:
    """Load meta model + meta JSON.

    Returns:
        (meta_model_instance, meta_metadata_dict)

    Raises:
        FileNotFoundError if joblib or meta JSON missing.
        MetaSchemaError if schema validation fails.
    """
    dir_path = Path(meta_dir)
    model_path = dir_path / f"meta_{version}.joblib"
    meta_path = dir_path / f"meta_{version}_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Meta model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta metadata not found: {meta_path}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    _validate_schema(metadata)
    model = joblib.load(model_path)
    return model, metadata

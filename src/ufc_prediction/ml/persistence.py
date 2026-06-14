"""Model persistence layer: save/load with joblib + JSON metadata.

Per D-07: Trained model persisted as joblib file in models/ directory
with version metadata JSON.
Per T-10-04: Only load models from local models/ directory.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import sklearn
import xgboost


def save_model(
    model,
    metrics: dict,
    feature_columns: list[str],
    best_params: dict,
    model_dir: str,
    version: str = "v1",
    cutoff_date: str = "2023-01-01",
    n_training_fights: int = 0,
    n_test_fights: int = 0,
    *,
    extra_meta: dict | None = None,
) -> Path:
    """Save trained model and metadata to model_dir.

    Creates:
        - model_dir/xgb_{version}.joblib  (serialized model)
        - model_dir/xgb_{version}_meta.json  (metadata)

    Args:
        model: Trained CalibratedClassifierCV model.
        metrics: Evaluation metrics dict.
        feature_columns: Ordered list of feature column names.
        best_params: Best hyperparameters from Optuna.
        model_dir: Directory to save model files.
        version: Model version tag (e.g., "v1").
        cutoff_date: Temporal split date used for training.
        n_training_fights: Number of fights in training set.
        n_test_fights: Number of fights in test set.
        extra_meta: Optional dict merged into the metadata JSON. Used by
            Phase 18 spike to record ``base_model_kind`` (variant identity)
            and ``applied_gate_contract_formula_hash`` per Open Q3 RESOLVED.
            Reserved keys (version, feature_columns, n_features, best_params,
            metrics, cutoff_date, n_training_fights, n_test_fights, trained_at,
            xgboost_version, sklearn_version, python_version) cannot be
            overridden — those are computed by save_model itself.

    Returns:
        Path to the saved model file.
    """
    dir_path = Path(model_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    model_path = dir_path / f"xgb_{version}.joblib"
    meta_path = dir_path / f"xgb_{version}_meta.json"

    # Save model
    joblib.dump(model, model_path)

    # Convert numpy types to Python native for JSON serialization
    serializable_params = _convert_numpy_types(best_params)
    serializable_metrics = _convert_numpy_types(metrics)

    # Save metadata
    metadata: dict = {
        "version": version,
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
        "feature_columns": list(feature_columns),
        "n_features": int(len(feature_columns)),  # HOUSE-01 (Pitfall #12 corroboration)
        "best_params": serializable_params,
        "metrics": serializable_metrics,
        "cutoff_date": cutoff_date,
        "n_training_fights": int(n_training_fights),
        "n_test_fights": int(n_test_fights),
    }
    # Phase 18 NET-V2-01: merge optional extra_meta (e.g., base_model_kind).
    # Reserved keys CANNOT be overridden — save_model is the source of truth
    # for them.
    if extra_meta:
        reserved = set(metadata.keys())
        for k, v in extra_meta.items():
            if k in reserved:
                continue
            metadata[k] = v
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return model_path


def load_model(model_dir: str, version: str):
    """Load a trained model from model_dir.

    Per T-10-04: Only loads from local models/ directory.

    Args:
        model_dir: Directory containing model files.
        version: Model version tag (e.g., "v1").

    Returns:
        The deserialized CalibratedClassifierCV model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = Path(model_dir) / f"xgb_{version}.joblib"
    if not model_path.exists():
        msg = f"Model file not found: {model_path}"
        raise FileNotFoundError(msg)

    return joblib.load(model_path)


def load_metadata(model_dir: str, version: str) -> dict:
    """Load model metadata JSON.

    Args:
        model_dir: Directory containing model files.
        version: Model version tag (e.g., "v1").

    Returns:
        Metadata dictionary.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
    """
    meta_path = Path(model_dir) / f"xgb_{version}_meta.json"
    if not meta_path.exists():
        msg = f"Metadata file not found: {meta_path}"
        raise FileNotFoundError(msg)

    return json.loads(meta_path.read_text(encoding="utf-8"))


def get_latest_version(model_dir: str) -> str:
    """Scan model_dir for xgb_v*.joblib files and return highest version.

    Args:
        model_dir: Directory to scan.

    Returns:
        Version string (e.g., "v2"). Returns "v1" if no files found.
    """
    dir_path = Path(model_dir)
    if not dir_path.exists():
        return "v1"

    versions: list[int] = []
    pattern = re.compile(r"^xgb_v(\d+)\.joblib$")

    for f in dir_path.iterdir():
        match = pattern.match(f.name)
        if match:
            versions.append(int(match.group(1)))

    if not versions:
        return "v1"

    return f"v{max(versions)}"


def _convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    import numpy as np

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


_FEATURE_COLUMNS_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def save_contract_json(
    model_dir: str,
    version: str,
    *,
    schema_version: str = "1.0.0",
    gate_contract_ref: str,
    feature_columns_hash: str,
    min_partner_version_supported: str = "1.0.0",
    deprecation_policy: str = "N >= 2 minor versions",
) -> Path:
    """Write *-contract.json sibling artifact next to *.joblib and *_meta.json.

    Per Phase 25 D-05: *-contract.json is a SIBLING (NOT a sub-object of meta).
    Per Phase 25 D-09(P15) discipline: meta.json is NEVER read, mutated, or overwritten.
    Per AUDIT-01: model_artifact_sha256 is computed FRESH from disk on every emit;
    the model joblib is never touched.

    WR-07 (Phase 25 review-fix): validates partner-trust inputs before write:
      - feature_columns_hash must be a 64-char lowercase hex SHA-256.
      - gate_contract_ref must point at an existing file under model_dir's
        repo (resolved from model_dir's parent — the repo root is one level
        above ``models/``). A typo or stale literal would otherwise silently
        produce a contract artifact partners cannot verify.

    Returns:
        Path to the written contract JSON.

    Raises:
        FileNotFoundError: model joblib or gate_contract_ref does not exist.
        ValueError: feature_columns_hash is not 64-char lowercase hex, or
            gate_contract_ref is empty.
    """
    import hashlib  # defer import — only this helper needs hashlib;
    # keeps cold-import cost out of load_model paths.

    if not feature_columns_hash or not _FEATURE_COLUMNS_HASH_RE.match(
        feature_columns_hash,
    ):
        msg = (
            "feature_columns_hash must be a 64-char lowercase hex SHA-256; "
            f"got {feature_columns_hash!r}"
        )
        raise ValueError(msg)

    if not gate_contract_ref:
        msg = f"gate_contract_ref must be non-empty; got {gate_contract_ref!r}"
        raise ValueError(msg)

    dir_path = Path(model_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    # gate_contract_ref is documented as a repo-relative path (e.g.
    # ".planning/gate_contract_v2.2.json"). Resolve against the parent of
    # model_dir (the repo root, by convention) and require existence so a
    # typo or stale literal is caught at emit time, not by partners.
    repo_root = dir_path.resolve().parent
    gate_path = (repo_root / gate_contract_ref).resolve()
    if not gate_path.is_file():
        msg = (
            f"gate_contract_ref does not resolve to an existing file: "
            f"{gate_contract_ref!r} (resolved to {gate_path})"
        )
        raise FileNotFoundError(msg)

    model_path = dir_path / f"xgb_{version}.joblib"
    contract_path = dir_path / f"xgb_{version}-contract.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"save_contract_json: model joblib not found: {model_path}"
        )

    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    contract = {
        "schema_version": schema_version,
        "gate_contract_ref": gate_contract_ref,
        "feature_columns_hash": feature_columns_hash,
        "min_partner_version_supported": min_partner_version_supported,
        "deprecation_policy": deprecation_policy,
        "model_artifact_sha256": model_sha,
        "created_at": datetime.now(tz=UTC).date().isoformat(),
    }
    contract_path.write_text(
        json.dumps(contract, indent=2) + "\n",
        encoding="utf-8",
    )
    return contract_path

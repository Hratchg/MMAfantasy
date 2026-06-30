"""Unit tests for save_contract_json (Phase 25 Plan 25-02).

Covers:
  - Structural fields (D-05 — 7-key contract)
  - D-09(P15) discipline — xgb_v2_meta.json NOT mutated by helper
  - model_artifact_sha256 equals fresh sha256 of joblib
  - WR-07 input validation: feature_columns_hash hex regex + gate_contract_ref
    file existence (Phase 25 review-fix).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ufc_prediction.ml.persistence import save_contract_json

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
META_PATH = MODELS_DIR / "xgb_v2_meta.json"
JOBLIB_PATH = MODELS_DIR / "xgb_v2.joblib"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_synth_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Build a minimal repo layout under tmp_path so that
    save_contract_json's gate_contract_ref file-existence check passes.

    Layout produced:
        tmp_path/
            models/
                xgb_v2.joblib       (copied from real repo)
            .planning/
                gate_contract_v2.2.json   (placeholder; content unused by helper)

    Returns:
        (model_dir, gate_contract_path) — both as Path objects.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "xgb_v2.joblib").write_bytes(JOBLIB_PATH.read_bytes())
    planning_dir = tmp_path / ".planning"
    planning_dir.mkdir()
    gate_path = planning_dir / "gate_contract_v2.2.json"
    gate_path.write_text('{"placeholder": true}\n', encoding="utf-8")
    return model_dir, gate_path


class TestSaveContractJson:
    def test_meta_json_byte_identity_preserved(self, tmp_path):
        """D-09(P15): save_contract_json does NOT mutate *_meta.json."""
        model_dir, _ = _seed_synth_repo(tmp_path)
        tmp_meta = model_dir / "xgb_v2_meta.json"
        tmp_meta.write_bytes(META_PATH.read_bytes())
        sha_meta_before = _sha(tmp_meta)

        save_contract_json(
            model_dir=str(model_dir),
            version="v2",
            gate_contract_ref=".planning/gate_contract_v2.2.json",
            feature_columns_hash="a" * 64,
        )

        sha_meta_after = _sha(tmp_meta)
        assert sha_meta_before == sha_meta_after, (
            "xgb_v2_meta.json was mutated — D-09(P15) discipline violated"
        )

    def test_contract_has_seven_fields(self, tmp_path):
        model_dir, _ = _seed_synth_repo(tmp_path)
        path = save_contract_json(
            model_dir=str(model_dir),
            version="v2",
            gate_contract_ref=".planning/gate_contract_v2.2.json",
            feature_columns_hash="a" * 64,
        )
        contract = json.loads(path.read_text())
        expected_keys = {
            "schema_version",
            "gate_contract_ref",
            "feature_columns_hash",
            "min_partner_version_supported",
            "deprecation_policy",
            "model_artifact_sha256",
            "created_at",
        }
        assert set(contract.keys()) == expected_keys

    def test_model_artifact_sha_matches_joblib(self, tmp_path):
        model_dir, _ = _seed_synth_repo(tmp_path)
        path = save_contract_json(
            model_dir=str(model_dir),
            version="v2",
            gate_contract_ref=".planning/gate_contract_v2.2.json",
            feature_columns_hash="b" * 64,
        )
        contract = json.loads(path.read_text())
        assert contract["model_artifact_sha256"] == _sha(
            model_dir / "xgb_v2.joblib",
        )

    def test_missing_joblib_raises(self, tmp_path):
        # Seed gate contract file so the validator passes; joblib is absent.
        _, _ = _seed_synth_repo(tmp_path)
        (tmp_path / "models" / "xgb_v2.joblib").unlink()
        with pytest.raises(FileNotFoundError):
            save_contract_json(
                model_dir=str(tmp_path / "models"),
                version="v2",
                gate_contract_ref=".planning/gate_contract_v2.2.json",
                feature_columns_hash="c" * 64,
            )

    def test_default_kwargs_per_research_pattern_4(self, tmp_path):
        model_dir, _ = _seed_synth_repo(tmp_path)
        path = save_contract_json(
            model_dir=str(model_dir),
            version="v2",
            gate_contract_ref=".planning/gate_contract_v2.2.json",
            feature_columns_hash="d" * 64,
        )
        contract = json.loads(path.read_text())
        assert contract["schema_version"] == "1.0.0"
        assert contract["min_partner_version_supported"] == "1.0.0"
        assert contract["deprecation_policy"] == "N >= 2 minor versions"

    # ── WR-07 validation tests (Phase 25 review-fix) ──────────────────────

    def test_rejects_non_hex_feature_columns_hash(self, tmp_path):
        model_dir, _ = _seed_synth_repo(tmp_path)
        with pytest.raises(ValueError, match="64-char lowercase hex"):
            save_contract_json(
                model_dir=str(model_dir),
                version="v2",
                gate_contract_ref=".planning/gate_contract_v2.2.json",
                feature_columns_hash="not-a-sha-256",
            )

    def test_rejects_uppercase_feature_columns_hash(self, tmp_path):
        model_dir, _ = _seed_synth_repo(tmp_path)
        with pytest.raises(ValueError, match="64-char lowercase hex"):
            save_contract_json(
                model_dir=str(model_dir),
                version="v2",
                gate_contract_ref=".planning/gate_contract_v2.2.json",
                feature_columns_hash="A" * 64,  # uppercase rejected
            )

    def test_rejects_short_feature_columns_hash(self, tmp_path):
        model_dir, _ = _seed_synth_repo(tmp_path)
        with pytest.raises(ValueError, match="64-char lowercase hex"):
            save_contract_json(
                model_dir=str(model_dir),
                version="v2",
                gate_contract_ref=".planning/gate_contract_v2.2.json",
                feature_columns_hash="a" * 63,  # one char short
            )

    def test_rejects_empty_feature_columns_hash(self, tmp_path):
        model_dir, _ = _seed_synth_repo(tmp_path)
        with pytest.raises(ValueError, match="64-char lowercase hex"):
            save_contract_json(
                model_dir=str(model_dir),
                version="v2",
                gate_contract_ref=".planning/gate_contract_v2.2.json",
                feature_columns_hash="",
            )

    def test_rejects_empty_gate_contract_ref(self, tmp_path):
        model_dir, _ = _seed_synth_repo(tmp_path)
        with pytest.raises(ValueError, match="non-empty"):
            save_contract_json(
                model_dir=str(model_dir),
                version="v2",
                gate_contract_ref="",
                feature_columns_hash="a" * 64,
            )

    def test_rejects_missing_gate_contract_file(self, tmp_path):
        model_dir, _ = _seed_synth_repo(tmp_path)
        with pytest.raises(FileNotFoundError, match="gate_contract_ref"):
            save_contract_json(
                model_dir=str(model_dir),
                version="v2",
                gate_contract_ref=".planning/does_not_exist.json",
                feature_columns_hash="a" * 64,
            )

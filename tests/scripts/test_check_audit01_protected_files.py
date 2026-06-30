"""Phase 50 DX-V26-04 — tests for the AUDIT-01 protected-file guard.

Verifies the script's three exit paths:
  - clean (no protected files staged) → exit 0
  - blocked (protected files staged, no override) → exit 1
  - override (protected files staged with AUDIT01_OVERRIDE=1) → exit 0

Locks the PROTECTED_FILES set against drift from CONTRIBUTING.md L62-74.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "check_audit01_protected_files.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "check_audit01_protected_files",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["check_audit01_protected_files"] = m
    spec.loader.exec_module(m)
    return m


def test_protected_files_set_matches_contributing_md(mod) -> None:
    """The protected set must enumerate exactly the CONTRIBUTING.md L62-74 list.

    Any drift between this assertion and the markdown means one source
    has changed without the other — operator must reconcile.
    """
    expected = {
        "models/xgb_v2.joblib",
        "models/meta/meta_v2.joblib",
        "models/meta/meta_v2_dedup.joblib",
        "src/ufc_prediction/ml/predictor.py",
        "src/ufc_prediction/ml/feature_matrix.py",
        "src/ufc_prediction/ml/persistence.py",
        "src/ufc_prediction/ml/train.py",
        "scripts/spike_noise_floor_v22.py",
        "scripts/spike_noise_floor_v23.py",
        "scripts/train_meta_v22.py",
        "src/ufc_prediction/contracts/predictor.schema.v1.0.0.json",
    }
    assert set(mod.PROTECTED_FILES) == expected


def test_clean_path_returns_zero(mod) -> None:
    assert mod.main(["README.md", "CONTRIBUTING.md"]) == 0


def test_no_args_returns_zero(mod) -> None:
    assert mod.main([]) == 0


def test_protected_file_blocks(mod, capsys, monkeypatch) -> None:
    monkeypatch.delenv("AUDIT01_OVERRIDE", raising=False)
    rc = mod.main(["src/ufc_prediction/ml/predictor.py"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "AUDIT-01 protected files staged — commit BLOCKED" in captured.err
    assert "src/ufc_prediction/ml/predictor.py" in captured.err
    assert "AUDIT01_OVERRIDE=1" in captured.err


def test_override_env_var_allows(mod, capsys, monkeypatch) -> None:
    monkeypatch.setenv("AUDIT01_OVERRIDE", "1")
    rc = mod.main(["src/ufc_prediction/ml/predictor.py"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "override — proceeding" in captured.err


def test_mixed_clean_and_protected_blocks(mod, capsys, monkeypatch) -> None:
    monkeypatch.delenv("AUDIT01_OVERRIDE", raising=False)
    rc = mod.main(["README.md", "models/xgb_v2.joblib"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "models/xgb_v2.joblib" in captured.err
    assert "README.md" not in captured.err  # only violations listed


def test_path_normalization_handles_dotslash_prefix(mod, monkeypatch) -> None:
    monkeypatch.delenv("AUDIT01_OVERRIDE", raising=False)
    # `./models/xgb_v2.joblib` must normalize to `models/xgb_v2.joblib`
    # and trip the guard. Otherwise an attacker could bypass with a
    # cosmetic prefix.
    rc = mod.main(["./models/xgb_v2.joblib"])
    assert rc == 1


def test_override_value_other_than_1_does_not_bypass(mod, capsys, monkeypatch) -> None:
    """Only literal '1' counts as override. `=true`, `=yes`, empty all block."""
    for value in ("true", "yes", "0", ""):
        monkeypatch.setenv("AUDIT01_OVERRIDE", value)
        rc = mod.main(["src/ufc_prediction/ml/predictor.py"])
        assert rc == 1, f"value={value!r} should NOT bypass the guard"

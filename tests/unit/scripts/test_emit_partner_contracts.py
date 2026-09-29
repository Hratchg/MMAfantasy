"""Guard `scripts/emit_partner_contracts.py`'s AUDIT-01 xgb_v2 SHA check.

The script previously hardcoded the pre-2026-07-06 `6e7641…` xgb_v2 SHA, so
every emit run raised "AUDIT-01 drift!" against the re-baselined frozen model.
It now re-exports `ufc_prediction.cli.predict.EXPECTED_XGB_V2_SHA`; these tests
pin that it agrees with the other canonical copy
(`scripts/spike_noise_floor_v23.py::EXPECTED_XGB_V2_SHA`) and with the bytes
of the committed `models/xgb_v2.joblib`.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SCRIPT_PATH: Path = REPO_ROOT / "scripts" / "emit_partner_contracts.py"
SPIKE_V23_PATH: Path = REPO_ROOT / "scripts" / "spike_noise_floor_v23.py"

_spec = importlib.util.spec_from_file_location("emit_partner_contracts", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
emit_mod = importlib.util.module_from_spec(_spec)
sys.modules["emit_partner_contracts"] = emit_mod
_spec.loader.exec_module(emit_mod)


def _spike_v23_expected_sha() -> str:
    """Read spike_v23's EXPECTED_XGB_V2_SHA literal without importing it."""
    tree = ast.parse(SPIKE_V23_PATH.read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "EXPECTED_XGB_V2_SHA"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise AssertionError("EXPECTED_XGB_V2_SHA not found in spike_noise_floor_v23.py")


def test_canonical_sha_matches_spike_v23() -> None:
    assert emit_mod.XGB_V2_CANONICAL_SHA == _spike_v23_expected_sha()


def test_check_passes_on_committed_frozen_model() -> None:
    assert emit_mod._check_audit01_xgb_v2_sha(REPO_ROOT) == emit_mod.XGB_V2_CANONICAL_SHA


def test_check_raises_on_drift(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    fake = tmp_path / "models" / "xgb_v2.joblib"
    fake.write_bytes(b"not the frozen model")
    assert hashlib.sha256(fake.read_bytes()).hexdigest() != emit_mod.XGB_V2_CANONICAL_SHA
    with pytest.raises(RuntimeError, match="AUDIT-01 drift"):
        emit_mod._check_audit01_xgb_v2_sha(tmp_path)

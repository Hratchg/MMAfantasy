"""Phase 72 METH-V261-01 — tests for the AUDIT-01 SOFT-PROTECT extension.

Verifies the new soft-protect path for refit-baseline siblings per
gate_methodology_v2.6.md §7.3:

  - matching file staged, no opt-in → exit 1 (BLOCKED)
  - matching file staged, GSD_REFIT_REEMIT=1 → exit 0 (WARN-but-allow)
  - matching file staged, commit message token → exit 0 (WARN-but-allow)
  - hard-protect path still exits 1 without override (regression check)
  - hard-protect path still exits 0 with AUDIT01_OVERRIDE=1 (regression check)
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
        "check_audit01_protected_files_soft",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["check_audit01_protected_files_soft"] = m
    spec.loader.exec_module(m)
    return m


# ── Soft-protect symbol exposure ──────────────────────────────────────────


def test_soft_protect_symbols_exposed(mod) -> None:
    """Module exposes the SOFT-protect patterns + token + env-var names."""
    assert hasattr(mod, "SOFT_PROTECTED_PATTERNS")
    assert hasattr(mod, "SOFT_PROTECT_TOKEN")
    assert hasattr(mod, "SOFT_PROTECT_ENV_VAR")
    assert mod.SOFT_PROTECT_TOKEN == "refit-driver-re-emit"
    assert mod.SOFT_PROTECT_ENV_VAR == "GSD_REFIT_REEMIT"
    assert callable(mod.find_soft_violations)


def test_soft_protect_patterns_match_refit_filenames(mod) -> None:
    """Regex matches the canonical refit sibling filenames."""
    matches = mod.find_soft_violations(
        [
            "models/meta/meta_v2_refit_v2.6.joblib",
            "models/meta/meta_v2_refit_v2.6_meta.json",
            "models/meta/meta_v2_refit_v2.7.joblib",  # future milestone
            "models/meta/meta_v2_refit_v3.0_meta.json",
        ]
    )
    assert "models/meta/meta_v2_refit_v2.6.joblib" in matches
    assert "models/meta/meta_v2_refit_v2.6_meta.json" in matches
    assert "models/meta/meta_v2_refit_v2.7.joblib" in matches
    assert "models/meta/meta_v2_refit_v3.0_meta.json" in matches


def test_soft_protect_patterns_do_not_match_canonical(mod) -> None:
    """Canonical meta_v2 + Phase 65 refv2 candidate are NOT soft-protect matches."""
    matches = mod.find_soft_violations(
        [
            "models/meta/meta_v2.joblib",  # canonical
            "models/meta/meta_v2_meta.json",  # canonical sidecar
            "models/meta/meta_v2_refv2.joblib",  # Phase 65 candidate
            "models/meta/meta_v22_travel.joblib",  # Phase 42 advisory sibling
            "README.md",
            "scripts/refit_meta_v22_v2.6.py",
        ]
    )
    assert matches == [], f"unexpected soft-protect matches: {matches}"


# ── Soft-protect block / warn-allow behaviour ─────────────────────────────


def test_soft_protect_blocks_without_opt_in(mod, monkeypatch, capsys, tmp_path) -> None:
    """Refit sibling staged + no opt-in → exit 1 with 'refit-driver-re-emit' message."""
    monkeypatch.delenv(mod.SOFT_PROTECT_ENV_VAR, raising=False)
    monkeypatch.delenv(mod.OVERRIDE_ENV_VAR, raising=False)
    # Point GIT_DIR at a tmp dir with no COMMIT_EDITMSG so the token sniff
    # returns empty.
    monkeypatch.setenv("GIT_DIR", str(tmp_path))

    rc = mod.main(["models/meta/meta_v2_refit_v2.6.joblib"])
    assert rc == 1
    captured = capsys.readouterr()
    combined = captured.err + captured.out
    assert "SOFT-protected" in combined
    assert mod.SOFT_PROTECT_TOKEN in combined
    assert mod.SOFT_PROTECT_ENV_VAR in combined


def test_soft_protect_allows_with_env_var(mod, monkeypatch, capsys, tmp_path) -> None:
    """GSD_REFIT_REEMIT=1 → exit 0 with warning."""
    monkeypatch.setenv(mod.SOFT_PROTECT_ENV_VAR, "1")
    monkeypatch.delenv(mod.OVERRIDE_ENV_VAR, raising=False)
    monkeypatch.setenv("GIT_DIR", str(tmp_path))

    rc = mod.main(
        [
            "models/meta/meta_v2_refit_v2.6.joblib",
            "models/meta/meta_v2_refit_v2.6_meta.json",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    combined = captured.err + captured.out
    assert "SOFT-protected" in combined
    assert "operator opt-in" in combined


def test_soft_protect_allows_with_commit_token(mod, monkeypatch, capsys, tmp_path) -> None:
    """COMMIT_EDITMSG containing 'refit-driver-re-emit' → exit 0 with warning."""
    monkeypatch.delenv(mod.SOFT_PROTECT_ENV_VAR, raising=False)
    monkeypatch.delenv(mod.OVERRIDE_ENV_VAR, raising=False)
    # Write a stub COMMIT_EDITMSG inside tmp_path; point GIT_DIR at tmp_path.
    msg_path = tmp_path / "COMMIT_EDITMSG"
    msg_path.write_text(
        "chore(72): re-emit refit baseline\n\n"
        "Triggered by METH-V27-01 corpus-growth gate. refit-driver-re-emit\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_DIR", str(tmp_path))

    rc = mod.main(["models/meta/meta_v2_refit_v2.6.joblib"])
    assert rc == 0, f"expected exit 0 with commit-token opt-in; got {rc}"
    captured = capsys.readouterr()
    combined = captured.err + captured.out
    assert "SOFT-protected" in combined


def test_soft_protect_blocks_when_token_missing_from_message(
    mod, monkeypatch, capsys, tmp_path
) -> None:
    """COMMIT_EDITMSG WITHOUT the token does NOT trigger opt-in."""
    monkeypatch.delenv(mod.SOFT_PROTECT_ENV_VAR, raising=False)
    monkeypatch.delenv(mod.OVERRIDE_ENV_VAR, raising=False)
    msg_path = tmp_path / "COMMIT_EDITMSG"
    msg_path.write_text(
        "chore(72): some unrelated commit\n\nNo token here.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_DIR", str(tmp_path))

    rc = mod.main(["models/meta/meta_v2_refit_v2.6.joblib"])
    assert rc == 1, "missing token should NOT trigger opt-in"


# ── Hard-protect path regression checks ───────────────────────────────────


def test_hard_protect_still_blocks(mod, monkeypatch, capsys, tmp_path) -> None:
    """Canonical anchor staged → exit 1 (hard-protect unchanged by Phase 72)."""
    monkeypatch.delenv(mod.OVERRIDE_ENV_VAR, raising=False)
    monkeypatch.delenv(mod.SOFT_PROTECT_ENV_VAR, raising=False)
    monkeypatch.setenv("GIT_DIR", str(tmp_path))

    rc = mod.main(["models/meta/meta_v2.joblib"])
    assert rc == 1
    captured = capsys.readouterr()
    combined = captured.err + captured.out
    assert "AUDIT-01 protected files staged" in combined
    assert "BLOCKED" in combined


def test_hard_protect_still_allows_with_audit01_override(
    mod, monkeypatch, capsys, tmp_path
) -> None:
    """AUDIT01_OVERRIDE=1 still allows hard-protect commit (regression check)."""
    monkeypatch.setenv(mod.OVERRIDE_ENV_VAR, "1")
    monkeypatch.delenv(mod.SOFT_PROTECT_ENV_VAR, raising=False)
    monkeypatch.setenv("GIT_DIR", str(tmp_path))

    rc = mod.main(["models/meta/meta_v2.joblib"])
    assert rc == 0


# ── Mixed: both hard + soft files staged ──────────────────────────────────


def test_mixed_hard_and_soft_blocks_when_hard_unauthorized(
    mod, monkeypatch, capsys, tmp_path
) -> None:
    """Hard-protect blocks first; soft path is reached only if hard is clear."""
    monkeypatch.delenv(mod.OVERRIDE_ENV_VAR, raising=False)
    monkeypatch.setenv(mod.SOFT_PROTECT_ENV_VAR, "1")  # soft authorized
    monkeypatch.setenv("GIT_DIR", str(tmp_path))

    rc = mod.main(
        [
            "models/meta/meta_v2.joblib",  # hard-protect
            "models/meta/meta_v2_refit_v2.6.joblib",  # soft-protect
        ]
    )
    assert rc == 1, "hard-protect block must fire even with soft opt-in active"


def test_mixed_hard_and_soft_allows_when_both_authorized(
    mod, monkeypatch, capsys, tmp_path
) -> None:
    """Both overrides set → hard + soft both pass."""
    monkeypatch.setenv(mod.OVERRIDE_ENV_VAR, "1")
    monkeypatch.setenv(mod.SOFT_PROTECT_ENV_VAR, "1")
    monkeypatch.setenv("GIT_DIR", str(tmp_path))

    rc = mod.main(
        [
            "models/meta/meta_v2.joblib",
            "models/meta/meta_v2_refit_v2.6.joblib",
        ]
    )
    assert rc == 0

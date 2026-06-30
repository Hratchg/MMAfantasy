"""Phase 26 Plan 26-04 — unit tests for `get_latest_meta_version()` helper.

Per RESEARCH §Runtime State Inventory Pitfall #5: replaces the hardcoded
`meta_v1.joblib` lookup in predictor.py so the predictor auto-picks whichever
meta version is promoted. Per D-10: `_candidate` artifacts are IGNORED.

Tests E1..E4 (per 26-04-PLAN <behavior>):
    E1: empty dir            -> None
    E2: meta_v1 only         -> "v1"
    E3: meta_v1 + meta_v2    -> "v2"
    E4: meta_v1 + candidate  -> "v1"  (candidate ignored)
"""

from __future__ import annotations

from pathlib import Path

from ufc_prediction.ml.meta_persistence import get_latest_meta_version


def _touch(p: Path) -> None:
    """Create an empty file at path p (parents already exist)."""
    p.write_bytes(b"")


def test_get_latest_meta_version_empty_dir(tmp_path: Path) -> None:
    """E1 — empty `models/meta/` returns None."""
    # tmp_path is empty
    assert get_latest_meta_version(str(tmp_path)) is None


def test_get_latest_meta_version_missing_dir(tmp_path: Path) -> None:
    """E1' — non-existent meta_dir returns None (does NOT raise)."""
    missing = tmp_path / "does_not_exist"
    assert get_latest_meta_version(str(missing)) is None


def test_get_latest_meta_version_v1_only(tmp_path: Path) -> None:
    """E2 — dir contains `meta_v1.joblib` only -> returns 'v1'."""
    _touch(tmp_path / "meta_v1.joblib")
    _touch(tmp_path / "meta_v1_meta.json")  # sibling JSON should be ignored
    assert get_latest_meta_version(str(tmp_path)) == "v1"


def test_get_latest_meta_version_v1_and_v2(tmp_path: Path) -> None:
    """E3 — both v1 + v2 promoted -> returns 'v2' (highest wins)."""
    _touch(tmp_path / "meta_v1.joblib")
    _touch(tmp_path / "meta_v2.joblib")
    assert get_latest_meta_version(str(tmp_path)) == "v2"


def test_get_latest_meta_version_ignores_candidate(tmp_path: Path) -> None:
    """E4 — `meta_v2_candidate.joblib` is NOT a promoted version (D-10)."""
    _touch(tmp_path / "meta_v1.joblib")
    _touch(tmp_path / "meta_v2_candidate.joblib")
    _touch(tmp_path / "meta_v2_candidate_meta.json")
    _touch(tmp_path / "meta_v2-contract.json")  # sibling contract also ignored
    assert get_latest_meta_version(str(tmp_path)) == "v1"


def test_get_latest_meta_version_double_digit(tmp_path: Path) -> None:
    """Edge case — numeric ordering, not lexicographic ('v10' > 'v2')."""
    _touch(tmp_path / "meta_v2.joblib")
    _touch(tmp_path / "meta_v10.joblib")
    assert get_latest_meta_version(str(tmp_path)) == "v10"


def test_get_latest_meta_version_ignores_non_joblib(tmp_path: Path) -> None:
    """Edge case — only files matching `meta_v{N}.joblib` count."""
    _touch(tmp_path / "meta_v1.txt")  # wrong extension
    _touch(tmp_path / "metaX_v1.joblib")  # wrong prefix
    _touch(tmp_path / "meta_v.joblib")  # no digits
    _touch(tmp_path / "README.md")
    assert get_latest_meta_version(str(tmp_path)) is None

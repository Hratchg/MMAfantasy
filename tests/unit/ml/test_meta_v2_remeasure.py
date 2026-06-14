"""TRUST-V24-02 — tests for META-V22 re-measurement append.

Plan 34-03 Task 3 — locks the v2_3_slice_metrics append contract + legacy-key value parity
(revision n2) against the pre-Task-2 git-fixture snapshot.

Fixture source = ``git show HEAD:models/meta/meta_v2_meta.json`` captured by Task 2 step 0
BEFORE Task 2 mutated the file. This catches legacy-key VALUE corruption that a pure
key-set comparison would miss.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
META_META_PATH = PROJECT_ROOT / "models" / "meta" / "meta_v2_meta.json"
META_JOBLIB_PATH = PROJECT_ROOT / "models" / "meta" / "meta_v2.joblib"
FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "meta_v2_meta_pre_phase_34_t2_snapshot.json"
)
SHA_START_PATH = (
    PROJECT_ROOT
    / ".planning"
    / "phases"
    / "34-trust-baseline-no-odds-fallback"
    / "34-META-V2-SHA-PHASE-34-START-PLAN-01.txt"
)


@pytest.fixture(scope="module")
def meta_doc() -> dict:
    if not META_META_PATH.exists():
        pytest.skip(f"{META_META_PATH} not found")
    return json.loads(META_META_PATH.read_text())


@pytest.fixture(scope="module")
def snapshot_doc() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"git-fixture snapshot {FIXTURE_PATH} missing")
    return json.loads(FIXTURE_PATH.read_text())


def test_v2_3_slice_metrics_key_present(meta_doc: dict) -> None:
    """The script must append a new `v2_3_slice_metrics` key (not overwrite anything)."""
    assert "v2_3_slice_metrics" in meta_doc


def test_three_slices_present(meta_doc: dict) -> None:
    """The new key must contain exactly the 3 v2.3 widened slices."""
    sm = meta_doc["v2_3_slice_metrics"]
    assert set(sm.keys()) == {"most_recent_12mo", "most_recent_24mo", "random_15pct"}


def test_per_slice_required_keys(meta_doc: dict) -> None:
    """Each slice block carries the documented metric fields."""
    required = {
        "brier_score",
        "auc_roc",
        "accuracy",
        "calibration_curve",
        "n",
        "measured_at",
    }
    for slice_name, body in meta_doc["v2_3_slice_metrics"].items():
        missing = required - set(body.keys())
        assert not missing, f"slice {slice_name} missing keys: {missing}"


def test_pre_existing_keys_preserved(meta_doc: dict, snapshot_doc: dict) -> None:
    """Revision n2 — load the pre-Task-2 git-fixture snapshot; assert every legacy key
    still present in current file AND every legacy value byte-equal to the snapshot value.
    This catches both deletion and in-place value corruption.

    Fixture source = ``git show HEAD:models/meta/meta_v2_meta.json`` captured by Task 2
    step 0 BEFORE Task 2 mutated the file.
    """
    # key set: every snapshot key present in current
    missing = set(snapshot_doc.keys()) - set(meta_doc.keys())
    assert not missing, f"legacy keys dropped: {missing}"
    # per-key values: snapshot values byte-equal to current values
    for k in snapshot_doc:
        assert meta_doc[k] == snapshot_doc[k], (
            f"legacy key {k!r} VALUE corrupted: snapshot != current"
        )


def test_meta_v2_joblib_sha_unchanged() -> None:
    """meta_v2.joblib SHA must match the START anchor (script never touches the .joblib)."""
    if not SHA_START_PATH.exists() or not META_JOBLIB_PATH.exists():
        pytest.skip("missing SHA anchor or .joblib")
    expected = SHA_START_PATH.read_text().strip()
    actual = hashlib.sha256(META_JOBLIB_PATH.read_bytes()).hexdigest()
    assert actual == expected, f"meta_v2.joblib SHA drift: {actual} vs {expected}"

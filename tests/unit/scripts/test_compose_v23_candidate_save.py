"""compose_v23 Path A must never auto-promote a meta model.

Code-review finding (meta_persistence.save_meta_model / compose_v23 Path A):
Path A used to call ``save_meta_model(meta_version="v3")`` with no operator
step, writing a promoted-name ``meta_v3.joblib`` that the serve path then
auto-discovered (and halted on, since predictor.py only validates v1/v2
meta columns). It also shared one OOF cache with the 730-day-window scripts.
"""

from __future__ import annotations

from pathlib import Path

from scripts import compose_v23_meta
from ufc_prediction.ml.meta_persistence import get_latest_meta_version


def test_path_a_saves_under_a_non_promoted_candidate_name(tmp_path: Path) -> None:
    """The Path A artifact name is never discoverable as a promoted version."""
    (tmp_path / "meta_v2.joblib").write_bytes(b"")
    (tmp_path / f"meta_{compose_v23_meta.META_V3_CANDIDATE_VERSION}.joblib").write_bytes(b"")
    # Even with the promoted allow-list disabled, the candidate name is ignored.
    assert get_latest_meta_version(str(tmp_path), allowed_versions=None) == "v2"


def test_path_a_candidate_does_not_clobber_tracked_meta_v3_candidate() -> None:
    """models/meta/meta_v3_candidate.joblib is a tracked Phase 45/48 artifact."""
    assert compose_v23_meta.META_V3_CANDIDATE_VERSION != "v3_candidate"


def test_compose_oof_cache_is_not_shared_with_730d_window_scripts() -> None:
    """compose (365d window) must not share the 730d scripts' OOF cache file."""
    shared_730d = Path(
        ".planning/phases/26-forward-stepwise-candidate-promotion/oof_predictions_v22.parquet"
    )
    assert compose_v23_meta.META_OOF_PARQUET_PATH != shared_730d

"""META-03: OOF-leakage CI regression test (Pitfall #11).

Asserts the invariant that the cached OOF parquet at
.planning/phases/19-meta-learner/oof_predictions.parquet contains predictions
whose training_accuracy < 0.75 — i.e., they look out-of-sample not in-sample.

Failure of this test means cross_val_predict was called with the wrong cv
parameter (StratifiedKFold instead of TimeSeriesSplit). Until Wave 2 builds
the parquet, individual tests skip via pytest.skip; the test file is RED-as-
tripwire, ship-passing at commit but firing if cache is regenerated incorrectly.
"""

import json
import pathlib
import pytest

pytest.importorskip("ufc_prediction.ml.oof")

PARQUET_PATH = pathlib.Path(".planning/phases/19-meta-learner/oof_predictions.parquet")
META_JSON_PATH = pathlib.Path("models/xgb_v2_meta.json")


def _read_oof_metadata():
    """Return parquet sidecar metadata or None if parquet not yet built."""
    if not PARQUET_PATH.exists():
        return None
    # We persist metadata as both parquet schema metadata AND a sidecar JSON
    # at .planning/phases/19-meta-learner/oof_predictions_meta.json (per OQ-4
    # serialization stability — JSON sidecar is the canonical readable form).
    sidecar = PARQUET_PATH.with_suffix(".meta.json")
    if not sidecar.exists():
        return None
    return json.loads(sidecar.read_text(encoding="utf-8"))


def test_oof_parquet_training_accuracy():
    """Pitfall #11 tripwire: cached OOF must look out-of-sample (acc < 0.75)."""
    meta = _read_oof_metadata()
    if meta is None:
        pytest.skip("OOF parquet not built yet (Wave 2 deliverable)")
    assert meta["training_accuracy"] < 0.75, (
        f"OOF training_accuracy={meta['training_accuracy']:.4f} ≥ 0.75 — "
        "predictions look in-sample (TimeSeriesSplit broken or wrong cv)"
    )


def test_oof_uses_timeseries_split():
    """Pitfall #11 tripwire: parquet metadata must record cv_kind=TimeSeriesSplit."""
    meta = _read_oof_metadata()
    if meta is None:
        pytest.skip("OOF parquet not built yet (Wave 2 deliverable)")
    assert meta.get("cv_kind") == "TimeSeriesSplit", (
        f"cv_kind={meta.get('cv_kind')!r} — only TimeSeriesSplit is leakage-free"
    )


def test_disjoint_train_meta_train_ids():
    """D-01(P19) persistent disjoint assertion: meta_train fight_ids ∩ base_train fight_ids == set()."""
    meta = _read_oof_metadata()
    if meta is None:
        pytest.skip("OOF parquet not built yet (Wave 2 deliverable)")
    if "meta_train_fight_ids" not in meta:
        pytest.skip("meta_train_fight_ids not recorded in parquet metadata")
    if not META_JSON_PATH.exists():
        pytest.skip("models/xgb_v2_meta.json missing")
    xgb_meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
    base_ids = set(xgb_meta.get("base_training_fight_ids", []))
    meta_ids = set(meta["meta_train_fight_ids"])
    if not base_ids:
        pytest.skip(
            "xgb_v2_meta.json does not record base_training_fight_ids "
            "(Phase 19 may add this in Wave 2 — currently skip)"
        )
    overlap = base_ids & meta_ids
    assert not overlap, (
        f"D-01(P19) violated: {len(overlap)} fight_ids appear in both base_train "
        "and meta_train; three-way split partition broken"
    )

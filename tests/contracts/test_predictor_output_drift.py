"""Phase 25 WR-05 (review-fix): catch drift between mock predictor output
and the keys the v1/predict route actually accesses.

The existing fixtures (xgb_v2_only_mock in tests/contracts/conftest.py,
_FAKE_PREDICT_RESULT in tests/api/test_predict_v1.py) are both
HAND-CONSTRUCTED. If a future phase renames a key in
ModelPredictor.predict()'s return dict (e.g., meta_skipped_reason →
meta_skip_reason), the mocks continue to validate the v1 schema but
ACTUAL production responses would 500 at the route adapter line
(result["fighter_a"] → KeyError) because the route still reads the old name.

The full-fidelity drift check requires a real ModelPredictor.predict()
call against a seeded DB — gated by Docker/testcontainers (see
tests/api/test_predict_v1.py::TestPredictV1 docstring), which is unavailable
in many CI environments per 25-SUMMARY.md.

This test compromises: statically extract the keys the route adapter reads
from `result` (via AST inspection of predict.py source), and assert those
keys are present in the locked _PREDICTOR_OUTPUT_REFERENCE dict committed
HERE. The reference is curated from the real ModelPredictor.predict() shape
(verified via the predictor.py return statement at lines 263-284, as of
the locking commit). Any future phase that renames a result key in
predictor.py without updating BOTH this reference and the route adapter
will trigger this test.

The reference is also asserted to be schema-valid against
PredictorOutputV1 (the partner-facing envelope).
"""

from __future__ import annotations

import ast
from pathlib import Path

from ufc_prediction.api.v1.models import PredictorOutputV1

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICT_ROUTE_PATH = REPO_ROOT / "src" / "ufc_prediction" / "api" / "v1" / "predict.py"


# Reference output for ModelPredictor.predict() under the xgb_v2-only
# regime (no meta artifact loaded). Synthesized from the predictor.py
# return statement at v25 lock-time. WR-05 (Phase 25 review-fix): if
# predictor.py renames a key, the next reviewer who touches predictor.py
# MUST update this reference (and bump the lock commit comment above).
#
# The 7 partner-facing keys (consumed by api/v1/predict.py adapter) are
# guarded by test_route_accessed_keys_present_in_reference below;
# remaining keys are predictor-internal back-compat for Phase 16+ Elo
# consumers and are not exposed via the v1 envelope.
_PREDICTOR_OUTPUT_REFERENCE: dict = {
    # Partner-facing envelope fields (consumed by v1 route adapter):
    "fighter_a": "Khabib Nurmagomedov",
    "fighter_b": "Conor McGregor",
    "win_probability": 0.6234,
    "base_prob": 0.6234,
    "meta_prob": None,
    "meta_learner_version": None,
    "meta_skipped_reason": "no_meta_artifact",
    # Phase 35 CONTRACT-V24-02 (Plan 35-04 + 35-05) — additive
    # prediction_metadata key on the predictor return shape; surfaced to
    # partners under the v1.2.0 contract. The route adapter reads this
    # key via result.get("prediction_metadata") and projects the inner
    # dict onto PredictionMetadataV12.
    "prediction_metadata": {
        "win_probability_source": "xgb_v2_no_odds",
        "odds_source": "nan",
        "odds_timestamp_iso": None,
        "fighter_a_n_ufc_fights": 0,
        "fighter_b_n_ufc_fights": 0,
        "is_debutant_either": False,
        "calibration_slice_brier": None,
    },
    # Predictor-internal back-compat fields (NOT exposed via v1 envelope
    # but produced by ModelPredictor.predict() — verified against
    # predictor.py return statement at lines 263-284):
    "meta_kind": None,
    "meta_skipped": True,
    "model_probability_a": 0.6234,
    "model_probability_b": 0.3766,
    "elo_probability_a": 0.5,
    "elo_probability_b": 0.5,
    "elo_a": 1500.0,
    "elo_b": 1500.0,
    "feature_importances": [],
    "odds_source": "nan",
}


def _extract_route_accessed_keys() -> set[str]:
    """Parse predict.py and return every string-literal key the route reads
    from a variable named `result` via subscript or .get().

    This is intentionally narrow: we only care about the adapter line
    surface — `result["..."]` and `result.get("...")`. If the route ever
    starts reading dynamic keys (e.g. `result[some_var]`), this test
    becomes incomplete but does NOT regress — the static check will
    simply miss the dynamic key.
    """
    tree = ast.parse(PREDICT_ROUTE_PATH.read_text(encoding="utf-8"))
    keys: set[str] = set()

    for node in ast.walk(tree):
        # result["key"]
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "result"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)
        # result.get("key", ...)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "result"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)

    return keys


class TestPredictorOutputDrift:
    """WR-05 drift guard: route-accessed keys must exist in reference."""

    def test_route_accessed_keys_extraction_nonempty(self):
        """Sanity: the AST extractor finds at least one key (e.g.
        win_probability). If this fails, the extractor is broken — NOT
        the route — and the test below would be a false negative."""
        accessed = _extract_route_accessed_keys()
        assert "win_probability" in accessed
        # Expect at least 7 partner-facing keys
        assert len(accessed) >= 7, (
            f"Expected >=7 route-accessed keys; got {len(accessed)}: {accessed}"
        )

    def test_route_accessed_keys_present_in_reference(self):
        """The route reads N keys from result; every one MUST be in the
        locked predictor-output reference. If a key is missing, either
        predictor.py renamed it (update reference + this test) or the
        route adapter is reading a non-existent key (BLOCKER for partners)."""
        accessed = _extract_route_accessed_keys()
        missing = accessed - _PREDICTOR_OUTPUT_REFERENCE.keys()
        assert not missing, (
            f"Route adapter reads keys not present in the predictor-output "
            f"reference: {sorted(missing)}. Either predictor.py renamed a "
            f"key (update _PREDICTOR_OUTPUT_REFERENCE in this test) or the "
            f"route adapter (src/ufc_prediction/api/v1/predict.py) is "
            f"accessing a non-existent key."
        )

    def test_reference_projects_to_valid_predictor_output_v1(self):
        """The reference dict, projected through the v1 route adapter
        logic, MUST validate against PredictorOutputV1. If this fails,
        the v1 schema and predictor.py output have drifted."""
        from datetime import date

        # Mirror the projection in src/ufc_prediction/api/v1/predict.py
        # (the only test in this suite that depends on the adapter logic
        # — keep in sync if the adapter changes).
        projected = {
            "schema_version": "1.0.0",
            "win_probability": _PREDICTOR_OUTPUT_REFERENCE["win_probability"],
            "fighter_a": _PREDICTOR_OUTPUT_REFERENCE["fighter_a"],
            "fighter_b": _PREDICTOR_OUTPUT_REFERENCE["fighter_b"],
            "event_date": date(2026, 5, 17).isoformat(),
            "base_prob": _PREDICTOR_OUTPUT_REFERENCE.get("base_prob"),
            "meta_prob": _PREDICTOR_OUTPUT_REFERENCE.get("meta_prob"),
            "meta_learner_version": _PREDICTOR_OUTPUT_REFERENCE.get(
                "meta_learner_version",
            ),
            "meta_skipped_reason": _PREDICTOR_OUTPUT_REFERENCE.get(
                "meta_skipped_reason",
            ),
        }
        # Validates or raises ValidationError → test fails loudly.
        obj = PredictorOutputV1.model_validate(projected)
        assert obj.win_probability == _PREDICTOR_OUTPUT_REFERENCE["win_probability"]
        assert obj.fighter_a == _PREDICTOR_OUTPUT_REFERENCE["fighter_a"]

    def test_reference_keys_match_real_predictor_return_signature(self):
        """The reference's KEY SET must match ModelPredictor.predict()'s
        actual return-dict key set, statically extracted from predictor.py's
        return statement. Catches the case where predictor.py adds OR
        removes a key without updating the reference here."""
        predictor_path = REPO_ROOT / "src" / "ufc_prediction" / "ml" / "predictor.py"
        tree = ast.parse(predictor_path.read_text(encoding="utf-8"))

        # Locate ModelPredictor.predict() and extract the (last) Return's
        # Dict key set. predict() returns a single literal dict at the end
        # of its happy path (verified at lock time).
        predict_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ModelPredictor":
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == "predict":
                        predict_func = child
                        break
                break

        assert predict_func is not None, (
            "ModelPredictor.predict not found in predictor.py — refactor "
            "moved the predict method; update this drift guard."
        )

        return_dict: ast.Dict | None = None
        for stmt in ast.walk(predict_func):
            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
                return_dict = stmt.value  # last Return wins; happy path is last
        assert return_dict is not None, (
            "ModelPredictor.predict has no literal-dict Return — refactor "
            "changed the return shape; update this drift guard."
        )

        actual_keys: set[str] = set()
        for key_node in return_dict.keys:
            if isinstance(key_node, ast.Constant) and isinstance(
                key_node.value,
                str,
            ):
                actual_keys.add(key_node.value)

        ref_keys = set(_PREDICTOR_OUTPUT_REFERENCE.keys())
        added = actual_keys - ref_keys
        removed = ref_keys - actual_keys
        assert not added and not removed, (
            f"ModelPredictor.predict() return-key drift detected. "
            f"Added in predictor.py: {sorted(added)}. "
            f"Removed from predictor.py: {sorted(removed)}. "
            f"Update _PREDICTOR_OUTPUT_REFERENCE in this test to match "
            f"the new return shape, then re-verify the v1 route adapter "
            f"in src/ufc_prediction/api/v1/predict.py reads the right keys."
        )


# Module-level smoke (callable from a future Docker-gated test that wants
# to compare real predictor output against the reference).
def get_reference_predictor_output() -> dict:
    """Public accessor for the locked predictor-output reference.

    Useful for an eventual integration test that calls real
    ModelPredictor.predict() against a seeded DB and asserts the result
    keys match this reference's keys (the full-fidelity drift check the
    Docker-gated TestPredictV1 suite cannot perform today).
    """
    return dict(_PREDICTOR_OUTPUT_REFERENCE)

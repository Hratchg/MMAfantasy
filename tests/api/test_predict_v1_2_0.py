"""Phase 35 CONTRACT-V24-02 — /api/v1/predict v1.2.0 negotiation tests.

Six behaviors (per Plan 35-05 Task 2):
  1. accept_schema_version="1.2.0" → prediction_metadata block populated.
  2. accept_schema_version omitted (default) → same as 1.
  3. accept_schema_version="1.1.0" → schema_version "1.1.0" + metadata None.
  4. accept_schema_version="1.0.0" → schema_version "1.0.0" + metadata None.
  5. accept_schema_version="2.0.0" → 422 ValidationError.
  6. Phase 32 D-04 additive trio still passes through unchanged.

Mirrors the mocking pattern from tests/api/test_predict_v1.py — the
ModelPredictor is replaced with an in-memory fake that returns the
post-Plan-35-04 prediction_metadata dict (7 keys). The route's job is to
project that dict onto the partner-facing PredictionMetadataV12 model
(6 fields) when v1.2.0 is negotiated, and to leave the field None on
older contract versions (forward-compat lock).
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ufc_prediction.api.deps import get_db
from ufc_prediction.api.v1 import predict as v1_predict


# Mirrors the predictor.predict() return shape after Plan 35-04: the dict
# now carries a `prediction_metadata` sub-dict with the 7 internal keys
# (including win_probability_source which the partner-facing model drops).
_FAKE_PREDICT_RESULT = {
    "fighter_a": "Khabib Nurmagomedov",
    "fighter_b": "Conor McGregor",
    "win_probability": 0.6234,
    "base_prob": 0.6234,
    "meta_prob": None,
    "meta_kind": None,
    "meta_learner_version": None,
    "meta_skipped": True,
    "meta_skipped_reason": "no_meta_artifact",
    "model_probability_a": 0.6234,
    "model_probability_b": 0.3766,
    "elo_probability_a": 0.5,
    "elo_probability_b": 0.5,
    "elo_a": 1500.0,
    "elo_b": 1500.0,
    "feature_importances": [],
    "odds_source": "live",
    "prediction_metadata": {
        # 7 internal keys per Plan 35-04 predictor.py — partner contract
        # surfaces only the 6 below (win_probability_source dropped at route).
        "win_probability_source": "xgb_v2_no_odds",
        "odds_source": "live",
        "odds_timestamp_iso": "2018-10-06T22:00:00+00:00",
        "fighter_a_n_ufc_fights": 10,
        "fighter_b_n_ufc_fights": 21,
        "is_debutant_either": False,
        "calibration_slice_brier": 0.19,
    },
}


class _FakePredictor:
    """In-memory fake mirroring ModelPredictor.predict()."""

    def predict(
        self,
        db,
        fighter_a_name,
        fighter_b_name,
        *,
        event_date=None,
        refresh=False,
    ):
        if "Nonexistent" in fighter_a_name or "Nonexistent" in fighter_b_name:
            raise ValueError(
                f"Fighter not found: {fighter_a_name}/{fighter_b_name}",
            )
        return {
            **_FAKE_PREDICT_RESULT,
            "fighter_a": fighter_a_name,
            "fighter_b": fighter_b_name,
        }


def _standalone_app_client() -> TestClient:
    """Minimal FastAPI app — mirrors tests/api/test_predict_v1.py helper."""
    app = FastAPI()
    app.include_router(v1_predict.router, prefix="/api/v1")

    def _fake_db():
        yield None

    app.dependency_overrides[get_db] = _fake_db
    return TestClient(app)


def _post(client: TestClient, **extra) -> "dict | None":
    """Helper: POST a happy-path matchup body + return response."""
    body = {
        "fighter_a": "Khabib Nurmagomedov",
        "fighter_b": "Conor McGregor",
        "event_date": "2018-10-06",
    }
    body.update(extra)
    with patch(
        "ufc_prediction.api.v1.predict._get_predictor",
        return_value=_FakePredictor(),
    ):
        return client.post("/api/v1/predict", json=body)


class TestPredictV1_2_0Negotiation:
    """CONTRACT-V24-02 schema-version negotiation surface."""

    def test_explicit_v120_populates_prediction_metadata(self):
        """Behavior 1 — v1.2.0 request → 6-field block populated."""
        c = _standalone_app_client()
        resp = _post(c, accept_schema_version="1.2.0")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["schema_version"] == "1.2.0"
        block = body["prediction_metadata"]
        assert block is not None
        # Exactly the 6 CONTEXT-listed fields (win_probability_source dropped).
        assert set(block.keys()) == {
            "odds_source",
            "odds_timestamp_iso",
            "fighter_a_n_ufc_fights",
            "fighter_b_n_ufc_fights",
            "is_debutant_either",
            "calibration_slice_brier",
        }
        assert block["odds_source"] == "live"
        assert block["odds_timestamp_iso"] == "2018-10-06T22:00:00+00:00"
        assert block["fighter_a_n_ufc_fights"] == 10
        assert block["fighter_b_n_ufc_fights"] == 21
        assert block["is_debutant_either"] is False
        assert block["calibration_slice_brier"] == 0.19

    def test_default_version_is_v120(self):
        """Behavior 2 — omitted accept_schema_version → same as v1.2.0."""
        c = _standalone_app_client()
        resp = _post(c)  # no accept_schema_version
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["schema_version"] == "1.2.0"
        assert body["prediction_metadata"] is not None
        assert body["prediction_metadata"]["odds_source"] == "live"

    def test_v110_request_leaves_metadata_none(self):
        """Behavior 3 — v1.1.0 → schema_version 1.1.0 + metadata None."""
        c = _standalone_app_client()
        resp = _post(c, accept_schema_version="1.1.0")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["schema_version"] == "1.1.0"
        assert body["prediction_metadata"] is None

    def test_v100_request_leaves_metadata_none(self):
        """Behavior 4 — v1.0.0 → schema_version 1.0.0 + metadata None."""
        c = _standalone_app_client()
        resp = _post(c, accept_schema_version="1.0.0")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["schema_version"] == "1.0.0"
        assert body["prediction_metadata"] is None

    def test_unknown_schema_version_returns_422(self):
        """Behavior 5 — Literal rejection → 422."""
        c = _standalone_app_client()
        resp = _post(c, accept_schema_version="2.0.0")
        assert resp.status_code == 422

    def test_phase32_additive_trio_unaffected(self):
        """Behavior 6 — gate_contract_ref/model_candidates/phase_chain_audit_sha
        still flow through as None on v1.0.0 + v1.1.0 routes (no regression)."""
        c = _standalone_app_client()
        for version in ("1.0.0", "1.1.0", "1.2.0"):
            resp = _post(c, accept_schema_version=version)
            assert resp.status_code == 200, resp.text
            body = resp.json()
            for key in (
                "gate_contract_ref",
                "model_candidates",
                "phase_chain_audit_sha",
            ):
                assert key in body, f"v{version} response missing Phase 32 D-04 field {key!r}"
                assert body[key] is None, (
                    f"v{version} field {key!r} should be None (none populated "
                    f"by the predictor); got {body[key]!r}"
                )


class TestPredictV1_2_0EdgeCases:
    """Defensive paths — empty metadata dict, partial metadata, etc."""

    def test_missing_metadata_falls_back_to_defaults(self):
        """When predictor returns no prediction_metadata key (legacy path),
        the v1.2.0 route synthesizes safe defaults rather than 500-ing."""
        legacy_result = {
            k: v for k, v in _FAKE_PREDICT_RESULT.items() if k != "prediction_metadata"
        }

        class _LegacyPredictor:
            def predict(self, db, a, b, *, event_date=None, refresh=False):
                return {**legacy_result, "fighter_a": a, "fighter_b": b}

        app = FastAPI()
        app.include_router(v1_predict.router, prefix="/api/v1")

        def _fake_db():
            yield None

        app.dependency_overrides[get_db] = _fake_db
        c = TestClient(app)
        with patch(
            "ufc_prediction.api.v1.predict._get_predictor",
            return_value=_LegacyPredictor(),
        ):
            resp = c.post(
                "/api/v1/predict",
                json={
                    "fighter_a": "A",
                    "fighter_b": "B",
                    "event_date": "2026-05-27",
                    "accept_schema_version": "1.2.0",
                },
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        block = body["prediction_metadata"]
        assert block is not None
        # Defaults: odds_source="nan", n_fights=0, is_debutant_either=False,
        # odds_timestamp_iso=None, calibration_slice_brier=None.
        assert block["odds_source"] == "nan"
        assert block["fighter_a_n_ufc_fights"] == 0
        assert block["fighter_b_n_ufc_fights"] == 0
        assert block["is_debutant_either"] is False
        assert block["odds_timestamp_iso"] is None
        assert block["calibration_slice_brier"] is None

    def test_v1_2_0_response_excludes_win_probability_source(self):
        """STRIDE T-35-05-02 — partner-facing block surfaces only the 6
        CONTEXT-listed fields; win_probability_source stays internal."""
        c = _standalone_app_client()
        resp = _post(c, accept_schema_version="1.2.0")
        assert resp.status_code == 200, resp.text
        block = resp.json()["prediction_metadata"]
        assert "win_probability_source" not in block

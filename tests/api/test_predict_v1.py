"""Integration tests for POST /api/v1/predict (Phase 25 Plan 25-02 D-06).

Note: ModelPredictor.predict() is mocked to avoid loading the real xgb_v2
joblib + executing full feature pipeline against the in-memory test DB
(seed_api_data lacks the BFO odds + computed feature substrate needed for
real inference). The route-shape contract (status codes, response Pydantic
shape, V11/V5 ASVS rejections, OpenAPI registration) is what matters for
Phase 25 D-06 verification.

Two test layers:
  - TestPredictV1NoDB: standalone TestClient without DB session (V5/V11 ASVS
    + OpenAPI shape; runs in any environment, no testcontainers required).
  - TestPredictV1: full TestClient via tests/api/conftest.py's seed_api_data
    fixture (requires Docker/PostgreSQL testcontainer; will skip cleanly in
    environments without Docker).
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ufc_prediction.api.deps import get_db
from ufc_prediction.api.v1 import predict as v1_predict

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
    "odds_source": "nan",
}


class _FakePredictor:
    """In-memory fake that mimics ModelPredictor.predict() return shape."""

    def predict(self, db, fighter_a_name, fighter_b_name, *, event_date=None, refresh=False):
        # Map fake-name "Nonexistent Fighter" to ValueError for the 404 test
        if "Nonexistent" in fighter_a_name or "Nonexistent" in fighter_b_name:
            raise ValueError(f"Fighter not found: {fighter_a_name}/{fighter_b_name}")
        return {
            **_FAKE_PREDICT_RESULT,
            "fighter_a": fighter_a_name,
            "fighter_b": fighter_b_name,
        }


def _standalone_app_client() -> TestClient:
    """Build a minimal FastAPI app with only the v1.predict router for
    no-DB tests. get_db is overridden to a no-op yielder."""
    app = FastAPI()
    app.include_router(v1_predict.router, prefix="/api/v1")

    def _fake_db():
        yield None

    app.dependency_overrides[get_db] = _fake_db
    return TestClient(app)


class TestPredictV1NoDB:
    """No-DB route shape tests (V5/V11 ASVS + Pydantic validation).

    These run anywhere — no testcontainers / Docker required.
    """

    def test_same_fighter_rejected_400(self):
        c = _standalone_app_client()
        resp = c.post(
            "/api/v1/predict",
            json={"fighter_a": "Khabib Nurmagomedov", "fighter_b": "khabib nurmagomedov"},
        )
        assert resp.status_code == 400

    def test_empty_fighter_returns_422(self):
        c = _standalone_app_client()
        resp = c.post(
            "/api/v1/predict",
            json={"fighter_a": "", "fighter_b": "Conor McGregor"},
        )
        assert resp.status_code == 422

    def test_oversize_fighter_returns_422(self):
        c = _standalone_app_client()
        resp = c.post(
            "/api/v1/predict",
            json={"fighter_a": "X" * 201, "fighter_b": "Y"},
        )
        assert resp.status_code == 422

    def test_openapi_includes_predict_path_and_is_31(self):
        """OpenAPI spec includes /api/v1/predict and emits 3.1.0.

        Uses the full create_app() to verify the production wiring registers
        the v1.predict router into the real app (not the standalone test app).
        """
        from ufc_prediction.api.app import create_app

        app = create_app()
        spec = app.openapi()
        assert spec["openapi"] == "3.1.0"
        assert "/api/v1/predict" in spec["paths"]
        predict_op = spec["paths"]["/api/v1/predict"]
        assert "post" in predict_op

    def test_happy_path_with_fake_predictor(self):
        """Happy path with mocked _get_predictor — no DB required."""
        c = _standalone_app_client()
        with patch(
            "ufc_prediction.api.v1.predict._get_predictor",
            return_value=_FakePredictor(),
        ):
            resp = c.post(
                "/api/v1/predict",
                json={
                    "fighter_a": "Khabib Nurmagomedov",
                    "fighter_b": "Conor McGregor",
                    "event_date": "2018-10-06",
                },
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Plan 35-05 CONTRACT-V24-02: default accept_schema_version is "1.2.0"
        # so the route now echoes "1.2.0" when no negotiation field is sent.
        assert body["schema_version"] == "1.2.0"
        assert 0.0 <= body["win_probability"] <= 1.0
        assert body["fighter_a"] == "Khabib Nurmagomedov"
        assert body["fighter_b"] == "Conor McGregor"
        assert "base_prob" in body
        assert "meta_prob" in body
        assert "meta_learner_version" in body
        assert "meta_skipped_reason" in body

    def test_unknown_fighter_returns_404(self):
        """ValueError from predictor → 404."""
        c = _standalone_app_client()
        with patch(
            "ufc_prediction.api.v1.predict._get_predictor",
            return_value=_FakePredictor(),
        ):
            resp = c.post(
                "/api/v1/predict",
                json={"fighter_a": "Nonexistent Fighter", "fighter_b": "Conor McGregor"},
            )
        assert resp.status_code == 404

    def test_distinct_inputs_resolving_to_same_fighter_rejected_400(self):
        """WR-01 regression: two different inputs that resolve to the same
        Fighter row must be rejected 400 (mirrors api/routers/matchup.py:74).

        Mocks search_fighters to return a single Fighter row with the same
        id for both inputs ("Khabib" vs "Khabib Nurmagomedov"), simulating
        the post-resolution collision case that the raw-string compare alone
        would miss.
        """
        from types import SimpleNamespace

        # Build an app that exposes a real (but no-op) DB session so the
        # resolution branch is reached. Using SimpleNamespace as a sentinel
        # — search_fighters is patched, so the session contents do not matter.
        app = FastAPI()
        app.include_router(v1_predict.router, prefix="/api/v1")

        def _sentinel_db():
            yield SimpleNamespace()

        app.dependency_overrides[get_db] = _sentinel_db
        c = TestClient(app)

        # Single Fighter row with id=42 — both lookups resolve to it.
        shared_fighter = SimpleNamespace(id=42, name="Khabib Nurmagomedov")

        with patch(
            "ufc_prediction.api.v1.predict.search_fighters",
            return_value=[shared_fighter],
        ):
            resp = c.post(
                "/api/v1/predict",
                json={"fighter_a": "Khabib", "fighter_b": "Khabib Nurmagomedov"},
            )

        assert resp.status_code == 400, resp.text
        assert "different fighters" in resp.json()["detail"]


class TestPredictV1:
    """Integration tests using tests/api/conftest.py TestClient fixture.

    These require Docker / testcontainers. In environments where Docker is
    unavailable, the `client` fixture itself fails and pytest reports an
    ERROR rather than a FAIL — those errors are environmental and pre-date
    Phase 25 (matches behavior of tests/api/test_fighters.py etc.).
    """

    def test_predict_happy_path(self, client: TestClient, seed_api_data):
        """Happy path — known fighters return 200 + PredictorOutputV1 shape."""
        with patch(
            "ufc_prediction.api.v1.predict._get_predictor",
            return_value=_FakePredictor(),
        ):
            resp = client.post(
                "/api/v1/predict",
                json={
                    "fighter_a": "Khabib Nurmagomedov",
                    "fighter_b": "Conor McGregor",
                    "event_date": "2018-10-06",
                },
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Plan 35-05 CONTRACT-V24-02: default accept_schema_version is "1.2.0"
        # so the route now echoes "1.2.0" when no negotiation field is sent.
        assert body["schema_version"] == "1.2.0"
        assert 0.0 <= body["win_probability"] <= 1.0
        assert body["fighter_a"] == "Khabib Nurmagomedov"
        assert body["fighter_b"] == "Conor McGregor"
        assert "base_prob" in body
        assert "meta_prob" in body
        assert "meta_learner_version" in body
        assert "meta_skipped_reason" in body

    def test_same_fighter_rejected_400(self, client: TestClient, seed_api_data):
        """V11 ASVS — same-fighter request must 400.

        Note: the route rejects same-fighter BEFORE calling _get_predictor,
        so this case doesn't need the patch — but we apply it defensively
        to ensure the singleton isn't lazy-initialized as a side effect.
        """
        with patch(
            "ufc_prediction.api.v1.predict._get_predictor",
            return_value=_FakePredictor(),
        ):
            resp = client.post(
                "/api/v1/predict",
                json={"fighter_a": "Khabib Nurmagomedov", "fighter_b": "khabib nurmagomedov"},
            )
        assert resp.status_code == 400

    def test_unknown_fighter_returns_404(self, client: TestClient, seed_api_data):
        """ValueError from predictor → 404."""
        with patch(
            "ufc_prediction.api.v1.predict._get_predictor",
            return_value=_FakePredictor(),
        ):
            resp = client.post(
                "/api/v1/predict",
                json={"fighter_a": "Nonexistent Fighter", "fighter_b": "Conor McGregor"},
            )
        assert resp.status_code == 404

    def test_empty_fighter_returns_422(self, client: TestClient, seed_api_data):
        """V5 ASVS — Pydantic Field(min_length=1) → 422.

        Validation runs before any route logic, so no patch needed.
        """
        resp = client.post(
            "/api/v1/predict",
            json={"fighter_a": "", "fighter_b": "Conor McGregor"},
        )
        assert resp.status_code == 422

    def test_openapi_includes_predict_path_and_is_31(
        self,
        client: TestClient,
    ):
        """OpenAPI spec includes /api/v1/predict and emits 3.1.0."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        spec = resp.json()
        assert spec["openapi"] == "3.1.0"
        assert "/api/v1/predict" in spec["paths"]

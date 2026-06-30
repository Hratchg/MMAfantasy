"""Phase 25 PARTNER-V22-03 CI gate: openapi-spec-validator>=0.8.5.

Validates the committed src/ufc_prediction/contracts/openapi.v1.0.0.json
against OpenAPI 3.1.0. Includes a negative test (Pitfall #7) to confirm
the validator isn't silently passing on garbage.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from openapi_spec_validator import validate
from openapi_spec_validator.validation.exceptions import (
    OpenAPISpecValidatorError,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OPENAPI_PATH = REPO_ROOT / "src" / "ufc_prediction" / "contracts" / "openapi.v1.0.0.json"


class TestOpenAPIValidation:
    def test_committed_openapi_validates(self):
        """The committed openapi.v1.0.0.json passes openapi-spec-validator."""
        spec = json.loads(OPENAPI_PATH.read_text())
        assert spec["openapi"] == "3.1.0"
        validate(spec)  # raises on failure

    def test_validator_rejects_missing_openapi_field(self):
        """Pitfall #7 negative test — validator MUST reject garbage.

        WR-06 (Phase 25 review-fix): narrowed from `pytest.raises(Exception)`
        to OpenAPISpecValidatorError (the canonical base class in
        openapi-spec-validator >=0.7.0). A bare Exception match also
        accepts AttributeError / TypeError / SystemExit which would mask
        bugs in the test-setup code path. If a future release changes the
        exception hierarchy, this assertion will fail loudly — that's the
        intended behavior of a validator-pinned CI gate.
        """
        spec = json.loads(OPENAPI_PATH.read_text())
        broken = copy.deepcopy(spec)
        del broken["openapi"]  # remove required top-level field
        with pytest.raises(OpenAPISpecValidatorError):
            validate(broken)

    def test_predict_path_present_in_committed_spec(self):
        """OpenAPI surface includes /api/v1/predict (PARTNER-V22-03)."""
        spec = json.loads(OPENAPI_PATH.read_text())
        assert "/api/v1/predict" in spec["paths"]
        predict_op = spec["paths"]["/api/v1/predict"]
        assert "post" in predict_op  # POST method registered

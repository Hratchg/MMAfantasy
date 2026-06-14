"""Phase 25 PARTNER-V22-04 forward-compat fuzz suite.

Four test classes:
  - TestRoundTripFuzz: Hypothesis @given 100 random valid responses
    (st.builds — pre-validated instances; catches serialization bugs).
  - TestRawDictFuzz: Hypothesis @given 100 random raw-JSON dicts fed
    DIRECTLY to PredictorOutputV1.model_validate(...) (WR-04 Phase 25
    review-fix: catches DESERIALIZATION bugs the pre-validated builds
    suite misses — float boundaries, unicode edge cases, type coercion).
  - TestXgbV2OnlyMock: current production response validates
  - TestMetaV22ActiveMock: forward-compat lock against mocked META response (D-09)

Per CONTEXT D-08 + D-09; RESEARCH Pattern 3 + Pitfall #5.
"""
from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from ufc_prediction.api.v1.models import PredictorOutputV1

# WR-04 (Phase 25 review-fix): exclude surrogate code points ("Cs" category)
# from text strategies. Lone surrogates round-trip through model_dump_json
# as \uD800-style escapes that json.loads accepts but downstream consumers
# may not — a known source of flaky Hypothesis seeds.
_SAFE_TEXT = st.text(
    min_size=1,
    max_size=200,
    alphabet=st.characters(blacklist_categories=("Cs",)),
)
_SAFE_TEXT_SHORT = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(blacklist_categories=("Cs",)),
)
_SAFE_TEXT_MEDIUM = st.text(
    min_size=1,
    max_size=100,
    alphabet=st.characters(blacklist_categories=("Cs",)),
)


class TestRoundTripFuzz:
    """100 Hypothesis-generated PredictorOutputV1 instances round-trip.

    Per RESEARCH Assumption A1 fallback: st.from_type(PredictorOutputV1) hits
    a Hypothesis limitation with Optional[float] + ge/le metadata (TypeError
    comparing None to bounded float). Use explicit st.builds(...) with per-
    field strategies — equivalent surface coverage, deterministic generation.

    NOTE: instances generated here are PRE-VALIDATED via the Pydantic
    constructor — the round-trip only catches serialization bugs (Pydantic
    is internally consistent). Deserialization-side edge cases are exercised
    separately in TestRawDictFuzz below (WR-04 Phase 25 review-fix).
    """

    @given(
        instance=st.builds(
            PredictorOutputV1,
            schema_version=st.just("1.0.0"),
            win_probability=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            fighter_a=_SAFE_TEXT,
            fighter_b=_SAFE_TEXT,
            event_date=st.dates(),
            base_prob=st.one_of(
                st.none(),
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            ),
            meta_prob=st.one_of(
                st.none(),
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            ),
            meta_learner_version=st.one_of(
                st.none(),
                _SAFE_TEXT_SHORT,
            ),
            meta_skipped_reason=st.one_of(
                st.none(),
                _SAFE_TEXT_MEDIUM,
            ),
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_round_trip_pydantic(self, instance: PredictorOutputV1):
        """Any valid PredictorOutputV1 → JSON → PredictorOutputV1 is identity."""
        payload = json.loads(instance.model_dump_json())
        rehydrated = PredictorOutputV1.model_validate(payload)
        assert rehydrated.model_dump() == instance.model_dump()


class TestRawDictFuzz:
    """100 Hypothesis-generated raw JSON dicts → PredictorOutputV1.model_validate.

    WR-04 (Phase 25 review-fix): the st.builds(...) suite above pre-validates
    every instance through the Pydantic constructor, so it only exercises
    SERIALIZATION (model_dump_json → json.loads → model_validate is identity
    by construction). The DESERIALIZATION surface (model_validate(payload)
    where payload is a freely-constructed dict) is where partner-shaped
    edge cases land — float boundaries (0.0, 1.0), unicode in string fields,
    optional-field absence vs explicit-null, ISO date strings — and that
    surface is not touched by st.builds.

    This suite builds raw dicts that match the v1 contract shape (per
    predictor.schema.v1.0.0.json) and feeds them directly to
    model_validate. The round-trip assertion confirms model_dump_json
    re-emits the same shape (modulo type coercion: floats stay floats,
    dates serialize back to ISO strings).
    """

    @given(
        payload=st.fixed_dictionaries({
            "schema_version": st.just("1.0.0"),
            "win_probability": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            "fighter_a": _SAFE_TEXT,
            "fighter_b": _SAFE_TEXT,
            "event_date": st.dates().map(lambda d: d.isoformat()),
            "base_prob": st.one_of(
                st.none(),
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            ),
            "meta_prob": st.one_of(
                st.none(),
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            ),
            "meta_learner_version": st.one_of(
                st.none(),
                _SAFE_TEXT_SHORT,
            ),
            "meta_skipped_reason": st.one_of(
                st.none(),
                _SAFE_TEXT_MEDIUM,
            ),
        }),
    )
    @settings(max_examples=100, deadline=None)
    def test_validate_from_raw_dict(self, payload: dict):
        """Raw dict (v1.0.0 partner-shaped JSON) round-trips through model_validate.

        Post-Phase-32 D-04: the live model emits a SUPERSET (v1.1.0 fields
        gate_contract_ref / model_candidates / phase_chain_audit_sha
        default to None). v1.0.0 partners send v1.0.0-shape payloads and
        their deserializers (built against predictor.schema.v1.0.0.json)
        either ignore the extra fields or surface them as None. The
        forward-compat guarantee is that the v1.0.0 KEY SUBSET round-trips
        identically — which is what this assertion now verifies.
        """
        obj = PredictorOutputV1.model_validate(payload)
        # After validation, re-serialize and project down to the v1.0.0
        # field set (Phase 25 forward-compat lock binding). The event_date
        # round-trips through date object; all other v1.0.0 fields are
        # JSON-native.
        rehydrated_full = json.loads(obj.model_dump_json())
        v100_keys = set(payload.keys())
        rehydrated_v100 = {k: rehydrated_full[k] for k in v100_keys}
        assert rehydrated_v100 == payload
        # Phase 32 D-04 invariant: when the v1.1.0 additive trio is not
        # present in the input payload, the dump surfaces them as None.
        for k in ("gate_contract_ref", "model_candidates", "phase_chain_audit_sha"):
            assert rehydrated_full[k] is None, (
                f"v1.0.0 payload should leave v1.1.0 field {k!r} as None; "
                f"got {rehydrated_full[k]!r}"
            )

    def test_validate_boundary_win_probability_zero(self):
        """ge=0.0 boundary — partner JSON with 0.0 must validate."""
        payload = {
            "schema_version": "1.0.0",
            "win_probability": 0.0,
            "fighter_a": "A",
            "fighter_b": "B",
            "event_date": "2026-01-01",
            "base_prob": 0.0,
            "meta_prob": None,
            "meta_learner_version": None,
            "meta_skipped_reason": "no_meta_artifact",
        }
        obj = PredictorOutputV1.model_validate(payload)
        assert obj.win_probability == 0.0
        assert obj.base_prob == 0.0

    def test_validate_boundary_win_probability_one(self):
        """le=1.0 boundary — partner JSON with 1.0 must validate."""
        payload = {
            "schema_version": "1.0.0",
            "win_probability": 1.0,
            "fighter_a": "A",
            "fighter_b": "B",
            "event_date": "2026-01-01",
            "base_prob": 1.0,
            "meta_prob": None,
            "meta_learner_version": None,
            "meta_skipped_reason": "no_meta_artifact",
        }
        obj = PredictorOutputV1.model_validate(payload)
        assert obj.win_probability == 1.0
        assert obj.base_prob == 1.0


class TestXgbV2OnlyMock:
    """Current production response (meta skipped, no meta artifact)."""

    def test_validates_against_committed_schema(
        self, schema_validator, xgb_v2_only_mock,
    ):
        """The committed schema file accepts the current production payload."""
        schema_validator.validate(xgb_v2_only_mock)

    def test_round_trips_through_pydantic(self, xgb_v2_only_mock):
        obj = PredictorOutputV1.model_validate(xgb_v2_only_mock)
        assert obj.meta_prob is None
        assert obj.meta_skipped_reason == "no_meta_artifact"
        # base_prob == win_probability in xgb_v2-only mode
        assert obj.base_prob == obj.win_probability


class TestMetaV22ActiveMock:
    """Forward-compat lock — schema must accept META-V22-active payload (D-09).

    If THIS test fails after Phase 26 lands the real META response, the
    regression is in Phase 26 (or this mock is out of date); Position 5a
    exists exactly to surface that mismatch BEFORE promotion.
    """

    def test_validates_against_committed_schema(
        self, schema_validator, meta_v22_active_mock,
    ):
        """The committed schema accepts the forward-compat META payload."""
        schema_validator.validate(meta_v22_active_mock)

    def test_round_trips_through_pydantic(self, meta_v22_active_mock):
        obj = PredictorOutputV1.model_validate(meta_v22_active_mock)
        assert obj.meta_learner_version == "v22.1"
        assert obj.meta_prob == 0.7234
        # META-active discriminator: base != win_probability
        assert obj.base_prob != obj.win_probability

    def test_meta_active_payload_has_required_discriminators(
        self, meta_v22_active_mock,
    ):
        """Sanity — the mock encodes the meta-active scenario correctly."""
        assert meta_v22_active_mock["meta_learner_version"] == "v22.1"
        assert meta_v22_active_mock["meta_prob"] is not None
        assert meta_v22_active_mock["meta_skipped_reason"] is None
        assert (
            meta_v22_active_mock["base_prob"]
            != meta_v22_active_mock["win_probability"]
        )

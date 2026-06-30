"""Partner-facing v1 contract models (Phase 25 D-01/02 + Phase 32 D-04 + Phase 35 CONTRACT-V24-02).

PredictorOutputV1 is the SOURCE OF TRUTH for the PARTNER schema. Phase 25
emitted predictor.schema.v1.0.0.json from this model. Phase 32 D-04 extends
the model with an additive trio (gate_contract_ref, model_candidates,
phase_chain_audit_sha — all Optional with None defaults) and emits
predictor.schema.v1.1.0.json as a sibling artifact. Phase 35 CONTRACT-V24-02
extends the model with an additive Optional `prediction_metadata` block (6
partner-facing observability fields) and emits predictor.schema.v1.2.0.json
as a sibling artifact. The v1.0.0 + v1.1.0 files are BYTE-FROZEN (Phase 25
forward-compat lock binding) — older partners see the response shape
unchanged because the additive fields default to None.

Forward-compat fields per D-05(P19): base_prob / meta_prob /
meta_learner_version / meta_skipped_reason all Optional so xgb_v2-only
AND META-V22-active responses both validate (Pitfall #5 prevention).
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ufc_prediction.api.disclaimer import DISCLAIMER_200W


class PredictMatchupRequestV1(BaseModel):
    """Request body for POST /api/v1/predict."""

    # extra="ignore" (Pydantic v2 default) — explicit for clarity; do NOT
    # set extra="allow" (RESEARCH Security Domain: schema-bypass risk).
    model_config = ConfigDict(extra="ignore")

    fighter_a: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Fighter A name (resolved via search_fighters).",
    )
    fighter_b: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Fighter B name.",
    )
    event_date: date | None = Field(
        None,
        description="As-of date for the prediction. Defaults to today() if omitted.",
    )

    # ── Phase 35 CONTRACT-V24-02 — schema-version negotiation ─────────────
    # Default "1.2.0" so new clients automatically get the additive
    # prediction_metadata block. Partners still on v1.0.0 / v1.1.0 contracts
    # opt out by passing "1.0.0" or "1.1.0" (Phase 25 forward-compat lock).
    accept_schema_version: Literal["1.0.0", "1.1.0", "1.2.0", "1.3.0"] = Field(
        default="1.2.0",
        description=(
            "Response schema_version to emit. Default 1.2.0; partners on "
            "older contracts pass '1.0.0' or '1.1.0' to opt out of new fields. "
            "v1.3.0 is identical to v1.2.0 on the success path (byte-identical "
            "via forward-compat lock); v1.3.0 partners additionally opt in to "
            "the RFC 7807 application/problem+json error wrapper by sending "
            "`Accept: application/problem+json` on the request. See Phase 52 "
            "API-V26-01 / API-V26-02."
        ),
    )


class PredictionMetadataV12(BaseModel):
    """CONTRACT-V24-02: additive v1.2.0 prediction_metadata block.

    Six partner-facing observability fields. Forward-compat: defaults to
    None on PredictorOutputV1, so v1.0.0 + v1.1.0 partners see this field
    absent / null (Phase 25 forward-compat lock).
    """

    model_config = ConfigDict(extra="ignore")

    odds_source: Literal["live", "cache", "nan"] = Field(
        ...,
        description=(
            "Source of the BFO closing-odds feature at predict time: "
            "'live' (just-fetched), 'cache' (DB-stashed), 'nan' (absent)."
        ),
    )
    odds_timestamp_iso: str | None = Field(
        default=None,
        description=(
            "ISO-8601 timestamp of when the odds row was fetched/cached. "
            "None when odds_source == 'nan'."
        ),
    )
    fighter_a_n_ufc_fights: int = Field(
        ...,
        ge=0,
        description="Count of fighter_a's UFC fights at predict time.",
    )
    fighter_b_n_ufc_fights: int = Field(
        ...,
        ge=0,
        description="Count of fighter_b's UFC fights at predict time.",
    )
    is_debutant_either: bool = Field(
        ...,
        description="True if either fighter has zero prior UFC fights.",
    )
    calibration_slice_brier: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Brier score of the calibration slice whose date-window "
            "contains event_date (from models/meta/meta_v2_meta.json). "
            "None when no slice metadata is available."
        ),
    )


class PredictorOutputV1(BaseModel):
    """Response body for POST /api/v1/predict.

    Two production response shapes (both validate against this single schema —
    base_prob / meta_prob / meta_learner_version / meta_skipped_reason are
    Optional per Phase 25 D-05(P19) forward-compat — Pitfall #5 prevention):

    1. **META-V22-active (default since Phase 26, 2026-05-17):** the
       meta-learner `models/meta/meta_v2.joblib` loads on startup; for any
       matchup where BFO closing odds yield a non-NaN `closing_prob_diff`
       feature, the meta is invoked and `win_probability == meta_prob`
       (calibrated logistic stack over xgb_v2 OOF + Elo + closing odds + style
       differentials). `meta_skipped=False`, `meta_skipped_reason=None`,
       `meta_learner_version="v2"`.

    2. **xgb_v2 base fallback:** triggers in two cases — (a) `closing_prob_diff`
       is NaN at predict time (no BFO closing odds for the matchup; typical
       for early-week or novel matchups before lines firm up), in which case
       `meta_skipped_reason="nan_closing_prob_diff"`; (b) the meta artifact is
       absent from disk, in which case `meta_skipped_reason="no_meta_artifact"`
       (a deployment misconfiguration; should not occur in production). In
       both cases `win_probability == base_prob == xgb_v2.predict_proba(...)`.

    The same PredictorOutputV1 type covers both. Clients reading
    `meta_skipped_reason` can detect which path served their request.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(
        "1.0.0",
        pattern=r"^\d+\.\d+\.\d+$",
        description="Envelope minor version (per D-03; URL is major).",
    )
    win_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Canonical user-facing probability fighter_a wins (D-05(P19)).",
    )
    fighter_a: str = Field(..., min_length=1, max_length=200)
    fighter_b: str = Field(..., min_length=1, max_length=200)
    event_date: date

    # D-05(P19) observability fields — Optional for forward-compat with META-V22
    base_prob: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Raw XGBoost probability before meta-learner wrapping.",
    )
    meta_prob: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Meta-learner output probability; None when meta is skipped.",
    )
    meta_learner_version: str | None = Field(
        None,
        description="Meta-learner version (e.g., 'v22.1') when meta is active; None otherwise.",
    )
    meta_skipped_reason: str | None = Field(
        None,
        description="Reason meta was skipped (e.g., 'no_meta_artifact'); None when meta is active.",
    )

    # ── Phase 32 D-04 additive trio (PARTNER-V23-01) ──────────────────────
    # All three Optional with None defaults so v1.0.0 partners are unaffected
    # (Phase 25 forward-compat lock binding). v1.1.0 partners can populate
    # them by opting into the v1.1.0 schema file; the default /api/v1/predict
    # route leaves them as None for v1.0.0-shape responses.
    gate_contract_ref: str | None = Field(
        default=None,
        description=(
            "Path/version of gate contract used at prediction time "
            "(e.g., '.planning/gate_contract_v2.3.json'). "
            "Phase 32 D-04 (PARTNER-V23-01)."
        ),
    )
    model_candidates: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "Supported model candidates with byte-identity SHAs. "
            "Each entry: {name: str, sha256: str, phase: str}. "
            "Phase 32 D-04 (PARTNER-V23-01)."
        ),
    )
    phase_chain_audit_sha: str | None = Field(
        default=None,
        description=(
            "AUDIT-01 chain leaf SHA (xgb_v2.joblib byte-identity). Phase 32 D-04 (PARTNER-V23-01)."
        ),
    )

    # ── Phase 35 CONTRACT-V24-02 additive v1.2.0 prediction_metadata block ──
    # Default None so v1.0.0 + v1.1.0 partners see this field as null/absent
    # (Phase 25 forward-compat lock binding). The /api/v1/predict route only
    # populates this when the request body carries accept_schema_version
    # == "1.2.0". Plan 35-06's CONTRACT-V24-03 regression test pins byte-
    # shape for older partner schemas with this field None.
    prediction_metadata: PredictionMetadataV12 | None = Field(
        default=None,
        description=(
            "v1.2.0 additive observability block. Populated when client "
            "passes accept_schema_version='1.2.0'; None for v1.0.0/v1.1.0."
        ),
    )

    # ── Phase 38 HYGIENE-V24-02 entertainment-disclaimer footer ──────────
    # Top-level disclaimer; populated by default with the shared
    # DISCLAIMER_200W constant from ufc_prediction.api.disclaimer.
    # Additive to v1.2.0 per CONTEXT.md `<decisions>` "Entertainment
    # Disclaimer → placement 3". The field is NOT gated on
    # accept_schema_version — it appears in every response. Forward-
    # compat: v1.0.0 + v1.1.0 partners using `extra="ignore"` validators
    # accept the new key transparently (Phase 25 lock binding); see
    # tests/contracts/test_forward_compat_v1_2_0.py Test 6 for
    # CONTRACT-V24-03 re-verification under the disclaimer addition.
    disclaimer: str = Field(
        default=DISCLAIMER_200W,
        description=(
            "Entertainment / no-warranty / no-financial-advice "
            "disclaimer (HYGIENE-V24-02). Always populated; "
            "partners may surface or ignore. Constant string equal "
            "to ufc_prediction.api.disclaimer.DISCLAIMER_200W."
        ),
    )


class ProblemDetailsV13(BaseModel):
    """Phase 52 API-V26-01 — RFC 7807 application/problem+json error wrapper.

    Additive opt-in surface for PARTNER schema v1.3.0. Existing partners
    on v1.0.0 / v1.1.0 / v1.2.0 continue to receive the FastAPI default
    ``{"detail": "..."}`` shape on errors; v1.3.0 partners who additionally
    send ``Accept: application/problem+json`` get this RFC-shaped error
    body with ``Content-Type: application/problem+json``.

    RFC 7807 §3.1 member set: ``type``, ``title``, ``status``, ``detail``,
    ``instance``. Extension members are allowed per §3.2 (``extra="allow"``).

    Forward-compat: this model is NEVER returned on the success path —
    only via the opt-in exception handler registered in
    ``ufc_prediction.api.app``. Success-path bytes remain byte-identical
    to v1.2.0 across the entire forward-compat lock (v1.0.0 / v1.1.0 /
    v1.2.0 / v1.3.0). See `tests/contracts/test_forward_compat_v1_2_0.py`
    (v2.6.1 follow-on will extend the regression to pin v1.3.0 bytes).
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(
        default="about:blank",
        description=(
            "URI reference identifying the problem type. Defaults to "
            '"about:blank" per RFC 7807 §4.2 when no problem-specific '
            "URI is defined for this error."
        ),
    )
    title: str = Field(
        ...,
        description=(
            "Short, human-readable summary of the problem type. "
            "Per RFC 7807 §3.1: SHOULD NOT change between occurrences "
            "of the same problem type."
        ),
    )
    status: int = Field(
        ...,
        ge=100,
        le=599,
        description="HTTP status code generated by the origin server.",
    )
    detail: str | None = Field(
        default=None,
        description=(
            "Human-readable explanation specific to this occurrence of "
            "the problem. Per RFC 7807 §3.1: focused on helping the "
            "client correct the problem; NOT a stable identifier."
        ),
    )
    instance: str | None = Field(
        default=None,
        description=(
            "URI reference identifying the specific occurrence of the "
            "problem. May be relative to the request URI (RFC 7807 §3.1)."
        ),
    )

"""Phase 25 D-06: /api/v1/predict adapter route.

LIVE-03 discipline — predictor.py is NEVER modified; this route imports
ModelPredictor and wraps its dict output in PredictorOutputV1 (D-01 +
D-02 forward-compat).
"""

from __future__ import annotations

from datetime import date
from functools import cache

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ufc_prediction.api.deps import get_db
from ufc_prediction.api.v1.models import (
    PredictionMetadataV12,
    PredictMatchupRequestV1,
    PredictorOutputV1,
)
from ufc_prediction.elo.fighter_queries import search_fighters
from ufc_prediction.ml.order_invariant import predict_order_invariant
from ufc_prediction.ml.predictor import ModelPredictor

router = APIRouter(tags=["predict"])


@cache
def _get_predictor() -> ModelPredictor:
    """Module-level singleton (Pitfall #6 mitigation).

    Lazy-init on first request; subsequent requests reuse the cached
    instance. Test fixtures can override via patch / dependency injection.

    WR-02 (Phase 25 review-fix): uses @functools.cache instead of a manual
    `global _PREDICTOR` sentinel. FastAPI runs sync route handlers in a
    threadpool, so two concurrent first-requests could both observe a
    None sentinel and each construct a ModelPredictor — racing through
    model load, meta-load, feature-column drift assertion, and SHA
    recomputation. @functools.cache is thread-safe under the GIL (lookup
    + call are atomic to other Python threads), so only one cold-start
    construction runs no matter how many concurrent first-requests arrive.

    To reset between tests, call `_get_predictor.cache_clear()`.
    """
    return ModelPredictor(model_dir="models", version="v2")


@router.post("/predict", response_model=PredictorOutputV1)
def predict(
    body: PredictMatchupRequestV1,
    db: Session = Depends(get_db),
) -> PredictorOutputV1:
    """Predict fight outcome for fighter_a vs fighter_b.

    Forward-compatible with future META-V22-active responses (Phase 25 D-02).
    """
    # V11 ASVS — reject same-fighter matchup. Two-tier check (WR-01 Phase 25
    # review-fix): (1) raw-string compare catches identical inputs without
    # any DB round-trip, including in tests/no-DB harnesses; (2) when a DB
    # session is available, resolve via search_fighters and compare ORM ids
    # — this matches api/routers/matchup.py:74 and catches the case where
    # two different inputs (e.g. "Khabib" vs "Khabib Nurmagomedov") both
    # resolve to the same Fighter row. Resolution failures are swallowed
    # here so the route falls through to predictor.predict() (which raises
    # ValueError → 404), preserving the existing error envelope.
    if body.fighter_a.strip().lower() == body.fighter_b.strip().lower():
        raise HTTPException(
            status_code=400,
            detail="fighter_a and fighter_b must be different fighters",
        )

    if db is not None:
        try:
            matches_a = search_fighters(db, body.fighter_a)
            matches_b = search_fighters(db, body.fighter_b)
        except Exception:  # noqa: BLE001 — fall through to predictor for any DB error
            matches_a = matches_b = []
        if len(matches_a) == 1 and len(matches_b) == 1 and matches_a[0].id == matches_b[0].id:
            raise HTTPException(
                status_code=400,
                detail="fighter_a and fighter_b must be different fighters",
            )

    ev_date = body.event_date or date.today()
    try:
        result = predict_order_invariant(
            _get_predictor(),
            db,
            body.fighter_a,
            body.fighter_b,
            event_date=ev_date,
        )
    except ValueError as exc:
        # Predictor raises ValueError on unknown fighter / no Elo / etc.
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # ── Phase 35 CONTRACT-V24-02 — schema-version negotiation ─────────────
    # Pydantic default on the request body is "1.2.0", so partners get the
    # new prediction_metadata block out of the box. Partners still on the
    # v1.0.0 / v1.1.0 contract opt out by passing accept_schema_version
    # explicitly. Forward-compat lock: when negotiated version <= "1.1.0"
    # the prediction_metadata field stays None — v1.0.0 + v1.1.0 partners
    # observe the byte-shape Phase 25 + Phase 32 D-04 promised.
    #
    # Phase 69 API-V261-01 (v2.6.1): v1.3.0 routes the SAME populated
    # prediction_metadata as v1.2.0. The v1.3.0 capability is on the error
    # path (RFC 7807 problem+json wrapper, Phase 52 API-V26-01), opt-in via
    # the Accept header; the success-path bytes are byte-identical to v1.2.0
    # per the v2.6.1 invariant #3. Phase 52 extended the Literal to admit
    # "1.3.0" but did NOT wire v1.3.0 through the negotiation branch — this
    # would leave v1.3.0 partners seeing prediction_metadata=null silently,
    # breaking the invariant. Phase 69 closes that gap.
    requested_version = body.accept_schema_version
    pred_meta_dict = result.get("prediction_metadata") or {}
    if requested_version in ("1.2.0", "1.3.0"):
        # Rename predictor dict → Pydantic model. The predictor's dict has
        # 7 keys (including win_probability_source); the partner-facing
        # model surfaces only the 6 CONTEXT-listed observability fields.
        prediction_metadata_block: PredictionMetadataV12 | None = PredictionMetadataV12(
            odds_source=pred_meta_dict.get("odds_source", "nan"),
            odds_timestamp_iso=pred_meta_dict.get("odds_timestamp_iso"),
            fighter_a_n_ufc_fights=pred_meta_dict.get(
                "fighter_a_n_ufc_fights",
                0,
            ),
            fighter_b_n_ufc_fights=pred_meta_dict.get(
                "fighter_b_n_ufc_fights",
                0,
            ),
            is_debutant_either=pred_meta_dict.get(
                "is_debutant_either",
                False,
            ),
            calibration_slice_brier=pred_meta_dict.get(
                "calibration_slice_brier",
            ),
        )
    else:
        # v1.0.0 / v1.1.0 forward-compat lock — leave the block null. Plan
        # 35-06 builds the CONTRACT-V24-03 regression test that pins this
        # byte-shape against the Phase 25 + Phase 32 D-04 reference fixtures.
        prediction_metadata_block = None

    return PredictorOutputV1(
        # Negotiated version echoed verbatim so the partner can audit which
        # contract served their request (STRIDE T-35-05-03 mitigation).
        schema_version=requested_version,
        win_probability=result["win_probability"],
        fighter_a=result["fighter_a"],
        fighter_b=result["fighter_b"],
        event_date=ev_date,
        base_prob=result.get("base_prob"),
        meta_prob=result.get("meta_prob"),
        meta_learner_version=result.get("meta_learner_version"),
        meta_skipped_reason=result.get("meta_skipped_reason"),
        prediction_metadata=prediction_metadata_block,
    )

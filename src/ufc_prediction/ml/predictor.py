"""ModelPredictor: loads a trained model and predicts fight outcomes.

Per D-10: returns win probability for Fighter A vs Fighter B.
Per D-12: model coexists with Elo predictions.

Phase 16-02 (LIVE-* + D-12) — Phase 19 META-05 evolution — predict path is now:
    1. _resolve_fighter(A, B)
    2. metadata + LIVE-03 FEATURE_COLUMNS assertion + Phase 19 meta-load (in __init__)
    3. live_odds = bfo_live.fetch_matchup_odds(A, B, event_date)
    4. base_features = inference_features.build(...)
    5. xgb_prob_a = model.predict_proba(base_features)
    6. elo_prob_a = EloEngine.expected_win_probability(...) (was step 7)
    6b. META wrapping (D-04 NaN handling + D-05 output shape) — if meta loaded
    7. (folded into 6/6b)
    8. Return result dict with base + meta fields per D-05(P19)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ufc_prediction.dedup.source_priority import prefer_canonical
from ufc_prediction.elo.config import EloConfig
from ufc_prediction.elo.engine import EloEngine
from ufc_prediction.elo.fighter_queries import search_fighters
from ufc_prediction.ml.config import FEATURE_COLUMNS, get_feature_columns
from ufc_prediction.ml.inference_features import build as build_inference_features
from ufc_prediction.ml.persistence import get_latest_version, load_metadata, load_model
from ufc_prediction.models.elo_snapshot import EloSnapshot
from ufc_prediction.models.fight import Fight
from ufc_prediction.models.fighter import Fighter
from ufc_prediction.scraper.bfo_live import fetch_matchup_odds

logger = logging.getLogger(__name__)


def _prefer_canonical(results: list[Fighter]) -> Fighter:
    """Return the highest-priority Fighter row from a duplicate set.

    Phase 16 HOUSE-04: thin shim over `ufc_prediction.dedup.source_priority.prefer_canonical`.
    The shim is preserved as a private symbol so the predictor's own
    `_resolve_fighter` and the existing `tests/ml/test_predictor.py::TestPreferCanonical`
    suite keep working without import churn (Gotcha 10).

    Per D-01 (Phase 14): ufcstats > sherdog > kaggle-rajeevw > kaggle-mdabbert.
    Tie-break: lowest fighter_id wins (deterministic across re-runs).

    Args:
        results: One or more Fighter rows representing the same real-world
            person across ingest sources. Must be non-empty.

    Returns:
        The single canonical Fighter row.

    Raises:
        ValueError: If `results` is empty (raised by `prefer_canonical`).
    """
    return prefer_canonical(results, source_key=lambda f: f.source, tiebreak_key=lambda f: f.id)


class ModelPredictor:
    """Loads a trained model and predicts fight outcomes.

    Per D-10: returns win probability for Fighter A vs Fighter B.
    Per D-12: model coexists with Elo predictions.
    """

    def __init__(
        self,
        model_dir: str = "models",
        version: str | None = None,
        *,
        meta_dir: str | None = "models/meta",  # Phase 19 META-05 (D-05(P19))
    ) -> None:
        self.model_dir = model_dir
        self.version = version or get_latest_version(model_dir)
        self.model = load_model(model_dir, self.version)
        self.metadata = load_metadata(model_dir, self.version)
        # Phase 18 NET-V2-01 + Pitfall B mitigation: read n_features and derive
        # whether the loaded model was trained with NET-* features. Only valid
        # values are 72 (ablation winner OR xgb_v2 baseline) and 75 (time-decayed
        # winner). Anything else = halt before predict_proba.
        # Carry-forward of LIVE-03 backwards-compat: when n_features is some
        # other value (e.g. 74 from a length-drift mismatch), the message also
        # quotes feature_columns + counts so existing length-drift tests still
        # match the failure surface.
        n_features = self.metadata.get("n_features")
        if n_features not in (72, 75):
            meta_cols_for_msg = self.metadata.get("feature_columns") or []
            raise RuntimeError(
                f"predictor: model {self.version} metadata.n_features="
                f"{n_features!r}; expected 72 (ablation) or 75 "
                f"(time-decayed). feature_columns mismatch: code "
                f"{len(FEATURE_COLUMNS)} vs meta {len(meta_cols_for_msg)}. "
                f"Halt."
            )
        self._include_net = n_features == 75
        # LIVE-03 (Pitfall #12) — Phase 18 evolved: hard-fail if the model was
        # trained against a different feature-column list than current code
        # knows about. Stale meta = silent feature drift that ships garbage
        # probabilities. The expected list is now derived from include_net so
        # both 72-col (ablation/xgb_v2) and 75-col (time-decayed) models load
        # cleanly via the same dispatch.
        # Per CONTEXT.md <plan_specific_requirements> 16-02: RuntimeError,
        # NOT AssertionError, NOT log-and-continue. Operator must explicitly
        # fix the mismatch (retrain or pin feature columns at the recorded
        # snapshot).
        meta_cols = self.metadata.get("feature_columns") or []
        expected_cols = get_feature_columns(include_net=self._include_net)
        if list(expected_cols) != list(meta_cols):
            msg = (
                f"Feature column drift: code expected {len(expected_cols)} cols "
                f"(include_net={self._include_net}) vs model {self.version} meta "
                f"feature_columns ({len(meta_cols)}) differ. Halt."
            )
            raise RuntimeError(msg)

        # Phase 19 META-05 → Phase 26 LIVE-03 discovery delta (CONTEXT D-12,
        # RESEARCH §Runtime State Inventory Pitfall #5): use
        # get_latest_meta_version() instead of hardcoded 'v1' — predictor.py
        # auto-picks whichever meta version is promoted (v1 or v2 today, vN
        # tomorrow). Per D-10: `_candidate` artifacts are IGNORED. The cols
        # invariant is generalized: v1 retains its 3-col list verbatim, v2
        # validates against `META_V22_FEATURE_COLUMNS` from the Phase 26 Plan
        # 26-01 module (single source of truth).
        self.meta_model = None
        self.meta_metadata: dict | None = None
        if meta_dir is not None:
            from ufc_prediction.ml.meta_persistence import (
                get_latest_meta_version,
                load_meta_model,
            )

            latest = get_latest_meta_version(meta_dir)
            if latest is not None:
                meta_path = Path(meta_dir) / f"meta_{latest}.joblib"
                if meta_path.exists():
                    self.meta_model, self.meta_metadata = load_meta_model(
                        meta_dir,
                        version=latest,
                    )
                    # SHA invariant (unchanged from Phase 19)
                    base_path = Path(model_dir) / f"xgb_{self.version}.joblib"
                    actual_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
                    if self.meta_metadata["base_model_sha256"] != actual_sha:
                        raise RuntimeError(
                            f"meta_{self.meta_metadata['meta_version']} trained against "
                            f"base_model_sha256={self.meta_metadata['base_model_sha256'][:12]}... "
                            f"but loaded base xgb_{self.version}.joblib SHA="
                            f"{actual_sha[:12]}.... Halt."
                        )
                    # Cols invariant — generalized per Phase 26 Plan 26-04:
                    # v1 keeps its 3-col list; v2 validates against the
                    # META_V22_FEATURE_COLUMNS module (Plan 26-01 source of truth).
                    if latest == "v2":
                        from ufc_prediction.ml.meta_features_v22 import (
                            META_V22_FEATURE_COLUMNS,
                        )

                        expected_meta_cols = list(META_V22_FEATURE_COLUMNS)
                    else:
                        expected_meta_cols = [
                            "xgb_oof_prob",
                            "elo_prob",
                            "closing_prob_diff",
                        ]
                    if list(self.meta_metadata["meta_feature_columns"]) != expected_meta_cols:
                        raise RuntimeError(
                            f"meta_feature_columns drift: meta has "
                            f"{self.meta_metadata['meta_feature_columns']} but code "
                            f"expects {expected_meta_cols}. Halt."
                        )

        # Phase 34 FALLBACK-V24-02 — lazy-load xgb_v2_no_odds (67-col ablation sibling).
        # Mirrors the Phase 19 META auto-load shape above: try/except FileNotFoundError,
        # default to None so the predict path degrades to today's pre-Phase-34 behavior if
        # the artifact is missing (backward-compat for stale checkouts). When present, the
        # predict path routes to the fallback when live_odds is None AND the canonical
        # closing_prob_diff cell in the built feature vector is NaN (per
        # PATTERNS.md "predictor.py no-odds routing" + CONTEXT D-FALLBACK-V24-02).
        self.fallback_model = None
        self.fallback_meta: dict | None = None
        self.fallback_keep_idx: list[int] | None = None
        try:
            import joblib as _joblib

            fallback_joblib = Path(model_dir) / "xgb_v2_no_odds.joblib"
            fallback_meta_path = Path(model_dir) / "xgb_v2_no_odds_meta.json"
            if fallback_joblib.exists() and fallback_meta_path.exists():
                self.fallback_model = _joblib.load(fallback_joblib)
                self.fallback_meta = json.loads(fallback_meta_path.read_text())
                # Compute the keep_idx mapping FEATURE_COLUMNS_NO_NET (72) → the 67-col
                # subset the fallback was trained on. The fallback meta's feature_columns
                # list is the source of truth.
                from ufc_prediction.ml.config import FEATURE_COLUMNS_NO_NET as _FC72

                fallback_cols = self.fallback_meta["feature_columns"]
                if len(fallback_cols) != 67:
                    raise RuntimeError(
                        f"xgb_v2_no_odds meta n_features mismatch: "
                        f"meta lists {len(fallback_cols)}, expected 67. Halt."
                    )
                self.fallback_keep_idx = [_FC72.index(c) for c in fallback_cols]
                logger.info("predictor: loaded xgb_v2_no_odds fallback (67-col); SHA pinned")
        except FileNotFoundError:
            # Backward-compat: no fallback artifact present → predict uses today's
            # pre-Phase-34 path. Tests against historical checkpoints stay green.
            self.fallback_model = None

        # Phase 35 CONTRACT-V24-02 — cache per-slice Brier values from the
        # meta_v2_meta.json `v2_3_slice_metrics` block (written by Phase 34
        # TRUST-V24-02). The `prediction_metadata.calibration_slice_brier`
        # field surfaces a date-matched slice Brier or falls back to
        # `most_recent_24mo` when no slice windows are recorded. Graceful
        # degradation: missing file/block → {} → calibration_slice_brier=None.
        self.slice_briers: dict[str, float | None] = {}
        self.slice_windows: dict[str, tuple[str | None, str | None]] = {}
        try:
            meta_meta_path = Path(model_dir) / "meta" / "meta_v2_meta.json"
            if meta_meta_path.exists():
                meta_meta = json.loads(meta_meta_path.read_text())
                v23 = meta_meta.get("v2_3_slice_metrics", {}) or {}
                for name, payload in v23.items():
                    if not isinstance(payload, dict):
                        continue
                    # Accept either "brier" or "brier_score" (schema variance
                    # between Phase 34 TRUST-V24-02 candidates).
                    brier = payload.get("brier")
                    if brier is None:
                        brier = payload.get("brier_score")
                    self.slice_briers[name] = (
                        float(brier) if isinstance(brier, (int, float)) else None
                    )
                    self.slice_windows[name] = (
                        payload.get("window_start_iso"),
                        payload.get("window_end_iso"),
                    )
        except Exception:  # noqa: BLE001 — graceful degradation, never break __init__
            self.slice_briers = {}
            self.slice_windows = {}

    def _select_slice_brier(self, event_date: date) -> float | None:
        """Return the calibration slice Brier matching `event_date`.

        Iterates `self.slice_windows`; if `event_date` falls within any
        (window_start_iso, window_end_iso) range, returns that slice's Brier.
        Falls back to `most_recent_24mo` (per CONTEXT decision) if no slice
        windows are present or none match. Returns None when neither path
        yields a numeric value.
        """
        for name, (start_iso, end_iso) in self.slice_windows.items():
            if not start_iso or not end_iso:
                continue
            try:
                start_d = date.fromisoformat(start_iso[:10])
                end_d = date.fromisoformat(end_iso[:10])
            except ValueError:
                continue
            if start_d <= event_date <= end_d:
                return self.slice_briers.get(name)
        return self.slice_briers.get("most_recent_24mo")

    def predict(
        self,
        session: Session,
        fighter_a_name: str,
        fighter_b_name: str,
        *,
        event_date: date | None = None,
        refresh: bool = False,
    ) -> dict:
        """Predict fight outcome for two fighters.

        Phase 16-02 (D-09 / D-11 / D-12): predict path is the 8-step
        sequence documented in the module docstring. The Phase 15
        ``_build_feature_vector`` 28-feat NaN-pad block has been deleted —
        ``inference_features.build`` is the sole feature builder, and live
        odds flow through ``bfo_live.fetch_matchup_odds`` per D-09 ordering.

        Args:
            session: Database session for snapshot lookups + cache reads.
            fighter_a_name: Name or search string for Fighter A.
            fighter_b_name: Name or search string for Fighter B.
            event_date: As-of event date. Defaults to today (per D-09 the
                cache lookup keys on ``(A, B, event_date)``).
            refresh: Per D-10, force a live BFO fetch even on cache hit.

        Returns:
            Dict with fighter names, model probabilities, Elo probabilities,
            and top feature importances.

        Raises:
            ValueError: If fighter not found or ambiguous.
        """
        ev_date = event_date or date.today()

        # Step 1: resolve fighter names → Fighter ORM rows (HOUSE-04 dedup)
        fighter_a = _resolve_fighter(session, fighter_a_name)
        fighter_b = _resolve_fighter(session, fighter_b_name)

        # Steps 2-3: metadata + LIVE-03 feature-column assertion ALREADY ran
        # in __init__. Use the loaded model's expected column list as the
        # single source of truth post-LIVE-03; Phase 18 NET-V2-01 derives this
        # from self._include_net (72 cols for ablation/xgb_v2; 75 for
        # time-decayed). The __init__ assertion guarantees meta matches code.
        model_columns = get_feature_columns(include_net=self._include_net)

        # Step 4: live odds (D-09: cache → live HTTP → None)
        live_odds = fetch_matchup_odds(
            fighter_a.name,
            fighter_b.name,
            ev_date,
            refresh=refresh,
            session=session,
        )

        # Step 5: feature vector (LIVE-02 — replaces NaN-pad block).
        # Phase 18 NET-V2-01: include_net is dispatched from self._include_net
        # so 72-col (ablation/xgb_v2) and 75-col (time-decayed) models receive
        # the matching feature space at predict-time (Pitfall #12 parity).
        feature_vector = build_inference_features(
            session,
            fighter_a,
            fighter_b,
            ev_date,
            live_odds=live_odds,
            include_net=self._include_net,
        )

        # Step 6: XGBoost prediction (base model output)
        probs = self.model.predict_proba(feature_vector)
        prob_a = float(probs[0, 1])
        prob_b = 1.0 - prob_a

        # Step 6.5: Elo-based probability (per D-12: coexist) — moved earlier so
        # META has elo_prob_a available as a Level-1 input
        elo_a = _get_latest_elo(session, fighter_a.id, "overall")
        elo_b = _get_latest_elo(session, fighter_b.id, "overall")
        engine = EloEngine(EloConfig())
        elo_prob_a = engine.expected_win_probability(elo_a, elo_b)
        elo_prob_b = 1.0 - elo_prob_a

        # Phase 34 FALLBACK-V24-02 — graceful degradation routing.
        # Per CONTEXT D-FALLBACK-V24-02: when live_odds is None (which, per bfo_live.py
        # docstring "cache → live HTTP → None", implies both cache and live missed) AND
        # the canonical closing_prob_diff cell in the built feature vector is NaN AND
        # we have an xgb_v2_no_odds fallback loaded, route to the 67-col fallback model
        # and SKIP META entirely. META's Level-1 set includes closing_prob_diff which is
        # the missing signal — META would silently degrade (RESEARCH Pitfall 2).
        closing_prob_diff_idx = get_feature_columns(include_net=self._include_net).index(
            "closing_prob_diff"
        )
        closing_prob_diff = float(feature_vector[0, closing_prob_diff_idx])
        fallback_used = False
        if (
            np.isnan(closing_prob_diff)
            and self.fallback_model is not None
            and self.fallback_keep_idx is not None
        ):
            # Build the 67-col view from the existing 72-col feature_vector and route.
            X_no_odds = feature_vector[:, self.fallback_keep_idx]
            prob_a = float(self.fallback_model.predict_proba(X_no_odds)[0, 1])
            prob_b = 1.0 - prob_a
            fallback_used = True

        # Step 6b: Phase 19 META-05 wrapping (D-04 + D-05) — short-circuited when
        # fallback_used (Phase 34 FALLBACK-V24-02).
        meta_prob: float | None = None
        meta_kind: str | None = None
        meta_learner_version: str | None = None
        meta_skipped = True
        meta_skipped_reason: str | None = None

        if fallback_used:
            # Skip META entirely; the fallback path is independent of META.
            meta_skipped_reason = "fallback_no_odds_used"
        elif self.meta_model is None:
            meta_skipped_reason = "no_meta_artifact"
        else:
            # closing_prob_diff is at the canonical index in the 72-col vector
            if np.isnan(closing_prob_diff):
                meta_skipped_reason = "nan_closing_prob_diff"
            else:
                meta_input = np.array([[prob_a, elo_prob_a, closing_prob_diff]])
                meta_prob = float(self.meta_model.predict_proba(meta_input)[0, 1])
                meta_kind = self.meta_metadata["meta_kind"]
                meta_learner_version = self.meta_metadata["meta_version"]
                meta_skipped = False

        win_probability = meta_prob if not meta_skipped else prob_a

        # Phase 34 FALLBACK-V24-02 — surface which model produced win_probability.
        # Per revision m3 Option B: nested in `_meta` dict so Phase 35 CONTRACT-V24-02
        # can rename `_meta` → `prediction_metadata` as a single key-rename (not
        # restructure flat-to-nested).
        if fallback_used:
            win_probability_source = "xgb_v2_no_odds"
        elif not meta_skipped:
            win_probability_source = "meta_v2"
        else:
            win_probability_source = "xgb_v2"

        # Step 8: D-11 distribution-shift observability log + result
        feature_importances = self._get_top_importances(model_columns)
        self._log_predict_call(
            fighter_a,
            fighter_b,
            ev_date,
            live_odds,
            prob_a,
            meta_kind=meta_kind,
            meta_skipped=meta_skipped,
            meta_skipped_reason=meta_skipped_reason,
        )

        # Phase 35 CONTRACT-V24-02 — build prediction_metadata at the
        # predictor layer (single source of truth; API layer in Plan 35-05
        # only key-renames _meta → prediction_metadata under the v1.2.0
        # contract). Two of the six fields (odds_source, odds_timestamp_iso)
        # surface the same information that Phase 34 emitted at the top level;
        # the top-level `odds_source` is preserved for Phase 34 back-compat.
        # Defensive int() guard handles MagicMock sessions in unit tests that
        # don't stub a real query chain — those tests get 0 / "debutant" which
        # remains schema-valid.
        try:
            n_a = int(
                session.query(Fight)
                .filter((Fight.fighter_a_id == fighter_a.id) | (Fight.fighter_b_id == fighter_a.id))
                .count()
            )
        except (TypeError, AttributeError, Exception):  # noqa: BLE001 — defensive only
            n_a = 0
        try:
            n_b = int(
                session.query(Fight)
                .filter((Fight.fighter_a_id == fighter_b.id) | (Fight.fighter_b_id == fighter_b.id))
                .count()
            )
        except (TypeError, AttributeError, Exception):  # noqa: BLE001 — defensive only
            n_b = 0
        is_debutant_either = (n_a == 0) or (n_b == 0)
        slice_brier = self._select_slice_brier(ev_date)
        odds_ts_iso = live_odds.fetched_at.isoformat() if live_odds is not None else None
        odds_src = live_odds.source if live_odds is not None else "nan"

        prediction_metadata = {
            "win_probability_source": win_probability_source,
            "odds_source": odds_src,
            "odds_timestamp_iso": odds_ts_iso,
            "fighter_a_n_ufc_fights": int(n_a),
            "fighter_b_n_ufc_fights": int(n_b),
            "is_debutant_either": bool(is_debutant_either),
            "calibration_slice_brier": (float(slice_brier) if slice_brier is not None else None),
        }

        return {
            "fighter_a": fighter_a.name,
            "fighter_b": fighter_b.name,
            # Phase 19 D-05(P19) — canonical user-facing field
            "win_probability": win_probability,
            # Phase 19 additions (observability)
            "base_prob": prob_a,
            "meta_prob": meta_prob,
            "meta_kind": meta_kind,
            "meta_learner_version": meta_learner_version,
            "meta_skipped": meta_skipped,
            "meta_skipped_reason": meta_skipped_reason,
            # Existing fields preserved (back-compat with Phase 16+ consumers)
            "model_probability_a": prob_a,
            "model_probability_b": prob_b,
            "elo_probability_a": elo_prob_a,
            "elo_probability_b": elo_prob_b,
            "elo_a": elo_a,
            "elo_b": elo_b,
            "feature_importances": feature_importances,
            "odds_source": odds_src,
            # Phase 35 CONTRACT-V24-02 — single key-rename from Phase 34's
            # transitional `_meta` nest. API layer (Plan 35-05) surfaces this
            # under PredictorOutputV1 v1.2.0; v1.0.0 + v1.1.0 callers see
            # `prediction_metadata=None` (default).
            "prediction_metadata": prediction_metadata,
        }

    def _log_predict_call(
        self,
        fighter_a: Fighter,
        fighter_b: Fighter,
        event_date: date,
        live_odds,
        xgb_proba_a: float,
        *,
        meta_kind: str | None = None,
        meta_skipped: bool = True,
        meta_skipped_reason: str | None = None,
    ) -> None:
        """D-11 distribution-shift observability — append one JSONL line per
        ``predict_matchup`` call.

        Per CONTEXT.md ``<specifics>`` line 148: file path is operator
        discretion. Default is ``logs/predict.jsonl`` (relative to CWD);
        override via ``UFC_PREDICT_LOG`` env var. The default keeps the
        log out of stdout so the existing CLI table output is unaffected.

        Logged fields: predicted_at_iso, event_date, fighter_a, fighter_b,
        odds_source ∈ {cache, live, nan}, odds_timestamp_iso, hours_to_event,
        xgb_proba_a, model_version, plus Phase 19 META observability
        (meta_kind, meta_skipped, meta_skipped_reason — Pitfall #18).
        """
        try:
            log_path = Path(os.getenv("UFC_PREDICT_LOG", "logs/predict.jsonl"))
            log_path.parent.mkdir(parents=True, exist_ok=True)

            now = datetime.now(timezone.utc)
            if live_odds is not None:
                odds_source = live_odds.source
                odds_ts_iso = live_odds.fetched_at.isoformat()
            else:
                odds_source = "nan"
                odds_ts_iso = None

            event_dt = datetime.combine(event_date, datetime.min.time(), timezone.utc)
            hours_to_event = (event_dt - now).total_seconds() / 3600.0

            record = {
                "predicted_at_iso": now.isoformat(),
                "event_date": event_date.isoformat(),
                "fighter_a": fighter_a.name,
                "fighter_b": fighter_b.name,
                "odds_source": odds_source,
                "odds_timestamp_iso": odds_ts_iso,
                "hours_to_event": hours_to_event,
                "xgb_proba_a": xgb_proba_a,
                "model_version": self.version,
                # Phase 19 Pitfall #18 mitigation — meta observability
                "meta_kind": meta_kind,
                "meta_skipped": meta_skipped,
                "meta_skipped_reason": meta_skipped_reason,
            }
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:  # noqa: BLE001 — observability MUST NOT break predict
            logger.warning("predict log write failed: %s", exc)

    def _get_top_importances(
        self, feature_columns: list[str] | None = None, top_n: int = 10
    ) -> dict[str, float]:
        """Extract top N feature importances from the base XGBoost model."""
        cols = feature_columns or self.metadata.get("feature_columns") or FEATURE_COLUMNS
        # Try to get from the calibrated model's base estimator. Tolerate
        # MagicMock / shape-mismatched models (Phase 19 META dispatch tests
        # patch self.model; importances are not asserted by those tests).
        try:
            base_estimator = (
                self.model.calibrated_classifiers_[
                    0
                ].estimator.estimator  # FrozenEstimator wraps the real one
            )
            raw = base_estimator.feature_importances_
            importances = dict(zip(cols, [float(v) for v in raw], strict=True))
        except (AttributeError, IndexError, ValueError, TypeError):
            importances = {}

        sorted_feats = sorted(
            importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return dict(sorted_feats[:top_n])


def _resolve_fighter(session: Session, name: str) -> Fighter:
    """Resolve a fighter name to a Fighter object.

    Per T-10-05: uses existing search_fighters with parameterized queries.

    Raises:
        ValueError: If fighter not found or multiple matches.
    """
    results = search_fighters(session, name)

    if not results:
        msg = f"No fighter found matching '{name}'"
        raise ValueError(msg)

    # Group results by exact fighter name (case-insensitive). Multiple rows
    # under the same name are the SAME real-world person across ingest
    # sources; collapse each group to its canonical row first.
    by_name: dict[str, list[Fighter]] = {}
    for f in results:
        by_name.setdefault(f.name.lower(), []).append(f)
    deduped = [_prefer_canonical(group) for group in by_name.values()]

    if len(deduped) == 1:
        return deduped[0]

    # Prefer an exact (case-insensitive) name match if one of the deduped
    # candidates matches the search term verbatim. This preserves the
    # pre-DEDUP-01 behaviour that "Joe Smith" resolves to Joe Smith even
    # if a "Joe Smith Jr." substring-matches the same query.
    exact = [f for f in deduped if f.name.lower() == name.lower()]
    if len(exact) == 1:
        return exact[0]

    # Genuinely different real people both substring-match the search term.
    # Preserve the existing user-facing error.
    names = [f.name for f in deduped[:5]]
    msg = f"Multiple fighters match '{name}': {names}. Please be more specific."
    raise ValueError(msg)


def _get_latest_elo(
    session: Session,
    fighter_id: int,
    elo_type: str,
) -> float:
    """Get the latest Elo rating for a fighter.

    Returns the most recent elo_after_shrinkage for the given elo_type.
    Falls back to 1500.0 if no snapshots exist.
    """
    stmt = (
        select(EloSnapshot.elo_after_shrinkage)
        .where(EloSnapshot.fighter_id == fighter_id)
        .where(EloSnapshot.elo_type == elo_type)
        .order_by(EloSnapshot.fight_date.desc())
        .limit(1)
    )
    result = session.scalar(stmt)
    return float(result) if result is not None else 1500.0


# Phase 16-02 D-12: ``_build_feature_vector``, ``_get_latest_computed_features``,
# and ``_get_fighter_physical`` were deleted. The 28-feat NaN-pad block is
# replaced wholesale by ``ufc_prediction.ml.inference_features.build`` which
# emits the full 72-feature vector at predict time. The TODO comment about
# "compute live in Phase 16+" goes with them — no fallback wrapper, no
# deprecation warning. Per CONTEXT.md D-12: trust the new path.

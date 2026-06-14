"""FALLBACK-V24-02 integration test: BFO unreachable → xgb_v2_no_odds route.

Phase 34 Plan 06 Task 3. End-to-end exercise of the predictor's no-odds
fallback path: both BFO cache miss AND live HTTP failure → ``live_odds=None``
flows to ``inference_features.build`` → ``closing_prob_diff`` is NaN →
predictor routes to ``models/xgb_v2_no_odds.joblib`` (67-col ablation) and
bypasses META-V22 entirely.

Acceptance per 34-06-PLAN.md must_haves
(Phase 35 CONTRACT-V24-02 renamed ``_meta`` → ``prediction_metadata``;
DEBT-V25-01 closure Phase 46, 2026-06-02):
1. ``response["prediction_metadata"]["win_probability_source"] == "xgb_v2_no_odds"``
2. ``response["odds_source"] == "nan"``
3. ``response["meta_skipped_reason"] == "fallback_no_odds_used"``
4. ``win_probability`` numerically matches a direct
   ``xgb_v2_no_odds.predict_proba(X_no_odds)`` call against the same 67-col
   feature vector view.
5. canonical ``xgb_v2.joblib`` + ``meta_v2.joblib`` SHA-256 byte-identical
   pre/post the predict call (no inference-time mutation of artifacts).

The test mocks the heavy DB and BFO surfaces (mirrors
``test_live_odds_acceptance.py`` pattern) so it runs without a Docker daemon
or live BFO connectivity. The model artifacts themselves are loaded from
the worktree's ``models/`` directory.
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest

from ufc_prediction.ml.config import FEATURE_COLUMNS

pytestmark = [pytest.mark.integration]


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def upcoming_event_date() -> date:
    return date.today() + timedelta(days=7)


@pytest.fixture
def stub_fighters():
    """Two duck-typed Fighter rows the inference pipeline can read."""
    fa = MagicMock(
        id=1, name="Khabib Nurmagomedov", source="ufcstats",
        height_inches=70.0, reach_inches=70.0, leg_reach_inches=39.0,
        stance="Orthodox", date_of_birth=date(1988, 9, 20),
    )
    fa.name = "Khabib Nurmagomedov"
    fb = MagicMock(
        id=2, name="Conor McGregor", source="ufcstats",
        height_inches=69.0, reach_inches=74.0, leg_reach_inches=40.0,
        stance="Southpaw", date_of_birth=date(1988, 7, 14),
    )
    fb.name = "Conor McGregor"
    return fa, fb


@pytest.fixture
def predict_log_path(tmp_path, monkeypatch) -> Path:
    """Isolate the D-11 predict-trace JSONL so the test can introspect it."""
    log_path = tmp_path / "predict-trace.jsonl"
    monkeypatch.setenv("UFC_PREDICT_LOG", str(log_path))
    return log_path


@pytest.fixture
def patched_predictor(stub_fighters):
    """Build a real ModelPredictor against the worktree's ``models/``
    artifacts, but bypass the DB-backed helpers (Elo + computed-feature
    snapshot + fighter resolver) so the test does not need Postgres.

    Returns ``(predictor, fa, fb, mock_session)``.
    """
    from ufc_prediction.ml import inference_features as inf
    from ufc_prediction.ml import predictor as pred_mod

    if not Path("models/xgb_v2.joblib").exists():
        pytest.skip("models/xgb_v2.joblib not present in this worktree")
    if not Path("models/xgb_v2_no_odds.joblib").exists():
        pytest.skip(
            "models/xgb_v2_no_odds.joblib not present — Phase 34 Plan 02 "
            "must land before this integration test can run."
        )

    fa, fb = stub_fighters

    def fake_get_elo(session, fighter_id, elo_type):
        is_a = (fighter_id == fa.id)
        if elo_type == "overall":
            return 1620.0 if is_a else 1480.0
        if elo_type == "striking":
            return 1600.0 if is_a else 1520.0
        if elo_type == "grappling":
            return 1700.0 if is_a else 1450.0
        return 1500.0

    def fake_get_perf(session, fighter_id):
        return {
            "sig_str_per_minute": 4.5 if fighter_id == fa.id else 5.5,
            "td_rate": 3.0 if fighter_id == fa.id else 0.5,
            "td_accuracy": 0.50 if fighter_id == fa.id else 0.30,
            "ctrl_time_per_fight": 200.0 if fighter_id == fa.id else 30.0,
        }

    def fake_resolve_fighter(session, name):
        return fa if name == fa.name else fb

    predictor_instance = pred_mod.ModelPredictor(
        model_dir="models", version="v2",
    )

    with patch.object(inf, "_get_latest_elo", fake_get_elo), \
         patch.object(inf, "_get_latest_computed_features", fake_get_perf), \
         patch.object(pred_mod, "_resolve_fighter", fake_resolve_fighter), \
         patch.object(pred_mod, "_get_latest_elo", fake_get_elo):
        yield predictor_instance, fa, fb, MagicMock()


# ── Helpers ────────────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capture_inference_vec(monkeypatch):
    """Wrap ``inference_features.build`` to expose the 72-col feature vector."""
    from ufc_prediction.ml import inference_features as inf
    from ufc_prediction.ml import predictor as pred_mod

    captured = {}
    real_build = inf.build

    def wrapper(*args, **kwargs):
        vec = real_build(*args, **kwargs)
        captured["vec"] = vec
        return vec

    monkeypatch.setattr(inf, "build", wrapper)
    monkeypatch.setattr(pred_mod, "build_inference_features", wrapper)
    return captured


# ── Test 1: BFO unreachable → xgb_v2_no_odds route ─────────────────────────


def test_bfo_unreachable_routes_to_no_odds_fallback(
    patched_predictor, predict_log_path, upcoming_event_date, monkeypatch,
):
    """End-to-end: cache miss + live failure → predictor returns a
    win_probability sourced from xgb_v2_no_odds, response[prediction_metadata]
    correctly reports the source, META is skipped.
    """
    from ufc_prediction.scraper import bfo_live

    pred, fa, fb, session = patched_predictor

    # Force BOTH cache miss AND live HTTP failure → fetch_matchup_odds
    # falls through to return None (per bfo_live.py docstring contract).
    monkeypatch.setattr(bfo_live, "_try_cache", lambda *a, **kw: None)
    monkeypatch.setattr(bfo_live, "_try_live", lambda *a, **kw: None)

    captured = _capture_inference_vec(monkeypatch)

    result = pred.predict(
        session, fa.name, fb.name,
        event_date=upcoming_event_date,
    )

    # ── Assertion 1: closing_prob_diff IS NaN (no live, no cache) ──────────
    vec = captured["vec"][0]
    cl_idx = FEATURE_COLUMNS.index("closing_prob_diff")
    assert np.isnan(vec[cl_idx]), (
        f"closing_prob_diff should be NaN when both BFO paths fail. "
        f"Got {vec[cl_idx]!r}."
    )

    # ── Assertion 2: response[prediction_metadata].win_probability_source ──
    # Phase 35 CONTRACT-V24-02 renamed `_meta` → `prediction_metadata`
    # (Plan 35-04, 2026-05-26). DEBT-V25-01 closure (Phase 46, 2026-06-02).
    assert "prediction_metadata" in result, (
        "Phase 35 CONTRACT-V24-02 response must include `prediction_metadata` "
        "block (renamed from `_meta` in Plan 35-04)."
    )
    assert result["prediction_metadata"]["win_probability_source"] == "xgb_v2_no_odds", (
        f"Fallback path must mark source as 'xgb_v2_no_odds'. "
        f"Got {result['prediction_metadata'].get('win_probability_source')!r}."
    )

    # ── Assertion 3: odds_source reports 'nan' ─────────────────────────────
    assert result["odds_source"] == "nan", (
        f"BFO-unreachable predict must report odds_source='nan'. "
        f"Got {result['odds_source']!r}."
    )

    # ── Assertion 4: META skipped with fallback reason ────────────────────
    assert result.get("meta_skipped") is True, (
        "META must be skipped when fallback route is taken (Pitfall 2 — "
        "META-V22 cannot handle NaN closing_prob_diff)."
    )
    assert result.get("meta_skipped_reason") == "fallback_no_odds_used", (
        f"meta_skipped_reason must distinguish the fallback path from the "
        f"plain `nan_closing_prob_diff` path. Got "
        f"{result.get('meta_skipped_reason')!r}."
    )

    # ── Assertion 5: win_probability matches direct xgb_v2_no_odds call ───
    # Reconstruct the 67-col view from the captured 72-col vector and call
    # the fallback model directly. The predictor's choice must equal this.
    fallback_model = joblib.load("models/xgb_v2_no_odds.joblib")
    assert pred.fallback_keep_idx is not None, (
        "predictor.fallback_keep_idx must be set when the fallback artifact "
        "is present in models/."
    )
    X_no_odds = captured["vec"][:, pred.fallback_keep_idx]
    expected_proba_a = float(fallback_model.predict_proba(X_no_odds)[0, 1])
    actual_proba_a = float(result["model_probability_a"])

    assert actual_proba_a == pytest.approx(expected_proba_a, abs=1e-9), (
        f"win_probability_a from fallback path ({actual_proba_a}) must "
        f"byte-identically match a direct xgb_v2_no_odds.predict_proba call "
        f"({expected_proba_a})."
    )

    # The top-level win_probability field mirrors fighter A's proba.
    assert float(result["win_probability"]) == pytest.approx(
        expected_proba_a, abs=1e-9,
    )


# ── Test 2: xgb_v2 + meta_v2 SHA byte-identity preserved across the call ───


def test_predict_call_does_not_mutate_protected_artifacts(
    patched_predictor, upcoming_event_date, monkeypatch,
):
    """AUDIT-01 chain integrity: a predict call must not mutate the
    on-disk SHA of xgb_v2.joblib or meta_v2.joblib. The fallback path
    is no exception — it loads a sibling model (xgb_v2_no_odds.joblib),
    not the canonicals.
    """
    from ufc_prediction.scraper import bfo_live

    xgb_path = Path("models/xgb_v2.joblib")
    meta_path = Path("models/meta/meta_v2.joblib")

    if not xgb_path.exists():
        pytest.skip("xgb_v2.joblib not present")

    sha_xgb_before = _sha256(xgb_path)
    sha_meta_before = _sha256(meta_path) if meta_path.exists() else None

    pred, fa, fb, session = patched_predictor

    monkeypatch.setattr(bfo_live, "_try_cache", lambda *a, **kw: None)
    monkeypatch.setattr(bfo_live, "_try_live", lambda *a, **kw: None)

    pred.predict(
        session, fa.name, fb.name,
        event_date=upcoming_event_date,
    )

    sha_xgb_after = _sha256(xgb_path)
    assert sha_xgb_after == sha_xgb_before, (
        f"xgb_v2.joblib SHA changed across predict call. "
        f"Before: {sha_xgb_before}; After: {sha_xgb_after}. "
        f"AUDIT-01 chain broken."
    )
    if sha_meta_before is not None:
        sha_meta_after = _sha256(meta_path)
        assert sha_meta_after == sha_meta_before, (
            f"meta_v2.joblib SHA changed across predict call. "
            f"AUDIT-01 chain broken."
        )

    # Sanity: the canonical xgb_v2.joblib SHA should still match the
    # AUDIT-01 baseline at the project root.
    canonical_sha = (
        "6e7641109524177c2f4efe556f6e29c38baa1ea996d68fac59879f4d6a1ba099"
    )
    assert sha_xgb_after == canonical_sha, (
        f"xgb_v2.joblib drifted from canonical AUDIT-01 baseline. "
        f"Expected {canonical_sha}, got {sha_xgb_after}."
    )

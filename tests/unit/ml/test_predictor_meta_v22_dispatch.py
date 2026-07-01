"""predict()-level guard for review CRITICAL #1 (and review finding tests-0).

Unlike test_meta_v22_inference_contract.py (which pins the artifact/assembler
shape), THIS test drives the real ``ModelPredictor.predict()`` width dispatch —
the exact code path where the 3-vs-13 bug lived. Reverting predictor.py's v2
branch to the old 3-wide input makes ``test_predict_engages_13col_meta`` raise
(3-wide into the 13-col meta), so this test fails on regression.

DB-free + non-slow: the DB-touching helpers and the inference feature builder
are monkeypatched; the real xgb_v2 + meta_v2 artifacts are used unchanged.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ufc_prediction.ml import predictor as pmod
from ufc_prediction.ml.config import FEATURE_COLUMNS_V22, get_feature_columns

pytestmark = pytest.mark.skipif(
    not Path("models/meta/meta_v2.joblib").exists(),
    reason="promoted meta_v2.joblib not present in this checkout",
)


def _make_predictor():
    return pmod.ModelPredictor(model_dir="models", version="v2", meta_dir="models/meta")


def _patch_common(monkeypatch, p):
    fa = SimpleNamespace(id=1, name="A")
    fb = SimpleNamespace(id=2, name="B")
    monkeypatch.setattr(pmod, "_resolve_fighter", lambda _s, n: fa if n == "A" else fb)
    monkeypatch.setattr(pmod, "fetch_matchup_odds", lambda *a, **k: None)
    monkeypatch.setattr(pmod, "_get_latest_elo", lambda *a, **k: 1500.0)
    monkeypatch.setattr(p, "_log_predict_call", lambda *a, **k: None)


def test_predict_engages_13col_meta(monkeypatch):
    """A finite 13-col v2.2 vector → META-V22 engages (no ValueError, no skip)."""
    p = _make_predictor()
    assert len(p.meta_metadata["meta_feature_columns"]) == 13, "expected the 13-col meta_v2"
    _patch_common(monkeypatch, p)

    def fake_build(_s, _a, _b, _ev, *, live_odds=None, include_net=None, feature_set="v1.0", **kw):
        n = (
            len(FEATURE_COLUMNS_V22)
            if feature_set == "v2.2"
            else len(get_feature_columns(include_net=include_net))
        )
        return np.full((1, n), 0.1)  # all finite → meta must engage

    monkeypatch.setattr(pmod, "build_inference_features", fake_build)

    res = p.predict(MagicMock(), "A", "B")
    assert res["meta_skipped"] is False, res.get("meta_skipped_reason")
    assert res["meta_prob"] is not None
    assert 0.0 <= res["meta_prob"] <= 1.0
    assert res["win_probability"] == res["meta_prob"]


def test_predict_skips_meta_on_nan_input(monkeypatch):
    """A NaN in any of the 13 Level-1 inputs → skip to base prob, never crash."""
    p = _make_predictor()
    _patch_common(monkeypatch, p)

    nan_idx = FEATURE_COLUMNS_V22.index("days_since_last_fight_diff")

    def fake_build(_s, _a, _b, _ev, *, live_odds=None, include_net=None, feature_set="v1.0", **kw):
        if feature_set == "v2.2":
            arr = np.full((1, len(FEATURE_COLUMNS_V22)), 0.1)
            arr[0, nan_idx] = np.nan
            return arr
        return np.full((1, len(get_feature_columns(include_net=include_net))), 0.1)

    monkeypatch.setattr(pmod, "build_inference_features", fake_build)

    res = p.predict(MagicMock(), "A", "B")
    assert res["meta_skipped"] is True
    assert res["meta_skipped_reason"] == "nan_meta_input"
    assert res["win_probability"] == res["base_prob"]

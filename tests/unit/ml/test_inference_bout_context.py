"""Serve-time bout context: weight class, rounds, title flag and physicals.

Covers four train/serve skews in ``inference_features.build``:

1. Bout context (``weight_class``, ``num_rounds``, ``is_title_fight``) was
   never supplied by ``ModelPredictor.predict``, so every served fight was a
   3-round non-title bout. ``build`` now reads the scheduled ``Fight`` row for
   ``(A, B, event_date)`` when the caller omits them (explicit args win).
2. With no weight class, the division came from fighter A's last fight even
   when that was 'Catch Weight' / 'Open Weight', which zeroes B's Elo to the
   seed/1500 and falls ``weight_class_ordinal`` back to its default.
3. Missing height/reach/leg-reach stayed NaN at serve while training imputes
   the division median first (D-01).
"""

from __future__ import annotations

import inspect
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest

from ufc_prediction.elo.asof import RatedFight, pre_fight_rating
from ufc_prediction.elo.config import EloConfig
from ufc_prediction.ml import inference_features
from ufc_prediction.ml.config import MLConfig, get_feature_columns

_COLS = get_feature_columns(feature_set="v2.1-no-net")
_REAL_GET_LATEST_ELO = inference_features._get_latest_elo


def _col(vec: np.ndarray, name: str) -> float:
    return float(vec[0, _COLS.index(name)])


def _fighter(fid: int, **kw) -> MagicMock:
    attrs = {
        "id": fid,
        "name": f"Fighter-{fid}",
        "height_inches": 70.0,
        "reach_inches": 72.0,
        "leg_reach_inches": 40.0,
        "stance": "Orthodox",
        "date_of_birth": date(1990, 1, 1),
    }
    attrs.update(kw)
    return MagicMock(**attrs)


def _record(fight_id, day, a, b, wc, winner=None):
    return {
        "fight_id": fight_id,
        "event_date": day,
        "fighter_a_id": a,
        "fighter_b_id": b,
        "winner_id": winner if winner is not None else a,
        "weight_class": wc,
        "method": "Decision",
        "is_title_fight": False,
        "num_rounds": 3,
    }


@pytest.fixture
def stub_db(monkeypatch):
    """Stub every DB reader ``build`` touches; tests override what they need."""
    monkeypatch.setattr(inference_features, "_get_latest_elo", lambda *a, **kw: 1500.0)
    monkeypatch.setattr(inference_features, "_get_pre_fight_performance", lambda *a: ({}, {}))
    monkeypatch.setattr(inference_features, "_get_cached_odds", lambda *a: (None, None))
    monkeypatch.setattr(
        inference_features, "_load_career_inputs", lambda *a: inference_features.CareerInputs()
    )
    monkeypatch.setattr(inference_features, "_query_scheduled_bout", lambda *a: None)
    monkeypatch.setattr(inference_features, "_query_division_physical_medians", lambda *a: {})


def _build(**kw) -> np.ndarray:
    fa = kw.pop("fa", _fighter(1))
    fb = kw.pop("fb", _fighter(2))
    return inference_features.build(
        MagicMock(), fa, fb, date(2026, 10, 3), feature_set="v2.1-no-net", **kw
    )


# ── Finding 1: bout context ──────────────────────────────────────────────────


class TestScheduledBoutContext:
    def test_scheduled_fight_row_supplies_context_when_omitted(self, stub_db, monkeypatch):
        seen = []

        def scheduled(session, fa_id, fb_id, event_date):
            seen.append((fa_id, fb_id, event_date))
            return ("Welterweight", 5, True)

        monkeypatch.setattr(inference_features, "_query_scheduled_bout", scheduled)
        vec = _build()
        assert seen == [(1, 2, date(2026, 10, 3))]
        assert _col(vec, "num_rounds") == 5.0
        assert _col(vec, "is_title_fight") == 1.0
        assert _col(vec, "weight_class_ordinal") == 6.0

    def test_explicit_args_override_scheduled_row(self, stub_db, monkeypatch):
        monkeypatch.setattr(
            inference_features, "_query_scheduled_bout", lambda *a: ("Welterweight", 5, True)
        )
        vec = _build(weight_class="Flyweight", num_rounds=3, is_title_fight=False)
        assert _col(vec, "num_rounds") == 3.0
        assert _col(vec, "is_title_fight") == 0.0
        assert _col(vec, "weight_class_ordinal") == 2.0

    def test_partial_explicit_args_fill_rest_from_scheduled_row(self, stub_db, monkeypatch):
        monkeypatch.setattr(
            inference_features, "_query_scheduled_bout", lambda *a: ("Welterweight", 5, True)
        )
        vec = _build(num_rounds=3)
        assert _col(vec, "num_rounds") == 3.0
        assert _col(vec, "is_title_fight") == 1.0
        assert _col(vec, "weight_class_ordinal") == 6.0

    def test_scheduled_lookup_skipped_when_all_context_given(self, stub_db, monkeypatch):
        def boom(*a):
            raise AssertionError("should not query the scheduled bout")

        monkeypatch.setattr(inference_features, "_query_scheduled_bout", boom)
        _build(weight_class="Lightweight", num_rounds=5, is_title_fight=True)

    def test_defaults_without_scheduled_row(self, stub_db):
        vec = _build()
        assert _col(vec, "num_rounds") == 3.0
        assert _col(vec, "is_title_fight") == 0.0

    def test_scheduled_division_drives_elo_replay(self, stub_db, monkeypatch):
        monkeypatch.setattr(
            inference_features, "_query_scheduled_bout", lambda *a: ("Middleweight", 5, False)
        )
        divisions = set()

        def spy(session, fid, et, as_of=None, division=None):
            divisions.add(division)
            return 1500.0

        monkeypatch.setattr(inference_features, "_get_latest_elo", spy)
        _build()
        assert divisions == {"Middleweight"}

    def test_query_scheduled_bout_degrades_on_unusable_session(self):
        assert (
            inference_features._query_scheduled_bout(MagicMock(), 1, 2, date(2026, 10, 3)) is None
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "NEEDS OPERATOR APPROVAL (AUDIT01_OVERRIDE): ModelPredictor.predict "
        "(AUDIT-01 protected predictor.py) does not yet accept/forward bout "
        "context. Remove this marker once the PR's proposed diff is applied."
    ),
)
def test_model_predictor_predict_accepts_bout_context():
    from ufc_prediction.ml.predictor import ModelPredictor

    params = inspect.signature(ModelPredictor.predict).parameters
    for name in ("weight_class", "num_rounds", "is_title_fight"):
        assert name in params, name


# ── Finding 2: catchweight / open-weight fallback ─────────────────────────────


class TestDivisionFallback:
    def test_skips_catch_weight_last_fight(self):
        records = [
            _record(1, date(2022, 1, 1), 1, 3, "Welterweight"),
            _record(2, date(2024, 1, 1), 1, 4, "Catch Weight"),
            _record(3, date(2023, 1, 1), 2, 5, "Welterweight"),
        ]
        assert inference_features._resolve_division(records, 1, 2) == "Welterweight"

    def test_skips_open_weight(self):
        records = [
            _record(1, date(2022, 1, 1), 1, 3, "Heavyweight"),
            _record(2, date(2024, 1, 1), 1, 4, "Open Weight"),
        ]
        assert inference_features._resolve_division(records, 1, 2) == "Heavyweight"

    def test_falls_back_to_b_when_a_only_fought_catch_weight(self):
        records = [
            _record(1, date(2024, 1, 1), 1, 4, "Catch Weight"),
            _record(2, date(2023, 1, 1), 2, 5, "Lightweight"),
        ]
        assert inference_features._resolve_division(records, 1, 2) == "Lightweight"

    def test_last_resort_is_any_division(self):
        records = [_record(1, date(2024, 1, 1), 1, 4, "Catch Weight")]
        assert inference_features._resolve_division(records, 1, 2) == "Catch Weight"

    def test_no_history_is_none(self):
        assert inference_features._resolve_division([], 1, 2) is None

    def test_prefers_a_division_both_fighters_share(self):
        # A moved up to Middleweight; B has only fought Welterweight, where A
        # also has history. Welterweight is the division both are rated in.
        records = [
            _record(1, date(2021, 1, 1), 1, 3, "Welterweight"),
            _record(2, date(2024, 1, 1), 1, 4, "Middleweight"),
            _record(3, date(2023, 1, 1), 2, 5, "Welterweight"),
        ]
        assert inference_features._resolve_division(records, 1, 2) == "Welterweight"
        assert inference_features._resolve_division(records, 2, 1) == "Welterweight"

    def test_unshared_divisions_resolve_symmetrically_to_most_recent(self):
        records = [
            _record(1, date(2022, 1, 1), 1, 3, "Lightweight"),
            _record(2, date(2024, 1, 1), 2, 4, "Welterweight"),
        ]
        assert inference_features._resolve_division(records, 1, 2) == "Welterweight"
        assert inference_features._resolve_division(records, 2, 1) == "Welterweight"

    def test_build_catchweight_last_fight_keeps_opponent_elo(self, stub_db, monkeypatch):
        """Weidman (last fight catchweight) vs Usman: Usman's Elo must come from
        his real division, not the 1500 default for an unrated catchweight key."""
        records = [
            _record(10, date(2021, 4, 1), 1, 3, "Middleweight"),
            _record(11, date(2024, 12, 7), 1, 4, "Catch Weight"),
            _record(12, date(2023, 3, 1), 2, 5, "Welterweight"),
            _record(13, date(2023, 10, 21), 2, 6, "Middleweight"),
        ]
        monkeypatch.setattr(
            inference_features,
            "_load_career_inputs",
            lambda *a: inference_features.CareerInputs(fight_records=records),
        )
        histories = {
            1: [
                RatedFight(date(2021, 4, 1), "Middleweight", 1560.0),
                RatedFight(date(2024, 12, 7), "Catch Weight", 1509.5),
            ],
            2: [
                RatedFight(date(2023, 3, 1), "Welterweight", 1580.0),
                RatedFight(date(2023, 10, 21), "Middleweight", 1542.7),
            ],
        }
        monkeypatch.setattr(
            inference_features,
            "_load_elo_history",
            lambda s, fid, et, before: histories[fid],
        )
        monkeypatch.setattr(inference_features, "_load_debutant_seeds", lambda: {})
        monkeypatch.setattr(inference_features, "_get_latest_elo", _REAL_GET_LATEST_ELO)
        ev = date(2026, 10, 3)

        def rating(fid: int, division: str) -> float:
            return pre_fight_rating(histories[fid], ev, division, config=EloConfig())

        # The pre-fix division: A's last fight. B has no Catch Weight key → 1500.
        assert rating(2, "Catch Weight") == 1500.0
        vec = _build()
        assert _col(vec, "weight_class_ordinal") == 7.0  # Middleweight, not default 5
        assert _col(vec, "elo_overall_diff") == pytest.approx(
            rating(1, "Middleweight") - rating(2, "Middleweight")
        )
        assert _col(vec, "elo_overall_diff") != pytest.approx(
            rating(1, "Catch Weight") - rating(2, "Catch Weight")
        )

    def test_query_fighter_division_skips_non_transfer(self):
        session = MagicMock()
        session.execute.return_value.scalars.return_value.all.return_value = [
            "Catch Weight",
            "Lightweight",
            "Featherweight",
        ]
        assert inference_features._query_fighter_division(session, 1) == "Lightweight"

    def test_query_fighter_division_last_resort_non_transfer(self):
        session = MagicMock()
        session.execute.return_value.scalars.return_value.all.return_value = ["Catch Weight"]
        assert inference_features._query_fighter_division(session, 1) == "Catch Weight"


# ── Finding 3: division-median imputation of physicals ───────────────────────


class TestPhysicalImputation:
    _MEDIANS = {"height_inches": 71.0, "reach_inches": 73.0, "leg_reach_inches": 41.0}

    def test_missing_reach_imputed_with_division_median(self, stub_db, monkeypatch):
        seen = []

        def medians(session, weight_class, cutoff):
            seen.append((weight_class, cutoff))
            return dict(self._MEDIANS)

        monkeypatch.setattr(inference_features, "_query_division_physical_medians", medians)
        vec = _build(
            fa=_fighter(1, reach_inches=76.0),
            fb=_fighter(2, reach_inches=None, height_inches=None),
            weight_class="Welterweight",
        )
        assert _col(vec, "reach_diff") == pytest.approx(76.0 - 73.0)
        assert _col(vec, "height_diff") == pytest.approx(70.0 - 71.0)
        assert _col(vec, "leg_reach_diff") == pytest.approx(0.0)
        assert seen == [("Welterweight", date.fromisoformat(MLConfig().cutoff_date))]

    def test_both_missing_imputes_both_sides(self, stub_db, monkeypatch):
        monkeypatch.setattr(
            inference_features, "_query_division_physical_medians", lambda *a: dict(self._MEDIANS)
        )
        vec = _build(
            fa=_fighter(1, reach_inches=None),
            fb=_fighter(2, reach_inches=None),
            weight_class="Welterweight",
        )
        assert _col(vec, "reach_diff") == 0.0

    def test_division_without_medians_stays_nan(self, stub_db):
        vec = _build(fb=_fighter(2, reach_inches=None), weight_class="Welterweight")
        assert np.isnan(_col(vec, "reach_diff"))

    def test_no_median_query_when_physicals_complete(self, stub_db, monkeypatch):
        def boom(*a):
            raise AssertionError("medians should only be read for a missing physical")

        monkeypatch.setattr(inference_features, "_query_division_physical_medians", boom)
        _build(weight_class="Welterweight")

    def test_median_query_degrades_on_unusable_session(self):
        assert (
            inference_features._query_division_physical_medians(
                MagicMock(), "Welterweight", date(2023, 1, 1)
            )
            == {}
        )

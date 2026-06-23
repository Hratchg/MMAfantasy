"""Regression: matchup prediction must be invariant to fighter argument order.

The raw ``ModelPredictor.predict`` is asymmetric (three style/stance features
are not anti-symmetric), so swapping fighter_a/fighter_b shifts the probability
by ~10 points. ``predict_order_invariant`` fixes this by averaging both
directions. These tests pin that contract with a fake predictor that encodes a
known asymmetry — no DB or model artifacts required.
"""

from __future__ import annotations

import pytest

from ufc_prediction.ml.order_invariant import predict_order_invariant


class _FakeAsymmetricPredictor:
    """Returns an order-dependent win prob: whoever is in slot B is favored.

    Mirrors the real bug numbers: A-vs-B -> P(a)=0.439, B-vs-A -> P(b)=0.453.
    """

    # P(fighter_a wins) keyed by (a, b) order.
    _WIN_A = {("Ades", "Strick"): 0.439, ("Strick", "Ades"): 0.453}

    def predict(self, session, fighter_a, fighter_b, **kwargs):
        wp = self._WIN_A[(fighter_a, fighter_b)]
        return {
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "win_probability": wp,
            "model_probability_a": wp,
            "model_probability_b": 1.0 - wp,
            "base_prob": wp,
            "meta_prob": None,
            "elo_probability_a": 0.469,
            "elo_probability_b": 0.531,
        }


def test_win_probability_is_order_invariant():
    p = _FakeAsymmetricPredictor()
    fwd = predict_order_invariant(p, None, "Ades", "Strick")
    rev = predict_order_invariant(p, None, "Strick", "Ades")

    # Adesanya's win prob is identical whether listed first or second.
    ades_first = fwd["win_probability"]
    ades_second = 1.0 - rev["win_probability"]
    assert ades_first == pytest.approx(ades_second, abs=1e-9)


def test_value_is_the_average_of_both_directions():
    p = _FakeAsymmetricPredictor()
    out = predict_order_invariant(p, None, "Ades", "Strick")
    # ½·(P(a|slotA) + P(a|slotB)) = ½·(0.439 + (1 - 0.453)) = 0.493
    assert out["win_probability"] == pytest.approx(0.493, abs=1e-9)
    assert out["model_probability_a"] == pytest.approx(0.493, abs=1e-9)
    # b-side mirror stays consistent with the symmetrized a-side
    assert out["model_probability_b"] == pytest.approx(0.507, abs=1e-9)
    assert out["order_invariant"] is True


def test_probabilities_still_sum_to_one():
    p = _FakeAsymmetricPredictor()
    out = predict_order_invariant(p, None, "Ades", "Strick")
    assert out["model_probability_a"] + out["model_probability_b"] == pytest.approx(1.0)


def test_none_meta_prob_is_left_untouched():
    p = _FakeAsymmetricPredictor()
    out = predict_order_invariant(p, None, "Ades", "Strick")
    assert out["meta_prob"] is None

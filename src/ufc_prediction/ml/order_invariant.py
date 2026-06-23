"""Order-invariant matchup prediction.

`ModelPredictor.predict()` is not symmetric under swapping `fighter_a` and
`fighter_b`. Most features are difference-based and flip sign cleanly on a swap,
but three style/stance features are not built anti-symmetrically:

  - ``stance_matchup``              (order-invariant constant, same value both ways)
  - ``a_striking_vs_b_grappling``   (swaps column with its mirror instead of negating)
  - ``a_grappling_vs_b_striking``   (likewise)

Because XGBoost learns independent weights per column, the prediction depends on
which fighter is passed first — a swing of ~10 points on close matchups, enough
to flip the pick. See the ``order_invariant`` regression test.

This wrapper restores order-invariance the cheap, no-retrain way: it runs the
predictor in BOTH directions and averages the ``fighter_a`` win-probability, so

    P(a beats b) = ½ · [ P(a wins | a in slot A) + P(a wins | a in slot B) ]

which is provably independent of argument order. It does NOT modify the
AUDIT-01 byte-locked ``predictor.py``. Meta/fallback routing depends only on the
(symmetric) matchup and its odds, so both directions route identically.
"""

from __future__ import annotations

from typing import Any

# Probability fields in the predict() result that are oriented to fighter_a and
# must be symmetrized. Each is "fighter_a's probability of winning" under some
# model stage; its reverse-run counterpart is the same quantity for fighter_b.
_FIGHTER_A_PROB_FIELDS = ("win_probability", "model_probability_a", "base_prob", "meta_prob")


def _sym(forward_a: float, reverse_a: float) -> float:
    """Average fighter_a's win-prob across both slots.

    ``forward_a`` is P(a wins) with a in slot A; ``reverse_a`` is P(b wins) with
    b in slot A, i.e. P(a loses) with a in slot B — so 1 - reverse_a is P(a wins
    | a in slot B). The mean of the two slots is order-invariant.
    """
    return 0.5 * (forward_a + (1.0 - reverse_a))


def predict_order_invariant(
    predictor: Any,
    session: Any,
    fighter_a_name: str,
    fighter_b_name: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Order-invariant drop-in for ``predictor.predict(session, a, b, **kwargs)``.

    Returns the same dict shape as ``ModelPredictor.predict`` (fighter_a/fighter_b
    orientation, Elo, feature importances, prediction_metadata all preserved from
    the forward run), with every fighter_a-oriented probability replaced by its
    order-invariant average and an ``order_invariant: True`` marker added.
    """
    forward = predictor.predict(session, fighter_a_name, fighter_b_name, **kwargs)
    reverse = predictor.predict(session, fighter_b_name, fighter_a_name, **kwargs)

    out = dict(forward)
    for field in _FIGHTER_A_PROB_FIELDS:
        fwd = forward.get(field)
        rev = reverse.get(field)
        if fwd is None or rev is None:
            continue  # meta_prob can be None when the meta-learner is skipped
        out[field] = _sym(fwd, rev)

    # Keep the b-side mirror consistent with the symmetrized a-side.
    if out.get("model_probability_a") is not None:
        out["model_probability_b"] = 1.0 - out["model_probability_a"]

    out["order_invariant"] = True
    return out

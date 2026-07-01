"""Unit tests for BFO vig-removal and implied-probability math (ODDS-02).

These tests are pure-logic — no network, no DB, no third-party scraping
dependencies. They assert the canonical examples from D-02 and the
worked example in .planning/phases/13-betting-odds-integration/13-RESEARCH.md
(Pattern 2: "Vig removal via proportional normalization").
"""

from __future__ import annotations

import pytest

from ufc_prediction.scraper.bfo_math import (
    american_ml_to_implied_prob,
    closing_prob_consensus,
    devig_closing_range,
    devig_proportional,
)


class TestAmericanMlToImpliedProb:
    """`american_ml_to_implied_prob` — raw implied probability incl. vig."""

    def test_ml_to_prob_favorite_minus_200(self) -> None:
        """Canonical favorite: -200 -> 2/3 (0.6667)."""
        assert american_ml_to_implied_prob(-200) == pytest.approx(0.6666667, abs=1e-6)

    def test_ml_to_prob_underdog_plus_170(self) -> None:
        """Canonical underdog: +170 -> 100/270 (0.3704)."""
        assert american_ml_to_implied_prob(170) == pytest.approx(0.3703704, abs=1e-6)

    def test_ml_to_prob_pickem_plus_100(self) -> None:
        """Exact pickem: +100 -> 0.5."""
        assert american_ml_to_implied_prob(100) == pytest.approx(0.5, abs=1e-9)

    def test_ml_to_prob_extreme_favorite_minus_1000(self) -> None:
        """Heavy favorite: -1000 -> 1000/1100 (0.9091)."""
        assert american_ml_to_implied_prob(-1000) == pytest.approx(0.9090909, abs=1e-6)


class TestDevigProportional:
    """`devig_proportional` — D-02 proportional vig removal to sum=1.0."""

    def test_devig_normalizes_to_one(self) -> None:
        """Output two probabilities always sum to 1.0 within 1e-9."""
        p_a, p_b = devig_proportional(-200, 170)
        assert abs((p_a + p_b) - 1.0) < 1e-9

    def test_devig_reclaims_vig_from_known_example(self) -> None:
        """D-02 worked example: (-200, +170) -> raw sum ~1.037; out (0.6429, 0.3571).

        Source: .planning/phases/13-betting-odds-integration/13-RESEARCH.md
        Pattern 2 and the D-02 docstring example.
        """
        p_a, p_b = devig_proportional(-200, 170)
        assert p_a == pytest.approx(0.6429, abs=1e-4)
        assert p_b == pytest.approx(0.3571, abs=1e-4)

    def test_devig_symmetric_pickem(self) -> None:
        """Symmetric vig (-110, -110) removes cleanly to (0.5, 0.5)."""
        p_a, p_b = devig_proportional(-110, -110)
        assert p_a == pytest.approx(0.5, abs=1e-9)
        assert p_b == pytest.approx(0.5, abs=1e-9)

    def test_devig_returns_tuple_of_two_floats(self) -> None:
        """Return type is a 2-tuple of floats."""
        result = devig_proportional(-200, 170)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


class TestAmericanMlGuard:
    """`american_ml_to_implied_prob` rejects invalid moneylines (review #3)."""

    @pytest.mark.parametrize("ml", [-99, -1, 0, 1, 50, 99])
    def test_rejects_values_inside_open_interval(self, ml: int) -> None:
        """Values in (-100, 100) are not valid American odds → ValueError.

        This is the guard that catches a midpoint that averaged two MLs across
        zero (e.g. -2) instead of silently returning a garbage probability.
        """
        with pytest.raises(ValueError, match="invalid American moneyline"):
            american_ml_to_implied_prob(ml)

    @pytest.mark.parametrize("ml", [-100, 100, -110, 250, -1000])
    def test_accepts_valid_moneylines(self, ml: int) -> None:
        """|ml| >= 100 (incl. even ±100) is accepted."""
        p = american_ml_to_implied_prob(ml)
        assert 0.0 < p < 1.0


class TestClosingProbConsensus:
    """`closing_prob_consensus` / `devig_closing_range` — probability-space
    closing consensus (review #3: supersedes the broken ML-midpoint method)."""

    def test_consensus_same_side_range(self) -> None:
        """(-200, -180) -> mean(0.6667, 0.6429) = 0.6548 (both favorite)."""
        assert closing_prob_consensus(-200, -180) == pytest.approx(0.6548, abs=1e-4)

    def test_straddle_zero_range_is_sane(self) -> None:
        """REGRESSION: a range straddling zero must NOT collapse to garbage.

        The old midpoint method turned (-105, 100) into midpoint -2 -> p=0.0196.
        Probability-space consensus gives mean(0.5122, 0.5000) = 0.5061.
        """
        assert closing_prob_consensus(-105, 100) == pytest.approx(0.5061, abs=1e-4)

    def test_devig_closing_range_near_even_fight(self) -> None:
        """The full near-even devig is ~50/50, not the old (0.0196, 0.9804)."""
        p_a, p_b = devig_closing_range(-105, 100, -100, 105)
        assert p_a == pytest.approx(0.5, abs=0.02)
        assert p_b == pytest.approx(0.5, abs=0.02)
        assert (p_a + p_b) == pytest.approx(1.0, abs=1e-9)

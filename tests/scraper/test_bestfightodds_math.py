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
    closing_ml_consensus,
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


class TestClosingMlConsensus:
    """`closing_ml_consensus` — D-01 midpoint of book-to-book closing range."""

    def test_closing_consensus_midpoint_even(self) -> None:
        """Even midpoint: (-200, -180) -> -190."""
        assert closing_ml_consensus(-200, -180) == -190

    def test_closing_consensus_midpoint_odd_rounds(self) -> None:
        """Odd midpoint uses Python 3.12 banker's rounding via int(round(...)).

        (-195, -180) -> raw midpoint -187.5 -> int(round(-187.5)) == -188
        (Python 3 rounds half to even: -187.5 -> -188 since -188 is even).
        """
        assert closing_ml_consensus(-195, -180) == -188

    def test_closing_consensus_mixed_signs(self) -> None:
        """Mixed-sign range: (-110, 120) -> midpoint 5."""
        assert closing_ml_consensus(-110, 120) == 5

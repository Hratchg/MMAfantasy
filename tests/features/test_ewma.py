"""Unit tests for EWMA computation."""

from __future__ import annotations

from ufc_prediction.features.ewma import compute_ewma


class TestComputeEwma:
    """Tests for the compute_ewma function."""

    def test_empty_list_returns_zero(self) -> None:
        assert compute_ewma([]) == 0.0

    def test_single_value_returns_that_value(self) -> None:
        assert compute_ewma([5.0]) == 5.0

    def test_three_values_with_default_alpha(self) -> None:
        """Manually computed EWMA with alpha=0.206:
        ewma1 = 2.0
        ewma2 = 0.206 * 4.0 + (1 - 0.206) * 2.0 = 0.824 + 1.588 = 2.412
        ewma3 = 0.206 * 6.0 + (1 - 0.206) * 2.412 = 1.236 + 1.915128 = 3.151128
        """
        result = compute_ewma([2.0, 4.0, 6.0], alpha=0.206)
        assert abs(result - 3.151128) < 0.001

    def test_default_alpha_is_0_206(self) -> None:
        """Default alpha should be 0.206 (3-fight half-life per D-03)."""
        # With a single value, alpha doesn't matter, so test with two values
        result_default = compute_ewma([1.0, 10.0])
        result_explicit = compute_ewma([1.0, 10.0], alpha=0.206)
        assert result_default == result_explicit

    def test_alpha_one_returns_last_value(self) -> None:
        """With alpha=1.0, EWMA always equals the latest value."""
        result = compute_ewma([1.0, 2.0, 3.0], alpha=1.0)
        assert result == 3.0

    def test_alpha_near_zero_returns_near_first_value(self) -> None:
        """With alpha near 0, EWMA stays close to the first value."""
        result = compute_ewma([10.0, 1.0, 1.0], alpha=0.01)
        assert result > 9.0  # should still be close to 10

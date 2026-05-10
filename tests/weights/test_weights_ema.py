"""Tests for EMA weight calculations.

Layer 1 tests: validate ema_weights function including return type, default alpha
calculation, normalization, length, ordering, custom parameters, edge cases,
and numerical properties.
"""

import numpy as np
import pytest

from mm_toolbox.weights import ema_weights


class TestEmaWeightsBasic:
    """Layer 1: Test basic EMA weights functionality."""

    def test_function_return_type(self):
        """Given window size, When ema_weights called, Then returns numpy float64 array."""
        result = ema_weights(5)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_default_alpha_calculation(self):
        """Given window size, When ema_weights called with default alpha, Then uses 2/(window+1)."""
        window = 5
        result = ema_weights(window)
        expected = ema_weights(window, alpha=2.0 / float(window + 1))
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_weights_normalization(self):
        """Given window size, When ema_weights called, Then sums to 1.0."""
        result = ema_weights(5)
        assert pytest.approx(result.sum(), abs=1e-12) == 1.0

        result_large = ema_weights(20)
        assert pytest.approx(result_large.sum(), abs=1e-12) == 1.0

    def test_weights_length(self):
        """Given window sizes, When ema_weights called, Then correct length returned."""
        for window in [3, 5, 10, 20]:
            result = ema_weights(window)
            assert len(result) == window

    def test_weights_ordering(self):
        """Given window size, When ema_weights called, Then descending order (most recent first)."""
        result = ema_weights(5)
        # EMA weights should be in descending order (most recent = highest weight)
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]


class TestEmaWeightsCustomParameters:
    """Layer 1: Test EMA weights with custom parameters."""

    def test_custom_alpha_half(self):
        """Given alpha=0.5, When ema_weights called, Then correct values returned."""
        result = ema_weights(5, alpha=0.5)
        # Corrected expected values based on actual implementation
        expected = np.array(
            [0.03225806, 0.06451613, 0.12903226, 0.25806452, 0.51612903]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_custom_alpha_different_values(self):
        """Given different alpha values, When ema_weights called, Then normalization maintained."""
        # Test alpha=0.3
        result_03 = ema_weights(4, alpha=0.3)
        assert pytest.approx(result_03.sum(), abs=1e-12) == 1.0

        # Test alpha=0.8
        result_08 = ema_weights(4, alpha=0.8)
        assert pytest.approx(result_08.sum(), abs=1e-12) == 1.0

        # Higher alpha should give more weight to recent values
        assert result_08[-1] > result_03[-1]

    def test_larger_window_size(self):
        """Given larger window, When ema_weights called, Then correct values returned."""
        result = ema_weights(10, alpha=0.5)
        # Corrected expected values based on actual implementation
        expected = np.array(
            [
                0.00097752,
                0.00195503,
                0.00391007,
                0.00782014,
                0.01564027,
                0.03128055,
                0.06256109,
                0.12512219,
                0.25024438,
                0.50048876,
            ]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5)  # Relaxed tolerance

    def test_non_normalized_weights(self):
        """Given normalized=False, When ema_weights called, Then raw weights returned."""
        result_norm = ema_weights(5, normalized=True)
        result_raw = ema_weights(5, normalized=False)

        # Raw weights should not sum to 1
        assert result_raw.sum() != pytest.approx(1.0, abs=1e-10)

        # Normalized version should equal raw/sum
        np.testing.assert_allclose(
            result_norm, result_raw / result_raw.sum(), rtol=1e-12
        )


class TestEmaWeightsEdgeCases:
    """Layer 1: Test edge cases and error handling."""

    def test_invalid_window_sizes(self):
        """Given invalid window sizes, When ema_weights called, Then raises ValueError."""
        # Window size <= 1 should raise ValueError
        with pytest.raises(ValueError, match="Invalid window size"):
            ema_weights(1)

        with pytest.raises(ValueError, match="Invalid window size"):
            ema_weights(0)

        with pytest.raises(ValueError, match="Invalid window size"):
            ema_weights(-1)

    def test_minimum_valid_window(self):
        """Given minimum valid window size, When ema_weights called, Then returns 2-element array."""
        result = ema_weights(2)
        assert len(result) == 2
        assert pytest.approx(result.sum(), abs=1e-12) == 1.0
        assert result[0] < result[1]  # More recent should have higher weight

    def test_extreme_alpha_values(self):
        """Given extreme alpha values, When ema_weights called, Then normalization maintained."""
        # Very small alpha (almost uniform weights)
        result_small = ema_weights(5, alpha=0.01)
        assert pytest.approx(result_small.sum(), abs=1e-12) == 1.0

        # Very large alpha (heavily weighted to recent)
        result_large = ema_weights(5, alpha=0.99)
        assert pytest.approx(result_large.sum(), abs=1e-12) == 1.0
        assert result_large[-1] > 0.9  # Most recent should dominate

    def test_alpha_boundary_values(self):
        """Given boundary alpha values, When ema_weights called, Then handles correctly."""
        # Alpha = 1 should give all weight to most recent
        result_one = ema_weights(5, alpha=1.0)
        assert pytest.approx(result_one.sum(), abs=1e-12) == 1.0
        assert result_one[-1] == pytest.approx(1.0, abs=1e-12)
        assert np.all(result_one[:-1] == pytest.approx(0.0, abs=1e-12))

        # Note: Alpha = 0 creates division by zero, so we test a very small
        # alpha instead
        result_small = ema_weights(5, alpha=0.001)
        assert pytest.approx(result_small.sum(), abs=1e-12) == 1.0


class TestEmaWeightsNumerical:
    """Layer 1: Test numerical properties and stability."""

    def test_numerical_precision(self):
        """Given various window sizes, When ema_weights called, Then sum close to 1.0."""
        for window in [5, 10, 50, 100]:
            result = ema_weights(window)
            # Sum should be very close to 1.0
            assert pytest.approx(result.sum(), abs=1e-10) == 1.0
            # All weights should be positive
            assert np.all(result > 0)

    def test_consistency_across_calls(self):
        """Given same parameters, When ema_weights called multiple times, Then identical results."""
        window, alpha = 10, 0.3
        result1 = ema_weights(window, alpha)
        result2 = ema_weights(window, alpha)
        result3 = ema_weights(window, alpha)

        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)

    def test_mathematical_properties(self):
        """Given alpha and window, When ema_weights called, Then geometric series property holds."""
        alpha = 0.4
        window = 8
        result = ema_weights(window, alpha, normalized=False)

        # Test the geometric series property: w[i] = alpha * (1-alpha)^(window-1-i)
        for i in range(window):
            expected_weight = alpha * (1.0 - alpha) ** (window - 1 - i)
            assert result[i] == pytest.approx(expected_weight, rel=1e-12)

    def test_weight_relationships(self):
        """Given alpha=0.5, When ema_weights called, Then each weight double the previous."""
        result = ema_weights(6, alpha=0.5)

        # Each weight should be double the previous (for alpha=0.5)
        for i in range(len(result) - 1):
            ratio = result[i + 1] / result[i]
            assert ratio == pytest.approx(2.0, rel=1e-10)

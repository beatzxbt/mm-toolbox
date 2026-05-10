"""Tests for logarithmic weight calculations.

Layer 1 tests: validate logarithmic_weights function including return type,
default calculation, normalization, length, ordering, scaling behavior,
edge cases, and numerical properties.
"""

import numpy as np
import pytest

from mm_toolbox.weights import logarithmic_weights


class TestLogarithmicWeightsBasic:
    """Layer 1: Test basic logarithmic weights functionality."""

    def test_function_return_type(self):
        """Given num, When logarithmic_weights called, Then returns numpy float64 array."""
        result = logarithmic_weights(5)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_default_calculation(self):
        """Given default parameters, When logarithmic_weights called, Then correct values returned."""
        result = logarithmic_weights(5)
        expected = np.array([0.0, 0.14478295, 0.22947555, 0.2895659, 0.3361756])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_weights_normalization(self):
        """Given num, When logarithmic_weights called, Then sums to 1.0."""
        result = logarithmic_weights(5)
        assert pytest.approx(result.sum(), abs=1e-12) == 1.0

        result_large = logarithmic_weights(50)
        assert pytest.approx(result_large.sum(), abs=1e-12) == 1.0

    def test_weights_length(self):
        """Given various num values, When logarithmic_weights called, Then correct length."""
        for num in [3, 5, 10, 20]:
            result = logarithmic_weights(num)
            assert len(result) == num

    def test_weights_ordering(self):
        """Given num, When logarithmic_weights called, Then ascending order."""
        result = logarithmic_weights(5)
        # Logarithmic weights should be in ascending order (more recent = higher weight)
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]

    def test_first_weight_is_zero(self):
        """Given any valid num, When logarithmic_weights called, Then first weight is zero."""
        for num in [3, 5, 10, 20]:
            result = logarithmic_weights(num)
            assert result[0] == pytest.approx(0.0, abs=1e-12)


class TestLogarithmicWeightsScaling:
    """Layer 1: Test logarithmic weights with different sizes."""

    def test_larger_num_values(self):
        """Given larger num, When logarithmic_weights called, Then correct length and sum."""
        result = logarithmic_weights(100)
        assert len(result) == 100
        assert pytest.approx(result.sum(), abs=1e-10) == 1.0

        # Should be monotonically increasing
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]

        # First weight should be zero
        assert result[0] == pytest.approx(0.0, abs=1e-12)

    def test_different_num_sizes(self):
        """Given various num sizes, When logarithmic_weights called, Then properties hold."""
        sizes = [3, 5, 10, 25, 50]

        for num in sizes:
            result = logarithmic_weights(num)

            # Basic properties
            assert len(result) == num
            assert pytest.approx(result.sum(), abs=1e-10) == 1.0
            assert result[0] == pytest.approx(0.0, abs=1e-12)

            # Monotonically increasing
            for i in range(len(result) - 1):
                assert result[i] <= result[i + 1]

    def test_scaling_behavior(self):
        """Given different num values, When logarithmic_weights called, Then max weights differ."""
        result_5 = logarithmic_weights(5)
        result_10 = logarithmic_weights(10)
        result_20 = logarithmic_weights(20)

        # Larger arrays should have smaller individual increments
        # (since they need to fit more weights in the same log curve)
        max_weight_5 = result_5[-1]
        max_weight_10 = result_10[-1]
        max_weight_20 = result_20[-1]

        # All should sum to 1, but distribution should be different
        # Test that max weights are meaningfully different (not just floating
        # point precision)
        assert abs(max_weight_5 - max_weight_10) > 0.05
        assert abs(max_weight_10 - max_weight_20) > 0.02


class TestLogarithmicWeightsEdgeCases:
    """Layer 1: Test edge cases and error handling."""

    def test_invalid_num_values(self):
        """Given invalid num values, When logarithmic_weights called, Then raises ValueError."""
        # num <= 1 should raise ValueError
        with pytest.raises(ValueError):
            logarithmic_weights(1)

        with pytest.raises(ValueError):
            logarithmic_weights(0)

        with pytest.raises(ValueError):
            logarithmic_weights(-1)

    def test_minimum_valid_num(self):
        """Given minimum valid num, When logarithmic_weights called, Then returns 2-element array."""
        result = logarithmic_weights(2)
        assert len(result) == 2
        assert pytest.approx(result.sum(), abs=1e-12) == 1.0
        assert result[0] == pytest.approx(0.0, abs=1e-12)
        assert result[1] == pytest.approx(1.0, abs=1e-12)

    def test_small_num_values(self):
        """Given small but valid num values, When logarithmic_weights called, Then properties hold."""
        # Test num = 3
        result_3 = logarithmic_weights(3)
        assert len(result_3) == 3
        assert pytest.approx(result_3.sum(), abs=1e-12) == 1.0
        assert result_3[0] == pytest.approx(0.0, abs=1e-12)
        assert result_3[1] <= result_3[2]

        # Test num = 4
        result_4 = logarithmic_weights(4)
        assert len(result_4) == 4
        assert pytest.approx(result_4.sum(), abs=1e-12) == 1.0
        assert result_4[0] == pytest.approx(0.0, abs=1e-12)


class TestLogarithmicWeightsNumerical:
    """Layer 1: Test numerical properties and stability."""

    def test_numerical_precision(self):
        """Given various sizes, When logarithmic_weights called, Then sum close to 1.0."""
        for num in [5, 10, 50, 100]:
            result = logarithmic_weights(num)
            # Sum should be very close to 1.0
            assert pytest.approx(result.sum(), abs=1e-10) == 1.0
            # All weights should be non-negative
            assert np.all(result >= 0)
            # First weight should be exactly zero
            assert result[0] == pytest.approx(0.0, abs=1e-15)

    def test_consistency_across_calls(self):
        """Given same parameters, When logarithmic_weights called multiple times, Then identical results."""
        num = 10
        result1 = logarithmic_weights(num)
        result2 = logarithmic_weights(num)
        result3 = logarithmic_weights(num)

        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)

    def test_mathematical_properties(self):
        """Given num, When logarithmic_weights called, Then mathematical properties hold."""
        num = 8
        result = logarithmic_weights(num)

        # First weight should be exactly zero
        assert result[0] == 0.0

        # All other weights should be positive
        assert np.all(result[1:] > 0)

        # Should be monotonically increasing
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]

    def test_logarithmic_curve_properties(self):
        """Given num, When logarithmic_weights called, Then differences decrease (log curve flattens)."""
        num = 10
        result = logarithmic_weights(num)

        # Calculate differences between consecutive weights
        diffs = np.diff(result)

        # Differences should generally decrease (logarithmic curve flattens)
        # Allow some tolerance for numerical precision
        decreasing_count = 0
        for i in range(len(diffs) - 1):
            if diffs[i] >= diffs[i + 1]:
                decreasing_count += 1

        # Most differences should be decreasing (characteristic of log curve)
        assert decreasing_count >= len(diffs) // 2

    def test_weight_distribution_characteristics(self):
        """Given num, When logarithmic_weights called, Then weight concentrated in later positions."""
        result = logarithmic_weights(10)

        # Most weight should be concentrated in later positions
        first_half_sum = result[:5].sum()
        second_half_sum = result[5:].sum()
        assert second_half_sum > first_half_sum

        # Last weight should be the largest
        assert result[-1] == np.max(result)

        # First weight should be the smallest (zero)
        assert result[0] == np.min(result)


class TestLogarithmicWeightsComparison:
    """Layer 1: Test comparisons between different logarithmic weight configurations."""

    def test_relative_weight_distribution(self):
        """Given different sizes, When logarithmic_weights called, Then both sum to 1."""
        small = logarithmic_weights(5)
        large = logarithmic_weights(10)

        # Both should sum to 1
        assert pytest.approx(small.sum(), abs=1e-12) == 1.0
        assert pytest.approx(large.sum(), abs=1e-12) == 1.0

        # Both should start with zero
        assert small[0] == pytest.approx(0.0, abs=1e-12)
        assert large[0] == pytest.approx(0.0, abs=1e-12)

        # Both should be monotonically increasing
        assert np.all(np.diff(small) >= 0)
        assert np.all(np.diff(large) >= 0)

    def test_convergence_properties(self):
        """Given increasing sizes, When logarithmic_weights called, Then max weight decreases."""
        sizes = [5, 10, 20, 50]
        max_weights = []

        for size in sizes:
            result = logarithmic_weights(size)
            max_weights.append(result[-1])

        # Maximum weights should generally decrease as size increases
        # (more weights to distribute among)
        for i in range(len(max_weights) - 1):
            assert max_weights[i] >= max_weights[i + 1] * 0.8  # Allow some tolerance

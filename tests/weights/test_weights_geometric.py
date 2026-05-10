"""Tests for geometric weight calculations.

Layer 1 tests: validate geometric_weights function including return type,
default ratio calculation, normalization, length, ordering, custom parameters,
edge cases, and numerical properties.
"""

import numpy as np
import pytest

import mm_toolbox.weights.geometric as geometric_module
from mm_toolbox.weights import geometric_weights


class TestGeometricWeightsBasic:
    """Layer 1: Test basic geometric weights functionality."""

    def test_function_return_type(self):
        """Given num, When geometric_weights called, Then returns numpy float64 array."""
        result = geometric_weights(5)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_default_ratio_calculation(self):
        """Given default ratio, When geometric_weights called, Then correct values returned."""
        result = geometric_weights(5)
        expected = np.array([0.32778489, 0.24583867, 0.184379, 0.13828425, 0.10371319])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_weights_normalization(self):
        """Given num, When geometric_weights called, Then sums to 1.0."""
        result = geometric_weights(5)
        assert pytest.approx(result.sum(), abs=1e-12) == 1.0

        result_large = geometric_weights(50)
        assert pytest.approx(result_large.sum(), abs=1e-12) == 1.0

    def test_weights_length(self):
        """Given various num values, When geometric_weights called, Then correct length."""
        for num in [3, 5, 10, 20]:
            result = geometric_weights(num)
            assert len(result) == num

    def test_weights_ordering(self):
        """Given num, When geometric_weights called, Then descending order."""
        result = geometric_weights(5)
        # Geometric weights should be in descending order
        for i in range(len(result) - 1):
            assert result[i] >= result[i + 1]

    def test_non_normalized_series(self):
        """Given normalized=False, When geometric_weights called, Then raw series returned."""
        result = geometric_weights(5, r=0.5, normalized=False)
        expected = np.array([1.0, 0.5, 0.25, 0.125, 0.0625], dtype=np.float64)
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.0)
        assert result.dtype == np.float64

    def test_vectorized_construction_uses_numpy(self, monkeypatch):
        """Given monkeypatched numpy, When geometric_weights called, Then uses vectorized ops."""
        calls: dict[str, int] = {"arange": 0, "power": 0}
        original_arange = geometric_module.np.arange
        original_power = geometric_module.np.power

        def tracked_arange(*args, **kwargs):
            calls["arange"] += 1
            return original_arange(*args, **kwargs)

        def tracked_power(*args, **kwargs):
            calls["power"] += 1
            return original_power(*args, **kwargs)

        monkeypatch.setattr(geometric_module.np, "arange", tracked_arange)
        monkeypatch.setattr(geometric_module.np, "power", tracked_power)

        result = geometric_weights(8, r=0.4, normalized=False)
        expected = original_power(np.float64(0.4), original_arange(8, dtype=np.float64))

        np.testing.assert_allclose(result, expected)
        assert calls["arange"] == 1
        assert calls["power"] == 1


class TestGeometricWeightsCustomParameters:
    """Layer 1: Test geometric weights with custom parameters."""

    def test_custom_ratio_half(self):
        """Given r=0.5, When geometric_weights called, Then correct values returned."""
        result = geometric_weights(5, r=0.5)
        expected = np.array(
            [0.51612903, 0.25806452, 0.12903226, 0.06451613, 0.03225806]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_custom_ratio_different_values(self):
        """Given different ratio values, When geometric_weights called, Then normalization maintained."""
        # Test r=0.3 (faster decay)
        result_03 = geometric_weights(5, r=0.3)
        assert pytest.approx(result_03.sum(), abs=1e-12) == 1.0

        # Test r=0.8 (slower decay)
        result_08 = geometric_weights(5, r=0.8)
        assert pytest.approx(result_08.sum(), abs=1e-12) == 1.0

        # Higher ratio should make weights more uniform
        variance_03 = np.var(result_03)
        variance_08 = np.var(result_08)
        assert variance_08 < variance_03  # More uniform = lower variance

    def test_larger_num_with_ratio(self):
        """Given larger num, When geometric_weights called, Then correct length and sum."""
        result = geometric_weights(100, r=0.95)
        assert len(result) == 100
        assert pytest.approx(result.sum(), abs=1e-10) == 1.0

        # Should be monotonically decreasing
        for i in range(len(result) - 1):
            assert result[i] >= result[i + 1]

    def test_ratio_effects_on_distribution(self):
        """Given different ratios, When geometric_weights called, Then distribution differs."""
        num = 10

        # Small ratio = steep decline
        result_small = geometric_weights(num, r=0.2)

        # Large ratio = gradual decline
        result_large = geometric_weights(num, r=0.9)

        # First weight should be larger for smaller ratio
        assert result_small[0] > result_large[0]

        # Last weight should be larger for larger ratio
        assert result_large[-1] > result_small[-1]


class TestGeometricWeightsEdgeCases:
    """Layer 1: Test edge cases and error handling."""

    def test_invalid_num_values(self):
        """Given invalid num values, When geometric_weights called, Then raises ValueError."""
        # num <= 1 should raise ValueError
        with pytest.raises(ValueError):
            geometric_weights(1)

        with pytest.raises(ValueError):
            geometric_weights(0)

        with pytest.raises(ValueError):
            geometric_weights(-1)

    def test_minimum_valid_num(self):
        """Given minimum valid num, When geometric_weights called, Then returns 2-element array."""
        result = geometric_weights(2)
        assert len(result) == 2
        assert pytest.approx(result.sum(), abs=1e-12) == 1.0
        assert result[0] >= result[1]  # Should be decreasing

    def test_extreme_ratio_values(self):
        """Given extreme ratio values, When geometric_weights called, Then handles correctly."""
        # Very small ratio (steep decline)
        result_small = geometric_weights(5, r=0.01)
        assert pytest.approx(result_small.sum(), abs=1e-12) == 1.0
        assert result_small[0] > 0.9  # First weight should dominate

        # Very large ratio (almost uniform)
        result_large = geometric_weights(5, r=0.99)
        assert pytest.approx(result_large.sum(), abs=1e-12) == 1.0
        # Weights should be relatively uniform
        assert np.std(result_large) < 0.1

    def test_ratio_boundary_values(self):
        """Given boundary ratio values, When geometric_weights called, Then handles correctly."""
        # Ratio = 0 should give all weight to first element
        result_zero = geometric_weights(5, r=0.0)
        assert pytest.approx(result_zero.sum(), abs=1e-12) == 1.0
        assert result_zero[0] == pytest.approx(1.0, abs=1e-12)
        assert np.all(result_zero[1:] == pytest.approx(0.0, abs=1e-12))

        # Ratio = 1 should give uniform weights
        result_one = geometric_weights(5, r=1.0)
        assert pytest.approx(result_one.sum(), abs=1e-12) == 1.0
        expected_uniform = 1.0 / 5
        assert np.all(result_one == pytest.approx(expected_uniform, abs=1e-12))


class TestGeometricWeightsNumerical:
    """Layer 1: Test numerical properties and stability."""

    def test_numerical_precision(self):
        """Given various parameters, When geometric_weights called, Then sum close to 1.0."""
        for num in [5, 10, 50]:
            for ratio in [0.1, 0.5, 0.9]:
                result = geometric_weights(num, r=ratio)
                # Sum should be very close to 1.0
                assert pytest.approx(result.sum(), abs=1e-10) == 1.0
                # All weights should be non-negative
                assert np.all(result >= 0)

    def test_consistency_across_calls(self):
        """Given same parameters, When geometric_weights called multiple times, Then identical results."""
        num, ratio = 10, 0.7
        result1 = geometric_weights(num, r=ratio)
        result2 = geometric_weights(num, r=ratio)
        result3 = geometric_weights(num, r=ratio)

        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)

    def test_mathematical_properties(self):
        """Given ratio, When geometric_weights called, Then ratio property holds."""
        num, ratio = 6, 0.6
        result = geometric_weights(num, r=ratio)

        # Test the geometric series property
        # Each weight should be ratio times the previous weight (before normalization)
        # After normalization, ratios should be consistent
        for i in range(1, len(result)):
            if result[i - 1] > 1e-10:  # Avoid division by very small numbers
                computed_ratio = result[i] / result[i - 1]
                assert computed_ratio == pytest.approx(ratio, rel=1e-10)

    def test_weight_relationships(self):
        """Given ratio=0.5, When geometric_weights called, Then each weight half the previous."""
        # Test with ratio = 0.5 (each weight is half the previous)
        result = geometric_weights(5, r=0.5)

        for i in range(1, len(result)):
            ratio = result[i] / result[i - 1]
            assert ratio == pytest.approx(0.5, rel=1e-10)

    def test_scaling_behavior(self):
        """Given different num values, When geometric_weights called, Then proportional relationships preserved."""
        ratio = 0.8

        # Compare weights for different numbers
        result_5 = geometric_weights(5, r=ratio)
        result_10 = geometric_weights(10, r=ratio)

        # First few weights should have similar proportional relationships
        # (though absolute values will differ due to normalization)
        for i in range(1, 5):
            ratio_5 = result_5[i] / result_5[i - 1]
            ratio_10 = result_10[i] / result_10[i - 1]
            assert ratio_5 == pytest.approx(ratio_10, rel=1e-10)

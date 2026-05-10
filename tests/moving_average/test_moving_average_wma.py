"""Tests for Weighted Moving Average (WMA) implementation."""

import numpy as np
import pytest

from mm_toolbox.moving_average import WeightedMovingAverage


class TestWmaInitialize:
    """Test WMA initialization behavior."""

    def test_initialize_correctness_with_linear_weights(self):
        """Test initialize() computes correct WMA with linear weights."""
        window = 5
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ma = WeightedMovingAverage(window=window)
        result = ma.initialize(values)
        weights = np.arange(1, window + 1, dtype=np.float64)
        expected = np.dot(weights, values) / np.sum(weights)
        assert result == pytest.approx(expected, abs=1e-12)

    def test_initialize_wrong_length_raises(self):
        """Test initialize() with wrong array length raises ValueError."""
        ma = WeightedMovingAverage(window=5)
        with pytest.raises(ValueError, match="Input array length must match window"):
            ma.initialize(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="Input array length must match window"):
            ma.initialize(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    def test_initialize_minimum_window(self):
        """Test WMA initialize() with minimum window size."""
        values = np.array([2.0, 4.0])
        ma = WeightedMovingAverage(window=2)
        result = ma.initialize(values)
        expected = (1 * 2.0 + 2 * 4.0) / 3.0
        assert result == pytest.approx(expected, abs=1e-12)


class TestWmaNext:
    """Test WMA next() behavior without state mutation."""

    def test_next_returns_correct_future_value(self):
        """Test next() returns correct value without mutating state."""
        ma = WeightedMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        initial_value = ma.get_value()
        result = ma.next(4.0)
        expected_window = np.array([2.0, 3.0, 4.0])
        weights = np.array([1.0, 2.0, 3.0])
        expected = np.dot(weights, expected_window) / np.sum(weights)
        assert result == pytest.approx(expected, abs=1e-12)
        assert ma.get_value() == pytest.approx(initial_value, abs=1e-12)

    def test_next_before_warm_raises(self):
        """Test next() before initialize() raises ValueError."""
        ma = WeightedMovingAverage(window=3)
        with pytest.raises(ValueError, match="initialized"):
            ma.next(1.0)

    def test_next_does_not_mutate_internal_state(self):
        """Test that next() does not change internal buffer or value."""
        ma = WeightedMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        pre_len = len(ma)
        pre_values = ma.get_values().copy()
        ma.next(10.0)
        post_len = len(ma)
        post_values = ma.get_values()
        assert post_len == pre_len
        np.testing.assert_array_equal(pre_values, post_values)
        assert ma.get_value() == pytest.approx(np.dot([1, 2, 3], [1, 2, 3]) / 6, abs=1e-12)


class TestWmaUpdate:
    """Test WMA update() behavior with rolling weighted sum."""

    def test_update_rolling_weighted_sum(self):
        """Test update() maintains correct rolling weighted sum."""
        ma = WeightedMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        result = ma.update(4.0)
        window = np.array([2.0, 3.0, 4.0])
        weights = np.array([1.0, 2.0, 3.0])
        expected = np.dot(weights, window) / np.sum(weights)
        assert result == pytest.approx(expected, abs=1e-12)

    def test_update_against_naive_weighted_mean(self):
        """Compare WMA update() against naive np.dot(weights, values) / sum(weights)."""
        window = 5
        np.random.seed(42)
        initial = np.random.randn(window)
        updates = np.random.randn(20)
        ma = WeightedMovingAverage(window=window)
        ma.initialize(initial)
        window_state = initial.copy()
        weights = np.arange(1, window + 1, dtype=np.float64)
        for new_val in updates:
            result = ma.update(new_val)
            window_state = np.append(window_state[1:], new_val)
            expected = np.dot(weights, window_state) / np.sum(weights)
            assert result == pytest.approx(expected, abs=1e-10)

    def test_update_tracks_historical_values(self):
        """Test that update() stores historical values correctly."""
        ma = WeightedMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        ma.update(4.0)
        ma.update(5.0)
        values = ma.get_values()
        expected = np.array([
            np.dot([1, 2, 3], [1, 2, 3]) / 6,
            np.dot([1, 2, 3], [2, 3, 4]) / 6,
            np.dot([1, 2, 3], [3, 4, 5]) / 6,
        ], dtype=np.float64)
        np.testing.assert_allclose(values, expected, rtol=1e-12)


class TestWmaEdgeCases:
    """Test WMA edge cases and numerical stability."""

    def test_constant_values(self):
        """Test WMA with constant input values."""
        ma = WeightedMovingAverage(window=4)
        ma.initialize(np.array([5.0, 5.0, 5.0, 5.0]))
        assert ma.get_value() == pytest.approx(5.0, abs=1e-12)
        for _ in range(10):
            result = ma.update(5.0)
            assert result == pytest.approx(5.0, abs=1e-12)

    def test_large_window(self):
        """Test WMA with a larger window size."""
        window = 50
        values = np.arange(window, dtype=np.float64)
        ma = WeightedMovingAverage(window=window)
        result = ma.initialize(values)
        weights = np.arange(1, window + 1, dtype=np.float64)
        expected = np.dot(weights, values) / np.sum(weights)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_increasing_values(self):
        """Test WMA with monotonically increasing values."""
        ma = WeightedMovingAverage(window=4)
        ma.initialize(np.array([1.0, 2.0, 3.0, 4.0]))
        result = ma.update(5.0)
        window = np.array([2.0, 3.0, 4.0, 5.0])
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        expected = np.dot(weights, window) / 10.0
        assert result == pytest.approx(expected, abs=1e-12)

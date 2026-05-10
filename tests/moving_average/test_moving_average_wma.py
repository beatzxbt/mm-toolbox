"""Layer 2 — Component tests for Weighted Moving Average (WMA).

Validates initialisation (linear weights, wrong length rejection, minimum
window), ``next()`` projection without mutation, ``update()`` rolling weighted
sum, comparison against a naive NumPy reference, historical value tracking, and
edge cases (constant values, large windows, monotonic sequences).
"""

import numpy as np
import pytest

from mm_toolbox.moving_average import WeightedMovingAverage


class TestWmaInitialize:
    """Layer 1 — WMA initialisation behaviour."""

    def test_initialize_correctness_with_linear_weights(self):
        """Given a 5-value array, the WMA equals the dot-product of linear weights and values."""
        window = 5
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ma = WeightedMovingAverage(window=window)
        result = ma.initialize(values)
        weights = np.arange(1, window + 1, dtype=np.float64)
        expected = np.dot(weights, values) / np.sum(weights)
        assert result == pytest.approx(expected, abs=1e-12)

    def test_initialize_wrong_length_raises(self):
        """Given an array shorter or longer than *window*, ``initialize()`` raises ``ValueError``."""
        ma = WeightedMovingAverage(window=5)
        with pytest.raises(ValueError, match="Input array length must match window"):
            ma.initialize(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="Input array length must match window"):
            ma.initialize(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    def test_initialize_minimum_window(self):
        """Given the minimum window (2), the WMA uses weights [1, 2]."""
        values = np.array([2.0, 4.0])
        ma = WeightedMovingAverage(window=2)
        result = ma.initialize(values)
        expected = (1 * 2.0 + 2 * 4.0) / 3.0
        assert result == pytest.approx(expected, abs=1e-12)


class TestWmaNext:
    """Layer 2 — ``next()`` projection without state mutation."""

    def test_next_returns_correct_future_value(self):
        """Given a warm WMA, ``next()`` returns the weighted mean of the future window."""
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
        """Given an uninitialised WMA, ``next()`` raises ``ValueError``."""
        ma = WeightedMovingAverage(window=3)
        with pytest.raises(ValueError, match="initialized"):
            ma.next(1.0)

    def test_next_does_not_mutate_internal_state(self):
        """Given ``next()``, internal buffer length and contents remain unchanged."""
        ma = WeightedMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        pre_len = len(ma)
        pre_values = ma.get_values().copy()
        ma.next(10.0)
        post_len = len(ma)
        post_values = ma.get_values()
        assert post_len == pre_len
        np.testing.assert_array_equal(pre_values, post_values)
        assert ma.get_value() == pytest.approx(
            np.dot([1, 2, 3], [1, 2, 3]) / 6, abs=1e-12
        )


class TestWmaUpdate:
    """Layer 2 — ``update()`` rolling weighted sum behaviour."""

    def test_update_rolling_weighted_sum(self):
        """Given successive updates, each result equals the weighted mean of the current window."""
        ma = WeightedMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        result = ma.update(4.0)
        window = np.array([2.0, 3.0, 4.0])
        weights = np.array([1.0, 2.0, 3.0])
        expected = np.dot(weights, window) / np.sum(weights)
        assert result == pytest.approx(expected, abs=1e-12)

    def test_update_against_naive_weighted_mean(self):
        """Given 20 random updates, WMA matches a hand-rolled rolling weighted mean."""
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
        """Given two updates, ``get_values()`` contains the rolling window of WMAs."""
        ma = WeightedMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        ma.update(4.0)
        ma.update(5.0)
        values = ma.get_values()
        expected = np.array(
            [
                np.dot([1, 2, 3], [1, 2, 3]) / 6,
                np.dot([1, 2, 3], [2, 3, 4]) / 6,
                np.dot([1, 2, 3], [3, 4, 5]) / 6,
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(values, expected, rtol=1e-12)


class TestWmaEdgeCases:
    """Layer 2 — Edge cases and numerical stability."""

    def test_constant_values(self):
        """Given constant input values, the WMA remains constant."""
        ma = WeightedMovingAverage(window=4)
        ma.initialize(np.array([5.0, 5.0, 5.0, 5.0]))
        assert ma.get_value() == pytest.approx(5.0, abs=1e-12)
        for _ in range(10):
            result = ma.update(5.0)
            assert result == pytest.approx(5.0, abs=1e-12)

    def test_large_window(self):
        """Given a 50-element window, the WMA equals the weighted sum of the initial array."""
        window = 50
        values = np.arange(window, dtype=np.float64)
        ma = WeightedMovingAverage(window=window)
        result = ma.initialize(values)
        weights = np.arange(1, window + 1, dtype=np.float64)
        expected = np.dot(weights, values) / np.sum(weights)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_increasing_values(self):
        """Given monotonically increasing values, the WMA tracks the weighted trend."""
        ma = WeightedMovingAverage(window=4)
        ma.initialize(np.array([1.0, 2.0, 3.0, 4.0]))
        result = ma.update(5.0)
        window = np.array([2.0, 3.0, 4.0, 5.0])
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        expected = np.dot(weights, window) / 10.0
        assert result == pytest.approx(expected, abs=1e-12)

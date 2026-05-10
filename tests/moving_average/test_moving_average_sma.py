"""Layer 2 — Component tests for Simple Moving Average (SMA).

Validates initialisation (exact window length, wrong length rejection),
``next()`` projection without mutation, ``update()`` rolling-sum correctness,
comparison against a naive NumPy reference, historical value tracking, and
edge cases (minimum window, constant values, large windows).
"""

import numpy as np
import pytest

from mm_toolbox.moving_average import SimpleMovingAverage


class TestSmaInitialize:
    """Layer 1 — SMA initialisation behaviour."""

    def test_initialize_exact_window_length(self):
        """Given an array of exactly *window* length, the SMA equals the mean and the MA is warm."""
        window = 5
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ma = SimpleMovingAverage(window=window)
        result = ma.initialize(values)
        expected = np.mean(values)
        assert result == pytest.approx(expected, abs=1e-12)
        assert ma.get_value() == pytest.approx(expected, abs=1e-12)
        assert ma.next(6.0) == pytest.approx(4.0, abs=1e-12)

    def test_initialize_wrong_length_raises(self):
        """Given an array shorter or longer than *window*, ``initialize()`` raises ``ValueError``."""
        ma = SimpleMovingAverage(window=5)
        with pytest.raises(ValueError, match="Input array length must match window"):
            ma.initialize(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="Input array length must match window"):
            ma.initialize(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    def test_initialize_computes_correct_value(self):
        """Given a 5-element array, the SMA is the arithmetic mean (30.0)."""
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        ma = SimpleMovingAverage(window=5)
        result = ma.initialize(values)
        assert result == pytest.approx(30.0, abs=1e-12)


class TestSmaNext:
    """Layer 2 — ``next()`` projection without state mutation."""

    def test_next_returns_correct_future_value(self):
        """Given a warm SMA, ``next()`` returns the mean of the future window without changing state."""
        ma = SimpleMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        initial_value = ma.get_value()
        result = ma.next(4.0)
        expected = np.mean(np.array([2.0, 3.0, 4.0]))
        assert result == pytest.approx(expected, abs=1e-12)
        assert ma.get_value() == pytest.approx(initial_value, abs=1e-12)

    def test_next_before_warm_raises(self):
        """Given an uninitialised SMA, ``next()`` raises ``ValueError``."""
        ma = SimpleMovingAverage(window=3)
        with pytest.raises(ValueError, match="initialized"):
            ma.next(1.0)

    def test_next_does_not_mutate_internal_state(self):
        """Given ``next()``, internal buffer length and contents remain unchanged."""
        ma = SimpleMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        pre_len = len(ma)
        pre_values = ma.get_values().copy()
        ma.next(10.0)
        post_len = len(ma)
        post_values = ma.get_values()
        assert post_len == pre_len
        np.testing.assert_array_equal(pre_values, post_values)
        assert ma.get_value() == pytest.approx(2.0, abs=1e-12)


class TestSmaUpdate:
    """Layer 2 — ``update()`` rolling-sum behaviour."""

    def test_update_rolling_sum_correctness(self):
        """Given successive updates, each result equals the mean of the current window."""
        ma = SimpleMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        result = ma.update(4.0)
        assert result == pytest.approx(3.0, abs=1e-12)
        result = ma.update(5.0)
        assert result == pytest.approx(4.0, abs=1e-12)
        result = ma.update(6.0)
        assert result == pytest.approx(5.0, abs=1e-12)

    def test_update_against_naive_mean(self):
        """Given 20 random updates, SMA matches a hand-rolled rolling window mean."""
        window = 5
        np.random.seed(42)
        initial = np.random.randn(window)
        updates = np.random.randn(20)
        ma = SimpleMovingAverage(window=window)
        ma.initialize(initial)
        window_state = initial.copy()
        for new_val in updates:
            result = ma.update(new_val)
            window_state = np.append(window_state[1:], new_val)
            expected = np.mean(window_state)
            assert result == pytest.approx(expected, abs=1e-10)

    def test_update_tracks_historical_values(self):
        """Given two updates, ``get_values()`` contains the rolling window of SMAs."""
        ma = SimpleMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        ma.update(4.0)
        ma.update(5.0)
        values = ma.get_values()
        expected = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        np.testing.assert_allclose(values, expected, rtol=1e-12)


class TestSmaEdgeCases:
    """Layer 2 — Edge cases and numerical stability."""

    def test_window_two(self):
        """Given the minimum valid window size (2), SMA behaves correctly."""
        ma = SimpleMovingAverage(window=2)
        ma.initialize(np.array([1.0, 3.0]))
        assert ma.get_value() == pytest.approx(2.0, abs=1e-12)
        result = ma.update(5.0)
        assert result == pytest.approx(4.0, abs=1e-12)

    def test_constant_values(self):
        """Given constant input values, the SMA remains constant."""
        ma = SimpleMovingAverage(window=4)
        ma.initialize(np.array([5.0, 5.0, 5.0, 5.0]))
        assert ma.get_value() == pytest.approx(5.0, abs=1e-12)
        for _ in range(10):
            result = ma.update(5.0)
            assert result == pytest.approx(5.0, abs=1e-12)

    def test_large_window(self):
        """Given a 100-element window, the SMA equals the mean of the initial array."""
        window = 100
        values = np.arange(window, dtype=np.float64)
        ma = SimpleMovingAverage(window=window)
        result = ma.initialize(values)
        expected = np.mean(values)
        assert result == pytest.approx(expected, abs=1e-10)

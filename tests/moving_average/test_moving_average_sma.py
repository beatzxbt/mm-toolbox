"""Tests for Simple Moving Average (SMA) implementation."""

import numpy as np
import pytest

from mm_toolbox.moving_average import SimpleMovingAverage


class TestSmaInitialize:
    """Test SMA initialization behavior."""

    def test_initialize_exact_window_length(self):
        """Test initialize() with array of exactly window length."""
        window = 5
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ma = SimpleMovingAverage(window=window)
        result = ma.initialize(values)
        expected = np.mean(values)
        assert result == pytest.approx(expected, abs=1e-12)
        assert ma.get_value() == pytest.approx(expected, abs=1e-12)
        # Warm state is verified by the fact that next()/update() succeed
        assert ma.next(6.0) == pytest.approx(4.0, abs=1e-12)

    def test_initialize_wrong_length_raises(self):
        """Test initialize() with wrong array length raises ValueError."""
        ma = SimpleMovingAverage(window=5)
        with pytest.raises(ValueError, match="Input array length must match window"):
            ma.initialize(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="Input array length must match window"):
            ma.initialize(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    def test_initialize_computes_correct_value(self):
        """Test that initialize() computes the correct SMA."""
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        ma = SimpleMovingAverage(window=5)
        result = ma.initialize(values)
        assert result == pytest.approx(30.0, abs=1e-12)


class TestSmaNext:
    """Test SMA next() behavior without state mutation."""

    def test_next_returns_correct_future_value(self):
        """Test next() returns correct value without mutating state."""
        ma = SimpleMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        initial_value = ma.get_value()
        result = ma.next(4.0)
        expected = np.mean(np.array([2.0, 3.0, 4.0]))
        assert result == pytest.approx(expected, abs=1e-12)
        assert ma.get_value() == pytest.approx(initial_value, abs=1e-12)

    def test_next_before_warm_raises(self):
        """Test next() before initialize() raises ValueError."""
        ma = SimpleMovingAverage(window=3)
        with pytest.raises(ValueError, match="initialized"):
            ma.next(1.0)

    def test_next_does_not_mutate_internal_state(self):
        """Test that next() does not change internal buffer or value."""
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
    """Test SMA update() behavior with rolling sum."""

    def test_update_rolling_sum_correctness(self):
        """Test update() maintains correct rolling sum."""
        ma = SimpleMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        result = ma.update(4.0)
        assert result == pytest.approx(3.0, abs=1e-12)
        result = ma.update(5.0)
        assert result == pytest.approx(4.0, abs=1e-12)
        result = ma.update(6.0)
        assert result == pytest.approx(5.0, abs=1e-12)

    def test_update_against_naive_mean(self):
        """Compare SMA update() against naive np.mean(window)."""
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
        """Test that update() stores historical values correctly."""
        ma = SimpleMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        ma.update(4.0)
        ma.update(5.0)
        values = ma.get_values()
        # initialize() pushes one value; each update() pushes one more
        expected = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        np.testing.assert_allclose(values, expected, rtol=1e-12)


class TestSmaEdgeCases:
    """Test SMA edge cases and numerical stability."""

    def test_window_two(self):
        """Test SMA with minimum valid window size."""
        ma = SimpleMovingAverage(window=2)
        ma.initialize(np.array([1.0, 3.0]))
        assert ma.get_value() == pytest.approx(2.0, abs=1e-12)
        result = ma.update(5.0)
        assert result == pytest.approx(4.0, abs=1e-12)

    def test_constant_values(self):
        """Test SMA with constant input values."""
        ma = SimpleMovingAverage(window=4)
        ma.initialize(np.array([5.0, 5.0, 5.0, 5.0]))
        assert ma.get_value() == pytest.approx(5.0, abs=1e-12)
        for _ in range(10):
            result = ma.update(5.0)
            assert result == pytest.approx(5.0, abs=1e-12)

    def test_large_window(self):
        """Test SMA with a larger window size."""
        window = 100
        values = np.arange(window, dtype=np.float64)
        ma = SimpleMovingAverage(window=window)
        result = ma.initialize(values)
        expected = np.mean(values)
        assert result == pytest.approx(expected, abs=1e-10)

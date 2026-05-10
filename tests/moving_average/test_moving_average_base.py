"""Tests for moving average base class behavior."""

import numpy as np
import pytest

from mm_toolbox.moving_average import SimpleMovingAverage, WeightedMovingAverage


class TestMovingAverageWindowValidation:
    """Test window size validation across moving average types."""

    def test_window_zero_raises(self):
        """Test that window=0 raises ValueError."""
        with pytest.raises(ValueError, match="window must be positive"):
            SimpleMovingAverage(window=0)

    def test_window_one_raises(self):
        """Test that window=1 raises ValueError."""
        with pytest.raises(ValueError, match="window must be positive"):
            SimpleMovingAverage(window=1)

    def test_window_negative_raises(self):
        """Test that negative window raises ValueError."""
        with pytest.raises(ValueError, match="window must be positive"):
            WeightedMovingAverage(window=-1)

    def test_window_two_ok(self):
        """Test that window=2 is accepted."""
        ma = SimpleMovingAverage(window=2)
        assert ma is not None


class TestMovingAverageFastMode:
    """Test fast mode behavior where historical storage is skipped."""

    def test_fast_mode_get_values_raises(self):
        """Test that get_values() raises ValueError when is_fast=True."""
        ma = SimpleMovingAverage(window=3, fast=True)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="fast mode"):
            ma.get_values()

    def test_fast_mode_len_raises(self):
        """Test that __len__() raises ValueError when is_fast=True."""
        ma = WeightedMovingAverage(window=3, fast=True)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="fast mode"):
            len(ma)


class TestMovingAverageOperationsBeforeWarm:
    """Test that operations before initialization raise errors for SMA/WMA."""

    def test_sma_next_before_warm_raises(self):
        """Test SMA.next() before initialize() raises ValueError."""
        ma = SimpleMovingAverage(window=3)
        with pytest.raises(ValueError, match="initialized"):
            ma.next(1.0)

    def test_sma_update_before_warm_raises(self):
        """Test SMA.update() before initialize() raises ValueError."""
        ma = SimpleMovingAverage(window=3)
        with pytest.raises(ValueError, match="initialized"):
            ma.update(1.0)

    def test_wma_next_before_warm_raises(self):
        """Test WMA.next() before initialize() raises ValueError."""
        ma = WeightedMovingAverage(window=3)
        with pytest.raises(ValueError, match="initialized"):
            ma.next(1.0)

    def test_wma_update_before_warm_raises(self):
        """Test WMA.update() before initialize() raises ValueError."""
        ma = WeightedMovingAverage(window=3)
        with pytest.raises(ValueError, match="initialized"):
            ma.update(1.0)


class TestMovingAverageHistoricalValues:
    """Test historical value storage and retrieval."""

    def test_get_values_returns_numpy_array(self):
        """Test that get_values() returns a numpy array."""
        ma = SimpleMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        values = ma.get_values()
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.float64

    def test_get_values_correct_after_updates(self):
        """Test get_values() returns correct array after multiple updates."""
        ma = SimpleMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        ma.update(4.0)
        ma.update(5.0)
        values = ma.get_values()
        # initialize() pushes one value; each update() pushes one more
        expected = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        np.testing.assert_allclose(values, expected, rtol=1e-12)

    def test_len_tracks_stored_values(self):
        """Test __len__() tracks number of stored values correctly."""
        ma = WeightedMovingAverage(window=3)
        assert len(ma) == 0
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        assert len(ma) == 1
        ma.update(4.0)
        assert len(ma) == 2
        ma.update(5.0)
        assert len(ma) == 3
        ma.update(6.0)
        assert len(ma) == 4

    def test_get_value_returns_current(self):
        """Test get_value() returns the current moving average value."""
        ma = SimpleMovingAverage(window=3)
        result = ma.initialize(np.array([1.0, 2.0, 3.0]))
        assert ma.get_value() == pytest.approx(result, abs=1e-12)
        ma.update(4.0)
        assert ma.get_value() == pytest.approx(3.0, abs=1e-12)

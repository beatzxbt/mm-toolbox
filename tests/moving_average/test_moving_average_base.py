"""Layer 1 — Primitives tests for moving-average base class behaviour.

Covers window-size validation (edge values: 0, 1, negative, minimum valid),
fast-mode restrictions (history access forbidden), pre-warm error handling
for SMA/WMA, and historical value storage/retrieval.
"""

import numpy as np
import pytest

from mm_toolbox.moving_average import SimpleMovingAverage, WeightedMovingAverage


class TestMovingAverageWindowValidation:
    """Layer 1 — Window size validation across moving average types."""

    def test_window_zero_raises(self):
        """Given window=0, construction raises ``ValueError``."""
        with pytest.raises(ValueError, match="window must be positive"):
            SimpleMovingAverage(window=0)

    def test_window_one_raises(self):
        """Given window=1, construction raises ``ValueError``."""
        with pytest.raises(ValueError, match="window must be positive"):
            SimpleMovingAverage(window=1)

    def test_window_negative_raises(self):
        """Given a negative window, construction raises ``ValueError``."""
        with pytest.raises(ValueError, match="window must be positive"):
            WeightedMovingAverage(window=-1)

    def test_window_two_ok(self):
        """Given window=2, construction succeeds (minimum valid window)."""
        ma = SimpleMovingAverage(window=2)
        assert ma is not None


class TestMovingAverageFastMode:
    """Layer 1 — Fast mode behaviour where historical storage is skipped."""

    def test_fast_mode_get_values_raises(self):
        """Given ``fast=True``, ``get_values()`` raises ``ValueError``."""
        ma = SimpleMovingAverage(window=3, fast=True)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="fast mode"):
            ma.get_values()

    def test_fast_mode_len_raises(self):
        """Given ``fast=True``, ``__len__`` raises ``ValueError``."""
        ma = WeightedMovingAverage(window=3, fast=True)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="fast mode"):
            len(ma)


class TestMovingAverageOperationsBeforeWarm:
    """Layer 1 — Pre-initialisation error handling for SMA/WMA."""

    def test_sma_next_before_warm_raises(self):
        """Given an uninitialised SMA, ``next()`` raises ``ValueError``."""
        ma = SimpleMovingAverage(window=3)
        with pytest.raises(ValueError, match="initialized"):
            ma.next(1.0)

    def test_sma_update_before_warm_raises(self):
        """Given an uninitialised SMA, ``update()`` raises ``ValueError``."""
        ma = SimpleMovingAverage(window=3)
        with pytest.raises(ValueError, match="initialized"):
            ma.update(1.0)

    def test_wma_next_before_warm_raises(self):
        """Given an uninitialised WMA, ``next()`` raises ``ValueError``."""
        ma = WeightedMovingAverage(window=3)
        with pytest.raises(ValueError, match="initialized"):
            ma.next(1.0)

    def test_wma_update_before_warm_raises(self):
        """Given an uninitialised WMA, ``update()`` raises ``ValueError``."""
        ma = WeightedMovingAverage(window=3)
        with pytest.raises(ValueError, match="initialized"):
            ma.update(1.0)


class TestMovingAverageHistoricalValues:
    """Layer 1 — Historical value storage and retrieval tests."""

    def test_get_values_returns_numpy_array(self):
        """Given an initialised MA, ``get_values()`` returns a ``float64`` ndarray."""
        ma = SimpleMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        values = ma.get_values()
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.float64

    def test_get_values_correct_after_updates(self):
        """Given two updates after initialisation, ``get_values()`` contains the rolling window."""
        ma = SimpleMovingAverage(window=3)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        ma.update(4.0)
        ma.update(5.0)
        values = ma.get_values()
        expected = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        np.testing.assert_allclose(values, expected, rtol=1e-12)

    def test_len_tracks_stored_values(self):
        """Given successive updates, ``__len__`` tracks the number of stored values."""
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
        """Given updates, ``get_value()`` matches the most recent computed value."""
        ma = SimpleMovingAverage(window=3)
        result = ma.initialize(np.array([1.0, 2.0, 3.0]))
        assert ma.get_value() == pytest.approx(result, abs=1e-12)
        ma.update(4.0)
        assert ma.get_value() == pytest.approx(3.0, abs=1e-12)

"""Tests for Exponential Moving Average (EMA) implementation."""

import numpy as np
import pytest

from mm_toolbox.moving_average import ExponentialMovingAverage


class TestEmaInitialize:
    """Test EMA initialization and warm behavior."""

    def test_initialize_warms_correctly(self):
        """Test that initialize() correctly warms the EMA."""
        window = 5
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ma = ExponentialMovingAverage(window=window)
        result = ma.initialize(values)
        # Verify warm state indirectly: next() should work without raising
        _ = ma.next(6.0)
        assert result == pytest.approx(ma.get_value(), abs=1e-12)

    def test_initialize_computes_correctly(self):
        """Test initialize() computes EMA correctly over the input sequence."""
        window = 3
        alpha = 2.0 / (window + 1)
        values = np.array([10.0, 20.0, 30.0])
        ma = ExponentialMovingAverage(window=window)
        result = ma.initialize(values)
        # EMA starts at first value, then updates
        expected = values[0]
        for v in values[1:]:
            expected = alpha * v + (1 - alpha) * expected
        assert result == pytest.approx(expected, abs=1e-12)

    def test_initialize_single_value(self):
        """Test initialize() with a single value sets that as the EMA."""
        ma = ExponentialMovingAverage(window=3)
        result = ma.initialize(np.array([42.0]))
        assert result == pytest.approx(42.0, abs=1e-12)
        assert ma.get_value() == pytest.approx(42.0, abs=1e-12)


class TestEmaColdStart:
    """Test EMA cold-start behavior without initialize()."""

    def test_can_operate_without_initialize(self):
        """Test that EMA can operate without calling initialize()."""
        ma = ExponentialMovingAverage(window=5)
        # Before any update, next() returns 0.0 (cold state)
        assert ma.next(10.0) == pytest.approx(0.0, abs=1e-12)
        result = ma.update(10.0)
        assert result == pytest.approx(10.0, abs=1e-12)
        # After first update, EMA is warm and next() uses the formula
        assert ma.next(20.0) != pytest.approx(0.0, abs=1e-12)

    def test_first_update_sets_value(self):
        """Test that first update() sets the EMA to the input value."""
        ma = ExponentialMovingAverage(window=5)
        result = ma.update(5.0)
        assert result == pytest.approx(5.0, abs=1e-12)
        assert ma.get_value() == pytest.approx(5.0, abs=1e-12)

    def test_next_before_warm_returns_zero(self):
        """Test next() before any update returns 0.0."""
        ma = ExponentialMovingAverage(window=5)
        result = ma.next(10.0)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_cold_start_sequence(self):
        """Test EMA behavior during a cold-start sequence."""
        ma = ExponentialMovingAverage(window=3)
        alpha = 2.0 / (3 + 1)
        # First update sets value directly
        r1 = ma.update(10.0)
        assert r1 == pytest.approx(10.0, abs=1e-12)
        # Second update uses EMA formula
        r2 = ma.update(20.0)
        expected = alpha * 20.0 + (1 - alpha) * 10.0
        assert r2 == pytest.approx(expected, abs=1e-12)
        # Third update continues
        r3 = ma.update(30.0)
        expected = alpha * 30.0 + (1 - alpha) * expected
        assert r3 == pytest.approx(expected, abs=1e-12)


class TestEmaCustomAlpha:
    """Test EMA with custom alpha parameter."""

    def test_custom_alpha_overrides_default(self):
        """Test that custom alpha overrides the default calculation."""
        window = 5
        custom_alpha = 0.5
        ma = ExponentialMovingAverage(window=window, alpha=custom_alpha)
        ma.initialize(np.array([100.0]))
        result = ma.update(200.0)
        # With custom alpha=0.5, EMA should be exactly 150.0
        expected = custom_alpha * 200.0 + (1.0 - custom_alpha) * 100.0
        assert result == pytest.approx(expected, abs=1e-12)

    def test_custom_alpha_affects_computation(self):
        """Test that custom alpha produces different results than default."""
        window = 5
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ma_default = ExponentialMovingAverage(window=window)
        ma_custom = ExponentialMovingAverage(window=window, alpha=0.8)
        result_default = ma_default.initialize(values.copy())
        result_custom = ma_custom.initialize(values.copy())
        assert result_default != pytest.approx(result_custom, abs=1e-6)

    def test_custom_alpha_formula(self):
        """Verify EMA formula with custom alpha."""
        alpha = 0.7
        ma = ExponentialMovingAverage(window=10, alpha=alpha)
        ma.initialize(np.array([100.0]))
        result = ma.update(200.0)
        expected = alpha * 200.0 + (1 - alpha) * 100.0
        assert result == pytest.approx(expected, abs=1e-12)


class TestEmaNext:
    """Test EMA next() behavior."""

    def test_next_returns_correct_value(self):
        """Test next() returns correct future EMA value."""
        window = 5
        ma = ExponentialMovingAverage(window=window)
        ma.initialize(np.array([10.0, 20.0, 30.0]))
        current = ma.get_value()
        result = ma.next(40.0)
        alpha = 2.0 / (window + 1)
        expected = alpha * 40.0 + (1 - alpha) * current
        assert result == pytest.approx(expected, abs=1e-12)

    def test_next_does_not_mutate_state(self):
        """Test that next() does not change internal value."""
        ma = ExponentialMovingAverage(window=5)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        pre_value = ma.get_value()
        pre_len = len(ma)
        ma.next(10.0)
        assert ma.get_value() == pytest.approx(pre_value, abs=1e-12)
        assert len(ma) == pre_len


class TestEmaFormula:
    """Test that EMA follows the standard recurrence formula."""

    def test_ema_recurrence_formula(self):
        """Verify EMA_t = alpha * x_t + (1-alpha) * EMA_{t-1}."""
        window = 5
        alpha = 2.0 / (window + 1)
        ma = ExponentialMovingAverage(window=window)
        ma.initialize(np.array([100.0]))
        for new_val in [110.0, 105.0, 115.0, 120.0]:
            prev_ema = ma.get_value()
            result = ma.update(new_val)
            expected = alpha * new_val + (1 - alpha) * prev_ema
            assert result == pytest.approx(expected, abs=1e-12)

    def test_ema_against_naive_computation(self):
        """Compare EMA against naive recurrence computation."""
        window = 10
        np.random.seed(42)
        values = np.random.randn(50)
        ma = ExponentialMovingAverage(window=window)
        # Cold start with first value
        ema = values[0]
        ma.update(values[0])
        assert ma.get_value() == pytest.approx(ema, abs=1e-12)
        alpha = 2.0 / (window + 1)
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema
            result = ma.update(v)
            assert result == pytest.approx(ema, abs=1e-10)

    def test_ema_with_high_alpha(self):
        """Test EMA with alpha close to 1 tracks recent values closely."""
        ma = ExponentialMovingAverage(window=10, alpha=0.9)
        ma.initialize(np.array([0.0]))
        ma.update(100.0)
        # With alpha=0.9, EMA should be close to 90
        assert ma.get_value() == pytest.approx(90.0, abs=1e-12)
        ma.update(100.0)
        assert ma.get_value() == pytest.approx(99.0, abs=1e-12)

    def test_ema_with_low_alpha(self):
        """Test EMA with alpha close to 0 changes slowly."""
        ma = ExponentialMovingAverage(window=10, alpha=0.1)
        ma.initialize(np.array([0.0]))
        ma.update(100.0)
        # With alpha=0.1, EMA should be close to 10
        assert ma.get_value() == pytest.approx(10.0, abs=1e-12)
        ma.update(100.0)
        assert ma.get_value() == pytest.approx(19.0, abs=1e-12)


class TestEmaHistoricalValues:
    """Test EMA historical value storage."""

    def test_get_values_after_updates(self):
        """Test get_values() returns correct array after EMA updates."""
        ma = ExponentialMovingAverage(window=5)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        ma.update(4.0)
        ma.update(5.0)
        values = ma.get_values()
        # initialize() inserts first value directly, then each update()
        # (including the two inside initialize for values[1:] and the
        # two explicit calls above) pushes one value
        assert len(values) == 5
        assert values[-1] == pytest.approx(ma.get_value(), abs=1e-12)

    def test_len_tracks_stored_values(self):
        """Test __len__() tracks number of stored EMA values."""
        ma = ExponentialMovingAverage(window=3)
        assert len(ma) == 0
        # First cold update sets value but does NOT push to ringbuffer
        ma.update(1.0)
        assert len(ma) == 0
        # Subsequent updates push to ringbuffer
        ma.update(2.0)
        assert len(ma) == 1
        ma.update(3.0)
        assert len(ma) == 2
        ma.update(4.0)
        assert len(ma) == 3

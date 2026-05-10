"""Layer 2 — Component tests for Exponential Moving Average (EMA).

Validates initialisation (warm-up), cold-start behaviour (zero initial EMA),
custom alpha overrides, the standard recurrence formula, numerical accuracy
against a naive reference, and historical value tracking.
"""

import numpy as np
import pytest

from mm_toolbox.moving_average import ExponentialMovingAverage


class TestEmaInitialize:
    """Layer 1 — EMA initialisation and warm-up behaviour."""

    def test_initialize_warms_correctly(self):
        """Given a full window of values, ``initialize()`` warms the EMA so ``next()`` succeeds."""
        window = 5
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ma = ExponentialMovingAverage(window=window)
        result = ma.initialize(values)
        _ = ma.next(6.0)
        assert result == pytest.approx(ma.get_value(), abs=1e-12)

    def test_initialize_computes_correctly(self):
        """Given a 3-value sequence, the EMA equals the recurrence over all inputs."""
        window = 3
        alpha = 2.0 / (window + 1)
        values = np.array([10.0, 20.0, 30.0])
        ma = ExponentialMovingAverage(window=window)
        result = ma.initialize(values)
        expected = values[0]
        for v in values[1:]:
            expected = alpha * v + (1 - alpha) * expected
        assert result == pytest.approx(expected, abs=1e-12)

    def test_initialize_single_value(self):
        """Given a single-value array, the EMA is initialised to that value."""
        ma = ExponentialMovingAverage(window=3)
        result = ma.initialize(np.array([42.0]))
        assert result == pytest.approx(42.0, abs=1e-12)
        assert ma.get_value() == pytest.approx(42.0, abs=1e-12)


class TestEmaColdStart:
    """Layer 2 — EMA cold-start behaviour without ``initialize()``."""

    def test_can_operate_without_initialize(self):
        """Given no warm-up, ``next()`` returns 0.0 and ``update()`` sets the first value."""
        ma = ExponentialMovingAverage(window=5)
        assert ma.next(10.0) == pytest.approx(0.0, abs=1e-12)
        result = ma.update(10.0)
        assert result == pytest.approx(10.0, abs=1e-12)
        assert ma.next(20.0) != pytest.approx(0.0, abs=1e-12)

    def test_first_update_sets_value(self):
        """Given the first ``update()``, the EMA becomes exactly the input value."""
        ma = ExponentialMovingAverage(window=5)
        result = ma.update(5.0)
        assert result == pytest.approx(5.0, abs=1e-12)
        assert ma.get_value() == pytest.approx(5.0, abs=1e-12)

    def test_next_before_warm_returns_zero(self):
        """Given a cold EMA, ``next()`` returns 0.0."""
        ma = ExponentialMovingAverage(window=5)
        result = ma.next(10.0)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_cold_start_sequence(self):
        """Given a cold start, the first update sets the value and subsequent updates use the EMA formula."""
        ma = ExponentialMovingAverage(window=3)
        alpha = 2.0 / (3 + 1)
        r1 = ma.update(10.0)
        assert r1 == pytest.approx(10.0, abs=1e-12)
        r2 = ma.update(20.0)
        expected = alpha * 20.0 + (1 - alpha) * 10.0
        assert r2 == pytest.approx(expected, abs=1e-12)
        r3 = ma.update(30.0)
        expected = alpha * 30.0 + (1 - alpha) * expected
        assert r3 == pytest.approx(expected, abs=1e-12)


class TestEmaCustomAlpha:
    """Layer 2 — EMA with a user-supplied alpha parameter."""

    def test_custom_alpha_overrides_default(self):
        """Given ``alpha=0.5``, the EMA is exactly the midpoint between old and new values."""
        window = 5
        custom_alpha = 0.5
        ma = ExponentialMovingAverage(window=window, alpha=custom_alpha)
        ma.initialize(np.array([100.0]))
        result = ma.update(200.0)
        expected = custom_alpha * 200.0 + (1.0 - custom_alpha) * 100.0
        assert result == pytest.approx(expected, abs=1e-12)

    def test_custom_alpha_affects_computation(self):
        """Given two EMAs with different alphas, their outputs diverge."""
        window = 5
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ma_default = ExponentialMovingAverage(window=window)
        ma_custom = ExponentialMovingAverage(window=window, alpha=0.8)
        result_default = ma_default.initialize(values.copy())
        result_custom = ma_custom.initialize(values.copy())
        assert result_default != pytest.approx(result_custom, abs=1e-6)

    def test_custom_alpha_formula(self):
        """Given ``alpha=0.7``, the update follows the weighted blend exactly."""
        alpha = 0.7
        ma = ExponentialMovingAverage(window=10, alpha=alpha)
        ma.initialize(np.array([100.0]))
        result = ma.update(200.0)
        expected = alpha * 200.0 + (1 - alpha) * 100.0
        assert result == pytest.approx(expected, abs=1e-12)


class TestEmaNext:
    """Layer 2 — ``next()`` projection without state mutation."""

    def test_next_returns_correct_value(self):
        """Given a warm EMA, ``next()`` returns the exact value that ``update()`` would produce."""
        window = 5
        ma = ExponentialMovingAverage(window=window)
        ma.initialize(np.array([10.0, 20.0, 30.0]))
        current = ma.get_value()
        result = ma.next(40.0)
        alpha = 2.0 / (window + 1)
        expected = alpha * 40.0 + (1 - alpha) * current
        assert result == pytest.approx(expected, abs=1e-12)

    def test_next_does_not_mutate_state(self):
        """Given ``next()``, internal value and history length remain unchanged."""
        ma = ExponentialMovingAverage(window=5)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        pre_value = ma.get_value()
        pre_len = len(ma)
        ma.next(10.0)
        assert ma.get_value() == pytest.approx(pre_value, abs=1e-12)
        assert len(ma) == pre_len


class TestEmaFormula:
    """Layer 2 — EMA recurrence formula correctness."""

    def test_ema_recurrence_formula(self):
        """Given sequential updates, each result matches ``alpha*x + (1-alpha)*prev``."""
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
        """Given 50 random values, the EMA aligns with a hand-rolled recurrence."""
        window = 10
        np.random.seed(42)
        values = np.random.randn(50)
        ma = ExponentialMovingAverage(window=window)
        ema = values[0]
        ma.update(values[0])
        assert ma.get_value() == pytest.approx(ema, abs=1e-12)
        alpha = 2.0 / (window + 1)
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema
            result = ma.update(v)
            assert result == pytest.approx(ema, abs=1e-10)

    def test_ema_with_high_alpha(self):
        """Given ``alpha=0.9``, the EMA converges to recent values within two steps."""
        ma = ExponentialMovingAverage(window=10, alpha=0.9)
        ma.initialize(np.array([0.0]))
        ma.update(100.0)
        assert ma.get_value() == pytest.approx(90.0, abs=1e-12)
        ma.update(100.0)
        assert ma.get_value() == pytest.approx(99.0, abs=1e-12)

    def test_ema_with_low_alpha(self):
        """Given ``alpha=0.1``, the EMA changes slowly and tracks the long-term mean."""
        ma = ExponentialMovingAverage(window=10, alpha=0.1)
        ma.initialize(np.array([0.0]))
        ma.update(100.0)
        assert ma.get_value() == pytest.approx(10.0, abs=1e-12)
        ma.update(100.0)
        assert ma.get_value() == pytest.approx(19.0, abs=1e-12)


class TestEmaHistoricalValues:
    """Layer 1 — EMA historical value storage."""

    def test_get_values_after_updates(self):
        """Given updates after initialisation, ``get_values()`` contains all computed EMAs."""
        ma = ExponentialMovingAverage(window=5)
        ma.initialize(np.array([1.0, 2.0, 3.0]))
        ma.update(4.0)
        ma.update(5.0)
        values = ma.get_values()
        assert len(values) == 5
        assert values[-1] == pytest.approx(ma.get_value(), abs=1e-12)

    def test_len_tracks_stored_values(self):
        """Given a cold start, the first update does not push to the ringbuffer."""
        ma = ExponentialMovingAverage(window=3)
        assert len(ma) == 0
        ma.update(1.0)
        assert len(ma) == 0
        ma.update(2.0)
        assert len(ma) == 1
        ma.update(3.0)
        assert len(ma) == 2
        ma.update(4.0)
        assert len(ma) == 3

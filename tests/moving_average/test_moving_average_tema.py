"""Tests for TimeExponentialMovingAverage (TEMA).

TEMA uses elapsed wall-clock time to compute the exponential decay weight
(alpha) of each new sample.  Tests here verify constructor validation,
initialization semantics, the read-only nature of ``next()``, state mutation
by ``update()``, and the fast-mode memory trade-off.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mm_toolbox.moving_average import TimeExponentialMovingAverage


class TestTemaConstructorValidation:
    """Layer 1 – primitive validation of the constructor."""

    def test_half_life_positive_ok(self):
        """Construction with a positive half-life succeeds."""
        tema = TimeExponentialMovingAverage(window=2, half_life_s=1.0)
        assert isinstance(tema, TimeExponentialMovingAverage)

    def test_half_life_zero_raises(self):
        """``half_life_s=0`` must raise ``ValueError``."""
        with pytest.raises(ValueError, match="Half life must be positive"):
            TimeExponentialMovingAverage(window=2, half_life_s=0.0)

    def test_half_life_negative_raises(self):
        """``half_life_s<0`` must raise ``ValueError``."""
        with pytest.raises(ValueError, match="Half life must be positive"):
            TimeExponentialMovingAverage(window=2, half_life_s=-0.5)


class TestTemaInitialize:
    """Layer 1 – validation and warm-up behaviour of ``initialize()``."""

    def test_initialize_too_short_raises(self):
        """Arrays of length 0 or 1 are rejected."""
        tema = TimeExponentialMovingAverage(window=2, half_life_s=1.0)

        with pytest.raises(ValueError, match="too short"):
            tema.initialize(np.array([], dtype=np.float64))

        with pytest.raises(ValueError, match="too short"):
            tema.initialize(np.array([1.0], dtype=np.float64))

    def test_initialize_warms_correctly(self):
        """After ``initialize()`` the MA is warm and the ringbuffer is populated."""
        tema = TimeExponentialMovingAverage(window=4, half_life_s=1.0)
        values = np.array([10.0, 10.0], dtype=np.float64)

        result = tema.initialize(values)

        assert tema.get_value() == result
        assert len(tema) == 2  # first value + one update

    def test_initialize_clears_previous_state(self):
        """``initialize()`` must discard any prior state."""
        tema = TimeExponentialMovingAverage(window=4, half_life_s=1.0)
        tema.initialize(np.array([1.0, 2.0], dtype=np.float64))
        old_value = tema.get_value()

        tema.initialize(np.array([100.0, 100.0], dtype=np.float64))

        assert tema.get_value() != old_value
        assert tema.get_value() == pytest.approx(100.0, abs=1e-9)


class TestTemaNext:
    """Layer 2 – ``next()`` must be a pure projection without side effects."""

    def test_next_before_warm_returns_input(self):
        """Calling ``next()`` on a cold MA returns the input without side effects."""
        tema = TimeExponentialMovingAverage(window=2, half_life_s=1.0)
        result = tema.next(42.0)

        assert result == pytest.approx(42.0)
        assert tema.get_value() == pytest.approx(0.0)  # visible state unchanged

    def test_next_does_not_mutate_state(self):
        """Critical regression test: ``next()`` must leave all internal state intact."""
        tema = TimeExponentialMovingAverage(window=2, half_life_s=1.0)
        # Use identical values so alpha has no effect on the numeric result.
        tema.initialize(np.array([10.0, 10.0], dtype=np.float64))

        value_before = tema.get_value()
        len_before = len(tema)
        values_before = tema.get_values().copy()

        # Sleep so that a meaningful alpha would be produced if state mutated.
        time.sleep(0.05)
        _ = tema.next(20.0)

        assert tema.get_value() == pytest.approx(value_before, abs=1e-12)
        assert len(tema) == len_before
        np.testing.assert_array_equal(tema.get_values(), values_before)

    def test_next_formula_is_consistent(self):
        """``next()`` result must match the hand-calculated alpha blend."""
        half_life = 1.0
        tema = TimeExponentialMovingAverage(window=2, half_life_s=half_life)
        tema.initialize(np.array([0.0, 0.0], dtype=np.float64))

        time.sleep(half_life)  # dt == half_life  =>  alpha == 0.5
        projected = tema.next(10.0)

        # alpha = 1 - 2^(-dt/hl) = 1 - 2^(-1) = 0.5
        expected = 0.5 * 10.0 + 0.5 * 0.0
        assert projected == pytest.approx(expected, abs=1e-3)


class TestTemaUpdate:
    """Layer 2 – ``update()`` must advance time and mutate state correctly."""

    def test_update_advances_time_s_and_value(self):
        """``update()`` changes ``_value`` and advances the internal timestamp."""
        half_life = 1.0
        tema = TimeExponentialMovingAverage(window=2, half_life_s=half_life)
        tema.initialize(np.array([0.0, 0.0], dtype=np.float64))

        time.sleep(half_life)
        result = tema.update(10.0)

        # alpha ~ 0.5 after one half-life
        assert result == pytest.approx(5.0, abs=1e-3)
        assert tema.get_value() == pytest.approx(5.0, abs=1e-3)

    def test_update_pushes_to_ringbuffer(self):
        """After ``update()`` the new value appears in history."""
        tema = TimeExponentialMovingAverage(window=4, half_life_s=1.0)
        tema.initialize(np.array([1.0, 1.0], dtype=np.float64))
        prev_len = len(tema)

        time.sleep(0.01)
        tema.update(2.0)

        assert len(tema) == prev_len + 1
        assert tema.get_values()[-1] == pytest.approx(tema.get_value(), abs=1e-9)

    def test_update_before_warm_sets_value(self):
        """First ``update()`` on a cold MA warms it with the input value."""
        tema = TimeExponentialMovingAverage(window=2, half_life_s=1.0)
        result = tema.update(99.0)

        assert result == pytest.approx(99.0)
        assert tema.get_value() == pytest.approx(99.0)

    def test_update_alpha_accuracy(self):
        """Verify the exact alpha for a known elapsed time."""
        half_life = 1.0
        tema = TimeExponentialMovingAverage(window=2, half_life_s=half_life)
        # Identical values keep the internal value at 10.0 regardless of alpha.
        tema.initialize(np.array([10.0, 10.0], dtype=np.float64))

        sleep_dt = 1.0  # 1.0 s with hl=1.0  =>  alpha = 1 - 2^(-1) = 0.5
        time.sleep(sleep_dt)
        result = tema.update(20.0)

        alpha = 1.0 - (2.0 ** (-sleep_dt / half_life))
        expected = alpha * 20.0 + (1.0 - alpha) * 10.0
        assert result == pytest.approx(expected, abs=1e-3)


class TestTemaFastVsNormalMode:
    """Layer 3 – fast mode must suppress history storage."""

    def test_normal_mode_keeps_history(self):
        """Normal mode permits ``get_values()`` and ``len()``."""
        tema = TimeExponentialMovingAverage(window=4, half_life_s=1.0, is_fast=False)
        tema.initialize(np.array([1.0, 2.0, 3.0], dtype=np.float64))

        history = tema.get_values()
        assert len(history) == 3
        assert len(tema) == 3

    def test_fast_mode_raises_on_history_access(self):
        """Fast mode must raise when querying historical values."""
        tema = TimeExponentialMovingAverage(window=4, half_life_s=1.0, is_fast=True)
        tema.initialize(np.array([1.0, 2.0, 3.0], dtype=np.float64))

        with pytest.raises(ValueError, match="fast mode"):
            tema.get_values()

        with pytest.raises(ValueError, match="fast mode"):
            len(tema)

    def test_fast_mode_still_computes_value(self):
        """Fast mode must produce the same current value as normal mode."""
        tema_fast = TimeExponentialMovingAverage(window=2, half_life_s=1.0, is_fast=True)
        tema_norm = TimeExponentialMovingAverage(window=2, half_life_s=1.0, is_fast=False)

        init_vals = np.array([0.0, 0.0], dtype=np.float64)
        tema_fast.initialize(init_vals.copy())
        tema_norm.initialize(init_vals.copy())

        time.sleep(0.5)
        fast_result = tema_fast.update(10.0)
        norm_result = tema_norm.update(10.0)

        assert fast_result == pytest.approx(norm_result, abs=1e-9)
        assert tema_fast.get_value() == pytest.approx(tema_norm.get_value(), abs=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

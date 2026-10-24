"""Layer 1–3 — Tests for TimeExponentialMovingAverage (TEMA).

TEMA uses elapsed monotonic time to compute the exponential decay weight
(alpha) of each new sample. Tests here cover constructor validation,
initialisation semantics, the read-only nature of ``next()``, state mutation
by ``update()``, and the fast-mode memory trade-off.
"""

from __future__ import annotations

import numpy as np
import pytest

from mm_toolbox.moving_average import TimeExponentialMovingAverage
from mm_toolbox.time import time_monotonic_ns
from .tema_clock import last_time_ns, next_at, set_time_ns, update_at


class TestTemaConstructorValidation:
    """Layer 1 — Primitive validation of the constructor."""

    def test_half_life_positive_ok(self):
        """Given a positive half-life, construction succeeds."""
        tema = TimeExponentialMovingAverage(window=2, half_life_s=1.0)
        assert isinstance(tema, TimeExponentialMovingAverage)

    def test_half_life_zero_raises(self):
        """Given ``half_life_s=0``, construction raises ``ValueError``."""
        with pytest.raises(ValueError, match="Half life must be positive"):
            TimeExponentialMovingAverage(window=2, half_life_s=0.0)

    def test_half_life_negative_raises(self):
        """Given ``half_life_s<0``, construction raises ``ValueError``."""
        with pytest.raises(ValueError, match="Half life must be positive"):
            TimeExponentialMovingAverage(window=2, half_life_s=-0.5)

    @pytest.mark.parametrize("half_life_s", [float("nan"), float("inf"), -float("inf")])
    def test_half_life_must_be_finite(self, half_life_s: float):
        """Non-finite decay configuration is rejected before the first update."""
        with pytest.raises(ValueError, match="finite"):
            TimeExponentialMovingAverage(window=2, half_life_s=half_life_s)


class TestTemaInitialize:
    """Layer 1 — Validation and warm-up behaviour of ``initialize()``."""

    def test_initialize_uses_one_clock_origin(self):
        """Untimed historical samples cannot acquire decay from CPU timing."""
        tema = TimeExponentialMovingAverage(window=4, half_life_s=1e-9)
        assert tema.initialize(np.array([10.0, 20.0, 30.0])) == 10.0
        np.testing.assert_array_equal(tema.get_values(), [10.0, 10.0, 10.0])

    def test_initialize_too_short_raises(self):
        """Given arrays of length 0 or 1, ``initialize()`` raises ``ValueError``."""
        tema = TimeExponentialMovingAverage(window=2, half_life_s=1.0)

        with pytest.raises(ValueError, match="too short"):
            tema.initialize(np.array([], dtype=np.float64))

        with pytest.raises(ValueError, match="too short"):
            tema.initialize(np.array([1.0], dtype=np.float64))

    def test_initialize_warms_correctly(self):
        """Given a valid array, the MA is warm and the ringbuffer is populated."""
        tema = TimeExponentialMovingAverage(window=4, half_life_s=1.0)
        values = np.array([10.0, 10.0], dtype=np.float64)

        result = tema.initialize(values)

        assert tema.get_value() == result
        assert len(tema) == 2

    def test_initialize_clears_previous_state(self):
        """Given a second ``initialize()`` call, prior state is fully discarded."""
        tema = TimeExponentialMovingAverage(window=4, half_life_s=1.0)
        tema.initialize(np.array([1.0, 2.0], dtype=np.float64))
        old_value = tema.get_value()

        tema.initialize(np.array([100.0, 100.0], dtype=np.float64))

        assert tema.get_value() != old_value
        assert tema.get_value() == pytest.approx(100.0, abs=1e-9)

    def test_initialize_resets_elapsed_time_origin(self):
        """Reinitialization starts a new elapsed-time baseline."""
        tema = TimeExponentialMovingAverage(window=2, half_life_s=1.0)
        set_time_ns(tema, 1)

        result = tema.initialize(np.array([100.0, 0.0], dtype=np.float64))

        assert last_time_ns(tema) > 1
        assert result == pytest.approx(100.0, abs=0.1)


class TestTemaNext:
    """Layer 2 — ``next()`` must be a pure projection without side effects."""

    def test_next_before_warm_returns_input(self):
        """Given a cold MA, ``next()`` returns the input and leaves visible state unchanged."""
        tema = TimeExponentialMovingAverage(window=2, half_life_s=1.0)
        result = tema.next(42.0)

        assert result == pytest.approx(42.0)
        assert tema.get_value() == pytest.approx(0.0)

    def test_next_does_not_mutate_state(self):
        """Given a warm MA, ``next()`` leaves all internal state intact.

        This is a critical regression test: if ``next()`` advanced the internal
        timestamp, subsequent ``update()`` calls would use a smaller dt and
        produce incorrect alpha values.
        """
        tema = TimeExponentialMovingAverage(window=2, half_life_s=1.0)
        tema.initialize(np.array([10.0, 10.0], dtype=np.float64))

        value_before = tema.get_value()
        len_before = len(tema)
        values_before = tema.get_values().copy()
        timestamp_before = time_monotonic_ns()
        set_time_ns(tema, timestamp_before)

        _ = tema.next(20.0)

        assert tema.get_value() == pytest.approx(value_before, abs=1e-12)
        assert len(tema) == len_before
        np.testing.assert_array_equal(tema.get_values(), values_before)
        assert last_time_ns(tema) == timestamp_before

    def test_next_formula_is_consistent(self):
        """A preview uses the exact subsecond interval supplied by the caller."""
        half_life = 1.0
        tema = TimeExponentialMovingAverage(window=2, half_life_s=half_life)
        tema.initialize(np.array([0.0, 0.0], dtype=np.float64))
        set_time_ns(tema, 1_000_000_000)

        projected = next_at(tema, 10.0, 1_500_000_000)
        alpha = -np.expm1(-np.log(2.0) * 0.5 / half_life)

        assert projected == pytest.approx(alpha * 10.0, abs=1e-12)
        assert last_time_ns(tema) == 1_000_000_000


class TestTemaUpdate:
    """Layer 2 — ``update()`` must advance time and mutate state correctly."""

    def test_update_advances_time_s_and_value(self):
        """The native update helper applies an exact half-second interval."""
        half_life = 1.0
        tema = TimeExponentialMovingAverage(window=2, half_life_s=half_life)
        tema.initialize(np.array([0.0, 0.0], dtype=np.float64))
        set_time_ns(tema, 1_000_000_000)

        result = update_at(tema, 10.0, 1_500_000_000)
        alpha = -np.expm1(-np.log(2.0) * 0.5 / half_life)

        assert result == pytest.approx(alpha * 10.0, abs=1e-12)
        assert tema.get_value() == pytest.approx(result)
        assert last_time_ns(tema) == 1_500_000_000

    def test_update_pushes_to_ringbuffer(self):
        """Given ``update()``, the new value appears in history."""
        tema = TimeExponentialMovingAverage(window=4, half_life_s=1.0)
        tema.initialize(np.array([1.0, 1.0], dtype=np.float64))
        prev_len = len(tema)

        tema.update(2.0)

        assert len(tema) == prev_len + 1
        assert tema.get_values()[-1] == pytest.approx(tema.get_value(), abs=1e-9)

    def test_update_before_warm_sets_value(self):
        """Given a cold MA, the first ``update()`` warms it with the input value."""
        tema = TimeExponentialMovingAverage(window=2, half_life_s=1.0)
        result = tema.update(99.0)

        assert result == pytest.approx(99.0)
        assert tema.get_value() == pytest.approx(99.0)

    def test_update_alpha_accuracy(self):
        """The update recurrence agrees with an exact monotonic interval."""
        half_life = 1.0
        tema = TimeExponentialMovingAverage(window=2, half_life_s=half_life)
        tema.initialize(np.array([10.0, 10.0], dtype=np.float64))
        set_time_ns(tema, 2_000_000_000)

        result = update_at(tema, 20.0, 2_500_000_000)
        alpha = -np.expm1(-np.log(2.0) * 0.5 / half_life)

        assert result == pytest.approx(10.0 + alpha * 10.0, abs=1e-12)
        assert last_time_ns(tema) == 2_500_000_000


class TestTemaFastVsNormalMode:
    """Layer 3 — Fast mode must suppress history storage."""

    def test_normal_mode_keeps_history(self):
        """Given normal mode, ``get_values()`` and ``len()`` succeed."""
        tema = TimeExponentialMovingAverage(window=4, half_life_s=1.0, is_fast=False)
        tema.initialize(np.array([1.0, 2.0, 3.0], dtype=np.float64))

        history = tema.get_values()
        assert len(history) == 3
        assert len(tema) == 3

    def test_fast_mode_raises_on_history_access(self):
        """Given fast mode, history queries raise ``ValueError``."""
        tema = TimeExponentialMovingAverage(window=4, half_life_s=1.0, is_fast=True)
        tema.initialize(np.array([1.0, 2.0, 3.0], dtype=np.float64))

        with pytest.raises(ValueError, match="fast mode"):
            tema.get_values()

        with pytest.raises(ValueError, match="fast mode"):
            len(tema)

    def test_fast_mode_still_computes_value(self):
        """Given identical inputs, fast and normal modes produce the same current value."""
        tema_fast = TimeExponentialMovingAverage(
            window=2, half_life_s=1.0, is_fast=True
        )
        tema_norm = TimeExponentialMovingAverage(
            window=2, half_life_s=1.0, is_fast=False
        )

        init_vals = np.array([0.0, 0.0], dtype=np.float64)
        tema_fast.initialize(init_vals.copy())
        tema_norm.initialize(init_vals.copy())

        set_time_ns(tema_fast, 1_000_000_000)
        set_time_ns(tema_norm, 1_000_000_000)
        fast_result = update_at(tema_fast, 10.0, 2_000_000_000)
        norm_result = update_at(tema_norm, 10.0, 2_000_000_000)

        assert fast_result == pytest.approx(norm_result, abs=1e-5)
        assert tema_fast.get_value() == pytest.approx(tema_norm.get_value(), abs=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Decimal-reference and adjacent-float checks for the power-of-ten grid."""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import math

import numpy as np
import pytest

from mm_toolbox.rounding import Rounder, RounderConfig


class TestPower10Grid:
    """Layer 2 — Scalar and vector rounding agree with an independent reference."""

    @pytest.mark.parametrize("exponent", [-8, -5, -2, -1, 0, 1, 6])
    def test_decimal_reference_and_adjacent_values(self, exponent):
        """Canonical grid points stay fixed and neighboring floats keep direction."""
        step = float(f"1e{exponent}")
        rounder = Rounder(RounderConfig.default(step, step))
        values = []
        for tick in [-1001, -29, -7, -1, 0, 1, 7, 29, 1001]:
            point = float(Decimal(tick).scaleb(exponent))
            values.extend(
                [
                    point,
                    math.nextafter(point, -math.inf),
                    math.nextafter(point, math.inf),
                ]
            )
        array = np.array(values, dtype=np.float64)
        expected_down, expected_up = [], []
        with localcontext() as context:
            context.prec = 400
            quantum = Decimal(1).scaleb(exponent)
            for value in values:
                decimal = Decimal(str(value))
                expected_down.append(
                    float(
                        (decimal / quantum).to_integral_value(rounding=ROUND_FLOOR)
                        * quantum
                    )
                )
                expected_up.append(
                    float(
                        (decimal / quantum).to_integral_value(rounding=ROUND_CEILING)
                        * quantum
                    )
                )
        assert [rounder.bid(value) for value in values] == expected_down
        assert [rounder.ask(value) for value in values] == expected_up
        assert rounder.bids(array).tolist() == expected_down
        assert rounder.asks(array).tolist() == expected_up
        assert rounder.sizes(array).tolist() == expected_up

    @pytest.mark.parametrize("exponent", [-308, -100, -15, 15, 100, 308])
    def test_supported_extreme_grid_points(self, exponent):
        """A canonical finite increment is stable even at extreme exponents."""
        step = float(f"1e{exponent}")
        rounder = Rounder(RounderConfig.default(step, step))
        assert rounder.bid(step) == step
        assert rounder.ask(step) == step

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 1e100])
    def test_unrepresentable_inputs_fail_explicitly(self, value):
        """Nonfinite or excessive tick counts cannot silently misround."""
        rounder = Rounder(RounderConfig.default(0.01, 0.01))
        with pytest.raises(ValueError):
            rounder.bid(value)
        with pytest.raises(ValueError):
            rounder.asks(np.array([value], dtype=np.float64))

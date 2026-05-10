"""Layer 1 — Primitives tests for bounds-based filter utilities.

Covers ``DataBoundsFilter`` construction validation, bound inclusivity,
out-of-range reset behaviour, and the explicit ``reset`` flag.
"""

from __future__ import annotations

import pytest

from mm_toolbox.misc.filter import DataBoundsFilter


class TestDataBoundsFilterInitialization:
    """Layer 1 — ``DataBoundsFilter`` initialisation and first-update behaviour."""

    def test_first_check_initializes_bounds(self) -> None:
        """Given the first value, bounds are initialised around it; the next value inside the band is rejected."""
        filt = DataBoundsFilter(10.0)

        assert filt.check_and_update(100.0) is True
        assert filt.check_and_update(105.0) is False
        assert filt.check_and_update(111.0) is True


class TestDataBoundsFilterBounds:
    """Layer 1 — Bound inclusivity and out-of-range update behaviour."""

    def test_bounds_are_inclusive(self) -> None:
        """Given values exactly at the ±threshold boundary, they are accepted without reset."""
        filt = DataBoundsFilter(10.0)
        filt.reset(100.0)

        assert filt.check_and_update(90.0) is False
        assert filt.check_and_update(110.0) is False

    def test_outside_bounds_triggers_update(self) -> None:
        """Given values just outside the ±threshold boundary, the filter resets and returns True."""
        filt = DataBoundsFilter(10.0)

        filt.reset(100.0)
        assert filt.check_and_update(89.99) is True

        filt.reset(100.0)
        assert filt.check_and_update(110.01) is True


class TestDataBoundsFilterResetFlag:
    """Layer 1 — Explicit reset-flag behaviour."""

    def test_reset_flag_forces_update(self) -> None:
        """Given ``reset=True``, the filter always re-bounds around the new value."""
        filt = DataBoundsFilter(5.0)
        filt.reset(100.0)

        assert filt.check_and_update(100.0, reset=True) is True
        assert filt.check_and_update(100.0) is False


class TestDataBoundsFilterThresholdValidation:
    """Layer 1 — Constructor threshold-validation tests."""

    @pytest.mark.parametrize("threshold_pct", [0.0, 100.0, -1.0, 150.0])
    def test_threshold_must_be_within_bounds(self, threshold_pct: float) -> None:
        """Given a threshold outside the open interval (0, 100), construction raises ``ValueError``."""
        with pytest.raises(ValueError):
            DataBoundsFilter(threshold_pct)

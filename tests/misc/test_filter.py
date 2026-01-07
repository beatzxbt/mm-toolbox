"""Tests for bounds-based filter utilities."""

from __future__ import annotations

import pytest

from mm_toolbox.misc.filter import DataBoundsFilter


class TestDataBoundsFilterInitialization:
    """Test initialization and first-update behavior."""

    def test_first_check_initializes_bounds(self) -> None:
        """First update should initialize bounds around the value."""
        filt = DataBoundsFilter(10.0)

        assert filt.check_and_update(100.0) is True
        assert filt.check_and_update(105.0) is False
        assert filt.check_and_update(111.0) is True


class TestDataBoundsFilterBounds:
    """Test bound inclusivity and out-of-range updates."""

    def test_bounds_are_inclusive(self) -> None:
        """Values at the bounds are accepted without reset."""
        filt = DataBoundsFilter(10.0)
        filt.reset(100.0)

        assert filt.check_and_update(90.0) is False
        assert filt.check_and_update(110.0) is False

    def test_outside_bounds_triggers_update(self) -> None:
        """Values outside the bounds reset the filter."""
        filt = DataBoundsFilter(10.0)

        filt.reset(100.0)
        assert filt.check_and_update(89.99) is True

        filt.reset(100.0)
        assert filt.check_and_update(110.01) is True


class TestDataBoundsFilterResetFlag:
    """Test explicit reset behavior."""

    def test_reset_flag_forces_update(self) -> None:
        """Reset flag should always refresh bounds."""
        filt = DataBoundsFilter(5.0)
        filt.reset(100.0)

        assert filt.check_and_update(100.0, reset=True) is True
        assert filt.check_and_update(100.0) is False


class TestDataBoundsFilterThresholdValidation:
    """Test threshold validation on construction."""

    @pytest.mark.parametrize("threshold_pct", [0.0, 100.0, -1.0, 150.0])
    def test_threshold_must_be_within_bounds(self, threshold_pct: float) -> None:
        """Threshold must be strictly within (0, 100)."""
        with pytest.raises(ValueError):
            DataBoundsFilter(threshold_pct)

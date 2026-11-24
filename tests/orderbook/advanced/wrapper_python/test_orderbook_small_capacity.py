"""Constructor capacity validation for the advanced Python wrapper.

Behavioral capacity and BBO cases live in the shared reference scenario suite.
"""

from __future__ import annotations

import pytest

from mm_toolbox.orderbook.advanced import (
    PyAdvancedOrderbook,
    PyOrderbookSortedness,
)
from tests.orderbook.advanced.conftest import (
    TICK_SIZE,
    LOT_SIZE,
)


@pytest.mark.boundary
class TestMinimumCapacityEnforcement:
    """Layer 2: Test that minimum orderbook size of 4 levels is enforced."""

    @pytest.mark.parametrize("invalid_size", [0, 1, 2, 3])
    def test_reject_sizes_below_minimum(self, invalid_size: int):
        """Given size < 4, When creating PyAdvancedOrderbook, Then raises ValueError."""
        with pytest.raises(ValueError, match="expected >=4"):
            PyAdvancedOrderbook(
                tick_size=TICK_SIZE,
                lot_size=LOT_SIZE,
                num_levels=invalid_size,
                delta_sortedness=PyOrderbookSortedness.UNKNOWN,
                snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
            )

    @pytest.mark.parametrize("valid_size", [4, 5, 16, 32, 64, 128, 1024])
    def test_accept_valid_sizes(self, valid_size: int):
        """Given size >= 4, When creating PyAdvancedOrderbook, Then succeeds."""
        book = PyAdvancedOrderbook(
            tick_size=TICK_SIZE,
            lot_size=LOT_SIZE,
            num_levels=valid_size,
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )
        assert book is not None

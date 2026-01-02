"""Type stubs for ladder.pyx - Orderbook ladder data structure."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

class OrderbookLadderView:
    """View into an OrderbookLadder's data (C struct exposed as class).

    This is a lightweight view that provides read access to the ladder's
    levels without copying data. The view is invalidated when the ladder
    is modified.

    Note: This struct is primarily used internally via cimport.
    """

    num_levels: int
    max_levels: int
    levels: Any  # OrderbookLevel* pointer
    is_price_ascending: bool

class OrderbookLadder:
    """Manages a single ladder (bids or asks) with fixed-size level storage.

    The ladder maintains levels in a contiguous, cache-aligned memory block.
    Bids are stored in descending price order; asks in ascending price order.

    Note: Most methods are cdef-only. Only cpdef methods are accessible from Python.
    """

    def __init__(self, max_levels: int, is_price_ascending: bool) -> None:
        """Initialize a new OrderbookLadder.

        Args:
            max_levels: Maximum number of levels to store
            is_price_ascending: True for asks (ascending), False for bids (descending)
        """
        ...

    def get_levels(self) -> npt.NDArray[np.void]:
        """Return a NumPy view of all levels as a structured array.

        The structured array has fields: price, size, norders, ticks, lots.
        """
        ...

    def get_prices(self) -> npt.NDArray[np.float64]:
        """Return a NumPy view of prices."""
        ...

    def get_sizes(self) -> npt.NDArray[np.float64]:
        """Return a NumPy view of sizes."""
        ...

    def get_norders(self) -> npt.NDArray[np.uint64]:
        """Return a NumPy view of norders."""
        ...

    # Note: The following methods are cdef-only and documented here for reference.
    # They must be accessed via cimport from Cython code.
    #
    # cdef OrderbookLadderView* get_view(self) noexcept nogil
    # cdef void insert_level(self, u64 index, OrderbookLevel level) noexcept nogil
    # cdef void roll_right(self, u64 start_index) noexcept nogil
    # cdef void roll_left(self, u64 start_index) noexcept nogil
    # cdef void reset(self) noexcept nogil
    # cdef void increment_count(self) noexcept nogil
    # cdef void decrement_count(self) noexcept nogil
    # cdef bint is_empty(self) noexcept nogil
    # cdef bint is_full(self) noexcept nogil

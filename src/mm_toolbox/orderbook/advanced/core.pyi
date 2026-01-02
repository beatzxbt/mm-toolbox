"""Type stubs for core.pyx - Core orderbook engine.

Note: CoreAdvancedOrderbook is an internal class primarily used by PyAdvancedOrderbook
and AdvancedOrderbook. Most methods are cdef-only and not callable from Python.
This stub documents the class interface for type checking purposes.
"""

from __future__ import annotations

from .enum.enums import CyOrderbookSortedness

class CoreAdvancedOrderbook:
    """Core orderbook engine managing bids and asks with efficient in-place updates.

    This is an internal class that provides the core orderbook functionality.
    It is wrapped by PyAdvancedOrderbook (for Python) and AdvancedOrderbook (for Cython).

    Note: Most methods are cdef-only and must be accessed via cimport from Cython.
    """

    # Internal state (for documentation purposes)
    _tick_size: float
    _lot_size: float
    _max_levels: int
    _delta_sortedness: CyOrderbookSortedness
    _snapshot_sortedness: CyOrderbookSortedness

    def __init__(
        self,
        tick_size: float,
        lot_size: float,
        num_levels: int,
        delta_sortedness: CyOrderbookSortedness,
        snapshot_sortedness: CyOrderbookSortedness,
    ) -> None:
        """Initialize the core orderbook engine.

        Args:
            tick_size: Minimum price increment (must be > 0)
            lot_size: Minimum size increment (must be > 0)
            num_levels: Maximum number of levels per side (must be > 0)
            delta_sortedness: Expected sort order for delta updates
            snapshot_sortedness: Expected sort order for snapshot updates
        Raises:
            ValueError: If tick_size, lot_size, or num_levels are invalid
        """
        ...

    # Note: The following methods are cdef-only and documented here for reference.
    # They must be accessed via cimport from Cython code.
    #
    # cdef void clear(self)
    # cdef void consume_snapshot(self, OrderbookLevels new_asks, OrderbookLevels new_bids)
    # cdef void consume_deltas(self, OrderbookLevels asks, OrderbookLevels bids)
    # cdef void consume_bbo(self, OrderbookLevel ask, OrderbookLevel bid)
    # cdef double get_mid_price(self)
    # cdef double get_bbo_spread(self)
    # cdef double get_wmid_price(self)
    # cdef double get_volume_weighted_mid_price(self, double size, bint is_base_currency)
    # cdef double get_price_impact(self, double size, bint is_buy, bint is_base_currency)
    # cdef bint is_bbo_crossed(self, double other_bid_price, double other_ask_price)
    # cdef bint does_bbo_price_change(self, double bid_price, double ask_price)
    # cdef OrderbookLadderView* get_bids_view(self)
    # cdef OrderbookLadderView* get_asks_view(self)

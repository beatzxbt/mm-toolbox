"""Type stubs for cython.pyx - Cython-facing advanced orderbook API.

Note: The AdvancedOrderbook class is designed for use from Cython code via cimport.
Most methods are cdef-only and not callable from Python. This stub documents the
class interface and the few Python-accessible static methods.
"""

from __future__ import annotations

from .enum.enums import CyOrderbookSortedness

class AdvancedOrderbook:
    """Cython-facing API for the advanced orderbook with efficient in-place updates.

    This class provides a Cython API wrapper around CoreAdvancedOrderbook, designed
    for use in Cython code that needs high-performance orderbook operations. Methods
    that modify the orderbook (consume_*) are designed for use in nogil contexts.
    Methods that query the orderbook may raise RuntimeError if the orderbook is empty.

    The orderbook maintains separate bid and ask sides, each with a fixed maximum
    number of levels. Levels are stored internally using integer arithmetic (ticks
    and lots) for precision and performance.

    Note: Most methods are cdef-only and must be accessed via cimport from Cython.
    """

    def __init__(
        self,
        tick_size: float,
        lot_size: float,
        num_levels: int,
        delta_sortedness: CyOrderbookSortedness = ...,
        snapshot_sortedness: CyOrderbookSortedness = ...,
    ) -> None:
        """Initialize a new AdvancedOrderbook instance.

        Args:
            tick_size: Minimum price increment (must be > 0)
            lot_size: Minimum size increment (must be > 0)
            num_levels: Maximum number of levels per side (must be > 0)
            delta_sortedness: Expected sort order for delta updates (default: UNKNOWN)
            snapshot_sortedness: Expected sort order for snapshot updates (default: UNKNOWN)
        """
        ...

    # Note: The following methods are cdef-only and documented here for reference.
    # They must be accessed via cimport from Cython code.
    #
    # cdef void clear(self)
    # cdef void consume_snapshot(self, OrderbookLevels asks, OrderbookLevels bids)
    # cdef void consume_deltas(self, OrderbookLevels asks, OrderbookLevels bids)
    # cdef void consume_bbo(self, OrderbookLevel ask, OrderbookLevel bid)
    # cdef double get_mid_price(self)
    # cdef double get_bbo_spread(self)
    # cdef double get_wmid_price(self)
    # cdef double get_volume_weighted_mid_price(self, double size, bint is_base_currency)
    # cdef double get_price_impact(self, double size, bint is_buy, bint is_base_currency)  # touch-anchored terminal impact
    # cdef double get_size_for_price_impact_bps(self, double impact_bps, bint is_buy, bint is_base_currency)  # touch-anchored depth band
    # cdef bint is_bbo_crossed(self, double other_bid_price, double other_ask_price)
    # cdef bint does_bbo_price_change(self, double bid_price, double ask_price)
    #
    # Static helper methods (cdef-only):
    # @staticmethod
    # cdef OrderbookLevel create_orderbook_level(double price, double size, u64 norders=1)
    # @staticmethod
    # cdef OrderbookLevel create_orderbook_level_with_ticks_and_lots(...)
    # @staticmethod
    # cdef OrderbookLevels create_orderbook_levels_from_array(OrderbookLevel* levels_ptr, u64 num_levels)
    # @staticmethod
    # cdef OrderbookLevels create_orderbook_levels_allocated(u64 num_levels)
    # @staticmethod
    # cdef void free_orderbook_levels(OrderbookLevels* levels)

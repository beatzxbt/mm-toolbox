from libc.stdint cimport uint64_t as u64
from libc.stdlib cimport free, malloc

import numpy as np
from numpy cimport ndarray

from .core cimport CoreAdvancedOrderbook
from .enum.enums cimport CyOrderbookSortedness
from .enum.enums import PyOrderbookSortedness  # Python import for Python class
from .level.level cimport (
    OrderbookLevel,
    OrderbookLevels,
    PyOrderbookLevel,
    PyOrderbookLevels,
    create_orderbook_level,
)
from .ladder.ladder cimport OrderbookLadderData


cdef class PyAdvancedOrderbook:
    """Python-facing API for the advanced orderbook with efficient in-place updates.
    
    This class provides a Python API wrapper around CoreAdvancedOrderbook, designed
    for use in Python code. It handles conversion between Python types (PyOrderbookLevel,
    PyOrderbookLevels) and the internal C structs.
    
    The orderbook maintains separate bid and ask sides, each with a fixed maximum
    number of levels. Levels are stored internally using integer arithmetic (ticks
    and lots) for precision and performance.
    """
    # Note: _core is declared in .pxd, don't redeclare here

    def __cinit__(
        self,
        double tick_size,
        double lot_size,
        u64 num_levels,
        object delta_sortedness=None,
        object snapshot_sortedness=None,
    ):
        """Initialize a new PyAdvancedOrderbook instance.
        
        Args:
            tick_size: Minimum price increment (must be > 0)
            lot_size: Minimum size increment (must be > 0)
            num_levels: Maximum number of levels per side (must be > 0)
            delta_sortedness: Expected sort order for delta updates (default: UNKNOWN)
            snapshot_sortedness: Expected sort order for snapshot updates (default: UNKNOWN)
        """
        if delta_sortedness is None:
            delta_sortedness = PyOrderbookSortedness.UNKNOWN
        if snapshot_sortedness is None:
            snapshot_sortedness = PyOrderbookSortedness.UNKNOWN
        cdef CyOrderbookSortedness delta_code = <CyOrderbookSortedness> <int> delta_sortedness
        cdef CyOrderbookSortedness snap_code = <CyOrderbookSortedness> <int> snapshot_sortedness

        self._core = CoreAdvancedOrderbook(
            tick_size,
            lot_size,
            num_levels,
            delta_code,
            snap_code,
        )

    cpdef void clear(self):
        """Clear all levels from both sides of the orderbook."""
        self._core.clear()

    cpdef void consume_snapshot(self, PyOrderbookLevels asks, PyOrderbookLevels bids):
        """Replace the entire orderbook state with new snapshot data.
        
        Args:
            asks: PyOrderbookLevels containing ask levels
            bids: PyOrderbookLevels containing bid levels
        """
        self._core.consume_snapshot(asks._levels, bids._levels)

    cpdef void consume_deltas(self, PyOrderbookLevels asks, PyOrderbookLevels bids):
        """Apply incremental updates to the orderbook.
        
        Args:
            asks: PyOrderbookLevels containing ask level updates
            bids: PyOrderbookLevels containing bid level updates
        """
        self._core.consume_deltas(asks._levels, bids._levels)

    cpdef void consume_bbo(self, PyOrderbookLevel ask, PyOrderbookLevel bid):
        """Update only the best bid and offer (top of book).
        
        Args:
            ask: PyOrderbookLevel for the best ask
            bid: PyOrderbookLevel for the best bid
        """
        self._core.consume_bbo(ask._level, bid._level)

    cpdef get_bbo(self):
        """Get the best bid and offer (top of book).
        
        Returns:
            Tuple of (best_bid, best_ask) as PyOrderbookLevel instances
        
        Raises:
            RuntimeError: If orderbook is empty
        """
        if self._core._bids.is_empty() or self._core._asks.is_empty():
            raise RuntimeError("Empty view on one/both sides of orderbook; cannot compute without data")
        cdef OrderbookLadderData* bids_data = self._core.get_bids_data()
        cdef OrderbookLadderData* asks_data = self._core.get_asks_data()
        cdef OrderbookLevel* bid0 = &bids_data.levels[0]
        cdef OrderbookLevel* ask0 = &asks_data.levels[0]
        return PyOrderbookLevel.from_struct(bid0[0]), PyOrderbookLevel.from_struct(ask0[0])

    cpdef get_bids(self):
        """Get all bid levels."""
        cdef OrderbookLadderData* v = self._core.get_bids_data()
        cdef u64 n = v.num_levels
        return PyOrderbookLevels.from_ptr(v.levels, n)

    cpdef get_asks(self):
        """Get all ask levels."""
        cdef OrderbookLadderData* v = self._core.get_asks_data()
        cdef u64 n = v.num_levels
        return PyOrderbookLevels.from_ptr(v.levels, n)

    cpdef get_bids_numpy(self):
        """Get bid levels as a NumPy array."""
        return (<object> self._core._bids).get_levels()

    cpdef get_asks_numpy(self):
        """Get ask levels as a NumPy array."""
        return (<object> self._core._asks).get_levels()

    cpdef double get_mid_price(self):
        """Calculate the mid price (average of best bid and ask).
        
        Returns:
            Mid price, or infinity if orderbook is empty
        
        Raises:
            RuntimeError: If orderbook is empty
        """
        return self._core.get_mid_price()

    cpdef double get_bbo_spread(self):
        """Calculate the spread between best bid and ask.
        
        Returns:
            Spread in price units, or infinity if orderbook is empty
        
        Raises:
            RuntimeError: If orderbook is empty
        """
        return self._core.get_bbo_spread()

    cpdef double get_wmid_price(self):
        """Calculate the weighted mid price (volume-weighted average of best bid and ask).
        
        Returns:
            Weighted mid price, or infinity if orderbook is empty
        
        Raises:
            RuntimeError: If orderbook is empty
        """
        return self._core.get_wmid_price()

    cpdef double get_volume_weighted_mid_price(self, double size, bint is_base_currency=True):
        """Calculate volume-weighted mid price for a given trade size.
        
        Args:
            size: Trade size to calculate weighted price for
            is_base_currency: If True, size is in base currency; if False, in quote currency
        
        Returns:
            Volume-weighted mid price, or infinity if orderbook is empty or size cannot be filled
        
        Raises:
            RuntimeError: If orderbook is empty
        """
        return self._core.get_volume_weighted_mid_price(size, is_base_currency)

    cpdef double get_price_impact(self, double size, bint is_buy, bint is_base_currency=True):
        """Calculate price impact of executing a trade of given size.
        
        Args:
            size: Trade size
            is_buy: If True, calculate impact for buying; if False, for selling
            is_base_currency: If True, size is in base currency; if False, in quote currency
        
        Returns:
            Price impact (absolute difference from mid price), or infinity if size cannot be filled
        
        Raises:
            RuntimeError: If orderbook is empty
        """
        return self._core.get_price_impact(size, is_buy, is_base_currency)

    cpdef bint is_bbo_crossed(self, double bid_price, double ask_price):
        """Check if this orderbook's BBO crosses with another orderbook's BBO.
        
        Args:
            bid_price: Best bid price from another orderbook
            ask_price: Best ask price from another orderbook
        
        Returns:
            True if the BBOs cross (this bid >= other ask or this ask <= other bid)
        
        Raises:
            RuntimeError: If orderbook is empty
        """
        return self._core.is_bbo_crossed(bid_price, ask_price)

    cpdef bint does_bbo_price_change(self, double bid_price, double ask_price):
        """Check if the given prices differ from current BBO.
        
        Args:
            bid_price: Bid price to compare
            ask_price: Ask price to compare
        
        Returns:
            True if either price differs from current BBO
        
        Raises:
            RuntimeError: If orderbook is empty
        """
        return self._core.does_bbo_price_change(bid_price, ask_price)

    cpdef void consume_snapshot_numpy(
        self,
        double[:] ask_prices,
        double[:] ask_sizes,
        double[:] bid_prices,
        double[:] bid_sizes,
        u64[:] ask_norders=None,
        u64[:] bid_norders=None,
    ):
        """Replace the entire orderbook state with new snapshot data from numpy arrays.
        
        This method provides a more efficient path for users who already have data
        in numpy arrays, avoiding the overhead of constructing PyOrderbookLevels.
        
        Args:
            ask_prices: 1D array of ask prices
            ask_sizes: 1D array of ask sizes (must match length of ask_prices)
            bid_prices: 1D array of bid prices
            bid_sizes: 1D array of bid sizes (must match length of bid_prices)
            ask_norders: Optional 1D array of ask norders (defaults to 1 for each level)
            bid_norders: Optional 1D array of bid norders (defaults to 1 for each level)
        """
        cdef:
            Py_ssize_t num_asks = ask_prices.shape[0]
            Py_ssize_t num_bids = bid_prices.shape[0]
            bint use_default_ask_norders = ask_norders is None
            bint use_default_bid_norders = bid_norders is None
            OrderbookLevel* ask_levels = NULL
            OrderbookLevel* bid_levels = NULL
            OrderbookLevels asks_struct
            OrderbookLevels bids_struct
            Py_ssize_t i

        if ask_sizes.shape[0] != num_asks:
            raise ValueError(
                f"Mismatched ask lengths; expected {num_asks} sizes but got {ask_sizes.shape[0]}"
            )
        if bid_sizes.shape[0] != num_bids:
            raise ValueError(
                f"Mismatched bid lengths; expected {num_bids} sizes but got {bid_sizes.shape[0]}"
            )
        if not use_default_ask_norders and ask_norders.shape[0] != num_asks:
            raise ValueError(
                f"Mismatched ask norders; expected {num_asks} but got {ask_norders.shape[0]}"
            )
        if not use_default_bid_norders and bid_norders.shape[0] != num_bids:
            raise ValueError(
                f"Mismatched bid norders; expected {num_bids} but got {bid_norders.shape[0]}"
            )

        # Allocate memory for ask levels
        if num_asks > 0:
            ask_levels = <OrderbookLevel*> malloc(num_asks * sizeof(OrderbookLevel))
            if ask_levels == NULL:
                raise MemoryError("Failed to allocate memory for ask levels")
            for i in range(num_asks):
                ask_levels[i] = create_orderbook_level(
                    ask_prices[i],
                    ask_sizes[i],
                    1 if use_default_ask_norders else ask_norders[i],
                )

        # Allocate memory for bid levels
        if num_bids > 0:
            bid_levels = <OrderbookLevel*> malloc(num_bids * sizeof(OrderbookLevel))
            if bid_levels == NULL:
                if ask_levels != NULL:
                    free(ask_levels)
                raise MemoryError("Failed to allocate memory for bid levels")
            for i in range(num_bids):
                bid_levels[i] = create_orderbook_level(
                    bid_prices[i],
                    bid_sizes[i],
                    1 if use_default_bid_norders else bid_norders[i],
                )

        # Create OrderbookLevels structs
        asks_struct.num_levels = <u64> num_asks
        asks_struct.levels = ask_levels
        bids_struct.num_levels = <u64> num_bids
        bids_struct.levels = bid_levels

        try:
            self._core.consume_snapshot(asks_struct, bids_struct)
        finally:
            if ask_levels != NULL:
                free(ask_levels)
            if bid_levels != NULL:
                free(bid_levels)

    cpdef void consume_deltas_numpy(
        self,
        double[:] ask_prices,
        double[:] ask_sizes,
        double[:] bid_prices,
        double[:] bid_sizes,
        u64[:] ask_norders=None,
        u64[:] bid_norders=None,
    ):
        """Apply incremental updates to the orderbook from numpy arrays.
        
        This method provides a more efficient path for users who already have data
        in numpy arrays, avoiding the overhead of constructing PyOrderbookLevels.
        
        Args:
            ask_prices: 1D array of ask prices
            ask_sizes: 1D array of ask sizes (must match length of ask_prices)
            bid_prices: 1D array of bid prices
            bid_sizes: 1D array of bid sizes (must match length of bid_prices)
            ask_norders: Optional 1D array of ask norders (defaults to 1 for each level)
            bid_norders: Optional 1D array of bid norders (defaults to 1 for each level)
        """
        cdef:
            Py_ssize_t num_asks = ask_prices.shape[0]
            Py_ssize_t num_bids = bid_prices.shape[0]
            bint use_default_ask_norders = ask_norders is None
            bint use_default_bid_norders = bid_norders is None
            OrderbookLevel* ask_levels = NULL
            OrderbookLevel* bid_levels = NULL
            OrderbookLevels asks_struct
            OrderbookLevels bids_struct
            Py_ssize_t i

        if ask_sizes.shape[0] != num_asks:
            raise ValueError(
                f"Mismatched ask lengths; expected {num_asks} sizes but got {ask_sizes.shape[0]}"
            )
        if bid_sizes.shape[0] != num_bids:
            raise ValueError(
                f"Mismatched bid lengths; expected {num_bids} sizes but got {bid_sizes.shape[0]}"
            )
        if not use_default_ask_norders and ask_norders.shape[0] != num_asks:
            raise ValueError(
                f"Mismatched ask norders; expected {num_asks} but got {ask_norders.shape[0]}"
            )
        if not use_default_bid_norders and bid_norders.shape[0] != num_bids:
            raise ValueError(
                f"Mismatched bid norders; expected {num_bids} but got {bid_norders.shape[0]}"
            )

        # Allocate memory for ask levels
        if num_asks > 0:
            ask_levels = <OrderbookLevel*> malloc(num_asks * sizeof(OrderbookLevel))
            if ask_levels == NULL:
                raise MemoryError("Failed to allocate memory for ask levels")
            for i in range(num_asks):
                ask_levels[i] = create_orderbook_level(
                    ask_prices[i],
                    ask_sizes[i],
                    1 if use_default_ask_norders else ask_norders[i],
                )

        # Allocate memory for bid levels
        if num_bids > 0:
            bid_levels = <OrderbookLevel*> malloc(num_bids * sizeof(OrderbookLevel))
            if bid_levels == NULL:
                if ask_levels != NULL:
                    free(ask_levels)
                raise MemoryError("Failed to allocate memory for bid levels")
            for i in range(num_bids):
                bid_levels[i] = create_orderbook_level(
                    bid_prices[i],
                    bid_sizes[i],
                    1 if use_default_bid_norders else bid_norders[i],
                )

        # Create OrderbookLevels structs
        asks_struct.num_levels = <u64> num_asks
        asks_struct.levels = ask_levels
        bids_struct.num_levels = <u64> num_bids
        bids_struct.levels = bid_levels

        try:
            self._core.consume_deltas(asks_struct, bids_struct)
        finally:
            if ask_levels != NULL:
                free(ask_levels)
            if bid_levels != NULL:
                free(bid_levels)

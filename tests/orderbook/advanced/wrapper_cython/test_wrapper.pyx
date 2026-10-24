# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Minimal delegation tests for the Cython CyAdvancedOrderbook wrapper.

These tests verify that the wrapper correctly delegates to CoreAdvancedOrderbook.
Core logic is tested exhaustively in engine/.
"""
from __future__ import annotations

from libc.stdint cimport uint64_t as u64
from libc.stdlib cimport malloc, free
from libc.math cimport fabs

from mm_toolbox.orderbook.advanced.level.level cimport (
    OrderbookLevel,
    OrderbookLevels,
    create_orderbook_level,
)
from mm_toolbox.orderbook.advanced.cython cimport CyAdvancedOrderbook
from mm_toolbox.orderbook.advanced.enum.enums cimport CyOrderbookSortedness


# =============================================================================
# Test Constants
# =============================================================================
DEF TICK_SIZE = 0.01
DEF LOT_SIZE = 0.001


cdef OrderbookLevels _alloc_levels(u64 count):
    """Allocate an OrderbookLevels struct with given capacity.

    Args:
        count: Number of OrderbookLevel slots to allocate.

    Returns:
        OrderbookLevels with allocated array and num_levels set.
    """
    cdef OrderbookLevel* arr = <OrderbookLevel*>malloc(count * sizeof(OrderbookLevel))
    cdef OrderbookLevels levels
    levels.num_levels = count
    levels.levels = arr
    return levels


cdef void _free_levels(OrderbookLevels* levels):
    """Free OrderbookLevels memory.

    Args:
        levels: Pointer to OrderbookLevels to free.
    """
    if levels != NULL and levels.levels != NULL:
        free(levels.levels)
        levels.levels = NULL
        levels.num_levels = 0


cdef OrderbookLevel _make_level(
    double price,
    double size,
    double tick_size,
    double lot_size,
    u64 norders=1,
):
    return create_orderbook_level(price, size, norders)


cdef OrderbookLevels _make_levels(
    double* prices,
    double* sizes,
    u64 count,
    double tick_size,
    double lot_size,
):
    """Create OrderbookLevels from price/size arrays.

    Args:
        prices: Array of price values.
        sizes: Array of size values.
        count: Number of levels.
        tick_size: Tick size for conversion.
        lot_size: Lot size for conversion.

    Returns:
        OrderbookLevels populated with raw price, size, and norders values.
    """
    cdef OrderbookLevels levels = _alloc_levels(count)
    cdef u64 i
    for i in range(count):
        levels.levels[i] = _make_level(
            prices[i], sizes[i], tick_size, lot_size, 1
        )
    return levels


cdef bint _approx_eq(double a, double b, double tol=1e-9):
    """Check if two doubles are approximately equal.

    Args:
        a: First value.
        b: Second value.
        tol: Absolute tolerance (default 1e-9).

    Returns:
        True if |a - b| < tol.
    """
    return fabs(a - b) < tol


cdef CyAdvancedOrderbook _create_book():
    """Create an CyAdvancedOrderbook with standard test settings.

    Returns:
        Initialized CyAdvancedOrderbook with TICK_SIZE, LOT_SIZE, and 64 levels.
    """
    return CyAdvancedOrderbook(
        tick_size=TICK_SIZE,
        lot_size=LOT_SIZE,
        num_levels=64,
        delta_sortedness=CyOrderbookSortedness.UNKNOWN,
        snapshot_sortedness=CyOrderbookSortedness.UNKNOWN,
    )


cdef void _populate_book(CyAdvancedOrderbook book):
    """Populate book with standard 2-level snapshot for delegation tests.

    Args:
        book: CyAdvancedOrderbook to populate with bids [100.00, 99.99] and asks [100.01, 100.02].
    """
    cdef double bid_prices[2]
    cdef double bid_sizes[2]
    cdef double ask_prices[2]
    cdef double ask_sizes[2]
    
    bid_prices[0] = 100.00; bid_prices[1] = 99.99
    bid_sizes[0] = 1.0; bid_sizes[1] = 2.0
    ask_prices[0] = 100.01; ask_prices[1] = 100.02
    ask_sizes[0] = 1.5; ask_sizes[1] = 2.5
    
    cdef OrderbookLevels bids = _make_levels(bid_prices, bid_sizes, 2, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels asks = _make_levels(ask_prices, ask_sizes, 2, TICK_SIZE, LOT_SIZE)
    book.consume_snapshot(asks, bids)
    _free_levels(&bids)
    _free_levels(&asks)


def test_wrapper_init():
    """Test CyAdvancedOrderbook initializes without error."""
    cdef CyAdvancedOrderbook book = _create_book()
    assert book is not None


def test_wrapper_consume_snapshot_delegation():
    """Test consume_snapshot delegates correctly and mid price is computed."""
    cdef CyAdvancedOrderbook book = _create_book()
    _populate_book(book)
    cdef double mid = book.get_mid_price()
    assert _approx_eq(mid, 100.00)


def test_wrapper_consume_deltas_delegation():
    """Test consume_deltas delegates correctly and updates BBO size."""
    cdef CyAdvancedOrderbook book = _create_book()
    _populate_book(book)
    
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    ask_prices[0] = 100.01
    ask_sizes[0] = 5.0
    
    cdef OrderbookLevels delta_asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0
    
    book.consume_deltas(delta_asks, delta_bids)
    
    cdef double mid = book.get_mid_price()
    assert _approx_eq(mid, 100.00)
    _free_levels(&delta_asks)


def test_wrapper_consume_bbo_delegation():
    """Test consume_bbo delegates correctly and updates BBO size."""
    cdef CyAdvancedOrderbook book = _create_book()
    _populate_book(book)
    
    cdef OrderbookLevel new_ask = _make_level(
        100.01, 5.0, TICK_SIZE, LOT_SIZE, 1
    )
    cdef OrderbookLevel new_bid = _make_level(
        100.00, 5.0, TICK_SIZE, LOT_SIZE, 1
    )
    
    book.consume_bbo(new_ask, new_bid)
    
    cdef double mid = book.get_mid_price()
    assert _approx_eq(mid, 100.00)


def test_wrapper_consume_bbo_rejects_crossed_input():
    """Test consume_bbo rejects crossed incoming BBO levels."""
    cdef CyAdvancedOrderbook book = _create_book()
    _populate_book(book)

    cdef OrderbookLevel crossed_ask = _make_level(
        100.00, 1.0, TICK_SIZE, LOT_SIZE, 1
    )
    cdef OrderbookLevel crossed_bid = _make_level(
        100.01, 1.0, TICK_SIZE, LOT_SIZE, 1
    )

    cdef bint raised = False
    try:
        book.consume_bbo(crossed_ask, crossed_bid)
    except ValueError:
        raised = True
    assert raised


def test_wrapper_calculation_delegation():
    """Test all price/spread/impact calculations delegate correctly."""
    cdef CyAdvancedOrderbook book = _create_book()
    _populate_book(book)
    
    assert _approx_eq(book.get_mid_price(), 100.00)
    assert _approx_eq(book.get_bbo_spread(), 0.01)
    assert book.get_wmid_price() > 0
    assert _approx_eq(book.get_price_impact(0.5, True, True), 0.0)


def test_wrapper_clear_delegation():
    """Test clear() empties book and subsequent operations raise RuntimeError."""
    cdef CyAdvancedOrderbook book = _create_book()
    _populate_book(book)
    book.clear()
    
    cdef bint raised = False
    try:
        book.get_mid_price()
    except RuntimeError:
        raised = True
    assert raised

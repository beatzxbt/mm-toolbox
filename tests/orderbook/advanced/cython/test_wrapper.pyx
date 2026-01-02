# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Layer 4: AdvancedOrderbook wrapper tests.

Tests Python wrapper API, snapshot/delta/BBO pass-through,
static helper methods, and integration tests.
"""
from __future__ import annotations

from libc.stdint cimport uint64_t as u64
from libc.stdlib cimport malloc, free
from libc.math cimport fabs

from mm_toolbox.orderbook.advanced.level.level cimport (
    OrderbookLevel,
    OrderbookLevels,
    create_orderbook_level,
    create_orderbook_level_with_ticks_and_lots,
)
from mm_toolbox.orderbook.advanced.cython cimport AdvancedOrderbook
from mm_toolbox.orderbook.advanced.enum.enums cimport CyOrderbookSortedness


# =============================================================================
# Test Constants
# =============================================================================
DEF TICK_SIZE = 0.01
DEF LOT_SIZE = 0.001
DEF DEFAULT_LEVELS = 64


# =============================================================================
# Helper Functions
# =============================================================================
cdef OrderbookLevels _alloc_levels(u64 count):
    """Allocate an OrderbookLevels struct with given capacity."""
    cdef OrderbookLevel* arr = <OrderbookLevel*>malloc(count * sizeof(OrderbookLevel))
    cdef OrderbookLevels levels
    levels.num_levels = count
    levels.levels = arr
    return levels


cdef void _free_levels(OrderbookLevels* levels):
    """Free OrderbookLevels memory."""
    if levels != NULL and levels.levels != NULL:
        free(levels.levels)
        levels.levels = NULL
        levels.num_levels = 0


cdef OrderbookLevels _make_levels(
    double* prices,
    double* sizes,
    u64 count,
    double tick_size,
    double lot_size,
):
    """Create OrderbookLevels from price/size arrays."""
    cdef OrderbookLevels levels = _alloc_levels(count)
    cdef u64 i
    for i in range(count):
        levels.levels[i] = create_orderbook_level_with_ticks_and_lots(
            prices[i], sizes[i], tick_size, lot_size, 1
        )
    return levels


cdef bint _approx_eq(double a, double b, double tol=1e-9):
    """Check if two doubles are approximately equal."""
    return fabs(a - b) < tol


# LAYER 4: AdvancedOrderbook Wrapper
# =============================================================================

def test_wrapper_init():
    """Test AdvancedOrderbook initialization."""
    cdef AdvancedOrderbook book = AdvancedOrderbook(
        tick_size=TICK_SIZE,
        lot_size=LOT_SIZE,
        num_levels=64,
        delta_sortedness=CyOrderbookSortedness.UNKNOWN,
        snapshot_sortedness=CyOrderbookSortedness.UNKNOWN,
    )
    # Should not raise


def test_wrapper_invalid_tick_raises():
    """Test wrapper with invalid tick raises."""
    cdef bint raised = False
    try:
        _dummy = AdvancedOrderbook(
            tick_size=-0.01,
            lot_size=LOT_SIZE,
            num_levels=64,
        )
    except ValueError:
        raised = True
    assert raised, "Expected ValueError"


def test_wrapper_consume_snapshot():
    """Test wrapper consume_snapshot pass-through."""
    cdef AdvancedOrderbook book = AdvancedOrderbook(
        tick_size=TICK_SIZE,
        lot_size=LOT_SIZE,
        num_levels=64,
        delta_sortedness=CyOrderbookSortedness.UNKNOWN,
        snapshot_sortedness=CyOrderbookSortedness.UNKNOWN,
    )
    
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
    
    cdef double mid = book.get_mid_price()
    assert _approx_eq(mid, 100.00)  # Integer tick arithmetic: (10000 + 10001) // 2 * 0.01
    
    _free_levels(&bids)
    _free_levels(&asks)


def test_wrapper_consume_deltas():
    """Test wrapper consume_deltas pass-through."""
    cdef AdvancedOrderbook book = AdvancedOrderbook(
        tick_size=TICK_SIZE,
        lot_size=LOT_SIZE,
        num_levels=64,
    )
    
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
    
    # Apply delta
    cdef double delta_ask_prices[1]
    cdef double delta_ask_sizes[1]
    delta_ask_prices[0] = 100.01
    delta_ask_sizes[0] = 5.0
    
    cdef OrderbookLevels delta_asks = _make_levels(delta_ask_prices, delta_ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0
    
    book.consume_deltas(delta_asks, delta_bids)
    
    _free_levels(&delta_asks)


def test_wrapper_consume_bbo():
    """Test wrapper consume_bbo pass-through."""
    cdef AdvancedOrderbook book = AdvancedOrderbook(
        tick_size=TICK_SIZE,
        lot_size=LOT_SIZE,
        num_levels=64,
    )
    
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
    
    cdef OrderbookLevel new_ask = create_orderbook_level_with_ticks_and_lots(
        100.01, 5.0, TICK_SIZE, LOT_SIZE, 1
    )
    cdef OrderbookLevel new_bid = create_orderbook_level_with_ticks_and_lots(
        100.00, 5.0, TICK_SIZE, LOT_SIZE, 1
    )
    
    book.consume_bbo(new_ask, new_bid)


def test_wrapper_clear():
    """Test wrapper clear pass-through."""
    cdef AdvancedOrderbook book = AdvancedOrderbook(
        tick_size=TICK_SIZE,
        lot_size=LOT_SIZE,
        num_levels=64,
    )
    
    cdef double bid_prices[1]
    cdef double bid_sizes[1]
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    
    bid_prices[0] = 100.00
    bid_sizes[0] = 1.0
    ask_prices[0] = 100.01
    ask_sizes[0] = 1.0
    
    cdef OrderbookLevels bids = _make_levels(bid_prices, bid_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    
    book.consume_snapshot(asks, bids)
    _free_levels(&bids)
    _free_levels(&asks)
    
    book.clear()
    
    try:
        book.get_mid_price()
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_wrapper_price_calculations():
    """Test wrapper price calculation pass-throughs."""
    cdef AdvancedOrderbook book = AdvancedOrderbook(
        tick_size=TICK_SIZE,
        lot_size=LOT_SIZE,
        num_levels=64,
    )
    
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
    
    assert _approx_eq(book.get_mid_price(), 100.00)  # Integer tick arithmetic
    assert _approx_eq(book.get_bbo_spread(), 0.01)
    assert book.get_wmid_price() > 0
    assert book.get_volume_weighted_mid_price(0.5, True) > 0
    assert book.get_price_impact(0.5, True, True) >= 0


def test_wrapper_crossing_and_change():
    """Test wrapper crossing and change detection pass-throughs."""
    cdef AdvancedOrderbook book = AdvancedOrderbook(
        tick_size=TICK_SIZE,
        lot_size=LOT_SIZE,
        num_levels=64,
    )
    
    cdef double bid_prices[1]
    cdef double bid_sizes[1]
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    
    bid_prices[0] = 100.00
    bid_sizes[0] = 1.0
    ask_prices[0] = 100.01
    ask_sizes[0] = 1.0
    
    cdef OrderbookLevels bids = _make_levels(bid_prices, bid_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    
    book.consume_snapshot(asks, bids)
    _free_levels(&bids)
    _free_levels(&asks)
    
    assert book.is_bbo_crossed(99.99, 100.02) == False
    assert book.does_bbo_price_change(100.00, 100.01) == False
    assert book.does_bbo_price_change(99.99, 100.01) == True



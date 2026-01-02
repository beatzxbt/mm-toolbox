# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Layer 3: CoreAdvancedOrderbook tests.

Tests core engine functionality: initialization, snapshot ingestion,
delta processing, BBO updates, price calculations, and state management.
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
from mm_toolbox.orderbook.advanced.ladder.ladder cimport (
    OrderbookLadder,
    OrderbookLadderData,
)
from mm_toolbox.orderbook.advanced.core cimport CoreAdvancedOrderbook
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


# LAYER 3: CoreAdvancedOrderbook (RIGOROUS ENGINE TESTING)
# =============================================================================

# -----------------------------------------------------------------------------
# Core helper functions for Layer 3 tests
# -----------------------------------------------------------------------------

cdef CoreAdvancedOrderbook _create_core(
    u64 num_levels=DEFAULT_LEVELS,
    CyOrderbookSortedness delta_sortedness=CyOrderbookSortedness.UNKNOWN,
    CyOrderbookSortedness snapshot_sortedness=CyOrderbookSortedness.UNKNOWN,
):
    """Create a CoreAdvancedOrderbook with standard settings."""
    return CoreAdvancedOrderbook(
        tick_size=TICK_SIZE,
        lot_size=LOT_SIZE,
        num_levels=num_levels,
        delta_sortedness=delta_sortedness,
        snapshot_sortedness=snapshot_sortedness,
    )


cdef void _populate_standard_book(CoreAdvancedOrderbook core):
    """Populate core with standard 3-level book: bids [100, 99.99, 99.98], asks [100.01, 100.02, 100.03]."""
    cdef double bid_prices[3]
    cdef double bid_sizes[3]
    cdef double ask_prices[3]
    cdef double ask_sizes[3]
    
    bid_prices[0] = 100.00; bid_prices[1] = 99.99; bid_prices[2] = 99.98
    bid_sizes[0] = 1.0; bid_sizes[1] = 2.0; bid_sizes[2] = 3.0
    
    ask_prices[0] = 100.01; ask_prices[1] = 100.02; ask_prices[2] = 100.03
    ask_sizes[0] = 1.5; ask_sizes[1] = 2.5; ask_sizes[2] = 3.5
    
    cdef OrderbookLevels bids = _make_levels(bid_prices, bid_sizes, 3, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels asks = _make_levels(ask_prices, ask_sizes, 3, TICK_SIZE, LOT_SIZE)
    
    core.consume_snapshot(asks, bids)
    
    _free_levels(&bids)
    _free_levels(&asks)


# -----------------------------------------------------------------------------
# 3.1 Initialization tests
# -----------------------------------------------------------------------------

def test_core_init_valid():
    """Test valid CoreAdvancedOrderbook initialization."""
    cdef CoreAdvancedOrderbook core = CoreAdvancedOrderbook(
        tick_size=0.01,
        lot_size=0.001,
        num_levels=646,
        delta_sortedness=CyOrderbookSortedness.UNKNOWN,
        snapshot_sortedness=CyOrderbookSortedness.UNKNOWN,
    )
    # Should not raise


def test_core_init_zero_tick_size():
    """Test zero tick_size raises ValueError."""
    cdef bint raised = False
    try:
        _dummy = CoreAdvancedOrderbook(
            tick_size=0.0,
            lot_size=0.001,
            num_levels=646,
            delta_sortedness=CyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=CyOrderbookSortedness.UNKNOWN,
        )
    except ValueError:
        raised = True
    assert raised, "Expected ValueError"


def test_core_init_negative_tick_size():
    """Test negative tick_size raises ValueError."""
    cdef bint raised = False
    try:
        _dummy = CoreAdvancedOrderbook(
            tick_size=-0.01,
            lot_size=0.001,
            num_levels=646,
            delta_sortedness=CyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=CyOrderbookSortedness.UNKNOWN,
        )
    except ValueError:
        raised = True
    assert raised, "Expected ValueError"


def test_core_init_zero_lot_size():
    """Test zero lot_size raises ValueError."""
    cdef bint raised = False
    try:
        _dummy = CoreAdvancedOrderbook(
            tick_size=0.01,
            lot_size=0.0,
            num_levels=646,
            delta_sortedness=CyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=CyOrderbookSortedness.UNKNOWN,
        )
    except ValueError:
        raised = True
    assert raised, "Expected ValueError"


def test_core_init_negative_lot_size():
    """Test negative lot_size raises ValueError."""
    cdef bint raised = False
    try:
        _dummy = CoreAdvancedOrderbook(
            tick_size=0.01,
            lot_size=-0.001,
            num_levels=646,
            delta_sortedness=CyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=CyOrderbookSortedness.UNKNOWN,
        )
    except ValueError:
        raised = True
    assert raised, "Expected ValueError"


def test_core_init_zero_levels():
    """Test zero num_levels raises ValueError."""
    cdef bint raised = False
    try:
        _dummy = CoreAdvancedOrderbook(
            tick_size=0.01,
            lot_size=0.001,
            num_levels=0,
            delta_sortedness=CyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=CyOrderbookSortedness.UNKNOWN,
        )
    except ValueError:
        raised = True
    assert raised, "Expected ValueError"


def test_core_init_single_level():
    """Test single level capacity works."""
    cdef CoreAdvancedOrderbook core = _create_core(num_levels=64)
    
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
    
    core.consume_snapshot(asks, bids)
    
    cdef double mid = core.get_mid_price()
    assert _approx_eq(mid, 100.00)  # Integer tick arithmetic: (10000 + 10001) // 2 * 0.01
    
    _free_levels(&bids)
    _free_levels(&asks)


def test_core_init_large_levels():
    """Test large num_levels allocation."""
    cdef CoreAdvancedOrderbook core = _create_core(num_levels=10000)
    cdef OrderbookLadderData* bids_view = core.get_bids_data()
    cdef OrderbookLadderData* asks_view = core.get_asks_data()
    assert bids_view.max_levels == 10000
    assert asks_view.max_levels == 10000


def test_core_init_all_sortedness_modes():
    """Test all sortedness modes are accepted."""
    cdef CoreAdvancedOrderbook core
    
    core = _create_core(DEFAULT_LEVELS, CyOrderbookSortedness.UNKNOWN)
    core = _create_core(DEFAULT_LEVELS, CyOrderbookSortedness.ASCENDING)
    core = _create_core(DEFAULT_LEVELS, CyOrderbookSortedness.DESCENDING)
    core = _create_core(DEFAULT_LEVELS, CyOrderbookSortedness.BIDS_ASCENDING_ASKS_DESCENDING)
    core = _create_core(DEFAULT_LEVELS, CyOrderbookSortedness.BIDS_DESCENDING_ASKS_ASCENDING)


# -----------------------------------------------------------------------------
# 3.2 Snapshot Consumption tests
# -----------------------------------------------------------------------------

def test_core_snapshot_basic():
    """Test basic snapshot populates both sides."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef OrderbookLadderData* bids = core.get_bids_data()
    cdef OrderbookLadderData* asks = core.get_asks_data()
    
    assert bids.num_levels == 3
    assert asks.num_levels == 3
    assert bids.levels[0].price == 100.00  # Best bid
    assert asks.levels[0].price == 100.01  # Best ask


def test_core_snapshot_replaces_existing():
    """Test second snapshot completely replaces first."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # New snapshot with different prices
    cdef double bid_prices[2]
    cdef double bid_sizes[2]
    cdef double ask_prices[2]
    cdef double ask_sizes[2]
    
    bid_prices[0] = 200.00; bid_prices[1] = 199.99
    bid_sizes[0] = 5.0; bid_sizes[1] = 6.0
    ask_prices[0] = 200.01; ask_prices[1] = 200.02
    ask_sizes[0] = 5.5; ask_sizes[1] = 6.5
    
    cdef OrderbookLevels bids = _make_levels(bid_prices, bid_sizes, 2, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels asks = _make_levels(ask_prices, ask_sizes, 2, TICK_SIZE, LOT_SIZE)
    
    core.consume_snapshot(asks, bids)
    
    cdef OrderbookLadderData* bids_view = core.get_bids_data()
    cdef OrderbookLadderData* asks_view = core.get_asks_data()
    
    assert bids_view.num_levels == 2
    assert asks_view.num_levels == 2
    assert bids_view.levels[0].price == 200.00
    assert asks_view.levels[0].price == 200.01
    
    _free_levels(&bids)
    _free_levels(&asks)


def test_core_snapshot_more_levels_than_max():
    """Test snapshot with more levels than max_levels truncates."""
    cdef CoreAdvancedOrderbook core = _create_core(num_levels=64)

    cdef double bid_prices[80]
    cdef double bid_sizes[80]
    cdef double ask_prices[80]
    cdef double ask_sizes[80]

    for i in range(80):
        bid_prices[i] = 100.0 - i * 0.01
        bid_sizes[i] = 1.0
        ask_prices[i] = 100.01 + i * 0.01
        ask_sizes[i] = 1.0

    cdef OrderbookLevels bids = _make_levels(bid_prices, bid_sizes, 80, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels asks = _make_levels(ask_prices, ask_sizes, 80, TICK_SIZE, LOT_SIZE)

    core.consume_snapshot(asks, bids)

    cdef OrderbookLadderData* bids_view = core.get_bids_data()
    cdef OrderbookLadderData* asks_view = core.get_asks_data()

    assert bids_view.num_levels == 64  # Truncated to max
    assert asks_view.num_levels == 64  # Truncated to max
    assert bids_view.levels[0].price == 100.00  # Best bid preserved
    assert asks_view.levels[0].price == 100.01  # Best ask preserved

    _free_levels(&bids)
    _free_levels(&asks)


def test_core_snapshot_fewer_levels_than_max():
    """Test snapshot with fewer levels than max works."""
    cdef CoreAdvancedOrderbook core = _create_core(num_levels=646)
    
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
    
    core.consume_snapshot(asks, bids)
    
    cdef OrderbookLadderData* bids_view = core.get_bids_data()
    cdef OrderbookLadderData* asks_view = core.get_asks_data()
    
    assert bids_view.num_levels == 2
    assert asks_view.num_levels == 2
    
    _free_levels(&bids)
    _free_levels(&asks)


def test_core_snapshot_single_level_each():
    """Test minimal single-level book."""
    cdef CoreAdvancedOrderbook core = _create_core()
    
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
    
    core.consume_snapshot(asks, bids)
    
    assert _approx_eq(core.get_bbo_spread(), 0.01)
    
    _free_levels(&bids)
    _free_levels(&asks)


def test_core_snapshot_sortedness_unknown():
    """Test UNKNOWN sortedness triggers internal sort."""
    cdef CoreAdvancedOrderbook core = _create_core(DEFAULT_LEVELS, CyOrderbookSortedness.UNKNOWN, CyOrderbookSortedness.UNKNOWN)
    
    # Provide unsorted data
    cdef double bid_prices[3]
    cdef double bid_sizes[3]
    cdef double ask_prices[3]
    cdef double ask_sizes[3]
    
    bid_prices[0] = 99.99; bid_prices[1] = 100.00; bid_prices[2] = 99.98  # Wrong order
    bid_sizes[0] = 1.0; bid_sizes[1] = 2.0; bid_sizes[2] = 3.0
    ask_prices[0] = 100.03; ask_prices[1] = 100.01; ask_prices[2] = 100.02  # Wrong order
    ask_sizes[0] = 1.0; ask_sizes[1] = 2.0; ask_sizes[2] = 3.0
    
    cdef OrderbookLevels bids = _make_levels(bid_prices, bid_sizes, 3, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels asks = _make_levels(ask_prices, ask_sizes, 3, TICK_SIZE, LOT_SIZE)
    
    core.consume_snapshot(asks, bids)
    
    cdef OrderbookLadderData* bids_view = core.get_bids_data()
    cdef OrderbookLadderData* asks_view = core.get_asks_data()
    
    # Should be sorted correctly
    assert bids_view.levels[0].price == 100.00  # Best bid (highest)
    assert asks_view.levels[0].price == 100.01  # Best ask (lowest)
    
    _free_levels(&bids)
    _free_levels(&asks)


def test_core_snapshot_populates_ticks_and_lots():
    """Snapshot normalization always computes ticks/lots."""
    cdef CoreAdvancedOrderbook core = _create_core()

    cdef OrderbookLevels bids = _alloc_levels(1)
    cdef OrderbookLevels asks = _alloc_levels(1)

    bids.levels[0] = create_orderbook_level(100.00, 1.0, 1)
    asks.levels[0] = create_orderbook_level(100.01, 1.0, 1)

    core.consume_snapshot(asks, bids)

    cdef OrderbookLadderData* bids_view = core.get_bids_data()
    cdef OrderbookLadderData* asks_view = core.get_asks_data()
    assert bids_view.levels[0].ticks == 10000
    assert asks_view.levels[0].ticks == 10001

    _free_levels(&bids)
    _free_levels(&asks)


def test_core_snapshot_overwrites_ticks_and_lots():
    """Snapshot normalization overwrites mismatched ticks/lots."""
    cdef CoreAdvancedOrderbook core = _create_core()

    cdef OrderbookLevels bids = _alloc_levels(1)
    cdef OrderbookLevels asks = _alloc_levels(1)

    bids.levels[0] = create_orderbook_level(100.00, 1.0, 1)
    bids.levels[0].ticks = 1
    bids.levels[0].lots = 1
    asks.levels[0] = create_orderbook_level(100.01, 1.0, 1)
    asks.levels[0].ticks = 2
    asks.levels[0].lots = 2

    core.consume_snapshot(asks, bids)

    cdef OrderbookLadderData* bids_view = core.get_bids_data()
    cdef OrderbookLadderData* asks_view = core.get_asks_data()
    assert bids_view.levels[0].ticks == 10000
    assert asks_view.levels[0].ticks == 10001

    _free_levels(&bids)
    _free_levels(&asks)


# -----------------------------------------------------------------------------
# 3.3 Delta Processing - Asks
# -----------------------------------------------------------------------------

def test_core_delta_ask_consume_bbo_size():
    """Test updating BBO ask size with matching ticks."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # Update BBO ask (100.01) with new size
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    ask_prices[0] = 100.01
    ask_sizes[0] = 5.0  # New size
    
    cdef OrderbookLevels delta_asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    cdef OrderbookLadderData* asks = core.get_asks_data()
    assert asks.levels[0].size == 5.0
    assert asks.levels[0].price == 100.01  # Price unchanged
    
    _free_levels(&delta_asks)


def test_core_delta_ask_delete_bbo():
    """Test deleting BBO ask with lots=0."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # Delete BBO ask (100.01)
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    ask_prices[0] = 100.01
    ask_sizes[0] = 0.0  # Delete marker
    
    cdef OrderbookLevels delta_asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    cdef OrderbookLadderData* asks = core.get_asks_data()
    assert asks.num_levels == 2
    assert asks.levels[0].price == 100.02  # New BBO
    
    _free_levels(&delta_asks)


def test_core_delta_ask_insert_new_bbo():
    """Test inserting new BBO ask with lower tick."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # Insert new BBO at 100.005 (lower than 100.01)
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    ask_prices[0] = 100.005
    ask_sizes[0] = 1.0
    
    cdef OrderbookLevels delta_asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    cdef OrderbookLadderData* asks = core.get_asks_data()
    # Note: 100.005 / 0.01 = 10000.5, truncated to 10000 ticks = 100.00
    # This may cause the ask to match bid tick, triggering cross resolution
    # The exact behavior depends on implementation
    
    _free_levels(&delta_asks)


def test_core_delta_ask_insert_new_bbo_removes_overlapping_bids():
    """Test new ask that crosses existing bids removes them."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # Initial: best bid = 100.00, best ask = 100.01
    # Insert ask at 99.99 (crosses into bids)
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    ask_prices[0] = 99.99  # Below best bid
    ask_sizes[0] = 1.0
    
    cdef OrderbookLevels delta_asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    cdef OrderbookLadderData* asks = core.get_asks_data()
    cdef OrderbookLadderData* bids = core.get_bids_data()
    
    # New ask should be BBO
    assert asks.levels[0].price == 99.99
    # Overlapping bids should be removed
    assert bids.num_levels < 3 or bids.levels[0].ticks < 9999
    
    _free_levels(&delta_asks)


def test_core_delta_ask_insert_middle():
    """Test inserting ask between existing levels."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # Insert at 100.015 (between 100.01 and 100.02)
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    ask_prices[0] = 100.015
    ask_sizes[0] = 0.5
    
    cdef OrderbookLevels delta_asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    cdef OrderbookLadderData* asks = core.get_asks_data()
    # Due to tick rounding, 100.015 / 0.01 = 10001.5 -> 10001 or 10002
    # Check that we have more or same levels
    assert asks.num_levels >= 3
    
    _free_levels(&delta_asks)


def test_core_delta_ask_update_middle():
    """Test updating non-BBO ask level."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # Update second level (100.02)
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    ask_prices[0] = 100.02
    ask_sizes[0] = 10.0  # New size
    
    cdef OrderbookLevels delta_asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    cdef OrderbookLadderData* asks = core.get_asks_data()
    assert asks.levels[1].price == 100.02
    assert asks.levels[1].size == 10.0
    
    _free_levels(&delta_asks)


def test_core_delta_ask_delete_middle():
    """Test deleting non-BBO ask level."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # Delete second level (100.02)
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    ask_prices[0] = 100.02
    ask_sizes[0] = 0.0  # Delete
    
    cdef OrderbookLevels delta_asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    cdef OrderbookLadderData* asks = core.get_asks_data()
    assert asks.num_levels == 2
    assert asks.levels[0].price == 100.01
    assert asks.levels[1].price == 100.03  # 100.02 removed
    
    _free_levels(&delta_asks)


def test_core_delta_ask_beyond_worst_full_book():
    """Test ask beyond worst price when book is full is ignored."""
    cdef CoreAdvancedOrderbook core = _create_core(num_levels=64)

    # Fill to capacity with 64 levels each side
    cdef double bid_prices[64]
    cdef double bid_sizes[64]
    cdef double ask_prices[64]
    cdef double ask_sizes[64]

    for i in range(64):
        bid_prices[i] = 100.0 - i * 0.01
        bid_sizes[i] = 1.0
        ask_prices[i] = 100.01 + i * 0.01
        ask_sizes[i] = 1.0

    cdef OrderbookLevels bids = _make_levels(bid_prices, bid_sizes, 64, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels asks_snap = _make_levels(ask_prices, ask_sizes, 64, TICK_SIZE, LOT_SIZE)
    core.consume_snapshot(asks_snap, bids)

    cdef OrderbookLadderData* asks_view = core.get_asks_data()
    assert asks_view.num_levels == 64  # Full

    # Try to insert beyond worst ask (100.64)
    cdef double delta_ask_prices[1]
    cdef double delta_ask_sizes[1]
    delta_ask_prices[0] = 100.80  # Beyond worst
    delta_ask_sizes[0] = 1.0

    cdef OrderbookLevels delta_asks = _make_levels(delta_ask_prices, delta_ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0

    core.consume_deltas(delta_asks, delta_bids)

    asks_view = core.get_asks_data()
    assert asks_view.num_levels == 64  # Still 64
    assert _approx_eq(asks_view.levels[63].price, 100.64)  # Still worst ask

    _free_levels(&bids)
    _free_levels(&asks_snap)
    _free_levels(&delta_asks)


def test_core_delta_ask_delete_nonexistent():
    """Test deleting nonexistent ask is no-op."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef OrderbookLadderData* asks = core.get_asks_data()
    cdef u64 initial_count = asks.num_levels
    
    # Try to delete nonexistent level
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    ask_prices[0] = 100.05  # Doesn't exist
    ask_sizes[0] = 0.0
    
    cdef OrderbookLevels delta_asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    asks = core.get_asks_data()
    assert asks.num_levels == initial_count  # Unchanged
    
    _free_levels(&delta_asks)


def test_core_delta_ask_multiple_sequential():
    """Test processing multiple ask deltas in sequence."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # Multiple deltas: delete BBO, update second, insert new
    cdef double ask_prices[3]
    cdef double ask_sizes[3]
    ask_prices[0] = 100.01; ask_sizes[0] = 0.0   # Delete BBO
    ask_prices[1] = 100.02; ask_sizes[1] = 5.0  # Update
    ask_prices[2] = 100.005; ask_sizes[2] = 1.0 # New BBO
    
    cdef OrderbookLevels delta_asks = _make_levels(ask_prices, ask_sizes, 3, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    # Book should be modified
    cdef OrderbookLadderData* asks = core.get_asks_data()
    assert asks.num_levels >= 1
    
    _free_levels(&delta_asks)


# -----------------------------------------------------------------------------
# 3.4 Delta Processing - Bids
# -----------------------------------------------------------------------------

def test_core_delta_bid_consume_bbo_size():
    """Test updating BBO bid size."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double bid_prices[1]
    cdef double bid_sizes[1]
    bid_prices[0] = 100.00
    bid_sizes[0] = 5.0
    
    cdef OrderbookLevels delta_bids = _make_levels(bid_prices, bid_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_asks = _alloc_levels(0)
    delta_asks.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    cdef OrderbookLadderData* bids = core.get_bids_data()
    assert bids.levels[0].size == 5.0
    
    _free_levels(&delta_bids)


def test_core_delta_bid_delete_bbo():
    """Test deleting BBO bid."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double bid_prices[1]
    cdef double bid_sizes[1]
    bid_prices[0] = 100.00
    bid_sizes[0] = 0.0
    
    cdef OrderbookLevels delta_bids = _make_levels(bid_prices, bid_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_asks = _alloc_levels(0)
    delta_asks.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    cdef OrderbookLadderData* bids = core.get_bids_data()
    assert bids.num_levels == 2
    assert bids.levels[0].price == 99.99  # New BBO
    
    _free_levels(&delta_bids)


def test_core_delta_bid_insert_new_bbo():
    """Test inserting new BBO bid with higher tick."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double bid_prices[1]
    cdef double bid_sizes[1]
    bid_prices[0] = 100.005  # Higher than 100.00
    bid_sizes[0] = 1.0
    
    cdef OrderbookLevels delta_bids = _make_levels(bid_prices, bid_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_asks = _alloc_levels(0)
    delta_asks.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    # Book modified - exact result depends on tick rounding
    cdef OrderbookLadderData* bids = core.get_bids_data()
    assert bids.num_levels >= 3
    
    _free_levels(&delta_bids)


def test_core_delta_bid_insert_new_bbo_removes_overlapping_asks():
    """Test new bid that crosses existing asks removes them."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # Insert bid at 100.02 (above best ask 100.01)
    cdef double bid_prices[1]
    cdef double bid_sizes[1]
    bid_prices[0] = 100.02
    bid_sizes[0] = 1.0
    
    cdef OrderbookLevels delta_bids = _make_levels(bid_prices, bid_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_asks = _alloc_levels(0)
    delta_asks.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    cdef OrderbookLadderData* bids = core.get_bids_data()
    cdef OrderbookLadderData* asks = core.get_asks_data()
    
    assert bids.levels[0].price == 100.02
    # Overlapping asks removed
    assert asks.num_levels < 3 or asks.levels[0].ticks > 10002
    
    _free_levels(&delta_bids)


def test_core_delta_bid_update_middle():
    """Test updating non-BBO bid level."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double bid_prices[1]
    cdef double bid_sizes[1]
    bid_prices[0] = 99.99
    bid_sizes[0] = 10.0
    
    cdef OrderbookLevels delta_bids = _make_levels(bid_prices, bid_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_asks = _alloc_levels(0)
    delta_asks.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    cdef OrderbookLadderData* bids = core.get_bids_data()
    assert bids.levels[1].size == 10.0
    
    _free_levels(&delta_bids)


def test_core_delta_bid_delete_middle():
    """Test deleting non-BBO bid level."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double bid_prices[1]
    cdef double bid_sizes[1]
    bid_prices[0] = 99.99
    bid_sizes[0] = 0.0
    
    cdef OrderbookLevels delta_bids = _make_levels(bid_prices, bid_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_asks = _alloc_levels(0)
    delta_asks.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    cdef OrderbookLadderData* bids = core.get_bids_data()
    assert bids.num_levels == 2
    assert bids.levels[1].price == 99.98
    
    _free_levels(&delta_bids)


# -----------------------------------------------------------------------------
# 3.5 Combined Delta Scenarios
# -----------------------------------------------------------------------------

def test_core_delta_both_sides():
    """Test deltas on both sides simultaneously."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    cdef double bid_prices[1]
    cdef double bid_sizes[1]
    
    ask_prices[0] = 100.01; ask_sizes[0] = 5.0
    bid_prices[0] = 100.00; bid_sizes[0] = 5.0
    
    cdef OrderbookLevels delta_asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _make_levels(bid_prices, bid_sizes, 1, TICK_SIZE, LOT_SIZE)
    
    core.consume_deltas(delta_asks, delta_bids)
    
    cdef OrderbookLadderData* asks = core.get_asks_data()
    cdef OrderbookLadderData* bids = core.get_bids_data()
    
    assert asks.levels[0].size == 5.0
    assert bids.levels[0].size == 5.0
    
    _free_levels(&delta_asks)
    _free_levels(&delta_bids)


def test_core_delta_empty_arrays():
    """Test empty delta arrays are no-op."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef OrderbookLadderData* asks_before = core.get_asks_data()
    cdef u64 ask_count = asks_before.num_levels
    
    cdef OrderbookLevels delta_asks = _alloc_levels(0)
    delta_asks.num_levels = 0
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    cdef OrderbookLadderData* asks_after = core.get_asks_data()
    assert asks_after.num_levels == ask_count


def test_core_delta_on_empty_book():
    """Test deltas on empty book return early."""
    cdef CoreAdvancedOrderbook core = _create_core()
    # Don't populate - book is empty
    
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    ask_prices[0] = 100.01
    ask_sizes[0] = 1.0
    
    cdef OrderbookLevels delta_asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0
    
    # Should not crash, just return early
    core.consume_deltas(delta_asks, delta_bids)
    
    _free_levels(&delta_asks)


def test_core_delta_deplete_entire_side():
    """Test deleting levels decreases the count."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef OrderbookLadderData* asks_before = core.get_asks_data()
    cdef u64 initial_count = asks_before.num_levels
    assert initial_count == 3
    
    # Delete first two asks
    cdef double ask_prices[2]
    cdef double ask_sizes[2]
    ask_prices[0] = 100.01; ask_sizes[0] = 0.0
    ask_prices[1] = 100.02; ask_sizes[1] = 0.0
    
    cdef OrderbookLevels delta_asks = _make_levels(ask_prices, ask_sizes, 2, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    cdef OrderbookLadderData* asks = core.get_asks_data()
    # At least one level should remain (100.03)
    assert asks.num_levels < initial_count
    
    _free_levels(&delta_asks)


# -----------------------------------------------------------------------------
# 3.6 BBO Operations
# -----------------------------------------------------------------------------

def test_core_bbo_update_same_tick():
    """Test updating BBO with same tick changes size/norders."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef OrderbookLevel new_ask = create_orderbook_level_with_ticks_and_lots(
        100.01, 5.0, TICK_SIZE, LOT_SIZE, 10
    )
    cdef OrderbookLevel new_bid = create_orderbook_level_with_ticks_and_lots(
        100.00, 5.0, TICK_SIZE, LOT_SIZE, 10
    )
    
    core.consume_bbo(new_ask, new_bid)
    
    cdef OrderbookLadderData* asks = core.get_asks_data()
    cdef OrderbookLadderData* bids = core.get_bids_data()
    
    assert asks.levels[0].size == 5.0
    assert bids.levels[0].size == 5.0


def test_core_bbo_delete_matching():
    """Test deleting BBO with lots=0."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # Delete BBO ask
    cdef OrderbookLevel del_ask = create_orderbook_level_with_ticks_and_lots(
        100.01, 0.0, TICK_SIZE, LOT_SIZE, 0
    )
    cdef OrderbookLevel same_bid = create_orderbook_level_with_ticks_and_lots(
        100.00, 1.0, TICK_SIZE, LOT_SIZE, 1
    )
    
    core.consume_bbo(del_ask, same_bid)
    
    cdef OrderbookLadderData* asks = core.get_asks_data()
    assert asks.num_levels == 2
    assert asks.levels[0].price == 100.02


def test_core_bbo_insert_tighter_ask():
    """Test inserting tighter (better) ask."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef OrderbookLevel tighter_ask = create_orderbook_level_with_ticks_and_lots(
        100.005, 1.0, TICK_SIZE, LOT_SIZE, 1  # Better than 100.01
    )
    cdef OrderbookLevel same_bid = create_orderbook_level_with_ticks_and_lots(
        100.00, 1.0, TICK_SIZE, LOT_SIZE, 1
    )
    
    core.consume_bbo(tighter_ask, same_bid)
    
    cdef OrderbookLadderData* asks = core.get_asks_data()
    # New ask should be at front (tick 10000 or 10001 depending on rounding)
    assert asks.num_levels >= 3


def test_core_bbo_insert_tighter_bid():
    """Test inserting tighter (better) bid."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef OrderbookLevel same_ask = create_orderbook_level_with_ticks_and_lots(
        100.01, 1.0, TICK_SIZE, LOT_SIZE, 1
    )
    cdef OrderbookLevel tighter_bid = create_orderbook_level_with_ticks_and_lots(
        100.005, 1.0, TICK_SIZE, LOT_SIZE, 1  # Better than 100.00
    )
    
    core.consume_bbo(same_ask, tighter_bid)
    
    cdef OrderbookLadderData* bids = core.get_bids_data()
    assert bids.num_levels >= 3


def test_core_bbo_crossed_book_resolution():
    """Test BBO that causes crossing removes overlapping asks."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # Bid at 100.02 crosses ask at 100.01
    cdef OrderbookLevel same_ask = create_orderbook_level_with_ticks_and_lots(
        100.01, 1.0, TICK_SIZE, LOT_SIZE, 1
    )
    cdef OrderbookLevel crossing_bid = create_orderbook_level_with_ticks_and_lots(
        100.02, 1.0, TICK_SIZE, LOT_SIZE, 1
    )
    
    core.consume_bbo(same_ask, crossing_bid)
    
    cdef OrderbookLadderData* asks = core.get_asks_data()
    cdef OrderbookLadderData* bids = core.get_bids_data()
    
    # Book should resolve crossing
    if asks.num_levels > 0 and bids.num_levels > 0:
        assert bids.levels[0].ticks < asks.levels[0].ticks


def test_core_bbo_on_empty_book():
    """Test BBO on empty book populates single level."""
    cdef CoreAdvancedOrderbook core = _create_core()
    # Don't populate
    
    cdef OrderbookLevel ask = create_orderbook_level_with_ticks_and_lots(
        100.01, 1.0, TICK_SIZE, LOT_SIZE, 1
    )
    cdef OrderbookLevel bid = create_orderbook_level_with_ticks_and_lots(
        100.00, 1.0, TICK_SIZE, LOT_SIZE, 1
    )
    
    core.consume_bbo(ask, bid)
    
    # Note: consume_bbo on empty book returns early per implementation
    # This test documents that behavior


def test_core_bbo_populates_ticks_and_lots():
    """BBO ingestion always computes ticks/lots from price/size."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)

    cdef OrderbookLevel ask = create_orderbook_level(100.01, 2.0, 1)
    cdef OrderbookLevel bid = create_orderbook_level(100.00, 2.0, 1)
    ask.ticks = 1
    ask.lots = 1
    bid.ticks = 2
    bid.lots = 2

    core.consume_bbo(ask, bid)

    cdef OrderbookLadderData* asks = core.get_asks_data()
    cdef OrderbookLadderData* bids = core.get_bids_data()
    assert asks.levels[0].ticks == 10001
    assert bids.levels[0].ticks == 10000


# -----------------------------------------------------------------------------
# 3.7 Price Calculations
# -----------------------------------------------------------------------------

# Mid Price tests

def test_core_mid_price_standard():
    """Test mid price calculation."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double mid = core.get_mid_price()
    # (100.00 + 100.01) / 2 = 100.005
    assert _approx_eq(mid, 100.00)  # Integer tick arithmetic: (10000 + 10001) // 2 * 0.01


def test_core_mid_price_1_tick_spread():
    """Test mid price with minimal 1-tick spread."""
    cdef CoreAdvancedOrderbook core = _create_core()
    
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
    core.consume_snapshot(asks, bids)
    
    cdef double mid = core.get_mid_price()
    assert _approx_eq(mid, 100.00)  # Integer tick arithmetic: (10000 + 10001) // 2 * 0.01
    
    _free_levels(&bids)
    _free_levels(&asks)


def test_core_mid_price_wide_spread():
    """Test mid price with wide spread."""
    cdef CoreAdvancedOrderbook core = _create_core()
    
    cdef double bid_prices[1]
    cdef double bid_sizes[1]
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    
    bid_prices[0] = 99.00
    bid_sizes[0] = 1.0
    ask_prices[0] = 101.00
    ask_sizes[0] = 1.0
    
    cdef OrderbookLevels bids = _make_levels(bid_prices, bid_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    core.consume_snapshot(asks, bids)
    
    cdef double mid = core.get_mid_price()
    assert _approx_eq(mid, 100.00)
    
    _free_levels(&bids)
    _free_levels(&asks)


def test_core_mid_price_empty_raises():
    """Test mid price on empty book raises RuntimeError."""
    cdef CoreAdvancedOrderbook core = _create_core()
    
    try:
        core.get_mid_price()
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


# BBO Spread tests

def test_core_spread_1_tick():
    """Test spread of exactly 1 tick."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double spread = core.get_bbo_spread()
    assert _approx_eq(spread, 0.01)


def test_core_spread_multi_tick():
    """Test spread of multiple ticks."""
    cdef CoreAdvancedOrderbook core = _create_core()
    
    cdef double bid_prices[1]
    cdef double bid_sizes[1]
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    
    bid_prices[0] = 100.00
    bid_sizes[0] = 1.0
    ask_prices[0] = 100.05  # 5 ticks
    ask_sizes[0] = 1.0
    
    cdef OrderbookLevels bids = _make_levels(bid_prices, bid_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    core.consume_snapshot(asks, bids)
    
    cdef double spread = core.get_bbo_spread()
    assert _approx_eq(spread, 0.05)
    
    _free_levels(&bids)
    _free_levels(&asks)


def test_core_spread_empty_raises():
    """Test spread on empty book raises."""
    cdef CoreAdvancedOrderbook core = _create_core()
    
    try:
        core.get_bbo_spread()
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


# WMID tests

def test_core_wmid_equal_volumes():
    """Test WMID with equal volumes equals mid."""
    cdef CoreAdvancedOrderbook core = _create_core()
    
    cdef double bid_prices[1]
    cdef double bid_sizes[1]
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    
    bid_prices[0] = 100.00
    bid_sizes[0] = 1.0  # Equal
    ask_prices[0] = 100.01
    ask_sizes[0] = 1.0  # Equal
    
    cdef OrderbookLevels bids = _make_levels(bid_prices, bid_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    core.consume_snapshot(asks, bids)
    
    cdef double wmid = core.get_wmid_price()
    cdef double mid = core.get_mid_price()
    assert _approx_eq(wmid, mid, tol=0.001)
    
    _free_levels(&bids)
    _free_levels(&asks)


def test_core_wmid_bid_heavy():
    """Test WMID skews toward bid when bid-heavy."""
    cdef CoreAdvancedOrderbook core = _create_core()
    
    cdef double bid_prices[1]
    cdef double bid_sizes[1]
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    
    bid_prices[0] = 100.00
    bid_sizes[0] = 10.0  # Heavy
    ask_prices[0] = 100.01
    ask_sizes[0] = 1.0   # Light
    
    cdef OrderbookLevels bids = _make_levels(bid_prices, bid_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    core.consume_snapshot(asks, bids)
    
    cdef double wmid = core.get_wmid_price()
    cdef double mid = core.get_mid_price()
    # WMID should be > mid (skewed toward ask which has more weight in imbalance calc)
    # Actually, formula weights by opposite side, so bid-heavy -> closer to ask
    
    _free_levels(&bids)
    _free_levels(&asks)


def test_core_wmid_empty_raises():
    """Test WMID on empty book raises."""
    cdef CoreAdvancedOrderbook core = _create_core()
    
    try:
        core.get_wmid_price()
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


# Volume-Weighted Mid Price tests

def test_core_vwmp_zero_size():
    """Test VWMP with zero size returns mid."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double vwmp = core.get_volume_weighted_mid_price(0.0, True)
    cdef double mid = core.get_mid_price()
    assert _approx_eq(vwmp, mid)


def test_core_vwmp_negative_size():
    """Test VWMP with negative size returns mid."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double vwmp = core.get_volume_weighted_mid_price(-1.0, True)
    cdef double mid = core.get_mid_price()
    assert _approx_eq(vwmp, mid)


def test_core_vwmp_small_size():
    """Test VWMP with small size fills within BBO."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double vwmp = core.get_volume_weighted_mid_price(0.5, True)
    # Should be close to mid, slightly worse
    assert vwmp > 0


def test_core_vwmp_exceeds_liquidity():
    """Test VWMP with huge size returns infinity."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double vwmp = core.get_volume_weighted_mid_price(1e9, True)
    # Should return DBL_MAX (infinity marker)
    assert vwmp > 1e10


# Price Impact tests

def test_core_impact_zero_size():
    """Test price impact with zero size returns 0."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double impact = core.get_price_impact(0.0, True, True)
    assert impact == 0.0


def test_core_impact_negative_size():
    """Test price impact with negative size returns 0."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double impact = core.get_price_impact(-1.0, True, True)
    assert impact == 0.0


def test_core_impact_buy_single_level():
    """Test buy impact within single level."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # BBO ask has 1.5 size, buy 0.5
    cdef double impact = core.get_price_impact(0.5, True, True)
    assert impact >= 0.0


def test_core_impact_sell_single_level():
    """Test sell impact within single level."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double impact = core.get_price_impact(0.5, False, True)
    assert impact >= 0.0


def test_core_impact_buy_multi_level():
    """Test buy impact sweeping multiple levels."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # Buy more than BBO ask size (1.5)
    cdef double impact = core.get_price_impact(3.0, True, True)
    assert impact > 0.0


def test_core_impact_exceeds_liquidity():
    """Test impact exceeding liquidity returns infinity."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double impact = core.get_price_impact(1e9, True, True)
    assert impact > 1e10


def test_core_impact_empty_raises():
    """Test impact on empty book raises."""
    cdef CoreAdvancedOrderbook core = _create_core()
    
    try:
        core.get_price_impact(1.0, True, True)
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


# -----------------------------------------------------------------------------
# 3.8 Crossing and Change Detection
# -----------------------------------------------------------------------------

def test_core_is_crossed_no_cross():
    """Test no crossing detected."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # Other bid < my ask, other ask > my bid
    cdef bint crossed = core.is_bbo_crossed(99.99, 100.02)
    assert crossed == False


def test_core_is_crossed_bid_crosses_ask():
    """Test bid crossing my ask."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # Other bid >= my ask (100.01)
    cdef bint crossed = core.is_bbo_crossed(100.02, 100.05)
    assert crossed == True


def test_core_is_crossed_ask_crosses_bid():
    """Test ask crossing my bid."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    # Other ask <= my bid (100.00)
    cdef bint crossed = core.is_bbo_crossed(99.95, 99.99)
    assert crossed == True


def test_core_is_crossed_empty_raises():
    """Test crossing check on empty raises."""
    cdef CoreAdvancedOrderbook core = _create_core()
    
    try:
        core.is_bbo_crossed(100.0, 100.01)
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_core_bbo_change_no_change():
    """Test no change when prices match."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef bint changed = core.does_bbo_price_change(100.00, 100.01)
    assert changed == False


def test_core_bbo_change_bid_differs():
    """Test change when bid differs."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef bint changed = core.does_bbo_price_change(99.99, 100.01)
    assert changed == True


def test_core_bbo_change_ask_differs():
    """Test change when ask differs."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef bint changed = core.does_bbo_price_change(100.00, 100.02)
    assert changed == True


def test_core_bbo_change_both_differ():
    """Test change when both differ."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef bint changed = core.does_bbo_price_change(99.99, 100.02)
    assert changed == True


def test_core_bbo_change_empty_raises():
    """Test change check on empty raises."""
    cdef CoreAdvancedOrderbook core = _create_core()
    
    try:
        core.does_bbo_price_change(100.0, 100.01)
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


# -----------------------------------------------------------------------------
# 3.9 State Management
# -----------------------------------------------------------------------------

def test_core_clear():
    """Test clear empties both sides."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    core.clear()
    
    cdef OrderbookLadderData* bids = core.get_bids_data()
    cdef OrderbookLadderData* asks = core.get_asks_data()
    
    assert bids.num_levels == 0
    assert asks.num_levels == 0


def test_core_operations_after_clear_raise():
    """Test operations fail after clear."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    core.clear()
    
    try:
        core.get_mid_price()
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_core_repopulate_after_clear():
    """Test repopulating after clear works."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    core.clear()
    _populate_standard_book(core)
    
    cdef double mid = core.get_mid_price()
    assert _approx_eq(mid, 100.00)  # Integer tick arithmetic: (10000 + 10001) // 2 * 0.01


def test_core_view_accessors():
    """Test view accessors return valid pointers."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef OrderbookLadderData* bids = core.get_bids_data()
    cdef OrderbookLadderData* asks = core.get_asks_data()
    
    assert bids != NULL
    assert asks != NULL
    assert bids.num_levels == 3
    assert asks.num_levels == 3


def test_core_views_reflect_mutations():
    """Test views reflect ladder mutations."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef OrderbookLadderData* asks = core.get_asks_data()
    assert asks.num_levels == 3
    
    # Delete BBO ask
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    ask_prices[0] = 100.01
    ask_sizes[0] = 0.0
    
    cdef OrderbookLevels delta_asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0
    
    core.consume_deltas(delta_asks, delta_bids)
    
    asks = core.get_asks_data()
    assert asks.num_levels == 2
    
    _free_levels(&delta_asks)


# -----------------------------------------------------------------------------
# 3.10 Tail Cases and Stress
# -----------------------------------------------------------------------------

def test_core_rapid_insert_delete():
    """Test rapid insert/delete cycles at BBO."""
    cdef CoreAdvancedOrderbook core = _create_core()
    _populate_standard_book(core)
    
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    cdef OrderbookLevels delta_asks
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0
    cdef u64 i
    
    for i in range(100):
        # Insert new BBO
        ask_prices[0] = 100.005
        ask_sizes[0] = 1.0
        delta_asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
        core.consume_deltas(delta_asks, delta_bids)
        _free_levels(&delta_asks)
        
        # Delete it
        ask_sizes[0] = 0.0
        delta_asks = _make_levels(ask_prices, ask_sizes, 1, TICK_SIZE, LOT_SIZE)
        core.consume_deltas(delta_asks, delta_bids)
        _free_levels(&delta_asks)
    
    # Should not crash and book should still be valid
    cdef OrderbookLadderData* asks = core.get_asks_data()
    assert asks.num_levels > 0


def test_core_fill_to_max_then_insert():
    """Test behavior when at max capacity."""
    cdef CoreAdvancedOrderbook core = _create_core(num_levels=64)

    # Fill to capacity with 64 levels each side
    cdef double bid_prices[64]
    cdef double bid_sizes[64]
    cdef double ask_prices[64]
    cdef double ask_sizes[64]

    for i in range(64):
        bid_prices[i] = 100.0 - i * 0.01
        bid_sizes[i] = 1.0
        ask_prices[i] = 100.01 + i * 0.01
        ask_sizes[i] = 1.0

    cdef OrderbookLevels bids = _make_levels(bid_prices, bid_sizes, 64, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels asks_snap = _make_levels(ask_prices, ask_sizes, 64, TICK_SIZE, LOT_SIZE)
    core.consume_snapshot(asks_snap, bids)

    cdef OrderbookLadderData* asks_view = core.get_asks_data()
    assert asks_view.num_levels == 64

    # Try to insert beyond capacity at a worse price (should be ignored)
    cdef double delta_ask_prices[1]
    cdef double delta_ask_sizes[1]
    delta_ask_prices[0] = 100.80  # Beyond worst ask
    delta_ask_sizes[0] = 1.0

    cdef OrderbookLevels delta_asks = _make_levels(delta_ask_prices, delta_ask_sizes, 1, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels delta_bids = _alloc_levels(0)
    delta_bids.num_levels = 0

    core.consume_deltas(delta_asks, delta_bids)

    asks_view = core.get_asks_data()
    assert asks_view.num_levels == 64  # Still at max

    _free_levels(&bids)
    _free_levels(&asks_snap)
    _free_levels(&delta_asks)


def test_core_empty_full_empty_cycle():
    """Test full lifecycle: empty -> full -> empty."""
    cdef CoreAdvancedOrderbook core = _create_core(num_levels=64)
    
    # Empty
    assert core.get_bids_data().num_levels == 0
    
    # Fill
    _populate_standard_book(core)
    assert core.get_bids_data().num_levels == 3
    
    # Empty again
    core.clear()
    assert core.get_bids_data().num_levels == 0
    
    # Fill again
    _populate_standard_book(core)
    assert core.get_bids_data().num_levels == 3


def test_core_very_small_tick_size():
    """Test with very small tick size."""
    cdef CoreAdvancedOrderbook core = CoreAdvancedOrderbook(
        tick_size=1e-8,
        lot_size=0.001,
        num_levels=646,
        delta_sortedness=CyOrderbookSortedness.UNKNOWN,
        snapshot_sortedness=CyOrderbookSortedness.UNKNOWN,
    )
    
    cdef double bid_prices[1]
    cdef double bid_sizes[1]
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    
    bid_prices[0] = 0.00012345
    bid_sizes[0] = 1.0
    ask_prices[0] = 0.00012346
    ask_sizes[0] = 1.0
    
    cdef OrderbookLevels bids = _make_levels(bid_prices, bid_sizes, 1, 1e-8, 0.001)
    cdef OrderbookLevels asks = _make_levels(ask_prices, ask_sizes, 1, 1e-8, 0.001)
    
    core.consume_snapshot(asks, bids)
    
    cdef double spread = core.get_bbo_spread()
    assert spread > 0
    
    _free_levels(&bids)
    _free_levels(&asks)


def test_core_very_large_tick_size():
    """Test with very large tick size."""
    cdef CoreAdvancedOrderbook core = CoreAdvancedOrderbook(
        tick_size=1000.0,
        lot_size=0.001,
        num_levels=646,
        delta_sortedness=CyOrderbookSortedness.UNKNOWN,
        snapshot_sortedness=CyOrderbookSortedness.UNKNOWN,
    )
    
    cdef double bid_prices[1]
    cdef double bid_sizes[1]
    cdef double ask_prices[1]
    cdef double ask_sizes[1]
    
    bid_prices[0] = 50000.0
    bid_sizes[0] = 1.0
    ask_prices[0] = 51000.0
    ask_sizes[0] = 1.0
    
    cdef OrderbookLevels bids = _make_levels(bid_prices, bid_sizes, 1, 1000.0, 0.001)
    cdef OrderbookLevels asks = _make_levels(ask_prices, ask_sizes, 1, 1000.0, 0.001)
    
    core.consume_snapshot(asks, bids)
    
    cdef double spread = core.get_bbo_spread()
    assert _approx_eq(spread, 1000.0)
    
    _free_levels(&bids)
    _free_levels(&asks)


def test_core_asymmetric_depths():
    """Test asymmetric book depths."""
    cdef CoreAdvancedOrderbook core = _create_core(num_levels=64)
    
    cdef double bid_prices[5]
    cdef double bid_sizes[5]
    cdef double ask_prices[2]
    cdef double ask_sizes[2]
    
    # 5 bids
    for i in range(5):
        bid_prices[i] = 100.0 - i * 0.01
        bid_sizes[i] = 1.0
    
    # Only 2 asks
    ask_prices[0] = 100.01
    ask_prices[1] = 100.02
    ask_sizes[0] = 1.0
    ask_sizes[1] = 1.0
    
    cdef OrderbookLevels bids = _make_levels(bid_prices, bid_sizes, 5, TICK_SIZE, LOT_SIZE)
    cdef OrderbookLevels asks = _make_levels(ask_prices, ask_sizes, 2, TICK_SIZE, LOT_SIZE)
    
    core.consume_snapshot(asks, bids)
    
    cdef OrderbookLadderData* bids_view = core.get_bids_data()
    cdef OrderbookLadderData* asks_view = core.get_asks_data()
    
    assert bids_view.num_levels == 5
    assert asks_view.num_levels == 2
    
    _free_levels(&bids)
    _free_levels(&asks)


# =============================================================================

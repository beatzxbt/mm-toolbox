# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Layer 2: OrderbookLadder tests.

Tests ladder initialization, insert/roll operations, state management,
and logical ring behavior.
"""
from __future__ import annotations

from libc.stdint cimport uint64_t as u64
from libc.math cimport fabs

from mm_toolbox.orderbook.advanced.level.level cimport (
    OrderbookEntry,
    create_orderbook_entry,
)
from mm_toolbox.orderbook.advanced.level.helpers cimport (
    convert_price_from_tick,
    convert_price_to_tick,
    convert_size_from_lot,
    convert_size_to_lot,
)
from mm_toolbox.orderbook.advanced.ladder.ladder cimport (
    OrderbookLadder,
    OrderbookLadderData,
    c_ladder_at,
)


# =============================================================================
# Test Constants
# =============================================================================
DEF TICK_SIZE = 0.01
DEF LOT_SIZE = 0.001


# =============================================================================
# Helper Functions
# =============================================================================
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


cdef OrderbookEntry _make_entry(
    double price,
    double size,
    double tick_size,
    double lot_size,
    u64 norders=1,
):
    return create_orderbook_entry(
        convert_price_to_tick(price, 1.0 / tick_size),
        convert_size_to_lot(size, 1.0 / lot_size),
        norders,
    )


cdef double _entry_price(OrderbookLadderData* data, u64 index):
    return convert_price_from_tick(c_ladder_at(data, index).ticks, TICK_SIZE)


cdef double _entry_size(OrderbookLadderData* data, u64 index):
    return convert_size_from_lot(c_ladder_at(data, index).lots, LOT_SIZE)


# =============================================================================
# LAYER 2: OrderbookLadder
# =============================================================================

# -----------------------------------------------------------------------------
# Initialization tests
# -----------------------------------------------------------------------------

def test_ladder_init_basic():
    """Test basic ladder initialization."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=10, is_price_ascending=True)
    assert ladder.is_empty()
    assert not ladder.is_full()


def test_ladder_init_ascending():
    """Test ladder with ascending prices (asks)."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.is_price_ascending == True
    assert data.max_levels == 5


def test_ladder_init_descending():
    """Test ladder with descending prices (bids)."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=False)
    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.is_price_ascending == False


def test_ladder_init_single_level():
    """Test ladder with single level capacity."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=1, is_price_ascending=True)
    assert ladder.is_empty()


def test_ladder_init_large():
    """Test ladder with large capacity."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=10000, is_price_ascending=True)
    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.max_levels == 10000


# -----------------------------------------------------------------------------
# insert_entry tests
# -----------------------------------------------------------------------------

def test_ladder_insert_entry():
    """Test inserting an entry."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookEntry entry = _make_entry(
        100.0, 1.0, TICK_SIZE, LOT_SIZE, 1
    )

    ladder.insert_entry(0, &entry)
    ladder.increment_count()

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 1
    assert _entry_price(data, 0) == 100.0


def test_ladder_insert_multiple():
    """Test inserting multiple levels."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookEntry entry
    cdef u64 i

    for i in range(3):
        entry = _make_entry(
            100.0 + i * 0.01, 1.0, TICK_SIZE, LOT_SIZE, 1
        )
        ladder.insert_entry(i, &entry)
        ladder.increment_count()

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 3
    assert _entry_price(data, 0) == 100.0
    assert _entry_price(data, 1) == 100.01
    assert _entry_price(data, 2) == 100.02


def test_ladder_accessor_methods():
    """Test C-level count, capacity, at, top, and bottom accessors."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookEntry entry
    cdef OrderbookEntry* entry_ptr

    assert ladder.count() == 0
    assert ladder.capacity() == 5

    entry = _make_entry(100.0, 1.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_entry(0, &entry)
    ladder.increment_count()
    entry = _make_entry(100.01, 2.0, TICK_SIZE, LOT_SIZE, 2)
    ladder.insert_entry(1, &entry)
    ladder.increment_count()

    entry_ptr = ladder.top()
    assert entry_ptr.ticks == convert_price_to_tick(100.0, 1.0 / TICK_SIZE)
    assert entry_ptr.norders == 1

    entry_ptr = ladder.at(1)
    assert entry_ptr.lots == convert_size_to_lot(2.0, 1.0 / LOT_SIZE)

    entry_ptr = ladder.bottom()
    assert entry_ptr.ticks == convert_price_to_tick(100.01, 1.0 / TICK_SIZE)
    assert entry_ptr.norders == 2


def test_ladder_assign_entry():
    """Test C-level entry assignment helper."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookEntry entry = _make_entry(100.0, 1.0, TICK_SIZE, LOT_SIZE, 1)
    cdef OrderbookEntry* entry_ptr

    ladder.assign_entry(0, &entry)
    ladder.set_count(1)

    entry_ptr = ladder.top()
    assert entry_ptr.ticks == entry.ticks
    assert entry_ptr.lots == entry.lots
    assert entry_ptr.norders == 1
    assert ladder.count() == 1


def test_ladder_seek_start_ascending():
    """Test ask-side insertion seek uses ascending price order."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookEntry entry
    cdef u64 i

    for i in range(3):
        entry = _make_entry(100.0 + i * 0.01, 1.0, TICK_SIZE, LOT_SIZE, 1)
        ladder.insert_entry(i, &entry)
        ladder.increment_count()

    assert ladder.seek_start(convert_price_to_tick(99.99, 1.0 / TICK_SIZE)) == 0
    assert ladder.seek_start(convert_price_to_tick(100.01, 1.0 / TICK_SIZE)) == 1
    assert ladder.seek_start(convert_price_to_tick(100.03, 1.0 / TICK_SIZE)) == 3


def test_ladder_seek_start_descending():
    """Test bid-side insertion seek uses descending price order."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=False)
    cdef OrderbookEntry entry
    cdef u64 i

    for i in range(3):
        entry = _make_entry(100.02 - i * 0.01, 1.0, TICK_SIZE, LOT_SIZE, 1)
        ladder.insert_entry(i, &entry)
        ladder.increment_count()

    assert ladder.seek_start(convert_price_to_tick(100.03, 1.0 / TICK_SIZE)) == 0
    assert ladder.seek_start(convert_price_to_tick(100.01, 1.0 / TICK_SIZE)) == 1
    assert ladder.seek_start(convert_price_to_tick(99.99, 1.0 / TICK_SIZE)) == 3


# -----------------------------------------------------------------------------
# roll_right tests
# -----------------------------------------------------------------------------

def test_ladder_roll_right_at_start():
    """Test rolling right from index 0."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookEntry entry

    # Add two entries: [100, 101]
    entry = _make_entry(100.0, 1.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_entry(0, &entry)
    ladder.increment_count()

    entry = _make_entry(101.0, 1.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_entry(1, &entry)
    ladder.increment_count()

    # Roll right at 0: makes room for new level at front
    ladder.roll_right(0)

    # Insert new level at 0
    entry = _make_entry(99.0, 1.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_entry(0, &entry)
    ladder.increment_count()

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 3
    assert _entry_price(data, 0) == 99.0
    assert _entry_price(data, 1) == 100.0
    assert _entry_price(data, 2) == 101.0


def test_ladder_roll_right_in_middle():
    """Test rolling right from middle index."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookEntry entry

    # Add three entries: [100, 101, 102]
    for i in range(3):
        entry = _make_entry(
            100.0 + i * 0.01, 1.0, TICK_SIZE, LOT_SIZE, 1
        )
        ladder.insert_entry(i, &entry)
        ladder.increment_count()

    # Roll right at 1
    ladder.roll_right(1)

    # Insert at 1
    entry = _make_entry(100.005, 2.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_entry(1, &entry)
    ladder.increment_count()

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 4
    assert _entry_price(data, 0) == 100.0
    assert _entry_size(data, 1) == 2.0  # New level
    assert data.head == 4


def test_ladder_roll_right_uses_prefix_shift():
    """Test rolling right from a front-side middle index."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=8, is_price_ascending=True)
    cdef OrderbookEntry entry
    cdef OrderbookLadderData* data = ladder.get_data()
    data.head = 6

    for i in range(5):
        entry = _make_entry(
            100.0 + i * 0.01, 1.0, TICK_SIZE, LOT_SIZE, 1
        )
        ladder.insert_entry(i, &entry)
        ladder.increment_count()

    ladder.roll_right(2)

    entry = _make_entry(150.0, 2.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_entry(2, &entry)
    ladder.increment_count()

    assert data.num_levels == 6
    assert data.head == 5
    assert _entry_price(data, 0) == 100.0
    assert _entry_price(data, 1) == 100.01
    assert _entry_size(data, 2) == 2.0
    assert _entry_price(data, 3) == 100.02
    assert _entry_price(data, 4) == 100.03
    assert _entry_price(data, 5) == 100.04


def test_ladder_roll_right_at_max_capacity():
    """Test rolling right when at max capacity drops last element."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=3, is_price_ascending=True)
    cdef OrderbookEntry entry

    # Fill to capacity: [100.0, 100.01, 100.02]
    for i in range(3):
        entry = _make_entry(
            100.0 + i * 0.01, 1.0, TICK_SIZE, LOT_SIZE, 1
        )
        ladder.insert_entry(i, &entry)
        ladder.increment_count()

    assert ladder.is_full()

    # Roll right at 0 (should drop 100.02)
    ladder.roll_right(0)
    entry = _make_entry(99.0, 2.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_entry(0, &entry)
    # Don't increment - we're replacing dropped element

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 3
    assert _entry_price(data, 0) == 99.0
    assert _entry_price(data, 1) == 100.0
    assert _approx_eq(_entry_price(data, 2), 100.01)  # 100.02 was dropped


# -----------------------------------------------------------------------------
# roll_left tests
# -----------------------------------------------------------------------------

def test_ladder_roll_left_at_start():
    """Test rolling left from index 0 (removes first element)."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookEntry entry

    # Add three entries: [100, 101, 102]
    for i in range(3):
        entry = _make_entry(
            100.0 + i * 0.01, 1.0, TICK_SIZE, LOT_SIZE, 1
        )
        ladder.insert_entry(i, &entry)
        ladder.increment_count()

    # Roll left at 0 removes first element
    ladder.roll_left(0)
    ladder.decrement_count()

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 2
    assert _entry_price(data, 0) == 100.01
    assert _entry_price(data, 1) == 100.02


def test_ladder_roll_left_in_middle():
    """Test rolling left from middle index."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookEntry entry

    # Add three entries: [100, 101, 102]
    for i in range(3):
        entry = _make_entry(
            100.0 + i * 0.01, 1.0, TICK_SIZE, LOT_SIZE, 1
        )
        ladder.insert_entry(i, &entry)
        ladder.increment_count()

    # Roll left at 1 removes middle element
    ladder.roll_left(1)
    ladder.decrement_count()

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 2
    assert _entry_price(data, 0) == 100.0
    assert _entry_price(data, 1) == 100.02


def test_ladder_roll_left_uses_prefix_shift():
    """Test rolling left from a front-side middle index."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=8, is_price_ascending=True)
    cdef OrderbookEntry entry
    cdef OrderbookLadderData* data = ladder.get_data()
    data.head = 6

    for i in range(5):
        entry = _make_entry(
            100.0 + i * 0.01, 1.0, TICK_SIZE, LOT_SIZE, 1
        )
        ladder.insert_entry(i, &entry)
        ladder.increment_count()

    ladder.roll_left(1)
    ladder.decrement_count()

    assert data.num_levels == 4
    assert data.head == 7
    assert _entry_price(data, 0) == 100.0
    assert _entry_price(data, 1) == 100.02
    assert _entry_price(data, 2) == 100.03
    assert _entry_price(data, 3) == 100.04


def test_ladder_roll_left_at_end():
    """Test rolling left from last index."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookEntry entry

    # Add three levels
    for i in range(3):
        entry = _make_entry(
            100.0 + i * 0.01, 1.0, TICK_SIZE, LOT_SIZE, 1
        )
        ladder.insert_entry(i, &entry)
        ladder.increment_count()

    # Roll left at last index
    ladder.roll_left(2)
    ladder.decrement_count()

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 2
    assert _entry_price(data, 0) == 100.0
    assert _entry_price(data, 1) == 100.01


# -----------------------------------------------------------------------------
# reset tests
# -----------------------------------------------------------------------------

def test_ladder_reset():
    """Test resetting ladder to empty."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookEntry entry

    # Add levels
    for i in range(3):
        entry = _make_entry(100.0 + i, 1.0, TICK_SIZE, LOT_SIZE, 1)
        ladder.insert_entry(i, &entry)
        ladder.increment_count()

    assert not ladder.is_empty()

    ladder.reset()

    assert ladder.is_empty()
    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 0


# -----------------------------------------------------------------------------
# is_empty / is_full tests
# -----------------------------------------------------------------------------

def test_ladder_is_empty():
    """Test is_empty on fresh ladder."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    assert ladder.is_empty()


def test_ladder_not_empty_after_insert():
    """Test is_empty after insert."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookEntry entry = _make_entry(100.0, 1.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_entry(0, &entry)
    ladder.increment_count()
    assert not ladder.is_empty()


def test_ladder_is_full():
    """Test is_full at capacity."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=2, is_price_ascending=True)
    cdef OrderbookEntry entry

    entry = _make_entry(100.0, 1.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_entry(0, &entry)
    ladder.increment_count()
    assert not ladder.is_full()

    entry = _make_entry(101.0, 1.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_entry(1, &entry)
    ladder.increment_count()
    assert ladder.is_full()


# -----------------------------------------------------------------------------
# increment/decrement count tests
# -----------------------------------------------------------------------------

def test_ladder_increment_count_respects_max():
    """Test increment_count doesn't exceed max."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=2, is_price_ascending=True)

    ladder.increment_count()
    ladder.increment_count()
    ladder.increment_count()  # Should be capped

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 2


def test_ladder_decrement_count_respects_zero():
    """Test decrement_count doesn't go below zero."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=2, is_price_ascending=True)

    ladder.decrement_count()  # Already at 0
    ladder.decrement_count()

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 0


# -----------------------------------------------------------------------------
# get_data tests
# -----------------------------------------------------------------------------

def test_ladder_get_data():
    """Test get_data returns valid pointer."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookLadderData* data = ladder.get_data()

    assert data != NULL
    assert data.max_levels == 5
    assert data.num_levels == 0


def test_ladder_data_reflects_changes():
    """Test data pointer reflects ladder mutations."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookLadderData* data = ladder.get_data()

    assert data.num_levels == 0

    cdef OrderbookEntry entry = _make_entry(100.0, 1.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_entry(0, &entry)
    ladder.increment_count()

    # Data should reflect change
    data = ladder.get_data()
    assert data.num_levels == 1

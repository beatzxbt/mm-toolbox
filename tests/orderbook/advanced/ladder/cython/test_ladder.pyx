# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Layer 2: OrderbookLadder tests.

Tests ladder initialization, insert/roll operations, state management,
and NumPy accessor methods.
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


# =============================================================================
# Test Constants
# =============================================================================
DEF TICK_SIZE = 0.01
DEF LOT_SIZE = 0.001


# =============================================================================
# Helper Functions
# =============================================================================
cdef bint _approx_eq(double a, double b, double tol=1e-9):
    """Check if two doubles are approximately equal."""
    return fabs(a - b) < tol


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
# insert_level tests
# -----------------------------------------------------------------------------

def test_ladder_insert_level():
    """Test inserting a level."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookLevel level = create_orderbook_level_with_ticks_and_lots(
        100.0, 1.0, TICK_SIZE, LOT_SIZE, 1
    )

    ladder.insert_level(0, level)
    ladder.increment_count()

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 1
    assert data.levels[0].price == 100.0


def test_ladder_insert_multiple():
    """Test inserting multiple levels."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookLevel level
    cdef u64 i

    for i in range(3):
        level = create_orderbook_level_with_ticks_and_lots(
            100.0 + i * 0.01, 1.0, TICK_SIZE, LOT_SIZE, 1
        )
        ladder.insert_level(i, level)
        ladder.increment_count()

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 3
    assert data.levels[0].price == 100.0
    assert data.levels[1].price == 100.01
    assert data.levels[2].price == 100.02


# -----------------------------------------------------------------------------
# roll_right tests
# -----------------------------------------------------------------------------

def test_ladder_roll_right_at_start():
    """Test rolling right from index 0."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookLevel level

    # Add two levels: [100, 101]
    level = create_orderbook_level_with_ticks_and_lots(100.0, 1.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_level(0, level)
    ladder.increment_count()

    level = create_orderbook_level_with_ticks_and_lots(101.0, 1.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_level(1, level)
    ladder.increment_count()

    # Roll right at 0: makes room for new level at front
    ladder.roll_right(0)

    # Insert new level at 0
    level = create_orderbook_level_with_ticks_and_lots(99.0, 1.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_level(0, level)
    ladder.increment_count()

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 3
    assert data.levels[0].price == 99.0
    assert data.levels[1].price == 100.0
    assert data.levels[2].price == 101.0


def test_ladder_roll_right_in_middle():
    """Test rolling right from middle index."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookLevel level

    # Add three levels: [100, 101, 102]
    for i in range(3):
        level = create_orderbook_level_with_ticks_and_lots(
            100.0 + i * 0.01, 1.0, TICK_SIZE, LOT_SIZE, 1
        )
        ladder.insert_level(i, level)
        ladder.increment_count()

    # Roll right at 1
    ladder.roll_right(1)

    # Insert at 1
    level = create_orderbook_level_with_ticks_and_lots(100.005, 2.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_level(1, level)
    ladder.increment_count()

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 4
    assert data.levels[0].price == 100.0
    assert data.levels[1].size == 2.0  # New level


def test_ladder_roll_right_at_max_capacity():
    """Test rolling right when at max capacity drops last element."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=3, is_price_ascending=True)
    cdef OrderbookLevel level

    # Fill to capacity: [100.0, 100.01, 100.02]
    for i in range(3):
        level = create_orderbook_level_with_ticks_and_lots(
            100.0 + i * 0.01, 1.0, TICK_SIZE, LOT_SIZE, 1
        )
        ladder.insert_level(i, level)
        ladder.increment_count()

    assert ladder.is_full()

    # Roll right at 0 (should drop 100.02)
    ladder.roll_right(0)
    level = create_orderbook_level_with_ticks_and_lots(99.0, 2.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_level(0, level)
    # Don't increment - we're replacing dropped element

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 3
    assert data.levels[0].price == 99.0
    assert data.levels[1].price == 100.0
    assert _approx_eq(data.levels[2].price, 100.01)  # 100.02 was dropped


# -----------------------------------------------------------------------------
# roll_left tests
# -----------------------------------------------------------------------------

def test_ladder_roll_left_at_start():
    """Test rolling left from index 0 (removes first element)."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookLevel level

    # Add three levels: [100, 101, 102]
    for i in range(3):
        level = create_orderbook_level_with_ticks_and_lots(
            100.0 + i * 0.01, 1.0, TICK_SIZE, LOT_SIZE, 1
        )
        ladder.insert_level(i, level)
        ladder.increment_count()

    # Roll left at 0 removes first element
    ladder.roll_left(0)
    ladder.decrement_count()

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 2
    assert data.levels[0].price == 100.01
    assert data.levels[1].price == 100.02


def test_ladder_roll_left_in_middle():
    """Test rolling left from middle index."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookLevel level

    # Add three levels: [100, 101, 102]
    for i in range(3):
        level = create_orderbook_level_with_ticks_and_lots(
            100.0 + i * 0.01, 1.0, TICK_SIZE, LOT_SIZE, 1
        )
        ladder.insert_level(i, level)
        ladder.increment_count()

    # Roll left at 1 removes middle element
    ladder.roll_left(1)
    ladder.decrement_count()

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 2
    assert data.levels[0].price == 100.0
    assert data.levels[1].price == 100.02


def test_ladder_roll_left_at_end():
    """Test rolling left from last index."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookLevel level

    # Add three levels
    for i in range(3):
        level = create_orderbook_level_with_ticks_and_lots(
            100.0 + i * 0.01, 1.0, TICK_SIZE, LOT_SIZE, 1
        )
        ladder.insert_level(i, level)
        ladder.increment_count()

    # Roll left at last index
    ladder.roll_left(2)
    ladder.decrement_count()

    cdef OrderbookLadderData* data = ladder.get_data()
    assert data.num_levels == 2
    assert data.levels[0].price == 100.0
    assert data.levels[1].price == 100.01


# -----------------------------------------------------------------------------
# reset tests
# -----------------------------------------------------------------------------

def test_ladder_reset():
    """Test resetting ladder to empty."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookLevel level

    # Add levels
    for i in range(3):
        level = create_orderbook_level(100.0 + i, 1.0)
        ladder.insert_level(i, level)
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
    cdef OrderbookLevel level = create_orderbook_level(100.0, 1.0)
    ladder.insert_level(0, level)
    ladder.increment_count()
    assert not ladder.is_empty()


def test_ladder_is_full():
    """Test is_full at capacity."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=2, is_price_ascending=True)
    cdef OrderbookLevel level

    level = create_orderbook_level(100.0, 1.0)
    ladder.insert_level(0, level)
    ladder.increment_count()
    assert not ladder.is_full()

    level = create_orderbook_level(101.0, 1.0)
    ladder.insert_level(1, level)
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
# get_data tests (replacing get_view)
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

    cdef OrderbookLevel level = create_orderbook_level(100.0, 1.0)
    ladder.insert_level(0, level)
    ladder.increment_count()

    # Data should reflect change
    data = ladder.get_data()
    assert data.num_levels == 1


# -----------------------------------------------------------------------------
# NumPy accessor tests (cpdef methods)
# -----------------------------------------------------------------------------

def test_ladder_get_levels():
    """Test get_levels returns NumPy view."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookLevel level

    for i in range(3):
        level = create_orderbook_level_with_ticks_and_lots(
            100.0 + i * 0.01, float(i + 1), TICK_SIZE, LOT_SIZE, 1
        )
        ladder.insert_level(i, level)
        ladder.increment_count()

    levels = ladder.get_levels()
    assert len(levels) == 3


def test_ladder_get_prices():
    """Test get_prices returns price array."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookLevel level

    level = create_orderbook_level_with_ticks_and_lots(100.0, 1.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_level(0, level)
    ladder.increment_count()

    level = create_orderbook_level_with_ticks_and_lots(101.0, 2.0, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_level(1, level)
    ladder.increment_count()

    prices = ladder.get_prices()
    assert len(prices) == 2
    assert prices[0] == 100.0
    assert prices[1] == 101.0


def test_ladder_get_sizes():
    """Test get_sizes returns size array."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)
    cdef OrderbookLevel level

    level = create_orderbook_level_with_ticks_and_lots(100.0, 1.5, TICK_SIZE, LOT_SIZE, 1)
    ladder.insert_level(0, level)
    ladder.increment_count()

    sizes = ladder.get_sizes()
    assert len(sizes) == 1
    assert sizes[0] == 1.5


def test_ladder_empty_accessors():
    """Test accessors on empty ladder."""
    cdef OrderbookLadder ladder = OrderbookLadder(max_levels=5, is_price_ascending=True)

    assert len(ladder.get_levels()) == 0
    assert len(ladder.get_prices()) == 0
    assert len(ladder.get_sizes()) == 0
    assert len(ladder.get_norders()) == 0

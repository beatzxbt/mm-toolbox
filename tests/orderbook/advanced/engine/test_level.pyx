# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Layer 1: Primitive tests for OrderbookLevel and helper functions.

Tests OrderbookLevel creation, conversion functions (price↔tick, size↔lot),
and level manipulation utilities (swap, reverse, sort).
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
    create_orderbook_levels,
    free_orderbook_levels,
)
from mm_toolbox.orderbook.advanced.level.helpers cimport (
    convert_price_to_tick,
    convert_price_from_tick,
    convert_size_to_lot,
    convert_size_from_lot,
    swap_levels,
    reverse_levels,
    inplace_sort_levels_by_ticks,
)


# =============================================================================
# Test Constants
# =============================================================================
DEF TICK_SIZE = 0.01
DEF LOT_SIZE = 0.001


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
    """Create OrderbookLevels from arrays with tick/lot conversion."""
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


# =============================================================================
# LAYER 1: Primitives - OrderbookLevel and Helpers
# =============================================================================

# -----------------------------------------------------------------------------
# OrderbookLevel struct tests
# -----------------------------------------------------------------------------

def test_create_orderbook_level_basic():
    """Test basic OrderbookLevel creation."""
    cdef OrderbookLevel level = create_orderbook_level(100.0, 1.5, 5)
    assert level.price == 100.0
    assert level.size == 1.5
    assert level.norders == 5
    assert level.ticks == 0  # Not computed without tick_size
    assert level.lots == 0   # Not computed without lot_size


def test_create_orderbook_level_default_norders():
    """Test OrderbookLevel with default norders=1."""
    cdef OrderbookLevel level = create_orderbook_level(50.0, 2.0)
    assert level.price == 50.0
    assert level.size == 2.0
    assert level.norders == 1


def test_create_orderbook_level_zero_price():
    """Test OrderbookLevel with zero price."""
    cdef OrderbookLevel level = create_orderbook_level(0.0, 1.0, 1)
    assert level.price == 0.0
    assert level.size == 1.0


def test_create_orderbook_level_zero_size():
    """Test OrderbookLevel with zero size (deletion marker)."""
    cdef OrderbookLevel level = create_orderbook_level(100.0, 0.0, 1)
    assert level.price == 100.0
    assert level.size == 0.0


def test_create_orderbook_level_large_values():
    """Test OrderbookLevel with large values."""
    cdef OrderbookLevel level = create_orderbook_level(999999.99, 1000000.0, 1000000)
    assert level.price == 999999.99
    assert level.size == 1000000.0
    assert level.norders == 1000000


def test_create_orderbook_level_with_ticks_and_lots_basic():
    """Test OrderbookLevel creation with tick/lot conversion."""
    cdef OrderbookLevel level = create_orderbook_level_with_ticks_and_lots(
        100.01, 1.5, TICK_SIZE, LOT_SIZE, 3
    )
    assert level.price == 100.01
    assert level.size == 1.5
    assert level.norders == 3
    assert level.ticks == 10001  # 100.01 / 0.01
    assert level.lots == 1500    # 1.5 / 0.001


def test_create_orderbook_level_with_ticks_and_lots_default_norders():
    """Test tick/lot conversion with default norders."""
    cdef OrderbookLevel level = create_orderbook_level_with_ticks_and_lots(
        50.00, 0.5, TICK_SIZE, LOT_SIZE
    )
    assert level.norders == 1
    assert level.ticks == 5000
    assert level.lots == 500


def test_create_orderbook_level_ticks_rounding():
    """Test tick conversion with various rounding scenarios."""
    cdef OrderbookLevel level
    # Exact multiple
    level = create_orderbook_level_with_ticks_and_lots(100.00, 1.0, 0.01, 0.001)
    assert level.ticks == 10000

    # Slightly above tick
    level = create_orderbook_level_with_ticks_and_lots(100.005, 1.0, 0.01, 0.001)
    assert level.ticks == 10000 or level.ticks == 10001  # Floor or round


def test_create_orderbook_level_very_small_tick():
    """Test with very small tick size."""
    cdef OrderbookLevel level = create_orderbook_level_with_ticks_and_lots(
        0.00012345, 100.0, 0.00000001, 0.001
    )
    # Floating point precision may give 12344 or 12345
    assert level.ticks == 12344 or level.ticks == 12345
    assert _approx_eq(level.price, 0.00012345, 1e-10)


def test_create_orderbook_level_large_tick():
    """Test with large tick size."""
    cdef OrderbookLevel level = create_orderbook_level_with_ticks_and_lots(
        50000.0, 1.0, 1000.0, 0.001
    )
    assert level.ticks == 50


# -----------------------------------------------------------------------------
# Helper function tests - Tick/Lot conversion
# -----------------------------------------------------------------------------

def test_convert_price_to_tick_basic():
    """Test price to tick conversion."""
    cdef u64 ticks = convert_price_to_tick(100.01, 0.01)
    assert ticks == 10001


def test_convert_price_from_tick_basic():
    """Test tick to price conversion."""
    cdef double price = convert_price_from_tick(10001, 0.01)
    assert _approx_eq(price, 100.01)


def test_tick_conversion_roundtrip():
    """Test price -> tick -> price roundtrip."""
    cdef double original = 123.45
    cdef u64 ticks = convert_price_to_tick(original, 0.01)
    cdef double recovered = convert_price_from_tick(ticks, 0.01)
    assert _approx_eq(original, recovered)


def test_convert_size_to_lot_basic():
    """Test size to lot conversion."""
    cdef u64 lots = convert_size_to_lot(1.5, 0.001)
    assert lots == 1500


def test_convert_size_from_lot_basic():
    """Test lot to size conversion."""
    cdef double size = convert_size_from_lot(1500, 0.001)
    assert _approx_eq(size, 1.5)


def test_lot_conversion_roundtrip():
    """Test size -> lot -> size roundtrip."""
    cdef double original = 99.999
    cdef u64 lots = convert_size_to_lot(original, 0.001)
    cdef double recovered = convert_size_from_lot(lots, 0.001)
    assert _approx_eq(original, recovered)


def test_convert_zero_price():
    """Test conversion of zero price."""
    assert convert_price_to_tick(0.0, 0.01) == 0
    assert convert_price_from_tick(0, 0.01) == 0.0


def test_convert_zero_size():
    """Test conversion of zero size."""
    assert convert_size_to_lot(0.0, 0.001) == 0
    assert convert_size_from_lot(0, 0.001) == 0.0


# -----------------------------------------------------------------------------
# Helper function tests - swap_levels
# -----------------------------------------------------------------------------

def test_swap_levels():
    """Test swapping two levels."""
    cdef OrderbookLevel a = create_orderbook_level(100.0, 1.0, 1)
    cdef OrderbookLevel b = create_orderbook_level(200.0, 2.0, 2)

    swap_levels(&a, &b)

    assert a.price == 200.0
    assert a.size == 2.0
    assert b.price == 100.0
    assert b.size == 1.0


# -----------------------------------------------------------------------------
# Helper function tests - reverse_levels
# -----------------------------------------------------------------------------

def test_reverse_levels_basic():
    """Test reversing an array of levels."""
    cdef double prices[3]
    cdef double sizes[3]
    prices[0] = 100.0; prices[1] = 101.0; prices[2] = 102.0
    sizes[0] = 1.0; sizes[1] = 2.0; sizes[2] = 3.0

    cdef OrderbookLevels levels = _make_levels(prices, sizes, 3, TICK_SIZE, LOT_SIZE)

    reverse_levels(levels)

    assert levels.levels[0].price == 102.0
    assert levels.levels[1].price == 101.0
    assert levels.levels[2].price == 100.0

    _free_levels(&levels)


def test_reverse_levels_single():
    """Test reversing single-element array (no-op)."""
    cdef double prices[1]
    cdef double sizes[1]
    prices[0] = 100.0
    sizes[0] = 1.0

    cdef OrderbookLevels levels = _make_levels(prices, sizes, 1, TICK_SIZE, LOT_SIZE)
    reverse_levels(levels)

    assert levels.levels[0].price == 100.0
    _free_levels(&levels)


def test_reverse_levels_even():
    """Test reversing even-length array."""
    cdef double prices[4]
    cdef double sizes[4]
    prices[0] = 1.0; prices[1] = 2.0; prices[2] = 3.0; prices[3] = 4.0
    sizes[0] = 1.0; sizes[1] = 1.0; sizes[2] = 1.0; sizes[3] = 1.0

    cdef OrderbookLevels levels = _make_levels(prices, sizes, 4, TICK_SIZE, LOT_SIZE)
    reverse_levels(levels)

    assert levels.levels[0].price == 4.0
    assert levels.levels[1].price == 3.0
    assert levels.levels[2].price == 2.0
    assert levels.levels[3].price == 1.0

    _free_levels(&levels)


# -----------------------------------------------------------------------------
# Helper function tests - inplace_sort_levels_by_ticks
# -----------------------------------------------------------------------------

def test_sort_levels_ascending():
    """Test sorting levels by ticks ascending."""
    cdef double prices[4]
    cdef double sizes[4]
    prices[0] = 103.0; prices[1] = 101.0; prices[2] = 104.0; prices[3] = 102.0
    sizes[0] = 1.0; sizes[1] = 1.0; sizes[2] = 1.0; sizes[3] = 1.0

    cdef OrderbookLevels levels = _make_levels(prices, sizes, 4, TICK_SIZE, LOT_SIZE)
    inplace_sort_levels_by_ticks(levels, ascending=True)

    assert levels.levels[0].price == 101.0
    assert levels.levels[1].price == 102.0
    assert levels.levels[2].price == 103.0
    assert levels.levels[3].price == 104.0

    _free_levels(&levels)


def test_sort_levels_descending():
    """Test sorting levels by ticks descending."""
    cdef double prices[4]
    cdef double sizes[4]
    prices[0] = 101.0; prices[1] = 103.0; prices[2] = 100.0; prices[3] = 102.0
    sizes[0] = 1.0; sizes[1] = 1.0; sizes[2] = 1.0; sizes[3] = 1.0

    cdef OrderbookLevels levels = _make_levels(prices, sizes, 4, TICK_SIZE, LOT_SIZE)
    inplace_sort_levels_by_ticks(levels, ascending=False)

    assert levels.levels[0].price == 103.0
    assert levels.levels[1].price == 102.0
    assert levels.levels[2].price == 101.0
    assert levels.levels[3].price == 100.0

    _free_levels(&levels)


def test_sort_levels_already_sorted():
    """Test sorting already-sorted levels."""
    cdef double prices[3]
    cdef double sizes[3]
    prices[0] = 100.0; prices[1] = 101.0; prices[2] = 102.0
    sizes[0] = 1.0; sizes[1] = 1.0; sizes[2] = 1.0

    cdef OrderbookLevels levels = _make_levels(prices, sizes, 3, TICK_SIZE, LOT_SIZE)
    inplace_sort_levels_by_ticks(levels, ascending=True)

    assert levels.levels[0].price == 100.0
    assert levels.levels[1].price == 101.0
    assert levels.levels[2].price == 102.0

    _free_levels(&levels)


# -----------------------------------------------------------------------------
# OrderbookLevels struct tests
# -----------------------------------------------------------------------------

def test_create_orderbook_levels():
    """Test creating OrderbookLevels struct."""
    cdef OrderbookLevel* arr = <OrderbookLevel*>malloc(3 * sizeof(OrderbookLevel))
    arr[0] = create_orderbook_level(100.0, 1.0)
    arr[1] = create_orderbook_level(101.0, 2.0)
    arr[2] = create_orderbook_level(102.0, 3.0)

    cdef OrderbookLevels levels = create_orderbook_levels(3, arr)

    assert levels.num_levels == 3
    assert levels.levels[0].price == 100.0
    assert levels.levels[1].price == 101.0
    assert levels.levels[2].price == 102.0

    free(arr)


def test_free_orderbook_levels():
    """Test freeing OrderbookLevels."""
    cdef OrderbookLevels levels = _alloc_levels(5)
    levels.levels[0] = create_orderbook_level(100.0, 1.0)

    free_orderbook_levels(&levels)

    assert levels.levels == NULL
    assert levels.num_levels == 0

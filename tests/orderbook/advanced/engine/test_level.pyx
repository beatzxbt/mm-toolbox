# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Layer 1: Primitive tests for OrderbookLevel and helper functions.

Tests OrderbookLevel creation and conversion functions (price↔tick, size↔lot).
"""
from __future__ import annotations

from libc.stdint cimport uint64_t as u64
from libc.stdlib cimport malloc, free
from libc.math cimport fabs

from mm_toolbox.orderbook.advanced.level.level cimport (
    OrderbookEntry,
    OrderbookLevel,
    OrderbookLevels,
    create_orderbook_entry,
    create_orderbook_level,
    create_orderbook_levels,
    free_orderbook_levels,
)
from mm_toolbox.orderbook.advanced.level.helpers cimport (
    convert_price_to_tick,
    convert_price_from_tick,
    convert_size_to_lot,
    convert_size_from_lot,
    validate_price,
    validate_size,
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


# -----------------------------------------------------------------------------
# OrderbookEntry struct tests
# -----------------------------------------------------------------------------

def test_create_orderbook_entry_basic():
    """Test basic OrderbookEntry creation."""
    cdef OrderbookEntry entry = create_orderbook_entry(10000, 1500, 5)
    assert entry.ticks == 10000
    assert entry.lots == 1500
    assert entry.norders == 5
    assert entry._pad == 0


def test_create_orderbook_entry_zero_ticks():
    """Test OrderbookEntry with zero ticks."""
    cdef OrderbookEntry entry = create_orderbook_entry(0, 1000, 1)
    assert entry.ticks == 0
    assert entry._pad == 0


def test_create_orderbook_entry_zero_lots():
    """Test OrderbookEntry with zero lots."""
    cdef OrderbookEntry entry = create_orderbook_entry(5000, 0, 1)
    assert entry.lots == 0
    assert entry._pad == 0


def test_create_orderbook_entry_large_values():
    """Test OrderbookEntry with large values."""
    cdef OrderbookEntry entry = create_orderbook_entry(9999999999, 1000000000, 1000000)
    assert entry.ticks == 9999999999
    assert entry.lots == 1000000000
    assert entry.norders == 1000000
    assert entry._pad == 0


def test_create_orderbook_entry_pad_always_zero():
    """Test that _pad is always initialized to 0 regardless of other fields."""
    cdef OrderbookEntry entry1 = create_orderbook_entry(1, 1, 1)
    cdef OrderbookEntry entry2 = create_orderbook_entry(0, 0, 0)
    cdef OrderbookEntry entry3 = create_orderbook_entry(99999999, 99999999, 99999999)
    assert entry1._pad == 0
    assert entry2._pad == 0
    assert entry3._pad == 0


# -----------------------------------------------------------------------------
# Helper function tests - Tick/Lot conversion
# -----------------------------------------------------------------------------

def test_convert_price_to_tick_basic():
    """Test price to tick conversion."""
    cdef u64 ticks = convert_price_to_tick(100.01, 1.0 / 0.01)
    assert ticks == 10001


def test_convert_price_from_tick_basic():
    """Test tick to price conversion."""
    cdef double price = convert_price_from_tick(10001, 0.01)
    assert _approx_eq(price, 100.01)


def test_tick_conversion_roundtrip():
    """Test price -> tick -> price roundtrip."""
    cdef double original = 123.45
    cdef u64 ticks = convert_price_to_tick(original, 1.0 / 0.01)
    cdef double recovered = convert_price_from_tick(ticks, 0.01)
    assert _approx_eq(original, recovered)


def test_convert_size_to_lot_basic():
    """Test size to lot conversion."""
    cdef u64 lots = convert_size_to_lot(1.5, 1.0 / 0.001)
    assert lots == 1500


def test_convert_size_from_lot_basic():
    """Test lot to size conversion."""
    cdef double size = convert_size_from_lot(1500, 0.001)
    assert _approx_eq(size, 1.5)


def test_lot_conversion_roundtrip():
    """Test size -> lot -> size roundtrip."""
    cdef double original = 99.999
    cdef u64 lots = convert_size_to_lot(original, 1.0 / 0.001)
    cdef double recovered = convert_size_from_lot(lots, 0.001)
    assert _approx_eq(original, recovered)


def test_convert_zero_price():
    """Test conversion of zero price."""
    assert convert_price_to_tick(0.0, 1.0 / 0.01) == 0
    assert convert_price_from_tick(0, 0.01) == 0.0


def test_convert_zero_size():
    """Test conversion of zero size."""
    assert convert_size_to_lot(0.0, 1.0 / 0.001) == 0
    assert convert_size_from_lot(0, 0.001) == 0.0


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


# -----------------------------------------------------------------------------
# Helper function tests - Validation
# -----------------------------------------------------------------------------

def test_validate_price_valid():
    """Test valid price passes validation."""
    validate_price(100.0)
    validate_price(0.0)
    validate_price(1e10)


def test_validate_price_negative_raises():
    """Test negative price raises ValueError."""
    cdef bint raised = False
    try:
        validate_price(-1.0)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for negative price"


def test_validate_price_nan_raises():
    """Test NaN price raises ValueError."""
    cdef bint raised = False
    try:
        validate_price(float('nan'))
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for NaN price"


def test_validate_price_inf_raises():
    """Test Inf price raises ValueError."""
    cdef bint raised = False
    try:
        validate_price(float('inf'))
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for Inf price"


def test_validate_size_valid():
    """Test valid size passes validation."""
    validate_size(0.0)
    validate_size(1.0)
    validate_size(1e10)


def test_validate_size_negative_raises():
    """Test negative size raises ValueError."""
    cdef bint raised = False
    try:
        validate_size(-1.0)
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for negative size"


def test_validate_size_nan_raises():
    """Test NaN size raises ValueError."""
    cdef bint raised = False
    try:
        validate_size(float('nan'))
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for NaN size"


def test_validate_size_inf_raises():
    """Test Inf size raises ValueError."""
    cdef bint raised = False
    try:
        validate_size(float('inf'))
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for Inf size"

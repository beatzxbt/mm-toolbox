"""Type stubs for helpers.pyx - Conversion helper functions."""

from __future__ import annotations

def py_convert_price_to_tick(price: float, tick_size: float) -> int:
    """Convert a price to ticks using integer arithmetic.

    Args:
        price: The price to convert
        tick_size: The tick size (minimum price increment)

    Returns:
        Number of ticks representing the price (floor division)
    """
    ...

def py_convert_size_to_lot(size: float, lot_size: float) -> int:
    """Convert a size to lots using integer arithmetic.

    Args:
        size: The size to convert
        lot_size: The lot size (minimum size increment)

    Returns:
        Number of lots representing the size (floor division)
    """
    ...

def py_convert_price_from_tick(tick: int, tick_size: float) -> float:
    """Convert ticks back to price.

    Args:
        tick: Number of ticks
        tick_size: The tick size (minimum price increment)

    Returns:
        Price value
    """
    ...

def py_convert_size_from_lot(lot: int, lot_size: float) -> float:
    """Convert lots back to size.

    Args:
        lot: Number of lots
        lot_size: The lot size (minimum size increment)

    Returns:
        Size value
    """
    ...

# Note: The following functions are cdef-only and documented here for reference.
# They must be accessed via cimport from Cython code.
#
# cdef u64 convert_price_to_tick(double price, double tick_size) noexcept nogil
# cdef u64 convert_size_to_lot(double size, double lot_size) noexcept nogil
# cdef double convert_price_from_tick(u64 tick, double tick_size) noexcept nogil
# cdef double convert_size_from_lot(u64 lot, double lot_size) noexcept nogil
# cdef void swap_levels(OrderbookLevel* a, OrderbookLevel* b) noexcept nogil
# cdef void reverse_levels(OrderbookLevels levels) noexcept nogil
# cdef void inplace_sort_levels_by_ticks(OrderbookLevels levels, bint ascending) noexcept nogil

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language = c
# distutils: sources = src/mm_toolbox/orderbook/advanced/c/orderbook_helpers.c
# distutils: include_dirs = src/mm_toolbox/orderbook/advanced/c

"""
Orderbook helper functions for price/size conversion and level manipulation.

This module provides Cython wrappers around C implementations for:
- Price to tick conversion and vice versa
- Size to lot conversion and vice versa
- Level swapping, reversing, and sorting operations
"""
from __future__ import annotations

from libc.stdint cimport uint64_t as u64

from .level cimport OrderbookLevel, OrderbookLevels


# C function declarations from orderbook_helpers.h
cdef extern from "orderbook_helpers.h":
    ctypedef struct OrderbookLevel:
        pass

    u64 price_to_tick(double price, double tick_size) nogil
    u64 price_to_tick_fast(double price, double tick_size_recip) nogil
    u64 size_to_lot(double size, double lot_size) nogil
    u64 size_to_lot_fast(double size, double lot_size_recip) nogil
    double tick_to_price(u64 tick, double tick_size) nogil
    double lot_to_size(u64 lot, double lot_size) nogil
    void swap_levels(OrderbookLevel* a, OrderbookLevel* b) nogil
    void reverse_levels_inplace(u64 num_levels, OrderbookLevel* levels) nogil
    void sort_levels_by_tick(u64 num_levels, OrderbookLevel* levels, bint ascending) nogil


cdef inline u64 convert_price_to_tick(double price, double tick_size) noexcept nogil:
    """Convert a price to ticks using integer arithmetic."""
    return price_to_tick(price, tick_size)


cdef inline u64 convert_price_to_tick_fast(double price, double tick_size_recip) noexcept nogil:
    """Convert a price to ticks using multiplication (faster, uses pre-computed reciprocal)."""
    return price_to_tick_fast(price, tick_size_recip)


cdef inline u64 convert_size_to_lot(double size, double lot_size) noexcept nogil:
    """Convert a size to lots using integer arithmetic."""
    return size_to_lot(size, lot_size)


cdef inline u64 convert_size_to_lot_fast(double size, double lot_size_recip) noexcept nogil:
    """Convert a size to lots using multiplication (faster, uses pre-computed reciprocal)."""
    return size_to_lot_fast(size, lot_size_recip)


cdef inline double convert_price_from_tick(u64 tick, double tick_size) noexcept nogil:
    """Convert ticks back to price."""
    return tick_to_price(tick, tick_size)


cdef inline double convert_size_from_lot(u64 lot, double lot_size) noexcept nogil:
    """Convert lots back to size."""
    return lot_to_size(lot, lot_size)


cdef void reverse_levels(OrderbookLevels levels) noexcept nogil:
    """Reverse the order of levels in-place."""
    reverse_levels_inplace(levels.num_levels, levels.levels)


cdef void inplace_sort_levels_by_ticks(OrderbookLevels levels, bint ascending) noexcept nogil:
    """Sort levels by tick in-place with smart algorithm."""
    sort_levels_by_tick(levels.num_levels, levels.levels, ascending)


# Python-accessible wrappers for conversion functions
cpdef u64 py_convert_price_to_tick(double price, double tick_size):
    """Convert a price to ticks using integer arithmetic (Python-accessible)."""
    return convert_price_to_tick(price, tick_size)


cpdef u64 py_convert_size_to_lot(double size, double lot_size):
    """Convert a size to lots using integer arithmetic (Python-accessible)."""
    return convert_size_to_lot(size, lot_size)


cpdef double py_convert_price_from_tick(u64 tick, double tick_size):
    """Convert ticks back to price (Python-accessible)."""
    return convert_price_from_tick(tick, tick_size)


cpdef double py_convert_size_from_lot(u64 lot, double lot_size):
    """Convert lots back to size (Python-accessible)."""
    return convert_size_from_lot(lot, lot_size)

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language = c
# distutils: sources = src/mm_toolbox/orderbook/advanced/c/orderbook_helpers.c
# distutils: include_dirs = src/mm_toolbox/orderbook/advanced/c

"""Cython-only helpers for normalized orderbook entries."""

from __future__ import annotations

from libc.math cimport isfinite
from libc.stdint cimport uint64_t as u64

from .level cimport OrderbookEntry

cdef extern from "orderbook_helpers.h":
    u64 price_to_tick(double price, double tick_size_recip) nogil
    u64 size_to_lot(double size, double lot_size_recip) nogil
    double tick_to_price(u64 tick, double tick_size) nogil
    double lot_to_size(u64 lot, double lot_size) nogil
    void reverse_entries_c "reverse_entries"(u64 num_entries, OrderbookEntry* entries) nogil


cdef inline u64 convert_price_to_tick(double price, double tick_size_recip) noexcept nogil:
    return price_to_tick(price, tick_size_recip)


cdef inline u64 convert_size_to_lot(double size, double lot_size_recip) noexcept nogil:
    return size_to_lot(size, lot_size_recip)


cdef inline double convert_price_from_tick(u64 tick, double tick_size) noexcept nogil:
    return tick_to_price(tick, tick_size)


cdef inline double convert_size_from_lot(u64 lot, double lot_size) noexcept nogil:
    return lot_to_size(lot, lot_size)


cdef void reverse_entries(OrderbookEntry* entries, u64 num_entries) noexcept nogil:
    reverse_entries_c(num_entries, entries)


cdef void validate_price(double price):
    if price < 0.0 or not isfinite(price):
        raise ValueError(f"Invalid price; expected finite >=0 but got {price}")


cdef void validate_size(double size):
    if size < 0.0 or not isfinite(size):
        raise ValueError(f"Invalid size; expected finite >=0 but got {size}")

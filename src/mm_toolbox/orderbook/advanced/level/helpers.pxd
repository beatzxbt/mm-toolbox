from libc.stdint cimport uint64_t as u64

from .level cimport OrderbookLevel, OrderbookLevels

cdef u64 convert_price_to_tick(double price, double tick_size) noexcept nogil
cdef u64 convert_price_to_tick_fast(double price, double tick_size_recip) noexcept nogil
cdef u64 convert_size_to_lot(double size, double lot_size) noexcept nogil
cdef u64 convert_size_to_lot_fast(double size, double lot_size_recip) noexcept nogil
cdef double convert_price_from_tick(u64 tick, double tick_size) noexcept nogil
cdef double convert_size_from_lot(u64 lot, double lot_size) noexcept nogil

cdef void swap_levels(OrderbookLevel* a, OrderbookLevel* b) noexcept nogil
cdef void reverse_levels(OrderbookLevels levels) noexcept nogil
cdef void inplace_sort_levels_by_ticks(OrderbookLevels levels, bint ascending) noexcept nogil

# Python-accessible wrappers
cpdef u64 py_convert_price_to_tick(double price, double tick_size)
cpdef u64 py_convert_size_to_lot(double size, double lot_size)
cpdef double py_convert_price_from_tick(u64 tick, double tick_size)
cpdef double py_convert_size_from_lot(u64 lot, double lot_size)

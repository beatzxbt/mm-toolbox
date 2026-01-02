from numpy cimport ndarray as cndarray
from libc.stdint cimport uint64_t as u64

from ..level.level cimport OrderbookLevel


cdef struct OrderbookLadderData:
    u64 num_levels
    u64 max_levels
    OrderbookLevel* levels
    bint is_price_ascending


cdef class OrderbookLadder:
    cdef:
        OrderbookLevel* _levels
        cndarray _levels_numpy

        OrderbookLadderData _data

    cdef OrderbookLadderData* get_data(self) noexcept nogil
    cdef void insert_level(self, u64 index, OrderbookLevel level) noexcept nogil
    cdef void roll_right(self, u64 start_index) noexcept nogil
    cdef void roll_left(self, u64 start_index) noexcept nogil
    cdef void reset(self) noexcept nogil
    cdef void increment_count(self) noexcept nogil
    cdef void decrement_count(self) noexcept nogil
    cdef bint is_empty(self) noexcept nogil
    cdef bint is_full(self) noexcept nogil

    cpdef get_levels(self, bint copy=*)
    cpdef get_prices(self, bint copy=*)
    cpdef get_sizes(self, bint copy=*)
    cpdef get_norders(self, bint copy=*)

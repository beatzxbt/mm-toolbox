# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language = c
# distutils: sources = src/mm_toolbox/orderbook/advanced/c/orderbook_ladder.c src/mm_toolbox/orderbook/advanced/c/orderbook_helpers.c
# distutils: include_dirs = src/mm_toolbox/orderbook/advanced/c

"""Internal ring-backed normalized orderbook ladder."""

from __future__ import annotations

from libc.stdint cimport uint64_t as u64
from libc.stdlib cimport free, malloc
from posix.stdlib cimport posix_memalign

from ..level.level cimport OrderbookEntry

cdef extern from "orderbook_types.h":
    u64 ORDERBOOK_MAX_LEVELS

cdef extern from "orderbook_ladder.h":
    ctypedef struct OrderbookLadderData:
        u64 num_levels
        u64 max_levels
        OrderbookEntry* levels
        u64 head
        int is_price_ascending

    OrderbookEntry* c_ladder_at(OrderbookLadderData* data, u64 index) noexcept nogil
    OrderbookEntry* c_ladder_top(OrderbookLadderData* data) noexcept nogil
    OrderbookEntry* c_ladder_bottom(OrderbookLadderData* data) noexcept nogil
    void c_ladder_roll_right(OrderbookLadderData* data, u64 start_index) noexcept nogil
    void c_ladder_roll_left(OrderbookLadderData* data, u64 start_index) noexcept nogil
    void c_ladder_insert_entry(OrderbookLadderData* data, u64 index, const OrderbookEntry* entry) noexcept nogil
    u64 c_ladder_ask_seek_start(OrderbookLadderData* data, u64 ticks) noexcept nogil
    u64 c_ladder_bid_seek_start(OrderbookLadderData* data, u64 ticks) noexcept nogil
    void c_ladder_apply_sorted_deltas(
        OrderbookLadderData* data,
        OrderbookEntry* updates,
        u64 update_count,
        OrderbookEntry* scratch,
    ) noexcept nogil


cdef class OrderbookLadder:
    """Fixed-capacity ladder storing normalized OrderbookEntry values."""

    def __cinit__(self, u64 max_levels, bint is_price_ascending) -> None:
        if max_levels == 0:
            raise ValueError(f"Invalid max_levels; expected >0 but got {max_levels}")
        if max_levels > ORDERBOOK_MAX_LEVELS:
            raise ValueError(
                f"Invalid max_levels; expected <={ORDERBOOK_MAX_LEVELS} but got {max_levels}"
            )

        cdef void* raw_levels
        cdef void* raw_scratch
        cdef int levels_errno = posix_memalign(&raw_levels, 64, max_levels * sizeof(OrderbookEntry))
        if levels_errno != 0:
            raw_levels = malloc(max_levels * sizeof(OrderbookEntry))
            if raw_levels == NULL:
                raise MemoryError(f"Cannot allocate ladder levels; posix_memalign errno: {levels_errno}")

        cdef int scratch_errno = posix_memalign(&raw_scratch, 64, max_levels * sizeof(OrderbookEntry))
        if scratch_errno != 0:
            raw_scratch = malloc(max_levels * sizeof(OrderbookEntry))
            if raw_scratch == NULL:
                free(raw_levels)
                raise MemoryError(f"Cannot allocate ladder scratch; posix_memalign errno: {scratch_errno}")

        self._levels = <OrderbookEntry*>raw_levels
        self._scratch = <OrderbookEntry*>raw_scratch
        self._data.num_levels = 0
        self._data.max_levels = max_levels
        self._data.levels = self._levels
        self._data.head = 0
        self._data.is_price_ascending = is_price_ascending

    def __dealloc__(self):
        if self._levels != NULL:
            free(<void*>self._levels)
        if self._scratch != NULL:
            free(<void*>self._scratch)
        self._levels = NULL
        self._scratch = NULL

    cdef inline OrderbookLadderData* get_data(self) noexcept nogil:
        return &self._data

    cdef inline void insert_entry(self, u64 index, const OrderbookEntry* entry) noexcept nogil:
        c_ladder_insert_entry(&self._data, index, entry)

    cdef inline void assign_entry(self, u64 index, const OrderbookEntry* entry) noexcept nogil:
        c_ladder_insert_entry(&self._data, index, entry)

    cdef inline void apply_sorted_deltas(self, OrderbookEntry* updates, u64 update_count) noexcept nogil:
        c_ladder_apply_sorted_deltas(&self._data, updates, update_count, self._scratch)

    cdef inline void roll_right(self, u64 start_index) noexcept nogil:
        c_ladder_roll_right(&self._data, start_index)

    cdef inline void roll_left(self, u64 start_index) noexcept nogil:
        c_ladder_roll_left(&self._data, start_index)

    cdef inline void reset(self) noexcept nogil:
        self._data.num_levels = 0
        self._data.head = 0

    cdef inline void set_count(self, u64 count) noexcept nogil:
        self._data.num_levels = count if count <= self._data.max_levels else self._data.max_levels

    cdef inline void increment_count(self) noexcept nogil:
        if self._data.num_levels < self._data.max_levels:
            self._data.num_levels += 1

    cdef inline void decrement_count(self) noexcept nogil:
        if self._data.num_levels > 0:
            self._data.num_levels -= 1

    cdef inline bint is_empty(self) noexcept nogil:
        return self._data.num_levels == 0

    cdef inline bint is_full(self) noexcept nogil:
        return self._data.num_levels == self._data.max_levels

    cdef inline u64 count(self) noexcept nogil:
        return self._data.num_levels

    cdef inline u64 capacity(self) noexcept nogil:
        return self._data.max_levels

    cdef inline OrderbookEntry* at(self, u64 index) noexcept nogil:
        return c_ladder_at(&self._data, index)

    cdef inline OrderbookEntry* top(self) noexcept nogil:
        return c_ladder_top(&self._data)

    cdef inline OrderbookEntry* bottom(self) noexcept nogil:
        return c_ladder_bottom(&self._data)

    cdef inline u64 seek_start(self, u64 ticks) noexcept nogil:
        if self._data.is_price_ascending:
            return c_ladder_ask_seek_start(&self._data, ticks)
        return c_ladder_bid_seek_start(&self._data, ticks)

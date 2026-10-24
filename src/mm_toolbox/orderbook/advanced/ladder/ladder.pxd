from libc.stdint cimport uint64_t as u64

from ..level.level cimport OrderbookEntry, OrderbookLevel

cdef extern from "c/orderbook_ladder.h":
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
    void c_ladder_export_levels(
        OrderbookLevel* out,
        OrderbookLadderData* data,
        double tick_size,
        double lot_size,
    ) noexcept nogil
    u64 c_ladder_ask_seek_start(OrderbookLadderData* data, u64 ticks) noexcept nogil
    u64 c_ladder_bid_seek_start(OrderbookLadderData* data, u64 ticks) noexcept nogil
    void c_ladder_apply_sorted_deltas(
        OrderbookLadderData* data,
        OrderbookEntry* updates,
        u64 update_count,
        OrderbookEntry* scratch,
    ) noexcept nogil

cdef class OrderbookLadder:
    cdef:
        OrderbookEntry* _levels
        OrderbookEntry* _scratch
        OrderbookLadderData _data

    cdef OrderbookLadderData* get_data(self) noexcept nogil
    cdef void insert_entry(self, u64 index, const OrderbookEntry* entry) noexcept nogil
    cdef void assign_entry(self, u64 index, const OrderbookEntry* entry) noexcept nogil
    cdef void apply_sorted_deltas(self, OrderbookEntry* updates, u64 update_count) noexcept nogil
    cdef void roll_right(self, u64 start_index) noexcept nogil
    cdef void roll_left(self, u64 start_index) noexcept nogil
    cdef void reset(self) noexcept nogil
    cdef void set_count(self, u64 count) noexcept nogil
    cdef void increment_count(self) noexcept nogil
    cdef void decrement_count(self) noexcept nogil
    cdef bint is_empty(self) noexcept nogil
    cdef bint is_full(self) noexcept nogil
    cdef u64 count(self) noexcept nogil
    cdef u64 capacity(self) noexcept nogil
    cdef OrderbookEntry* at(self, u64 index) noexcept nogil
    cdef OrderbookEntry* top(self) noexcept nogil
    cdef OrderbookEntry* bottom(self) noexcept nogil
    cdef u64 seek_start(self, u64 ticks) noexcept nogil

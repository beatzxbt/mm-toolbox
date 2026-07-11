from libc.stdint cimport uint64_t as u64

from .enum.enums cimport CyOrderbookSortedness
from .ladder.ladder cimport OrderbookLadder, OrderbookLadderData
from .level.level cimport OrderbookEntry, OrderbookLevel, OrderbookLevels


cdef class CoreAdvancedOrderbook:
    cdef:
        double _tick_size
        double _lot_size
        double _tick_size_recip
        double _lot_size_recip
        u64 _max_levels

        CyOrderbookSortedness _delta_sortedness
        CyOrderbookSortedness _snapshot_sortedness
        int _snapshot_ask_order
        int _snapshot_bid_order
        int _delta_ask_order
        int _delta_bid_order

        OrderbookLadder _bids
        OrderbookLadder _asks
        OrderbookLadderData* _bids_data
        OrderbookLadderData* _asks_data

    cdef void _ensure_not_empty(self)
    cdef bint _check_if_empty(self)
    cdef OrderbookLevel _level_from_entry(self, OrderbookEntry* entry)
    cdef OrderbookEntry* _ordered_entry(
        self,
        OrderbookEntry* entries,
        u64 count,
        int source_order,
        int target_order,
        u64 index,
    ) noexcept
    cdef u64 _fill_snapshot_entries(
        self,
        OrderbookEntry* entries,
        u64 count,
        OrderbookLadderData* data,
        int source_order,
        int target_order,
    ) noexcept
    cdef void _normalize_levels(self, OrderbookLevels levels, OrderbookEntry* entries)
    cdef bint _has_nonzero_entries(self, OrderbookEntry* entries, u64 count) noexcept
    cdef bint _should_ignore_crossed_deltas(
        self,
        OrderbookEntry* asks,
        u64 ask_count,
        int ask_order,
        OrderbookEntry* bids,
        u64 bid_count,
        int bid_order,
    ) noexcept
    cdef bint _can_apply_replacement_side(
        self,
        OrderbookLadder ladder,
        OrderbookLadderData* data,
        OrderbookEntry* entries,
        u64 count,
        int source_order,
        int target_order,
    ) noexcept
    cdef void _apply_replacement_side(
        self,
        OrderbookLadder ladder,
        OrderbookLadderData* data,
        OrderbookEntry* entries,
        u64 count,
        int source_order,
        int target_order,
    ) noexcept
    cdef bint _try_apply_replacement_deltas(
        self,
        OrderbookEntry* asks,
        u64 ask_count,
        int ask_order,
        OrderbookEntry* bids,
        u64 bid_count,
        int bid_order,
    ) noexcept
    cdef void _apply_entry_delta(
        self,
        OrderbookLadder ladder,
        OrderbookLadderData* data,
        OrderbookEntry* entry,
    ) noexcept
    cdef void _apply_small_deltas(
        self,
        OrderbookEntry* asks,
        u64 ask_count,
        OrderbookEntry* bids,
        u64 bid_count,
    ) noexcept
    cdef void _remove_crossed_bids(self)
    cdef void _remove_crossed_asks(self)
    cdef void _assign_top(self, OrderbookLadder ladder, OrderbookEntry* entry)

    cdef void clear(self)
    cdef void consume_snapshot_entries(
        self,
        OrderbookEntry* asks,
        u64 ask_count,
        OrderbookEntry* bids,
        u64 bid_count,
    )
    cdef void consume_deltas_entries(
        self,
        OrderbookEntry* asks,
        u64 ask_count,
        OrderbookEntry* bids,
        u64 bid_count,
    )
    cdef void consume_bbo_entries(self, OrderbookEntry ask_entry, OrderbookEntry bid_entry)
    cdef void consume_snapshot(self, OrderbookLevels new_asks, OrderbookLevels new_bids)
    cdef void consume_deltas(self, OrderbookLevels asks, OrderbookLevels bids)
    cdef void consume_bbo(self, OrderbookLevel ask, OrderbookLevel bid)
    cdef void consume_bbo_values(
        self,
        double ask_price,
        double ask_size,
        double bid_price,
        double bid_size,
        u64 ask_norders,
        u64 bid_norders,
    )

    cdef double get_mid_price(self)
    cdef double get_bbo_spread(self)
    cdef double get_wmid_price(self)
    cdef double get_volume_weighted_mid_price(self, double size, bint is_base_currency)
    cdef double get_price_impact(self, double size, bint is_buy, bint is_base_currency)
    cdef double get_size_for_price_impact_bps(
        self,
        double impact_bps,
        bint is_buy,
        bint is_base_currency,
    )
    cdef bint is_bbo_crossed(self, double other_bid_price, double other_ask_price)
    cdef bint does_bbo_price_change(self, double bid_price, double ask_price)
    cdef OrderbookLadderData* get_bids_data(self) noexcept
    cdef OrderbookLadderData* get_asks_data(self) noexcept

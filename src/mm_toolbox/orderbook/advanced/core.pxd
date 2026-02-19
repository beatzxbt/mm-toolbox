cimport cython
from libc.stdint cimport uint64_t as u64

from .level.level cimport OrderbookLevels, OrderbookLevel
from .ladder.ladder cimport OrderbookLadder, OrderbookLadderData
from .enum.enums cimport CyOrderbookSortedness


cdef class CoreAdvancedOrderbook:
    cdef:
        double _tick_size
        double _lot_size
        double _tick_size_recip  # Pre-computed 1.0 / tick_size for fast conversion
        double _lot_size_recip   # Pre-computed 1.0 / lot_size for fast conversion
        u64 _max_levels

        CyOrderbookSortedness _delta_sortedness
        CyOrderbookSortedness _snapshot_sortedness

        OrderbookLadder _bids
        OrderbookLadder _asks
        OrderbookLadderData* _bids_data
        OrderbookLadderData* _asks_data

    # Internal helpers
    cdef void _ensure_not_empty(self)
    cdef bint _check_if_empty(self)
    cdef void _normalize_incoming_levels(self, OrderbookLevels asks, OrderbookLevels bids, bint is_snapshot)
    cdef void _process_matching_ask_ticks(self, OrderbookLevel* ask)
    cdef void _process_matching_bid_ticks(self, OrderbookLevel* bid)
    cdef void _process_lower_ask_ticks(self, OrderbookLevel* ask)
    cdef void _process_higher_bid_ticks(self, OrderbookLevel* bid)
    cdef void _process_middle_ask_ticks(self, OrderbookLevel* ask)
    cdef void _process_middle_bid_ticks(self, OrderbookLevel* bid)
    cdef void _assign_bbo_level(self, OrderbookLevel* target, OrderbookLevel* source, u64 ticks, u64 lots) noexcept nogil

    cdef void clear(self) 
    cdef void consume_snapshot(self, OrderbookLevels new_asks, OrderbookLevels new_bids)
    cdef void consume_deltas(self, OrderbookLevels asks, OrderbookLevels bids)
    cdef void consume_bbo(self, OrderbookLevel ask, OrderbookLevel bid)

    cdef double get_mid_price(self)
    cdef double get_bbo_spread(self)
    cdef double get_wmid_price(self)
    cdef double get_volume_weighted_mid_price(self, double size, bint is_base_currency)
    cdef double get_price_impact(self, double size, bint is_buy, bint is_base_currency)
    cdef double get_size_for_price_impact_bps(self, double impact_bps, bint is_buy, bint is_base_currency)
    cdef bint is_bbo_crossed(self, double other_bid_price, double other_ask_price)
    cdef bint does_bbo_price_change(self, double bid_price, double ask_price)
    cdef OrderbookLadderData* get_bids_data(self) noexcept
    cdef OrderbookLadderData* get_asks_data(self) noexcept 

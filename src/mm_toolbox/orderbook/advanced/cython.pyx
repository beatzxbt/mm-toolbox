# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.stdint cimport uint64_t as u64
from libc.stdlib cimport free, malloc, realloc

from .core cimport CoreAdvancedOrderbook
from .enum.enums cimport CyOrderbookSortedness
from .level.helpers cimport (
    convert_price_to_tick_trusted,
    convert_size_to_lot_trusted,
    validate_price,
    validate_size,
)
from .level.level cimport (
    OrderbookEntry,
    OrderbookLevel,
    OrderbookLevels,
    create_orderbook_entry,
)


cdef class CyAdvancedOrderbook:
    """Cython-facing advanced orderbook wrapper."""

    def __cinit__(
        self,
        double tick_size,
        double lot_size,
        u64 num_levels,
        CyOrderbookSortedness delta_sortedness=CyOrderbookSortedness.UNKNOWN,
        CyOrderbookSortedness snapshot_sortedness=CyOrderbookSortedness.UNKNOWN,
    ):
        self._core = CoreAdvancedOrderbook(
            tick_size,
            lot_size,
            num_levels,
            delta_sortedness,
            snapshot_sortedness,
        )
        self._entry_buffer_capacity = num_levels if num_levels > 0 else 1
        self._ask_entry_buffer = <OrderbookEntry*>malloc(
            self._entry_buffer_capacity * sizeof(OrderbookEntry)
        )
        if self._ask_entry_buffer == NULL:
            raise MemoryError("Failed to allocate ask entry buffer")
        self._bid_entry_buffer = <OrderbookEntry*>malloc(
            self._entry_buffer_capacity * sizeof(OrderbookEntry)
        )
        if self._bid_entry_buffer == NULL:
            free(self._ask_entry_buffer)
            self._ask_entry_buffer = NULL
            raise MemoryError("Failed to allocate bid entry buffer")

    def __dealloc__(self):
        if self._ask_entry_buffer != NULL:
            free(self._ask_entry_buffer)
        if self._bid_entry_buffer != NULL:
            free(self._bid_entry_buffer)
        self._ask_entry_buffer = NULL
        self._bid_entry_buffer = NULL

    cdef void _ensure_entry_capacity(self, u64 required):
        cdef:
            u64 new_capacity
            void* new_asks
            void* new_bids

        if required <= self._entry_buffer_capacity:
            return

        new_capacity = self._entry_buffer_capacity
        while new_capacity < required:
            new_capacity *= 2

        new_asks = realloc(self._ask_entry_buffer, new_capacity * sizeof(OrderbookEntry))
        if new_asks == NULL:
            raise MemoryError("Failed to grow ask entry buffer")
        self._ask_entry_buffer = <OrderbookEntry*>new_asks

        new_bids = realloc(self._bid_entry_buffer, new_capacity * sizeof(OrderbookEntry))
        if new_bids == NULL:
            raise MemoryError("Failed to grow bid entry buffer")
        self._bid_entry_buffer = <OrderbookEntry*>new_bids
        self._entry_buffer_capacity = new_capacity

    cdef void _normalize_levels_to_entries(
        self,
        OrderbookLevel* levels,
        u64 count,
        OrderbookEntry* entries,
    ):
        cdef u64 i
        for i in range(count):
            validate_price(levels[i].price)
            validate_size(levels[i].size)
            entries[i] = create_orderbook_entry(
                convert_price_to_tick_trusted(levels[i].price, self._core._tick_size_recip),
                convert_size_to_lot_trusted(levels[i].size, self._core._lot_size_recip),
                levels[i].norders,
            )

    cdef void clear(self):
        self._core.clear()

    cdef void consume_snapshot(self, OrderbookLevels asks, OrderbookLevels bids):
        cdef u64 required = (
            asks.num_levels if asks.num_levels >= bids.num_levels else bids.num_levels
        )
        self._ensure_entry_capacity(required)
        self._normalize_levels_to_entries(
            asks.levels,
            asks.num_levels,
            self._ask_entry_buffer,
        )
        self._normalize_levels_to_entries(
            bids.levels,
            bids.num_levels,
            self._bid_entry_buffer,
        )
        self._core.consume_snapshot_entries(
            self._ask_entry_buffer,
            asks.num_levels,
            self._bid_entry_buffer,
            bids.num_levels,
        )

    cdef void consume_deltas(self, OrderbookLevels asks, OrderbookLevels bids):
        cdef u64 required = (
            asks.num_levels if asks.num_levels >= bids.num_levels else bids.num_levels
        )
        self._ensure_entry_capacity(required)
        self._normalize_levels_to_entries(
            asks.levels,
            asks.num_levels,
            self._ask_entry_buffer,
        )
        self._normalize_levels_to_entries(
            bids.levels,
            bids.num_levels,
            self._bid_entry_buffer,
        )
        self._core.consume_deltas_entries(
            self._ask_entry_buffer,
            asks.num_levels,
            self._bid_entry_buffer,
            bids.num_levels,
        )

    cdef void consume_bbo(self, OrderbookLevel ask, OrderbookLevel bid):
        cdef:
            u64 ask_ticks
            u64 ask_lots
            u64 bid_ticks
            u64 bid_lots
            OrderbookEntry ask_entry
            OrderbookEntry bid_entry

        validate_price(ask.price)
        validate_size(ask.size)
        validate_price(bid.price)
        validate_size(bid.size)
        ask_ticks = convert_price_to_tick_trusted(ask.price, self._core._tick_size_recip)
        ask_lots = convert_size_to_lot_trusted(ask.size, self._core._lot_size_recip)
        bid_ticks = convert_price_to_tick_trusted(bid.price, self._core._tick_size_recip)
        bid_lots = convert_size_to_lot_trusted(bid.size, self._core._lot_size_recip)
        if bid_ticks >= ask_ticks:
            raise ValueError("Crossed BBO; bid price must be below ask price")
        ask_entry = create_orderbook_entry(
            ask_ticks,
            ask_lots,
            ask.norders,
        )
        bid_entry = create_orderbook_entry(
            bid_ticks,
            bid_lots,
            bid.norders,
        )
        self._core.consume_bbo_entries(ask_entry, bid_entry)

    cdef double get_mid_price(self):
        return self._core.get_mid_price()

    cdef double get_bbo_spread(self):
        return self._core.get_bbo_spread()

    cdef double get_wmid_price(self):
        return self._core.get_wmid_price()

    cdef double get_volume_weighted_mid_price(self, double size, bint is_base_currency):
        return self._core.get_volume_weighted_mid_price(size, is_base_currency)

    cdef double get_price_impact(self, double size, bint is_buy, bint is_base_currency):
        return self._core.get_price_impact(size, is_buy, is_base_currency)

    cdef double get_size_for_price_impact_bps(
        self,
        double impact_bps,
        bint is_buy,
        bint is_base_currency,
    ):
        return self._core.get_size_for_price_impact_bps(impact_bps, is_buy, is_base_currency)

    cdef bint is_bbo_crossed(self, double other_bid_price, double other_ask_price):
        return self._core.is_bbo_crossed(other_bid_price, other_ask_price)

    cdef bint does_bbo_price_change(self, double bid_price, double ask_price):
        return self._core.does_bbo_price_change(bid_price, ask_price)

    cdef tuple get_bbo(self):
        self._core._ensure_not_empty()
        return (
            self._core._level_from_entry(self._core._bids.top()),
            self._core._level_from_entry(self._core._asks.top()),
        )

    cdef OrderbookLevel get_bid(self, u64 index):
        return self._core._level_from_entry(self._core._bids.at(index))

    cdef OrderbookLevel get_ask(self, u64 index):
        return self._core._level_from_entry(self._core._asks.at(index))

    cdef u64 get_num_bids(self):
        return self._core.get_bids_data().num_levels

    cdef u64 get_num_asks(self):
        return self._core.get_asks_data().num_levels

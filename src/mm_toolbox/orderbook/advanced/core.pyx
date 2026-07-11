# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""Core advanced orderbook engine using compact normalized entries."""

from __future__ import annotations

from libc.float cimport DBL_MAX as INFINITY_DOUBLE
from libc.math cimport fabs, isfinite
from libc.stdint cimport uint64_t as u64
from libc.stdlib cimport free, malloc

from .enum.enums cimport CyOrderbookSortedness
from .ladder.ladder cimport OrderbookLadder, OrderbookLadderData
from .level.helpers cimport (
    convert_price_from_tick,
    convert_price_to_tick,
    convert_price_to_tick_trusted,
    convert_size_from_lot,
    convert_size_to_lot,
    convert_size_to_lot_trusted,
    reverse_entries,
    validate_price,
    validate_size,
)
from .level.level cimport (
    OrderbookEntry,
    OrderbookLevel,
    OrderbookLevels,
    create_orderbook_entry,
    create_orderbook_level,
)

cdef enum:
    SMALL_DELTA_LIMIT = 8
    ORDER_UNKNOWN = 0
    ORDER_ASCENDING = 1
    ORDER_DESCENDING = 2


cdef inline int _ask_order_from_sortedness(CyOrderbookSortedness sortedness) noexcept:
    if (
        sortedness == CyOrderbookSortedness.ASCENDING
        or sortedness == CyOrderbookSortedness.BIDS_DESCENDING_ASKS_ASCENDING
    ):
        return ORDER_ASCENDING
    if (
        sortedness == CyOrderbookSortedness.DESCENDING
        or sortedness == CyOrderbookSortedness.BIDS_ASCENDING_ASKS_DESCENDING
    ):
        return ORDER_DESCENDING
    return ORDER_UNKNOWN


cdef inline int _bid_order_from_sortedness(CyOrderbookSortedness sortedness) noexcept:
    if (
        sortedness == CyOrderbookSortedness.ASCENDING
        or sortedness == CyOrderbookSortedness.BIDS_ASCENDING_ASKS_DESCENDING
    ):
        return ORDER_ASCENDING
    if (
        sortedness == CyOrderbookSortedness.DESCENDING
        or sortedness == CyOrderbookSortedness.BIDS_DESCENDING_ASKS_ASCENDING
    ):
        return ORDER_DESCENDING
    return ORDER_UNKNOWN


cdef int _infer_entries_order(OrderbookEntry* entries, u64 count):
    cdef:
        u64 i
        bint ascending = True
        bint descending = True
        u64 previous
        u64 ticks

    if count < 2:
        return ORDER_UNKNOWN

    previous = entries[0].ticks
    for i in range(1, count):
        ticks = entries[i].ticks
        if previous > ticks:
            ascending = False
        if previous < ticks:
            descending = False
        previous = ticks

    if ascending:
        return ORDER_ASCENDING
    if descending:
        return ORDER_DESCENDING
    raise ValueError("Unsorted orderbook levels; expected stable per-side sorted input")


cdef class CoreAdvancedOrderbook:
    """Core orderbook engine managing normalized bid and ask ladders."""

    def __cinit__(
        self,
        double tick_size,
        double lot_size,
        u64 num_levels,
        CyOrderbookSortedness delta_sortedness,
        CyOrderbookSortedness snapshot_sortedness,
    ):
        if tick_size <= 0.0 or not isfinite(tick_size):
            raise ValueError(f"Invalid tick_size; expected finite >0 but got {tick_size}")
        if lot_size <= 0.0 or not isfinite(lot_size):
            raise ValueError(f"Invalid lot_size; expected finite >0 but got {lot_size}")
        if num_levels < 4:
            raise ValueError(f"Invalid num_levels; expected >=4 but got {num_levels}")

        self._tick_size = tick_size
        self._lot_size = lot_size
        self._tick_size_recip = 1.0 / tick_size
        self._lot_size_recip = 1.0 / lot_size
        self._max_levels = num_levels
        self._delta_sortedness = delta_sortedness
        self._snapshot_sortedness = snapshot_sortedness
        self._snapshot_ask_order = _ask_order_from_sortedness(snapshot_sortedness)
        self._snapshot_bid_order = _bid_order_from_sortedness(snapshot_sortedness)
        self._delta_ask_order = _ask_order_from_sortedness(delta_sortedness)
        self._delta_bid_order = _bid_order_from_sortedness(delta_sortedness)
        self._bids = OrderbookLadder(max_levels=num_levels, is_price_ascending=False)
        self._asks = OrderbookLadder(max_levels=num_levels, is_price_ascending=True)
        self._bids_data = self._bids.get_data()
        self._asks_data = self._asks.get_data()

    cdef inline void _ensure_not_empty(self):
        if self._bids.is_empty() or self._asks.is_empty():
            raise RuntimeError(
                "Empty view on one/both sides of orderbook; cannot compute without data"
            )

    cdef inline bint _check_if_empty(self):
        return not self._bids.is_empty() and not self._asks.is_empty()

    cdef inline OrderbookLevel _level_from_entry(self, OrderbookEntry* entry):
        return create_orderbook_level(
            convert_price_from_tick(entry.ticks, self._tick_size),
            convert_size_from_lot(entry.lots, self._lot_size),
            entry.norders,
        )

    cdef inline OrderbookEntry* _ordered_entry(
        self,
        OrderbookEntry* entries,
        u64 count,
        int source_order,
        int target_order,
        u64 index,
    ) noexcept:
        if source_order == target_order:
            return &entries[index]
        return &entries[count - 1 - index]

    cdef u64 _fill_snapshot_entries(
        self,
        OrderbookEntry* entries,
        u64 count,
        OrderbookLadderData* data,
        int source_order,
        int target_order,
    ) noexcept:
        cdef:
            u64 copy_n = count if count <= data.max_levels else data.max_levels
            u64 i

        for i in range(copy_n):
            data.levels[i] = self._ordered_entry(entries, count, source_order, target_order, i)[0]
        data.num_levels = copy_n
        return copy_n

    cdef void _normalize_levels(self, OrderbookLevels levels, OrderbookEntry* entries):
        cdef u64 i
        for i in range(levels.num_levels):
            validate_price(levels.levels[i].price)
            validate_size(levels.levels[i].size)
            entries[i] = create_orderbook_entry(
                convert_price_to_tick_trusted(levels.levels[i].price, self._tick_size_recip),
                convert_size_to_lot_trusted(levels.levels[i].size, self._lot_size_recip),
                levels.levels[i].norders,
            )

    cdef bint _has_nonzero_entries(self, OrderbookEntry* entries, u64 count) noexcept:
        cdef u64 i
        for i in range(count):
            if entries[i].lots != 0:
                return True
        return False

    cdef inline bint _should_ignore_crossed_deltas(
        self,
        OrderbookEntry* asks,
        u64 ask_count,
        int ask_order,
        OrderbookEntry* bids,
        u64 bid_count,
        int bid_order,
    ) noexcept:
        cdef:
            bint has_ask_replacements = self._has_nonzero_entries(asks, ask_count)
            bint has_bid_replacements = self._has_nonzero_entries(bids, bid_count)
            u64 best_ask_ticks = self._asks.top().ticks
            u64 worst_ask_ticks = self._asks.bottom().ticks
            u64 best_bid_ticks = self._bids.top().ticks
            u64 worst_bid_ticks = self._bids.bottom().ticks
            OrderbookEntry* incoming_ask = NULL
            OrderbookEntry* incoming_bid = NULL

        if ask_count > 0:
            incoming_ask = self._ordered_entry(
                asks,
                ask_count,
                ask_order,
                ORDER_ASCENDING,
                0,
            )
        if bid_count > 0:
            incoming_bid = self._ordered_entry(
                bids,
                bid_count,
                bid_order,
                ORDER_DESCENDING,
                0,
            )

        if (
            ask_count > 0
            and incoming_ask[0].lots != 0
            and incoming_ask[0].ticks < best_ask_ticks
            and incoming_ask[0].ticks <= worst_bid_ticks
            and not has_bid_replacements
        ):
            return True

        if (
            bid_count > 0
            and incoming_bid[0].lots != 0
            and incoming_bid[0].ticks > best_bid_ticks
            and incoming_bid[0].ticks >= worst_ask_ticks
            and not has_ask_replacements
        ):
            return True

        return False

    cdef bint _can_apply_replacement_side(
        self,
        OrderbookLadder ladder,
        OrderbookLadderData* data,
        OrderbookEntry* entries,
        u64 count,
        int source_order,
        int target_order,
    ) noexcept:
        cdef:
            u64 update_i
            u64 ladder_i = 0
            OrderbookEntry* entry
            u64 current_ticks

        if count == 0:
            return True
        if data.num_levels == 0:
            return False

        for update_i in range(count):
            entry = self._ordered_entry(entries, count, source_order, target_order, update_i)
            if entry.lots == 0:
                return False
            if update_i == 0:
                ladder_i = ladder.seek_start(entry.ticks)
            while ladder_i < data.num_levels:
                current_ticks = ladder.at(ladder_i).ticks
                if current_ticks == entry.ticks:
                    break
                if data.is_price_ascending:
                    if current_ticks > entry.ticks:
                        return False
                else:
                    if current_ticks < entry.ticks:
                        return False
                ladder_i += 1
            if ladder_i >= data.num_levels or ladder.at(ladder_i).ticks != entry.ticks:
                return False
            ladder_i += 1

        return True

    cdef void _apply_replacement_side(
        self,
        OrderbookLadder ladder,
        OrderbookLadderData* data,
        OrderbookEntry* entries,
        u64 count,
        int source_order,
        int target_order,
    ) noexcept:
        cdef:
            u64 update_i
            u64 ladder_i = 0
            OrderbookEntry* entry

        for update_i in range(count):
            entry = self._ordered_entry(entries, count, source_order, target_order, update_i)
            if update_i == 0:
                ladder_i = ladder.seek_start(entry.ticks)
            while ladder.at(ladder_i).ticks != entry.ticks:
                ladder_i += 1
            ladder.assign_entry(ladder_i, entry)
            ladder_i += 1

    cdef bint _try_apply_replacement_deltas(
        self,
        OrderbookEntry* asks,
        u64 ask_count,
        int ask_order,
        OrderbookEntry* bids,
        u64 bid_count,
        int bid_order,
    ) noexcept:
        if not self._can_apply_replacement_side(
            self._asks,
            self._asks_data,
            asks,
            ask_count,
            ask_order,
            ORDER_ASCENDING,
        ):
            return False
        if not self._can_apply_replacement_side(
            self._bids,
            self._bids_data,
            bids,
            bid_count,
            bid_order,
            ORDER_DESCENDING,
        ):
            return False

        self._apply_replacement_side(
            self._asks,
            self._asks_data,
            asks,
            ask_count,
            ask_order,
            ORDER_ASCENDING,
        )
        self._apply_replacement_side(
            self._bids,
            self._bids_data,
            bids,
            bid_count,
            bid_order,
            ORDER_DESCENDING,
        )
        if ask_count > 0:
            self._remove_crossed_bids()
        if bid_count > 0:
            self._remove_crossed_asks()
        return True

    cdef inline void _apply_entry_delta(
        self,
        OrderbookLadder ladder,
        OrderbookLadderData* data,
        OrderbookEntry* entry,
    ) noexcept:
        cdef u64 index

        if data.num_levels == 0:
            if entry.lots != 0:
                ladder.insert_entry(0, entry)
                ladder.increment_count()
            return

        index = ladder.seek_start(entry.ticks)
        if index < data.num_levels and ladder.at(index).ticks == entry.ticks:
            if entry.lots == 0:
                ladder.roll_left(index)
                ladder.decrement_count()
            else:
                ladder.assign_entry(index, entry)
        elif entry.lots != 0 and index < data.max_levels:
            ladder.roll_right(index)
            ladder.insert_entry(index, entry)
            ladder.increment_count()

    cdef inline void _apply_small_deltas(
        self,
        OrderbookEntry* asks,
        u64 ask_count,
        OrderbookEntry* bids,
        u64 bid_count,
    ) noexcept:
        cdef u64 i

        for i in range(ask_count):
            self._apply_entry_delta(self._asks, self._asks_data, &asks[i])
        for i in range(bid_count):
            self._apply_entry_delta(self._bids, self._bids_data, &bids[i])
        if ask_count > 0:
            self._remove_crossed_bids()
        if bid_count > 0:
            self._remove_crossed_asks()

    cdef void _remove_crossed_bids(self):
        while (
            self._bids_data.num_levels > 0
            and self._asks_data.num_levels > 0
            and self._bids.top().ticks >= self._asks.top().ticks
        ):
            self._bids.roll_left(0)
            self._bids.decrement_count()

    cdef void _remove_crossed_asks(self):
        while (
            self._bids_data.num_levels > 0
            and self._asks_data.num_levels > 0
            and self._bids.top().ticks >= self._asks.top().ticks
        ):
            self._asks.roll_left(0)
            self._asks.decrement_count()

    cdef inline void _assign_top(self, OrderbookLadder ladder, OrderbookEntry* entry):
        ladder.roll_right(0)
        ladder.insert_entry(0, entry)
        ladder.increment_count()

    cdef inline void clear(self):
        self._bids.reset()
        self._asks.reset()

    cdef void consume_snapshot_entries(
        self,
        OrderbookEntry* asks,
        u64 ask_count,
        OrderbookEntry* bids,
        u64 bid_count,
    ):
        cdef:
            int inferred
            int ask_order = self._snapshot_ask_order
            int bid_order = self._snapshot_bid_order
            OrderbookEntry* best_ask = NULL
            OrderbookEntry* best_bid = NULL

        if ask_order == ORDER_UNKNOWN:
            inferred = _infer_entries_order(asks, ask_count)
            if inferred != ORDER_UNKNOWN:
                self._snapshot_ask_order = inferred
                ask_order = inferred
        if ask_order == ORDER_UNKNOWN:
            ask_order = ORDER_ASCENDING

        if bid_order == ORDER_UNKNOWN:
            inferred = _infer_entries_order(bids, bid_count)
            if inferred != ORDER_UNKNOWN:
                self._snapshot_bid_order = inferred
                bid_order = inferred
        if bid_order == ORDER_UNKNOWN:
            bid_order = ORDER_DESCENDING

        if ask_count > 0:
            best_ask = self._ordered_entry(asks, ask_count, ask_order, ORDER_ASCENDING, 0)
        if bid_count > 0:
            best_bid = self._ordered_entry(bids, bid_count, bid_order, ORDER_DESCENDING, 0)

        if (
            ask_count > 0
            and bid_count > 0
            and best_bid[0].ticks >= best_ask[0].ticks
        ):
            raise ValueError("Crossed snapshot; best bid must be below best ask")

        self._asks.reset()
        self._bids.reset()
        self._fill_snapshot_entries(
            asks,
            ask_count,
            self._asks_data,
            ask_order,
            ORDER_ASCENDING,
        )
        self._fill_snapshot_entries(
            bids,
            bid_count,
            self._bids_data,
            bid_order,
            ORDER_DESCENDING,
        )

    cdef void consume_deltas_entries(
        self,
        OrderbookEntry* asks,
        u64 ask_count,
        OrderbookEntry* bids,
        u64 bid_count,
    ):
        if not self._check_if_empty():
            return

        cdef:
            int inferred
            int ask_order = self._delta_ask_order
            int bid_order = self._delta_bid_order

        if ask_order == ORDER_UNKNOWN:
            inferred = _infer_entries_order(asks, ask_count)
            if inferred != ORDER_UNKNOWN:
                self._delta_ask_order = inferred
                ask_order = inferred
        if ask_order == ORDER_UNKNOWN:
            ask_order = ORDER_ASCENDING

        if bid_order == ORDER_UNKNOWN:
            inferred = _infer_entries_order(bids, bid_count)
            if inferred != ORDER_UNKNOWN:
                self._delta_bid_order = inferred
                bid_order = inferred
        if bid_order == ORDER_UNKNOWN:
            bid_order = ORDER_DESCENDING

        if ask_count + bid_count <= SMALL_DELTA_LIMIT:
            if self._should_ignore_crossed_deltas(
                asks,
                ask_count,
                ask_order,
                bids,
                bid_count,
                bid_order,
            ):
                return
            self._apply_small_deltas(asks, ask_count, bids, bid_count)
            return

        if self._should_ignore_crossed_deltas(
            asks,
            ask_count,
            ask_order,
            bids,
            bid_count,
            bid_order,
        ):
            return

        if self._try_apply_replacement_deltas(
            asks,
            ask_count,
            ask_order,
            bids,
            bid_count,
            bid_order,
        ):
            return

        if ask_count > 0:
            if ask_order != ORDER_ASCENDING:
                reverse_entries(asks, ask_count)
            self._asks.apply_sorted_deltas(asks, ask_count)
        if bid_count > 0:
            if bid_order != ORDER_DESCENDING:
                reverse_entries(bids, bid_count)
            self._bids.apply_sorted_deltas(bids, bid_count)
        if ask_count > 0:
            self._remove_crossed_bids()
        if bid_count > 0:
            self._remove_crossed_asks()

    cdef void consume_snapshot(self, OrderbookLevels new_asks, OrderbookLevels new_bids):
        cdef:
            OrderbookEntry* ask_entries = NULL
            OrderbookEntry* bid_entries = NULL
            u64 ask_count = new_asks.num_levels
            u64 bid_count = new_bids.num_levels

        if new_asks.num_levels > 0:
            ask_entries = <OrderbookEntry*>malloc(new_asks.num_levels * sizeof(OrderbookEntry))
            if ask_entries == NULL:
                raise MemoryError("Failed to allocate normalized ask entries")
        if new_bids.num_levels > 0:
            bid_entries = <OrderbookEntry*>malloc(new_bids.num_levels * sizeof(OrderbookEntry))
            if bid_entries == NULL:
                if ask_entries != NULL:
                    free(ask_entries)
                raise MemoryError("Failed to allocate normalized bid entries")

        try:
            self._normalize_levels(new_asks, ask_entries)
            self._normalize_levels(new_bids, bid_entries)
            self.consume_snapshot_entries(ask_entries, ask_count, bid_entries, bid_count)
        finally:
            if ask_entries != NULL:
                free(ask_entries)
            if bid_entries != NULL:
                free(bid_entries)

    cdef void consume_deltas(self, OrderbookLevels asks, OrderbookLevels bids):
        cdef:
            OrderbookEntry* ask_entries = NULL
            OrderbookEntry* bid_entries = NULL
            u64 ask_count = asks.num_levels
            u64 bid_count = bids.num_levels

        if asks.num_levels > 0:
            ask_entries = <OrderbookEntry*>malloc(asks.num_levels * sizeof(OrderbookEntry))
            if ask_entries == NULL:
                raise MemoryError("Failed to allocate normalized ask delta entries")
        if bids.num_levels > 0:
            bid_entries = <OrderbookEntry*>malloc(bids.num_levels * sizeof(OrderbookEntry))
            if bid_entries == NULL:
                if ask_entries != NULL:
                    free(ask_entries)
                raise MemoryError("Failed to allocate normalized bid delta entries")

        try:
            self._normalize_levels(asks, ask_entries)
            self._normalize_levels(bids, bid_entries)
            self.consume_deltas_entries(ask_entries, ask_count, bid_entries, bid_count)
        finally:
            if ask_entries != NULL:
                free(ask_entries)
            if bid_entries != NULL:
                free(bid_entries)

    cdef void consume_bbo_entries(self, OrderbookEntry ask_entry, OrderbookEntry bid_entry):
        if not self._check_if_empty():
            return

        cdef OrderbookEntry* top

        if self._asks_data.num_levels > 0:
            top = self._asks.top()
            if ask_entry.lots == 0 and ask_entry.ticks == top.ticks:
                self._asks.roll_left(0)
                self._asks.decrement_count()
            elif ask_entry.ticks == top.ticks:
                self._asks.assign_entry(0, &ask_entry)
            elif ask_entry.ticks < top.ticks:
                if ask_entry.lots != 0:
                    self._assign_top(self._asks, &ask_entry)
            else:
                self._asks.roll_left(0)
                self._asks.decrement_count()
                if self._asks_data.num_levels == 0 and ask_entry.lots != 0:
                    self._assign_top(self._asks, &ask_entry)
        elif ask_entry.lots != 0:
            self._assign_top(self._asks, &ask_entry)

        if self._bids_data.num_levels > 0:
            top = self._bids.top()
            if bid_entry.lots == 0 and bid_entry.ticks == top.ticks:
                self._bids.roll_left(0)
                self._bids.decrement_count()
            elif bid_entry.ticks == top.ticks:
                self._bids.assign_entry(0, &bid_entry)
            elif bid_entry.ticks > top.ticks:
                if bid_entry.lots != 0:
                    self._assign_top(self._bids, &bid_entry)
            else:
                self._bids.roll_left(0)
                self._bids.decrement_count()
                if self._bids_data.num_levels == 0 and bid_entry.lots != 0:
                    self._assign_top(self._bids, &bid_entry)
        elif bid_entry.lots != 0:
            self._assign_top(self._bids, &bid_entry)

        self._remove_crossed_asks()
        if self._asks_data.num_levels == 0 and ask_entry.lots != 0:
            self._assign_top(self._asks, &ask_entry)
        if self._bids_data.num_levels == 0 and bid_entry.lots != 0:
            self._assign_top(self._bids, &bid_entry)

    cdef void consume_bbo(self, OrderbookLevel ask, OrderbookLevel bid):
        self.consume_bbo_values(
            ask.price,
            ask.size,
            bid.price,
            bid.size,
            ask.norders,
            bid.norders,
        )

    cdef void consume_bbo_values(
        self,
        double ask_price,
        double ask_size,
        double bid_price,
        double bid_size,
        u64 ask_norders,
        u64 bid_norders,
    ):
        validate_price(ask_price)
        validate_size(ask_size)
        validate_price(bid_price)
        validate_size(bid_size)

        cdef:
            u64 ask_ticks = convert_price_to_tick_trusted(ask_price, self._tick_size_recip)
            u64 bid_ticks = convert_price_to_tick_trusted(bid_price, self._tick_size_recip)
            OrderbookEntry ask_entry = create_orderbook_entry(
                ask_ticks,
                convert_size_to_lot_trusted(ask_size, self._lot_size_recip),
                ask_norders,
            )
            OrderbookEntry bid_entry = create_orderbook_entry(
                bid_ticks,
                convert_size_to_lot_trusted(bid_size, self._lot_size_recip),
                bid_norders,
            )

        if bid_ticks >= ask_ticks:
            raise ValueError("Crossed BBO; bid price must be below ask price")
        if not self._check_if_empty():
            return

        self.consume_bbo_entries(ask_entry, bid_entry)

    cdef inline double get_mid_price(self):
        self._ensure_not_empty()
        cdef u64 bid_ticks = self._bids.top().ticks
        cdef u64 ask_ticks = self._asks.top().ticks
        return convert_price_from_tick((bid_ticks + ask_ticks) // 2, self._tick_size)

    cdef inline double get_bbo_spread(self):
        self._ensure_not_empty()
        cdef u64 bid_ticks = self._bids.top().ticks
        cdef u64 ask_ticks = self._asks.top().ticks
        if ask_ticks >= bid_ticks:
            return convert_price_from_tick(ask_ticks - bid_ticks, self._tick_size)
        return -convert_price_from_tick(bid_ticks - ask_ticks, self._tick_size)

    cdef inline double get_wmid_price(self):
        self._ensure_not_empty()
        cdef:
            OrderbookEntry* bid = self._bids.top()
            OrderbookEntry* ask = self._asks.top()
            u64 total_lots = bid.lots + ask.lots
            double weighted_ticks
        if total_lots == 0:
            return 0.0
        weighted_ticks = (
            <double>bid.ticks * <double>bid.lots + <double>ask.ticks * <double>ask.lots
        ) / <double>total_lots
        return convert_price_from_tick(<u64>weighted_ticks, self._tick_size)

    cdef inline double get_volume_weighted_mid_price(self, double size, bint is_base_currency):
        self._ensure_not_empty()
        cdef double mid_price = self.get_mid_price()
        if size <= 0.0:
            return mid_price
        cdef:
            double target = size if is_base_currency else (size / mid_price)
            u64 target_lots = convert_size_to_lot(target, self._lot_size_recip)
            u64 cum_ask_lots = 0
            u64 cum_bid_lots = 0
            u64 final_buy_ticks = 0
            u64 final_sell_ticks = 0
            u64 i
        for i in range(self._asks_data.num_levels):
            cum_ask_lots += self._asks.at(i).lots
            if cum_ask_lots >= target_lots:
                final_buy_ticks = self._asks.at(i).ticks
                break
        for i in range(self._bids_data.num_levels):
            cum_bid_lots += self._bids.at(i).lots
            if cum_bid_lots >= target_lots:
                final_sell_ticks = self._bids.at(i).ticks
                break
        if final_buy_ticks == 0 or final_sell_ticks == 0:
            return INFINITY_DOUBLE
        return convert_price_from_tick((final_buy_ticks + final_sell_ticks) // 2, self._tick_size)

    cdef inline double get_price_impact(self, double size, bint is_buy, bint is_base_currency):
        self._ensure_not_empty()
        if size <= 0.0:
            return 0.0
        cdef:
            OrderbookLadder side = self._asks if is_buy else self._bids
            OrderbookLadderData* side_data = side.get_data()
            u64 touch_anchor_ticks = side.top().ticks
            double touch_anchor_price = convert_price_from_tick(touch_anchor_ticks, self._tick_size)
            double target_base = size if is_base_currency else (size / touch_anchor_price)
            u64 target_lots = convert_size_to_lot(target_base, self._lot_size_recip)
            u64 remaining_lots = target_lots
            u64 consumed_lots
            u64 available_lots
            u64 last_touched_ticks = touch_anchor_ticks
            u64 i
        if target_lots == 0:
            return 0.0
        for i in range(side_data.num_levels):
            available_lots = side.at(i).lots
            consumed_lots = available_lots if available_lots < remaining_lots else remaining_lots
            if consumed_lots > 0:
                last_touched_ticks = side.at(i).ticks
            remaining_lots -= consumed_lots
            if remaining_lots == 0:
                break
        if remaining_lots > 0:
            return INFINITY_DOUBLE
        return fabs(
            convert_price_from_tick(last_touched_ticks, self._tick_size)
            - touch_anchor_price
        )

    cdef inline double get_size_for_price_impact_bps(
        self,
        double impact_bps,
        bint is_buy,
        bint is_base_currency,
    ):
        self._ensure_not_empty()
        if impact_bps <= 0.0:
            return 0.0
        cdef:
            OrderbookLadder side = self._asks if is_buy else self._bids
            OrderbookLadderData* side_data = side.get_data()
            u64 touch_anchor_ticks = side.top().ticks
            double touch_anchor_price = convert_price_from_tick(
                touch_anchor_ticks,
                self._tick_size,
            )
            double limit_price
            u64 limit_ticks
            u64 ticks
            u64 lots
            u64 total_lots = 0
            double total_ticks_times_lots = 0.0
            u64 i
        if is_buy:
            limit_price = touch_anchor_price * (1.0 + impact_bps / 10000.0)
            limit_ticks = convert_price_to_tick(limit_price, self._tick_size_recip)
            for i in range(side_data.num_levels):
                ticks = side.at(i).ticks
                if ticks > limit_ticks:
                    break
                lots = side.at(i).lots
                total_lots += lots
                total_ticks_times_lots += <double>ticks * <double>lots
        else:
            limit_price = touch_anchor_price * (1.0 - impact_bps / 10000.0)
            limit_ticks = convert_price_to_tick(limit_price, self._tick_size_recip)
            if convert_price_from_tick(limit_ticks, self._tick_size) < limit_price:
                limit_ticks += 1
            for i in range(side_data.num_levels):
                ticks = side.at(i).ticks
                if ticks < limit_ticks:
                    break
                lots = side.at(i).lots
                total_lots += lots
                total_ticks_times_lots += <double>ticks * <double>lots
        if is_base_currency:
            return convert_size_from_lot(total_lots, self._lot_size)
        return (self._tick_size * self._lot_size) * total_ticks_times_lots

    cdef inline bint is_bbo_crossed(
        self,
        double other_bid_price,
        double other_ask_price,
    ):
        self._ensure_not_empty()
        validate_price(other_bid_price)
        validate_price(other_ask_price)
        cdef u64 other_bid_ticks = convert_price_to_tick(
            other_bid_price,
            self._tick_size_recip,
        )
        cdef u64 other_ask_ticks = convert_price_to_tick(
            other_ask_price,
            self._tick_size_recip,
        )
        return (
            self._bids.top().ticks >= other_ask_ticks
            or self._asks.top().ticks <= other_bid_ticks
        )

    cdef inline bint does_bbo_price_change(self, double bid_price, double ask_price):
        self._ensure_not_empty()
        validate_price(bid_price)
        validate_price(ask_price)
        cdef u64 other_bid_ticks = convert_price_to_tick(
            bid_price,
            self._tick_size_recip,
        )
        cdef u64 other_ask_ticks = convert_price_to_tick(
            ask_price,
            self._tick_size_recip,
        )
        return (
            self._bids.top().ticks != other_bid_ticks
            or self._asks.top().ticks != other_ask_ticks
        )

    cdef inline OrderbookLadderData* get_bids_data(self) noexcept:
        return self._bids_data

    cdef inline OrderbookLadderData* get_asks_data(self) noexcept:
        return self._asks_data

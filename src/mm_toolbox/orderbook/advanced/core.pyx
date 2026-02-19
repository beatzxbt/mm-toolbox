# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
Core orderbook engine for managing bids and asks with tick-based pricing.

Implements CoreAdvancedOrderbook class which handles orderbook state management,
level normalization, snapshot/delta ingestion, and price calculations. Uses
fixed-size pre-allocated ladders with in-place updates for optimal performance.
"""
from __future__ import annotations

from libc.math cimport floor
from libc.float cimport DBL_MAX as INFINITY_DOUBLE
from libc.stdint cimport uint64_t as u64

from .level.level cimport OrderbookLevel, OrderbookLevels
from .level.helpers cimport (
    convert_price_from_tick,
    convert_price_to_tick,
    convert_price_to_tick_fast,
    convert_size_from_lot,
    convert_size_to_lot,
    convert_size_to_lot_fast,
    inplace_sort_levels_by_ticks,
    reverse_levels,
)
from .ladder.ladder cimport OrderbookLadder, OrderbookLadderData
from .enum.enums cimport CyOrderbookSortedness


cdef class CoreAdvancedOrderbook:
    """Core orderbook engine managing bids and asks with efficient in-place updates."""
    def __cinit__(
        self,
        double tick_size,
        double lot_size,
        u64 num_levels,
        CyOrderbookSortedness delta_sortedness,
        CyOrderbookSortedness snapshot_sortedness,
    ):
        if tick_size <= 0.0:
            raise ValueError(f"Invalid tick_size; expected >0 but got {tick_size}")
        if lot_size <= 0.0:
            raise ValueError(f"Invalid lot_size; expected >0 but got {lot_size}")
        if num_levels < 64:
            raise ValueError(f"Invalid num_levels; expected >=64 but got {num_levels}")
        self._tick_size = tick_size
        self._lot_size = lot_size
        self._tick_size_recip = 1.0 / tick_size
        self._lot_size_recip = 1.0 / lot_size
        self._max_levels = num_levels
        self._delta_sortedness = delta_sortedness
        self._snapshot_sortedness = snapshot_sortedness
        self._bids = OrderbookLadder(max_levels=self._max_levels, is_price_ascending=False)
        self._asks = OrderbookLadder(max_levels=self._max_levels, is_price_ascending=True)
        self._bids_data = self._bids.get_data()
        self._asks_data = self._asks.get_data()

    cdef inline void _ensure_not_empty(self):
        """Ensures the orderbook has been populated."""
        if self._bids.is_empty() or self._asks.is_empty():
            raise RuntimeError("Empty view on one/both sides of orderbook; cannot compute without data")

    cdef inline bint _check_if_empty(self):
        """Checks if the orderbook is not empty."""
        return not self._bids.is_empty() and not self._asks.is_empty()

    cdef void _normalize_incoming_levels(
        self,
        OrderbookLevels asks,
        OrderbookLevels bids,
        bint is_snapshot,
    ):
        """Normalize incoming levels to the orderbook's internal representation."""
        cdef:
            CyOrderbookSortedness sortedness_code = (
                self._snapshot_sortedness 
                if is_snapshot else 
                self._delta_sortedness
            )
            OrderbookLevel* ask_level
            OrderbookLevel* bid_level
            u64 i

        for i in range(asks.num_levels):
            ask_level = &asks.levels[i]
            ask_level.ticks = convert_price_to_tick_fast(ask_level.price, self._tick_size_recip)
            ask_level.lots = convert_size_to_lot_fast(ask_level.size, self._lot_size_recip)
        for i in range(bids.num_levels):
            bid_level = &bids.levels[i]
            bid_level.ticks = convert_price_to_tick_fast(bid_level.price, self._tick_size_recip)
            bid_level.lots = convert_size_to_lot_fast(bid_level.size, self._lot_size_recip)

        # Likely most common user choice due to sortedness being unspecified
        if sortedness_code == CyOrderbookSortedness.UNKNOWN:
            inplace_sort_levels_by_ticks(levels=asks, ascending=True)
            inplace_sort_levels_by_ticks(levels=bids, ascending=False)
        
        # Used by most exchanges for delta updates, preferred path internally
        elif sortedness_code == CyOrderbookSortedness.BIDS_DESCENDING_ASKS_ASCENDING:
            pass

        # Used by most exchanges for snapshot updates
        elif sortedness_code == CyOrderbookSortedness.ASCENDING:
            reverse_levels(levels=bids)
        
        # Unlikely, should never really happen.
        elif sortedness_code == CyOrderbookSortedness.DESCENDING:
            reverse_levels(levels=asks)

        # Unlikely, should never really happen.
        elif sortedness_code == CyOrderbookSortedness.BIDS_ASCENDING_ASKS_DESCENDING:
            reverse_levels(levels=asks)
            reverse_levels(levels=bids)

    cdef void _process_matching_ask_ticks(self, OrderbookLevel* ask):
        """Rolls the ask level array left (removing the top-of-book ask) if lots=0, otherwise updates size/lots/norders."""
        cdef:
            OrderbookLadderData* asks = self._asks_data
            OrderbookLevel* top_of_book_ask = &asks.levels[0]
        if ask.lots == 0:
            self._asks.roll_left(0)
            self._asks.decrement_count()
        else:
            top_of_book_ask.size = ask.size
            top_of_book_ask.lots = ask.lots
            top_of_book_ask.norders = ask.norders

    cdef void _process_matching_bid_ticks(self, OrderbookLevel* bid):
        """Rolls the bid level array left (removing the top-of-book bid) if lots=0, otherwise updates size/lots/norders."""
        cdef:
            OrderbookLadderData* bids = self._bids_data
            OrderbookLevel* top_of_book_bid = &bids.levels[0]
        if bid.lots == 0:
            self._bids.roll_left(0)
            self._bids.decrement_count()
        else:
            top_of_book_bid.size = bid.size
            top_of_book_bid.lots = bid.lots
            top_of_book_bid.norders = bid.norders

    cdef void _process_lower_ask_ticks(self, OrderbookLevel* ask):
        """Rolls the ask level array right (adding a new ask level) then corrects for any overlapping bids"""
        cdef:
            OrderbookLadderData* bids = self._bids_data
            OrderbookLadderData* asks = self._asks_data
            OrderbookLevel* top_of_book_bid
            OrderbookLevel* top_of_book_ask

        if ask.lots == 0:
            return

        self._asks.roll_right(0)
        self._asks.increment_count()

        top_of_book_ask = &asks.levels[0]
        top_of_book_ask.price = ask.price
        top_of_book_ask.ticks = ask.ticks
        top_of_book_ask.size = ask.size
        top_of_book_ask.lots = ask.lots
        top_of_book_ask.norders = ask.norders

        # Remove overlapping bids (fix: check num_levels before dereferencing)
        while bids.num_levels > 0:
            top_of_book_bid = &bids.levels[0]
            if ask.ticks > top_of_book_bid.ticks:
                break
            self._bids.roll_left(0)
            self._bids.decrement_count()

    cdef void _process_higher_bid_ticks(self, OrderbookLevel* bid):
        """Rolls the bid level array right (adding a new bid level) then corrects for any overlapping asks"""
        cdef:
            OrderbookLadderData* bids = self._bids_data
            OrderbookLadderData* asks = self._asks_data
            OrderbookLevel* top_of_book_bid
            OrderbookLevel* top_of_book_ask

        if bid.lots == 0:
            return

        self._bids.roll_right(0)
        self._bids.increment_count()

        top_of_book_bid = &bids.levels[0]
        top_of_book_bid.price = bid.price
        top_of_book_bid.ticks = bid.ticks
        top_of_book_bid.size = bid.size
        top_of_book_bid.lots = bid.lots
        top_of_book_bid.norders = bid.norders

        # Remove overlapping asks (fix: check num_levels before dereferencing)
        while asks.num_levels > 0:
            top_of_book_ask = &asks.levels[0]
            if bid.ticks < top_of_book_ask.ticks:
                break
            self._asks.roll_left(0)
            self._asks.decrement_count()

    cdef void _process_middle_ask_ticks(self, OrderbookLevel* ask):
        """Process an ask level that falls in the middle of the existing ask levels."""
        cdef:
            OrderbookLadderData* bids = self._bids_data
            OrderbookLadderData* asks = self._asks_data
            OrderbookLevel* ask_insertion_level
            u64 i
            u64 current_ask_ticks
            u64 last_idx = asks.num_levels
            u64 insert_idx = last_idx
            bint is_matching = False

        for i in range(1, last_idx):
            current_ask_ticks = asks.levels[i].ticks
            if current_ask_ticks >= ask.ticks:
                insert_idx = i
                if current_ask_ticks == ask.ticks:
                    is_matching = True
                break

        if is_matching:
            if ask.lots == 0:
                self._asks.roll_left(insert_idx)
                self._asks.decrement_count()
            else:
                ask_insertion_level = &asks.levels[insert_idx]
                ask_insertion_level.size = ask.size
                ask_insertion_level.lots = ask.lots
                ask_insertion_level.norders = ask.norders
        else:
            if ask.lots == 0:
                return
            self._asks.roll_right(insert_idx)
            self._asks.increment_count()
            ask_insertion_level = &asks.levels[insert_idx]
            ask_insertion_level.price = ask.price
            ask_insertion_level.ticks = ask.ticks
            ask_insertion_level.size = ask.size
            ask_insertion_level.lots = ask.lots
            ask_insertion_level.norders = ask.norders

    cdef void _process_middle_bid_ticks(self, OrderbookLevel* bid):
        """Process a bid level that falls in the middle of the existing bid levels."""
        cdef:
            OrderbookLadderData* bids = self._bids_data
            OrderbookLadderData* asks = self._asks_data
            OrderbookLevel* bid_insertion_level
            u64 i
            u64 current_bid_ticks
            u64 last_idx = bids.num_levels
            u64 insert_idx = last_idx
            bint is_matching = False

        for i in range(1, last_idx):
            current_bid_ticks = bids.levels[i].ticks
            if current_bid_ticks <= bid.ticks:
                insert_idx = i
                if current_bid_ticks == bid.ticks:
                    is_matching = True
                break

        if is_matching:
            if bid.lots == 0:
                self._bids.roll_left(insert_idx)
                self._bids.decrement_count()
            else:
                bid_insertion_level = &bids.levels[insert_idx]
                bid_insertion_level.size = bid.size
                bid_insertion_level.lots = bid.lots
                bid_insertion_level.norders = bid.norders
        else:
            if bid.lots == 0:
                return
                
            self._bids.roll_right(insert_idx)
            self._bids.increment_count()
            bid_insertion_level = &bids.levels[insert_idx]
            bid_insertion_level.price = bid.price
            bid_insertion_level.ticks = bid.ticks
            bid_insertion_level.size = bid.size
            bid_insertion_level.lots = bid.lots
            bid_insertion_level.norders = bid.norders

    cdef inline void clear(self):
        """Clear all levels from both sides of the orderbook."""
        self._bids.reset()
        self._asks.reset()

    cdef inline void consume_snapshot(self, OrderbookLevels new_asks, OrderbookLevels new_bids):
        """Replace the entire orderbook state with new snapshot data."""
        cdef:
            OrderbookLadderData* bids_data = self._bids.get_data()
            OrderbookLadderData* asks_data = self._asks.get_data()
            u64 i, copy_n

        self._normalize_incoming_levels(new_asks, new_bids, True)

        # Asks: direct assignment without intermediate copies, set count once
        copy_n = new_asks.num_levels if new_asks.num_levels <= asks_data.max_levels else asks_data.max_levels
        for i in range(copy_n):
            asks_data.levels[i] = new_asks.levels[i]
        asks_data.num_levels = copy_n

        # Bids: direct assignment without intermediate copies, set count once
        copy_n = new_bids.num_levels if new_bids.num_levels <= bids_data.max_levels else bids_data.max_levels
        for i in range(copy_n):
            bids_data.levels[i] = new_bids.levels[i]
        bids_data.num_levels = copy_n

    cdef inline void consume_deltas(self, OrderbookLevels asks, OrderbookLevels bids):
        """Apply incremental delta updates to the orderbook."""
        if not self._check_if_empty():
            return

        cdef:
            OrderbookLadderData* bids_data = self._bids.get_data()
            OrderbookLadderData* asks_data = self._asks.get_data()
            OrderbookLevel* ask_level
            OrderbookLevel* bid_level
            OrderbookLevel* target  # Cached pointer for multi-field updates
            u64 best_bid_ticks, best_ask_ticks
            u64 worst_bid_ticks, worst_ask_ticks
            u64 ask_count, bid_count  # Cached counts for inner loops
            bint has_bid_replacements = False
            bint has_ask_replacements = False
            u64 j
            u64 insert_idx
            u64 ask_idx
            u64 i

        self._normalize_incoming_levels(asks, bids, False)

        best_bid_ticks = bids_data.levels[0].ticks
        best_ask_ticks = asks_data.levels[0].ticks
        worst_bid_ticks = bids_data.levels[bids_data.num_levels - 1].ticks
        worst_ask_ticks = asks_data.levels[asks_data.num_levels - 1].ticks

        for j in range(bids.num_levels):
            if bids.levels[j].lots != 0:
                has_bid_replacements = True
                break
        for j in range(asks.num_levels):
            if asks.levels[j].lots != 0:
                has_ask_replacements = True
                break

        # Reject deltas that would wipe the opposite side without replacements.
        if asks.num_levels > 0:
            if asks.levels[0].ticks < best_ask_ticks and asks.levels[0].ticks <= worst_bid_ticks:
                if not has_bid_replacements:
                    return

        if bids.num_levels > 0:
            if bids.levels[0].ticks > best_bid_ticks and bids.levels[0].ticks >= worst_ask_ticks:
                if not has_ask_replacements:
                    return

        i = 0
        while i < asks.num_levels:
            ask_level = &asks.levels[i]
            if ask_level.ticks < best_ask_ticks:
                self._process_lower_ask_ticks(ask_level)
                best_ask_ticks = ask_level.ticks
                i += 1
            else:
                break

        if i < asks.num_levels:
            ask_level = &asks.levels[i]
            if ask_level.ticks == best_ask_ticks:
                self._process_matching_ask_ticks(ask_level)
                if asks_data.num_levels > 0:
                    best_ask_ticks = asks_data.levels[0].ticks
                i += 1

        ask_count = asks_data.num_levels
        if ask_count == 0:
            if i < asks.num_levels:
                insert_idx = 0
                for j in range(i, asks.num_levels):
                    ask_level = &asks.levels[j]
                    if ask_level.lots == 0:
                        continue
                    asks_data.levels[insert_idx] = ask_level[0]
                    insert_idx += 1
                    if insert_idx == asks_data.max_levels:
                        break
                asks_data.num_levels = insert_idx

                while bids_data.num_levels > 0 and asks_data.num_levels > 0:
                    if asks_data.levels[0].ticks > bids_data.levels[0].ticks:
                        break
                    self._bids.roll_left(0)
                    self._bids.decrement_count()
            i = asks.num_levels
        else:
            worst_ask_ticks = asks_data.levels[ask_count - 1].ticks

            ask_idx = 0
            while i < asks.num_levels:
                ask_level = &asks.levels[i]
                ask_count = asks_data.num_levels
                if ask_level.ticks > worst_ask_ticks and ask_count == asks_data.max_levels:
                    break
                while ask_idx < ask_count and asks_data.levels[ask_idx].ticks < ask_level.ticks:
                    ask_idx += 1
                if ask_idx < ask_count and asks_data.levels[ask_idx].ticks == ask_level.ticks:
                    if ask_level.lots == 0:
                        self._asks.roll_left(ask_idx)
                        self._asks.decrement_count()
                    else:
                        target = &asks_data.levels[ask_idx]
                        target.size = ask_level.size
                        target.lots = ask_level.lots
                        target.norders = ask_level.norders
                else:
                    if ask_level.lots != 0:
                        self._asks.roll_right(ask_idx)
                        self._asks.increment_count()
                        target = &asks_data.levels[ask_idx]
                        target.price = ask_level.price
                        target.ticks = ask_level.ticks
                        target.size = ask_level.size
                        target.lots = ask_level.lots
                        target.norders = ask_level.norders
                        ask_idx += 1
                ask_count = asks_data.num_levels
                if ask_count > 0:
                    worst_ask_ticks = asks_data.levels[ask_count - 1].ticks
                i += 1

        bid_count = bids_data.num_levels
        if bid_count == 0:
            if bids.num_levels == 0:
                return
            insert_idx = 0
            for i in range(bids.num_levels):
                bid_level = &bids.levels[i]
                if bid_level.lots == 0:
                    continue
                bids_data.levels[insert_idx] = bid_level[0]
                insert_idx += 1
                if insert_idx == bids_data.max_levels:
                    break
            bids_data.num_levels = insert_idx
            while bids_data.num_levels > 0 and asks_data.num_levels > 0:
                if bids_data.levels[0].ticks < asks_data.levels[0].ticks:
                    break
                self._asks.roll_left(0)
                self._asks.decrement_count()
            return

        best_bid_ticks = bids_data.levels[0].ticks
        worst_bid_ticks = bids_data.levels[bid_count - 1].ticks

        i = 0
        if i < bids.num_levels:
            bid_level = &bids.levels[i]
            if bid_level.ticks > best_bid_ticks:
                self._process_higher_bid_ticks(bid_level)
                best_bid_ticks = bid_level.ticks
                i += 1
        while i < bids.num_levels:
            bid_level = &bids.levels[i]
            if bid_level.ticks == best_bid_ticks:
                self._process_matching_bid_ticks(bid_level)
                if bids_data.num_levels > 0:
                    best_bid_ticks = bids_data.levels[0].ticks
                i += 1
            else:
                break

        bid_count = bids_data.num_levels
        if bid_count == 0:
            if i < bids.num_levels:
                insert_idx = 0
                for j in range(i, bids.num_levels):
                    bid_level = &bids.levels[j]
                    if bid_level.lots == 0:
                        continue
                    bids_data.levels[insert_idx] = bid_level[0]
                    insert_idx += 1
                    if insert_idx == bids_data.max_levels:
                        break
                bids_data.num_levels = insert_idx

                while bids_data.num_levels > 0 and asks_data.num_levels > 0:
                    if bids_data.levels[0].ticks < asks_data.levels[0].ticks:
                        break
                    self._asks.roll_left(0)
                    self._asks.decrement_count()
            return
        if bid_count > 0:
            worst_bid_ticks = bids_data.levels[bid_count - 1].ticks

        cdef u64 bid_idx = 0
        while i < bids.num_levels:
            bid_level = &bids.levels[i]
            bid_count = bids_data.num_levels
            if bid_level.ticks < worst_bid_ticks and bid_count == bids_data.max_levels:
                break
            while bid_idx < bid_count and bids_data.levels[bid_idx].ticks > bid_level.ticks:
                bid_idx += 1
            if bid_idx < bid_count and bids_data.levels[bid_idx].ticks == bid_level.ticks:
                if bid_level.lots == 0:
                    self._bids.roll_left(bid_idx)
                    self._bids.decrement_count()
                else:
                    target = &bids_data.levels[bid_idx]
                    target.size = bid_level.size
                    target.lots = bid_level.lots
                    target.norders = bid_level.norders
            else:
                if bid_level.lots != 0:
                    self._bids.roll_right(bid_idx)
                    self._bids.increment_count()
                    target = &bids_data.levels[bid_idx]
                    target.price = bid_level.price
                    target.ticks = bid_level.ticks
                    target.size = bid_level.size
                    target.lots = bid_level.lots
                    target.norders = bid_level.norders
                    bid_idx += 1
            bid_count = bids_data.num_levels
            if bid_count > 0:
                worst_bid_ticks = bids_data.levels[bid_count - 1].ticks
            i += 1

    cdef inline void _assign_bbo_level(
        self,
        OrderbookLevel* target,
        OrderbookLevel* source,
        u64 ticks,
        u64 lots,
    ) noexcept nogil:
        """Assign BBO level fields from source to target with converted ticks/lots."""
        target.price = source.price
        target.ticks = ticks
        target.size = source.size
        target.lots = lots
        target.norders = source.norders

    cdef inline void consume_bbo(self, OrderbookLevel ask, OrderbookLevel bid):
        """Update only the best bid and offer (top of book)."""
        if not self._check_if_empty():
            return

        cdef:
            OrderbookLadderData* asks_data = self._asks.get_data()
            OrderbookLadderData* bids_data = self._bids.get_data()
            OrderbookLevel* top
            u64 ask_ticks, bid_ticks, ask_lots, bid_lots

        ask_ticks = convert_price_to_tick_fast(ask.price, self._tick_size_recip)
        bid_ticks = convert_price_to_tick_fast(bid.price, self._tick_size_recip)
        ask_lots = convert_size_to_lot_fast(ask.size, self._lot_size_recip)
        bid_lots = convert_size_to_lot_fast(bid.size, self._lot_size_recip)

        # Process ask side
        if asks_data.num_levels > 0:
            top = &asks_data.levels[0]
            if ask_lots == 0 and ask_ticks == top.ticks:
                self._asks.roll_left(0)
                self._asks.decrement_count()
            elif top.ticks == ask_ticks:
                top.size = ask.size
                top.lots = ask_lots
                top.norders = ask.norders
            elif ask_ticks < top.ticks:
                self._asks.roll_right(0)
                self._asks.increment_count()
                self._assign_bbo_level(&asks_data.levels[0], &ask, ask_ticks, ask_lots)
            else:
                self._asks.roll_left(0)
                self._asks.decrement_count()
                if asks_data.num_levels > 0:
                    self._assign_bbo_level(&asks_data.levels[0], &ask, ask_ticks, ask_lots)
        elif ask_lots != 0:
            self._asks.roll_right(0)
            self._asks.increment_count()
            self._assign_bbo_level(&asks_data.levels[0], &ask, ask_ticks, ask_lots)

        # Process bid side
        if bids_data.num_levels > 0:
            top = &bids_data.levels[0]
            if bid_lots == 0 and bid_ticks == top.ticks:
                self._bids.roll_left(0)
                self._bids.decrement_count()
            elif top.ticks == bid_ticks:
                top.size = bid.size
                top.lots = bid_lots
                top.norders = bid.norders
            elif bid_ticks > top.ticks:
                self._bids.roll_right(0)
                self._bids.increment_count()
                self._assign_bbo_level(&bids_data.levels[0], &bid, bid_ticks, bid_lots)
            else:
                self._bids.roll_left(0)
                self._bids.decrement_count()
                if bids_data.num_levels > 0:
                    self._assign_bbo_level(&bids_data.levels[0], &bid, bid_ticks, bid_lots)
        elif bid_lots != 0:
            self._bids.roll_right(0)
            self._bids.increment_count()
            self._assign_bbo_level(&bids_data.levels[0], &bid, bid_ticks, bid_lots)

        # Remove crossed bids/asks, but preserve at least the incoming BBO
        while (
            bids_data.num_levels > 0
            and asks_data.num_levels > 0
            and bids_data.levels[0].ticks >= asks_data.levels[0].ticks
        ):
            self._asks.roll_left(0)
            self._asks.decrement_count()

        # If ask side was emptied by cross-removal, restore with incoming ask BBO
        if asks_data.num_levels == 0 and ask_lots != 0:
            self._asks.increment_count()
            self._assign_bbo_level(&asks_data.levels[0], &ask, ask_ticks, ask_lots)

        # If bid side was emptied by cross-removal, restore with incoming bid BBO
        if bids_data.num_levels == 0 and bid_lots != 0:
            self._bids.increment_count()
            self._assign_bbo_level(&bids_data.levels[0], &bid, bid_ticks, bid_lots)

    cdef inline double get_mid_price(self):
        """Calculate the mid price from best bid and ask."""
        self._ensure_not_empty()
        cdef:
            OrderbookLadderData* bids_data = self._bids_data
            OrderbookLadderData* asks_data = self._asks_data
            u64 bid_ticks = bids_data.levels[0].ticks
            u64 ask_ticks = asks_data.levels[0].ticks
        return convert_price_from_tick(
            tick=(bid_ticks + ask_ticks) // 2,
            tick_size=self._tick_size,
        )

    cdef inline double get_bbo_spread(self):
        """Calculate the spread between best bid and ask."""
        self._ensure_not_empty()
        cdef:
            OrderbookLadderData* bids_data = self._bids_data
            OrderbookLadderData* asks_data = self._asks_data
            u64 bid_ticks = bids_data.levels[0].ticks
            u64 ask_ticks = asks_data.levels[0].ticks
        return convert_price_from_tick(
            tick=ask_ticks - bid_ticks,
            tick_size=self._tick_size,
        )

    cdef inline double get_wmid_price(self):
        """Calculate weighted mid price using best bid/ask volumes."""
        self._ensure_not_empty()
        cdef:
            OrderbookLadderData* bids_data = self._bids_data
            OrderbookLadderData* asks_data = self._asks_data
            u64 bid_ticks = bids_data.levels[0].ticks
            u64 bid_lots = bids_data.levels[0].lots
            u64 ask_ticks = asks_data.levels[0].ticks
            u64 ask_lots = asks_data.levels[0].lots
            u64 total_lots = bid_lots + ask_lots
        if total_lots == 0:
            return 0.0
        return convert_price_from_tick(
            tick=(bid_ticks * bid_lots + ask_ticks * ask_lots) // total_lots,
            tick_size=self._tick_size,
        )

    cdef inline double get_volume_weighted_mid_price(self, double size, bint is_base_currency):
        """Calculate volume-weighted mid price for a given trade size."""
        self._ensure_not_empty()
        cdef double mid_price = self.get_mid_price()
        if size <= 0.0:
            return mid_price
        cdef:
            double target = size if is_base_currency else (size / mid_price)
            u64 target_lots = convert_size_to_lot(target, self._lot_size)
            OrderbookLadderData* asks_data = self._asks_data
            OrderbookLadderData* bids_data = self._bids_data
            u64 cum_ask_lots = 0
            u64 cum_bid_lots = 0
            u64 final_buy_ticks = 0
            u64 final_sell_ticks = 0
            u64 i
        for i in range(asks_data.num_levels):
            cum_ask_lots += asks_data.levels[i].lots
            if cum_ask_lots >= target_lots:
                final_buy_ticks = asks_data.levels[i].ticks
                break
        for i in range(bids_data.num_levels):
            cum_bid_lots += bids_data.levels[i].lots
            if cum_bid_lots >= target_lots:
                final_sell_ticks = bids_data.levels[i].ticks
                break
        if final_buy_ticks == 0 or final_sell_ticks == 0:
            return INFINITY_DOUBLE
        return convert_price_from_tick((final_buy_ticks + final_sell_ticks) // 2, self._tick_size)

    cdef inline double get_price_impact(self, double size, bint is_buy, bint is_base_currency):
        """Calculate terminal touch-relative impact for a trade of given size."""
        self._ensure_not_empty()
        if size <= 0.0:
            return 0.0
        cdef:
            OrderbookLadderData* side_data = self._asks_data if is_buy else self._bids_data
            u64 touch_anchor_ticks = side_data.levels[0].ticks
            double touch_anchor_price = convert_price_from_tick(
                touch_anchor_ticks,
                self._tick_size,
            )
            double target_base = size if is_base_currency else (size / touch_anchor_price)
            u64 target_lots = convert_size_to_lot(target_base, self._lot_size)
            u64 remaining_lots = target_lots
            u64 consumed_lots, available_lots
            u64 last_touched_ticks = touch_anchor_ticks
            u64 i
        if target_lots == 0:
            return 0.0
        for i in range(side_data.num_levels):
            available_lots = side_data.levels[i].lots
            consumed_lots = available_lots if available_lots < remaining_lots else remaining_lots
            if consumed_lots > 0:
                last_touched_ticks = side_data.levels[i].ticks
            remaining_lots -= consumed_lots
            if remaining_lots == 0:
                break
        if remaining_lots > 0:
            return INFINITY_DOUBLE
        return abs(
            convert_price_from_tick(last_touched_ticks, self._tick_size) - touch_anchor_price
        )

    cdef inline double get_size_for_price_impact_bps(
        self,
        double impact_bps,
        bint is_buy,
        bint is_base_currency,
    ):
        """Get cumulative size available within a touch-anchored impact band."""
        self._ensure_not_empty()
        if impact_bps <= 0.0:
            return 0.0
        cdef:
            OrderbookLadderData* side_data = self._asks_data if is_buy else self._bids_data
            u64 touch_anchor_ticks = side_data.levels[0].ticks
            double touch_anchor_price = convert_price_from_tick(touch_anchor_ticks, self._tick_size)
            double limit_price
            u64 limit_ticks
            u64 ticks
            u64 lots
            u64 total_lots = 0
            u64 total_ticks_times_lots = 0
            u64 i
        if is_buy:
            limit_price = touch_anchor_price * (1.0 + impact_bps / 10_000.0)
            limit_ticks = convert_price_to_tick_fast(limit_price, self._tick_size_recip)
            for i in range(side_data.num_levels):
                ticks = side_data.levels[i].ticks
                if ticks > limit_ticks:
                    break
                lots = side_data.levels[i].lots
                total_lots += lots
                total_ticks_times_lots += ticks * lots
        else:
            limit_price = touch_anchor_price * (1.0 - impact_bps / 10_000.0)
            limit_ticks = convert_price_to_tick_fast(limit_price, self._tick_size_recip)
            if convert_price_from_tick(limit_ticks, self._tick_size) < limit_price:
                limit_ticks += 1
            for i in range(side_data.num_levels):
                ticks = side_data.levels[i].ticks
                if ticks < limit_ticks:
                    break
                lots = side_data.levels[i].lots
                total_lots += lots
                total_ticks_times_lots += ticks * lots
        if is_base_currency:
            return convert_size_from_lot(total_lots, self._lot_size)
        return (self._tick_size * self._lot_size) * <double> total_ticks_times_lots

    cdef inline bint is_bbo_crossed(self, double other_bid_price, double other_ask_price):
        """Check if this orderbook's BBO crosses with another orderbook's BBO."""
        self._ensure_not_empty()
        cdef:
            OrderbookLadderData* bids_data = self._bids_data
            OrderbookLadderData* asks_data = self._asks_data
            u64 my_bid_ticks = bids_data.levels[0].ticks
            u64 my_ask_ticks = asks_data.levels[0].ticks
            u64 other_bid_ticks = convert_price_to_tick(other_bid_price, self._tick_size)
            u64 other_ask_ticks = convert_price_to_tick(other_ask_price, self._tick_size)
        return my_bid_ticks > other_ask_ticks or my_ask_ticks < other_bid_ticks

    cdef inline bint does_bbo_price_change(self, double bid_price, double ask_price):
        """Check if the given prices differ from current BBO."""
        self._ensure_not_empty()
        cdef:
            OrderbookLadderData* bids_data = self._bids_data
            OrderbookLadderData* asks_data = self._asks_data
            u64 my_bid_ticks = bids_data.levels[0].ticks
            u64 my_ask_ticks = asks_data.levels[0].ticks
            u64 other_bid_ticks = convert_price_to_tick(bid_price, self._tick_size)
            u64 other_ask_ticks = convert_price_to_tick(ask_price, self._tick_size)
        return my_bid_ticks != other_bid_ticks or my_ask_ticks != other_ask_ticks

    cdef inline OrderbookLadderData* get_bids_data(self) noexcept:
        """Get the bids ladder data.

        Returns:
            Pointer to OrderbookLadderData for bids
        """
        return self._bids_data

    cdef inline OrderbookLadderData* get_asks_data(self) noexcept:
        """Get the asks ladder data.

        Returns:
            Pointer to OrderbookLadderData for asks
        """
        return self._asks_data

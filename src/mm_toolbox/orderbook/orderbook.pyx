from libc.stdint cimport (
    uint16_t as u16,
    int16_t as i16
)
from libc.stdlib cimport malloc, free

cdef struct OrderbookLevel:
    double  px
    double  sz
    u16     num_orders

cpdef OrderbookLevel make_orderbook_level(
    double px, 
    double sz, 
    u16 num_orders
):
    return OrderbookLevel(
        px=px, 
        sz=sz, 
        num_orders=num_orders
    )

cdef struct OrderbookSnapshot:
    u16 num_levels
    OrderbookLevel* bids
    OrderbookLevel* asks

cpdef OrderbookSnapshot make_orderbook_snapshot(
    u16 num_levels,
    OrderbookLevel* bids,
    OrderbookLevel* asks,
):
    return OrderbookSnapshot(
        num_levels=num_levels,
        bids=bids,
        asks=asks,
    )

cdef class Orderbook:
    def __init__(
        self,   
        double  tick_sz, 
        u16     max_levels=500, 
    ):
        if max_levels <= 5:
            raise ValueError(f"Invalid max_levels; expected >5 but got...fuck all, whats the point?!")

        self._tick_sz = tick_sz
        self._max_levels = max_levels
        self._size = 0

        cdef OrderbookLevel* bids = <OrderbookLevel*>malloc(self._max_levels * sizeof(OrderbookLevel))
        cdef OrderbookLevel* asks = <OrderbookLevel*>malloc(self._max_levels * sizeof(OrderbookLevel))
        if bids is NULL or asks is NULL:
            raise MemoryError("Failed to allocate memory for orderbook levels")

        self._bids = bids
        self._asks = asks

        self._is_populated = False

    cdef inline void _reset(self):
        cdef i16 i
        for i in range(self._max_levels):
            self._asks[i].px = 0.0
            self._asks[i].sz = 0.0
            self._asks[i].num_orders = 0
            self._bids[i].px = 0.0
            self._bids[i].sz = 0.0
            self._bids[i].num_orders = 0

        self._is_populated = False

    cdef inline void _roll_bids(
        self,
        u16 start_idx = 0,
        bint shift_right = True
    ):
        """
        Roll bids right or left from `start_idx`, creating space for a new
        (price, size) entry to be inserted into that open index.

        Example for shift_right = True:
            Suppose self._bids = [[65, 1], [64, 1], [62, 2], [59, 1]]
            and we want to insert [63, 1] at i=2.

            Rolling the array right from start_idx=2 produces:
                [[65, 1], [64, 1], [62, 2], [62, 2]] 
            then we overwrite start_idx=2 with [63, 1]:
                [[65, 1], [64, 1], [63, 1], [62, 2]]

        Example for shift_right = False:
            If the best bid at start_idx=0 is removed, we shift everything left from start_idx=0:
                old: [[65, 1], [64, 1], [62, 2], [59, 1]]
                new: [[64, 1], [62, 2], [59, 1], [??, ??]] 
            and you may fill the final row with dummy (e.g., px=(old last px - 1 tick), sz=0).
        """
        cdef: 
            Py_ssize_t i
            Py_ssize_t end = self._size

        if shift_right:
            # Move rows from the end down to start_idx+1
            for i in range(end - 1, start_idx, -1):
                self._bids[i, 0] = self._bids[i - 1, 0]
                self._bids[i, 1] = self._bids[i - 1, 1]
        else:
            # Shift everything left, overwriting row i with row i+1
            for i in range(start_idx, end - 1):
                self._bids[i, 0] = self._bids[i + 1, 0]
                self._bids[i, 1] = self._bids[i + 1, 1]

    cdef inline void _roll_asks(
        self,
        Py_ssize_t start_idx = 0,
        bint shift_right = True
    ):
        """
        Roll asks right or left from 'start_idx', creating space for a new 
        price & size to be inserted or removing the best ask.

        Example (shift_right = True):
            Suppose self._asks = [[100, 1], [101, 2], [103, 1], [105, 1]]
            and we want to insert [102, 1] at start_idx=2.

            Rolling the array right from start_idx=2 results in:
                [[100, 1], [101, 2], [103, 1], [103, 1]]
            then we overwrite row=2 with [102, 1]:
                [[100, 1], [101, 2], [102, 1], [103, 1]]

        Example (shift_right = False):
            If removing the best ask at start_idx=0, we shift everything left:
                old: [[100, 1], [101, 2], [103, 1], [105, 1]]
                new: [[101, 2], [103, 1], [105, 1], [??, ??]]
            and you may fill the final row with dummy (e.g., px=(old last px + 1 tick), sz=0).
        """
        cdef: 
            Py_ssize_t i
            Py_ssize_t end = self._size

        if shift_right:
            # Shift rows from the end downward to start_idx+1
            for i in range(end - 1, start_idx, -1):
                self._asks[i, 0] = self._asks[i - 1, 0]
                self._asks[i, 1] = self._asks[i - 1, 1]

        else:
            # Shift everything left, overwriting row i with row i+1
            for i in range(start_idx, end - 1):
                self._asks[i, 0] = self._asks[i + 1, 0]
                self._asks[i, 1] = self._asks[i + 1, 1]

    cdef inline void _process_matching_ask(self, double ask_sz):
        """
        Update or remove the current best ask based on the new size.

        Args:
            ask_sz (float): The updated size for the best ask price. If 0.0,
                the best ask level is removed and shifted.

        Notes:
            - If `ask_sz == 0.0`, we shift the entire asks array left from index 0.
            We then set the final row's price to 1 tick higher than the new best ask
            and its size to 0.
            - Otherwise, if `ask_sz > 0.0`, we simply update the existing best ask
            size to `ask_sz`.
        """
        cdef Py_ssize_t last_idx = self._size - 1

        if ask_sz == 0.0:
            # Some feeds rarely send size=0 for best ask, 
            # but we handle it just in case.
            self._roll_asks(start_idx=0, shift_right=False)
            self._asks[last_idx, 0] = self._asks[last_idx - 1, 0] + self._tick_size
            self._asks[last_idx, 1] = 0.0
        else:
            self._asks[0, 1] = ask_sz

    cdef inline void _process_matching_bid(self, double bid_sz):
        """
        Update or remove the current best bid based on the new size.

        Args:
            bid_sz (float): The updated size for the best bid price. If 0.0,
                the best bid level is removed and shifted.

        Notes:
            - If `bid_sz == 0.0`, we shift the entire bids array left from index 0.
            We then set the final row's price to 1 tick lower than the new best bid
            and its size to 0.
            - Otherwise, if `bid_sz > 0.0`, we simply update the existing best bid
            size to `bid_sz`.
        """
        cdef Py_ssize_t last_idx = self._size - 1

        if bid_sz == 0.0:
            # Some feeds rarely send size=0 for best bid, 
            # but we handle it just in case.
            self._roll_bids(start_idx=0, shift_right=False)
            self._bids[last_idx, 0] = self._bids[last_idx - 1, 0] - self._tick_size
            self._bids[last_idx, 1] = 0.0
        else:
            self._bids[0, 1] = bid_sz

    cdef inline void _process_middle_ask(self, double ask_px, double ask_sz):
        """
        Insert, update, or remove an ask level in the middle of the asks array.

        We assume:
        - best_ask < ask_px <= worst_ask
        - The array is sorted ascending by price
        - All higher prices were already skipped for efficiency

        Args:
            ask_px (float): The ask price to insert/update/remove.
            ask_sz (float): The new size. If 0.0, remove the level.

        Notes:
            - We do a linear scan from i=1 to i=(self._size - 2), 
            because index 0 is best ask and is handled separately.
            - If we find the exact price, we update or remove.
            - If we find a strictly higher price, we insert just before it.
            - If we never find a higher or equal price, 
            we handle fallback logic at the end.
        """
        cdef:
            Py_ssize_t last_idx = self._size - 1
            Py_ssize_t i
            double curr_px
            Py_ssize_t insert_idx = -1
            bint is_matching = False
            bint remove_level = (ask_sz == 0.0)

        # Scan from i=1 up to last_idx
        for i in range(1, last_idx):
            curr_px = self._asks[i, 0]
            if curr_px >= ask_px:
                if curr_px == ask_px:
                    insert_idx = i
                    is_matching = True
                else:
                    # Insert just before i
                    insert_idx = i - 1
                break

        # If we never broke out, i.e. we never found curr_px >= ask_px.
        # Technically impossible case, ideally throw error for logging.
        if insert_idx == -1:
            # raise IndexError(f"Orderbook insertion error")
            return

        # Handle remove/update/insert. As these are bints, Cython already
        # compiles this to native switch-case statements so no need for 
        # further optimization (reordering the statements etc).
        if is_matching and remove_level:
            # Remove the level by rolling array left from insert_idx
            # then fill the last row with a dummy (price + tick_size, 0.0).
            self._roll_asks(insert_idx, shift_right=False)
            self._asks[last_idx, 0] = self._asks[last_idx - 1, 0] + self._tick_size
            self._asks[last_idx, 1] = 0.0

        elif is_matching and not remove_level:
            # Just update the size of the existing price.
            self._asks[insert_idx, 1] = ask_sz

        elif not is_matching and not remove_level:
            # Insert a brand-new price by rolling array right from insert_idx.
            self._roll_asks(insert_idx, shift_right=True)
            self._asks[insert_idx, 0] = ask_px
            self._asks[insert_idx, 1] = ask_sz

        else:
            # Not matching & remove_level => do nothing, or handle if needed.
            pass

    cdef inline void _process_middle_bid(self, double bid_px, double bid_sz):
        """
        Insert, update, or remove a bid level in the middle of the bids array.

        Assumptions:
        - The array is sorted in descending order by price.
        - We skip all bids that are bigger than best_bid or smaller than worst_bid in earlier checks.
        - For an exact price match, we update or remove that level.
        - Otherwise, we insert just after the next-lower price.

        Args:
            bid_px (float): The bid price to be inserted, updated, or removed.
            bid_sz (float): The new size. If 0.0, remove the level.

        Notes:
            - We do a linear scan from i=1 to i=(self._size - 2),
            skipping i=0 because it's the best bid (handled separately).
            - If we find a price <= bid_px, we break.
            - If we never break, we fallback to i=(last_idx - 1).
        """
        cdef:
            Py_ssize_t last_idx = self._size - 1
            Py_ssize_t i
            double curr_px
            Py_ssize_t insert_idx = -1
            bint is_matching = False
            bint remove_level = (bid_sz == 0.0)

        # Scan from i=1 up to last_idx
        for i in range(1, last_idx):
            curr_px = self._bids[i, 0]
            if curr_px <= bid_px:
                if curr_px == bid_px:
                    insert_idx = i
                    is_matching = True
                else:
                    # Insert just after i
                    insert_idx = i + 1
                break

        # If we never broke out, i.e. we never found curr_px >= ask_px.
        # Technically impossible case, ideally throw error for logging.
        if insert_idx == -1:
            # raise IndexError(f"Orderbook insertion error")
            return

        # Handle remove/update/insert. As these are bints, Cython already
        # compiles this to native switch-case statements so no need for 
        # further optimization (reordering the statements etc).
        if is_matching and remove_level:
            # Remove the level by rolling array left from insert_idx.
            self._roll_bids(insert_idx, shift_right=False)
            self._bids[last_idx, 0] = self._bids[last_idx - 1, 0] - self._tick_size
            self._bids[last_idx, 1] = 0.0

        elif is_matching and not remove_level:
            # Just update the size.
            self._bids[insert_idx, 1] = bid_sz

        elif not is_matching and not remove_level:
            # Insert a brand-new price by rolling array right from insert_idx.
            self._roll_bids(insert_idx, shift_right=True)
            self._bids[insert_idx, 0] = bid_px
            self._bids[insert_idx, 1] = bid_sz

        else:
            # Not matching & remove_level => do nothing, or handle if needed.
            pass

    cdef inline void _process_lower_ask(self, double ask_px, double ask_sz):
        """
        Insert a new best ask at the front (index 0) of the asks array,
        shifting existing asks to the right.

        If the new best ask overlaps with the best bid, remove those bid rows
        until there is no overlap.

        Args:
            ask_px (float): The new best ask price.
            ask_sz (float): The new best ask size.
        """
        cdef Py_ssize_t last_idx = self._size - 1

        # Roll the asks array right from index=0.
        self._roll_asks(start_idx=0, shift_right=True)

        # Place the new best ask.
        self._asks[0, 0] = ask_px
        self._asks[0, 1] = ask_sz

        # If ask_px <= best bid, remove those bid rows to avoid overlap
        #    (In future, consider batch rolling for efficiency.)
        while ask_px <= self._bids[0, 0]:
            self._roll_bids(start_idx=0, shift_right=False)
            self._bids[last_idx, 0] = self._bids[last_idx - 1, 0] - self._tick_size
            self._bids[last_idx, 1] = 0.0

    cdef inline void _process_higher_bid(self, double bid_px, double bid_sz):
        """
        Insert a new best bid at the front (index 0) of the bids array,
        shifting existing bids to the right.

        If the new best bid overlaps with the best ask, remove those ask rows
        until there is no overlap.

        Args:
            bid_px (float): The new best bid price.
            bid_sz (float): The new best bid size.
        """
        cdef Py_ssize_t last_idx = self._size - 1

        # 1) Roll the bids array right from index=0.
        self._roll_bids(start_idx=0, shift_right=True)

        # 2) Place the new best bid.
        self._bids[0, 0] = bid_px
        self._bids[0, 1] = bid_sz

        # 3) If bid_px >= best ask, remove those ask rows to avoid overlap
        #    (In future, consider batch rolling for efficiency.)
        while bid_px >= self._asks[0, 0]:
            self._roll_asks(start_idx=0, shift_right=False)
            self._asks[last_idx, 0] = self._asks[last_idx - 1, 0] + self._tick_size
            self._asks[last_idx, 1] = 0.0

    cdef inline void _process_higher_ask(self, double ask_px, double ask_sz):
        """
        We simply ignore asks outside the handled price region.
        """
        return 

    cdef inline void _process_lower_bid(self, double bid_px, double bid_sz):
        """
        We simply ignore bids outside the handled price region.
        """
        return 

    cpdef void consume_snapshot(self, OrderbookSnapshot snapshot):
        """
        Refresh the order book with new bid and ask levels.
        """
        self._reset()

        cdef:
            u16 num_levels = snapshot.num_levels
            OrderbookLevel* new_asks = snapshot.asks
            OrderbookLevel* new_bids = snapshot.bids

        self._asks = new_asks
        self._bids = new_bids
        self._is_populated = True

    cpdef void update_bbo(
        self,
        double bid_px,
        double bid_sz,
        double ask_px,
        double ask_sz
    ):
        """
        Update the best bid and best ask (BBO) based on a new quote.

        Args:
            bid_px (float): The new best bid price.
            bid_sz (float): The new best bid size.
            ask_px (float): The new best ask price.
            ask_sz (float): The new best ask size.

        Notes:
            - We first check if the new bid/ask matches the existing best 
            bid/ask. If both match, we simply update their sizes. 
            If both matched, we return early, since no other changes 
            would apply.
            - Otherwise, if the new bid is higher, we insert a 'higher bid'. 
            If the new ask is lower, we insert a 'lower ask'.
        """
        self.ensure_populated()

        cdef:
            double best_ask_px = self._asks[0, 0]
            double best_bid_px = self._bids[0, 0]

            # If both bid+ask prices match existing ones, 
            # this will count up to 2. If it does, we can 
            # do an return to save time
            Py_ssize_t bbo_px_match = 0

        # 1) Check if the new bid matches the current best bid
        if best_bid_px == bid_px:
            self._process_matching_bid(bid_sz)
            bbo_px_match += 1

        # 2) Check if the new ask matches the current best ask
        if best_ask_px == ask_px:
            self._process_matching_ask(ask_sz)
            bbo_px_match += 1

        # If both matched, we updated them and can leave
        if bbo_px_match == 2:
            return

        # 3) If new bid is higher than best bid
        if bid_px > best_bid_px:
            self._process_higher_bid(bid_px, bid_sz)

        # 4) If new ask is lower than best ask
        if ask_px < best_ask_px:
            self._process_lower_ask(ask_px, ask_sz)

    cpdef void update(
        self, 
        list[OrderbookLevel] asks,
        list[OrderbookLevel] bids
    ):
        """
        Update both the asks and the bids of the orderbook using new arrays.

        Args:
            updated_asks (cnp.ndarray): A 2D array of ask levels, 
                each row typically [price, size] (plus possibly other columns).
            updated_bids (cnp.ndarray): A 2D array of bid levels,
                each row typically [price, size] (plus possibly other columns).

        Raises:
            IndexError: If the provided arrays are not 2D or have no rows.
        """
        self.ensure_populated()

        cdef:
            double[:, :] new_bids_view = bids
            double[:, :] new_asks_view = asks
            Py_ssize_t new_bids_len = new_bids_view.shape[0]
            Py_ssize_t new_asks_len = new_asks_view.shape[0]
            Py_ssize_t new_bids_dims = new_bids_view.ndim
            Py_ssize_t new_asks_dims = new_asks_view.ndim

            double best_bid_px = self._bids[0, 0]
            double worst_bid_px = self._bids[self._size - 1, 0]
            double best_ask_px = self._asks[0, 0]
            double worst_ask_px = self._asks[self._size - 1, 0]
            double new_ask_px, new_ask_sz, new_bid_px, new_bid_sz
            Py_ssize_t i

        if new_asks_dims != 2:
            raise IndexError(f"Invalid asks dimensions; expected 2D but got {new_asks_dims}D")

        if new_asks_len < 1:
            raise IndexError(f"Invalid asks length; expected >=1 but got {new_asks_len}")

        if new_bids_dims != 2:
            raise IndexError(f"Invalid bids dimensions; expected 2D but got {new_bids_dims}D")

        if new_bids_len < 1:
            raise IndexError(f"Invalid bids length; expected >=1 but got {new_bids_len}")

        # Copy pasted from self.update_asks()
        for i in range(new_asks_len):
            new_ask_px = new_asks_view[i, 0]
            new_ask_sz = new_asks_view[i, 1]

            if new_ask_px < best_ask_px:
                self._process_lower_ask(new_ask_px, new_ask_sz)
                best_ask_px = new_ask_px
                worst_ask_px = self._asks[self._size - 1, 0]

            if new_ask_px == best_ask_px:
                self._process_matching_ask(new_ask_sz)

            elif new_ask_px <= worst_ask_px:
                self._process_middle_ask(new_ask_px, new_ask_sz)
                worst_ask_px = self._asks[self._size - 1, 0]

            else:
                # Price is above worst ask => new "higher ask".
                # In practice, future updates won't change the book. 
                # Might do an early return, but we keep iterating for completeness.
                self._process_higher_ask(new_ask_px, new_ask_sz)
                # break  

        # Copy pasted from self.update_bids()
        for i in range(new_bids_len):
            new_bid_px = new_bids_view[i, 0]
            new_bid_sz = new_bids_view[i, 1]

            if new_bid_px < best_bid_px:
                self._process_lower_bid(new_bid_px, new_bid_sz)
                best_bid_px = self._bids[0, 0]
                worst_bid_px = self._bids[self._size - 1, 0]

            elif new_bid_px == best_bid_px:
                self._process_matching_bid(new_bid_sz)

            elif new_bid_px <= worst_bid_px:
                self._process_middle_bid(new_bid_px, new_bid_sz)
                worst_bid_px = self._bids[self._size - 1, 0]
            else:
                # new_bid_px > worst_bid_px => "higher bid"
                # Typically not relevant for subsequent updates,
                # but we keep iterating for completeness.
                self._process_higher_bid(new_bid_px, new_bid_sz)
                # break

    cpdef double get_mid_px(self):
        """
        Return the midpoint between the best bid and best ask prices.

        Returns:
            float: 
                The midpoint, calculated as (best_bid + best_ask) / 2.0.

        Raises:
            RuntimeError: If the orderbook is not populated (caught via `ensure_populated()`).
        """
        self.ensure_populated()
        cdef double best_bid_px = self._bids[0].px
        cdef double best_ask_px = self._asks[0].px
        return (best_bid_px + best_ask_px) / 2.0

    cpdef double get_wmid_px(self):
        """
        Return a 'weighted mid' price considering size at the top bid and ask.

        This method computes an imbalance ratio based on the top bid size and top ask size, 
        and uses that ratio to do a linear interpolation between best_bid and best_ask prices.

        Returns:
            float:
                The weighted mid, computed as 
                (best_bid_price * imbalance + best_ask_price * (1 - imbalance)),
                where imbalance = top_bid_size / (top_bid_size + top_ask_size).

        Raises:
            RuntimeError: If the orderbook is not populated (caught via `ensure_populated()`).
        """
        self.ensure_populated()

        cdef double top_bid_sz = self._bids[0].sz
        cdef double top_ask_sz = self._asks[0].sz
        cdef double imbalance = top_bid_sz / (top_bid_sz + top_ask_sz)

        return (self._bids[0].px * imbalance) + (self._asks[0].px * (1.0 - imbalance))

    cpdef double get_bbo_spread(self):
        """
        Calculate the spread between the best ask and the best bid.

        Returns:
            float: The difference between the best ask price (asks[0,0])
                and the best bid price (bids[0,0]).
        """
        cdef double best_ask_px = self._asks[0].px
        cdef double best_bid_px = self._bids[0].px
        return best_ask_px - best_bid_px
    
    cpdef double get_vamp(self, double sz, bint is_base_currency=False):
        """
        Calculate the volume-weighted average market price (VAMP) up to a specified 'size' 
        from both bid and ask sides.

        Args:
            size (float): 
                The target size (or 'depth') for which we compute the volume-weighted average price.
            is_base_currency (bool, optional): 
                If True, the 'size' is in base currency units; we convert it to quote 
                currency by multiplying with the mid price. Defaults to False.

        Returns:
            float:
                The volume-weighted average of top bids and asks, up to the 'size' 
                on each side. If 'size' is in base currency, it's converted to quote 
                currency first.

        Raises:
            RuntimeError: 
                - If the orderbook is not populated (caught via `ensure_populated()`).
                - If total cumulative size from both sides ends up zero (cannot compute VAMP).
        """
        self.ensure_populated()

        cdef:
            double sum_weighted_bsize = 0.0
            double sum_weighted_asize = 0.0
            double cum_bsize = 0.0
            double cum_asize = 0.0
            Py_ssize_t bid_iter_idx = 0
            Py_ssize_t ask_iter_idx = 0
            double bprice, bsize
            double aprice, asize

        # Convert base -> quote if needed
        if is_base_currency:
            sz *= self.get_mid_px()  

        # Please dont be this dumb...
        if sz == 0.0:
            return self.get_mid()

        # Bids accumulation (partial fill from top bids)
        while cum_bsize < size and bid_iter_idx < self._size:
            bprice = self._bids_view[bid_iter_idx, 0]
            bsize = self._bids_view[bid_iter_idx, 1]

            if (cum_bsize + bsize) > size:
                remaining_size = size - cum_bsize
                sum_weighted_bsize += bprice * remaining_size
                cum_bsize += remaining_size
                break

            sum_weighted_bsize += bprice * bsize
            cum_bsize += bsize
            bid_iter_idx += 1

        # Asks accumulation (partial fill from top asks)
        while cum_asize < size and ask_iter_idx < self._size:
            aprice = self._asks_view[ask_iter_idx, 0]
            asize = self._asks_view[ask_iter_idx, 1]

            if (cum_asize + asize) > size:
                remaining_size = size - cum_asize
                sum_weighted_asize += aprice * remaining_size
                cum_asize += remaining_size
                break

            sum_weighted_asize += aprice * asize
            cum_asize += asize
            ask_iter_idx += 1

        cdef double total_cum_size = cum_bsize + cum_asize
        if total_cum_size == 0.0:
            raise RuntimeError("Total cumulative size is zero; cannot compute VAMP")

        return (sum_weighted_bsize + sum_weighted_asize) / total_cum_size
    
    cpdef double get_impact(self, double sz, bint is_bid, bint is_base_currency=False):
        """
        Calculate the average impact for a hypothetical order of 'sz' 
        on either the bid or ask side, relative to the mid price.

        Args:
            sz (float): 
                The size of the order for which impact is being calculated.
            is_bid (bool): 
                If True, we treat this as a sell order (filling the bid side). 
                If False, a buy order (filling the ask side).
            is_base_currency (bool, optional): 
                If True, 'sz' is in base currency, converted to quote currency 
                by multiplying with `get_mid()`. Defaults to False.

        Returns:
            float:
                The average impact, defined as the volume-weighted difference 
                from the mid price.

        Raises:
            RuntimeError:
                - If the orderbook is not populated (caught via `ensure_populated()`).
                - If total cumulative size is zero after processing (meaning no liquidity).
        """
        self.ensure_populated()

        cdef:
            double mid_px = self.get_mid()
            double cum_size = 0.0
            double weighted_impact = 0.0
            double level_price, level_size
            double available_size
            Py_ssize_t level = 0

            double[:, :] book = self._bids_view if is_bid else self._asks_view

        # convert base -> quote if needed
        if is_base_currency:
            sz *= mid_px

        # Iterate over levels to fill partial or entire 'size'
        while cum_size < sz and level < self._size:
            level_price = book[level, 0]
            level_size = book[level, 1]

            available_size = min(level_size, sz - cum_size)
            weighted_impact += abs(level_price - mid_px) * available_size

            cum_size += available_size
            level += 1

        if cum_size == 0.0:
            raise RuntimeError("Total cumulative size is zero; cannot compute impact.")

        cdef double impact = weighted_impact / cum_size

        return impact

    cpdef double get_imbalance(self, double depth_pct):
        """
        Compute the volume imbalance of the orderbook within a symmetric 
        price band around the mid price, proportional to `depth_pct`.

        This function:
        1) Calculates mid = (best_bid + best_ask) / 2
        2) Defines a band around mid by a fraction of the spread: 
            spread = best_ask - best_bid
            min_px = mid - (spread * depth_pct / 2)
            max_px = mid + (spread * depth_pct / 2)
        3) Sums volumes of all bid levels with price >= min_px,
            sums volumes of all ask levels with price <= max_px.
        4) Defines imbalance = bid_vol / (bid_vol + ask_vol).

        Args:
            depth_pct (float): 
                A fraction (0 < depth_pct <= 1, typically) indicating the fraction 
                of the half-spread around mid to consider. E.g., depth_pct=0.2 
                means ±10% of the spread around the mid.

        Returns:
            float:
                A ratio between 0 and 1 representing the fraction of volume 
                on the bid side vs. total volume in that band.

        Raises:
            RuntimeError: If the orderbook is not populated or if total volume is zero.
            ValueError: If depth_pct <= 0.
        """
        self.ensure_populated()

        if depth_pct <= 0.0:
            raise ValueError(f"Invalid depth_pct; expected >0 but got {depth_pct}")

        cdef:
            double best_bid = self._bids_view[0, 0]
            double best_ask = self._asks_view[0, 0]
            double spread = self.get_bbo_spread()
            double mid = self.get_mid()

            double min_px = mid - (spread * depth_pct / 2.0)
            double max_px = mid + (spread * depth_pct / 2.0)

            double bid_volume = 0.0
            double ask_volume = 0.0

            Py_ssize_t i

            double px, sz

        # Sum up volumes on the bid side for levels >= min_px
        for i in range(self._size):
            px = self._bids_view[i, 0]
            sz = self._bids_view[i, 1]

            if px < min_px:
                break

            bid_volume += sz

        # Sum up volumes on the ask side for levels <= max_px
        for i in range(self._size):
            px = self._asks_view[i, 0]
            sz = self._asks_view[i, 1]

            if px > max_px:
                break

            ask_volume += sz

        cdef double total_vol = bid_volume + ask_volume
        if total_vol == 0.0:
            raise RuntimeError("Total volume in the defined price band is zero; cannot compute imbalance.")

        return bid_volume / total_vol

    # ----- Public array access methods ----- #

    cpdef cnp.ndarray get_bids(self):
        """
        Return the entire bids array.

        Returns:
            np.ndarray: The NumPy array representing the bid side 
                of the orderbook.
        """
        return self._bids.copy()

    cpdef cnp.ndarray get_asks(self):
        """
        Return the entire asks array.

        Returns:
            np.ndarray: The NumPy array representing the ask side 
                of the orderbook.
        """
        return self._asks.copy()

    cpdef list[OrderbookLevel] get_bbo(self):
        """
        Return the top (best) bid and ask price/size pairs.

        Returns:
            list[OrderbookLevel]: A list of two OrderbookLevel objects, 
                representing the best bid and ask levels.
        """
        return [self._bids[0], self._asks[0]]

    cpdef bint is_crossed(self, Orderbook other):
        """
        Check if this orderbook is crossed with another orderbook.
        
        A cross occurs when:
        - This orderbook's best bid >= other orderbook's best ask, OR
        - Other orderbook's best bid >= this orderbook's best ask

        Args:
            other (Orderbook): The other orderbook to compare against.

        Returns:
            bool: True if the orderbooks are crossed, False otherwise.
        """
        self.ensure_populated()
        
        cdef list[OrderbookLevel] other_bbo = other.get_bbo()
        
        return (self._bids[0].px >= other_bbo[1].px or 
                other_bbo[0].px >= self._asks[0].px) 
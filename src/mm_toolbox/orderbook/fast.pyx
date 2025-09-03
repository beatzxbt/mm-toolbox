# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from numpy cimport (
    ndarray as cndarray,
    PyArray_EMPTY as cPyArray_EMPTY,
    PyArray_BYTES as cPyArray_BYTES,
    PyArray_Cast as cPyArray_Cast,
)
from libc.math cimport floor, fmin
from libc.stdint cimport (
    uint64_t as u64, uint32_t as u32, int64_t as i64
)
from libc.string cimport memmove


# Side view struct (module scope, matches pxd)
ctypedef struct _SideView:
    u64* prices
    u64* sizes
    u32* norders
    u64* count_ptr
    u64  max_levels
    bint ascending


cdef inline void _memmove_u64(u64* base, u64 start, u64 end) noexcept nogil:
    memmove(<void*>(base + start + 1), <const void*>(base + start), (end - start) * sizeof(u64))


cdef inline void _memmove_u32(u32* base, u64 start, u64 end) noexcept nogil:
    memmove(<void*>(base + start + 1), <const void*>(base + start), (end - start) * sizeof(u32))


# Future optimizations:
# - If the user can guarantee that incoming deltas/snapshots are sorted in some order,
#   we can skip the sorting for snapshots, and reduce branching *significantly* when iterating over deltas.


cdef class COrderbook:
    """Array-backed orderbook optimized for update throughput.

    Stores prices (ticks) and sizes (lots) in contiguous arrays for cache-friendly updates.
    """

    def __cinit__(self, double tick_size, double lot_size, u64 max_num_levels=500):
        """
        Args:
            tick_size: size of a tick in real units
            lot_size: size of a lot in real units
            max_num_levels: maximum number of levels to store
        """
        if max_num_levels <= 0:
            raise ValueError(f"Invalid max_num_levels; expected >0 but got {max_num_levels}")
        if 1 <= max_num_levels < 5:
            raise ValueError(
                f"Invalid max_num_levels; expected >=5 but got...{max_num_levels}? What's the point?!"
            )
        self._tick_size = tick_size
        self._lot_size = lot_size
        self._max_levels = max_num_levels
        
        self._num_bids = 0
        self._num_asks = 0

        # PyArray_EMPTY returns a python object, so we need to type cast it to a 
        # C ndarray. This needs to be casted back to a py ndarray if read from outside.
        cdef cnp.npy_intp _dim = <cnp.npy_intp> max_num_levels
        self._bids_price_arr = <cndarray> cPyArray_EMPTY(1, &_dim, cnp.NPY_UINT64, 0)
        self._bids_size_arr = <cndarray> cPyArray_EMPTY(1, &_dim, cnp.NPY_UINT64, 0)
        self._bids_norders_arr = <cndarray> cPyArray_EMPTY(1, &_dim, cnp.NPY_UINT32, 0)
        self._asks_price_arr = <cndarray> cPyArray_EMPTY(1, &_dim, cnp.NPY_UINT64, 0)
        self._asks_size_arr = <cndarray> cPyArray_EMPTY(1, &_dim, cnp.NPY_UINT64, 0)
        self._asks_norders_arr = <cndarray> cPyArray_EMPTY(1, &_dim, cnp.NPY_UINT32, 0)

    cdef inline void __ensure_orderbook_not_empty(self):
        """Raise RuntimeError if orderbook is empty."""
        if self._num_bids == 0 or self._num_asks == 0:
            raise RuntimeError("Empty book; cannot compute without data")

    cdef inline u64 _convert_price_to_ticks(self, double price) nogil:
        return <u64> (floor(price / self._tick_size))

    cdef inline u64 _convert_size_to_lots(self, double size) nogil:
        return <u64> (floor(size / self._lot_size))

    cdef inline cndarray _convert_prices_to_ticks(self, cndarray prices):
        # Remove the np.rint to then add nogil
        return cPyArray_Cast(<cndarray> np.rint(prices / self._tick_size), cnp.NPY_UINT64)

    cdef inline cndarray _convert_sizes_to_lots(self, cndarray sizes):
        # Remove the np.rint to then add nogil
        return cPyArray_Cast(<cndarray> np.rint(sizes / self._lot_size), cnp.NPY_UINT64)

    cdef inline double _convert_ticks_to_real(self, u64 ticks) nogil:
        return <double> (ticks * self._tick_size)

    cdef inline double _convert_lots_to_real(self, u64 lots) nogil:
        return <double> (lots * self._lot_size)

    cdef inline _SideView _make_asks_view(self) nogil:
        cdef _SideView sv
        sv.prices = <u64*> cPyArray_BYTES(self._asks_price_arr)
        sv.sizes = <u64*> cPyArray_BYTES(self._asks_size_arr)
        sv.norders = <u32*> cPyArray_BYTES(self._asks_norders_arr)
        sv.count_ptr = &self._num_asks
        sv.max_levels = self._max_levels
        sv.ascending = True
        return sv

    cdef inline _SideView _make_bids_view(self) nogil:
        cdef _SideView sv
        sv.prices = <u64*> cPyArray_BYTES(self._bids_price_arr)
        sv.sizes = <u64*> cPyArray_BYTES(self._bids_size_arr)
        sv.norders = <u32*> cPyArray_BYTES(self._bids_norders_arr)
        sv.count_ptr = &self._num_bids
        sv.max_levels = self._max_levels
        sv.ascending = False
        return sv

    cdef inline void _sv_roll_right(self, _SideView sv, u64 start_idx) noexcept nogil:
        """
        Shift a side (bids or asks) right by one from ``start_idx`` to make an insert slot.

        Copies elements in [start_idx, N-1] to [start_idx+1, N] across price/size/norders,
        then increments the count by 1. This creates a free slot at ``start_idx`` for the
        caller to overwrite.

        If the side is full (count == max_levels), the last element is dropped (tail-drop)
        to avoid overflow.

        Visual (for bids, but applies to asks as well):
            before (prices): [b0, b1, b2, b3], count=4, start_idx=1
            after  (prices): [b0, [free], b1, b2, b3], count=5

        No-op if ``start_idx >= count``.

        Args:
            sv (_SideView): Pointer to the side view (bids or asks).
            start_idx (u64): Index at which to create the free slot.
        """
        cdef u64 end = sv.count_ptr[0]
        if end <= start_idx:
            return
        if end < sv.max_levels:
            _memmove_u64(sv.prices, start_idx, end)
            _memmove_u64(sv.sizes, start_idx, end)
            _memmove_u32(sv.norders, start_idx, end)
            sv.count_ptr[0] = end + 1
        else:
            # Tail-drop the last element to avoid overflow
            if end - start_idx > 0:
                _memmove_u64(sv.prices, start_idx, end)
                _memmove_u64(sv.sizes, start_idx, end)
                _memmove_u32(sv.norders, start_idx, end)

    cdef inline void _sv_roll_left(self, _SideView sv, u64 start_idx) noexcept nogil:
        """
        Delete an entry at ``start_idx`` by shifting left by one.

        Copies elements in [start_idx+1, N-1] to [start_idx, N-2] across price/size/norders,
        then decrements the count by 1.

        Visual (for asks, but applies to bids as well):
            before (prices): [a0, a1, a2, a3], count=4, start_idx=1
            after  (prices): [a0, a2, a3, ...], count=3

        No-op if ``start_idx >= count``.

        Args:
            sv (_SideView): Pointer to the side view (bids or asks).
            start_idx (u64): Index of the entry to delete.
        """
        cdef u64 end = sv.count_ptr[0]
        if start_idx >= end:
            return
        if end - start_idx > 1:
            _memmove_u64(sv.prices, start_idx, end)
            _memmove_u64(sv.sizes, start_idx, end)
            _memmove_u32(sv.norders, start_idx, end)
            sv.count_ptr[0] = end - 1

    cdef void _process_matching_ask_ticks(self, u64 new_size, u32 new_norder) noexcept nogil:
        """If ``new_size == 0``, remove the current best ask, otherwise update the size."""
        cdef _SideView ask_sv = self._make_asks_view()
        if new_size == 0:
            self._sv_roll_left(ask_sv, 0)
        else:
            ask_sv.sizes[0] = new_size
            ask_sv.norders[0] = new_norder

    cdef void _process_matching_bid_ticks(self, u64 new_size, u32 new_norder) noexcept nogil:
        """If ``new_size == 0``, remove the current best bid, otherwise update the size."""
        cdef _SideView bid_sv = self._make_bids_view()
        if new_size == 0:
            self._sv_roll_left(bid_sv, 0)
        else:
            bid_sv.sizes[0] = new_size
            bid_sv.norders[0] = new_norder

    cdef void _process_lower_ask_ticks(self, u64 new_price, u64 new_size, u32 new_norder) noexcept nogil:
        """
        Insert a new best ask at the front (index 0) of the asks array,
        shifting existing asks to the right.

        If the new best ask overlaps with the best bid, remove those bid rows
        until there is no overlap.
        """
        cdef _SideView ask_sv = self._make_asks_view()
        cdef _SideView bid_sv = self._make_bids_view()
        self._sv_roll_right(ask_sv, 0)
        ask_sv.prices[0] = new_price
        ask_sv.sizes[0] = new_size
        ask_sv.norders[0] = new_norder
        while bid_sv.count_ptr[0] > 0 and new_price <= bid_sv.prices[0]:
            self._sv_roll_left(bid_sv, 0)

    cdef void _process_higher_bid_ticks(self, u64 new_price, u64 new_size, u32 new_norder) noexcept nogil:
        """
        Insert a new best bid at the front (index 0) of the bids array,
        shifting existing bids to the right.

        If the new best bid overlaps with the best ask, remove those ask rows
        until there is no overlap.
        """
        cdef _SideView bid_sv = self._make_bids_view()
        cdef _SideView ask_sv = self._make_asks_view()
        self._sv_roll_right(bid_sv, 0)
        bid_sv.prices[0] = new_price
        bid_sv.sizes[0] = new_size
        bid_sv.norders[0] = new_norder
        while ask_sv.count_ptr[0] > 0 and new_price >= ask_sv.prices[0]:
            self._sv_roll_left(bid_sv, 0)

    cdef void _process_middle_ask_ticks(self, u64 new_price, u64 new_size, u32 new_norder) noexcept nogil:
        """
        Insert, update, or remove an ask level in the middle of the asks array.

        We assume ``best_ask < new_price <= worst_ask``.
        """
        cdef _SideView ask_sv = self._make_asks_view()
        cdef u64 i, curr_price
        cdef u64 insert_idx = 1
        cdef u64 last_idx = ask_sv.count_ptr[0] - 1
        cdef bint is_matching = False

        for i in range(1, last_idx + 1):
            curr_price = ask_sv.prices[i]
            if curr_price >= new_price:
                insert_idx = i
                if curr_price == new_price:
                    is_matching = True
                break

        if is_matching:
            if new_size == 0:
                self._sv_roll_left(ask_sv, insert_idx)
            else:
                ask_sv.sizes[insert_idx] = new_size
                ask_sv.norders[insert_idx] = new_norder
        else:
            # Mismatch between deltas and current snapshot, ignore
            if new_size == 0:
                return
            self._sv_roll_right(ask_sv, insert_idx)
            ask_sv.prices[insert_idx] = new_price
            ask_sv.sizes[insert_idx] = new_size
            ask_sv.norders[insert_idx] = new_norder

    cdef void _process_middle_bid_ticks(self, u64 new_price, u64 new_size, u32 new_norder) noexcept nogil:
        """
        Insert, update, or remove a bid level in the middle of the bids array.

        We assume ``best_bid > new_price >= worst_bid``.
        """
        cdef _SideView bid_sv = self._make_bids_view()
        cdef u64 i, curr_price
        cdef u64 insert_idx = 1
        cdef u64 last_idx = bid_sv.count_ptr[0] - 1
        cdef bint is_matching = False   

        for i in range(1, last_idx + 1):
            curr_price = bid_sv.prices[i]
            if curr_price <= new_price:
                if curr_price == new_price:
                    is_matching = True
                insert_idx = i
                break

        if is_matching:
            if new_size == 0:
                self._sv_roll_left(bid_sv, insert_idx)
            else:
                bid_sv.sizes[insert_idx] = new_size
                bid_sv.norders[insert_idx] = new_norder
        else:
            # Mismatch between deltas and current snapshot, ignore
            if new_size == 0:
                return
            self._sv_roll_right(bid_sv, insert_idx)
            bid_sv.prices[insert_idx] = new_price
            bid_sv.sizes[insert_idx] = new_size
            bid_sv.norders[insert_idx] = new_norder

    cpdef void clear(self):
        """Reset the book to empty state."""
        self._num_bids = 0
        self._num_asks = 0

    cpdef void consume_snapshot_raw(
        self,
        cndarray asks_price_ticks,
        cndarray asks_size_lots,
        cndarray asks_norders,
        cndarray bids_price_ticks,
        cndarray bids_size_lots,
        cndarray bids_norders,
    ):
        """Refresh the book with raw ticks/lots arrays (ascending asks, descending bids)."""
        cdef:
            u64 i
            u64 num_asks = <u64> fmin(<double>asks_price_ticks.shape[0], <double>self._max_levels)
            u64 num_bids = <u64> fmin(<double>bids_price_ticks.shape[0], <double>self._max_levels)
            _SideView ask_sv = self._make_asks_view()
            _SideView bid_sv = self._make_bids_view()

        for i in range(num_asks):
            ask_sv.prices[i] = <u64> asks_price_ticks[i]
            ask_sv.sizes[i] = <u64> asks_size_lots[i]
            ask_sv.norders[i] = <u32> asks_norders[i]

        for i in range(num_bids):
            bid_sv.prices[i] = <u64> bids_price_ticks[i]
            bid_sv.sizes[i] = <u64> bids_size_lots[i]
            bid_sv.norders[i] = <u32> bids_norders[i]

    cpdef void consume_deltas_raw(
        self,
        cndarray asks_price_ticks,
        cndarray asks_size_lots,
        cndarray asks_norders,
        cndarray bids_price_ticks,
        cndarray bids_size_lots,
        cndarray bids_norders,
    ):
        """Apply deltas in ticks/lots."""
        cdef:
            _SideView ask_sv = self._make_asks_view()
            _SideView bid_sv = self._make_bids_view()
            u64 top_bid = bid_sv.prices[0]
            u64 top_ask = ask_sv.prices[0]
            u64 bottom_bid = bid_sv.prices[bid_sv.count_ptr[0] - 1]
            u64 bottom_ask = ask_sv.prices[ask_sv.count_ptr[0] - 1]
            u64 i, price, size
            u32 norder
            u64 num_new_asks = asks_price_ticks.shape[0]
            u64 num_new_bids = bids_price_ticks.shape[0]

        for i in range(num_new_asks):
            price = <u64> asks_price_ticks[i]
            size = <u64> asks_size_lots[i]
            norder = <u32> asks_norders[i]

            if price < top_ask:
                self._process_lower_ask_ticks(price, size, norder)
                top_ask = price
            elif price == top_ask:
                self._process_matching_ask_ticks(size, norder)
            elif price <= bottom_ask:
                self._process_middle_ask_ticks(price, size, norder)
                bottom_ask = price
            else:
                break

        for i in range(num_new_bids):
            price = <u64> bids_price_ticks[i]
            size = <u64> bids_size_lots[i]
            norder = <u32> bids_norders[i]

            if price > top_bid:
                self._process_higher_bid_ticks(price, size, norder)
                top_bid = price
            elif price == top_bid:
                self._process_matching_bid_ticks(size, norder)
            elif price >= bottom_bid:
                self._process_middle_bid_ticks(price, size, norder)
                bottom_bid = price
            else:
                break

    cpdef void consume_bbo_raw(
        self, 
        u64 bid_price_ticks, 
        u64 bid_size_lots, 
        u32 bid_norder, 
        u64 ask_price_ticks, 
        u64 ask_size_lots, 
        u32 ask_norder
    ):
        """Update top-of-book given tick/lot values."""
        cdef:
            _SideView ask_sv = self._make_asks_view()
            _SideView bid_sv = self._make_bids_view()
            u64 top_bid = bid_sv.prices[0]
            u64 top_ask = ask_sv.prices[0]

        if top_ask == ask_price_ticks:
            ask_sv.sizes[0] = ask_size_lots
            ask_sv.norders[0] = ask_norder
        elif ask_price_ticks < top_ask:
            self._sv_roll_right(ask_sv, 0)
            ask_sv.prices[0] = ask_price_ticks
            ask_sv.sizes[0] = ask_size_lots
            ask_sv.norders[0] = ask_norder
            top_ask = ask_price_ticks

            if top_bid > top_ask:
                while bid_sv.count_ptr[0] > 0 and bid_price_ticks >= top_bid:
                    self._sv_roll_left(bid_sv, 0)
                    top_bid = bid_sv.prices[0]
        else:
            while ask_price_ticks > top_ask:
                self._sv_roll_left(ask_sv, 0)
                top_ask = ask_sv.prices[0]

        if top_bid == bid_price_ticks:
            bid_sv.sizes[0] = bid_size_lots
            bid_sv.norders[0] = bid_norder
        elif bid_price_ticks > top_bid:
            self._sv_roll_right(bid_sv, 0)
            bid_sv.prices[0] = bid_price_ticks
            bid_sv.sizes[0] = bid_size_lots
            bid_sv.norders[0] = bid_norder
            top_bid = bid_price_ticks

            if top_bid > top_ask:
                while bid_sv.count_ptr[0] > 0 and bid_price_ticks >= top_bid:
                    self._sv_roll_left(bid_sv, 0)
                    top_bid = bid_sv.prices[0]
        else:
            while bid_price_ticks < top_bid:
                self._sv_roll_left(bid_sv, 0)
                top_bid = bid_sv.prices[0]

    cpdef void consume_snapshot(
        self, 
        cndarray asks_prices, 
        cndarray asks_sizes, 
        cndarray asks_norders, 
        cndarray bids_prices, 
        cndarray bids_sizes, 
        cndarray bids_norders
    ):
        """Refresh the book with real prices/sizes/norders arrays (ascending asks, descending bids)."""
        if asks_prices.ndim != 1 or asks_sizes.ndim != 1 or asks_norders.ndim != 1:
            raise ValueError("Illegal ask array(s) ndim; must be 1D")
        if bids_prices.ndim != 1 or bids_sizes.ndim != 1 or bids_norders.ndim != 1:
            raise ValueError("Illegal bid array(s) ndim; must be 1D")
        if asks_prices.shape[0] != asks_sizes.shape[0] or asks_prices.shape[0] != asks_norders.shape[0]:
            raise ValueError("Illegal ask array(s) shape; must have the same length")
        if bids_prices.shape[0] != bids_sizes.shape[0] or bids_prices.shape[0] != bids_norders.shape[0]:
            raise ValueError("Illegal bid array(s) shape; must have the same length")

        cdef cndarray asks_price_ticks = self._convert_prices_to_ticks(asks_prices)
        cdef cndarray asks_size_lots = self._convert_sizes_to_lots(asks_sizes)
        asks_norders = cPyArray_Cast(<cndarray> asks_norders, cnp.NPY_UINT32)

        cdef cndarray bids_price_ticks = self._convert_prices_to_ticks(bids_prices)
        cdef cndarray bids_size_lots = self._convert_sizes_to_lots(bids_sizes)
        bids_norders = cPyArray_Cast(<cndarray> bids_norders, cnp.NPY_UINT32)

        self.consume_snapshot_raw(
            asks_price_ticks, 
            asks_size_lots, 
            asks_norders, 
            bids_price_ticks, 
            bids_size_lots, 
            bids_norders
        )

    cpdef void consume_snapshot_auto(self, cndarray asks_prices, cndarray asks_sizes, cndarray bids_prices, cndarray bids_sizes):
        """Refresh the book when norders are not provided (fill with 1 norder per level)."""
        cdef cndarray ones_asks = <cndarray> cPyArray_EMPTY(1, &asks_prices.shape[0], cnp.NPY_UINT32, 0)
        cdef cndarray ones_bids = <cndarray> cPyArray_EMPTY(1, &bids_prices.shape[0], cnp.NPY_UINT32, 0)
        cdef u32* asks_data = <u32*> ones_asks.data
        cdef u32* bids_data = <u32*> ones_bids.data
        cdef u64 i
        for i in range(asks_prices.shape[0]):
            asks_data[i] = 1
        for i in range(bids_prices.shape[0]):
            bids_data[i] = 1
        
        self.consume_snapshot(
            asks_prices, 
            asks_sizes, 
            ones_asks, 
            bids_prices, 
            bids_sizes, 
            ones_bids
        )

    cpdef void consume_deltas(
        self, 
        cndarray asks_prices, 
        cndarray asks_sizes, 
        cndarray asks_norders, 
        cndarray bids_prices, 
        cndarray bids_sizes, 
        cndarray bids_norders
    ):
        """Apply deltas in real prices/sizes/norders."""
        if asks_prices.ndim != 1 or asks_sizes.ndim != 1 or asks_norders.ndim != 1:
            raise ValueError("Illegal ask array(s) ndim; must be 1D")
        if bids_prices.ndim != 1 or bids_sizes.ndim != 1 or bids_norders.ndim != 1:
            raise ValueError("Illegal bid array(s) ndim; must be 1D")
        if asks_prices.shape[0] != asks_sizes.shape[0] or asks_prices.shape[0] != asks_norders.shape[0]:
            raise ValueError("Illegal ask array(s) shape; must have the same length")
        if bids_prices.shape[0] != bids_sizes.shape[0] or bids_prices.shape[0] != bids_norders.shape[0]:
            raise ValueError("Illegal bid array(s) shape; must have the same length")

        cdef cndarray asks_price_ticks = self._convert_prices_to_ticks(asks_prices)
        cdef cndarray asks_size_lots = self._convert_sizes_to_lots(asks_sizes)
        asks_norders = cPyArray_Cast(<cndarray> asks_norders, cnp.NPY_UINT32)

        cdef cndarray bids_price_ticks = self._convert_prices_to_ticks(bids_prices)
        cdef cndarray bids_size_lots = self._convert_sizes_to_lots(bids_sizes)
        bids_norders = cPyArray_Cast(<cndarray> bids_norders, cnp.NPY_UINT32)

        self.consume_deltas_raw(
            asks_price_ticks, 
            asks_size_lots, 
            asks_norders, 
            bids_price_ticks, 
            bids_size_lots, 
            bids_norders
        )

    cpdef void consume_deltas_auto(self, cndarray asks_prices, cndarray asks_sizes, cndarray bids_prices, cndarray bids_sizes):
        """Apply deltas when norders are not provided (assume 1 per level)."""
        cdef cndarray ones_asks = <cndarray> cPyArray_EMPTY(1, &asks_prices.shape[0], cnp.NPY_UINT32, 0)
        cdef cndarray ones_bids = <cndarray> cPyArray_EMPTY(1, &bids_prices.shape[0], cnp.NPY_UINT32, 0)
        cdef u32* asks_data = <u32*> ones_asks.data
        cdef u32* bids_data = <u32*> ones_bids.data
        cdef u64 i
        for i in range(asks_prices.shape[0]):
            asks_data[i] = 1
        for i in range(bids_prices.shape[0]):
            bids_data[i] = 1
        
        self.consume_deltas(asks_prices, asks_sizes, ones_asks, bids_prices, bids_sizes, ones_bids)

    cpdef void consume_bbo(self, double bid_price, double bid_size, u32 bid_norder, double ask_price, double ask_size, u32 ask_norder):
        """Update top-of-book given real prices and sizes."""
        cdef:
            u64 bid_price_ticks = self._convert_price_to_ticks(bid_price)
            u64 ask_price_ticks = self._convert_price_to_ticks(ask_price)
            u64 bid_size_lots = self._convert_size_to_lots(bid_size)
            u64 ask_size_lots = self._convert_size_to_lots(ask_size)
        self.consume_bbo_raw(bid_price_ticks, bid_size_lots, bid_norder, ask_price_ticks, ask_size_lots, ask_norder)

    cpdef void consume_bbo_auto(self, double bid_price, double bid_size, double ask_price, double ask_size):
        """Update BBO assuming norders=1 when not provided."""
        self.consume_bbo(bid_price, bid_size, <u32>1, ask_price, ask_size, <u32>1)

    cpdef tuple get_bbo(self):
        """Return (best_bid_price, best_bid_size, best_ask_price, best_ask_size)."""
        self.__ensure_orderbook_not_empty()
        cdef _SideView bid_sv = self._make_bids_view()
        cdef _SideView ask_sv = self._make_asks_view()
        return (
            self._convert_ticks_to_real(bid_sv.prices[0]), 
            self._convert_lots_to_real(bid_sv.sizes[0]), 
            self._convert_ticks_to_real(ask_sv.prices[0]), 
            self._convert_lots_to_real(ask_sv.sizes[0])
        )

    cpdef double get_mid_price(self):
        """Return the price between best bid and best ask."""
        self.__ensure_orderbook_not_empty()
        cdef _SideView bid_sv = self._make_bids_view()
        cdef _SideView ask_sv = self._make_asks_view()
        cdef u64 mid_price_ticks = (bid_sv.prices[0] + ask_sv.prices[0]) // 2
        return self._convert_ticks_to_real(mid_price_ticks)

    cpdef double get_bbo_spread(self):
        """Return the difference between best bid and best ask."""
        self.__ensure_orderbook_not_empty()
        cdef _SideView bid_sv = self._make_bids_view()
        cdef _SideView ask_sv = self._make_asks_view()
        cdef u64 spread_ticks = ask_sv.prices[0] - bid_sv.prices[0]
        return self._convert_ticks_to_real(spread_ticks)

    cpdef bint is_crossed(self, double other_bid_price, double other_ask_price):
        """Return True if bids cross asks."""
        self.__ensure_orderbook_not_empty()
        cdef _SideView bid_sv = self._make_bids_view()
        cdef _SideView ask_sv = self._make_asks_view()
        cdef u64 bid_price_ticks = bid_sv.prices[0]
        cdef u64 ask_price_ticks = ask_sv.prices[0]
        return bid_price_ticks > other_ask_price and ask_price_ticks < other_bid_price

    cpdef double get_imbalance(self, double depth_pct):
        """Compute bid/ask volume imbalance within a spread-based band.
        """
        if depth_pct <= 0.0:
            raise ValueError(f"Negative depth_pct; expected >0 but got {depth_pct}")
        
        self.__ensure_orderbook_not_empty()

        cdef:
            _SideView bid_sv = self._make_bids_view()
            _SideView ask_sv = self._make_asks_view()
            u64 top_ask_price = ask_sv.prices[0]
            u64 top_bid_price = bid_sv.prices[0]
            u64 mid_price = (top_ask_price + top_bid_price) // 2
            double band = (top_ask_price - top_bid_price) * (depth_pct / 100.0)
            double upper_price = mid_price - band
            double lower_price = mid_price + band
            double cum_ask_size = 0.0
            double cum_bid_size = 0.0
            u64 i

        for i in range(bid_sv.count_ptr[0]):
            if bid_sv.prices[i] >= lower_price:
                cum_bid_size += bid_sv.sizes[i]
        for i in range(ask_sv.count_ptr[0]):
            if ask_sv.prices[i] <= upper_price:
                cum_ask_size += ask_sv.sizes[i]

        if cum_bid_size + cum_ask_size == 0.0:
            # raise RuntimeError("No volume within band; cannot compute imbalance.")
            return 0.0
        return cum_bid_size / (cum_bid_size + cum_ask_size)

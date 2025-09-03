from numpy cimport ndarray as cndarray
from libc.stdint cimport uint64_t as u64, uint32_t as u32

cdef void _memmove_u64(u64* base, u64 start, u64 end) noexcept nogil  
cdef void _memmove_u32(u32* base, u64 start, u64 end) noexcept nogil

ctypedef struct _SideView:
    u64* prices
    u64* sizes
    u32* norders
    u64* count_ptr
    u64  max_levels
    bint ascending

cdef class COrderbook:
    cdef:
        double      _tick_size
        double      _lot_size
        u64         _max_levels
        u64         _num_bids
        u64         _num_asks

        # Backing NumPy arrays to keep memory alive
        cndarray     _bids_price_arr
        cndarray     _bids_size_arr
        cndarray     _bids_norders_arr
        cndarray     _asks_price_arr
        cndarray     _asks_size_arr
        cndarray     _asks_norders_arr

    # def __cinit__(self, double tick_size, double lot_size, u64 max_num_levels=*)
    cdef inline void __ensure_orderbook_not_empty(self)
    cdef inline u64 _convert_price_to_ticks(self, double price) nogil
    cdef inline u64 _convert_size_to_lots(self, double size) nogil
    cdef inline cndarray _convert_prices_to_ticks(self, cndarray prices)
    cdef inline cndarray _convert_sizes_to_lots(self, cndarray sizes)
    cdef inline double _convert_ticks_to_real(self, u64 ticks) nogil
    cdef inline double _convert_lots_to_real(self, u64 lots) nogil
    cdef inline _SideView _make_asks_view(self) nogil
    cdef inline _SideView _make_bids_view(self) nogil
    cdef inline void _sv_roll_right(self, _SideView sv, u64 start_idx) noexcept nogil
    cdef inline void _sv_roll_left(self, _SideView sv, u64 start_idx) noexcept nogil

    cdef void _process_matching_ask_ticks(self, u64 size_lots, u32 norder) noexcept nogil
    cdef void _process_matching_bid_ticks(self, u64 size_lots, u32 norder) noexcept nogil
    cdef void _process_lower_ask_ticks(self, u64 price_ticks, u64 size_lots, u32 norder) noexcept nogil
    cdef void _process_higher_bid_ticks(self, u64 price_ticks, u64 size_lots, u32 norder) noexcept nogil
    cdef void _process_middle_ask_ticks(self, u64 price_ticks, u64 size_lots, u32 norder) noexcept nogil
    cdef void _process_middle_bid_ticks(self, u64 price_ticks, u64 size_lots, u32 norder) noexcept nogil
    
    cpdef void consume_snapshot_raw(self, cndarray asks_price_ticks, cndarray asks_size_lots, cndarray asks_norders, cndarray bids_price_ticks, cndarray bids_size_lots, cndarray bids_norders)
    cpdef void consume_deltas_raw(self, cndarray asks_price_ticks, cndarray asks_size_lots, cndarray asks_norders, cndarray bids_price_ticks, cndarray bids_size_lots, cndarray bids_norders)
    cpdef void consume_bbo_raw(self, u64 bid_price_ticks, u64 bid_size_lots, u32 bid_norder, u64 ask_price_ticks, u64 ask_size_lots, u32 ask_norder)
    cpdef void consume_snapshot(self, cndarray asks_prices, cndarray asks_sizes, cndarray asks_norders, cndarray bids_prices, cndarray bids_sizes, cndarray bids_norders)
    cpdef void consume_snapshot_auto(self, cndarray asks_prices, cndarray asks_sizes, cndarray bids_prices, cndarray bids_sizes)
    cpdef void consume_deltas(self, cndarray asks_prices, cndarray asks_sizes, cndarray asks_norders, cndarray bids_prices, cndarray bids_sizes, cndarray bids_norders)
    cpdef void consume_deltas_auto(self, cndarray asks_prices, cndarray asks_sizes, cndarray bids_prices, cndarray bids_sizes)
    cpdef void consume_bbo(self, double bid_price, double bid_size, u32 bid_norder, double ask_price, double ask_size, u32 ask_norder)
    cpdef void consume_bbo_auto(self, double bid_price, double bid_size, double ask_price, double ask_size)
    cpdef void clear(self)

    cpdef tuple get_bbo(self)
    cpdef double get_mid_price(self)
    cpdef double get_bbo_spread(self)
    cpdef bint is_crossed(self, double other_bid_price, double other_ask_price)
    cpdef double get_imbalance(self, double depth_pct)
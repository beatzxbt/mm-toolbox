
from libc.stdint cimport uint16_t as u16, int16_t as i16

cdef struct OrderbookLevel:
    double px
    double sz
    u16 num_orders

cdef struct OrderbookSnapshot:
    u16 num_levels
    OrderbookLevel* bids
    OrderbookLevel* asks

cpdef OrderbookLevel make_orderbook_level(double px, double sz, u16 num_orders)
cpdef OrderbookSnapshot make_orderbook_snapshot(u16 num_levels, OrderbookLevel* bids, OrderbookLevel* asks)

cdef class Orderbook:
    cdef double _tick_sz
    cdef u16 _max_levels
    cdef int _size
    cdef OrderbookLevel* _bids
    cdef OrderbookLevel* _asks
    cdef bint _is_populated

    cdef void _reset(self)
    cdef void _roll_bids(self, u16 start_idx=*, bint shift_right=*)
    cdef void _roll_asks(self, Py_ssize_t start_idx=*, bint shift_right=*)
    cdef void _process_matching_ask(self, double ask_sz)
    cdef void _process_matching_bid(self, double bid_sz)
    cdef void _process_middle_ask(self, double ask_px, double ask_sz)
    cdef void _process_middle_bid(self, double bid_px, double bid_sz)
    cdef void _process_lower_ask(self, double ask_px, double ask_sz)
    cdef void _process_higher_bid(self, double bid_px, double bid_sz)
    cdef void _process_higher_ask(self, double ask_px, double ask_sz)
    cdef void _process_lower_bid(self, double bid_px, double bid_sz)

    cpdef void update(self, list asks, list bids)
    cpdef double get_mid_px(self)
    cpdef double get_wmid_px(self)
    cpdef double get_bbo_spread(self)
    cpdef double get_vamp(self, double sz, bint is_base_currency=?)
    cpdef double get_impact(self, double sz, bint is_bid, bint is_base_currency=?)
    cpdef double get_imbalance(self, double depth_pct)
    cpdef get_bids(self)
    cpdef get_asks(self)
    cpdef list get_bbo(self)
    cpdef bint is_crossed(self, Orderbook other)


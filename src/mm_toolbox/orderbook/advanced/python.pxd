from libc.stdint cimport uint64_t as u64

from numpy cimport ndarray

from .core cimport CoreAdvancedOrderbook
from .level.level cimport PyOrderbookLevel, PyOrderbookLevels


cdef class PyAdvancedOrderbook:
    cdef CoreAdvancedOrderbook _core

    cpdef void clear(self)
    cpdef void consume_snapshot(self, PyOrderbookLevels asks, PyOrderbookLevels bids)
    cpdef void consume_deltas(self, PyOrderbookLevels asks, PyOrderbookLevels bids)
    cpdef void consume_bbo(self, PyOrderbookLevel ask, PyOrderbookLevel bid)
    cpdef void consume_snapshot_numpy(
        self,
        double[:] ask_prices,
        double[:] ask_sizes,
        double[:] bid_prices,
        double[:] bid_sizes,
        u64[:] ask_norders=?,
        u64[:] bid_norders=?,
    )
    cpdef void consume_deltas_numpy(
        self,
        double[:] ask_prices,
        double[:] ask_sizes,
        double[:] bid_prices,
        double[:] bid_sizes,
        u64[:] ask_norders=?,
        u64[:] bid_norders=?,
    )

    cpdef get_bbo(self)
    cpdef get_bids(self)
    cpdef get_asks(self)
    cpdef get_bids_numpy(self)
    cpdef get_asks_numpy(self)
    cpdef double get_mid_price(self)
    cpdef double get_bbo_spread(self)
    cpdef double get_wmid_price(self)
    cpdef double get_volume_weighted_mid_price(self, double size, bint is_base_currency=?)
    cpdef double get_price_impact(self, double size, bint is_buy, bint is_base_currency=?)
    cpdef double get_size_for_price_impact_bps(self, double impact_bps, bint is_buy, bint is_base_currency=?)
    cpdef bint is_bbo_crossed(self, double bid_price, double ask_price)
    cpdef bint does_bbo_price_change(self, double bid_price, double ask_price)


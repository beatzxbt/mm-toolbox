from numpy cimport ndarray

cdef class Rounder:
    cdef:
        public object   config
        public double   tick_size
        public double   lot_size

        double          (*_bid_round_func)(double)
        double          (*_ask_round_func)(double)
        double          (*_size_round_func)(double)

        double          _inverse_tick_size
        double          _inverse_lot_size
        double          _tick_rounding_factor
        double          _lot_rounding_factor
    
    # def               __cinit__(self, double tick_size, double lot_size)

    cdef inline double  _round_to(self, double num, double factor) nogil
    
    cpdef double        bid(self, double price)
    cpdef double        ask(self, double price)
    cpdef double        size(self, double size)
    cpdef ndarray       bids(self, ndarray prices)
    cpdef ndarray       asks(self, ndarray prices)
    cpdef ndarray       sizes(self, ndarray sizes)

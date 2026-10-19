from numpy cimport ndarray

cdef class Rounder:
    cdef:
        public object   config
        readonly double tick_size
        readonly double lot_size

        bint _bid_up
        bint _ask_up
        bint _size_up

        double          _inverse_tick_size
        double          _inverse_lot_size
        double          _tick_rounding_factor
        double          _lot_rounding_factor
    
    # def               __cinit__(self, double tick_size, double lot_size)

    cdef inline double _grid_value(self, double ticks, double step, double factor) noexcept nogil
    cdef inline double _quantize(self, double value, double step, double inverse, double factor, bint up)
    
    cpdef double        bid(self, double price)
    cpdef double        ask(self, double price)
    cpdef double        size(self, double size)
    cpdef ndarray       bids(self, ndarray prices)
    cpdef ndarray       asks(self, ndarray prices)
    cpdef ndarray       sizes(self, ndarray sizes)

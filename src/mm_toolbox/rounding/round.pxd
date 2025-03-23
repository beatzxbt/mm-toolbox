cimport numpy as cnp

cdef class Round:
    cdef:
        # Public attributes.
        public double   tick_sz
        public double   lot_sz

        # Private attributes.
        double          _inverse_tick_sz
        double          _inverse_lot_sz
        double          _tick_rounding_factor
        double          _lot_rounding_factor
    
    # def               __cinit__(self, double tick_sz, double lot_sz)

    cdef inline double  _round_to(self, double num, double factor) nogil
    
    cpdef double        bid(self, double px)
    cpdef double        ask(self, double px)
    cpdef double        sz(self, double sz)
    cpdef cnp.ndarray   bids(self, cnp.ndarray pxs)
    cpdef cnp.ndarray   asks(self, cnp.ndarray pxs)
    cpdef cnp.ndarray   szs(self, cnp.ndarray szs)

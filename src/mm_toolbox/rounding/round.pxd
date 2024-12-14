cimport numpy as np

cdef class Round:
    cdef:
        # Public attributes.
        double          tick_size
        double          lot_size

        # Private attributes.
        double          _inverse_tick_size
        double          _inverse_lot_size
        double          _tick_rounding_factor
        double          _lot_rounding_factor
    
    # Private helper method for process functions.
    cdef inline double  _round_to(self, double num, double factor)

    # Core public methods.
    cpdef double        bid(self, double px)
    cpdef double        ask(self, double px)
    cpdef double        size(self, double sz)
    cpdef np.ndarray    bids(self, np.ndarray pxs)
    cpdef np.ndarray    asks(self, np.ndarray pxs)
    cpdef np.ndarray    sizes(self, np.ndarray szs)

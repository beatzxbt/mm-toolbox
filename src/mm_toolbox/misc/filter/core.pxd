cdef class DataBoundsFilter:
    cdef:
        double _threshold
        double _lower_bound
        double _upper_bound

    cpdef void reset(self, double value)
    cpdef bint check_and_update(self, double value, bint reset=*)



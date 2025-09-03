cdef class DataBoundsFilter:
    cdef:
        double _threshold
        double _lower_bound
        double _upper_bound

    cpdef bint check_and_update(self, float value, bint reset=*)

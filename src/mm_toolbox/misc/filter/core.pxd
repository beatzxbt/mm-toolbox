cdef class DataBoundsFilter:
    cdef:
        double _threshold
        double _lower_bound
        double _upper_bound

    cpdef void reset(self, double value) noexcept
    cpdef bint check_and_update(self, double value, bint reset=*) noexcept



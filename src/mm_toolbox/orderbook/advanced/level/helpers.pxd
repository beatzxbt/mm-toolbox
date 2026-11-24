from libc.stdint cimport uint64_t as u64
from libc.float cimport DBL_EPSILON
from libc.math cimport fabs, floor, fmin, isfinite

from .level cimport OrderbookEntry

cdef u64 convert_price_to_tick(double price, double tick_size_recip) noexcept nogil
cdef u64 convert_size_to_lot(double size, double lot_size_recip) noexcept nogil
cdef double convert_price_from_tick(u64 tick, double tick_size) noexcept nogil
cdef double convert_size_from_lot(u64 lot, double lot_size) noexcept nogil
cdef void reverse_entries(OrderbookEntry* entries, u64 num_entries) noexcept nogil
cdef u64 compact_entries(OrderbookEntry* entries, u64 num_entries, bint drop_zero) noexcept nogil
cdef void validate_price(double price)
cdef void validate_size(double size)


cdef inline u64 _checked_scaled_units(double value) except? -1 nogil:
    cdef double integer
    cdef double epsilon
    if not isfinite(value) or value < 0.0 or value >= 18446744073709551616.0:
        with gil:
            raise ValueError("Normalized price/size must fit in unsigned 64-bit units")
    integer = floor(value)
    if value == integer:
        return <u64>integer
    epsilon = fmin(fabs(value) * DBL_EPSILON * 4.0, 0.5)
    if 1.0 - (value - integer) <= epsilon:
        integer += 1.0
    return <u64>integer


cdef inline u64 convert_price_to_tick_trusted(double price, double tick_size_recip) except? -1 nogil:
    return _checked_scaled_units(price * tick_size_recip)


cdef inline u64 convert_size_to_lot_trusted(double size, double lot_size_recip) except? -1 nogil:
    return _checked_scaled_units(size * lot_size_recip)

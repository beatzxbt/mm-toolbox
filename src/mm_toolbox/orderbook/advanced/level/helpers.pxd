from libc.stdint cimport uint64_t as u64
from libc.float cimport DBL_EPSILON
from libc.math cimport fabs, floor

from .level cimport OrderbookEntry

cdef u64 convert_price_to_tick(double price, double tick_size_recip) noexcept nogil
cdef u64 convert_size_to_lot(double size, double lot_size_recip) noexcept nogil
cdef double convert_price_from_tick(u64 tick, double tick_size) noexcept nogil
cdef double convert_size_from_lot(u64 lot, double lot_size) noexcept nogil
cdef void reverse_entries(OrderbookEntry* entries, u64 num_entries) noexcept nogil
cdef void validate_price(double price)
cdef void validate_size(double size)


cdef inline u64 convert_price_to_tick_trusted(double price, double tick_size_recip) noexcept nogil:
    cdef double value = price * tick_size_recip
    return <u64>floor(value + fabs(value) * DBL_EPSILON * 4.0)


cdef inline u64 convert_size_to_lot_trusted(double size, double lot_size_recip) noexcept nogil:
    cdef double value = size * lot_size_recip
    return <u64>floor(value + fabs(value) * DBL_EPSILON * 4.0)

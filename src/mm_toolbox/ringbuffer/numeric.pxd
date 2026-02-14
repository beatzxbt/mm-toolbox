cimport numpy as cnp
from libc.stdint cimport uint64_t as u64

ctypedef fused uint_t:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t

ctypedef fused int_t:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t

ctypedef fused float_t:
    cnp.float32_t
    cnp.float64_t

ctypedef fused numeric_t:
    uint_t
    int_t
    float_t

cdef class NumericRingBuffer:
    cdef:
        u64 _max_capacity
        u64 _mask
        bint _disable_async
        u64 _tail
        u64 _head
        u64 _size
        cnp.ndarray _buffer
        object _dtype
        object _buffer_not_empty_event

    # def __cinit__(self, int max_capacity, object dtype, bint disable_async=False)
    cpdef cnp.ndarray raw(self, bint copy=*)
    cpdef cnp.ndarray unwrapped(self)
    cpdef void overwrite_latest(self, numeric_t item, bint increment_count=*)
    cpdef void insert(self, numeric_t item)
    cpdef void insert_batch(self, numeric_t[::1] items)
    cpdef bint contains(self, numeric_t item)
    cpdef object consume(self)
    cpdef cnp.ndarray consume_all(self)
    # def consume_iterable(self)
    # async def aconsume(self)
    # async def aconsume_iterable(self)
    cpdef object peekright(self)
    cpdef object peekleft(self)
    cpdef void clear(self)
    cpdef bint is_empty(self)
    cpdef bint is_full(self)

    # def __contains__(self, object item)
    # def __len__(self)
    # def __getitem__(self, int idx)
    cdef inline bint __enforce_ringbuffer_not_empty(self)
    cdef inline bint __enforce_async_not_disabled(self)


cpdef bint _contains_u64(NumericRingBuffer ringbuffer, cnp.uint64_t item)
cpdef void _insert_u64(NumericRingBuffer ringbuffer, cnp.uint64_t item)

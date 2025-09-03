from libc.stdint cimport uint64_t as u64

cdef class GenericRingBuffer:
    cdef:
        u64         _max_capacity
        u64         _mask
        u64         _tail
        u64         _head
        u64         _size
        list        _buffer
        object      _buffer_not_empty_event
        bint        _disable_async

    # def void      __cinit__(self, int max_capacity, bint disable_async=False)
    cpdef list      raw(self, bint copy=*)
    cpdef list      unwrapped(self)
    cpdef void      overwrite_latest(self, object item, bint increment_count=*)
    cpdef void      insert(self, object item)
    cpdef void      insert_batch(self, list items)
    cpdef bint      contains(self, object item)
    cpdef object    consume(self)
    cpdef list      consume_all(self)
    # def consume_iterable(self)
    # async def     aconsume(self)
    # async def     aconsume_iterable(self)
    cpdef object    peekright(self)
    cpdef object    peekleft(self)
    cpdef void      clear(self)
    cpdef bint      is_empty(self)
    cpdef bint      is_full(self)

    # def           __contains__(self, object item)
    # def           __len__(self)
    # def           __getitem__(self, int idx)
    cdef inline bint __enforce_ringbuffer_not_empty(self)
    cdef inline bint __enforce_async_not_disabled(self)

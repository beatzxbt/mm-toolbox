from libc.stdint cimport uint64_t as u64

cdef class BytesRingBuffer:
    cdef:
        bint        _only_insert_unique
        u64         _max_capacity
        u64         _mask
        u64         _head
        u64         _tail
        u64         _size
        u64         _latest_insert_time_ns
        u64         _latest_consume_time_ns
        list        _buffer
        object      _buffer_not_empty_event
        bint        _disable_async
    cpdef list unwrapped(self)
    cpdef void overwrite_latest(self, bytes item, bint increment_count=*)
    cpdef bint insert(self, bytes item)
    cpdef bint insert_batch(self, list items)
    cpdef bint contains(self, bytes item)
    cpdef bytes consume(self)
    cpdef list consume_all(self)
    # def       consume_iterable(self)
    # async def aconsume(self)
    # async def aconsume_iterable(self)
    cpdef bytes peekright(self)
    cpdef bytes peekleft(self)
    cpdef void clear(self)
    cpdef bint is_empty(self)
    cpdef bint is_full(self)

    # def __contains__(self, bytes item)
    # def __len__(self)
    cdef inline bint __enforce_ringbuffer_not_empty(self)
    cdef inline bint __enforce_async_not_disabled(self)


cdef class BytesRingBufferFast:
    cdef:
        u64         _max_capacity
        u64         _mask
        u64         _slot_size
        int         _slot_size_log2
        u64         _head
        u64         _tail
        u64         _size
        u64         _latest_insert_time_ns
        u64         _latest_consume_time_ns
        char*       _buffer
        u64*        _lengths
        object      _buffer_not_empty_event
        bint        _disable_async
        bint        _only_insert_unique
    cpdef list unwrapped(self)
    cpdef void overwrite_latest(self, bytes item, bint increment_count=*)
    cpdef bint insert(self, bytes item)
    cpdef bint insert_char(self, const char* item, Py_ssize_t item_len)
    cpdef bint insert_batch(self, list items)
    cpdef bint contains(self, bytes item)
    cpdef bytes consume(self)
    cpdef list consume_all(self)
    # def       consume_iterable(self)
    # async def aconsume(self)
    # async def aconsume_iterable(self)
    cpdef bytes peekright(self)
    cpdef bytes peekleft(self)
    cpdef void clear(self)
    cpdef bint is_empty(self)
    cpdef bint is_full(self)
    # def __contains__(self, bytes item)
    # def __len__(self)
    cdef inline char* _get_slot_ptr(self, u64 idx) nogil
    cdef inline bytes _make_bytes(self, u64 idx)
    cdef inline u64 _next_power_of_2(self, u64 n) nogil

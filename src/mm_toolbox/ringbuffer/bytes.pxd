cdef class BytesRingBuffer:
    cdef:
        bint        _only_insert_unique
        Py_ssize_t  _max_capacity
        Py_ssize_t  _mask
        Py_ssize_t  _head
        Py_ssize_t  _tail
        Py_ssize_t  _size
        list        _buffer
        object      _buffer_not_empty_event
        bint        _disable_async

    cpdef list raw(self, bint copy=*)
    cpdef list unwrapped(self)
    cpdef void overwrite_latest(self, bytes item, bint increment_count=*)
    cpdef void insert(self, bytes item)
    cpdef void insert_batch(self, list items)
    cpdef bint contains(self, bytes item)
    cpdef bytes consume(self)
    cpdef list consume_all(self)
    # def consume_iterable(self)
    # async def aconsume(self)
    # async def aconsume_iterable(self)
    cpdef bytes peekright(self)
    cpdef bytes peekleft(self)
    cpdef void clear(self)
    cpdef bint is_empty(self)
    cpdef bint is_full(self)

    # def __contains__(self, bytes item)
    # def __len__(self)
    # def __getitem__(self, int idx)
    cdef inline bint __enforce_ringbuffer_not_empty(self)
    cdef inline bint __enforce_async_not_disabled(self)
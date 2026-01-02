from libc.stdint cimport uint64_t as u64

cdef extern from *:
    unsigned long long __atomic_load_n(unsigned long long* ptr, int memorder) nogil
    void __atomic_store_n(unsigned long long* ptr, unsigned long long val, int memorder) nogil
    unsigned long long __atomic_add_fetch(unsigned long long* ptr, unsigned long long val, int memorder) nogil
    unsigned long long __atomic_sub_fetch(unsigned long long* ptr, unsigned long long val, int memorder) nogil


cdef int _ATOMIC_RELAXED = 0
cdef int _ATOMIC_ACQUIRE = 2
cdef int _ATOMIC_RELEASE = 3
cdef int _ATOMIC_ACQ_REL = 4


cdef u64 atomic_load_acquire(u64* p) nogil:
    """Load value with acquire semantics."""
    return <u64>__atomic_load_n(<unsigned long long*>p, _ATOMIC_ACQUIRE)


cdef void atomic_store_release(u64* p, u64 v) nogil:
    """Store value with release semantics."""
    __atomic_store_n(<unsigned long long*>p, <unsigned long long>v, _ATOMIC_RELEASE)


cdef u64 atomic_add(u64* p, u64 delta) nogil:
    """Add delta and return the new value."""
    return <u64>__atomic_add_fetch(<unsigned long long*>p, <unsigned long long>delta, _ATOMIC_ACQ_REL)


cdef u64 atomic_sub(u64* p, u64 delta) nogil:
    """Subtract delta and return the new value."""
    return <u64>__atomic_sub_fetch(<unsigned long long*>p, <unsigned long long>delta, _ATOMIC_ACQ_REL)

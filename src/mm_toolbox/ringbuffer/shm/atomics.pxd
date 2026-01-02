from libc.stdint cimport uint64_t as u64


cdef extern from *:
    unsigned long long __atomic_load_n(unsigned long long* ptr, int memorder)
    void __atomic_store_n(unsigned long long* ptr, unsigned long long val, int memorder)
    unsigned long long __atomic_add_fetch(unsigned long long* ptr, unsigned long long val, int memorder)
    unsigned long long __atomic_sub_fetch(unsigned long long* ptr, unsigned long long val, int memorder)


cdef int _ATOMIC_RELAXED
cdef int _ATOMIC_ACQUIRE
cdef int _ATOMIC_RELEASE
cdef int _ATOMIC_ACQ_REL

cdef u64 atomic_load_acquire(u64* p) nogil
cdef void atomic_store_release(u64* p, u64 v) nogil
cdef u64 atomic_add(u64* p, u64 delta) nogil
cdef u64 atomic_sub(u64* p, u64 delta) nogil

from libc.stdint cimport uint64_t as u64
from libc.string cimport memcmp, memcpy, memset
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from mm_toolbox.misc.parsers.crypto.binance._bbo_cache cimport _BboShapeEntry


cdef int _CACHE_CAP = 4096
cdef int _CACHE_MASK = 4096 - 1
cdef _BboShapeEntry* _CACHE = NULL
cdef bint _CACHE_INITIALIZED = False


cdef void cache_init():
    """Initialize C-level cache table."""
    global _CACHE_INITIALIZED, _CACHE
    if _CACHE_INITIALIZED:
        return
    _CACHE = <_BboShapeEntry*>PyMem_Malloc(<size_t>(_CACHE_CAP) * <size_t>sizeof(_BboShapeEntry))
    if _CACHE != NULL:
        memset(_CACHE, 0, <size_t>(_CACHE_CAP) * <size_t>sizeof(_BboShapeEntry))
        _CACHE_INITIALIZED = True


cdef inline u64 hash_symbol(const unsigned char* s, Py_ssize_t n) nogil:
    """FNV-1a 64-bit hash."""
    cdef u64 h = 0xCBF29CE484222325
    cdef Py_ssize_t i = 0
    while i < n:
        h ^= <u64>s[i]
        h *= <u64>0x100000001B3
        i += 1
    return h


cdef inline _BboShapeEntry* cache_find_nogil(const unsigned char* s, Py_ssize_t n, u64 h) nogil:
    """Find cache entry for symbol; return NULL if not found."""
    if not _CACHE_INITIALIZED:
        return NULL
    cdef int idx = <int>(h & <u64>_CACHE_MASK)
    cdef int start = idx
    cdef _BboShapeEntry* e
    while True:
        e = &_CACHE[idx]
        if e.used == 0:
            return NULL
        if e.used == 1 and e.hash == h and e.sym_len == n and memcmp(e.sym, s, <size_t>n) == 0:
            return e
        idx = (idx + 1) & _CACHE_MASK
        if idx == start:
            return NULL


cdef _BboShapeEntry* cache_get_or_create(const unsigned char* s, Py_ssize_t n, u64 h):
    """Lookup entry; create if missing. GIL must be held."""
    cache_init()
    cdef int idx = <int>(h & <u64>_CACHE_MASK)
    cdef int start = idx
    cdef _BboShapeEntry* e
    cdef int first_free = -1

    while True:
        e = &_CACHE[idx]
        if e.used == 0:
            if first_free < 0:
                first_free = idx
            break
        if e.used == 1 and e.hash == h and e.sym_len == n and memcmp(e.sym, s, <size_t>n) == 0:
            return e
        idx = (idx + 1) & _CACHE_MASK
        if idx == start:
            break

    if first_free < 0:
        first_free = <int>(h & <u64>_CACHE_MASK)
    e = &_CACHE[first_free]
    if e.used == 1 and e.sym != NULL:
        PyMem_Free(e.sym)
    e.hash = h
    e.sym_len = n
    e.sym = <char*>PyMem_Malloc(<size_t>n)
    if e.sym != NULL:
        memcpy(e.sym, s, <size_t>n)
    e.u_pos = -1
    e.u_digits = 0
    e.b_pos = -1
    e.B_pos = -1
    e.a_pos = -1
    e.A_pos = -1
    e.used = 1
    return e




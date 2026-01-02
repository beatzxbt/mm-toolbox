from libc.stdint cimport uint64_t as u64

cdef struct _BboShapeEntry:
    u64        hash
    char*           sym
    Py_ssize_t      sym_len
    Py_ssize_t      u_pos
    int             u_digits
    Py_ssize_t      b_pos
    Py_ssize_t      B_pos
    Py_ssize_t      a_pos
    Py_ssize_t      A_pos
    unsigned char   used

cdef void cache_init()
cdef inline u64 hash_symbol(const unsigned char* s, Py_ssize_t n) nogil
cdef _BboShapeEntry* cache_find_nogil(const unsigned char* s, Py_ssize_t n, u64 h) nogil
cdef _BboShapeEntry* cache_get_or_create(const unsigned char* s, Py_ssize_t n, u64 h)




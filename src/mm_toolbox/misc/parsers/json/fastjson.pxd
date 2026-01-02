from libc.stdint cimport uint64_t as u64, int64_t as i64

cdef inline bint is_ws(unsigned char c) nogil:
    """True if ASCII whitespace."""
    return c == 32 or c == 10 or c == 13 or c == 9


cdef inline const unsigned char* skip_ws(const unsigned char* p, const unsigned char* end) nogil:
    """Advance pointer past ASCII whitespace."""
    while p < end and is_ws(p[0]):
        p += 1
    return p


cdef inline double ten_pow_neg(int n) nogil:
    """Return 10^-n for small n via constants to avoid pow()."""
    if n == 0:
        return 1.0
    elif n == 1:
        return 0.1
    elif n == 2:
        return 0.01
    elif n == 3:
        return 0.001
    elif n == 4:
        return 0.0001
    elif n == 5:
        return 0.00001
    elif n == 6:
        return 0.000001
    elif n == 7:
        return 0.0000001
    elif n == 8:
        return 0.00000001
    elif n == 9:
        return 0.000000001
    elif n == 10:
        return 0.0000000001
    elif n == 11:
        return 0.00000000001
    elif n == 12:
        return 0.000000000001
    elif n == 13:
        return 0.0000000000001
    elif n == 14:
        return 0.00000000000001
    elif n == 15:
        return 0.000000000000001
    elif n == 16:
        return 0.0000000000000001
    elif n == 17:
        return 0.00000000000000001
    else:
        return 0.000000000000000001


cdef inline double parse_quoted_decimal(const unsigned char* p,
                                        const unsigned char* end,
                                        const unsigned char** out_next) nogil:
    """Parse digits/decimal inside quotes; set out_next to char after closing quote."""
    cdef:
        u64 int_part = 0
        u64 frac_part = 0
        int frac_len = 0
        bint seen_dot = False
        unsigned char ch

    while p < end:
        ch = p[0]
        if ch == ord('"'):
            p += 1
            break
        if ch == ord('.'):
            seen_dot = True
            p += 1
            continue
        if ch >= ord('0') and ch <= ord('9'):
            if not seen_dot:
                int_part = int_part * 10 + (ch - ord('0'))
            else:
                frac_part = frac_part * 10 + (ch - ord('0'))
                frac_len += 1
        p += 1

    out_next[0] = p
    if frac_len == 0:
        return <double>int_part
    return <double>int_part + (<double>frac_part * ten_pow_neg(frac_len))


cdef inline int count_digits(const unsigned char* p, const unsigned char* end) nogil:
    """Count consecutive [0-9] digits starting at p."""
    cdef int c = 0
    while p < end and p[0] >= ord('0') and p[0] <= ord('9'):
        c += 1
        p += 1
    return c




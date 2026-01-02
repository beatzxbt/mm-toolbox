# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""Ultra-fast parser for Binance bookTicker BBO payloads.

Parses only the required keys ("b", "B", "a", "A") from a JSON payload
and returns a 4-tuple of floats. Designed for zero-allocation scanning over
bytes and suitable for use in high-frequency websocket consumers.

Example:
    >>> from mm_toolbox.misc.parsers.crypto.binance.binance_tob_parser import parse_bbo
    >>> payload = (b'{"e":"bookTicker","u":400900217,"E":1568014460893,'
    ...            b'"T":1568014460891,"s":"BNBUSDT","b":"25.35190000",'
    ...            b'"B":"31.21000000","a":"25.36520000","A":"40.66000000"}')
    >>> parse_bbo(payload)
    (25.3519, 31.21, 25.3652, 40.66)
"""

from libc.stdint cimport uint64_t as u64
from libc.string cimport memchr
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_GET_SIZE
from mm_toolbox.misc.parsers.crypto.binance.binance_tob_parser cimport BestBidOffer
from mm_toolbox.misc.parsers.crypto.binance._bbo_cache cimport (
    _BboShapeEntry,
    cache_find_nogil,
    cache_get_or_create,
    hash_symbol,
)
from mm_toolbox.misc.parsers.json.fastjson cimport (
    is_ws,
    skip_ws,
    ten_pow_neg,
    parse_quoted_decimal,
    count_digits,
)




cdef int _calibrate_bbo_shape(
    const unsigned char* buf,
    Py_ssize_t n,
    Py_ssize_t* u_pos,
    int* u_digits,
    Py_ssize_t* b_pos,
    Py_ssize_t* B_pos,
    Py_ssize_t* a_pos,
    Py_ssize_t* A_pos,
) nogil:
    """Scan once to record offsets for u,b,B,a,A; returns 0 on success."""
    cdef:
        const unsigned char* p = buf
        const unsigned char* end = buf + n
        const unsigned char* q
        unsigned char key
        const unsigned char* nextp

    u_pos[0] = -1
    u_digits[0] = 0
    b_pos[0] = -1
    B_pos[0] = -1
    a_pos[0] = -1
    A_pos[0] = -1

    while True:
        q = <const unsigned char*>memchr(p, ord('"'), <size_t>(end - p))
        if q is NULL or q + 2 >= end:
            break
        if q[2] != ord('"'):
            p = q + 1
            continue
        key = q[1]
        p = q + 3
        p = skip_ws(p, end)
        if p >= end or p[0] != ord(':'):
            p += 1
            continue
        p += 1
        p = skip_ws(p, end)

        if key == ord('u'):
            u_pos[0] = p - buf
            u_digits[0] = count_digits(p, end)
            p += u_digits[0]
            continue

        if key == ord('b') or key == ord('B') or key == ord('a') or key == ord('A'):
            if p >= end or p[0] != ord('"'):
                p += 1
                continue
            p += 1
            if key == ord('b'):
                b_pos[0] = p - buf
            elif key == ord('B'):
                B_pos[0] = p - buf
            elif key == ord('a'):
                a_pos[0] = p - buf
            else:
                A_pos[0] = p - buf
            # advance p to after closing quote quickly
            _ = parse_quoted_decimal(p, end, &nextp)
            p = nextp
            continue

    if u_pos[0] >= 0 and u_digits[0] > 0 and b_pos[0] >= 0 and B_pos[0] >= 0 and a_pos[0] >= 0 and A_pos[0] >= 0:
        return 0
    return -1


cdef BestBidOffer parse_bbo_ptr(const unsigned char* buf, Py_ssize_t n) nogil:
    """Parse Binance bookTicker payload into BestBidOffer (nogil)."""
    cdef:
        const unsigned char* p = buf
        const unsigned char* end = buf + n
        BestBidOffer out
        bint f_b = False
        bint f_B = False
        bint f_a = False
        bint f_A = False
        unsigned char key
        double val
        const unsigned char* nextp

    out.bid_price = 0.0
    out.bid_qty = 0.0
    out.ask_price = 0.0
    out.ask_qty = 0.0

    while True:
        q = <const unsigned char*>memchr(p, ord('"'), <size_t>(end - p))
        if q is NULL or q + 2 >= end:
            break
        if q[2] != ord('"'):
            p = q + 1
            continue
        key = q[1]
        if key == ord('b') or key == ord('B') or key == ord('a') or key == ord('A'):
            p = q + 3
            p = skip_ws(p, end)
            if p >= end or p[0] != ord(':'):
                p += 1
                continue
            p += 1
            p = skip_ws(p, end)
            if p >= end or p[0] != ord('"'):
                p += 1
                continue
            p += 1
            val = parse_quoted_decimal(p, end, &nextp)
            p = nextp
            if key == ord('b'):
                out.bid_price = val
                f_b = True
            elif key == ord('B'):
                out.bid_qty = val
                f_B = True
            elif key == ord('a'):
                out.ask_price = val
                f_a = True
            elif key == ord('A'):
                out.ask_qty = val
                f_A = True
            if f_b and f_B and f_a and f_A:
                break
            continue
        p = q + 1
    return out


cpdef tuple parse_bbo(bytes payload):
    """Parse Binance bookTicker bytes to (bid_price, bid_qty, ask_price, ask_qty).

    Args:
        payload: Raw websocket message as bytes.

    Returns:
        A tuple of 4 floats: (bid_price, bid_qty, ask_price, ask_qty).
    """
    cdef:
        const unsigned char* buf = <const unsigned char*>PyBytes_AS_STRING(payload)
        Py_ssize_t n = PyBytes_GET_SIZE(payload)
        BestBidOffer bbo
    bbo = parse_bbo_ptr(buf, n)
    return (bbo.bid_price, bbo.bid_qty, bbo.ask_price, bbo.ask_qty)


cdef BestBidOffer parse_bbo_cached_ptr(
    const unsigned char* buf,
    Py_ssize_t n,
    const unsigned char* sym,
    Py_ssize_t sym_n,
) nogil:
    """Nogil fast path using C cache; falls back to full scan if missing."""
    cdef:
        _BboShapeEntry* e
        u64 h
        const unsigned char* end = buf + n
        int cur_digits
        Py_ssize_t pb
        Py_ssize_t pB
        Py_ssize_t pa
        Py_ssize_t pA
        const unsigned char* p
        const unsigned char* nextp
        BestBidOffer out
        bint ok = True

    h = hash_symbol(sym, sym_n)
    e = cache_find_nogil(sym, sym_n, h)
    if e is NULL:
        return parse_bbo_ptr(buf, n)

    cur_digits = count_digits(buf + e.u_pos, end)
    pb = e.b_pos + (cur_digits - e.u_digits)
    pB = e.B_pos + (cur_digits - e.u_digits)
    pa = e.a_pos + (cur_digits - e.u_digits)
    pA = e.A_pos + (cur_digits - e.u_digits)

    if pb <= 0 or pB <= 0 or pa <= 0 or pA <= 0 or pb >= n or pB >= n or pa >= n or pA >= n:
        return parse_bbo_ptr(buf, n)

    p = buf + pb
    if p[-1] != ord('"'):
        return parse_bbo_ptr(buf, n)
    out.bid_price = parse_quoted_decimal(p, end, &nextp)

    p = buf + pB
    if p[-1] != ord('"'):
        return parse_bbo_ptr(buf, n)
    out.bid_qty = parse_quoted_decimal(p, end, &nextp)

    p = buf + pa
    if p[-1] != ord('"'):
        return parse_bbo_ptr(buf, n)
    out.ask_price = parse_quoted_decimal(p, end, &nextp)

    p = buf + pA
    if p[-1] != ord('"'):
        return parse_bbo_ptr(buf, n)
    out.ask_qty = parse_quoted_decimal(p, end, &nextp)
    return out

cpdef tuple parse_bbo_cached(bytes payload, bytes symbol, bint unsafe_fast_path=False):
    """Parse using cached offsets per symbol; falls back to scan if shape drifted.

    Args:
        payload: Raw websocket message as bytes.
        symbol: Symbol bytes key for cache.

    Returns:
        A tuple of 4 floats: (bid_price, bid_qty, ask_price, ask_qty).
    """
    cdef:
        const unsigned char* buf = <const unsigned char*>PyBytes_AS_STRING(payload)
        Py_ssize_t n = PyBytes_GET_SIZE(payload)
        const unsigned char* sym = <const unsigned char*>PyBytes_AS_STRING(symbol)
        Py_ssize_t sym_n = PyBytes_GET_SIZE(symbol)
        _BboShapeEntry* e
        int rc
        BestBidOffer bbo
        BestBidOffer bbo2
        int cur_digits
        const unsigned char* end
        Py_ssize_t pb
        Py_ssize_t pB
        Py_ssize_t pa
        Py_ssize_t pA
        const unsigned char* p
        const unsigned char* nextp
        double bid, bid_qty, ask, ask_qty
        bint ok
        u64 h

    h = hash_symbol(sym, sym_n)
    e = cache_find_nogil(sym, sym_n, h)
    if e is NULL:
        e = cache_get_or_create(sym, sym_n, h)
        if e is NULL:
            with nogil:
                bbo = parse_bbo_ptr(buf, n)
            return (bbo.bid_price, bbo.bid_qty, bbo.ask_price, bbo.ask_qty)
        with nogil:
            rc = _calibrate_bbo_shape(buf, n, &e.u_pos, &e.u_digits, &e.b_pos, &e.B_pos, &e.a_pos, &e.A_pos)
        if rc != 0:
            with nogil:
                bbo = parse_bbo_ptr(buf, n)
            return (bbo.bid_price, bbo.bid_qty, bbo.ask_price, bbo.ask_qty)

    end = buf + n
    ok = True
    with nogil:
        cur_digits = count_digits(buf + e.u_pos, end)
        pb = e.b_pos + (cur_digits - e.u_digits)
        pB = e.B_pos + (cur_digits - e.u_digits)
        pa = e.a_pos + (cur_digits - e.u_digits)
        pA = e.A_pos + (cur_digits - e.u_digits)
        if pb <= 0 or pB <= 0 or pa <= 0 or pA <= 0 or pb >= n or pB >= n or pa >= n or pA >= n:
            ok = False
        else:
            p = buf + pb
            if (not unsafe_fast_path) and p[-1] != ord('"'):
                ok = False
            else:
                bid = parse_quoted_decimal(p, end, &nextp)
                p = buf + pB
                if (not unsafe_fast_path) and p[-1] != ord('"'):
                    ok = False
                else:
                    bid_qty = parse_quoted_decimal(p, end, &nextp)
                    p = buf + pa
                    if (not unsafe_fast_path) and p[-1] != ord('"'):
                        ok = False
                    else:
                        ask = parse_quoted_decimal(p, end, &nextp)
                        p = buf + pA
                        if (not unsafe_fast_path) and p[-1] != ord('"'):
                            ok = False
                        else:
                            ask_qty = parse_quoted_decimal(p, end, &nextp)

    if ok:
        if cur_digits != e.u_digits:
            e.u_digits = cur_digits
        return (bid, bid_qty, ask, ask_qty)

    with nogil:
        bbo2 = parse_bbo_ptr(buf, n)
        rc = _calibrate_bbo_shape(buf, n, &e.u_pos, &e.u_digits, &e.b_pos, &e.B_pos, &e.a_pos, &e.A_pos)
    return (bbo2.bid_price, bbo2.bid_qty, bbo2.ask_price, bbo2.ask_qty)



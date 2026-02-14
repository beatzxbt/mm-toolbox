# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""Unsafe fast parser for Binance futures `@bookTicker` payloads.

Assumes canonical layout with no whitespace:
{"e":"bookTicker","u":...,"s":"...","b":"...","B":"...","a":"...","A":"...","T":...,"E":...}

Output type contract for this fast path is aligned with a msgspec schema using:
- e: str
- u, E, T: int
- s: str
- b, B, a, A: float
"""

from libc.string cimport memcmp, memcpy
from libc.stdlib cimport strtod
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from cpython.unicode cimport PyUnicode_DecodeASCII
from .utils cimport parse_quoted_span, parse_u64_until_char

cdef class BinanceBookTickerView:
    """Mutable parsed view reused by the parser for zero-allocation returns."""

    cdef:
        unsigned long long _u
        object _u_obj
        object _E_obj
        object _T_obj
        unsigned long long _seq
        unsigned long long _u_seq

        public object s
        public object b
        public object B
        public object a
        public object A

    def __cinit__(self):
        self._u = 0
        self._u_obj = None
        self._E_obj = None
        self._T_obj = None
        self._seq = 1
        self._u_seq = 0

    @property
    def e(self):
        return "bookTicker"

    @property
    def u(self):
        if self._u_seq != self._seq:
            self._u_obj = self._u
            self._u_seq = self._seq
        return self._u_obj

    @property
    def E(self):
        return self._E_obj

    @property
    def T(self):
        return self._T_obj

    cpdef tuple as_tuple(self):
        return (
            "bookTicker",
            self.u,
            self.E,
            self.T,
            self.s,
            self.b,
            self.B,
            self.a,
            self.A,
        )


cdef class BinanceBookTickerStrictParser:
    """Unsafe fast parser for canonical Binance `@bookTicker` payloads."""

    cdef:
        # symbol cache (multi-symbol friendly)
        char _symbol_buf[64]
        int _symbol_len
        bint _has_symbol
        object _symbol_str
        list _symbol_keys
        list _symbol_vals
        int _symbol_cache_capacity
        int _symbol_cache_next

        # numeric-string caches + parsed float objects
        char _b_buf[128]
        int _b_len
        bint _has_b
        object _b_obj

        char _B_buf[128]
        int _B_len
        bint _has_B
        object _B_obj

        char _a_buf[128]
        int _a_len
        bint _has_a
        object _a_obj

        char _A_buf[128]
        int _A_len
        bint _has_A
        object _A_obj

        # repeated integer object caches
        unsigned long long _last_t
        unsigned long long _last_e
        bint _has_t
        bint _has_e
        object _t_obj
        object _e_obj

        BinanceBookTickerView _view

    def __cinit__(self):
        self._symbol_len = 0
        self._has_symbol = False
        self._symbol_str = None
        self._symbol_keys = []
        self._symbol_vals = []
        self._symbol_cache_capacity = 16
        self._symbol_cache_next = 0

        self._b_len = 0
        self._has_b = False
        self._b_obj = None

        self._B_len = 0
        self._has_B = False
        self._B_obj = None

        self._a_len = 0
        self._has_a = False
        self._a_obj = None

        self._A_len = 0
        self._has_A = False
        self._A_obj = None

        self._last_t = 0
        self._last_e = 0
        self._has_t = False
        self._has_e = False
        self._t_obj = None
        self._e_obj = None

        self._view = BinanceBookTickerView()

    cdef inline object _intern_symbol(
        self,
        const char* ptr,
        Py_ssize_t value_len,
    ):
        cdef:
            int n
            int i
            bytes sym_key
            object symbol

        if value_len >= 64:
            raise ValueError("Symbol too long for fast path")

        if (
            self._has_symbol
            and self._symbol_len == value_len
            and memcmp(ptr, self._symbol_buf, value_len) == 0
        ):
            return self._symbol_str

        n = len(self._symbol_keys)
        for i in range(n):
            sym_key = <bytes>self._symbol_keys[i]
            if len(sym_key) == value_len and memcmp(ptr, <const char*>sym_key, value_len) == 0:
                symbol = self._symbol_vals[i]
                memcpy(self._symbol_buf, ptr, value_len)
                self._symbol_len = <int>value_len
                self._has_symbol = True
                self._symbol_str = symbol
                return symbol

        symbol = PyUnicode_DecodeASCII(ptr, value_len, NULL)
        sym_key = PyBytes_FromStringAndSize(ptr, value_len)

        if n < self._symbol_cache_capacity:
            self._symbol_keys.append(sym_key)
            self._symbol_vals.append(symbol)
        else:
            self._symbol_keys[self._symbol_cache_next] = sym_key
            self._symbol_vals[self._symbol_cache_next] = symbol
            self._symbol_cache_next += 1
            if self._symbol_cache_next == self._symbol_cache_capacity:
                self._symbol_cache_next = 0

        memcpy(self._symbol_buf, ptr, value_len)
        self._symbol_len = <int>value_len
        self._has_symbol = True
        self._symbol_str = symbol
        return symbol

    cpdef object parse(self, bytes payload):
        """Parse one payload and return a mutable typed view."""
        cdef:
            const char* data
            const unsigned char* cursor

            unsigned long long u_value = 0
            unsigned long long t_value = 0
            unsigned long long e_value = 0

            const char* s_start
            const char* s_end
            const char* b_start
            const char* b_end
            const char* B_start
            const char* B_end
            const char* a_start
            const char* a_end
            const char* A_start
            const char* A_end
            Py_ssize_t value_len

            object symbol
            object b_obj
            object B_obj
            object a_obj
            object A_obj
            object t_obj
            object e_obj

            BinanceBookTickerView view

        data = <const char*>PyBytes_AS_STRING(payload)

        cursor = <const unsigned char*>data + 22  # {"e":"bookTicker","u":
        u_value = parse_u64_until_char(&cursor, 44)

        cursor += 6  # ,"s":"
        parse_quoted_span(&cursor, &s_start, &s_end)

        cursor += 7  # ","b":"
        parse_quoted_span(&cursor, &b_start, &b_end)

        cursor += 7  # ","B":"
        parse_quoted_span(&cursor, &B_start, &B_end)

        cursor += 7  # ","a":"
        parse_quoted_span(&cursor, &a_start, &a_end)

        cursor += 7  # ","A":"
        parse_quoted_span(&cursor, &A_start, &A_end)

        cursor += 6  # ","T":
        t_value = parse_u64_until_char(&cursor, 44)

        cursor += 5  # ,"E":
        e_value = parse_u64_until_char(&cursor, 125)

        symbol = self._intern_symbol(s_start, s_end - s_start)

        # b
        value_len = b_end - b_start
        if value_len >= 128:
            raise ValueError("b too long for fast path")
        if self._has_b and self._b_len == value_len and memcmp(b_start, self._b_buf, value_len) == 0:
            b_obj = self._b_obj
        else:
            memcpy(self._b_buf, b_start, value_len)
            self._b_len = <int>value_len
            self._has_b = True
            self._b_obj = strtod(b_start, NULL)
            b_obj = self._b_obj

        # B
        value_len = B_end - B_start
        if value_len >= 128:
            raise ValueError("B too long for fast path")
        if self._has_B and self._B_len == value_len and memcmp(B_start, self._B_buf, value_len) == 0:
            B_obj = self._B_obj
        else:
            memcpy(self._B_buf, B_start, value_len)
            self._B_len = <int>value_len
            self._has_B = True
            self._B_obj = strtod(B_start, NULL)
            B_obj = self._B_obj

        # a
        value_len = a_end - a_start
        if value_len >= 128:
            raise ValueError("a too long for fast path")
        if self._has_a and self._a_len == value_len and memcmp(a_start, self._a_buf, value_len) == 0:
            a_obj = self._a_obj
        else:
            memcpy(self._a_buf, a_start, value_len)
            self._a_len = <int>value_len
            self._has_a = True
            self._a_obj = strtod(a_start, NULL)
            a_obj = self._a_obj

        # A
        value_len = A_end - A_start
        if value_len >= 128:
            raise ValueError("A too long for fast path")
        if self._has_A and self._A_len == value_len and memcmp(A_start, self._A_buf, value_len) == 0:
            A_obj = self._A_obj
        else:
            memcpy(self._A_buf, A_start, value_len)
            self._A_len = <int>value_len
            self._has_A = True
            self._A_obj = strtod(A_start, NULL)
            A_obj = self._A_obj

        # repeated integer object caches
        if self._has_t and t_value == self._last_t:
            t_obj = self._t_obj
        else:
            self._last_t = t_value
            self._t_obj = t_value
            self._has_t = True
            t_obj = self._t_obj

        if self._has_e and e_value == self._last_e:
            e_obj = self._e_obj
        else:
            self._last_e = e_value
            self._e_obj = e_value
            self._has_e = True
            e_obj = self._e_obj

        view = self._view
        view._u = u_value

        # Avoid unnecessary refcount churn on the reusable view object.
        if view._T_obj is not t_obj:
            view._T_obj = t_obj
        if view._E_obj is not e_obj:
            view._E_obj = e_obj
        if view.s is not symbol:
            view.s = symbol
        if view.b is not b_obj:
            view.b = b_obj
        if view.B is not B_obj:
            view.B = B_obj
        if view.a is not a_obj:
            view.a = a_obj
        if view.A is not A_obj:
            view.A = A_obj
        view._seq += 1
        return view


cpdef object parse_binance_bookticker_strict(bytes payload):
    """Parse one Binance `@bookTicker` payload via unsafe fast path."""
    return _DEFAULT_PARSER.parse(payload)


cdef BinanceBookTickerStrictParser _DEFAULT_PARSER = BinanceBookTickerStrictParser()

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""Unsafe fast parser for Binance futures `@trade` payloads.

Assumes canonical layout with no whitespace:
{"e":"trade","E":...,"T":...,"s":"...","t":...,"p":"...","q":"...","X":"...","m":...}

Output type contract is aligned with a msgspec schema using:
- e: str
- E, T, t: int
- s: str
- p, q: float
- X: str
- m: bool
"""

from libc.string cimport memcmp, memcpy
from libc.stdlib cimport strtod
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from cpython.unicode cimport PyUnicode_DecodeASCII
from .utils cimport is_market_token, parse_quoted_span, parse_u64_until_char


cdef class BinanceTradeView:
    """Mutable parsed view reused by the parser for zero-allocation returns."""

    cdef:
        unsigned long long _E
        unsigned long long _T
        unsigned long long _t

        public object s
        public object p
        public object q
        public object X
        public object m

    def __cinit__(self):
        self._E = 0
        self._T = 0
        self._t = 0
        self.s = ""
        self.p = 0.0
        self.q = 0.0
        self.X = ""
        self.m = False

    @property
    def e(self):
        return "trade"

    @property
    def E(self):
        return self._E

    @property
    def T(self):
        return self._T

    @property
    def t(self):
        return self._t

    cpdef tuple as_tuple(self):
        return (
            "trade",
            self.E,
            self.T,
            self.s,
            self.t,
            self.p,
            self.q,
            self.X,
            self.m,
        )


cdef class BinanceTradeStrictParser:
    """Unsafe fast parser for canonical Binance `@trade` payloads."""

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

        # price and quantity string caches + parsed float objects
        char _p_buf[128]
        int _p_len
        bint _has_p
        object _p_obj

        char _q_buf[128]
        int _q_len
        bint _has_q
        object _q_obj

        # order type cache (usually "MARKET")
        char _x_buf[32]
        int _x_len
        bint _has_x
        object _x_obj
        object _market_obj

        BinanceTradeView _view

    def __cinit__(self):
        self._symbol_len = 0
        self._has_symbol = False
        self._symbol_str = None
        self._symbol_keys = []
        self._symbol_vals = []
        self._symbol_cache_capacity = 16
        self._symbol_cache_next = 0

        self._p_len = 0
        self._has_p = False
        self._p_obj = None

        self._q_len = 0
        self._has_q = False
        self._q_obj = None

        self._x_len = 0
        self._has_x = False
        self._x_obj = None
        self._market_obj = "MARKET"

        self._view = BinanceTradeView()

    cdef inline object _intern_symbol(self, const char* ptr, Py_ssize_t value_len):
        cdef:
            int n
            int i
            bytes sym_key
            object symbol

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

    cdef inline object _intern_order_type(self, const char* ptr, Py_ssize_t value_len):
        cdef:
            object x_value

        if (
            self._has_x
            and self._x_len == value_len
            and memcmp(ptr, self._x_buf, value_len) == 0
        ):
            return self._x_obj

        x_value = PyUnicode_DecodeASCII(ptr, value_len, NULL)
        memcpy(self._x_buf, ptr, value_len)
        self._x_len = <int>value_len
        self._has_x = True
        self._x_obj = x_value
        return x_value

    cpdef object parse(self, bytes payload):
        """Parse one payload and return a mutable typed view."""
        cdef:
            const char* data
            const unsigned char* cursor
            const char* s_start
            const char* s_end
            const char* p_start
            const char* p_end
            const char* q_start
            const char* q_end
            const char* x_start
            const char* x_end

            unsigned long long e_value = 0
            unsigned long long t_value = 0
            unsigned long long trade_id = 0

            Py_ssize_t value_len

            object symbol
            object p_obj
            object q_obj
            object x_obj

            bint m_value
            BinanceTradeView view

        data = <const char*>PyBytes_AS_STRING(payload)

        cursor = <const unsigned char*>data + 17  # {"e":"trade","E":
        e_value = parse_u64_until_char(&cursor, 44)

        cursor += 5  # ,"T":
        t_value = parse_u64_until_char(&cursor, 44)

        cursor += 6  # ,"s":"
        parse_quoted_span(&cursor, &s_start, &s_end)

        cursor += 6  # ,"t":
        trade_id = parse_u64_until_char(&cursor, 44)

        cursor += 6  # ,"p":"
        parse_quoted_span(&cursor, &p_start, &p_end)

        cursor += 7  # ","q":"
        parse_quoted_span(&cursor, &q_start, &q_end)

        cursor += 7  # ","X":"
        x_start = <const char*>cursor
        if is_market_token(cursor):
            x_obj = self._market_obj
            cursor += 6
        else:
            parse_quoted_span(&cursor, &x_start, &x_end)
            x_obj = self._intern_order_type(x_start, x_end - x_start)

        cursor += 6  # ,"m":
        m_value = cursor[0] == 116  # 't' in true

        symbol = self._intern_symbol(s_start, s_end - s_start)

        value_len = p_end - p_start
        if (
            self._has_p
            and self._p_len == value_len
            and memcmp(p_start, self._p_buf, value_len) == 0
        ):
            p_obj = self._p_obj
        else:
            memcpy(self._p_buf, p_start, value_len)
            self._p_len = <int>value_len
            self._has_p = True
            self._p_obj = strtod(p_start, NULL)
            p_obj = self._p_obj

        value_len = q_end - q_start
        if (
            self._has_q
            and self._q_len == value_len
            and memcmp(q_start, self._q_buf, value_len) == 0
        ):
            q_obj = self._q_obj
        else:
            memcpy(self._q_buf, q_start, value_len)
            self._q_len = <int>value_len
            self._has_q = True
            self._q_obj = strtod(q_start, NULL)
            q_obj = self._q_obj

        view = self._view
        view._E = e_value
        view._T = t_value
        view._t = trade_id

        # Avoid unnecessary refcount churn on the reusable view object.
        if view.s is not symbol:
            view.s = symbol
        if view.p is not p_obj:
            view.p = p_obj
        if view.q is not q_obj:
            view.q = q_obj
        if view.X is not x_obj:
            view.X = x_obj
        if m_value:
            if view.m is not True:
                view.m = True
        else:
            if view.m is not False:
                view.m = False

        return view


cpdef object parse_binance_trade_strict(bytes payload):
    """Parse one Binance `@trade` payload via unsafe fast path."""
    return _DEFAULT_TRADE_PARSER.parse(payload)


cdef BinanceTradeStrictParser _DEFAULT_TRADE_PARSER = BinanceTradeStrictParser()

cimport numpy as cnp

cdef class ArrayOrderbook:
    cdef: 
        double        _tick_size
        double        _lot_size
        Py_ssize_t    _size

        cnp.ndarray         _asks      
        cnp.ndarray         _bids      
        double[:, :]        _asks_view
        double[:, :]        _bids_view

        Py_ssize_t          _seq_id
        bint                _is_warm

    cdef inline void        _reset(self)

    cdef inline void        _roll_bids(self, Py_ssize_t start_idx=*, bint shift_right=*)
    cdef inline void        _roll_asks(self, Py_ssize_t start_idx=*, bint shift_right=*)

    cdef inline void        _process_matching_ask(self, double ask_sz)
    cdef inline void        _process_matching_bid(self, double bid_sz)
    cdef inline void        _process_middle_ask(self, double ask_px, double ask_sz)
    cdef inline void        _process_middle_bid(self, double bid_px, double bid_sz)
    cdef inline void        _process_lower_ask(self, double ask_px, double ask_sz)
    cdef inline void        _process_higher_bid(self, double bid_px, double bid_sz)
    cdef inline void        _process_higher_ask(self, double ask_px, double ask_sz)
    cdef inline void        _process_lower_bid(self, double bid_px, double bid_sz)

    cpdef void              warmup(self, cnp.ndarray new_asks, cnp.ndarray new_bids, Py_ssize_t new_seq_id)
    cpdef void              update_bbo(self, double bid_px, double bid_sz, double ask_px, double ask_sz, Py_ssize_t new_seq_id)
    cpdef void              update_asks(self, cnp.ndarray updated_asks, Py_ssize_t new_seq_id)
    cpdef void              update_bids(self, cnp.ndarray updated_bids, Py_ssize_t new_seq_id)
    cpdef void              update_full(self, cnp.ndarray updated_asks, cnp.ndarray updated_bids, Py_ssize_t new_seq_id)

    cpdef double            get_mid(self)
    cpdef double            get_wmid(self)
    cpdef double            get_bbo_spread(self)
    cpdef double            get_vamp(self, double size, bint is_base_currency=*)
    cpdef double            get_slippage(self, double size, bint is_bid, bint is_base_currency=*)
    cpdef double            get_imbalance(self, double depth_pct)

    cpdef cnp.ndarray       get_bids(self)
    cpdef cnp.ndarray       get_asks(self)
    cpdef Py_ssize_t        get_seq_id(self)
    cpdef cnp.ndarray       get_best_bid(self)
    cpdef cnp.ndarray       get_best_ask(self)
    cpdef cnp.ndarray       get_bbo(self)

    cpdef bint              is_warm(self)
    cpdef void              ensure_warm(self)

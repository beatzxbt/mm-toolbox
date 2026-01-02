cdef struct BestBidOffer:
    double bid_price
    double bid_qty
    double ask_price
    double ask_qty

cdef BestBidOffer parse_bbo_ptr(const unsigned char* buf, Py_ssize_t n) nogil

cdef BestBidOffer parse_bbo_cached_ptr(
    const unsigned char* buf,
    Py_ssize_t n,
    const unsigned char* sym,
    Py_ssize_t sym_n,
) nogil




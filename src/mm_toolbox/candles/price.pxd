from mm_toolbox.candles.base cimport BaseCandles

cdef class PriceCandles(BaseCandles):
    cdef:
        double px_bucket
        double upper_px_bound
        double lower_px_bound

    cpdef void process_trade(self, double time_ms, bint is_buy, double px, double sz)

from mm_toolbox.candles.base cimport BaseCandles

cdef class MultiCandles(BaseCandles):
    cdef:
        double max_duration_millis
        int    max_ticks
        double max_sz

    cpdef void process_trade(self, double time_ms, bint is_buy, double px, double sz)
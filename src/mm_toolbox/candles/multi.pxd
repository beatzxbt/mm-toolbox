from mm_toolbox.candles.base cimport BaseCandles

cdef class MultiCandles(BaseCandles):
    cdef:
        double max_duration_millis
        int    max_ticks
        double max_size

    cpdef void process_trade(self, object trade)
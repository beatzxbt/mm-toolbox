from mm_toolbox.candles.base cimport BaseCandles

cdef class TickCandles(BaseCandles):
    cdef:
        int ticks_per_bucket

    cpdef void process_trade(self, double time_ms, bint is_buy, double px, double sz)
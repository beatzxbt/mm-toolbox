from mm_toolbox.candles.base cimport BaseCandles

cdef class TimeCandles(BaseCandles):
    cdef:
        double millis_per_bucket
        double next_candle_close_time

    cpdef void process_trade(self, object trade)
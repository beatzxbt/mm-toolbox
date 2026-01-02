from mm_toolbox.candles.base cimport BaseCandles

cdef class TimeCandles(BaseCandles):
    cdef:
        double millis_per_bucket
        double next_candle_close_time
    # def __init__(self, double secs_per_bucket, int num_candles=1000)

    cpdef void process_trade(self, object trade)
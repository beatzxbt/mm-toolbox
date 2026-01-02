from mm_toolbox.candles.base cimport BaseCandles

cdef class TickCandles(BaseCandles):
    cdef int ticks_per_bucket
    # def __init__(self, int ticks_per_bucket, int num_candles=1000)

    cpdef void process_trade(self, object trade)
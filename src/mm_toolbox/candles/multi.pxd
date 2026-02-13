from mm_toolbox.candles.base cimport BaseCandles

cdef class MultiCandles(BaseCandles):
    cdef:
        double max_duration_millis
        int    max_ticks
        double max_size
    # def __init__(
    #     self,
    #     double max_duration_secs,
    #     int max_ticks,
    #     double max_size,
    #     int num_candles=1000,
    #     bint store_trades=True
    # )

    cpdef void process_trade(self, object trade)

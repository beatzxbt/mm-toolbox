from mm_toolbox.candles.base cimport BaseCandles

cdef class VolumeCandles(BaseCandles):
    cdef:
        double volume_per_bucket
    # def __init__(
    #     self,
    #     double volume_per_bucket,
    #     int num_candles=1000,
    #     bint store_trades=True
    # )

    cpdef void process_trade(self, object trade)

from mm_toolbox.candles.base cimport BaseCandles

cdef class VolumeCandles(BaseCandles):
    cdef:
        double volume_per_bucket

    cpdef void process_trade(self, object trade)
from .base cimport BaseCandles

cdef class VolumeTriggerCandles(BaseCandles):
    cdef:
        double volume_per_bucket

    cpdef void process_trade(self, double time, bint is_buy, double px, double sz)
from .base cimport BaseCandles

cdef class MultiTriggerCandles(BaseCandles):
    cdef:
        double max_duration_millis
        double max_ticks
        double max_volume

    cpdef void process_trade(self, double time, bint is_buy, double px, double sz)
from .base cimport BaseCandles

cdef class MultiTriggerCandles(BaseCandles):
    cdef:
        double millis_per_bucket

    cpdef void process_trade(self, double time, bint is_buy, double px, double sz)
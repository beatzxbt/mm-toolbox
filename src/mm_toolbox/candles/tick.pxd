from .base cimport BaseCandles

cdef class TickCandles(BaseCandles):
    cdef:
        double ticks_per_bucket

    cpdef void process_trade(self, double time, bint is_buy, double px, double sz)
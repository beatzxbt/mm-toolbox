from mm_toolbox.candles.base cimport BaseCandles

cdef class PriceCandles(BaseCandles):
    cdef:
        double price_bucket
        double upper_price_bound
        double lower_price_bound

    cpdef void process_trade(self, object trade)
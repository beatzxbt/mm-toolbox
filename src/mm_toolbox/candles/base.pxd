cimport numpy as cnp
from mm_toolbox.ringbuffer.twodim cimport RingBufferTwoDim

cdef class BaseCandles:
    cdef:
        double open_time_ms
        double open_px
        double high_px
        double low_px
        double close_px
        double close_time_ms
        double buy_sz
        double sell_sz
        double vwap_px
        double num_trades
        
        double _cum_volume
        double _total_sz

        RingBufferTwoDim _ringbuffer

    cdef inline double  calculate_vwap(self, double px, double sz) 
    cdef inline bint    is_stale_trade(self, double time_ms) 
    cdef void           insert_candle(self)
    cpdef void          initialize(self, cnp.ndarray trades)
    cpdef void          process_trade(self, double time_ms, bint is_buy, double px, double sz)
    cpdef cnp.ndarray   get_current(self)
    cpdef cnp.ndarray   get_all(self)
import numpy as np
cimport numpy as np

from mm_toolbox.ringbuffer.twodim cimport RingBufferTwoDim

cdef class BaseCandles:
    cdef:
        double              open_price
        double              high_price
        double              low_price
        double              close_price
        double              buy_volume
        double              sell_volume
        double              vwap_price
        double              total_trades
        double              open_time
        double              close_time

        double             _cum_price_volume
        double             _total_volume
        RingBufferTwoDim   _ringbuffer

    # Core funcs
    cdef double             calculate_vwap(self, double px, double sz)
    cdef void               insert_candle(self)
    cpdef void              process_trade(self, double time, bint is_buy, double px, double sz)
    cpdef np.ndarray        unwrapped(self)
    cpdef void              initialize(self, np.ndarray trades)
    
    # Helpers
    cpdef np.ndarray        durations(self)
    cpdef np.ndarray        imbalances(self)
    cpdef np.ndarray        current_candle(self)
    cpdef np.ndarray        open_prices(self)
    cpdef np.ndarray        high_prices(self)
    cpdef np.ndarray        low_prices(self)
    cpdef np.ndarray        close_prices(self)
    cpdef np.ndarray        buy_volumes(self)
    cpdef np.ndarray        sell_volumes(self)
    cpdef np.ndarray        vwap_prices(self)
    cpdef np.ndarray        all_trades(self)
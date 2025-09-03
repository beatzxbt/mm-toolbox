from mm_toolbox.ringbuffer.generic cimport GenericRingBuffer

cdef class BaseCandles:
    cdef:
        object latest_candle
        GenericRingBuffer ringbuffer
        object candle_push_event

        double __cum_volume
        double __total_size

    cdef inline double  calculate_vwap(self, double price, double size) 
    cdef inline bint    is_stale_trade(self, double time_ms) 
    cdef inline void    insert_and_reset_candle(self)
    cpdef void          initialize(self, list trades)
    # def void          process_trade(self, trade: Trade)
from mm_toolbox.ringbuffer.generic cimport GenericRingBuffer

cdef class BaseCandles:
    cdef:
        object latest_candle
        GenericRingBuffer ringbuffer
        object candle_push_event

        double __cum_volume
        double __total_size

    cdef inline double  calculate_vwap(self, double price, double size) noexcept nogil
    cdef inline bint    is_stale_trade(self, double time_ms) 
    cdef inline void    insert_and_reset_candle(self)
    cpdef void          initialize(self, list[object] trades)
    cpdef void          process_trade(self, object trade)

    # def               __len__(self) -> int
    # def               __getitem__(self, index: int) -> Candle
    # def               __aiter__(self) -> AsyncIterator[Candle]
    # async def         __anext__(self) -> Candle
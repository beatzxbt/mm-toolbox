from libc.stdint cimport uint64_t as u64

cdef struct OrderbookLevel:
    double price
    double size
    u64 norders
    u64 ticks
    u64 lots
    u64 __padding1
    u64 __padding2
    u64 __padding3

cdef struct OrderbookLevels:
    u64 num_levels
    OrderbookLevel* levels

cdef OrderbookLevel create_orderbook_level(double price, double size, u64 norders=*) noexcept nogil
cdef OrderbookLevel create_orderbook_level_with_ticks_and_lots(double price, double size, double tick_size, double lot_size, u64 norders=*) noexcept nogil

cdef OrderbookLevels create_orderbook_levels(u64 num_levels, OrderbookLevel* levels) noexcept nogil
cdef void free_orderbook_levels(OrderbookLevels* levels) noexcept nogil

cdef class PyOrderbookLevel:
    cdef OrderbookLevel _level

    @staticmethod
    cdef PyOrderbookLevel from_struct(OrderbookLevel level)
    
    cdef OrderbookLevel to_c_struct(self)

cdef class PyOrderbookLevels:
    cdef OrderbookLevels _levels

    @staticmethod
    cdef PyOrderbookLevels _create(u64 num_levels, OrderbookLevel* levels)
    
    @staticmethod
    cdef PyOrderbookLevels from_ptr(OrderbookLevel* levels_ptr, u64 num_levels)
    
    cdef OrderbookLevels to_c_struct(self)


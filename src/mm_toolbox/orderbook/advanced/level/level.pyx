# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""Raw public orderbook level wrappers and internal entry factories."""

from __future__ import annotations

from typing import Self

from libc.stdint cimport uint64_t as u64
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

from .level cimport OrderbookEntry, OrderbookLevel, OrderbookLevels
from .helpers cimport validate_price, validate_size


cdef inline OrderbookLevel create_orderbook_level(
    double price,
    double size,
    u64 norders=1,
) noexcept nogil:
    cdef OrderbookLevel level
    level.price = price
    level.size = size
    level.norders = norders
    return level


cdef inline OrderbookEntry create_orderbook_entry(
    u64 ticks,
    u64 lots,
    u64 norders=1,
) noexcept nogil:
    cdef OrderbookEntry entry
    entry.ticks = ticks
    entry.lots = lots
    entry.norders = norders
    entry._pad = 0
    return entry


cdef inline OrderbookLevels create_orderbook_levels(
    u64 num_levels,
    OrderbookLevel* levels,
) noexcept nogil:
    cdef OrderbookLevels levels_struct
    levels_struct.num_levels = num_levels
    levels_struct.levels = levels
    return levels_struct


cdef inline void free_orderbook_levels(OrderbookLevels* levels) noexcept nogil:
    if levels != NULL and levels.levels != NULL:
        free(<void*>levels.levels)
        levels.levels = NULL
        levels.num_levels = 0


cdef class PyOrderbookLevel:
    """Python wrapper around a raw public OrderbookLevel."""

    def __cinit__(
        self,
        double price,
        double size,
        u64 norders=1,
        bint verify_values=True,
    ) -> None:
        if verify_values:
            validate_price(price)
            validate_size(size)
        self._level = create_orderbook_level(price, size, norders)

    @staticmethod
    cdef PyOrderbookLevel from_struct(OrderbookLevel level):
        cdef PyOrderbookLevel wrapper = PyOrderbookLevel.__new__(
            PyOrderbookLevel,
            level.price,
            level.size,
            level.norders,
            False,
        )
        return wrapper

    cdef OrderbookLevel to_c_struct(self):
        return self._level

    @property
    def price(self) -> float:
        return self._level.price

    @property
    def size(self) -> float:
        return self._level.size

    @property
    def norders(self) -> int:
        return self._level.norders

    def __repr__(self) -> str:
        return (
            f"PyOrderbookLevel(price={self._level.price}, "
            f"size={self._level.size}, norders={self._level.norders})"
        )


cdef class PyOrderbookLevels:
    """Python wrapper around an owned array of raw public OrderbookLevel values."""

    @staticmethod
    cdef PyOrderbookLevels _create(u64 num_levels, OrderbookLevel* levels):
        cdef PyOrderbookLevels obj = PyOrderbookLevels.__new__(PyOrderbookLevels)
        obj._levels = create_orderbook_levels(num_levels, levels)
        return obj

    def __cinit__(self) -> None:
        self._levels.num_levels = 0
        self._levels.levels = NULL

    def __dealloc__(self) -> None:
        free_orderbook_levels(&self._levels)

    @staticmethod
    cdef PyOrderbookLevels from_ptr(OrderbookLevel* levels_ptr, u64 num_levels):
        if levels_ptr == NULL:
            raise ValueError("Invalid levels_ptr; expected non-null")
        if num_levels == 0:
            raise ValueError("Invalid num_levels; expected >0")

        cdef u64 size = sizeof(OrderbookLevel) * num_levels
        cdef OrderbookLevel* new_levels = <OrderbookLevel*>malloc(size)
        if new_levels == NULL:
            raise MemoryError("Failed to allocate memory for new levels")
        memcpy(new_levels, levels_ptr, size)
        return PyOrderbookLevels._create(num_levels, new_levels)

    @staticmethod
    def from_list(
        list[double] prices,
        list[double] sizes,
        list[u64] norders=None,
        bint verify_values=True,
    ) -> Self:
        cdef Py_ssize_t num = len(prices)
        cdef Py_ssize_t len_sizes = len(sizes)
        cdef Py_ssize_t len_norders = len(norders) if norders is not None else 0
        cdef bint use_default = norders is None

        if num <= 0:
            raise ValueError(f"Invalid prices; expected >0 but got {num}")
        if len_sizes != num:
            raise ValueError(
                "Mismatched lengths; expected same length for prices and sizes "
                f"but got {num} and {len_sizes}"
            )
        if not use_default and len_norders != num:
            raise ValueError(
                "Mismatched lengths; expected same length for prices and norders "
                f"but got {num} and {len_norders}"
            )

        cdef:
            Py_ssize_t i
            u64 size = sizeof(OrderbookLevel) * num
            OrderbookLevel* new_levels = <OrderbookLevel*>malloc(size)

        if new_levels == NULL:
            raise MemoryError("Failed to allocate memory for new levels")

        try:
            for i in range(num):
                if verify_values:
                    validate_price(prices[i])
                    validate_size(sizes[i])
                new_levels[i] = create_orderbook_level(
                    prices[i],
                    sizes[i],
                    norders[i] if not use_default else 1,
                )
        except:
            free(new_levels)
            raise

        return PyOrderbookLevels._create(<u64>num, new_levels)

    @staticmethod
    def from_numpy(
        double[:] prices,
        double[:] sizes,
        u64[:] norders=None,
        bint verify_values=True,
    ) -> Self:
        cdef Py_ssize_t num = prices.shape[0]
        cdef bint use_default = norders is None

        if num <= 0:
            raise ValueError(f"Invalid prices; expected >0 but got {num}")
        if sizes.shape[0] != num:
            raise ValueError(
                "Mismatched lengths; expected same length for prices and sizes "
                f"but got {num} and {sizes.shape[0]}"
            )
        if not use_default and norders.shape[0] != num:
            raise ValueError(
                "Mismatched lengths; expected same length for prices and norders "
                f"but got {num} and {norders.shape[0]}"
            )

        cdef:
            Py_ssize_t i
            u64 size = sizeof(OrderbookLevel) * num
            OrderbookLevel* new_levels = <OrderbookLevel*>malloc(size)

        if new_levels == NULL:
            raise MemoryError("Failed to allocate memory for new levels")

        try:
            for i in range(num):
                if verify_values:
                    validate_price(prices[i])
                    validate_size(sizes[i])
                new_levels[i] = create_orderbook_level(
                    prices[i],
                    sizes[i],
                    norders[i] if not use_default else 1,
                )
        except:
            free(new_levels)
            raise

        return PyOrderbookLevels._create(<u64>num, new_levels)

    cdef OrderbookLevels to_c_struct(self):
        return self._levels

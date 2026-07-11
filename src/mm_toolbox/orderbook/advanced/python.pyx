# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.stdint cimport uint64_t as u64
from libc.stdlib cimport free, malloc, realloc

import numpy as np
cimport numpy as cnp
from cpython.ref cimport Py_INCREF

from .core cimport CoreAdvancedOrderbook
from .enum.enums cimport CyOrderbookSortedness
from .enum.enums import PyOrderbookSortedness
from .ladder.ladder cimport OrderbookLadderData, c_ladder_export_levels
from .level.helpers cimport (
    convert_price_to_tick_trusted,
    convert_size_to_lot_trusted,
    validate_price,
    validate_size,
)
from .level.level cimport (
    OrderbookEntry,
    OrderbookLevel,
    OrderbookLevels,
    PyOrderbookLevel,
    PyOrderbookLevels,
    create_orderbook_entry,
    create_orderbook_level,
)

cnp.import_array()

cdef extern from "numpy/arrayobject.h":
    int PyArray_SetBaseObject(cnp.ndarray arr, object obj)


cdef class PyAdvancedOrderbook:
    """Python-facing advanced orderbook API using raw public levels."""

    def __cinit__(
        self,
        double tick_size,
        double lot_size,
        u64 num_levels,
        object delta_sortedness=None,
        object snapshot_sortedness=None,
    ):
        self._bid_export = NULL
        self._ask_export = NULL
        self._ask_entry_buffer = NULL
        self._bid_entry_buffer = NULL
        self._entry_buffer_capacity = 0
        self._max_levels = num_levels

        if delta_sortedness is None:
            delta_sortedness = PyOrderbookSortedness.UNKNOWN
        if snapshot_sortedness is None:
            snapshot_sortedness = PyOrderbookSortedness.UNKNOWN

        cdef CyOrderbookSortedness delta_code = <CyOrderbookSortedness><int>delta_sortedness
        cdef CyOrderbookSortedness snap_code = <CyOrderbookSortedness><int>snapshot_sortedness
        self._core = CoreAdvancedOrderbook(
            tick_size,
            lot_size,
            num_levels,
            delta_code,
            snap_code,
        )

        self._entry_buffer_capacity = num_levels if num_levels > 0 else 1
        self._ask_entry_buffer = <OrderbookEntry*>malloc(
            self._entry_buffer_capacity * sizeof(OrderbookEntry)
        )
        if self._ask_entry_buffer == NULL:
            raise MemoryError("Failed to allocate ask entry buffer")
        self._bid_entry_buffer = <OrderbookEntry*>malloc(
            self._entry_buffer_capacity * sizeof(OrderbookEntry)
        )
        if self._bid_entry_buffer == NULL:
            free(self._ask_entry_buffer)
            self._ask_entry_buffer = NULL
            raise MemoryError("Failed to allocate bid entry buffer")

        self._bid_export = <OrderbookLevel*>malloc(num_levels * sizeof(OrderbookLevel))
        if self._bid_export == NULL:
            free(self._ask_entry_buffer)
            free(self._bid_entry_buffer)
            self._ask_entry_buffer = NULL
            self._bid_entry_buffer = NULL
            raise MemoryError("Failed to allocate bid export buffer")
        self._ask_export = <OrderbookLevel*>malloc(num_levels * sizeof(OrderbookLevel))
        if self._ask_export == NULL:
            free(self._bid_export)
            free(self._ask_entry_buffer)
            free(self._bid_entry_buffer)
            self._bid_export = NULL
            self._ask_entry_buffer = NULL
            self._bid_entry_buffer = NULL
            raise MemoryError("Failed to allocate ask export buffer")

    def __dealloc__(self):
        if self._bid_export != NULL:
            free(self._bid_export)
        if self._ask_export != NULL:
            free(self._ask_export)
        if self._ask_entry_buffer != NULL:
            free(self._ask_entry_buffer)
        if self._bid_entry_buffer != NULL:
            free(self._bid_entry_buffer)
        self._bid_export = NULL
        self._ask_export = NULL
        self._ask_entry_buffer = NULL
        self._bid_entry_buffer = NULL

    cdef object _array_from_export(self, OrderbookLevel* levels, u64 count):
        cdef cnp.npy_intp dims[1]
        dims[0] = <cnp.npy_intp>(count * sizeof(OrderbookLevel))
        cdef cnp.ndarray base_array = cnp.PyArray_SimpleNewFromData(
            1,
            dims,
            cnp.NPY_UINT8,
            <void*>levels,
        )
        Py_INCREF(self)
        PyArray_SetBaseObject(base_array, self)
        return base_array.view(
            dtype=np.dtype(
                [
                    ("price", np.float64),
                    ("size", np.float64),
                    ("norders", np.uint64),
                ],
                align=True,
            )
        )

    cdef void _ensure_entry_capacity(self, u64 required):
        cdef:
            u64 new_capacity
            void* new_asks
            void* new_bids

        if required <= self._entry_buffer_capacity:
            return

        new_capacity = self._entry_buffer_capacity
        while new_capacity < required:
            new_capacity *= 2

        new_asks = realloc(self._ask_entry_buffer, new_capacity * sizeof(OrderbookEntry))
        if new_asks == NULL:
            raise MemoryError("Failed to grow ask entry buffer")
        self._ask_entry_buffer = <OrderbookEntry*>new_asks

        new_bids = realloc(self._bid_entry_buffer, new_capacity * sizeof(OrderbookEntry))
        if new_bids == NULL:
            raise MemoryError("Failed to grow bid entry buffer")
        self._bid_entry_buffer = <OrderbookEntry*>new_bids
        self._entry_buffer_capacity = new_capacity

    cdef void _normalize_levels_to_entries(
        self,
        OrderbookLevel* levels,
        u64 count,
        OrderbookEntry* entries,
    ):
        cdef u64 i
        for i in range(count):
            validate_price(levels[i].price)
            validate_size(levels[i].size)
            entries[i] = create_orderbook_entry(
                convert_price_to_tick_trusted(levels[i].price, self._core._tick_size_recip),
                convert_size_to_lot_trusted(levels[i].size, self._core._lot_size_recip),
                levels[i].norders,
            )

    cpdef void clear(self):
        self._core.clear()

    cpdef void consume_snapshot(self, PyOrderbookLevels asks, PyOrderbookLevels bids):
        self._ensure_entry_capacity(max(asks._levels.num_levels, bids._levels.num_levels))
        self._normalize_levels_to_entries(
            asks._levels.levels,
            asks._levels.num_levels,
            self._ask_entry_buffer,
        )
        self._normalize_levels_to_entries(
            bids._levels.levels,
            bids._levels.num_levels,
            self._bid_entry_buffer,
        )
        self._core.consume_snapshot_entries(
            self._ask_entry_buffer,
            asks._levels.num_levels,
            self._bid_entry_buffer,
            bids._levels.num_levels,
        )

    cpdef void consume_deltas(self, PyOrderbookLevels asks, PyOrderbookLevels bids):
        self._ensure_entry_capacity(max(asks._levels.num_levels, bids._levels.num_levels))
        self._normalize_levels_to_entries(
            asks._levels.levels,
            asks._levels.num_levels,
            self._ask_entry_buffer,
        )
        self._normalize_levels_to_entries(
            bids._levels.levels,
            bids._levels.num_levels,
            self._bid_entry_buffer,
        )
        self._core.consume_deltas_entries(
            self._ask_entry_buffer,
            asks._levels.num_levels,
            self._bid_entry_buffer,
            bids._levels.num_levels,
        )

    cpdef void consume_bbo(self, PyOrderbookLevel ask, PyOrderbookLevel bid):
        cdef:
            u64 ask_ticks
            u64 ask_lots
            u64 bid_ticks
            u64 bid_lots
            OrderbookEntry ask_entry
            OrderbookEntry bid_entry

        validate_price(ask._level.price)
        validate_size(ask._level.size)
        validate_price(bid._level.price)
        validate_size(bid._level.size)
        ask_ticks = convert_price_to_tick_trusted(ask._level.price, self._core._tick_size_recip)
        ask_lots = convert_size_to_lot_trusted(ask._level.size, self._core._lot_size_recip)
        bid_ticks = convert_price_to_tick_trusted(bid._level.price, self._core._tick_size_recip)
        bid_lots = convert_size_to_lot_trusted(bid._level.size, self._core._lot_size_recip)
        if bid_ticks >= ask_ticks:
            raise ValueError("Crossed BBO; bid price must be below ask price")
        ask_entry = create_orderbook_entry(
            ask_ticks,
            ask_lots,
            ask._level.norders,
        )
        bid_entry = create_orderbook_entry(
            bid_ticks,
            bid_lots,
            bid._level.norders,
        )
        self._core.consume_bbo_entries(ask_entry, bid_entry)

    cpdef void consume_bbo_values(
        self,
        double ask_price,
        double ask_size,
        double bid_price,
        double bid_size,
        u64 ask_norders=1,
        u64 bid_norders=1,
    ):
        cdef:
            u64 ask_ticks
            u64 ask_lots
            u64 bid_ticks
            u64 bid_lots

        validate_price(ask_price)
        validate_size(ask_size)
        validate_price(bid_price)
        validate_size(bid_size)
        ask_ticks = convert_price_to_tick_trusted(ask_price, self._core._tick_size_recip)
        ask_lots = convert_size_to_lot_trusted(ask_size, self._core._lot_size_recip)
        bid_ticks = convert_price_to_tick_trusted(bid_price, self._core._tick_size_recip)
        bid_lots = convert_size_to_lot_trusted(bid_size, self._core._lot_size_recip)
        if bid_ticks >= ask_ticks:
            raise ValueError("Crossed BBO; bid price must be below ask price")
        self._core.consume_bbo_entries(
            create_orderbook_entry(
                ask_ticks,
                ask_lots,
                ask_norders,
            ),
            create_orderbook_entry(
                bid_ticks,
                bid_lots,
                bid_norders,
            ),
        )

    cpdef get_bbo(self):
        self._core._ensure_not_empty()
        return (
            PyOrderbookLevel.from_struct(self._core._level_from_entry(self._core._bids.top())),
            PyOrderbookLevel.from_struct(self._core._level_from_entry(self._core._asks.top())),
        )

    cpdef get_bids(self):
        cdef OrderbookLadderData* data = self._core.get_bids_data()
        if data.num_levels == 0:
            raise RuntimeError("Empty bid side")
        c_ladder_export_levels(self._bid_export, data, self._core._tick_size, self._core._lot_size)
        return PyOrderbookLevels.from_ptr(self._bid_export, data.num_levels)

    cpdef get_asks(self):
        cdef OrderbookLadderData* data = self._core.get_asks_data()
        if data.num_levels == 0:
            raise RuntimeError("Empty ask side")
        c_ladder_export_levels(self._ask_export, data, self._core._tick_size, self._core._lot_size)
        return PyOrderbookLevels.from_ptr(self._ask_export, data.num_levels)

    cpdef get_bids_numpy(self):
        cdef OrderbookLadderData* data = self._core.get_bids_data()
        c_ladder_export_levels(self._bid_export, data, self._core._tick_size, self._core._lot_size)
        return self._array_from_export(self._bid_export, data.num_levels)

    cpdef get_asks_numpy(self):
        cdef OrderbookLadderData* data = self._core.get_asks_data()
        c_ladder_export_levels(self._ask_export, data, self._core._tick_size, self._core._lot_size)
        return self._array_from_export(self._ask_export, data.num_levels)

    cpdef double get_mid_price(self):
        return self._core.get_mid_price()

    cpdef double get_bbo_spread(self):
        return self._core.get_bbo_spread()

    cpdef double get_wmid_price(self):
        return self._core.get_wmid_price()

    cpdef double get_volume_weighted_mid_price(self, double size, bint is_base_currency=True):
        return self._core.get_volume_weighted_mid_price(size, is_base_currency)

    cpdef double get_price_impact(self, double size, bint is_buy, bint is_base_currency=True):
        return self._core.get_price_impact(size, is_buy, is_base_currency)

    cpdef double get_size_for_price_impact_bps(
        self,
        double impact_bps,
        bint is_buy,
        bint is_base_currency=True,
    ):
        return self._core.get_size_for_price_impact_bps(impact_bps, is_buy, is_base_currency)

    cpdef bint is_bbo_crossed(self, double bid_price, double ask_price):
        return self._core.is_bbo_crossed(bid_price, ask_price)

    cpdef bint does_bbo_price_change(self, double bid_price, double ask_price):
        return self._core.does_bbo_price_change(bid_price, ask_price)

    cpdef void consume_snapshot_numpy(
        self,
        double[:] ask_prices,
        double[:] ask_sizes,
        double[:] bid_prices,
        double[:] bid_sizes,
        u64[:] ask_norders=None,
        u64[:] bid_norders=None,
    ):
        cdef:
            Py_ssize_t num_asks = ask_prices.shape[0]
            Py_ssize_t num_bids = bid_prices.shape[0]
            bint use_default_ask_norders = ask_norders is None
            bint use_default_bid_norders = bid_norders is None
            Py_ssize_t i

        if ask_sizes.shape[0] != num_asks:
            raise ValueError(
                f"Mismatched ask lengths; expected {num_asks} sizes "
                f"but got {ask_sizes.shape[0]}"
            )
        if bid_sizes.shape[0] != num_bids:
            raise ValueError(
                f"Mismatched bid lengths; expected {num_bids} sizes "
                f"but got {bid_sizes.shape[0]}"
            )
        if not use_default_ask_norders and ask_norders.shape[0] != num_asks:
            raise ValueError(
                f"Mismatched ask norders; expected {num_asks} "
                f"but got {ask_norders.shape[0]}"
            )
        if not use_default_bid_norders and bid_norders.shape[0] != num_bids:
            raise ValueError(
                f"Mismatched bid norders; expected {num_bids} "
                f"but got {bid_norders.shape[0]}"
            )

        self._ensure_entry_capacity(<u64>max(num_asks, num_bids))
        for i in range(num_asks):
            validate_price(ask_prices[i])
            validate_size(ask_sizes[i])
            self._ask_entry_buffer[i] = create_orderbook_entry(
                convert_price_to_tick_trusted(
                    ask_prices[i],
                    self._core._tick_size_recip,
                ),
                convert_size_to_lot_trusted(ask_sizes[i], self._core._lot_size_recip),
                1 if use_default_ask_norders else ask_norders[i],
            )
        for i in range(num_bids):
            validate_price(bid_prices[i])
            validate_size(bid_sizes[i])
            self._bid_entry_buffer[i] = create_orderbook_entry(
                convert_price_to_tick_trusted(
                    bid_prices[i],
                    self._core._tick_size_recip,
                ),
                convert_size_to_lot_trusted(bid_sizes[i], self._core._lot_size_recip),
                1 if use_default_bid_norders else bid_norders[i],
            )

        self._core.consume_snapshot_entries(
            self._ask_entry_buffer,
            <u64>num_asks,
            self._bid_entry_buffer,
            <u64>num_bids,
        )

    cpdef void consume_deltas_numpy(
        self,
        double[:] ask_prices,
        double[:] ask_sizes,
        double[:] bid_prices,
        double[:] bid_sizes,
        u64[:] ask_norders=None,
        u64[:] bid_norders=None,
    ):
        cdef:
            Py_ssize_t num_asks = ask_prices.shape[0]
            Py_ssize_t num_bids = bid_prices.shape[0]
            bint use_default_ask_norders = ask_norders is None
            bint use_default_bid_norders = bid_norders is None
            Py_ssize_t i

        if ask_sizes.shape[0] != num_asks:
            raise ValueError(
                f"Mismatched ask lengths; expected {num_asks} sizes "
                f"but got {ask_sizes.shape[0]}"
            )
        if bid_sizes.shape[0] != num_bids:
            raise ValueError(
                f"Mismatched bid lengths; expected {num_bids} sizes "
                f"but got {bid_sizes.shape[0]}"
            )
        if not use_default_ask_norders and ask_norders.shape[0] != num_asks:
            raise ValueError(
                f"Mismatched ask norders; expected {num_asks} "
                f"but got {ask_norders.shape[0]}"
            )
        if not use_default_bid_norders and bid_norders.shape[0] != num_bids:
            raise ValueError(
                f"Mismatched bid norders; expected {num_bids} "
                f"but got {bid_norders.shape[0]}"
            )

        self._ensure_entry_capacity(<u64>max(num_asks, num_bids))
        for i in range(num_asks):
            validate_price(ask_prices[i])
            validate_size(ask_sizes[i])
            self._ask_entry_buffer[i] = create_orderbook_entry(
                convert_price_to_tick_trusted(
                    ask_prices[i],
                    self._core._tick_size_recip,
                ),
                convert_size_to_lot_trusted(ask_sizes[i], self._core._lot_size_recip),
                1 if use_default_ask_norders else ask_norders[i],
            )
        for i in range(num_bids):
            validate_price(bid_prices[i])
            validate_size(bid_sizes[i])
            self._bid_entry_buffer[i] = create_orderbook_entry(
                convert_price_to_tick_trusted(
                    bid_prices[i],
                    self._core._tick_size_recip,
                ),
                convert_size_to_lot_trusted(bid_sizes[i], self._core._lot_size_recip),
                1 if use_default_bid_norders else bid_norders[i],
            )

        self._core.consume_deltas_entries(
            self._ask_entry_buffer,
            <u64>num_asks,
            self._bid_entry_buffer,
            <u64>num_bids,
        )

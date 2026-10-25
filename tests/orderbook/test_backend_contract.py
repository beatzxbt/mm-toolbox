"""Independent feed and execution-price regressions for both book backends."""

from math import isinf

import numpy as np
import pytest

from mm_toolbox.orderbook.advanced import (
    OrderbookLevels,
    PyAdvancedOrderbook,
    PyOrderbookSortedness,
)
from mm_toolbox.orderbook.standard import Orderbook, OrderbookLevel


class _Book:
    def __init__(self, backend, descending, capacity=64, lot_size=1.0):
        self.backend = backend
        self.descending = descending
        if backend == "standard":
            self.book = Orderbook(tick_size=1.0, lot_size=lot_size, size=1)
        else:
            order = (
                PyOrderbookSortedness.DESCENDING
                if descending
                else PyOrderbookSortedness.ASCENDING
            )
            self.book = PyAdvancedOrderbook(
                tick_size=1.0,
                lot_size=lot_size,
                num_levels=capacity,
                snapshot_sortedness=order,
                delta_sortedness=order,
            )

    def ingest(self, method, asks, bids):
        # Sorting normalized ticks is stable: raw prices within one tick retain
        # their original feed order in either configured side orientation.
        asks = sorted(asks, key=lambda row: int(row[0]), reverse=self.descending)
        bids = sorted(bids, key=lambda row: int(row[0]), reverse=self.descending)
        if self.backend == "numpy":

            def columns(rows, index, dtype):
                return np.array([row[index] for row in rows], dtype=dtype)

            getattr(self.book, method + "_numpy")(
                columns(asks, 0, np.float64),
                columns(asks, 1, np.float64),
                columns(bids, 0, np.float64),
                columns(bids, 1, np.float64),
                columns(asks, 2, np.uint64),
                columns(bids, 2, np.uint64),
            )
        elif self.backend == "standard":
            getattr(self.book, method)(
                [OrderbookLevel(*row) for row in asks],
                [OrderbookLevel(*row) for row in bids],
            )
        else:

            def levels(rows):
                if not rows:
                    return OrderbookLevels()
                return OrderbookLevels.from_list(
                    [row[0] for row in rows],
                    [row[1] for row in rows],
                    [row[2] for row in rows],
                )

            getattr(self.book, method)(levels(asks), levels(bids))

    def rows(self, side):
        levels = getattr(self.book, "get_" + side)()
        if self.backend == "standard":
            return [(level.ticks, level.lots, level.norders) for level in levels]
        rows = getattr(self.book, "get_" + side + "_numpy")()
        return [
            (int(row["price"]), int(row["size"]), int(row["norders"])) for row in rows
        ]


@pytest.fixture(
    params=[
        (backend, descending)
        for backend in ("standard", "advanced", "numpy")
        for descending in (False, True)
    ],
    ids=lambda case: f"{case[0]}-{'descending' if case[1] else 'ascending'}",
)
def book(request):
    return _Book(*request.param)


def test_snapshot_duplicate_ticks_use_last_absolute_quantity_and_zero_deletes(book):
    book.ingest(
        "consume_snapshot",
        [
            (100.2, 2.0, 2),
            (100.1, 7.0, 7),
            (110.0, 4.0, 4),
            (110.2, 0.0, 0),
            (120.0, 5.0, 5),
        ],
        [
            (99.1, 2.0, 2),
            (99.2, 5.0, 5),
            (80.0, 3.0, 3),
            (80.1, 0.0, 0),
            (70.0, 3.0, 3),
        ],
    )
    assert book.rows("asks") == [(100, 7, 7), (120, 5, 5)]
    assert book.rows("bids") == [(99, 5, 5), (70, 3, 3)]
    assert sum(row[1] for row in book.rows("asks")) == 12
    assert sum(row[1] for row in book.rows("bids")) == 8


@pytest.mark.parametrize("padding", [0, 10], ids=["small", "bulk"])
def test_delta_duplicate_ticks_replace_delete_and_restore(book, padding):
    book.ingest(
        "consume_snapshot",
        [(100.0, 2.0, 2), (110.0, 3.0, 3)],
        [(99.0, 4.0, 4), (90.0, 3.0, 3)],
    )
    book.ingest(
        "consume_deltas",
        [(100.2, 9.0, 9), (100.1, 4.0, 4), (110.0, 5.0, 5), (110.2, 0.0, 0)]
        + [(float(120 + i), 1.0, 1) for i in range(padding)],
        [(99.0, 6.0, 6), (99.2, 0.0, 0), (90.0, 0.0, 0), (90.1, 8.0, 8)],
    )
    assert book.rows("asks") == [(100, 4, 4)] + [
        (120 + i, 1, 1) for i in range(padding)
    ]
    assert book.rows("bids") == [(90, 8, 8)]


def test_final_duplicate_deletion_cannot_remove_opposite_depth(book):
    book.ingest("consume_snapshot", [(100.0, 2.0, 1)], [(99.0, 3.0, 1), (90.0, 4.0, 1)])
    book.ingest("consume_deltas", [(89.0, 2.0, 1), (89.2, 0.0, 0)], [])
    assert book.rows("asks") == [(100, 2, 1)]
    assert book.rows("bids") == [(99, 3, 1), (90, 4, 1)]


@pytest.mark.parametrize("padding", [0, 10], ids=["small", "bulk"])
def test_cross_cleanup_uses_completed_delta_batch(book, padding):
    book.ingest("consume_snapshot", [(100.0, 2.0, 1)], [(99.0, 3.0, 1), (90.0, 4.0, 1)])
    book.ingest(
        "consume_deltas",
        [(95.0, 5.0, 5)] + [(float(110 + i), 1.0, 1) for i in range(padding)],
        [(99.0, 0.0, 0), (94.0, 7.0, 7)],
    )
    assert book.rows("asks")[0] == (95, 5, 5)
    assert book.rows("bids") == [(94, 7, 7), (90, 4, 1)]


def test_partial_cross_removes_only_overlapping_opposite_levels(book):
    book.ingest(
        "consume_snapshot",
        [(100.0, 2.0, 1)],
        [(99.0, 3.0, 1), (98.0, 4.0, 1), (90.0, 5.0, 1)],
    )
    book.ingest("consume_deltas", [(98.0, 6.0, 6)], [])
    assert book.rows("asks")[0] == (98, 6, 6)
    assert book.rows("bids") == [(90, 5, 1)]


def test_cross_wiping_side_without_replacement_is_ignored(book):
    book.ingest("consume_snapshot", [(100.0, 2.0, 1)], [(99.0, 3.0, 1), (90.0, 4.0, 1)])
    book.ingest("consume_deltas", [(89.0, 5.0, 5)], [])
    assert book.rows("asks") == [(100, 2, 1)]
    assert book.rows("bids") == [(99, 3, 1), (90, 4, 1)]


@pytest.mark.parametrize("invalid", ["crossed", "invalid_raw"])
def test_rejected_snapshot_preserves_previous_book(book, invalid):
    book.ingest("consume_snapshot", [(100.0, 2.0, 2)], [(99.0, 3.0, 3)])
    asks = [(98.0, 5.0, 5)]
    if invalid == "invalid_raw":
        # Keep invalid raw data out of the helper's integer sorting key.
        asks = [(100.0, float("nan"), 5)]
    with pytest.raises(ValueError):
        book.ingest("consume_snapshot", asks, [(99.0, 9.0, 9)])
    assert book.rows("asks") == [(100, 2, 2)]
    assert book.rows("bids") == [(99, 3, 3)]


def test_execution_vwap_midpoint_is_not_terminal_level_midpoint(book):
    book.ingest(
        "consume_snapshot",
        [(100.0, 1.0, 1), (110.0, 9.0, 1)],
        [(99.0, 5.0, 1), (80.0, 5.0, 1)],
    )
    # Buy VWAP = 109; sell VWAP = 89.5; their midpoint = 99.25.
    # The old terminal-level calculation returned (110 + 80) / 2 = 95.
    assert book.book.get_volume_weighted_mid_price(10.0) == pytest.approx(99.25)


@pytest.mark.parametrize("size,expected", [(0.5, 99.5), (2.5, 102.5), (10.0, 99.25)])
def test_execution_vwap_partial_fills_and_quote_units(book, size, expected):
    book.ingest(
        "consume_snapshot",
        [(100.0, 1.0, 1), (110.0, 9.0, 1)],
        [(99.0, 5.0, 1), (80.0, 5.0, 1)],
    )
    assert book.book.get_volume_weighted_mid_price(size) == pytest.approx(expected)
    quote_notional = size * book.book.get_mid_price()
    assert book.book.get_volume_weighted_mid_price(
        quote_notional, False
    ) == pytest.approx(expected)


def test_vwap_insufficient_depth_and_zero_price_bid(book):
    book.ingest("consume_snapshot", [(2.0, 2.0, 1)], [(0.0, 1.0, 1)])
    assert book.book.get_volume_weighted_mid_price(1.0) == pytest.approx(1.0)
    assert isinf(book.book.get_volume_weighted_mid_price(1.5))


@pytest.mark.parametrize("backend", ["standard", "advanced", "numpy"])
def test_vwap_exact_fractional_lot_depth_ignores_roundoff_residual(backend):
    book = _Book(backend, False, lot_size=0.1)
    book.ingest(
        "consume_snapshot",
        [(100.0, 0.3, 1), (110.0, 0.6, 1)],
        [(99.0, 0.3, 1), (90.0, 0.6, 1)],
    )
    # 0.9 - 0.3 - 0.6 leaves a positive binary floating-point residual.
    assert book.book.get_volume_weighted_mid_price(0.9) == pytest.approx(299.5 / 3.0)
    assert isinf(book.book.get_volume_weighted_mid_price(0.9001))


def test_midpoint_and_weighted_midpoint_intentionally_floor_to_tick(book):
    book.ingest("consume_snapshot", [(100.0, 1.0, 1)], [(99.0, 1.0, 1)])
    assert book.book.get_mid_price() == 99.0
    assert book.book.get_wmid_price() == 99.0
    assert book.book.get_volume_weighted_mid_price(0.0) == 99.0
    assert book.book.get_volume_weighted_mid_price(-1.0) == 99.0
    assert book.book.get_volume_weighted_mid_price(1.0) == 99.5


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("backend", ["advanced", "numpy"])
def test_snapshot_duplicates_are_resolved_before_capacity_truncation(
    backend, descending
):
    book = _Book(backend, descending, capacity=4)
    asks = [(100.0, float(i), i) for i in range(1, 7)] + [
        (float(tick), 1.0, 1) for tick in (110, 120, 130, 140)
    ]
    book.ingest("consume_snapshot", asks, [(99.0, 2.0, 2)])
    assert book.rows("asks") == [(100, 6, 6), (110, 1, 1), (120, 1, 1), (130, 1, 1)]

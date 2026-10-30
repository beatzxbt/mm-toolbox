"""Thin raw-row adapters; no production implementation serves as an oracle."""

from importlib import import_module
from decimal import Decimal, ROUND_FLOOR
import sys
from pathlib import Path

from mm_toolbox.orderbook.standard import Orderbook, OrderbookLevel
from mm_toolbox.orderbook.advanced import (
    PyAdvancedOrderbook,
    OrderbookLevel as AdvancedLevel,
    OrderbookLevels,
    PyOrderbookSortedness,
)

TARGETS = ("standard", "advanced-python", "advanced-cython")


def ordered(rows, side, sortedness, tick=1.0):
    # Stable sorting keeps final-write precedence for equal-price duplicates.
    reverse = (
        sortedness == 2
        or (sortedness == 3 and side == "asks")
        or (sortedness in (0, 4) and side == "bids")
    )

    def key(row):
        price = Decimal(str(row[0]))
        if not price.is_finite():
            return Decimal(0)  # Preserve invalid payloads for ingestion validation.
        return (price / Decimal(str(tick))).to_integral_value(rounding=ROUND_FLOOR)

    if any(not Decimal(str(row[0])).is_finite() or row[0] < 0 for row in rows):
        return list(
            rows
        )  # Invalid input must reach validation in its original position.
    return sorted(rows, key=key, reverse=reverse)


class StandardAdapter:
    def __init__(self, tick_size, lot_size, capacity=64, sortedness=0):
        self.book = Orderbook(tick_size, lot_size, size=1)
        self.initialized = False

    @staticmethod
    def levels(rows):
        return [OrderbookLevel(*row) for row in rows]

    def snapshot(self, asks, bids):
        self.book.consume_snapshot(self.levels(asks), self.levels(bids))
        self.initialized = True

    def deltas(self, asks, bids):
        self.book.consume_deltas(self.levels(asks), self.levels(bids))

    def bbo(self, ask, bid):
        self.book.consume_bbo(OrderbookLevel(*ask), OrderbookLevel(*bid))
        self.initialized = True

    def clear(self):
        self.book.reset()
        self.initialized = False

    def rows(self, side):
        if not self.initialized:
            return []
        return [
            (x.price, x.size, x.norders) for x in getattr(self.book, "get_" + side)()
        ]

    def query(self, name, *args):
        if name == "get_bbo":
            return tuple((x.price, x.size, x.norders) for x in self.book.get_bbo())
        if name == "counts":
            return (len(self.book.get_bids()), len(self.book.get_asks()))
        return getattr(self.book, name)(*args)


class AdvancedPythonAdapter:
    def __init__(self, tick_size, lot_size, capacity=64, sortedness=0):
        self.sortedness = sortedness
        self.tick = tick_size
        self.book = PyAdvancedOrderbook(
            tick_size,
            lot_size,
            capacity,
            PyOrderbookSortedness(sortedness),
            PyOrderbookSortedness(sortedness),
        )

    def levels(self, rows, side):
        if not rows:
            return OrderbookLevels()
        rows = ordered(rows, side, self.sortedness, self.tick)
        return OrderbookLevels.from_list(
            [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
        )

    def snapshot(self, asks, bids):
        self.book.consume_snapshot(self.levels(asks, "asks"), self.levels(bids, "bids"))

    def deltas(self, asks, bids):
        self.book.consume_deltas(self.levels(asks, "asks"), self.levels(bids, "bids"))

    def bbo(self, ask, bid):
        self.book.consume_bbo(AdvancedLevel(*ask), AdvancedLevel(*bid))

    def clear(self):
        self.book.clear()

    def rows(self, side):
        result = getattr(self.book, "get_" + side + "_numpy")()
        return [
            (float(r["price"]), float(r["size"]), int(r["norders"])) for r in result
        ]

    def query(self, name, *args):
        if name == "get_bbo":
            return tuple((x.price, x.size, x.norders) for x in self.book.get_bbo())
        if name == "counts":
            return (len(self.rows("bids")), len(self.rows("asks")))
        if name == "does_bbo_cross":
            name = "is_bbo_crossed"
        return getattr(self.book, name)(*args)


class AdvancedCythonAdapter:
    def __init__(self, tick_size, lot_size, capacity=64, sortedness=0):
        test_extensions = str(Path(__file__).resolve().parents[1])
        if test_extensions not in sys.path:
            sys.path.append(test_extensions)
        # Required by design: a missing native build must fail the shared suite.
        native = import_module("cython_test_wrapper")
        self.book = native.NativeBookAdapter(tick_size, lot_size, capacity, sortedness)
        self.sortedness = sortedness
        self.tick = tick_size

    def snapshot(self, asks, bids):
        self.book.snapshot(
            ordered(asks, "asks", self.sortedness, self.tick),
            ordered(bids, "bids", self.sortedness, self.tick),
        )

    def deltas(self, asks, bids):
        self.book.deltas(
            ordered(asks, "asks", self.sortedness, self.tick),
            ordered(bids, "bids", self.sortedness, self.tick),
        )

    def bbo(self, ask, bid):
        self.book.bbo(ask, bid)

    def clear(self):
        self.book.clear()

    def rows(self, side):
        return self.book.rows(side)

    def query(self, name, *args):
        return self.book.query(name, *args)


def make_target(target, tick=1.0, lot=1.0, capacity=64, sortedness=0):
    adapter = {
        "standard": StandardAdapter,
        "advanced-python": AdvancedPythonAdapter,
        "advanced-cython": AdvancedCythonAdapter,
    }[target]
    return adapter(tick, lot, capacity, sortedness)

"""Focused input conversion and ownership checks beside the shared scenarios.

The full behavioral suite exercises three book targets. These tests cover the
additional Python array/scalar boundaries and the lifetime of exported views.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from mm_toolbox.orderbook.advanced import (
    AdvancedOrderbook,
    OrderbookLevel,
    OrderbookLevels,
)
from mm_toolbox.orderbook.standard import (
    Orderbook as StandardBook,
    OrderbookLevel as StandardLevel,
)


def payload(*, layout="contiguous", changed=False):
    """Build independent arrays with distinct price, quantity and order fields."""
    values = {
        "ask_prices": [105.0, 106.0],
        "ask_sizes": [75.0, 85.0] if changed else [15.0, 25.0],
        "bid_prices": [100.0, 99.0],
        "bid_sizes": [70.0, 80.0] if changed else [10.0, 20.0],
        "ask_norders": [5, 6],
        "bid_norders": [1, 2],
    }
    result = {}
    for name, rows in values.items():
        dtype = np.uint64 if name.endswith("norders") else np.float64
        array = np.array(rows, dtype=dtype)
        if layout == "strided":
            backing = np.zeros(len(array) * 2, dtype=dtype)
            backing[::2] = array
            array = backing[::2]
        result[name] = array
    return result


def state(book):
    """Copy both exports so the assertion cannot alias reused export buffers."""
    return (book.get_bids_numpy().copy(), book.get_asks_numpy().copy())


def assert_state_equal(actual, expected):
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)


def make_book():
    book = AdvancedOrderbook(tick_size=1.0, lot_size=1.0, num_levels=4)
    book.consume_snapshot_numpy(**payload())
    return book


@pytest.mark.parametrize("method", ["snapshot", "deltas"])
@pytest.mark.parametrize(
    "fault",
    [
        "ask_sizes",
        "bid_sizes",
        "ask_norders",
        "bid_norders",
        "rank",
        "price_dtype",
        "orders_dtype",
        "readonly",
    ],
)
def test_numpy_rejection_preserves_both_sides(method, fault):
    """N6: malformed arrays reject before any of their valid rows take effect."""
    book = make_book()
    previous = state(book)
    arrays = payload(changed=True)
    if fault in {"ask_sizes", "bid_sizes", "ask_norders", "bid_norders"}:
        arrays[fault] = arrays[fault][:-1]
    elif fault == "rank":
        arrays["ask_prices"] = arrays["ask_prices"].reshape(-1, 1)
    elif fault == "price_dtype":
        arrays["ask_prices"] = arrays["ask_prices"].astype(np.float32)
    elif fault == "orders_dtype":
        arrays["bid_norders"] = arrays["bid_norders"].astype(np.int64)
    else:
        # The writable memoryview signature explicitly rejects read-only input.
        arrays["ask_prices"].flags.writeable = False
    with pytest.raises(ValueError):
        getattr(book, f"consume_{method}_numpy")(**arrays)
    assert_state_equal(state(book), previous)
    # A rejected call must not poison the reusable ingestion buffers.
    book.consume_deltas_numpy(**payload(changed=True))
    bids, asks = state(book)
    assert bids["size"].tolist() == [70.0, 80.0]
    assert asks["size"].tolist() == [75.0, 85.0]


@pytest.mark.parametrize("layout", ["contiguous", "strided"])
@pytest.mark.parametrize("method", ["snapshot", "deltas"])
def test_supported_numpy_layouts_preserve_inputs_and_exact_rows(layout, method):
    """N7/N8: supported strides read every field and never mutate caller arrays."""
    book = make_book()
    arrays = payload(layout=layout, changed=True)
    original = {name: array.copy() for name, array in arrays.items()}
    getattr(book, f"consume_{method}_numpy")(**arrays)
    bids, asks = state(book)
    assert bids.tolist() == [(100.0, 70.0, 1), (99.0, 80.0, 2)]
    assert asks.tolist() == [(105.0, 75.0, 5), (106.0, 85.0, 6)]
    for name, array in arrays.items():
        np.testing.assert_array_equal(array, original[name])
        array[:] = 0
    assert_state_equal(state(book), (bids, asks))


@pytest.mark.parametrize("method", ["snapshot", "deltas"])
def test_optional_numpy_order_counts_default_to_one(method):
    """Omitted order arrays populate every accepted row with the documented default."""
    book = make_book()
    arrays = payload(changed=True)
    arrays.pop("ask_norders")
    arrays.pop("bid_norders")
    getattr(book, f"consume_{method}_numpy")(**arrays)
    for side in state(book):
        assert side["norders"].tolist() == [1, 1]


def test_python_ingestion_routes_have_identical_full_state():
    """Array, level and scalar routes preserve metadata across a short mixed trace."""
    arrays_book = make_book()
    levels_book = AdvancedOrderbook(tick_size=1.0, lot_size=1.0, num_levels=4)
    asks = OrderbookLevels.from_list([105.0, 106.0], [15.0, 25.0], [5, 6])
    bids = OrderbookLevels.from_list([100.0, 99.0], [10.0, 20.0], [1, 2])
    levels_book.consume_snapshot(asks, bids)
    assert_state_equal(state(arrays_book), state(levels_book))
    arrays_book.consume_deltas_numpy(**payload(changed=True))
    levels_book.consume_deltas(
        OrderbookLevels.from_list([105.0, 106.0], [75.0, 85.0], [5, 6]),
        OrderbookLevels.from_list([100.0, 99.0], [70.0, 80.0], [1, 2]),
    )
    assert_state_equal(state(arrays_book), state(levels_book))
    arrays_book.consume_bbo_values(106.0, 8.0, 99.0, 7.0, 17, 19)
    levels_book.consume_bbo(
        OrderbookLevel(106.0, 8.0, 17), OrderbookLevel(99.0, 7.0, 19)
    )
    assert_state_equal(state(arrays_book), state(levels_book))
    assert state(arrays_book)[0].tolist() == [(99.0, 7.0, 19)]
    assert state(arrays_book)[1].tolist() == [(106.0, 8.0, 17)]


def test_numpy_export_reuse_copy_side_isolation_and_owner_lifetime():
    """N9: reusable views follow their owner while explicit copies stay independent."""
    book = make_book()
    bid_view, ask_view = book.get_bids_numpy(), book.get_asks_numpy()
    bid_copy, ask_copy = bid_view.copy(), ask_view.copy()
    assert bid_view.dtype.names == ("price", "size", "norders")
    assert bid_view.dtype.itemsize == 24
    assert not np.shares_memory(bid_view, ask_view)
    book.consume_deltas_numpy(**payload(changed=True))
    refreshed_bid = book.get_bids_numpy()
    assert np.shares_memory(bid_view, refreshed_bid)
    assert bid_view["size"].tolist() == [70.0, 80.0]
    # Only a bid export occurred, so the ask export buffer is still untouched.
    np.testing.assert_array_equal(ask_view, ask_copy)
    refreshed_ask = book.get_asks_numpy()
    assert np.shares_memory(ask_view, refreshed_ask)
    assert ask_view["size"].tolist() == [75.0, 85.0]
    assert bid_copy["size"].tolist() == [10.0, 20.0]
    assert ask_copy["size"].tolist() == [15.0, 25.0]
    del book
    gc.collect()
    assert bid_view["size"].tolist() == [70.0, 80.0]
    assert ask_view["size"].tolist() == [75.0, 85.0]


@pytest.mark.parametrize("operation", ["snapshot", "deltas", "bbo"])
def test_standard_book_owns_ingested_mutable_levels(operation):
    """N8: reusing caller-owned level objects cannot rewrite retained state."""
    book = StandardBook(tick_size=1.0, lot_size=1.0, size=1)
    book.consume_snapshot(
        [StandardLevel(105.0, 15.0, 5)], [StandardLevel(100.0, 10.0, 1)]
    )
    ask, bid = StandardLevel(105.0, 8.0, 17), StandardLevel(100.0, 7.0, 19)
    if operation == "bbo":
        book.consume_bbo(ask, bid)
    else:
        getattr(book, f"consume_{operation}")([ask], [bid])
    for level in (ask, bid):
        level.price = level.size = 1.0
        level.norders = 99
        level.ticks = level.lots = 1
    actual_bid, actual_ask = book.get_bbo()
    assert (actual_bid.price, actual_bid.size, actual_bid.norders) == (100.0, 7.0, 19)
    assert (actual_ask.price, actual_ask.size, actual_ask.norders) == (105.0, 8.0, 17)

"""Tests for NumPy array ingestion methods.

Layer 2 tests: validate direct numpy array ingestion for snapshots and deltas,
including mismatched lengths, norders handling, modifications, deletions,
and equivalence with PyOrderbookLevels ingestion.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.orderbook.advanced.conftest import (
    _bids_asks_arrays,
    _make_levels,
    _mk_book,
)


class TestNumpyIngestion:
    """Layer 2: Test direct numpy array ingestion methods."""

    def test_basic_snapshot_with_numpy_arrays(self):
        """Given numpy arrays, When consume_snapshot_numpy called, Then book populated correctly."""
        ob = _mk_book(num_levels=64)

        bid_prices = np.array([100.0, 99.99, 99.98], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ask_prices = np.array([100.01, 100.02, 100.03], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5, 3.5], dtype=np.float64)

        ob.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        bid_arr, ask_arr = _bids_asks_arrays(ob)
        assert list(bid_arr["price"]) == pytest.approx([100.0, 99.99, 99.98])
        assert list(ask_arr["price"]) == pytest.approx([100.01, 100.02, 100.03])
        assert list(bid_arr["size"]) == pytest.approx([1.0, 2.0, 3.0])
        assert list(ask_arr["size"]) == pytest.approx([1.5, 2.5, 3.5])
        assert ob.get_bbo_spread() == pytest.approx(0.01)

    def test_snapshot_mismatched_lengths_raise(self):
        """Given mismatched numpy array lengths, When consume_snapshot_numpy called, Then raises ValueError."""
        ob = _mk_book(num_levels=64)

        bid_prices = np.array([100.0, 99.99], dtype=np.float64)
        bid_sizes = np.array([1.0], dtype=np.float64)
        ask_prices = np.array([100.01], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5], dtype=np.float64)

        with pytest.raises(ValueError):
            ob.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

    def test_snapshot_norders_length_mismatch_raise(self):
        """Given norders arrays not matching price/size length, When consume_snapshot_numpy called, Then raises ValueError."""
        ob = _mk_book(num_levels=64)

        bid_prices = np.array([100.0, 99.99], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0], dtype=np.float64)
        bid_norders = np.array([1], dtype=np.uint64)
        ask_prices = np.array([100.01, 100.02], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5], dtype=np.float64)
        ask_norders = np.array([1, 2, 3], dtype=np.uint64)

        with pytest.raises(ValueError):
            ob.consume_snapshot_numpy(
                ask_prices, ask_sizes, bid_prices, bid_sizes, ask_norders, bid_norders
            )

    def test_snapshot_with_norders(self):
        """Given explicit norders arrays, When consume_snapshot_numpy called, Then norders stored correctly."""
        ob = _mk_book(num_levels=64)

        bid_prices = np.array([100.0, 99.99], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0], dtype=np.float64)
        bid_norders = np.array([5, 10], dtype=np.uint64)
        ask_prices = np.array([100.01, 100.02], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5], dtype=np.float64)
        ask_norders = np.array([3, 7], dtype=np.uint64)

        ob.consume_snapshot_numpy(
            ask_prices, ask_sizes, bid_prices, bid_sizes, ask_norders, bid_norders
        )

        bid_arr, ask_arr = _bids_asks_arrays(ob)
        assert list(bid_arr["norders"]) == [5, 10]
        assert list(ask_arr["norders"]) == [3, 7]

    def test_numpy_export_uses_raw_public_fields(self):
        """Given numpy arrays, When exported, Then public fields are price/size/norders."""
        ob = _mk_book(num_levels=64)

        bid_prices = np.array([100.0, 99.99], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0], dtype=np.float64)
        ask_prices = np.array([100.01, 100.02], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5], dtype=np.float64)

        ob.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        bid_arr, ask_arr = _bids_asks_arrays(ob)
        assert bid_arr.dtype.names == ("price", "size", "norders")
        assert ask_arr.dtype.names == ("price", "size", "norders")
        assert bid_arr.dtype.itemsize == 24
        assert ask_arr.dtype.itemsize == 24

    def test_snapshot_without_norders_defaults_to_one(self):
        """Given no norders arrays, When consume_snapshot_numpy called, Then defaults to 1."""
        ob = _mk_book(num_levels=64)

        bid_prices = np.array([100.0, 99.99], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0], dtype=np.float64)
        ask_prices = np.array([100.01, 100.02], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5], dtype=np.float64)

        ob.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        bid_arr, ask_arr = _bids_asks_arrays(ob)
        assert list(bid_arr["norders"]) == [1, 1]
        assert list(ask_arr["norders"]) == [1, 1]

    def test_basic_deltas_with_numpy_arrays(self):
        """Given numpy delta arrays, When consume_deltas_numpy called, Then levels updated correctly."""
        ob = _mk_book(num_levels=64)

        # Initial snapshot
        bid_prices = np.array([100.0, 99.99, 99.98], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ask_prices = np.array([100.01, 100.02, 100.03], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        ob.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        # Apply delta: add a new ask level
        delta_ask_prices = np.array([100.04], dtype=np.float64)
        delta_ask_sizes = np.array([1.0], dtype=np.float64)
        # Use far-away price with zero size for no-op bid delta
        delta_bid_prices = np.array([1.0], dtype=np.float64)
        delta_bid_sizes = np.array([0.0], dtype=np.float64)

        ob.consume_deltas_numpy(
            delta_ask_prices, delta_ask_sizes, delta_bid_prices, delta_bid_sizes
        )

        _, ask_arr = _bids_asks_arrays(ob)
        assert list(ask_arr["price"]) == pytest.approx([100.01, 100.02, 100.03, 100.04])

    def test_deltas_mismatched_lengths_raise(self):
        """Given mismatched numpy delta array lengths, When consume_deltas_numpy called, Then raises ValueError."""
        ob = _mk_book(num_levels=64)

        ask_prices = np.array([100.01], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5], dtype=np.float64)
        bid_prices = np.array([100.0], dtype=np.float64)
        bid_sizes = np.array([1.0], dtype=np.float64)

        with pytest.raises(ValueError):
            ob.consume_deltas_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

    def test_deltas_norders_length_mismatch_raise(self):
        """Given norders arrays not matching delta length, When consume_deltas_numpy called, Then raises ValueError."""
        ob = _mk_book(num_levels=64)

        ask_prices = np.array([100.01, 100.02], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5], dtype=np.float64)
        ask_norders = np.array([1], dtype=np.uint64)
        bid_prices = np.array([100.0, 99.99], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0], dtype=np.float64)
        bid_norders = np.array([1, 2, 3], dtype=np.uint64)

        with pytest.raises(ValueError):
            ob.consume_deltas_numpy(
                ask_prices, ask_sizes, bid_prices, bid_sizes, ask_norders, bid_norders
            )

    def test_deltas_modify_existing_level(self):
        """Given delta modifying existing level, When consume_deltas_numpy called, Then level updated."""
        ob = _mk_book(num_levels=64)

        # Initial snapshot
        bid_prices = np.array([100.0, 99.99, 99.98], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ask_prices = np.array([100.01, 100.02, 100.03], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        ob.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        # Modify the best bid
        delta_bid_prices = np.array([100.0], dtype=np.float64)
        delta_bid_sizes = np.array([5.0], dtype=np.float64)
        delta_bid_norders = np.array([10], dtype=np.uint64)
        # No-op for asks
        delta_ask_prices = np.array([999.0], dtype=np.float64)
        delta_ask_sizes = np.array([0.0], dtype=np.float64)

        ob.consume_deltas_numpy(
            delta_ask_prices,
            delta_ask_sizes,
            delta_bid_prices,
            delta_bid_sizes,
            bid_norders=delta_bid_norders,
        )

        bid_arr, _ = _bids_asks_arrays(ob)
        assert bid_arr["size"][0] == pytest.approx(5.0)
        assert bid_arr["norders"][0] == 10

    def test_deltas_delete_level_with_zero_size(self):
        """Given delta with size=0, When consume_deltas_numpy called, Then level deleted."""
        ob = _mk_book(num_levels=64)

        # Initial snapshot
        bid_prices = np.array([100.0, 99.99, 99.98], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ask_prices = np.array([100.01, 100.02, 100.03], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        ob.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        # Delete the best bid
        delta_bid_prices = np.array([100.0], dtype=np.float64)
        delta_bid_sizes = np.array([0.0], dtype=np.float64)
        # No-op for asks
        delta_ask_prices = np.array([999.0], dtype=np.float64)
        delta_ask_sizes = np.array([0.0], dtype=np.float64)

        ob.consume_deltas_numpy(
            delta_ask_prices, delta_ask_sizes, delta_bid_prices, delta_bid_sizes
        )

        bid_arr, _ = _bids_asks_arrays(ob)
        assert list(bid_arr["price"]) == pytest.approx([99.99, 99.98])

    def test_numpy_vs_pyorderbooklevels_equivalence(self):
        """Given same data via numpy and PyOrderbookLevels, When consumed, Then produce identical results."""
        ob_numpy = _mk_book(num_levels=64)
        ob_py = _mk_book(num_levels=64)

        bid_prices_list = [100.0, 99.99, 99.98, 99.97]
        bid_sizes_list = [1.0, 2.0, 3.0, 4.0]
        ask_prices_list = [100.01, 100.02, 100.03, 100.04]
        ask_sizes_list = [1.5, 2.5, 3.5, 4.5]

        # Numpy ingestion
        bid_prices = np.array(bid_prices_list, dtype=np.float64)
        bid_sizes = np.array(bid_sizes_list, dtype=np.float64)
        ask_prices = np.array(ask_prices_list, dtype=np.float64)
        ask_sizes = np.array(ask_sizes_list, dtype=np.float64)
        ob_numpy.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        # PyOrderbookLevels ingestion
        bids_py, _ = _make_levels(
            prices=bid_prices_list,
            sizes=bid_sizes_list,
            norders=[1] * 4,
            with_precision=False,
        )
        asks_py, _ = _make_levels(
            prices=ask_prices_list,
            sizes=ask_sizes_list,
            norders=[1] * 4,
            with_precision=False,
        )
        ob_py.consume_snapshot(asks_py, bids_py)

        # Compare results
        bid_arr_numpy, ask_arr_numpy = _bids_asks_arrays(ob_numpy)
        bid_arr_py, ask_arr_py = _bids_asks_arrays(ob_py)

        assert list(bid_arr_numpy["price"]) == pytest.approx(list(bid_arr_py["price"]))
        assert list(ask_arr_numpy["price"]) == pytest.approx(list(ask_arr_py["price"]))
        assert list(bid_arr_numpy["size"]) == pytest.approx(list(bid_arr_py["size"]))
        assert list(ask_arr_numpy["size"]) == pytest.approx(list(ask_arr_py["size"]))
        assert ob_numpy.get_mid_price() == ob_py.get_mid_price()
        assert ob_numpy.get_bbo_spread() == ob_py.get_bbo_spread()

    def test_empty_arrays_snapshot(self):
        """Given empty arrays for one side, When consume_snapshot_numpy called, Then that side is empty."""
        ob = _mk_book(num_levels=64)

        # Only populate asks, empty bids
        bid_prices = np.array([], dtype=np.float64)
        bid_sizes = np.array([], dtype=np.float64)
        ask_prices = np.array([100.01, 100.02], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5], dtype=np.float64)

        ob.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        # Should have asks but no bids
        _, ask_arr = _bids_asks_arrays(ob)
        assert len(ask_arr) == 2

        # Orderbook is half-empty, so mid_price should raise
        with pytest.raises(RuntimeError):
            ob.get_mid_price()

    def test_single_level_numpy(self):
        """Given single level per side via numpy, When consumed, Then book has 1 level each side."""
        ob = _mk_book(num_levels=64)

        bid_prices = np.array([100.0], dtype=np.float64)
        bid_sizes = np.array([1.0], dtype=np.float64)
        ask_prices = np.array([100.01], dtype=np.float64)
        ask_sizes = np.array([1.5], dtype=np.float64)

        ob.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        bid_arr, ask_arr = _bids_asks_arrays(ob)
        assert len(bid_arr) == 1
        assert len(ask_arr) == 1
        assert bid_arr["price"][0] == pytest.approx(100.0)
        assert ask_arr["price"][0] == pytest.approx(100.01)

    def test_repeated_numpy_snapshots_smaller_and_larger_do_not_keep_stale_rows(self):
        """Given reusable ingestion scratch, When lengths change, Then exported rows match current book."""
        ob = _mk_book(num_levels=64)

        ob.consume_snapshot_numpy(
            np.array([100.01, 100.02, 100.03, 100.04], dtype=np.float64),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            np.array([100.0, 99.99, 99.98, 99.97], dtype=np.float64),
            np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64),
        )
        ob.consume_snapshot_numpy(
            np.array([101.01], dtype=np.float64),
            np.array([5.0], dtype=np.float64),
            np.array([101.0], dtype=np.float64),
            np.array([5.5], dtype=np.float64),
        )
        bid_arr, ask_arr = _bids_asks_arrays(ob)
        assert list(bid_arr["price"]) == pytest.approx([101.0])
        assert list(ask_arr["price"]) == pytest.approx([101.01])

        ob.consume_snapshot_numpy(
            np.array([102.01, 102.02, 102.03], dtype=np.float64),
            np.array([6.0, 7.0, 8.0], dtype=np.float64),
            np.array([102.0, 101.99, 101.98], dtype=np.float64),
            np.array([6.5, 7.5, 8.5], dtype=np.float64),
        )
        bid_arr, ask_arr = _bids_asks_arrays(ob)
        assert list(bid_arr["price"]) == pytest.approx([102.0, 101.99, 101.98])
        assert list(ask_arr["price"]) == pytest.approx([102.01, 102.02, 102.03])

    def test_numpy_export_reuses_same_side_scratch_buffer(self):
        """Given exported bid view, When bid export runs again, Then prior view reflects scratch rewrite."""
        ob = _mk_book(num_levels=64)
        ob.consume_snapshot_numpy(
            np.array([100.01, 100.02], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([100.0, 99.99], dtype=np.float64),
            np.array([1.5, 2.5], dtype=np.float64),
        )
        first_bid_view = ob.get_bids_numpy()

        ob.consume_snapshot_numpy(
            np.array([101.01, 101.02], dtype=np.float64),
            np.array([3.0, 4.0], dtype=np.float64),
            np.array([101.0, 100.99], dtype=np.float64),
            np.array([3.5, 4.5], dtype=np.float64),
        )
        second_bid_view = ob.get_bids_numpy()

        assert np.shares_memory(first_bid_view, second_bid_view)
        assert list(first_bid_view["price"]) == pytest.approx([101.0, 100.99])
        assert list(second_bid_view["price"]) == pytest.approx([101.0, 100.99])

    def test_numpy_export_copy_preserves_previous_values(self):
        """Given copied export, When scratch is rewritten, Then copy keeps old values."""
        ob = _mk_book(num_levels=64)
        ob.consume_snapshot_numpy(
            np.array([100.01, 100.02], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([100.0, 99.99], dtype=np.float64),
            np.array([1.5, 2.5], dtype=np.float64),
        )
        first_bid_copy = ob.get_bids_numpy().copy()

        ob.consume_snapshot_numpy(
            np.array([101.01, 101.02], dtype=np.float64),
            np.array([3.0, 4.0], dtype=np.float64),
            np.array([101.0, 100.99], dtype=np.float64),
            np.array([3.5, 4.5], dtype=np.float64),
        )
        second_bid_view = ob.get_bids_numpy()

        assert not np.shares_memory(first_bid_copy, second_bid_view)
        assert list(first_bid_copy["price"]) == pytest.approx([100.0, 99.99])
        assert list(second_bid_view["price"]) == pytest.approx([101.0, 100.99])

    def test_numpy_export_sides_use_independent_scratch_buffers(self):
        """Given bid and ask exports, When one side is exported, Then other side view is not rewritten."""
        ob = _mk_book(num_levels=64)
        ob.consume_snapshot_numpy(
            np.array([100.01, 100.02], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([100.0, 99.99], dtype=np.float64),
            np.array([1.5, 2.5], dtype=np.float64),
        )
        ask_view = ob.get_asks_numpy()
        bid_view = ob.get_bids_numpy()

        ob.consume_snapshot_numpy(
            np.array([101.01, 101.02], dtype=np.float64),
            np.array([3.0, 4.0], dtype=np.float64),
            np.array([101.0, 100.99], dtype=np.float64),
            np.array([3.5, 4.5], dtype=np.float64),
        )
        refreshed_bid_view = ob.get_bids_numpy()

        assert np.shares_memory(bid_view, refreshed_bid_view)
        assert not np.shares_memory(ask_view, refreshed_bid_view)
        assert list(ask_view["price"]) == pytest.approx([100.01, 100.02])
        assert list(refreshed_bid_view["price"]) == pytest.approx([101.0, 100.99])

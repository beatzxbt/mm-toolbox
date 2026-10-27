"""Futures U/u/pu bridging and invalidation, with no network calls."""

from types import SimpleNamespace

import msgspec
import pytest

from examples.binance_stream import BinanceDataProcessor, OrderbookDelta, StreamMessage
from examples._stream_guard import FeedIntegrityError


@pytest.fixture
def processor():
    processor = BinanceDataProcessor.__new__(BinanceDataProcessor)
    processor.orderbook = None
    processor.applied = []
    processor._pending_orderbook = SimpleNamespace(
        consume_deltas=lambda **levels: processor.applied.append(levels)
    )
    processor.last_update_id = 100
    processor._depth_started = False
    processor._depth_update_count = 0
    processor.decoder = msgspec.json.Decoder(type=StreamMessage)
    return processor


def delta(first, final, previous):
    return OrderbookDelta(final, first, previous, [(100.0, 1.0)], [(101.0, 1.0)])


def test_initial_bridge_includes_snapshot_id_and_publishes_book_only_after_apply(
    processor,
):
    processor._handle_depth(delta(90, 99, 89))
    assert processor.orderbook is None
    assert processor.applied == []
    processor._handle_depth(delta(90, 100, 89))
    assert processor._depth_started
    assert processor.orderbook is not None
    assert len(processor.applied) == 1
    # Futures IDs need not advance by one; pu establishes continuity.
    processor._handle_depth(delta(108, 110, 100))
    assert processor.last_update_id == 110
    assert len(processor.applied) == 2


def test_unbridgeable_initial_delta_fails_before_exposing_snapshot(processor):
    with pytest.raises(FeedIntegrityError, match="cannot bridge"):
        processor._handle_depth(delta(101, 110, 100))
    assert processor.orderbook is None
    assert not processor._depth_started
    assert processor.applied == []


def test_previous_final_id_gap_propagates_through_dispatch_and_invalidates_book(
    processor,
):
    processor._handle_depth(delta(90, 105, 89))
    message = StreamMessage("depth", msgspec.to_builtins(delta(106, 110, 104)))
    with pytest.raises(FeedIntegrityError, match="expected pu=105"):
        processor._process_message(msgspec.json.encode(message))
    assert processor.orderbook is None
    assert processor.last_update_id == 105
    assert len(processor.applied) == 1


def test_duplicate_depth_is_not_applied_twice(processor):
    message = delta(90, 105, 89)
    processor._handle_depth(message)
    processor._handle_depth(message)
    assert len(processor.applied) == 1

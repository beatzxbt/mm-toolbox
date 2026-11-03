"""Offline regressions for the independent Binance replay reference."""

import pytest

from tests.orderbook.standard.binance_replay import (
    BookTickerEvent,
    DepthEvent,
    _apply_depth_to_reference,
    _lots,
    _replay_depth_with_reference,
    _replay_interleaved_with_book_ticker,
    _snapshot_reference,
    _ticks,
)


TICK_SIZE = "0.01000000"
LOT_SIZE = "0.00001000"


def test_reference_price_preserves_an_exact_tick_boundary() -> None:
    """80789.76 * 100 is slightly below 8078976 in binary floating point."""
    assert _ticks("80789.76000000", TICK_SIZE) == 8078976


@pytest.mark.parametrize(
    ("quantity", "expected_lots"),
    [
        ("0.00000000", 0),
        ("0.00001000", 1),
        ("0.00003000", 3),
        ("1.23456000", 123456),
    ],
)
def test_reference_quantity_preserves_exact_lot_boundaries(
    quantity: str, expected_lots: int
) -> None:
    """One exact lot must remain present; only zero quantity deletes a level."""
    assert _lots(quantity, LOT_SIZE) == expected_lots


@pytest.fixture
def snapshot() -> dict:
    return {
        "lastUpdateId": 100,
        "bids": [
            ["80789.76000000", "0.00001000"],
            ["80789.75000000", "0.00003000"],
        ],
        "asks": [
            ["80789.77000000", "1.23456000"],
            ["80789.78000000", "0.00007000"],
        ],
    }


def test_reference_snapshot_and_zero_quantity_deletion(snapshot: dict) -> None:
    bids, asks = _snapshot_reference(snapshot, TICK_SIZE, LOT_SIZE)
    assert bids == {8078976: 1, 8078975: 3}
    assert asks == {8078977: 123456, 8078978: 7}

    event = DepthEvent(
        seq=0,
        U=101,
        u=101,
        bids=[
            ["80789.76000000", "0.00000000"],
            ["80789.75000000", "0.00001000"],
        ],
        asks=[
            ["80789.77000000", "0.00000000"],
            ["80789.78000000", "1.23456000"],
        ],
    )
    _apply_depth_to_reference(event, bids, asks, TICK_SIZE, LOT_SIZE)
    assert bids == {8078975: 1}
    assert asks == {8078978: 123456}

    stats, errors = _replay_depth_with_reference(snapshot, [event], TICK_SIZE, LOT_SIZE)
    assert stats.applied_depth_events == 1
    assert stats.sequence_gap_events == 0
    assert stats.state_mismatches == 0
    assert stats.raised_errors == 0
    assert errors == []


def test_reference_book_ticker_retains_decimal_strings(snapshot: dict) -> None:
    event = BookTickerEvent(
        seq=0,
        u=101,
        bid_price="80789.75000000",
        bid_qty="0.00001000",
        ask_price="80789.78000000",
        ask_qty="0.00003000",
    )
    stats, errors = _replay_interleaved_with_book_ticker(
        snapshot, [event], TICK_SIZE, LOT_SIZE
    )
    assert stats["book_ticker_checked"] == 1
    assert stats["book_ticker_mismatches"] == 0
    assert errors == []

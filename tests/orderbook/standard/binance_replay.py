"""Binance orderbook replay types and helpers shared by offline and live tests."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from mm_toolbox.orderbook.standard import Orderbook, OrderbookLevel


@dataclass(slots=True)
class DepthEvent:
    """Buffered Binance spot diff-depth event."""

    seq: int
    U: int
    u: int
    bids: list[list[str]]
    asks: list[list[str]]


@dataclass(slots=True)
class BookTickerEvent:
    """Buffered Binance spot bookTicker event."""

    seq: int
    u: int
    bid_price: str
    bid_qty: str
    ask_price: str
    ask_qty: str


@dataclass(slots=True)
class ReplayStats:
    """Summary stats for depth replay validation."""

    applied_depth_events: int = 0
    skipped_old_events: int = 0
    skipped_prestart_events: int = 0
    sequence_gap_events: int = 0
    state_mismatches: int = 0
    raised_errors: int = 0


def _ticks(price: str, tick_size: str) -> int:
    """Convert original exchange decimal strings without binary rounding."""
    return int(Decimal(price) / Decimal(tick_size))


def _lots(size: str, lot_size: str) -> int:
    """Convert original exchange decimal strings without binary rounding."""
    return int(Decimal(size) / Decimal(lot_size))


def _snapshot_levels(
    snapshot: dict[str, Any],
) -> tuple[list[OrderbookLevel], list[OrderbookLevel]]:
    """Convert snapshot to OrderbookLevel lists.

    Args:
        snapshot: Binance depth snapshot.

    Returns:
        Tuple of (asks, bids) as OrderbookLevel lists.
    """
    bids = [
        OrderbookLevel(price=float(price), size=float(size), norders=0)
        for price, size in snapshot["bids"]
    ]
    asks = [
        OrderbookLevel(price=float(price), size=float(size), norders=0)
        for price, size in snapshot["asks"]
    ]
    return asks, bids


def _snapshot_reference(
    snapshot: dict[str, Any],
    tick_size: str,
    lot_size: str,
) -> tuple[dict[int, int], dict[int, int]]:
    """Build reference bid/ask dictionaries from snapshot.

    Args:
        snapshot: Binance depth snapshot.
        tick_size: Original exchange tick-size decimal string.
        lot_size: Original exchange lot-size decimal string.

    Returns:
        Tuple of (bids_dict, asks_dict) mapping ticks to lots.
    """
    bids: dict[int, int] = {}
    asks: dict[int, int] = {}

    for price, size in snapshot["bids"]:
        t = _ticks(price, tick_size)
        lots = _lots(size, lot_size)
        if lots > 0:
            bids[t] = lots

    for price, size in snapshot["asks"]:
        t = _ticks(price, tick_size)
        lots = _lots(size, lot_size)
        if lots > 0:
            asks[t] = lots

    return bids, asks


def _apply_depth_to_reference(
    event: DepthEvent,
    bids: dict[int, int],
    asks: dict[int, int],
    tick_size: str,
    lot_size: str,
) -> None:
    """Apply depth event to reference dictionaries.

    Args:
        event: DepthEvent to apply.
        bids: Reference bids dictionary (modified in-place).
        asks: Reference asks dictionary (modified in-place).
        tick_size: Original exchange tick-size decimal string.
        lot_size: Original exchange lot-size decimal string.
    """
    for price, size in event.bids:
        t = _ticks(price, tick_size)
        lots = _lots(size, lot_size)
        if lots == 0:
            bids.pop(t, None)
        else:
            bids[t] = lots

    for price, size in event.asks:
        t = _ticks(price, tick_size)
        lots = _lots(size, lot_size)
        if lots == 0:
            asks.pop(t, None)
        else:
            asks[t] = lots


def _apply_depth_to_orderbook(ob: Orderbook, event: DepthEvent) -> None:
    """Apply depth event to Orderbook instance.

    Args:
        ob: Orderbook to modify.
        event: DepthEvent to apply.
    """
    bids = [
        OrderbookLevel(price=float(price), size=float(size), norders=0)
        for price, size in event.bids
    ]
    asks = [
        OrderbookLevel(price=float(price), size=float(size), norders=0)
        for price, size in event.asks
    ]
    ob.consume_deltas(asks=asks, bids=bids)


def _replay_depth_with_reference(
    snapshot: dict[str, Any],
    depth_events: list[DepthEvent],
    tick_size: str,
    lot_size: str,
) -> tuple[ReplayStats, list[str]]:
    """Replay depth events and validate against reference state.

    Args:
        snapshot: Initial snapshot.
        depth_events: List of depth events to replay.
        tick_size: Tick size.
        lot_size: Lot size.

    Returns:
        Tuple of (stats, errors).
    """
    ob = Orderbook(tick_size=float(tick_size), lot_size=float(lot_size), size=1)
    asks_levels, bids_levels = _snapshot_levels(snapshot)
    ob.consume_snapshot(asks=asks_levels, bids=bids_levels)

    ref_bids, ref_asks = _snapshot_reference(snapshot, tick_size, lot_size)
    stats = ReplayStats()
    errors: list[str] = []

    def record_error(msg: str) -> None:
        """Record an error if under limit."""
        stats.state_mismatches += 1
        if len(errors) < 20:
            errors.append(msg)

    def validate_state(context: str) -> None:
        """Validate orderbook state against reference."""
        bid_keys = sorted(ref_bids)
        ask_keys = sorted(ref_asks)

        if ob._sorted_bid_ticks != bid_keys:
            record_error(f"{context}: sorted bid ticks mismatch")
        if ob._sorted_ask_ticks != ask_keys:
            record_error(f"{context}: sorted ask ticks mismatch")
        if set(ob._bids) != set(ref_bids):
            record_error(f"{context}: bid key set mismatch")
        if set(ob._asks) != set(ref_asks):
            record_error(f"{context}: ask key set mismatch")

        for tick, lots in ref_bids.items():
            level = ob._bids.get(tick)
            if level is None or level.lots != lots:
                record_error(f"{context}: bid lots mismatch at tick={tick}")
                break

        for tick, lots in ref_asks.items():
            level = ob._asks.get(tick)
            if level is None or level.lots != lots:
                record_error(f"{context}: ask lots mismatch at tick={tick}")
                break

        both_sides = bool(ref_bids) and bool(ref_asks)
        if ob.is_populated() != both_sides:
            record_error(f"{context}: is_populated mismatch")

        if both_sides:
            expected_bid = bid_keys[-1]
            expected_ask = ask_keys[0]
            try:
                best_bid, best_ask = ob.get_bbo()
            except Exception as exc:
                record_error(
                    f"{context}: get_bbo raised unexpectedly: {type(exc).__name__}"
                )
                return
            if best_bid.ticks != expected_bid or best_ask.ticks != expected_ask:
                record_error(f"{context}: bbo mismatch")
        else:
            try:
                ob.get_bbo()
                record_error(f"{context}: get_bbo should fail when side is empty")
            except ValueError:
                pass

    validate_state("snapshot")

    last_update_id = int(snapshot["lastUpdateId"])
    started = False

    for event in depth_events:
        if event.u <= last_update_id:
            stats.skipped_old_events += 1
            continue

        if not started:
            if event.U <= last_update_id + 1 <= event.u:
                started = True
            else:
                stats.skipped_prestart_events += 1
                continue
        else:
            if event.u < last_update_id + 1:
                stats.skipped_old_events += 1
                continue
            if event.U > last_update_id + 1:
                stats.sequence_gap_events += 1
                record_error(
                    f"sequence gap at seq={event.seq}; expected={last_update_id + 1} got=[{event.U},{event.u}]"
                )
                break

        try:
            _apply_depth_to_reference(event, ref_bids, ref_asks, tick_size, lot_size)
            _apply_depth_to_orderbook(ob, event)
            stats.applied_depth_events += 1
            last_update_id = event.u
            validate_state(f"depth seq={event.seq}")
        except Exception as exc:
            stats.raised_errors += 1
            record_error(
                f"depth apply failed at seq={event.seq}: {type(exc).__name__}: {exc}"
            )
            break

    return stats, errors


def _replay_interleaved_with_book_ticker(
    snapshot: dict[str, Any],
    ordered_events: list[DepthEvent | BookTickerEvent],
    tick_size: str,
    lot_size: str,
) -> tuple[dict[str, int], list[str]]:
    """Replay interleaved depth and bookTicker events.

    Args:
        snapshot: Initial snapshot.
        ordered_events: Interleaved depth and bookTicker events.
        tick_size: Tick size.
        lot_size: Lot size.

    Returns:
        Tuple of (stats, errors).
    """
    ob = Orderbook(tick_size=float(tick_size), lot_size=float(lot_size), size=1)
    asks_levels, bids_levels = _snapshot_levels(snapshot)
    ob.consume_snapshot(asks=asks_levels, bids=bids_levels)

    stats = {
        "depth_events": 0,
        "book_ticker_events": 0,
        "book_ticker_checked": 0,
        "book_ticker_stale_or_out_of_order": 0,
        "book_ticker_mismatches": 0,
    }
    errors: list[str] = []

    last_ticker_u = -1
    for event in ordered_events:
        if isinstance(event, DepthEvent):
            _apply_depth_to_orderbook(ob, event)
            stats["depth_events"] += 1
            continue

        stats["book_ticker_events"] += 1
        if event.u < last_ticker_u:
            stats["book_ticker_stale_or_out_of_order"] += 1
        last_ticker_u = max(last_ticker_u, event.u)

        ob.consume_bbo(
            ask=OrderbookLevel(
                price=float(event.ask_price), size=float(event.ask_qty), norders=0
            ),
            bid=OrderbookLevel(
                price=float(event.bid_price), size=float(event.bid_qty), norders=0
            ),
        )

        bid_lots = _lots(event.bid_qty, lot_size)
        ask_lots = _lots(event.ask_qty, lot_size)
        if bid_lots <= 0 or ask_lots <= 0:
            continue

        stats["book_ticker_checked"] += 1
        expected_bid_tick = _ticks(event.bid_price, tick_size)
        expected_ask_tick = _ticks(event.ask_price, tick_size)

        try:
            best_bid, best_ask = ob.get_bbo()
        except Exception as exc:
            stats["book_ticker_mismatches"] += 1
            if len(errors) < 20:
                errors.append(f"bookTicker get_bbo failed: {type(exc).__name__}: {exc}")
            continue

        if best_bid.ticks != expected_bid_tick or best_ask.ticks != expected_ask_tick:
            stats["book_ticker_mismatches"] += 1
            if len(errors) < 20:
                errors.append(
                    f"bookTicker mismatch seq={event.seq} ob=({best_bid.ticks},{best_ask.ticks}) evt=({expected_bid_tick},{expected_ask_tick})"
                )

        if ob._sorted_bid_ticks and ob._sorted_bid_ticks[-1] != expected_bid_tick:
            stats["book_ticker_mismatches"] += 1
            if len(errors) < 20:
                errors.append(
                    f"stale better bid remained top={ob._sorted_bid_ticks[-1]} expected={expected_bid_tick}"
                )

        if ob._sorted_ask_ticks and ob._sorted_ask_ticks[0] != expected_ask_tick:
            stats["book_ticker_mismatches"] += 1
            if len(errors) < 20:
                errors.append(
                    f"stale better ask remained top={ob._sorted_ask_ticks[0]} expected={expected_ask_tick}"
                )

    return stats, errors

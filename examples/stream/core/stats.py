"""Statistics tracking for multi-venue streaming.

Tracks per-venue latency distributions, message counts, and processing
timings while providing summary reporting at shutdown.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from .models import MsgType


@dataclass
class VenueStats:
    """Track statistics for a single venue.

    Attributes:
        venue: Venue name.
        latencies: Latency samples keyed by stream type.
        message_counts: Message counters keyed by message type.
        processing_times: Processing time samples in milliseconds.
    """

    venue: str
    latencies: dict[str, list[float]] = field(
        default_factory=lambda: {"bbo": [], "trade": [], "orderbook": []}
    )
    message_counts: dict[str, int] = field(
        default_factory=lambda: {
            "bbo": 0,
            "trade": 0,
            "orderbook": 0,
        }
    )
    processing_times: list[float] = field(default_factory=list)

    def record_latency(self, stream_type: MsgType | str, latency_ms: float) -> None:
        """Record a latency sample.

        Args:
            stream_type: Stream identifier or message type.
            latency_ms: Latency in milliseconds.
        """
        key = self._stream_key(stream_type)
        if key is None:
            return
        self.latencies.setdefault(key, []).append(latency_ms)

    def record_message(self, msg_type: MsgType | str) -> None:
        """Record a message count.

        Args:
            msg_type: Message type enum or string label.
        """
        key = self._message_key(msg_type)
        if key is None:
            return
        self.message_counts[key] = self.message_counts.get(key, 0) + 1

    def record_processing_time(self, time_ms: float) -> None:
        """Record a processing time sample.

        Args:
            time_ms: Processing time in milliseconds.
        """
        self.processing_times.append(time_ms)

    def get_latency_percentiles(self, stream_type: str) -> tuple[float, float, float]:
        """Return latency percentiles for the requested stream.

        Args:
            stream_type: Stream key (bbo, trade, orderbook).

        Returns:
            tuple[float, float, float]: (p50, p90, p99) in milliseconds.
        """
        values = self.latencies.get(stream_type, [])
        return self._percentiles(values, (0.5, 0.9, 0.99))

    def get_average_processing_time(self) -> float:
        """Compute the average processing time.

        Returns:
            float: Average processing time in milliseconds.
        """
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    def _stream_key(self, stream_type: MsgType | str) -> str | None:
        """Normalize a stream type to a latency key.

        Args:
            stream_type: Stream identifier.

        Returns:
            str | None: Normalized key.
        """
        if isinstance(stream_type, MsgType):
            if stream_type == MsgType.BBO:
                return "bbo"
            if stream_type == MsgType.TRADE:
                return "trade"
            if stream_type == MsgType.ORDERBOOK:
                return "orderbook"
            return None
        if isinstance(stream_type, str):
            if stream_type in {"bbo", "trade", "orderbook"}:
                return stream_type
            if "orderbook" in stream_type:
                return "orderbook"
        return None

    def _message_key(self, msg_type: MsgType | str) -> str | None:
        """Normalize a message type to a message counter key.

        Args:
            msg_type: Message identifier.

        Returns:
            str | None: Normalized key.
        """
        if isinstance(msg_type, MsgType):
            return {
                MsgType.BBO: "bbo",
                MsgType.TRADE: "trade",
                MsgType.ORDERBOOK: "orderbook",
            }.get(msg_type)
        if isinstance(msg_type, str):
            return msg_type
        return None

    def _percentiles(
        self, values: Iterable[float], percentiles: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Compute percentiles for a list of values.

        Args:
            values: Iterable of samples.
            percentiles: Percentiles to compute (0-1 range).

        Returns:
            tuple[float, float, float]: Computed percentiles.
        """
        values_list = list(values)
        if not values_list:
            return (0.0, 0.0, 0.0)
        values_list.sort()
        count = len(values_list)

        def pick(p: float) -> float:
            index = int(round((count - 1) * p))
            return values_list[min(max(index, 0), count - 1)]

        return tuple(pick(p) for p in percentiles)


class StatsCollector:
    """Aggregate statistics across venues."""

    def __init__(self) -> None:
        """Initialize the collector."""
        self.venue_stats: dict[str, VenueStats] = {}
        self._start_time_s = time.time()
        self._tick_candles_created = 0
        self._volume_candles_created = 0

    def get_or_create(self, venue: str) -> VenueStats:
        """Get or create stats for a venue.

        Args:
            venue: Venue name.

        Returns:
            VenueStats: Stats container for the venue.
        """
        if venue not in self.venue_stats:
            self.venue_stats[venue] = VenueStats(venue=venue)
        return self.venue_stats[venue]

    def set_candle_counts(self, tick_candles: int, volume_candles: int) -> None:
        """Set candle counts for summary reporting.

        Args:
            tick_candles: Total tick candles created.
            volume_candles: Total volume candles created.
        """
        self._tick_candles_created = tick_candles
        self._volume_candles_created = volume_candles

    def print_summary(self) -> None:
        """Print a summary of statistics to stdout."""
        runtime_s = max(time.time() - self._start_time_s, 0.0)
        total_messages = 0

        print("\n================== STREAM STATISTICS ==================")
        for venue in sorted(self.venue_stats.keys()):
            stats = self.venue_stats[venue]
            total_messages += sum(stats.message_counts.values())

            bbo_p50, bbo_p90, bbo_p99 = stats.get_latency_percentiles("bbo")
            trade_p50, trade_p90, trade_p99 = stats.get_latency_percentiles("trade")
            ob_p50, ob_p90, ob_p99 = stats.get_latency_percentiles("orderbook")

            print(f"\n[{venue.upper()}]")
            print(
                "  Latency (BBO):       "
                f"p50={bbo_p50:.1f}ms  p90={bbo_p90:.1f}ms  p99={bbo_p99:.1f}ms"
            )
            print(
                "  Latency (Trade):     "
                f"p50={trade_p50:.1f}ms  p90={trade_p90:.1f}ms  p99={trade_p99:.1f}ms"
            )
            print(
                "  Latency (Orderbook): "
                f"p50={ob_p50:.1f}ms  p90={ob_p90:.1f}ms  p99={ob_p99:.1f}ms"
            )
            print(
                "  Messages: "
                f"BBO={stats.message_counts.get('bbo', 0):,}  "
                f"Trade={stats.message_counts.get('trade', 0):,}  "
                f"Orderbook={stats.message_counts.get('orderbook', 0):,}"
            )
            print(
                "  Avg Processing Time: "
                f"{stats.get_average_processing_time():.2f}ms"
            )

        print("\n[TOTAL]")
        print(f"  Total Messages: {total_messages:,}")
        print(f"  Runtime: {runtime_s:.1f}s")
        print(f"  Tick Candles Created: {self._tick_candles_created:,}")
        print(f"  Volume Candles Created: {self._volume_candles_created:,}")
        print("=======================================================")

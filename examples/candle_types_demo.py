"""Example demonstrating multiple candle builders on a single Binance trade stream.

This example shows:
- Using WsPool for Binance futures @trade stream
- Converting trade messages to Trade structs
- Feeding the same Trade to TimeCandles, TickCandles, VolumeCandles,
  PriceCandles, and MultiCandles simultaneously
- Capturing completed candles via async iteration
- Periodic summary of candle completion counts and latest details
- Graceful shutdown on Ctrl+C
"""

from __future__ import annotations

import argparse
import asyncio
import math
import signal
import sys
from typing import Any

import msgspec

from mm_toolbox.candles import (
    Candle,
    MultiCandles,
    PriceCandles,
    TickCandles,
    TimeCandles,
    Trade,
    VolumeCandles,
)
from mm_toolbox.websocket import WsConnectionConfig, WsPool, WsPoolConfig


class BinanceTradeMsg(msgspec.Struct):
    """Binance futures trade stream message."""

    E: int
    T: int
    s: str
    p: str
    q: str
    m: bool


class CandleAggregator:
    """Holds all five candle builders and processes incoming trades.

    Attributes:
        time_candles: Time-based candle aggregator.
        tick_candles: Tick-count-based candle aggregator.
        volume_candles: Volume-based candle aggregator.
        price_candles: Price-movement-based candle aggregator.
        multi_candles: Multi-trigger candle aggregator.
    """

    def __init__(self) -> None:
        """Initialize all candle aggregators with demo parameters."""
        self.time_candles = TimeCandles(secs_per_bucket=5.0, num_candles=1000)
        self.tick_candles = TickCandles(ticks_per_bucket=100, num_candles=1000)
        self.volume_candles = VolumeCandles(volume_per_bucket=0.1, num_candles=1000)
        self.price_candles = PriceCandles(price_bucket=5.0, num_candles=1000)
        self.multi_candles = MultiCandles(
            max_duration_secs=10.0,
            max_ticks=50,
            max_size=0.05,
            num_candles=1000,
        )

        self._total_trades = 0
        self._total_completed: dict[str, int] = {
            "TimeCandles": 0,
            "TickCandles": 0,
            "VolumeCandles": 0,
            "PriceCandles": 0,
            "MultiCandles": 0,
        }
        self._latest_completed: dict[str, Candle | None] = {
            "TimeCandles": None,
            "TickCandles": None,
            "VolumeCandles": None,
            "PriceCandles": None,
            "MultiCandles": None,
        }
        self._consumer_tasks: list[asyncio.Task[None]] = []

    async def _consume_builder(self, name: str, builder: Any) -> None:
        """Async iterate over a candle builder to capture completed candles.

        Args:
            name: Human-readable name of the builder.
            builder: The candle builder instance to consume.
        """
        try:
            async for candle in builder:
                self._latest_completed[name] = candle
                self._total_completed[name] += 1
        except asyncio.CancelledError:
            pass

    def start_consumers(self) -> None:
        """Start async consumer tasks for all builders."""
        builders = [
            ("TimeCandles", self.time_candles),
            ("TickCandles", self.tick_candles),
            ("VolumeCandles", self.volume_candles),
            ("PriceCandles", self.price_candles),
            ("MultiCandles", self.multi_candles),
        ]
        for name, builder in builders:
            task = asyncio.create_task(self._consume_builder(name, builder))
            self._consumer_tasks.append(task)

    async def stop_consumers(self) -> None:
        """Cancel all async consumer tasks and wait for them to finish."""
        for task in self._consumer_tasks:
            task.cancel()
        if self._consumer_tasks:
            await asyncio.gather(*self._consumer_tasks, return_exceptions=True)

    def process_trade(self, trade: Trade) -> None:
        """Feed a single trade to all five candle aggregators.

        Args:
            trade: The trade to process.
        """
        self.time_candles.process_trade(trade)
        self.tick_candles.process_trade(trade)
        self.volume_candles.process_trade(trade)
        self.price_candles.process_trade(trade)
        self.multi_candles.process_trade(trade)
        self._total_trades += 1

    def print_summary(self) -> None:
        """Print a summary of all candle builders."""
        print(f"\n{'=' * 80}")
        print(f"Summary after {self._total_trades} trades")
        print(f"{'=' * 80}")

        builders = [
            ("TimeCandles", self.time_candles),
            ("TickCandles", self.tick_candles),
            ("VolumeCandles", self.volume_candles),
            ("PriceCandles", self.price_candles),
            ("MultiCandles", self.multi_candles),
        ]

        for name, candles in builders:
            completed = self._total_completed[name]
            latest = self._latest_completed[name]

            if latest is not None and latest.num_trades > 0:
                candle_str = (
                    f"O={latest.open_price:.2f} H={latest.high_price:.2f} "
                    f"L={latest.low_price:.2f} C={latest.close_price:.2f} "
                    f"VWAP={latest.vwap:.2f} Trades={latest.num_trades}"
                )
            else:
                candle_str = "No completed candles yet"

            current = candles.latest_candle
            current_str = (
                f"Current: trades={current.num_trades}"
                if current.num_trades > 0
                else "Current: empty"
            )

            print(
                f"{name:15s} | Completed: {completed:4d} | {candle_str} | {current_str}"
            )

        print(f"{'=' * 80}\n")


async def _consume_trades(
    pool: WsPool,
    aggregator: CandleAggregator,
    shutdown_event: asyncio.Event,
) -> None:
    """Consume trade messages from the WebSocket pool.

    Args:
        pool: The WsPool connected to Binance.
        aggregator: The candle aggregator to feed trades into.
        shutdown_event: Event to signal graceful shutdown.
    """
    decoder = msgspec.json.Decoder(type=BinanceTradeMsg)

    async for msg in pool:
        if shutdown_event.is_set():
            break

        try:
            trade_msg = decoder.decode(msg)
            price, size = float(trade_msg.p), float(trade_msg.q)
            if not (
                math.isfinite(price) and price > 0 and math.isfinite(size) and size > 0
            ):
                print(
                    f"Skipping non-execution trade: price={trade_msg.p}, quantity={trade_msg.q}",
                    file=sys.stderr,
                )
                continue
            trade = Trade(
                time_ms=trade_msg.T,
                is_buy=not trade_msg.m,
                price=price,
                size=size,
            )
            aggregator.process_trade(trade)
        except Exception as e:
            print(f"Error processing trade: {e}", file=sys.stderr)


async def _print_periodic_summary(
    aggregator: CandleAggregator,
    shutdown_event: asyncio.Event,
) -> None:
    """Print a summary every 30 seconds.

    Args:
        aggregator: The candle aggregator to summarize.
        shutdown_event: Event to signal graceful shutdown.
    """
    while not shutdown_event.is_set():
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            aggregator.print_summary()


async def _run(duration_s: float | None = None) -> None:
    """Set up the WsPool and run the event loop."""
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        print("\nReceived shutdown signal, stopping...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)
    duration_handle = (
        loop.call_later(duration_s, _signal_handler) if duration_s is not None else None
    )

    symbol = "btcusdt"

    print(f"Connecting to Binance futures @trade stream for {symbol.upper()}...")

    config = WsConnectionConfig.default(
        wss_url=f"wss://fstream.binance.com/ws/{symbol}@trade"
    )
    pool_config = WsPoolConfig(
        num_connections=2,
        evict_interval_s=60,
        hash_capacity=16_384,
    )
    pool = await WsPool.new(
        config=config,
        pool_config=pool_config,
    )

    aggregator = CandleAggregator()
    aggregator.start_consumers()
    trade_task = asyncio.create_task(_consume_trades(pool, aggregator, shutdown_event))
    summary_task = asyncio.create_task(
        _print_periodic_summary(aggregator, shutdown_event)
    )
    shutdown_task = asyncio.create_task(shutdown_event.wait())
    tasks = (trade_task, summary_task, shutdown_task)

    try:
        async with pool:
            print("Connected. Streaming trades and building candles...")
            print("Press Ctrl+C to stop.\n")
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task is not shutdown_task:
                    task.result()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await aggregator.stop_consumers()
        aggregator.print_summary()
        if duration_handle is not None:
            duration_handle.cancel()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        print("Shutdown complete.")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build candle types from public Binance futures trades"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds (default: run until interrupted)",
    )
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be > 0")
    try:
        asyncio.run(_run(args.duration))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

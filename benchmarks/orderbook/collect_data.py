"""Collects Binance BTCUSDT Futures orderbook data for benchmarking.

Usage:
    uv run python benchmarks/orderbook/collect_data.py [--output FILE] [--max-messages N]

Connects to Binance Futures WebSocket streams and REST API to collect:
- Initial depth snapshot (REST)
- BBO updates (@bookTicker stream)
- Depth deltas (@depth@100ms stream)

Output is JSON Lines format with type tags and nanosecond receive timestamps.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import TextIO

import aiohttp
import msgspec


class CollectorConfig(msgspec.Struct):
    """Configuration for data collection."""

    output_path: str
    max_messages: int
    symbol: str
    rest_depth_limit: int


class DataMessage(msgspec.Struct):
    """Message wrapper for JSON output."""

    type: str
    recv_ts_ns: int
    data: dict


class BinanceDataCollector:
    """Collects Binance orderbook data and writes to JSON Lines file."""

    def __init__(self, config: CollectorConfig) -> None:
        """Initialize the data collector.

        Args:
            config: Collection configuration.
        """
        self.config = config
        self.output_file: TextIO | None = None
        self.message_count = 0
        self.encoder = msgspec.json.Encoder()
        self._stop_event = asyncio.Event()

    def _write_message(self, msg_type: str, data: dict) -> None:
        """Write a timestamped message to the output file.

        Args:
            msg_type: Message type (snapshot, bbo, delta).
            data: Raw message data.
        """
        if self.output_file is None:
            return

        msg = DataMessage(
            type=msg_type,
            recv_ts_ns=time.perf_counter_ns(),
            data=data,
        )
        line = self.encoder.encode(msg).decode("utf-8")
        self.output_file.write(line + "\n")
        self.message_count += 1

        if self.message_count % 1000 == 0:
            print(f"Collected {self.message_count} messages...")

        if self.message_count >= self.config.max_messages:
            self._stop_event.set()

    async def fetch_snapshot(self, session: aiohttp.ClientSession) -> None:
        """Fetch initial orderbook snapshot via REST API.

        Args:
            session: aiohttp client session.
        """
        url = (
            f"https://fapi.binance.com/fapi/v1/depth"
            f"?symbol={self.config.symbol}&limit={self.config.rest_depth_limit}"
        )
        print(f"Fetching snapshot from {url}...")

        async with session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to fetch snapshot: {resp.status}")
            data = await resp.json()
            self._write_message("snapshot", data)
            print(
                f"Snapshot received: {len(data['bids'])} bids, {len(data['asks'])} asks"
            )

    async def _consume_bbo_stream(self, session: aiohttp.ClientSession) -> None:
        """Consume BBO WebSocket stream.

        Args:
            session: aiohttp client session.
        """
        symbol_lower = self.config.symbol.lower()
        url = f"wss://fstream.binance.com/ws/{symbol_lower}@bookTicker"

        try:
            async with session.ws_connect(url) as ws:
                print(f"Connected to BBO stream: {url}")
                async for msg in ws:
                    if self._stop_event.is_set():
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = msgspec.json.decode(msg.data, type=dict)
                        self._write_message("bbo", data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"BBO stream error: {ws.exception()}")
                        break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"BBO stream error: {e}")

    async def _consume_depth_stream(self, session: aiohttp.ClientSession) -> None:
        """Consume depth WebSocket stream.

        Args:
            session: aiohttp client session.
        """
        symbol_lower = self.config.symbol.lower()
        url = f"wss://fstream.binance.com/ws/{symbol_lower}@depth@100ms"

        try:
            async with session.ws_connect(url) as ws:
                print(f"Connected to depth stream: {url}")
                async for msg in ws:
                    if self._stop_event.is_set():
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = msgspec.json.decode(msg.data, type=dict)
                        self._write_message("delta", data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"Depth stream error: {ws.exception()}")
                        break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Depth stream error: {e}")

    async def run(self) -> None:
        """Run the data collection process."""
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Writing output to: {output_path}")
        print(f"Target messages: {self.config.max_messages}")

        with open(output_path, "w") as f:
            self.output_file = f

            async with aiohttp.ClientSession() as session:
                # Fetch initial snapshot
                await self.fetch_snapshot(session)

                print("Connecting to WebSocket streams...")

                # Run streams concurrently
                bbo_task = asyncio.create_task(self._consume_bbo_stream(session))
                depth_task = asyncio.create_task(self._consume_depth_stream(session))

                print("Connected. Collecting data...")

                # Wait for stop event or interruption
                try:
                    await self._stop_event.wait()
                except asyncio.CancelledError:
                    pass

                # Cancel stream tasks
                bbo_task.cancel()
                depth_task.cancel()

                # Wait for cancellation to complete
                await asyncio.gather(bbo_task, depth_task, return_exceptions=True)

            self.output_file = None

        print(f"\nCollection complete. Total messages: {self.message_count}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect Binance BTCUSDT Futures orderbook data"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="benchmarks/orderbook/data/btcusdt_orderbook.jsonl",
        help="Output file path (default: benchmarks/orderbook/data/btcusdt_orderbook.jsonl)",
    )
    parser.add_argument(
        "--max-messages",
        "-n",
        type=int,
        default=10000,
        help="Maximum number of messages to collect (default: 10000)",
    )
    parser.add_argument(
        "--symbol",
        "-s",
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--depth-limit",
        type=int,
        default=1000,
        help="REST snapshot depth limit (default: 1000)",
    )

    args = parser.parse_args()

    config = CollectorConfig(
        output_path=args.output,
        max_messages=args.max_messages,
        symbol=args.symbol,
        rest_depth_limit=args.depth_limit,
    )

    collector = BinanceDataCollector(config)

    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        print(f"\nInterrupted. Collected {collector.message_count} messages.")


if __name__ == "__main__":
    main()

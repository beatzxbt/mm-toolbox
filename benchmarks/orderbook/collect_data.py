"""Collects Binance BTCUSDT Futures orderbook data for benchmarking.

Usage:
    uv run python benchmarks/orderbook/collect_data.py [--output FILE] [--max-messages N]

Connects to Binance Futures WebSocket streams and REST API to collect:
- Initial depth snapshot (REST)
- BBO updates (@bookTicker stream)
- Depth deltas (@depth@100ms stream)

Output is JSON Lines with type tags and nanosecond receive timestamps. The
snapshot is written first for replay, followed by the first bridging depth
event. Future BBOs wait for covering depth and publish in increasing update-ID
order; receive timestamps are preserved and can precede the snapshot response.
An incomplete capture is not published if collection fails or is interrupted.
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import TextIO

import aiohttp
import msgspec

_STREAM_START_TIMEOUT_S = 10.0
_FIRST_DEPTH_TIMEOUT_S = 10.0
_SNAPSHOT_TIMEOUT_S = 10.0
_BRIDGE_TIMEOUT_S = 10.0
_MAX_BUFFERED_EVENTS = 4096
_MAX_BUFFERED_BYTES = 16 * 1024 * 1024
_MAX_PENDING_BBO_EVENTS = 4096
_MAX_PENDING_BBO_BYTES = 16 * 1024 * 1024
_PENDING_DEPTH_TIMEOUT_S = 10.0


class CollectorConfig(msgspec.Struct):
    """Configuration for data collection."""

    output_path: str
    max_messages: int
    symbol: str
    rest_depth_limit: int

    def __post_init__(self) -> None:
        if self.max_messages < 1:
            raise ValueError("max_messages must be positive")


class DataMessage(msgspec.Struct):
    """Message wrapper for JSON output."""

    type: str
    recv_ts_ns: int
    data: dict


class _BufferedEvents:
    """Apply backpressure while bounding queued message count and payload bytes."""

    def __init__(
        self,
        max_events: int = _MAX_BUFFERED_EVENTS,
        max_bytes: int = _MAX_BUFFERED_BYTES,
    ) -> None:
        if max_events <= 0 or max_bytes <= 0:
            raise ValueError("Collector buffer limits must be positive")
        self._queue: asyncio.Queue[tuple[str, str, int, int]] = asyncio.Queue(
            maxsize=max_events
        )
        self._max_bytes = max_bytes
        self._bytes = 0
        self._space = asyncio.Event()

    async def put(self, message_type: str, raw: str, recv_ts_ns: int) -> None:
        size = len(raw.encode("utf-8"))
        if size > self._max_bytes:
            raise ValueError("WebSocket event exceeds the collector buffer limit")
        while self._queue.full() or size > self._max_bytes - self._bytes:
            self._space.clear()
            await self._space.wait()
        self._bytes += size
        self._queue.put_nowait((message_type, raw, recv_ts_ns, size))

    async def get(self) -> tuple[str, str, int]:
        message_type, raw, recv_ts_ns, size = await self._queue.get()
        self._bytes -= size
        self._space.set()
        return message_type, raw, recv_ts_ns


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

    def _write_message(
        self, msg_type: str, data: dict, recv_ts_ns: int | None = None
    ) -> None:
        """Write a timestamped message to the output file.

        Args:
            msg_type: Message type (snapshot, bbo, delta).
            data: Raw message data.
        """
        if self.output_file is None:
            return

        msg = DataMessage(
            type=msg_type,
            recv_ts_ns=time.perf_counter_ns() if recv_ts_ns is None else recv_ts_ns,
            data=data,
        )
        line = self.encoder.encode(msg).decode("utf-8")
        self.output_file.write(line + "\n")
        self.message_count += 1

        if self.message_count % 1000 == 0:
            print(f"Collected {self.message_count} messages...")

        if self.message_count >= self.config.max_messages:
            self._stop_event.set()

    async def fetch_snapshot(self, session: aiohttp.ClientSession) -> dict:
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
            print(
                f"Snapshot received: {len(data['bids'])} bids, {len(data['asks'])} asks"
            )
            return data

    async def _consume_stream(
        self,
        session: aiohttp.ClientSession,
        *,
        stream_name: str,
        suffix: str,
        message_type: str,
        ready: asyncio.Event,
        start: asyncio.Event,
        first_depth: asyncio.Event,
        events: _BufferedEvents,
    ) -> None:
        """Consume one feed after both connections are ready, propagating failures."""
        symbol_lower = self.config.symbol.lower()
        url = f"wss://fstream.binance.com/public/ws/{symbol_lower}@{suffix}"
        try:
            async with session.ws_connect(url) as ws:
                print(f"Connected to {stream_name} stream: {url}")
                ready.set()
                await start.wait()
                async for msg in ws:
                    # An already pending read can report failure in the same
                    # turn that the publication owner reaches its target.
                    # Cancellation ends healthy feeds; observed errors cannot
                    # be hidden by that target being reached.
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError(
                            "WebSocket reported an error"
                        ) from ws.exception()
                    if self._stop_event.is_set():
                        return
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        recv_ts_ns = time.perf_counter_ns()
                        if message_type == "delta":
                            first_depth.set()
                        await events.put(message_type, msg.data, recv_ts_ns)
                raise RuntimeError("WebSocket closed before collection completed")
        except Exception as exc:
            raise RuntimeError(f"{stream_name} stream failed: {exc}") from exc

    async def run(self) -> None:
        """Publish the output only after the requested capture completes."""
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Writing output to: {output_path}")
        print(f"Target messages: {self.config.max_messages}")

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".partial",
                delete=False,
            ) as f:
                temporary_path = Path(f.name)
                self.output_file = f
                try:
                    async with aiohttp.ClientSession() as session:
                        if self.config.max_messages == 1:
                            snapshot = await asyncio.wait_for(
                                self.fetch_snapshot(session), _SNAPSHOT_TIMEOUT_S
                            )
                            snapshot_id = snapshot.get("lastUpdateId")
                            if type(snapshot_id) is not int or snapshot_id < 0:
                                raise ValueError("Snapshot has an invalid lastUpdateId")
                            self._write_message("snapshot", snapshot)
                        else:
                            await self._collect_streams(session)
                finally:
                    self.output_file = None
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        print(f"\nCollection complete. Total messages: {self.message_count}")

    async def _collect_streams(self, session: aiohttp.ClientSession) -> None:
        """Start both feeds, buffer depth, then fetch and bridge the snapshot."""
        print("Connecting to WebSocket streams...")
        bbo_ready, depth_ready, start = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )
        first_depth = asyncio.Event()
        events = _BufferedEvents()
        bbo_task = asyncio.create_task(
            self._consume_stream(
                session,
                stream_name="BBO",
                suffix="bookTicker",
                message_type="bbo",
                ready=bbo_ready,
                start=start,
                first_depth=first_depth,
                events=events,
            )
        )
        depth_task = asyncio.create_task(
            self._consume_stream(
                session,
                stream_name="Depth",
                suffix="depth@100ms",
                message_type="delta",
                ready=depth_ready,
                start=start,
                first_depth=first_depth,
                events=events,
            )
        )

        async def wait_until_ready() -> None:
            await asyncio.gather(bbo_ready.wait(), depth_ready.wait())

        streams = (bbo_task, depth_task)
        tasks: list[asyncio.Task] = list(streams)

        async def wait_for_phase(task: asyncio.Task, timeout: float | None, name: str):
            done, _ = await asyncio.wait(
                (*streams, task), timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                raise TimeoutError(f"Timed out waiting for {name}")
            for stream in streams:
                if stream in done:
                    stream.result()
                    if not self._stop_event.is_set():
                        raise RuntimeError(f"A stream ended while waiting for {name}")
            if task not in done:
                raise RuntimeError(f"Collection ended while waiting for {name}")
            return task.result()

        try:
            ready_task = asyncio.create_task(wait_until_ready())
            tasks.append(ready_task)
            await wait_for_phase(
                ready_task, _STREAM_START_TIMEOUT_S, "both WebSocket feeds to start"
            )
            start.set()
            first_depth_task = asyncio.create_task(first_depth.wait())
            tasks.append(first_depth_task)
            await wait_for_phase(
                first_depth_task, _FIRST_DEPTH_TIMEOUT_S, "the first depth event"
            )

            snapshot_task = asyncio.create_task(self.fetch_snapshot(session))
            tasks.append(snapshot_task)
            snapshot = await wait_for_phase(
                snapshot_task, _SNAPSHOT_TIMEOUT_S, "the depth snapshot"
            )
            snapshot_id = snapshot.get("lastUpdateId")
            if type(snapshot_id) is not int or snapshot_id < 0:
                raise ValueError("Snapshot has an invalid lastUpdateId")
            self._write_message("snapshot", snapshot)
            print("Connected. Bridging the snapshot and collecting data...")

            processor = asyncio.create_task(self._process_events(events, snapshot_id))
            tasks.append(processor)
            await wait_for_phase(processor, None, "collection to complete")
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # A stream can fail between the completion check and cancellation.
        for result in results[: len(streams)]:
            if isinstance(result, Exception):
                raise result

    async def _process_events(self, events: _BufferedEvents, snapshot_id: int) -> None:
        """Sequence publication without allowing BBOs to hide required depth.

        The receive FIFO supplies bounded startup staging and backpressure. This
        single owner controls insertion into the capture: future BBOs wait for
        a covering depth event, then publish in ID order before that depth.
        """
        depth_id = published_id = snapshot_id
        bridged = False
        loop = asyncio.get_running_loop()
        bridge_deadline = loop.time() + _BRIDGE_TIMEOUT_S
        pending_deadline: float | None = None
        pending: dict[int, tuple[dict, int, int]] = {}
        pending_ids: list[int] = []
        pending_bytes = 0

        while not self._stop_event.is_set():
            deadline = pending_deadline if bridged else bridge_deadline
            phase = (
                "depth progress for pending BBOs"
                if bridged
                else "a depth snapshot bridge"
            )
            if deadline is None:
                message_type, raw, recv_ts_ns = await events.get()
            else:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for {phase}")
                try:
                    message_type, raw, recv_ts_ns = await asyncio.wait_for(
                        events.get(), remaining
                    )
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(f"Timed out waiting for {phase}") from exc

            data = msgspec.json.decode(raw, type=dict)
            if message_type == "bbo":
                update_id = data.get("u")
                if type(update_id) is not int or update_id < 0:
                    raise ValueError("BBO event has an invalid update ID")
                if update_id <= published_id or update_id in pending:
                    continue
                size = len(raw.encode("utf-8"))
                if (
                    len(pending) >= _MAX_PENDING_BBO_EVENTS
                    or size > _MAX_PENDING_BBO_BYTES - pending_bytes
                ):
                    raise RuntimeError(
                        "Pending BBO buffer limit exceeded before depth progress"
                    )
                pending[update_id] = (data, recv_ts_ns, size)
                heapq.heappush(pending_ids, update_id)
                pending_bytes += size
                if pending_deadline is None:
                    pending_deadline = loop.time() + _PENDING_DEPTH_TIMEOUT_S
                continue

            first, final, previous = data.get("U"), data.get("u"), data.get("pu")
            if (
                type(first) is not int
                or type(final) is not int
                or type(previous) is not int
                or first < 0
                or final < first
                or previous < 0
                or final < previous
            ):
                raise ValueError("Depth event has invalid sequence IDs")
            if not bridged:
                if final < snapshot_id:
                    continue
                if first > snapshot_id:
                    raise RuntimeError("Depth stream cannot bridge the snapshot")
            else:
                # Repeated or late depth packets cannot mutate the book again
                # and do not extend the pending-depth deadline.
                if final <= depth_id:
                    continue
                if previous != depth_id:
                    raise RuntimeError(
                        f"Depth sequence gap: expected pu={depth_id}, got {previous}"
                    )

            while pending_ids and pending_ids[0] <= final:
                update_id = heapq.heappop(pending_ids)
                bbo, received, size = pending.pop(update_id)
                pending_bytes -= size
                # The initial bridge is always second. Depth also wins all
                # equal-ID ties; stale BBOs never enter the final stream.
                if not bridged or update_id <= published_id or update_id == final:
                    continue
                self._write_message("bbo", bbo, received)
                published_id = update_id
                if self._stop_event.is_set():
                    return

            self._write_message("delta", data, recv_ts_ns)
            depth_id = published_id = final
            bridged = True
            pending_deadline = (
                loop.time() + _PENDING_DEPTH_TIMEOUT_S if pending else None
            )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect Binance BTCUSDT Futures orderbook data"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="benchmarks/orderbook/data/btcusdt_capture.jsonl",
        help=(
            "Output file path "
            "(default: benchmarks/orderbook/data/btcusdt_capture.jsonl)"
        ),
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
        print(
            f"\nInterrupted. Discarded {collector.message_count} partial messages.",
            file=sys.stderr,
        )
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()

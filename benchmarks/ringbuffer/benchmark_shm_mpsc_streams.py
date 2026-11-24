"""Replay independent Binance symbol traffic through SHM MPSC producers.

This benchmark captures public Binance USD-M book-ticker traffic, freezes it,
then compares direct paced MPSC writers with process-per-connection WsSingle
writers. A second, seeded schedule redistributes each symbol's captured messages
independently over time at a higher fixed aggregate rate.

Run with:
    uv run python benchmarks/ringbuffer/benchmark_shm_mpsc_streams.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import queue
import random
import resource
import shutil
import statistics
import struct
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

from mm_toolbox.ringbuffer.shm import ShmMpscConsumer, ShmMpscProducer


_CAPACITY_BYTES = 64 * 1024 * 1024
_PRODUCER_COUNTS = (1, 4, 8, 16, 64, 256)
_SAMPLE_EVERY = 16
_SENTINEL = (1 << 64) - 1
_MESSAGE_HEADER = struct.Struct("<QIIQQQQ")
_SCHEDULE_HEADER = struct.Struct("<QQIII")


@dataclass(frozen=True, slots=True)
class Event:
    """One captured market payload and its frozen replay position."""

    offset_ns: int
    sequence: int
    symbol_id: int
    symbol_sequence: int
    payload: bytes


def _resource_usage() -> dict[str, float | int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if sys.platform != "darwin":
        rss *= 1024
    return {
        "cpu_seconds": float(usage.ru_utime + usage.ru_stime),
        "peak_rss_bytes": rss,
    }


def _percentiles(values: list[int] | tuple[int, ...]) -> dict[str, float | int] | None:
    if not values:
        return None
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(array.size),
        "p50_ns": float(np.percentile(array, 50)),
        "p95_ns": float(np.percentile(array, 95)),
        "p99_ns": float(np.percentile(array, 99)),
        "max_ns": int(array.max()),
    }


async def _get_active_usdt_symbols(limit: int) -> list[str]:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get("https://fapi.binance.com/fapi/v1/exchangeInfo") as resp:
            resp.raise_for_status()
            info = await resp.json()
        async with session.get("https://fapi.binance.com/fapi/v1/ticker/24hr") as resp:
            resp.raise_for_status()
            tickers = await resp.json()

    eligible = {
        item["symbol"]
        for item in info["symbols"]
        if item.get("status") == "TRADING"
        and item.get("contractType") == "PERPETUAL"
        and item.get("quoteAsset") == "USDT"
    }
    ranked = sorted(
        (
            (float(item.get("quoteVolume", 0.0)), item["symbol"])
            for item in tickers
            if item.get("symbol") in eligible
        ),
        reverse=True,
    )
    symbols = [symbol for _, symbol in ranked[:limit]]
    if len(symbols) != limit:
        raise RuntimeError(
            f"Found {len(symbols)} active USDT perpetual symbols; expected {limit}"
        )
    return symbols


async def _capture_bookticker_trace(
    symbols: list[str], capture_seconds: float
) -> tuple[list[Event], int, dict[str, Any]]:
    """Retry a transient public WebSocket close before giving up the capture."""
    from websockets.exceptions import ConnectionClosed

    retryable = (ConnectionClosed, OSError, TimeoutError)
    for attempt in range(1, 4):
        try:
            events, duration_ns, metadata = await _capture_bookticker_trace_once(
                symbols, capture_seconds
            )
            minimum_duration_ns = int(max(0.1, capture_seconds * 0.5) * 1e9)
            if duration_ns < minimum_duration_ns:
                raise TimeoutError(
                    f"capture lasted {duration_ns / 1e9:.2f}s; "
                    f"need at least {minimum_duration_ns / 1e9:.2f}s"
                )
            metadata["capture_attempts"] = attempt
            return events, duration_ns, metadata
        except retryable as exc:
            if attempt == 3:
                raise RuntimeError(
                    f"Binance capture failed after {attempt} attempts: {exc}"
                ) from exc
            print(
                f"Binance capture attempt {attempt}/3 failed ({exc}); retrying",
                flush=True,
            )
            await asyncio.sleep(0.5 * attempt)
    raise AssertionError("unreachable capture retry loop")


async def _capture_bookticker_trace_once(
    symbols: list[str], capture_seconds: float
) -> tuple[list[Event], int, dict[str, Any]]:
    """Capture one real multi-symbol socket and retain receive-time offsets."""
    import websockets
    from websockets.exceptions import ConnectionClosed

    stream_names = [f"{symbol.lower()}@bookTicker" for symbol in symbols]
    symbol_ids = {symbol: i for i, symbol in enumerate(symbols)}
    symbol_sequences = [0] * len(symbols)
    events: list[Event] = []
    capture_start_ns = 0
    deadline_ns = 0
    capture_close_error: str | None = None

    async with websockets.connect(
        "wss://fstream.binance.com/public/ws",
        open_timeout=20,
        close_timeout=5,
        max_size=1_048_576,
        ping_interval=None,
        compression=None,
    ) as websocket:
        for request_id, start in enumerate(range(0, len(stream_names), 128), start=1):
            await websocket.send(
                json.dumps(
                    {
                        "method": "SUBSCRIBE",
                        "params": stream_names[start : start + 128],
                        "id": request_id,
                    },
                    separators=(",", ":"),
                )
            )
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                decoded_response = json.loads(response)
                if (
                    decoded_response.get("id") == request_id
                    and "result" in decoded_response
                ):
                    if decoded_response["result"] is not None:
                        raise RuntimeError(
                            f"Binance rejected the stream subscription: {decoded_response}"
                        )
                    break
                if "code" in decoded_response and "msg" in decoded_response:
                    raise RuntimeError(
                        "Binance rejected the stream subscription: "
                        f"{decoded_response['code']} {decoded_response['msg']}"
                    )
        capture_start_ns = time.perf_counter_ns()
        deadline_ns = capture_start_ns + int(capture_seconds * 1e9)
        while True:
            remaining_ns = deadline_ns - time.perf_counter_ns()
            if remaining_ns <= 0:
                break
            timeout_s = remaining_ns / 1_000_000_000
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout_s)
            except TimeoutError:
                break
            except ConnectionClosed as exc:
                capture_close_error = f"{type(exc).__name__}: {exc}"
                break
            received_ns = time.perf_counter_ns()
            if isinstance(raw, str):
                payload = raw.encode("utf-8")
                decoded = json.loads(raw)
            else:
                payload = bytes(raw)
                decoded = json.loads(payload)
            data = decoded.get("data", decoded)
            symbol = data.get("s")
            symbol_id = symbol_ids.get(symbol)
            if symbol_id is None:
                continue
            if received_ns >= deadline_ns:
                break
            events.append(
                Event(
                    offset_ns=received_ns - capture_start_ns,
                    sequence=len(events),
                    symbol_id=symbol_id,
                    symbol_sequence=symbol_sequences[symbol_id],
                    payload=payload,
                )
            )
            symbol_sequences[symbol_id] += 1
    if not events:
        raise RuntimeError("Binance capture returned no book-ticker messages")
    capture_end_ns = min(deadline_ns, time.perf_counter_ns())
    duration_ns = max(1, capture_end_ns - capture_start_ns)
    counts = Counter(event.symbol_id for event in events)
    metadata = {
        "source": "Binance USD-M public multi-symbol @bookTicker socket",
        "symbols": symbols,
        "capture_duration_ns": duration_ns,
        "capture_truncated": capture_close_error is not None,
        "capture_close_error": capture_close_error,
        "message_count": len(events),
        "messages_per_second": len(events) / (duration_ns / 1e9),
        "symbols_with_messages": len(counts),
        "symbol_message_counts": {
            str(i): counts.get(i, 0) for i in range(len(symbols))
        },
        "median_payload_bytes": int(statistics.median(len(e.payload) for e in events)),
        "max_payload_bytes": max(len(event.payload) for event in events),
    }
    return events, duration_ns, metadata


def _make_independent_schedule(
    captured: list[Event], duration_ns: int, multiplier: float, seed: int
) -> tuple[list[Event], int]:
    """Condition independent per-symbol Poisson arrivals on captured counts."""
    if multiplier <= 1.0:
        raise ValueError("independent schedule multiplier must exceed one")
    target_duration_ns = max(1, int(duration_ns / multiplier))
    grouped: dict[int, list[Event]] = defaultdict(list)
    for event in captured:
        grouped[event.symbol_id].append(event)

    rng = random.Random(seed)
    generated: list[Event] = []
    for symbol_id, symbol_events in grouped.items():
        offsets = sorted(rng.randrange(target_duration_ns) for _ in symbol_events)
        for event, offset_ns in zip(symbol_events, offsets, strict=True):
            generated.append(replace(event, offset_ns=offset_ns))
    generated.sort(key=lambda event: (event.offset_ns, event.sequence))
    return generated, target_duration_ns


def _balance_symbols(
    events: list[Event], symbol_count: int, producers: int
) -> list[int]:
    """Assign each symbol to a producer, balancing captured message counts."""
    counts = Counter(event.symbol_id for event in events)
    loads = [0] * producers
    assignment = [-1] * symbol_count
    for symbol_id in sorted(range(symbol_count), key=lambda sid: (-counts[sid], sid)):
        producer_id = min(range(producers), key=lambda pid: (loads[pid], pid))
        assignment[symbol_id] = producer_id
        loads[producer_id] += counts[symbol_id]
    return assignment


def _write_schedules(
    events: list[Event], symbol_count: int, producers: int, directory: Path
) -> tuple[list[str], list[int], list[int]]:
    assignment = _balance_symbols(events, symbol_count, producers)
    groups: list[list[Event]] = [[] for _ in range(producers)]
    for event in events:
        groups[assignment[event.symbol_id]].append(event)

    paths: list[str] = []
    counts: list[int] = []
    for producer_id, group in enumerate(groups):
        group.sort(key=lambda event: (event.offset_ns, event.sequence))
        path = directory / f"producer_{producer_id:03d}.schedule"
        with path.open("wb") as output:
            for event in group:
                output.write(
                    _SCHEDULE_HEADER.pack(
                        event.offset_ns,
                        event.sequence,
                        event.symbol_id,
                        event.symbol_sequence,
                        len(event.payload),
                    )
                )
                output.write(event.payload)
        paths.append(str(path))
        counts.append(len(group))
    return paths, counts, assignment


def _read_schedule(path: str) -> Iterator[tuple[int, int, int, int, bytes]]:
    with open(path, "rb") as schedule:
        while True:
            header = schedule.read(_SCHEDULE_HEADER.size)
            if not header:
                return
            if len(header) != _SCHEDULE_HEADER.size:
                raise RuntimeError(f"Truncated schedule header in {path}")
            offset_ns, sequence, symbol_id, symbol_sequence, payload_size = (
                _SCHEDULE_HEADER.unpack(header)
            )
            payload = schedule.read(payload_size)
            if len(payload) != payload_size:
                raise RuntimeError(f"Truncated payload in {path}")
            yield offset_ns, sequence, symbol_id, symbol_sequence, payload


def _load_schedule(path: str) -> list[tuple[int, int, int, int, bytes]]:
    """Move schedule file I/O outside each producer's timed replay loop."""
    return list(_read_schedule(path))


def _message_bytes(
    sequence: int,
    symbol_id: int,
    symbol_sequence: int,
    intended_ns: int,
    source_send_ns: int,
    callback_ns: int,
    pre_insert_ns: int,
    payload: bytes,
) -> bytes:
    header = _MESSAGE_HEADER.pack(
        sequence,
        symbol_id,
        symbol_sequence,
        intended_ns,
        source_send_ns,
        callback_ns,
        pre_insert_ns,
    )
    return header + payload


def _send_sentinel(producer: ShmMpscProducer) -> tuple[int, int, int]:
    start_ns = time.perf_counter_ns()
    succeeded = producer.insert(_MESSAGE_HEADER.pack(_SENTINEL, 0, 0, 0, 0, 0, 0))
    elapsed_ns = time.perf_counter_ns() - start_ns
    return (
        int(succeeded),
        elapsed_ns,
        int(producer.last_insert_dropped if succeeded else 0),
    )


def _consumer_worker(
    path: str,
    expected_messages: int,
    producer_count: int,
    sample_every: int,
    ready_queue: Any,
    result_queue: Any,
) -> None:
    consumer = ShmMpscConsumer(path, spin_wait=1024)
    ready_queue.put(("ready", "consumer", -1))
    seen = bytearray(expected_messages)
    last_symbol_sequence: dict[int, int] = {}
    consumed_raw = 0
    unique = 0
    duplicates = 0
    invalid = 0
    reordered_per_symbol = 0
    sentinel_count = 0
    preinsert_to_consume: list[int] = []
    source_to_consume: list[int] = []
    callback_to_consume: list[int] = []
    deadline_to_consume: list[int] = []
    first_consume_ns = 0
    last_consume_ns = 0
    try:
        while sentinel_count < producer_count:
            message = consumer.consume()
            consumed_ns = time.perf_counter_ns()
            if not first_consume_ns:
                first_consume_ns = consumed_ns
            last_consume_ns = consumed_ns
            if len(message) < _MESSAGE_HEADER.size:
                invalid += 1
                continue
            (
                sequence,
                symbol_id,
                symbol_sequence,
                intended_ns,
                source_send_ns,
                callback_ns,
                pre_insert_ns,
            ) = _MESSAGE_HEADER.unpack_from(message)
            if sequence == _SENTINEL:
                sentinel_count += 1
                ready_queue.put(("progress", consumed_raw, sentinel_count))
                continue
            consumed_raw += 1
            if consumed_raw % 10_000 == 0:
                ready_queue.put(("progress", consumed_raw, sentinel_count))
            if not 0 <= sequence < expected_messages:
                invalid += 1
                continue
            if seen[sequence]:
                duplicates += 1
            else:
                seen[sequence] = 1
                unique += 1
            previous = last_symbol_sequence.get(symbol_id, -1)
            if symbol_sequence <= previous:
                reordered_per_symbol += 1
            last_symbol_sequence[symbol_id] = symbol_sequence
            if sequence % sample_every == 0:
                if intended_ns:
                    deadline_to_consume.append(consumed_ns - intended_ns)
                if source_send_ns:
                    source_to_consume.append(consumed_ns - source_send_ns)
                if callback_ns:
                    callback_to_consume.append(consumed_ns - callback_ns)
                if pre_insert_ns:
                    preinsert_to_consume.append(consumed_ns - pre_insert_ns)

        remaining = len(consumer)
    finally:
        consumer.close()
    result_queue.put(
        (
            "result",
            "consumer",
            {
                "consumed_raw_messages": consumed_raw,
                "unique_messages": unique,
                "missing_unique_messages": expected_messages - unique,
                "duplicates": duplicates,
                "invalid_messages": invalid,
                "per_symbol_reorders": reordered_per_symbol,
                "sentinels": sentinel_count,
                "remaining_messages": remaining,
                "first_consume_ns": first_consume_ns,
                "last_consume_ns": last_consume_ns,
                "deadline_to_consume": deadline_to_consume,
                "source_send_to_consume": source_to_consume,
                "callback_to_consume": callback_to_consume,
                "preinsert_to_consume": preinsert_to_consume,
                **_resource_usage(),
            },
        )
    )


def _wait_until_ns(deadline_ns: int) -> None:
    while True:
        remaining_ns = deadline_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > 1_000_000:
            time.sleep((remaining_ns - 500_000) / 1_000_000_000)
        else:
            while time.perf_counter_ns() < deadline_ns:
                pass
            return


async def _run_ws_producer(
    producer: ShmMpscProducer,
    producer_id: int,
    schedule_path: str,
    url: str,
    ready_queue: Any,
    sample_every: int,
    timeout_s: float,
) -> dict[str, Any]:
    from mm_toolbox.websocket.connection import WsConnectionConfig
    from mm_toolbox.websocket.single import WsSingle

    callback_state: dict[str, Any] = {
        "events_received": 0,
        "sentinel_received": False,
        "insert_successes": 0,
        "insert_failures": 0,
        "overwritten": 0,
        "insert_call_samples": [],
        "first_preinsert_ns": 0,
        "last_preinsert_ns": 0,
        "callback_error": None,
    }
    done = asyncio.Event()
    all_events_received = asyncio.Event()
    expected_events = sum(1 for _ in _read_schedule(schedule_path))

    def on_message(message: bytes) -> None:
        try:
            if len(message) < _MESSAGE_HEADER.size:
                callback_state["callback_error"] = "short WebSocket payload"
                done.set()
                return
            (
                sequence,
                symbol_id,
                symbol_sequence,
                intended_ns,
                source_send_ns,
                _old_callback_ns,
                _old_preinsert_ns,
            ) = _MESSAGE_HEADER.unpack_from(message)
            if sequence == _SENTINEL:
                callback_state["sentinel_received"] = True
                done.set()
                return
            callback_ns = time.perf_counter_ns()
            preinsert_ns = time.perf_counter_ns()
            outgoing = _message_bytes(
                sequence,
                symbol_id,
                symbol_sequence,
                intended_ns,
                source_send_ns,
                callback_ns,
                preinsert_ns,
                message[_MESSAGE_HEADER.size :],
            )
            call_start_ns = time.perf_counter_ns()
            succeeded = producer.insert(outgoing)
            call_end_ns = time.perf_counter_ns()
            callback_state["events_received"] += 1
            if callback_state["events_received"] >= expected_events:
                all_events_received.set()
            callback_state["first_preinsert_ns"] = (
                callback_state["first_preinsert_ns"] or preinsert_ns
            )
            callback_state["last_preinsert_ns"] = preinsert_ns
            if sequence % sample_every == 0:
                callback_state["insert_call_samples"].append(
                    call_end_ns - call_start_ns
                )
            if succeeded:
                callback_state["insert_successes"] += 1
                callback_state["overwritten"] += producer.last_insert_dropped
            else:
                callback_state["insert_failures"] += 1
        except BaseException:
            callback_state["callback_error"] = traceback.format_exc()
            done.set()

    config = WsConnectionConfig.default(
        url,
        conn_id=producer_id,
        auto_reconnect=False,
        latency_ping_interval_ms=60_000,
        receive_buffer_capacity=4096,
        receive_buffer_max_bytes=16 * 1024 * 1024,
    )
    client = WsSingle(config, on_message=on_message)
    start_task = asyncio.create_task(client.start())
    try:
        connect_deadline = time.monotonic() + timeout_s
        while client.get_connection_count() != 1:
            if start_task.done():
                await start_task
                raise RuntimeError(f"WsSingle {producer_id} stopped before connecting")
            if time.monotonic() >= connect_deadline:
                raise TimeoutError(f"WsSingle {producer_id} connection timed out")
            await asyncio.sleep(0.001)
        ready_queue.put(("ready", "producer", producer_id))
        done_task = asyncio.create_task(done.wait())
        all_events_task = asyncio.create_task(all_events_received.wait())
        try:
            completed, pending = await asyncio.wait(
                (done_task, all_events_task),
                timeout=timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if not completed:
                raise TimeoutError(
                    f"WsSingle {producer_id} received "
                    f"{callback_state['events_received']} of {expected_events} events"
                )
            if all_events_task in completed and not done_task.done():
                try:
                    await asyncio.wait_for(done.wait(), timeout=0.5)
                except TimeoutError:
                    pass
        finally:
            for task in (done_task, all_events_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(done_task, all_events_task, return_exceptions=True)
        if callback_state["callback_error"]:
            raise RuntimeError(callback_state["callback_error"])
    finally:
        client.close()
        await asyncio.gather(start_task, return_exceptions=True)
    return {
        **callback_state,
        "native_received_frames": client.get_seq_id(),
        "receive_buffer_drops": client.get_dropped_messages(),
    }


def _producer_worker(
    path: str,
    producer_count: int,
    producer_id: int,
    schedule_path: str,
    mode: str,
    url: str | None,
    start_value: Any,
    ready_queue: Any,
    result_queue: Any,
    sample_every: int,
    timeout_s: float,
) -> None:
    producer: ShmMpscProducer | None = None
    stats: dict[str, Any] = {
        "producer_id": producer_id,
        "events_scheduled": 0,
        "events_received": 0,
        "insert_successes": 0,
        "insert_failures": 0,
        "overwritten": 0,
        "sentinel_success": 0,
        "sentinel_overwritten": 0,
        "sentinel_insert_latency_ns": 0,
        "insert_call_samples": [],
        "source_lateness_samples": [],
        "first_schedule_send_ns": 0,
        "last_schedule_send_ns": 0,
        "receive_buffer_drops": 0,
        "native_received_frames": 0,
        "error": None,
    }
    try:
        producer = ShmMpscProducer(
            path,
            _CAPACITY_BYTES,
            create=False,
            unlink_on_close=False,
            spin_wait=1024,
        )
        if mode == "websocket":
            if url is None:
                raise ValueError("WebSocket producer is missing its local source URL")
            ws_stats = asyncio.run(
                _run_ws_producer(
                    producer,
                    producer_id,
                    schedule_path,
                    url,
                    ready_queue,
                    sample_every,
                    timeout_s,
                )
            )
            for key in (
                "events_received",
                "sentinel_received",
                "insert_successes",
                "insert_failures",
                "overwritten",
                "insert_call_samples",
                "receive_buffer_drops",
                "native_received_frames",
            ):
                stats[key] = ws_stats[key]
            stats["events_scheduled"] = sum(1 for _ in _read_schedule(schedule_path))
        elif mode == "direct":
            schedule = _load_schedule(schedule_path)
            stats["events_scheduled"] = len(schedule)
            ready_queue.put(("ready", "producer", producer_id))
            while True:
                with start_value.get_lock():
                    start_ns = int(start_value.value)
                if start_ns:
                    break
                time.sleep(0.001)
            for offset_ns, sequence, symbol_id, symbol_sequence, payload in schedule:
                deadline_ns = start_ns + offset_ns
                _wait_until_ns(deadline_ns)
                actual_ns = time.perf_counter_ns()
                stats["events_received"] += 1
                stats["first_schedule_send_ns"] = (
                    stats["first_schedule_send_ns"] or actual_ns
                )
                stats["last_schedule_send_ns"] = actual_ns
                preinsert_ns = time.perf_counter_ns()
                outgoing = _message_bytes(
                    sequence,
                    symbol_id,
                    symbol_sequence,
                    deadline_ns,
                    0,
                    0,
                    preinsert_ns,
                    payload,
                )
                call_start_ns = time.perf_counter_ns()
                succeeded = producer.insert(outgoing)
                call_end_ns = time.perf_counter_ns()
                if sequence % sample_every == 0:
                    stats["insert_call_samples"].append(call_end_ns - call_start_ns)
                    stats["source_lateness_samples"].append(actual_ns - deadline_ns)
                if succeeded:
                    stats["insert_successes"] += 1
                    stats["overwritten"] += producer.last_insert_dropped
                else:
                    stats["insert_failures"] += 1
        else:
            raise ValueError(f"Unknown producer mode: {mode}")
    except BaseException:
        stats["error"] = traceback.format_exc()
        try:
            ready_queue.put(("error", "producer", producer_id, stats["error"]))
        except Exception:
            pass
    finally:
        if producer is not None:
            try:
                succeeded, sentinel_latency, sentinel_drops = _send_sentinel(producer)
                stats["sentinel_success"] = succeeded
                stats["sentinel_overwritten"] = sentinel_drops
                stats["sentinel_insert_latency_ns"] = sentinel_latency
            except BaseException:
                stats["error"] = stats["error"] or traceback.format_exc()
            stats["failed_inserts_counter"] = producer.failed_inserts
            producer.close()
        stats.update(_resource_usage())
        result_queue.put(("result", "producer", stats))


def _websocket_path_id(websocket: Any) -> int:
    request = getattr(websocket, "request", None)
    path = getattr(request, "path", None) or getattr(websocket, "path", "")
    return int(path.rsplit("/", 1)[-1])


async def _wait_until_async(deadline_ns: int) -> None:
    while True:
        remaining_ns = deadline_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > 1_000_000:
            await asyncio.sleep((remaining_ns - 500_000) / 1_000_000_000)
        else:
            await asyncio.sleep(0)


async def _send_schedule(
    websocket: Any,
    schedule_path: str,
    start_ns: int,
    producer_id: int,
) -> dict[str, Any]:
    count = 0
    first_send_ns = 0
    last_send_ns = 0
    lateness_samples: list[int] = []
    send_call_samples: list[int] = []
    schedule = _load_schedule(schedule_path)
    for offset_ns, sequence, symbol_id, symbol_sequence, payload in schedule:
        deadline_ns = start_ns + offset_ns
        await _wait_until_async(deadline_ns)
        send_start_ns = time.perf_counter_ns()
        frame = bytearray(_MESSAGE_HEADER.size + len(payload))
        _MESSAGE_HEADER.pack_into(
            frame,
            0,
            sequence,
            symbol_id,
            symbol_sequence,
            deadline_ns,
            send_start_ns,
            0,
            0,
        )
        frame[_MESSAGE_HEADER.size :] = payload
        send_call_start_ns = time.perf_counter_ns()
        await websocket.send(bytes(frame))
        send_call_end_ns = time.perf_counter_ns()
        count += 1
        first_send_ns = first_send_ns or send_start_ns
        last_send_ns = send_start_ns
        if sequence % _SAMPLE_EVERY == 0:
            lateness_samples.append(send_start_ns - deadline_ns)
            send_call_samples.append(send_call_end_ns - send_call_start_ns)

    eos = _MESSAGE_HEADER.pack(_SENTINEL, 0, 0, 0, 0, 0, 0)
    await websocket.send(eos)
    await websocket.close()
    return {
        "producer_id": producer_id,
        "sent": count,
        "first_send_ns": first_send_ns,
        "last_send_ns": last_send_ns,
        "source_lateness_samples": lateness_samples,
        "send_call_samples": send_call_samples,
    }


async def _serve_local_source(connection: Any, host: str) -> None:
    import websockets

    connections: dict[int, Any] = {}
    changed = asyncio.Event()

    async def handle(websocket: Any) -> None:
        producer_id = _websocket_path_id(websocket)
        connections[producer_id] = websocket
        changed.set()
        try:
            await websocket.wait_closed()
        finally:
            connections.pop(producer_id, None)
            changed.set()

    server = await websockets.serve(
        handle,
        host,
        0,
        compression=None,
        ping_interval=None,
        max_size=1_048_576,
    )
    sockets = server.sockets
    if not sockets:
        raise RuntimeError("Local replay source did not bind a socket")
    port = int(sockets[0].getsockname()[1])
    connection.send(("ready", port))
    try:
        command = await asyncio.to_thread(connection.recv)
        if command[0] == "stop":
            return
        if command[0] != "run":
            raise ValueError(f"Unexpected local source command: {command[0]}")
        _, schedule_paths, start_ns, timeout_s = command
        expected = len(schedule_paths)
        deadline = time.monotonic() + timeout_s
        while len(connections) < expected:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Local replay accepted {len(connections)} of {expected} clients"
                )
            changed.clear()
            if len(connections) >= expected:
                break
            await asyncio.wait_for(changed.wait(), timeout=remaining)
        tasks = [
            asyncio.create_task(
                _send_schedule(
                    connections[producer_id], schedule_path, start_ns, producer_id
                )
            )
            for producer_id, schedule_path in enumerate(schedule_paths)
        ]
        results = await asyncio.gather(*tasks)
        sent_count = sum(result["sent"] for result in results)
        lateness = [
            sample for result in results for sample in result["source_lateness_samples"]
        ]
        send_calls = [
            sample for result in results for sample in result["send_call_samples"]
        ]
        first_send = min(
            (result["first_send_ns"] for result in results if result["first_send_ns"]),
            default=0,
        )
        last_send = max((result["last_send_ns"] for result in results), default=0)
        connection.send(
            (
                "done",
                {
                    "sent": sent_count,
                    "first_send_ns": first_send,
                    "last_send_ns": last_send,
                    "source_lateness_samples": lateness,
                    "send_call_samples": send_calls,
                    **_resource_usage(),
                },
            )
        )
    finally:
        server.close()
        await server.wait_closed()
        connection.close()


def _source_process(connection: Any, host: str) -> None:
    try:
        asyncio.run(_serve_local_source(connection, host))
    except BaseException as exc:
        try:
            connection.send(
                ("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise


def _receive_pipe(connection: Any, timeout_s: float) -> tuple[Any, ...]:
    if not connection.poll(timeout_s):
        raise TimeoutError("Timed out waiting for local replay source")
    response = connection.recv()
    if response[0] == "error":
        raise RuntimeError(response[1])
    return response


def _run_case(
    context: Any,
    root: Path,
    workload_name: str,
    events: list[Event],
    duration_ns: int,
    symbol_count: int,
    producer_count: int,
    arm: str,
    repetition: int,
    timeout_s: float,
) -> dict[str, Any]:
    case_dir = root / f"{workload_name}_{arm}_p{producer_count}_r{repetition}"
    case_dir.mkdir()
    schedule_paths, producer_event_counts, _ = _write_schedules(
        events, symbol_count, producer_count, case_dir
    )
    ring_path = str(case_dir / "mpsc.shm")

    bootstrap = ShmMpscProducer(
        ring_path,
        _CAPACITY_BYTES,
        num_rings=producer_count,
        create=True,
        unlink_on_close=False,
    )
    bootstrap.close()

    ready_queue = context.Queue()
    result_queue = context.Queue()
    start_value = context.Value("Q", 0)
    consumer = context.Process(
        target=_consumer_worker,
        args=(
            ring_path,
            len(events),
            producer_count,
            _SAMPLE_EVERY,
            ready_queue,
            result_queue,
        ),
        name=f"mpsc-consumer-{producer_count}",
    )
    consumer.start()

    source_process = None
    source_parent = None
    source_port = 0
    producer_processes = []
    try:
        if arm == "websocket":
            source_parent, source_child = context.Pipe()
            source_process = context.Process(
                target=_source_process,
                args=(source_child, "127.0.0.1"),
                name=f"mpsc-replay-source-{producer_count}",
            )
            source_process.start()
            source_child.close()
            source_response = _receive_pipe(source_parent, timeout_s)
            if source_response[0] != "ready":
                raise RuntimeError(
                    f"Unexpected local source startup response: {source_response}"
                )
            source_port = int(source_response[1])

        for producer_id, schedule_path in enumerate(schedule_paths):
            url = (
                f"ws://127.0.0.1:{source_port}/producer/{producer_id}"
                if arm == "websocket"
                else None
            )
            process = context.Process(
                target=_producer_worker,
                args=(
                    ring_path,
                    producer_count,
                    producer_id,
                    schedule_path,
                    arm,
                    url,
                    start_value,
                    ready_queue,
                    result_queue,
                    _SAMPLE_EVERY,
                    timeout_s,
                ),
                name=f"mpsc-{arm}-producer-{producer_id}",
            )
            process.start()
            producer_processes.append(process)

        ready_count = 0
        expected_ready = producer_count + 1
        ready_deadline = time.monotonic() + timeout_s
        while ready_count < expected_ready:
            remaining = ready_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Only {ready_count} of {expected_ready} workers became ready"
                )
            status = ready_queue.get(timeout=remaining)
            if status[0] == "error":
                raise RuntimeError(status[3])
            if status[0] == "ready":
                ready_count += 1

        run_start_ns = time.perf_counter_ns() + 500_000_000
        with start_value.get_lock():
            start_value.value = run_start_ns
        if arm == "websocket":
            assert source_parent is not None
            source_parent.send(("run", schedule_paths, run_start_ns, timeout_s))

        source_stats: dict[str, Any] | None = None
        if arm == "websocket":
            assert source_parent is not None
            response = _receive_pipe(source_parent, duration_ns / 1e9 + timeout_s)
            if response[0] != "done":
                raise RuntimeError(f"Unexpected local source result: {response}")
            source_stats = response[1]

        producer_results: list[dict[str, Any]] = []
        consumer_result = None
        results_deadline = time.monotonic() + duration_ns / 1e9 + timeout_s + 30.0
        while consumer_result is None or len(producer_results) < producer_count:
            remaining = results_deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise queue.Empty
                result = result_queue.get(timeout=remaining)
            except queue.Empty:
                consumer_progress = []
                while True:
                    try:
                        status = ready_queue.get_nowait()
                    except queue.Empty:
                        break
                    if status[0] == "progress":
                        consumer_progress.append(
                            {"consumed_raw_messages": status[1], "sentinels": status[2]}
                        )
                for process in producer_processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(2.0)
                if consumer.is_alive():
                    consumer.terminate()
                    consumer.join(2.0)
                raise TimeoutError(
                    "Timed out draining MPSC worker results; "
                    f"consumer progress={consumer_progress}; "
                    f"producer results={len(producer_results)}/{producer_count}"
                )
            if result[0] != "result":
                raise RuntimeError(f"Unexpected worker result: {result}")
            if result[1] == "consumer":
                consumer_result = result[2]
            else:
                producer_results.append(result[2])
        if consumer_result is None or len(producer_results) != producer_count:
            raise RuntimeError("Incomplete producer/consumer result set")
        for process in producer_processes:
            process.join(5.0)
            if process.is_alive():
                process.terminate()
                process.join(2.0)
                raise TimeoutError(f"Producer process {process.name} did not stop")
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Producer process {process.name} exited with {process.exitcode}"
                )
        consumer.join(5.0)
        if consumer.is_alive():
            consumer.terminate()
            consumer.join(2.0)
            raise TimeoutError("MPSC consumer did not stop after reporting its result")
        if consumer.exitcode != 0:
            raise RuntimeError(f"MPSC consumer exited with {consumer.exitcode}")

        insert_samples = [
            value
            for result in producer_results
            for value in result["insert_call_samples"]
        ]
        direct_lateness = [
            value
            for result in producer_results
            for value in result["source_lateness_samples"]
        ]
        event_successes = sum(result["insert_successes"] for result in producer_results)
        event_failures = sum(result["insert_failures"] for result in producer_results)
        event_overwrites = sum(result["overwritten"] for result in producer_results)
        sentinel_successes = sum(
            result["sentinel_success"] for result in producer_results
        )
        sentinel_overwrites = sum(
            result["sentinel_overwritten"] for result in producer_results
        )
        overwritten = event_overwrites + sentinel_overwrites
        producer_attempts = sum(
            result["insert_successes"] + result["insert_failures"]
            for result in producer_results
        )
        callback_events = sum(result["events_received"] for result in producer_results)
        source_sent = source_stats["sent"] if source_stats else len(events)
        native_received = sum(
            result["native_received_frames"] for result in producer_results
        )
        receive_drops = sum(
            result["receive_buffer_drops"] for result in producer_results
        )
        consumer_accounting = (
            consumer_result["consumed_raw_messages"]
            + consumer_result["sentinels"]
            + overwritten
            + consumer_result["remaining_messages"]
        )
        accounting_ok = (
            event_successes + sentinel_successes == consumer_accounting
            and consumer_result["sentinels"] == producer_count
            and consumer_result["unique_messages"]
            + consumer_result["missing_unique_messages"]
            == len(events)
        )
        websocket_fidelity = arm == "direct" or (
            source_sent == len(events)
            and callback_events == len(events)
            and native_received == len(events) + producer_count
            and receive_drops == 0
        )
        no_loss = (
            accounting_ok
            and producer_attempts == len(events)
            and websocket_fidelity
            and event_failures == 0
            and overwritten == 0
            and consumer_result["missing_unique_messages"] == 0
            and consumer_result["duplicates"] == 0
            and consumer_result["invalid_messages"] == 0
            and consumer_result["per_symbol_reorders"] == 0
        )
        cpu_seconds = sum(result["cpu_seconds"] for result in producer_results)
        peak_rss_bytes = sum(result["peak_rss_bytes"] for result in producer_results)
        cpu_seconds += consumer_result["cpu_seconds"]
        peak_rss_bytes += consumer_result["peak_rss_bytes"]
        if source_stats:
            cpu_seconds += source_stats["cpu_seconds"]
            peak_rss_bytes += source_stats["peak_rss_bytes"]
        expected_rate = len(events) / (duration_ns / 1e9)
        actual_duration_ns = max(
            1,
            consumer_result["last_consume_ns"] - run_start_ns,
        )
        if source_stats and source_stats["first_send_ns"]:
            source_elapsed_ns = max(
                1, source_stats["last_send_ns"] - source_stats["first_send_ns"]
            )
            actual_source_rate = source_stats["sent"] / (source_elapsed_ns / 1e9)
        else:
            direct_first = min(
                (
                    result["first_schedule_send_ns"]
                    for result in producer_results
                    if result["first_schedule_send_ns"]
                ),
                default=0,
            )
            direct_last = max(
                (result["last_schedule_send_ns"] for result in producer_results),
                default=0,
            )
            source_elapsed_ns = max(1, direct_last - direct_first)
            actual_source_rate = len(events) / (source_elapsed_ns / 1e9)
        result = {
            "workload": workload_name,
            "arm": arm,
            "producer_count": producer_count,
            "repetition": repetition,
            "symbols": symbol_count,
            "symbols_per_producer_mean": symbol_count / producer_count,
            "scheduled_messages": len(events),
            "duration_ns": duration_ns,
            "expected_aggregate_rate_per_second": expected_rate,
            "actual_consumer_rate_per_second": consumer_result["unique_messages"]
            / (actual_duration_ns / 1e9),
            "actual_source_or_producer_rate_per_second": actual_source_rate,
            "message_payload_bytes_median": int(
                statistics.median(len(event.payload) for event in events)
            ),
            "queue_capacity_bytes": _CAPACITY_BYTES,
            "ring_count": producer_count,
            "ring_capacity_bytes": _CAPACITY_BYTES // producer_count,
            "accounting_ok": accounting_ok,
            "no_loss": no_loss,
            "producer_event_attempts": producer_attempts,
            "event_insert_successes": event_successes,
            "event_insert_failures": event_failures,
            "queue_overwrites": overwritten,
            "sentinel_overwrites": sentinel_overwrites,
            "sentinel_insert_successes": sentinel_successes,
            "consumer": {
                key: value
                for key, value in consumer_result.items()
                if not key.endswith("_ns")
                and key
                not in {
                    "deadline_to_consume",
                    "source_send_to_consume",
                    "callback_to_consume",
                    "preinsert_to_consume",
                }
            },
            "insert_call_latency": _percentiles(insert_samples),
            "preinsert_to_consume_latency": _percentiles(
                consumer_result["preinsert_to_consume"]
            ),
            "deadline_to_consume_latency": _percentiles(
                consumer_result["deadline_to_consume"]
            ),
            "source_send_to_consume_latency": _percentiles(
                consumer_result["source_send_to_consume"]
            ),
            "callback_to_consume_latency": _percentiles(
                consumer_result["callback_to_consume"]
            ),
            "source_send_lateness": _percentiles(
                source_stats["source_lateness_samples"] if source_stats else []
            ),
            "producer_schedule_lateness": _percentiles(
                direct_lateness if arm == "direct" else []
            ),
            "websocket_receive_drops": receive_drops,
            "websocket_native_received_frames": native_received,
            "websocket_callback_events": callback_events
            if arm == "websocket"
            else None,
            "producer_event_counts": producer_event_counts,
            "producer_insert_success_counts": [
                result["insert_successes"] for result in producer_results
            ],
            "producer_insert_call_latency": [
                _percentiles(result["insert_call_samples"])
                for result in producer_results
            ],
            "process_cpu_seconds_sum": cpu_seconds,
            "process_peak_rss_bytes_sum": peak_rss_bytes,
            "source": {
                key: value
                for key, value in source_stats.items()
                if key not in {"source_lateness_samples", "send_call_samples"}
            }
            if source_stats
            else None,
            "producer_errors": [
                result["error"] for result in producer_results if result["error"]
            ],
        }
        return result
    finally:
        for process in producer_processes:
            if process.is_alive():
                process.terminate()
                process.join(2.0)
            process.close()
        if consumer.is_alive():
            consumer.terminate()
            consumer.join(2.0)
        consumer.close()
        if source_process is not None:
            if source_process.is_alive():
                try:
                    if source_parent is not None:
                        source_parent.send(("stop",))
                except (BrokenPipeError, EOFError, OSError):
                    pass
                source_process.join(2.0)
            if source_process.is_alive():
                source_process.terminate()
                source_process.join(2.0)
            source_process.close()
        if source_parent is not None:
            source_parent.close()
        for channel in (ready_queue, result_queue):
            channel.close()
            channel.join_thread()
        try:
            Path(ring_path).unlink(missing_ok=True)
        except OSError:
            pass
        shutil.rmtree(case_dir, ignore_errors=True)


def _print_case(result: dict[str, Any]) -> None:
    insert = result["insert_call_latency"] or {}
    preinsert = result["preinsert_to_consume_latency"] or {}
    deadline = result["deadline_to_consume_latency"] or {}
    lateness = (
        result["source_send_lateness"]
        if result["arm"] == "websocket"
        else result["producer_schedule_lateness"]
    ) or {}
    print(
        f"{result['workload']:>18} {result['arm']:>9} "
        f"P={result['producer_count']:>3} r={result['repetition']} "
        f"rate={result['actual_consumer_rate_per_second']:>9.0f}/s "
        f"insert p50/p95/p99={insert.get('p50_ns', 0):.0f}/"
        f"{insert.get('p95_ns', 0):.0f}/{insert.get('p99_ns', 0):.0f}ns "
        f"pre-insert→consume p50/p99={preinsert.get('p50_ns', 0) / 1e6:.3f}/"
        f"{preinsert.get('p99_ns', 0) / 1e6:.3f}ms "
        f"deadline→consume p99={deadline.get('p99_ns', 0) / 1e6:.3f}ms "
        f"schedule/send-late p99={lateness.get('p99_ns', 0) / 1e6:.3f}ms "
        f"loss={result['websocket_receive_drops'] + result['queue_overwrites'] + result['event_insert_failures']} "
        f"accounting={'ok' if result['accounting_ok'] else 'FAIL'} "
        f"no-loss={'yes' if result['no_loss'] else 'no'}",
        flush=True,
    )


def _parse_counts(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value not in _PRODUCER_COUNTS for value in values):
        raise ValueError(f"Producer counts must be selected from {_PRODUCER_COUNTS}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=256)
    parser.add_argument("--capture-seconds", type=float, default=10.0)
    parser.add_argument("--high-rate-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--producer-counts", default=",".join(map(str, _PRODUCER_COUNTS))
    )
    parser.add_argument("--workloads", default="capture-native,independent-2x")
    parser.add_argument("--arms", default="direct,websocket")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument(
        "--output",
        default="/tmp/mm_mpsc_ws_benchmark_results.json",
        help="JSON result path (default: /tmp/mm_mpsc_ws_benchmark_results.json)",
    )
    args = parser.parse_args()
    if args.symbols != 256:
        parser.error("This experiment fixes the logical market universe at 256 symbols")
    if args.capture_seconds <= 0 or args.repeats < 1 or args.startup_timeout <= 0:
        parser.error("capture duration, repeats, and startup timeout must be positive")
    producer_counts = _parse_counts(args.producer_counts)
    selected_workloads = tuple(
        value.strip() for value in args.workloads.split(",") if value.strip()
    )
    selected_arms = tuple(
        value.strip() for value in args.arms.split(",") if value.strip()
    )
    allowed_workloads = {
        "capture-native",
        f"independent-{args.high_rate_multiplier:g}x",
    }
    if not selected_workloads or set(selected_workloads) - allowed_workloads:
        parser.error(f"Workloads must be selected from {sorted(allowed_workloads)}")
    if not selected_arms or set(selected_arms) - {"direct", "websocket"}:
        parser.error("Arms must be selected from ['direct', 'websocket']")

    symbols = asyncio.run(_get_active_usdt_symbols(args.symbols))
    captured_events, native_duration_ns, capture_metadata = asyncio.run(
        _capture_bookticker_trace(symbols, args.capture_seconds)
    )
    independent_events, independent_duration_ns = _make_independent_schedule(
        captured_events,
        native_duration_ns,
        args.high_rate_multiplier,
        seed=20261124,
    )
    workloads = [
        ("capture-native", captured_events, native_duration_ns),
        (
            f"independent-{args.high_rate_multiplier:g}x",
            independent_events,
            independent_duration_ns,
        ),
    ]
    workloads = [item for item in workloads if item[0] in selected_workloads]

    run_cases = [
        (workload_name, events, duration_ns, producer_count, arm, repetition)
        for workload_name, events, duration_ns in workloads
        for arm in selected_arms
        for producer_count in producer_counts
        for repetition in range(1, args.repeats + 1)
    ]
    random.Random(20261124).shuffle(run_cases)

    context = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    output_path = Path(args.output)
    report = {
        "capture": capture_metadata,
        "configuration": {
            "producer_counts": producer_counts,
            "repeats": args.repeats,
            "capacity_bytes": _CAPACITY_BYTES,
            "sample_every": _SAMPLE_EVERY,
            "higher_rate_multiplier": args.high_rate_multiplier,
            "arms": selected_arms,
            "workloads": [name for name, _, _ in workloads],
            "notes": [
                "Captured native arrival timestamps are replayed locally.",
                "The independent schedule gives each symbol independent uniform Poisson-conditioned arrival times while preserving its captured message count and payload sequence.",
                "preinsert-to-consume is a pre-insert timestamp to consumer receipt measurement, not exact queue residence time.",
                "Each WsSingle producer runs in its own process; the P=256 result includes process and event-loop scheduling costs.",
            ],
        },
        "results": results,
    }

    def save_report() -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save_report()
    with tempfile.TemporaryDirectory(prefix="mm-mpsc-ws-") as temp_dir:
        root = Path(temp_dir)
        for index, (
            workload_name,
            events,
            duration_ns,
            producer_count,
            arm,
            repetition,
        ) in enumerate(run_cases, start=1):
            print(
                f"[{index}/{len(run_cases)}] {workload_name} {arm} "
                f"P={producer_count} repeat={repetition}",
                flush=True,
            )
            try:
                result = _run_case(
                    context,
                    root,
                    workload_name,
                    events,
                    duration_ns,
                    len(symbols),
                    producer_count,
                    arm,
                    repetition,
                    args.startup_timeout,
                )
                _print_case(result)
            except Exception as exc:
                result = {
                    "workload": workload_name,
                    "arm": arm,
                    "producer_count": producer_count,
                    "repetition": repetition,
                    "symbols": len(symbols),
                    "accounting_ok": False,
                    "no_loss": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"CASE FAILED: {result['error']}", flush=True)
            results.append(result)
            save_report()

    save_report()
    print(f"\nSaved JSON report: {output_path}")

    failures = [
        result
        for result in results
        if not result.get("accounting_ok")
        or result.get("producer_errors")
        or result.get("error")
    ]
    if failures:
        raise SystemExit(
            f"{len(failures)} benchmark cases had accounting or worker errors"
        )


if __name__ == "__main__":
    main()

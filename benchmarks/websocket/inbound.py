"""Independent-process inbound WebSocket delivery scenarios and accounting."""

from __future__ import annotations

import asyncio
import math
import multiprocessing
import random
import struct
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import websockets

from mm_toolbox.websocket.connection import WsConnectionConfig
from mm_toolbox.websocket.pool import WsPool, WsPoolConfig
from mm_toolbox.websocket.single import WsSingle


_INBOUND_HEADER = struct.Struct(">QQ")
_MAX_LATENCY_SAMPLES = 4_096


@dataclass(slots=True)
class InboundBenchmarkConfig:
    """Bounded configuration for local inbound WebSocket scenarios."""

    server_host: str = "127.0.0.1"
    pool_connections: int = 3
    message_size: int = 256
    num_messages: int = 512
    warmup_messages: int = 32
    paced_interval_ms: float = 1.0
    idle_interval_ms: float = 20.0
    timeout_s: float = 3.0
    require_no_loss: bool = True

    def __post_init__(self) -> None:
        if self.pool_connections <= 1:
            raise ValueError("pool_connections must be greater than one")
        if self.message_size < _INBOUND_HEADER.size:
            raise ValueError(
                f"message_size must be at least {_INBOUND_HEADER.size} bytes"
            )
        if not 129 <= self.num_messages <= 1_000_000:
            raise ValueError("num_messages must be between 129 and 1,000,000")
        if not 0 <= self.warmup_messages <= 1_000:
            raise ValueError("warmup_messages must be between 0 and 1,000")
        if (
            not math.isfinite(self.paced_interval_ms)
            or not math.isfinite(self.idle_interval_ms)
            or self.paced_interval_ms < 0
            or self.idle_interval_ms < 0
        ):
            raise ValueError("intervals cannot be negative")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")


@dataclass(slots=True)
class InboundScenarioResult:
    """Exact source, receive-buffer, deduplication, and app-delivery counts."""

    receiver: str
    scenario: str
    logical_payloads: int
    message_size: int
    connection_count: int
    source_sent_per_connection: tuple[int, ...]
    native_received_per_connection: tuple[int, ...]
    receive_buffer_drops: int
    peak_buffered_frames: int
    peak_buffered_bytes: int
    read_paused_samples: int
    raw_frames_consumed: int
    deduplicated_frames: int
    app_received_frames: int
    unique_payloads_received: int
    missing_unique_payloads: int
    missing_id_samples: tuple[int, ...]
    reordered_unique_payloads: int
    duplicate_app_payloads: int
    malformed_payloads: int
    delivery_elapsed_ns: int
    delivery_completed: bool
    latency_samples_ns: tuple[int, ...]
    latency_observations: int
    receive_timed_out: bool

    @property
    def wire_frames_sent(self) -> int:
        return sum(self.source_sent_per_connection)

    @property
    def native_frames_received(self) -> int:
        return sum(self.native_received_per_connection)

    @property
    def wire_frames_not_received(self) -> int:
        return max(0, self.wire_frames_sent - self.native_frames_received)

    @property
    def pending_frames(self) -> int:
        return max(
            0,
            self.native_frames_received
            - self.receive_buffer_drops
            - self.raw_frames_consumed,
        )

    @property
    def lost_wire_frames(self) -> int:
        """Frames missing at the native listener or overwritten in its buffer."""
        return self.wire_frames_not_received + self.receive_buffer_drops

    @property
    def unaccounted_frames(self) -> int:
        """Nonzero indicates the wire→buffer→iterator accounting failed to close."""
        accounted = (
            self.wire_frames_not_received
            + self.receive_buffer_drops
            + self.raw_frames_consumed
            + self.pending_frames
        )
        return self.wire_frames_sent - accounted


@dataclass(slots=True)
class InboundBufferStats:
    """Sampled peak occupancy and transport pause state for a receive buffer."""

    peak_frames: int = 0
    peak_bytes: int = 0
    read_paused_samples: int = 0

    async def monitor(self, client: Any) -> None:
        """Sample buffer pressure at a low rate without retaining frame data."""
        ringbuffer = client._ringbuffer
        while True:
            self.peak_frames = max(self.peak_frames, len(ringbuffer))
            self.peak_bytes = max(self.peak_bytes, ringbuffer.get_buffered_bytes())
            if ringbuffer.is_reading_paused():
                self.read_paused_samples += 1
            await asyncio.sleep(0.0005)


def _make_inbound_payload(sequence: int, sent_ns: int, message_size: int) -> bytes:
    """Build a fixed-size frame carrying a sequence and monotonic send time."""
    if message_size < _INBOUND_HEADER.size:
        raise ValueError(f"message_size must be at least {_INBOUND_HEADER.size} bytes")
    header = _INBOUND_HEADER.pack(sequence, sent_ns)
    return header + bytes([sequence & 0xFF]) * (message_size - len(header))


def _decode_inbound_payload(payload: bytes) -> tuple[int, int]:
    """Read the sequence and same-host monotonic send timestamp."""
    if len(payload) < _INBOUND_HEADER.size:
        raise ValueError("inbound payload is shorter than its sequence header")
    return _INBOUND_HEADER.unpack_from(payload)


class LocalSourceServer:
    """Hold local clients and send paced or burst inbound frames to them."""

    def __init__(self, host: str = "127.0.0.1") -> None:
        self.host = host
        self.port = 0
        self._server: websockets.Server | None = None
        self._connections: dict[int, Any] = {}
        self._next_connection_id = 0
        self._connections_changed = asyncio.Event()

    async def _handle_connection(self, websocket) -> None:
        connection_id = self._next_connection_id
        self._next_connection_id += 1
        self._connections[connection_id] = websocket
        self._connections_changed.set()
        try:
            await websocket.wait_closed()
        finally:
            self._connections.pop(connection_id, None)
            self._connections_changed.set()

    async def start(self) -> None:
        """Start a no-compression local source on an ephemeral port."""
        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            0,
            compression=None,
            ping_interval=None,
        )
        sockets = self._server.sockets
        if not sockets:
            raise RuntimeError("Local source did not bind a listening socket")
        self.port = int(sockets[0].getsockname()[1])

    async def wait_for_connections(self, target: int, timeout_s: float) -> None:
        """Wait for all expected clients, with a finite connection deadline."""
        deadline = time.monotonic() + timeout_s
        while len(self._connections) < target:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Local source accepted {len(self._connections)} of "
                    f"{target} connections"
                )
            self._connections_changed.clear()
            if len(self._connections) >= target:
                return
            try:
                await asyncio.wait_for(
                    self._connections_changed.wait(), timeout=remaining
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"Local source accepted {len(self._connections)} of "
                    f"{target} connections"
                ) from exc

    async def send_frames(
        self,
        sequence_start: int,
        count: int,
        message_size: int,
        interval_ms: float = 0.0,
    ) -> tuple[tuple[int, ...], int, int]:
        """Send each unique payload on every socket and return exact send counts.

        The returned timestamps are the first source-send timestamp and the
        final completed send timestamp, both from this process's monotonic clock.
        """
        connections = tuple(sorted(self._connections.items()))
        if not connections:
            raise RuntimeError("Local source has no connected receivers")
        sent_per_connection = [0] * len(connections)
        first_send_ns = 0

        for offset in range(count):
            sequence = sequence_start + offset
            sent_ns = time.perf_counter_ns()
            if first_send_ns == 0:
                first_send_ns = sent_ns
            payload = _make_inbound_payload(sequence, sent_ns, message_size)
            for index, (_, websocket) in enumerate(connections):
                await websocket.send(payload)
                sent_per_connection[index] += 1
            if interval_ms > 0:
                await asyncio.sleep(interval_ms / 1_000)

        return tuple(sent_per_connection), first_send_ns, time.perf_counter_ns()

    async def stop(self, timeout_s: float = 2.0) -> None:
        """Close the listener and clients within a finite cleanup deadline."""
        server = self._server
        self._server = None
        if server is None:
            return
        server.close()
        try:
            await asyncio.wait_for(server.wait_closed(), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("Local source shutdown exceeded its deadline") from exc

    @property
    def url(self) -> str:
        if self.port == 0:
            raise RuntimeError("Local source has not started")
        return f"ws://{self.host}:{self.port}"


async def _serve_inbound_source(connection, host: str) -> None:
    """Run a controllable inbound source in its own process and event loop."""
    server = LocalSourceServer(host)
    try:
        await server.start()
        connection.send(("ready", server.port))
        while True:
            try:
                command = await asyncio.to_thread(connection.recv)
            except EOFError:
                return
            if command[0] == "wait-for-connections":
                await server.wait_for_connections(command[1], command[2])
                connection.send(("ok",))
            elif command[0] == "send":
                sent, first_ns, last_ns = await server.send_frames(
                    command[1], command[2], command[3], command[4]
                )
                connection.send(("ok", sent, first_ns, last_ns))
            elif command[0] == "stop":
                connection.send(("ok",))
                return
            else:
                raise ValueError(f"Unknown inbound source command: {command[0]}")
    finally:
        await server.stop(timeout_s=2.0)


def _inbound_source_process(connection, host: str) -> None:
    """Multiprocessing entry point for the separate inbound source process."""
    try:
        asyncio.run(_serve_inbound_source(connection, host))
    except BaseException as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()


class InboundSourceProcess:
    """Small bounded control channel for a source isolated from the receiver."""

    def __init__(self, host: str) -> None:
        context = multiprocessing.get_context("spawn")
        self._parent_connection, child_connection = context.Pipe()
        self._process = context.Process(
            target=_inbound_source_process,
            args=(child_connection, host),
            name="websocket-inbound-source",
        )
        try:
            self._process.start()
        except BaseException:
            child_connection.close()
            self._parent_connection.close()
            raise
        else:
            child_connection.close()
        self.host = host
        self.port = 0
        self.start_method = context.get_start_method()
        self.exitcode: int | None = None
        self._stopped = False

    @classmethod
    async def start(cls, host: str, timeout_s: float) -> InboundSourceProcess:
        source = cls(host)
        try:
            response = await source._receive(timeout_s)
            if response[0] != "ready":
                raise RuntimeError(f"Inbound source failed to start: {response}")
            source.port = int(response[1])
            return source
        except BaseException:
            await source.stop(timeout_s=1.0)
            raise

    @property
    def url(self) -> str:
        if self.port == 0:
            raise RuntimeError("Inbound source process has not started")
        return f"ws://{self.host}:{self.port}"

    async def _receive(self, timeout_s: float) -> tuple:
        try:
            ready = await asyncio.to_thread(self._parent_connection.poll, timeout_s)
        except (EOFError, OSError) as exc:
            raise RuntimeError("Inbound source control channel closed") from exc
        if not ready:
            if not self._process.is_alive():
                raise RuntimeError(
                    "Inbound source exited before replying "
                    f"(status {self._process.exitcode})"
                )
            raise TimeoutError("Inbound source control request timed out")
        try:
            response = self._parent_connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError("Inbound source control channel closed") from exc
        if response[0] == "error":
            raise RuntimeError(f"Inbound source process failed: {response[1]}")
        return response

    async def _request(self, command: tuple, timeout_s: float) -> tuple:
        if not self._process.is_alive():
            raise RuntimeError(
                f"Inbound source process exited (status {self._process.exitcode})"
            )
        self._parent_connection.send(command)
        response = await self._receive(timeout_s)
        if response[0] != "ok":
            raise RuntimeError(f"Unexpected inbound source response: {response}")
        return response[1:]

    async def wait_for_connections(self, target: int, timeout_s: float) -> None:
        await self._request(
            ("wait-for-connections", target, timeout_s), timeout_s + 0.5
        )

    async def send_frames(
        self,
        sequence_start: int,
        count: int,
        message_size: int,
        interval_ms: float,
        timeout_s: float,
    ) -> tuple[tuple[int, ...], int, int]:
        sent, first_ns, last_ns = await self._request(
            ("send", sequence_start, count, message_size, interval_ms), timeout_s
        )
        return tuple(sent), first_ns, last_ns

    async def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the source process or terminate it within a fixed deadline."""
        if self._stopped:
            return
        process = self._process
        if process.is_alive():
            try:
                await self._request(("stop",), timeout_s=min(0.5, timeout_s))
            except (RuntimeError, TimeoutError, BrokenPipeError, OSError):
                pass
            process.join(timeout=timeout_s)
        if process.is_alive():
            process.terminate()
            process.join(timeout=timeout_s)
        if process.is_alive():
            process.kill()
            process.join(timeout=timeout_s)
        self._parent_connection.close()
        self.exitcode = process.exitcode
        self._stopped = not process.is_alive()
        if self._stopped:
            process.close()
        else:
            raise TimeoutError("Inbound source process did not stop after kill")


class InboundAccounting:
    """Bounded exact message accounting with a bounded latency reservoir."""

    def __init__(
        self,
        sequence_start: int,
        expected_count: int,
        message_size: int,
        latency_sample_limit: int = _MAX_LATENCY_SAMPLES,
    ) -> None:
        self.sequence_start = sequence_start
        self.expected_count = expected_count
        self.message_size = message_size
        self._seen = bytearray(expected_count)
        self.raw_frames_consumed = 0
        self.deduplicated_frames = 0
        self.app_received_frames = 0
        self.unique_payloads_received = 0
        self.reordered_unique_payloads = 0
        self.duplicate_app_payloads = 0
        self.malformed_payloads = 0
        self._maximum_sequence: int | None = None
        self._latency_sample_limit = latency_sample_limit
        self._latency_rng = random.Random(0)
        self.latency_samples_ns: list[int] = []
        self.latency_observations = 0
        self.delivery_end_ns: int | None = None
        self.progress = asyncio.Event()

    def record_raw_frame(self, duplicate: bool = False) -> None:
        self.raw_frames_consumed += 1
        if duplicate:
            self.deduplicated_frames += 1
        self.progress.set()

    def _record_latency(self, value_ns: int) -> None:
        self.latency_observations += 1
        if len(self.latency_samples_ns) < self._latency_sample_limit:
            self.latency_samples_ns.append(value_ns)
            return
        replacement = self._latency_rng.randrange(self.latency_observations)
        if replacement < self._latency_sample_limit:
            self.latency_samples_ns[replacement] = value_ns

    def record_app_payload(self, payload: bytes) -> None:
        self.app_received_frames += 1
        received_ns = time.perf_counter_ns()
        try:
            if len(payload) != self.message_size:
                raise ValueError("unexpected payload size")
            sequence, sent_ns = _decode_inbound_payload(payload)
        except (ValueError, struct.error):
            self.malformed_payloads += 1
            self.progress.set()
            return

        offset = sequence - self.sequence_start
        if not 0 <= offset < self.expected_count:
            self.malformed_payloads += 1
            self.progress.set()
            return
        if self._seen[offset]:
            self.duplicate_app_payloads += 1
            self.progress.set()
            return

        if self._maximum_sequence is not None and sequence < self._maximum_sequence:
            self.reordered_unique_payloads += 1
        if self._maximum_sequence is None or sequence > self._maximum_sequence:
            self._maximum_sequence = sequence
        self._seen[offset] = 1
        self.unique_payloads_received += 1
        if 0 <= sent_ns <= received_ns:
            self._record_latency(received_ns - sent_ns)
        else:
            self.malformed_payloads += 1
        if self.unique_payloads_received == self.expected_count:
            self.delivery_end_ns = received_ns
        self.progress.set()

    def missing_ids(self, sample_limit: int = 8) -> tuple[int, tuple[int, ...]]:
        missing_count = 0
        samples = []
        for offset, seen in enumerate(self._seen):
            if not seen:
                missing_count += 1
                if len(samples) < sample_limit:
                    samples.append(self.sequence_start + offset)
        return missing_count, tuple(samples)


class InboundCollector:
    """Consume client messages and count pool duplicates before API delivery."""

    def __init__(self, receiver: str, client: Any, accounting: InboundAccounting):
        self.receiver = receiver
        self.client = client
        self.accounting = accounting
        self.waiting_for_next = asyncio.Event()
        self._original_hash_check = None
        self._had_instance_hash_check = False
        self._previous_instance_hash_check = None
        if receiver == "pool":
            self._original_hash_check = client._is_seen_hash
            self._had_instance_hash_check = "_is_seen_hash" in vars(client)
            if self._had_instance_hash_check:
                self._previous_instance_hash_check = vars(client)["_is_seen_hash"]

            def _count_hash_check(payload: bytes) -> bool:
                duplicate = self._original_hash_check(payload)
                self.accounting.record_raw_frame(duplicate)
                return duplicate

            # Benchmark-only instrumentation at the existing pool dedup boundary.
            client._is_seen_hash = _count_hash_check

    def restore_hash_check(self) -> None:
        """Remove benchmark instrumentation before the client is closed."""
        if self._original_hash_check is None:
            return
        if self._had_instance_hash_check:
            self.client._is_seen_hash = self._previous_instance_hash_check
        else:
            self.client.__dict__.pop("_is_seen_hash", None)
        self._original_hash_check = None

    def set_accounting(self, accounting: InboundAccounting) -> None:
        self.accounting = accounting

    async def run(self) -> None:
        while True:
            self.waiting_for_next.clear()
            self.waiting_for_next.set()
            try:
                payload = await self.client.__anext__()
            except StopAsyncIteration:
                return
            if self.receiver == "single":
                self.accounting.record_raw_frame()
            self.accounting.record_app_payload(payload)


def _client_native_snapshot(client: Any, receiver: str) -> tuple[tuple[int, ...], int]:
    """Return per-connection frame counts and total receive-buffer overwrites."""
    if receiver == "single":
        connections = (client._ws_conn,) if client._ws_conn is not None else ()
    else:
        connections = tuple(
            connection for _, connection in sorted(client._conns.items())
        )
    return (
        tuple(connection.get_seq_id() for connection in connections),
        client.get_dropped_messages(),
    )


async def _wait_for_inbound_accounting(
    accounting: InboundAccounting,
    expected_raw_frames: int,
    timeout_s: float,
    consumer_task: asyncio.Task | None = None,
) -> None:
    """Wait until all expected wire frames and unique payloads reach the app."""

    async def wait_until_complete() -> None:
        while (
            accounting.raw_frames_consumed < expected_raw_frames
            or accounting.unique_payloads_received < accounting.expected_count
        ):
            accounting.progress.clear()
            if (
                accounting.raw_frames_consumed >= expected_raw_frames
                and accounting.unique_payloads_received >= accounting.expected_count
            ):
                return
            await accounting.progress.wait()

    accounting_task = asyncio.create_task(wait_until_complete())
    watched = {accounting_task}
    if consumer_task is not None:
        watched.add(consumer_task)
    done, _ = await asyncio.wait(
        watched,
        timeout=timeout_s,
        return_when=asyncio.FIRST_COMPLETED,
    )
    try:
        if consumer_task is not None and consumer_task in done:
            if consumer_task.cancelled():
                raise RuntimeError("Inbound collector was cancelled before completion")
            error = consumer_task.exception()
            if error is not None:
                raise RuntimeError("Inbound collector failed") from error
            if accounting_task.done():
                await accounting_task
            raise RuntimeError("Inbound collector stopped before completion")
        if accounting_task not in done:
            raise TimeoutError(
                "Timed out waiting for inbound frames: "
                f"raw={accounting.raw_frames_consumed}/{expected_raw_frames}, "
                f"unique={accounting.unique_payloads_received}/"
                f"{accounting.expected_count}"
            )
        await accounting_task
    finally:
        if not accounting_task.done():
            accounting_task.cancel()
            try:
                await accounting_task
            except asyncio.CancelledError:
                pass
    if accounting_task.cancelled():
        raise RuntimeError("Inbound accounting wait was cancelled")


def _make_inbound_result(
    receiver: str,
    scenario: str,
    logical_payloads: int,
    message_size: int,
    connection_count: int,
    source_sent_per_connection: tuple[int, ...],
    native_received_per_connection: tuple[int, ...],
    receive_buffer_drops: int,
    buffer_stats: InboundBufferStats,
    accounting: InboundAccounting,
    delivery_elapsed_ns: int,
    receive_timed_out: bool,
) -> InboundScenarioResult:
    missing_count, missing_samples = accounting.missing_ids()
    return InboundScenarioResult(
        receiver=receiver,
        scenario=scenario,
        logical_payloads=logical_payloads,
        message_size=message_size,
        connection_count=connection_count,
        source_sent_per_connection=source_sent_per_connection,
        native_received_per_connection=native_received_per_connection,
        receive_buffer_drops=receive_buffer_drops,
        peak_buffered_frames=buffer_stats.peak_frames,
        peak_buffered_bytes=buffer_stats.peak_bytes,
        read_paused_samples=buffer_stats.read_paused_samples,
        raw_frames_consumed=accounting.raw_frames_consumed,
        deduplicated_frames=accounting.deduplicated_frames,
        app_received_frames=accounting.app_received_frames,
        unique_payloads_received=accounting.unique_payloads_received,
        missing_unique_payloads=missing_count,
        missing_id_samples=missing_samples,
        reordered_unique_payloads=accounting.reordered_unique_payloads,
        duplicate_app_payloads=accounting.duplicate_app_payloads,
        malformed_payloads=accounting.malformed_payloads,
        delivery_elapsed_ns=delivery_elapsed_ns,
        delivery_completed=accounting.delivery_end_ns is not None,
        latency_samples_ns=tuple(accounting.latency_samples_ns),
        latency_observations=accounting.latency_observations,
        receive_timed_out=receive_timed_out,
    )


async def _run_inbound_scenario(
    receiver: str,
    scenario: str,
    config: InboundBenchmarkConfig,
) -> InboundScenarioResult:
    """Run one finite paced, startup-burst, or post-idle receive scenario."""
    if receiver not in {"single", "pool"}:
        raise ValueError(f"Unknown inbound receiver: {receiver}")
    if scenario not in {"paced", "startup-burst", "post-idle-burst"}:
        raise ValueError(f"Unknown inbound scenario: {scenario}")

    connection_count = 1 if receiver == "single" else config.pool_connections
    server: InboundSourceProcess | None = None
    client = None
    collector: InboundCollector | None = None
    consumer_task: asyncio.Task | None = None
    buffer_stats: InboundBufferStats | None = None
    buffer_monitor_task: asyncio.Task | None = None
    client_entered = False
    try:
        server = await InboundSourceProcess.start(
            config.server_host, timeout_s=config.timeout_s
        )
        connection_config = WsConnectionConfig.default(
            server.url,
            conn_id=1,
            auto_reconnect=False,
        )
        if receiver == "single":
            client = WsSingle(connection_config)
        else:
            client = await WsPool.new(
                config=connection_config,
                pool_config=WsPoolConfig(
                    num_connections=config.pool_connections,
                    evict_interval_s=60,
                ),
            )

        await asyncio.wait_for(client.__aenter__(), timeout=config.timeout_s)
        client_entered = True
        await server.wait_for_connections(connection_count, config.timeout_s)

        initial_accounting = InboundAccounting(
            sequence_start=0,
            expected_count=(
                config.num_messages
                if scenario == "startup-burst"
                else config.warmup_messages
            ),
            message_size=config.message_size,
        )
        collector = InboundCollector(receiver, client, initial_accounting)
        consumer_task = asyncio.create_task(collector.run())
        await asyncio.wait_for(
            collector.waiting_for_next.wait(), timeout=config.timeout_s
        )

        uses_warmup = scenario != "startup-burst" and config.warmup_messages > 0
        sequence_start = 0
        if uses_warmup:
            native_before, drops_before = _client_native_snapshot(client, receiver)
            sent, _, _ = await server.send_frames(
                sequence_start=0,
                count=config.warmup_messages,
                message_size=config.message_size,
                interval_ms=0.0,
                timeout_s=config.timeout_s,
            )
            expected_warmup_wire = config.warmup_messages * connection_count
            try:
                await _wait_for_inbound_accounting(
                    initial_accounting,
                    expected_raw_frames=expected_warmup_wire,
                    timeout_s=config.timeout_s,
                    consumer_task=consumer_task,
                )
            except TimeoutError as exc:
                raise RuntimeError(f"Inbound warmup failed: {exc}") from exc

            native_after, drops_after = _client_native_snapshot(client, receiver)
            warmup_received = tuple(
                after - before
                for before, after in zip(native_before, native_after, strict=True)
            )
            expected_deduplicated = expected_warmup_wire - config.warmup_messages
            if (
                sent != (config.warmup_messages,) * connection_count
                or warmup_received != (config.warmup_messages,) * connection_count
                or drops_after != drops_before
                or initial_accounting.raw_frames_consumed != expected_warmup_wire
                or initial_accounting.deduplicated_frames != expected_deduplicated
                or initial_accounting.unique_payloads_received != config.warmup_messages
            ):
                raise RuntimeError(
                    "Inbound warmup accounting failed: "
                    f"sent={sent}, native={warmup_received}, "
                    f"deduplicated={initial_accounting.deduplicated_frames}/"
                    f"{expected_deduplicated}, "
                    f"unique={initial_accounting.unique_payloads_received}/"
                    f"{config.warmup_messages}, drops={drops_after - drops_before}"
                )
            sequence_start += config.warmup_messages

        accounting = (
            initial_accounting
            if scenario == "startup-burst"
            else InboundAccounting(
                sequence_start=sequence_start,
                expected_count=config.num_messages,
                message_size=config.message_size,
            )
        )
        collector.set_accounting(accounting)
        await asyncio.wait_for(
            collector.waiting_for_next.wait(), timeout=config.timeout_s
        )
        if scenario == "post-idle-burst":
            await asyncio.sleep(config.idle_interval_ms / 1_000)

        native_before, drops_before = _client_native_snapshot(client, receiver)
        buffer_stats = InboundBufferStats()
        buffer_monitor_task = asyncio.create_task(buffer_stats.monitor(client))
        interval_ms = config.paced_interval_ms if scenario == "paced" else 0.0
        source_sent, first_send_ns, _ = await server.send_frames(
            sequence_start=sequence_start,
            count=config.num_messages,
            message_size=config.message_size,
            interval_ms=interval_ms,
            timeout_s=(config.timeout_s + config.num_messages * interval_ms / 1_000),
        )
        expected_wire_frames = config.num_messages * connection_count
        receive_timed_out = False
        try:
            await _wait_for_inbound_accounting(
                accounting,
                expected_raw_frames=expected_wire_frames,
                timeout_s=config.timeout_s + config.num_messages * interval_ms / 1_000,
                consumer_task=consumer_task,
            )
        except TimeoutError:
            receive_timed_out = True
        buffer_monitor_task.cancel()
        try:
            await buffer_monitor_task
        except asyncio.CancelledError:
            pass
        buffer_monitor_task = None

        native_after, drops_after = _client_native_snapshot(client, receiver)
        native_received = tuple(
            after - before
            for before, after in zip(native_before, native_after, strict=True)
        )
        end_ns = accounting.delivery_end_ns or time.perf_counter_ns()
        elapsed_ns = max(0, end_ns - first_send_ns)
        return _make_inbound_result(
            receiver=receiver,
            scenario=scenario,
            logical_payloads=config.num_messages,
            message_size=config.message_size,
            connection_count=connection_count,
            source_sent_per_connection=source_sent,
            native_received_per_connection=native_received,
            receive_buffer_drops=drops_after - drops_before,
            buffer_stats=buffer_stats,
            accounting=accounting,
            delivery_elapsed_ns=elapsed_ns,
            receive_timed_out=receive_timed_out,
        )
    finally:
        try:
            if buffer_monitor_task is not None:
                buffer_monitor_task.cancel()
                try:
                    await asyncio.wait_for(buffer_monitor_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            if consumer_task is not None:
                consumer_task.cancel()
                try:
                    await asyncio.wait_for(consumer_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
        finally:
            try:
                if collector is not None:
                    collector.restore_hash_check()
                if client_entered:
                    await asyncio.wait_for(
                        client.__aexit__(None, None, None), timeout=2.0
                    )
            finally:
                if server is not None:
                    await server.stop(timeout_s=2.0)


def _inbound_failure_reasons(result: InboundScenarioResult) -> list[str]:
    """Return every violation of the strict no-loss acceptance contract."""
    reasons = []
    expected_sends = (result.logical_payloads,) * result.connection_count
    if result.source_sent_per_connection != expected_sends:
        reasons.append(f"per-connection sends={result.source_sent_per_connection}")
    expected_native = (result.logical_payloads,) * result.connection_count
    if result.native_received_per_connection != expected_native:
        reasons.append(
            f"per-connection native frames={result.native_received_per_connection}"
        )
    expected_deduplicated = result.wire_frames_sent - result.logical_payloads
    if result.deduplicated_frames != expected_deduplicated:
        reasons.append(
            f"deduplicated={result.deduplicated_frames}/"
            f"{expected_deduplicated} expected"
        )
    if result.receive_buffer_drops:
        reasons.append(f"receive-buffer overwrites={result.receive_buffer_drops}")
    if result.wire_frames_not_received:
        reasons.append(f"wire frames not observed={result.wire_frames_not_received}")
    if result.pending_frames:
        reasons.append(f"frames pending at deadline={result.pending_frames}")
    if result.missing_unique_payloads:
        reasons.append(
            f"missing unique payloads={result.missing_unique_payloads} "
            f"(sample {result.missing_id_samples})"
        )
    if result.reordered_unique_payloads:
        reasons.append(f"reordered unique payloads={result.reordered_unique_payloads}")
    if result.duplicate_app_payloads:
        reasons.append(f"duplicate app deliveries={result.duplicate_app_payloads}")
    if result.malformed_payloads:
        reasons.append(f"malformed/unexpected payloads={result.malformed_payloads}")
    if result.unaccounted_frames:
        reasons.append(f"unaccounted wire frames={result.unaccounted_frames}")
    if result.receive_timed_out:
        reasons.append("receive deadline expired")
    if not result.delivery_completed:
        reasons.append("not all unique payloads reached the application")
    return reasons


def _require_inbound_no_loss(results: list[InboundScenarioResult]) -> None:
    """Fail acceptance mode if any unique payload was lost or reordered."""
    failures = []
    for result in results:
        reasons = _inbound_failure_reasons(result)
        if reasons:
            failures.append(
                f"{result.receiver}/{result.scenario}: " + "; ".join(reasons)
            )
    if failures:
        raise RuntimeError(
            "Inbound WebSocket no-loss acceptance failed: " + " | ".join(failures)
        )


def _print_inbound_result(result: InboundScenarioResult) -> None:
    """Print physical-frame accounting separately from unique delivery rate."""
    elapsed_s = result.delivery_elapsed_ns / 1_000_000_000
    rate = result.unique_payloads_received / elapsed_s if elapsed_s > 0 else 0.0
    print(f"\n  {result.receiver} / {result.scenario}")
    print(f"    payload size                 : {result.message_size} bytes")
    print(f"    unique payloads sent         : {result.logical_payloads:,}")
    print(
        f"    successful source wire sends : {result.wire_frames_sent:,} "
        f"(per socket {list(result.source_sent_per_connection)})"
    )
    print(
        f"    native frames received      : {result.native_frames_received:,} "
        f"(per socket {list(result.native_received_per_connection)})"
    )
    print(f"    receive-buffer overwrites   : {result.receive_buffer_drops:,}")
    print(
        f"    sampled peak frames/bytes   : {result.peak_buffered_frames:,}/"
        f"{result.peak_buffered_bytes:,}; read-paused samples="
        f"{result.read_paused_samples:,} (0.5ms requested polling)"
    )
    print(f"    raw frames consumed         : {result.raw_frames_consumed:,}")
    print(f"    duplicate frames filtered   : {result.deduplicated_frames:,}")
    print(
        f"    unique app deliveries       : {result.unique_payloads_received:,}; "
        f"missing={result.missing_unique_payloads:,}; "
        f"reordered={result.reordered_unique_payloads:,}; "
        f"duplicate app={result.duplicate_app_payloads:,}"
    )
    print(
        f"    wire not observed/lost      : {result.wire_frames_not_received:,}/"
        f"{result.lost_wire_frames:,}; pending={result.pending_frames:,}; "
        f"unaccounted={result.unaccounted_frames:,}"
    )
    print(
        f"    unique delivery rate        : {rate:,.0f} unique payloads/s "
        f"over {elapsed_s:.6f}s from first source send to last unique app delivery"
    )
    if result.latency_samples_ns:
        samples = np.asarray(result.latency_samples_ns, dtype=np.float64) / 1_000_000
        sample_kind = (
            "all deliveries"
            if result.latency_observations == len(result.latency_samples_ns)
            else "bounded reservoir sample"
        )
        print(
            "    one-way source-to-app delay  : "
            f"p50={np.percentile(samples, 50):.3f}ms "
            f"p95={np.percentile(samples, 95):.3f}ms "
            f"p99={np.percentile(samples, 99):.3f}ms "
            f"({len(samples):,}/{result.latency_observations:,} {sample_kind})"
        )
    if result.receive_timed_out:
        print("    receive status              : deadline expired; counts are partial")


async def _benchmark_inbound_receiver(
    receiver: str, config: InboundBenchmarkConfig
) -> list[InboundScenarioResult]:
    """Run all receive modes for one receiver in an isolated process."""
    scenarios = ("paced", "startup-burst", "post-idle-burst")
    results = []
    for scenario in scenarios:
        results.append(await _run_inbound_scenario(receiver, scenario, config))
    return results


def _run_single_inbound_benchmark(config_dict: dict) -> None:
    config = InboundBenchmarkConfig(**config_dict)
    results = asyncio.run(_benchmark_inbound_receiver("single", config))
    for result in results:
        _print_inbound_result(result)
    if config.require_no_loss:
        _require_inbound_no_loss(results)


def _run_pool_inbound_benchmark(config_dict: dict) -> None:
    config = InboundBenchmarkConfig(**config_dict)
    results = asyncio.run(_benchmark_inbound_receiver("pool", config))
    for result in results:
        _print_inbound_result(result)
    if config.require_no_loss:
        _require_inbound_no_loss(results)

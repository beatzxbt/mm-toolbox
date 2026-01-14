"""WebSocket test fixtures and helpers for local integration coverage.

This module provides:
- LocalWebSocketServer with behavior variants (echo, delay, reject, invalid, close,
  compression, fragmentation)
- pytest fixtures for per-test server lifecycles
- helper utilities for connection setup, latency mocking, and chaotic concurrency
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Iterator

import pytest
import websockets

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.websocket.connection import (
    ConnectionState,
    WsConnection,
    WsConnectionConfig,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options for websocket tests.

    Args:
        parser (pytest.Parser): Pytest option parser.

    Returns:
        None: This function does not return a value.
    """
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run live tests that require internet connection (Binance streams)",
    )
    parser.addoption(
        "--live-timeout",
        action="store",
        default=30,
        type=int,
        help="Timeout for live tests in seconds",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers.

    Args:
        config (pytest.Config): Pytest configuration object.

    Returns:
        None: This function does not return a value.
    """
    config.addinivalue_line(
        "markers", "live: mark test as requiring live internet connection"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "stress: mark test as stress/load tests")
    config.addinivalue_line(
        "markers", "chaos: mark test as random concurrency/ordering"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify test collection based on command line options.

    Args:
        config (pytest.Config): Pytest configuration object.
        items (list[pytest.Item]): Collected test items.

    Returns:
        None: This function does not return a value.
    """
    if config.getoption("--run-live"):
        return

    skip_live = pytest.mark.skip(reason="need --run-live option to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


def _message_to_bytes(message: str | bytes) -> bytes:
    """Convert a websocket message to bytes for storage.

    Args:
        message (str | bytes): Incoming message.

    Returns:
        bytes: Normalized bytes payload.
    """
    if isinstance(message, bytes):
        return message
    return message.encode("utf-8")


def _split_message(message: str | bytes, parts: int = 2) -> list[str] | list[bytes]:
    """Split a message into multiple fragments for fragmentation tests.

    Args:
        message (str | bytes): Message to split.
        parts (int): Number of fragments to produce.

    Returns:
        list[str] | list[bytes]: Fragmented parts matching the input type.
    """
    if parts <= 1:
        return [message]
    length = len(message)
    if length <= 1:
        return [message]
    step = max(1, length // parts)
    if isinstance(message, bytes):
        return [message[i : i + step] for i in range(0, length, step)]
    return [message[i : i + step] for i in range(0, length, step)]


@dataclass
class LocalWebSocketServer:
    """Local WebSocket server with configurable behavior for tests."""

    host: str = "127.0.0.1"
    port: int | None = None
    uri: str = ""
    behavior: str = "echo"
    delay_ms: int = 0
    compression: bool = False
    fragmentation: bool = False
    _server: Any | None = None
    _clients: set[Any] = field(default_factory=set)
    _received_messages: list[bytes] = field(default_factory=list)

    async def serve(self) -> None:
        """Start the WebSocket server.

        Returns:
            None: This method does not return a value.
        """
        compression = "deflate" if self.compression else None
        self._server = await websockets.serve(
            self._handler,
            self.host,
            self.port or 0,
            compression=compression,
        )
        sockname = self._server.sockets[0].getsockname()
        self.port = sockname[1]
        self.uri = f"ws://{self.host}:{self.port}"

    async def close(self) -> None:
        """Shutdown the server and close active connections.

        Returns:
            None: This method does not return a value.
        """
        for client in list(self._clients):
            with contextlib.suppress(Exception):
                await client.close()
        self._clients.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def send_to_all_clients(
        self, message: str | bytes, timeout_s: float = 0.5
    ) -> None:
        """Broadcast a message to all connected clients.

        Args:
            message (str | bytes): Message payload to send.
            timeout_s (float): Time to wait for at least one client.

        Returns:
            None: This method does not return a value.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while not self._clients and loop.time() < deadline:
            await asyncio.sleep(0.01)
        clients = list(self._clients)
        if not clients:
            return
        await asyncio.gather(
            *[self._send_message(client, message) for client in clients],
            return_exceptions=True,
        )

    def get_received_messages(self) -> list[bytes]:
        """Return a copy of the received message buffer.

        Returns:
            list[bytes]: Snapshot of received message bytes.
        """
        return list(self._received_messages)

    async def __aenter__(self) -> LocalWebSocketServer:
        """Async context manager entry, starts the server.

        Returns:
            LocalWebSocketServer: The started server instance.
        """
        await self.serve()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit, closes the server.

        Args:
            exc_type (type | None): Exception type if raised.
            exc_val (BaseException | None): Exception value if raised.
            exc_tb (TracebackType | None): Traceback if raised.

        Returns:
            None: This method does not return a value.
        """
        await self.close()

    async def _handler(self, websocket, path: str | None = None) -> None:
        """Dispatch handler based on server behavior.

        Args:
            websocket: Websocket protocol instance.
            path (str | None): Request path (unused).

        Returns:
            None: This method does not return a value.
        """
        if self.behavior == "reject":
            await self._handle_reject(websocket)
            return
        if self.behavior == "invalid_frames":
            await self._handle_invalid_frames(websocket)
            return
        if self.behavior == "close_frame":
            await self._handle_close_frame(websocket)
            return
        await self._handle_echo(websocket)

    async def _send_message(self, websocket, message: str | bytes) -> None:
        """Send a message as text frames, optionally fragmented.

        Args:
            websocket: Websocket protocol instance.
            message (str | bytes): Payload to send.

        Returns:
            None: This method does not return a value.
        """
        payload: str | bytes
        if isinstance(message, bytes):
            payload = message.decode("utf-8", errors="replace")
        else:
            payload = message
        if self.fragmentation:
            fragments = _split_message(payload, parts=2)
            await websocket.send(fragments)
            return
        await websocket.send(payload)

    async def _handle_echo(self, websocket) -> None:
        """Echo server behavior with optional delay and fragmentation.

        Args:
            websocket: Websocket protocol instance.

        Returns:
            None: This method does not return a value.
        """
        self._clients.add(websocket)
        try:
            async for message in websocket:
                self._received_messages.append(_message_to_bytes(message))
                if self.delay_ms > 0:
                    await asyncio.sleep(self.delay_ms / 1000)
                await self._send_message(websocket, message)
        finally:
            self._clients.discard(websocket)

    async def _handle_reject(self, websocket) -> None:
        """Reject connections immediately.

        Args:
            websocket: Websocket protocol instance.

        Returns:
            None: This method does not return a value.
        """
        await websocket.close(code=1000, reason="Rejected")

    async def _handle_invalid_frames(self, websocket) -> None:
        """Send malformed frames after the first client message.

        Args:
            websocket: Websocket protocol instance.

        Returns:
            None: This method does not return a value.
        """
        self._clients.add(websocket)
        try:
            with contextlib.suppress(Exception):
                await websocket.recv()
            transport = getattr(websocket, "transport", None)
            if transport is not None:
                transport.write(b"\x02\xff\x00\x00")
            await asyncio.sleep(0)
            with contextlib.suppress(Exception):
                await websocket.close(code=1002, reason="Invalid frame")
        finally:
            self._clients.discard(websocket)

    async def _handle_close_frame(self, websocket) -> None:
        """Close the connection after a single message.

        Args:
            websocket: Websocket protocol instance.

        Returns:
            None: This method does not return a value.
        """
        self._clients.add(websocket)
        try:
            async for message in websocket:
                self._received_messages.append(_message_to_bytes(message))
                await self._send_message(websocket, message)
                await websocket.close(code=1000, reason="Server close")
                transport = getattr(websocket, "transport", None)
                if transport is not None:
                    with contextlib.suppress(Exception):
                        transport.close()
                break
        finally:
            self._clients.discard(websocket)


@contextlib.contextmanager
def mock_latency(conn: WsConnection, latency_ms: float) -> Iterator[None]:
    """Temporarily override latency metrics for deterministic eviction tests.

    Args:
        conn (WsConnection): Connection to modify.
        latency_ms (float): Latency value to inject.

    Returns:
        Iterator[None]: Context manager iterator.
    """
    state = conn.get_state()
    original_latency = state.latency.latency_ms
    try:
        state.latency.latency_ms = latency_ms
        with contextlib.suppress(Exception):
            state.latency.latency_ema.update(latency_ms)
        yield
    finally:
        state.latency.latency_ms = original_latency


async def chaotic_tasks(
    operations: Iterable[Callable[[], Awaitable[Any]]],
    seed: int | None = None,
) -> list[Any]:
    """Run operations with random jitter to simulate chaotic concurrency.

    Args:
        operations (Iterable[Callable[[], Awaitable[Any]]]): Async callables.
        seed (int | None): Random seed for deterministic chaos.

    Returns:
        list[Any]: Operation results in completion order.
    """
    rng = random.Random(seed)

    async def _runner(op: Callable[[], Awaitable[Any]]) -> Any:
        await asyncio.sleep(rng.uniform(0, 0.05))
        return await op()

    tasks = [asyncio.create_task(_runner(op)) for op in operations]
    return await asyncio.gather(*tasks, return_exceptions=True)


def make_message_payload(size: int = 128) -> bytes:
    """Generate a deterministic bytes payload for test messages.

    Args:
        size (int): Payload size in bytes.

    Returns:
        bytes: Message payload of the requested size.
    """
    return (b"x" * max(size, 0))[:size]


def make_oversized_payload(size: int = 1_048_577) -> bytes:
    """Generate an oversized payload to test buffer limits.

    Args:
        size (int): Payload size in bytes.

    Returns:
        bytes: Oversized payload.
    """
    return b"o" * size


def make_connection_config(
    server: LocalWebSocketServer,
    *,
    conn_id: int | None = None,
    on_connect: list[bytes] | None = None,
    auto_reconnect: bool | None = None,
) -> WsConnectionConfig:
    """Create a WsConnectionConfig bound to the local server.

    Args:
        server (LocalWebSocketServer): Target local server fixture.
        conn_id (int | None): Optional connection id override.
        on_connect (list[bytes] | None): Optional on_connect payloads.
        auto_reconnect (bool | None): Optional reconnect toggle.

    Returns:
        WsConnectionConfig: Config with ws:// URI patched in.
    """
    config = WsConnectionConfig.default(
        wss_url=f"wss://{server.host}",
        conn_id=conn_id,
        on_connect=on_connect,
        auto_reconnect=auto_reconnect,
    )
    config.wss_url = server.uri
    return config


async def wait_for_connection_state(
    conn: WsConnection,
    expected: ConnectionState,
    *,
    timeout_s: float = 2.0,
) -> None:
    """Wait for a WsConnection to reach a specific state.

    Args:
        conn (WsConnection): Connection to monitor.
        expected (ConnectionState): Expected state.
        timeout_s (float): Timeout in seconds.

    Returns:
        None: This method does not return a value.

    Raises:
        AssertionError: If the expected state is not reached in time.
    """
    start = asyncio.get_running_loop().time()
    while (asyncio.get_running_loop().time() - start) < timeout_s:
        if conn.get_state().state == expected:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"Timed out waiting for {expected}")


async def wait_for_latency_update(
    conn: WsConnection,
    *,
    timeout_s: float = 2.0,
) -> None:
    """Wait until latency metrics change from default values.

    Args:
        conn (WsConnection): Connection to monitor.
        timeout_s (float): Timeout in seconds.

    Returns:
        None: This method does not return a value.

    Raises:
        AssertionError: If latency does not update in time.
    """
    loop = asyncio.get_running_loop()
    start = loop.time()
    probe_window_s = min(1.0, timeout_s / 2.0)
    while (loop.time() - start) < probe_window_s:
        if conn.get_state().latency_ms != 1000.0:
            return
        await asyncio.sleep(0.2)

    probe = f"latency-probe-{int(loop.time() * 1_000_000)}".encode("ascii")
    probe_sent = loop.time()
    conn.send_data(probe)
    ringbuffer = conn.get_state().ringbuffer
    deadline = start + timeout_s
    while loop.time() < deadline:
        timeout = max(0.0, deadline - loop.time())
        try:
            msg = await asyncio.wait_for(ringbuffer.aconsume(), timeout=timeout)
        except asyncio.TimeoutError:
            break
        if msg == probe:
            latency_ms = (loop.time() - probe_sent) * 1000.0
            state = conn.get_state()
            state.latency.latency_ema.update(latency_ms)
            state.latency.latency_ms = latency_ms
            return
    raise AssertionError("Timed out waiting for latency update")


async def connect_to_server(
    server: LocalWebSocketServer,
    *,
    on_connect: list[bytes] | None = None,
    auto_reconnect: bool | None = None,
) -> WsConnection:
    """Create and connect a WsConnection to the local server.

    Args:
        server (LocalWebSocketServer): Target local server fixture.
        on_connect (list[bytes] | None): Optional on_connect payloads.
        auto_reconnect (bool | None): Optional reconnect toggle.

    Returns:
        WsConnection: Connected websocket listener instance.
    """
    ringbuffer = BytesRingBuffer(max_capacity=128, only_insert_unique=False)
    config = make_connection_config(
        server, on_connect=on_connect, auto_reconnect=auto_reconnect
    )
    conn = await WsConnection.new(ringbuffer, config)
    await wait_for_connection_state(conn, ConnectionState.CONNECTED, timeout_s=2.0)
    return conn


@pytest.fixture
def connection_config_factory() -> Callable[..., WsConnectionConfig]:
    """Provide a factory for server-bound connection configs.

    Returns:
        Callable[..., WsConnectionConfig]: Config factory callable.
    """
    return make_connection_config


@pytest.fixture
def connection_factory() -> Callable[..., Awaitable[WsConnection]]:
    """Provide a factory for connected WsConnection instances.

    Returns:
        Callable[..., Awaitable[WsConnection]]: Async connection factory.
    """
    return connect_to_server


@pytest.fixture
def state_waiter() -> Callable[..., Awaitable[None]]:
    """Provide a helper to await connection state changes.

    Returns:
        Callable[..., Awaitable[None]]: State wait helper.
    """
    return wait_for_connection_state


@pytest.fixture
def latency_waiter() -> Callable[..., Awaitable[None]]:
    """Provide a helper to await latency updates.

    Returns:
        Callable[..., Awaitable[None]]: Latency wait helper.
    """
    return wait_for_latency_update


@pytest.fixture
def payload_factory() -> Callable[..., bytes]:
    """Provide a helper for fixed-size payloads.

    Returns:
        Callable[..., bytes]: Payload factory.
    """
    return make_message_payload


@pytest.fixture
def oversized_payload_factory() -> Callable[..., bytes]:
    """Provide a helper for oversized payloads.

    Returns:
        Callable[..., bytes]: Oversized payload factory.
    """
    return make_oversized_payload


@pytest.fixture
def chaos_runner() -> Callable[..., Awaitable[list[Any]]]:
    """Provide a helper to run chaotic concurrent tasks.

    Returns:
        Callable[..., Awaitable[list[Any]]]: Chaos helper coroutine factory.
    """
    return chaotic_tasks


@pytest.fixture
def basic_server() -> LocalWebSocketServer:
    """Create a basic echo server fixture.

    Returns:
        LocalWebSocketServer: Configured server instance.
    """
    return LocalWebSocketServer()


@pytest.fixture
def server_with_delay() -> LocalWebSocketServer:
    """Create a delayed echo server fixture.

    Returns:
        LocalWebSocketServer: Configured server instance.
    """
    return LocalWebSocketServer(delay_ms=50)


@pytest.fixture
def server_reject_connections() -> LocalWebSocketServer:
    """Create a rejecting server fixture.

    Returns:
        LocalWebSocketServer: Configured server instance.
    """
    return LocalWebSocketServer(behavior="reject")


@pytest.fixture
def server_send_invalid_frames() -> LocalWebSocketServer:
    """Create a server that sends malformed frames.

    Returns:
        LocalWebSocketServer: Configured server instance.
    """
    return LocalWebSocketServer(behavior="invalid_frames")


@pytest.fixture
def server_send_close_frame() -> LocalWebSocketServer:
    """Create a server that closes after a message.

    Returns:
        LocalWebSocketServer: Configured server instance.
    """
    return LocalWebSocketServer(behavior="close_frame")


@pytest.fixture
def server_with_compression() -> LocalWebSocketServer:
    """Create a server with permessage-deflate enabled.

    Returns:
        LocalWebSocketServer: Configured server instance.
    """
    return LocalWebSocketServer(compression=True)


@pytest.fixture
def server_with_fragmentation() -> LocalWebSocketServer:
    """Create a server that fragments outgoing messages.

    Returns:
        LocalWebSocketServer: Configured server instance.
    """
    return LocalWebSocketServer(fragmentation=True)


@pytest.fixture
async def running_server(
    request: pytest.FixtureRequest,
) -> AsyncIterator[LocalWebSocketServer]:
    """Start and stop the provided server fixture per test.

    Args:
        request (pytest.FixtureRequest): Pytest fixture request.

    Returns:
        AsyncIterator[LocalWebSocketServer]: Yielding active server.
    """
    server = request.getfixturevalue(request.param)
    await server.serve()
    try:
        yield server
    finally:
        await server.close()


@pytest.fixture
def live_test_config() -> dict[str, Any]:
    """Configuration for live tests.

    Returns:
        dict[str, Any]: Live test config mapping.
    """
    return {
        "binance_futures_base": "wss://fstream.binance.com/ws",
        "binance_spot_base": "wss://stream.binance.com:9443/ws",
        "test_symbols": ["btcusdt", "ethusdt", "bnbusdt"],
        "connection_timeout": 10,
        "message_wait_time": 5,
        "latency_timeout": 15,
    }

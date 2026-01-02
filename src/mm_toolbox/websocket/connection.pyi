from collections.abc import AsyncIterable
from enum import IntEnum
from typing import Any, Self

import msgspec
from picows.picows import WSFrame, WSListener, WSTransport

from mm_toolbox.moving_average.ema import ExponentialMovingAverage as Ema
from mm_toolbox.ringbuffer.bytes import BytesRingBuffer

class ConnectionState(IntEnum):
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2

class WsConnectionConfig(msgspec.Struct):
    conn_id: int
    wss_url: str
    on_connect: list[bytes]
    auto_reconnect: bool
    @classmethod
    def default(
        cls,
        wss_url: str,
        conn_id: int | None = None,
        on_connect: list[bytes] | None = None,
        auto_reconnect: bool | None = None,
    ) -> WsConnectionConfig: ...

class LatencyTrackerState(msgspec.Struct):
    latency_ema: Ema
    latency_ms: float
    @classmethod
    def default(cls) -> LatencyTrackerState: ...

class WsConnectionState(msgspec.Struct):
    seq_id: int
    state: ConnectionState
    ringbuffer: BytesRingBuffer
    latency: LatencyTrackerState
    @property
    def is_connected(self) -> bool: ...
    @property
    def latency_ms(self) -> float: ...
    @property
    def recent_message(self) -> bytes: ...

class WsConnection(WSListener):
    """Abstract Websocket connection class, wrapping PicoWs."""

    def __init__(self, ringbuffer: BytesRingBuffer, config: WsConnectionConfig) -> None:
        """Initializes a new Websocket connection."""
        ...
    def _timed_operations(self) -> None:
        """Performs timed operations for the connection."""
        ...
    def set_on_connect(self, on_connect: list[bytes]) -> None:
        """Sets the on_connect list."""
        ...
    def send_ping(self, msg: bytes = b"") -> None:
        """Sends a PING frame to the remote endpoint."""
        ...
    def send_pong(self, msg: bytes = b"") -> None:
        """Sends a PONG frame to the remote endpoint."""
        ...
    def send_data(self, msg: bytes) -> None:
        """Sends data as a TEXT frame over the Websocket connection."""
        ...
    def close(self) -> None:
        """Closes the Websocket connection."""
        ...
    def get_config(self) -> WsConnectionConfig:
        """Returns the current connection config."""
        ...
    def get_state(self) -> WsConnectionState:
        """Returns the current connection state."""
        ...
    def on_ws_connected(self, transport: WSTransport) -> Any:
        """Called when the handshake completes successfully."""
        ...
    def on_ws_frame(self, transport: WSTransport, frame: WSFrame) -> Any:
        """Called upon receiving a new frame."""
        ...
    def on_ws_disconnected(self, transport: WSTransport) -> Any:
        """Called when the Websocket connection is closed."""
        ...
    @classmethod
    async def new(cls, ringbuffer: BytesRingBuffer, config: WsConnectionConfig) -> Self:
        """Opens a Websocket connection to the specified URL."""
        ...
    @classmethod
    async def new_with_reconnect(
        cls, ringbuffer: BytesRingBuffer, config: WsConnectionConfig
    ) -> AsyncIterable[Self]:
        """Opens a Websocket connection to the specified URL with reconnect."""
        ...

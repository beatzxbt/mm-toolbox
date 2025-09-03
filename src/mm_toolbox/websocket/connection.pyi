from collections.abc import AsyncIterable
from enum import IntEnum
from typing import Any, Self

import msgspec
from picows.picows import WSListener

from mm_toolbox.moving_average.tema import TimeExponentialMovingAverage as Tema
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
        conn_id: int = None,
        on_connect: list[bytes] = None,
        auto_reconnect: bool = True,
    ) -> WsConnectionConfig: ...

class LatencyTrackerState(msgspec.Struct):
    latency_ema: Tema
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
    def _timed_operations(self) -> Any:
        """Performs timed operations for the connection."""
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

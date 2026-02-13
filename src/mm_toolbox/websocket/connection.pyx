import os
import random
import asyncio
from typing import AsyncIterable, Self, Optional, cast

from msgspec import Struct
from picows import ws_connect
from picows.picows cimport (
    WSFrame, 
    WSTransport, 
    WSListener, 
    WSMsgType, 
)

from mm_toolbox.time.time cimport time_ns, time_ms
from mm_toolbox.websocket.connection cimport ConnectionState
from mm_toolbox.ringbuffer.bytes cimport BytesRingBuffer
from mm_toolbox.moving_average.ema cimport ExponentialMovingAverage as Ema

DEFAULT_MAX_FRAME_SIZE = 1_048_576
DEFAULT_LATENCY_PING_INTERVAL_MS = 100


cdef inline bint _is_data_frame_type(WSMsgType msg_type):
    return (
        msg_type == WSMsgType.TEXT
        or msg_type == WSMsgType.BINARY
        or msg_type == WSMsgType.CONTINUATION
    )

class WsConnectionConfig(Struct):
    conn_id: int
    wss_url: str
    on_connect: list[bytes]
    auto_reconnect: bool
    max_frame_size: int = DEFAULT_MAX_FRAME_SIZE
    latency_ping_interval_ms: int = DEFAULT_LATENCY_PING_INTERVAL_MS

    def __post_init__(self):
        if not self.wss_url.startswith("wss://"):
            raise ValueError("Invalid wss_url; must start with 'wss://'")
        if self.max_frame_size <= 0:
            raise ValueError(
                f"Invalid max_frame_size; expected >0 but got {self.max_frame_size}"
            )
        if self.latency_ping_interval_ms <= 0:
            raise ValueError(
                "Invalid latency_ping_interval_ms; expected >0 but got "
                f"{self.latency_ping_interval_ms}"
            )
    
    @classmethod
    def default(
        cls, 
        wss_url: str, 
        conn_id: Optional[int]=None, 
        on_connect: Optional[list[bytes]]=None, 
        auto_reconnect: Optional[bool]=None,
        max_frame_size: Optional[int]=None,
        latency_ping_interval_ms: Optional[int]=None,
    ) -> WsConnectionConfig:
        return WsConnectionConfig(
            conn_id=conn_id if conn_id is not None else (time_ns() + os.getpid() + random.randint(1, 10000)),
            wss_url=wss_url,
            on_connect=on_connect if on_connect is not None else [],
            auto_reconnect=auto_reconnect if auto_reconnect is not None else True,
            max_frame_size=max_frame_size if max_frame_size is not None else DEFAULT_MAX_FRAME_SIZE,
            latency_ping_interval_ms=(
                latency_ping_interval_ms
                if latency_ping_interval_ms is not None
                else DEFAULT_LATENCY_PING_INTERVAL_MS
            ),
        )

class LatencyTrackerState(Struct):
    latency_ema: Ema
    latency_ms: float
    
    @classmethod
    def default(cls) -> LatencyTrackerState:
        return LatencyTrackerState(
            latency_ema=Ema(
                window=60,
                is_fast=False,
            ),
            latency_ms=1000.0,
        )
    
class WsConnectionState(Struct):
    seq_id: int
    state: ConnectionState
    ringbuffer: BytesRingBuffer
    latency: LatencyTrackerState

    @property
    def is_connected(self) -> bool:
        return self.state == ConnectionState.CONNECTED

    @property
    def latency_ms(self) -> float:
        return self.latency.latency_ms

    @property
    def recent_message(self) -> bytes:
        return self.ringbuffer.peekright()

cdef class WsConnection(WSListener):
    """Abstract Websocket connection class, wrapping PicoWs."""

    def __cinit__(
        self, 
        BytesRingBuffer ringbuffer,
        object config,
    ):
        """Initializes a new Websocket connection."""
        # Single source of truth for state - no duplicate state variables
        self._seq_id = 0
        self._ringbuffer = ringbuffer
        self._latency_tracker = LatencyTrackerState.default()

        self._state = WsConnectionState(
            seq_id=0,
            state=ConnectionState.DISCONNECTED,
            ringbuffer=ringbuffer,
            latency=self._latency_tracker,
        )

        self._config = cast('WsConnectionConfig', config)
        
        # Use atomic-like operations for ping/pong tracking (single writes)
        self._tracker_ping_sent_time_ms = 0.0  # 0.0 means no ping sent
        self._tracker_pong_recv_time_ms = 0.0  # 0.0 means no pong received
        
        self._unfin_msg_buffer = bytearray()
        self._unfin_msg_size = 0  # Track buffer size for memory safety

        self._transport: Optional[WSTransport] = None
        self._reconnect_attempts: int = 0
        self._should_stop = False  # Lightweight stop signal
        self._loop = None

        self._latency_task = None

    def _start_latency_task(self):
        """Starts periodic internal latency pings on the connection loop."""
        if self._loop is None or self._should_stop:
            return
        if self._latency_task is None or self._latency_task.done():
            self._latency_task = self._loop.create_task(self._latency_loop())

    def _cancel_latency_task(self):
        """Cancels latency task in a loop-safe way."""
        cdef object task = self._latency_task
        cdef object loop = self._loop

        self._latency_task = None
        if task is None or task.done():
            return

        if loop is not None and loop.is_running():
            try:
                if asyncio.get_running_loop() is loop:
                    task.cancel()
                else:
                    loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                loop.call_soon_threadsafe(task.cancel)
        else:
            task.cancel()

    async def _latency_loop(self) -> None:
        """Periodically sends ping and updates latency when pong arrives."""
        cdef double interval_s = self._config.latency_ping_interval_ms / 1000.0

        try:
            while not self._should_stop:
                await asyncio.sleep(interval_s)

                if (
                    self._should_stop
                    or self._state.state != ConnectionState.CONNECTED
                    or self._transport is None
                ):
                    continue

                if self._tracker_ping_sent_time_ms > 0.0:
                    continue

                try:
                    self._transport.send_ping()
                    self._tracker_ping_sent_time_ms = time_ms()
                except Exception:
                    pass
        except asyncio.CancelledError:
            return
  
    cpdef void set_on_connect(self, list[bytes] on_connect):
        """
        Sets the on_connect list.
        """
        self._config.on_connect = on_connect

    cpdef void send_ping(self, bytes msg=b""):
        """
        Sends a PING frame to the remote endpoint.

        Args:
            msg (bytes, optional): Optional payload for the PING frame.
        """
        if self._state.state == ConnectionState.CONNECTED and self._transport is not None:
            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._send_ping_safe, msg)
            else:
                self._transport.send_ping(msg)

    def _send_ping_safe(self, bytes msg):
        """Send ping on the event loop thread to avoid cross-thread transport access."""
        if self._state.state == ConnectionState.CONNECTED and self._transport is not None:
            self._transport.send_ping(msg)

    cpdef void send_pong(self, bytes msg=b""):
        """
        Sends a PONG frame to the remote endpoint.

        Args:
            msg (bytes, optional): Optional payload for the PONG frame.
        """
        if self._state.state == ConnectionState.CONNECTED and self._transport is not None:
            self._transport.send_pong(msg)

    cpdef void send_data(self, bytes msg):
        """
        Sends data as a TEXT frame over the Websocket connection.

        Args:
            msg (bytes): The data to send as TEXT.
        """
        if self._state.state == ConnectionState.CONNECTED and self._transport is not None:
            self._transport.send(
                msg_type=WSMsgType.TEXT, 
                message=msg,
            )

    cpdef void send_data_bytearray(self, bytearray msg):
        """
        Sends a bytearray as a TEXT frame over the Websocket connection.

        Args:
            msg (bytearray): The data to send as TEXT.
        """
        cdef:
            bytearray transport_buffer
            Py_ssize_t msg_len

        if self._state.state == ConnectionState.CONNECTED and self._transport is not None:
            msg_len = len(msg)
            transport_buffer = bytearray(14 + msg_len)
            transport_buffer[14:] = msg
            self._transport.send_reuse_external_bytearray(
                WSMsgType.TEXT,
                transport_buffer,
                14,
            )

    cpdef void close(self):
        """Closes the Websocket connection."""
        # Signal task to stop first (cheapest operation)
        self._should_stop = True
        self._state.state = ConnectionState.DISCONNECTED
        self._cancel_latency_task()
        
        # Disconnect transport if available
        if self._transport is not None:
            self._transport.disconnect(graceful=True)

    cpdef object get_config(self):
        """Returns the current connection state."""
        return self._config

    cpdef object get_state(self):
        """Returns the current connection state."""
        return self._state

    # ---------- WSListener Callbacks ---------- #

    cpdef on_ws_connected(self, WSTransport transport):
        """Called when the handshake completes successfully."""
        self._should_stop = False
        self._seq_id = 0
        self._transport = transport
        self._reconnect_attempts = 0
        self._state.state = ConnectionState.CONNECTED
        self._loop = asyncio.get_running_loop()
        self._tracker_ping_sent_time_ms = 0.0
        self._tracker_pong_recv_time_ms = 0.0
        self._start_latency_task()

        for payload in self._config.on_connect:
            self.send_data(payload)

    cpdef on_ws_frame(self, WSTransport transport, WSFrame frame):
        """Called upon receiving a new frame."""
        cdef: 
            WSMsgType frame_msg_type = frame.msg_type
            bint      frame_unfinished = frame.fin == 0
            bint      frame_is_data = _is_data_frame_type(frame_msg_type)
            int       frame_size = 0
            int       max_frame_size = self._config.max_frame_size
            double    pong_recv_time_ms = 0.0
            double    ping_sent_time_ms = 0.0
            double    latency_ms = 0.0
            object    frame_payload_mv

        if frame_msg_type == WSMsgType.PONG:
            pong_recv_time_ms = time_ms()
            ping_sent_time_ms = self._tracker_ping_sent_time_ms
            self._tracker_pong_recv_time_ms = pong_recv_time_ms
            if ping_sent_time_ms > 0.0 and pong_recv_time_ms >= ping_sent_time_ms:
                latency_ms = pong_recv_time_ms - ping_sent_time_ms
                self._latency_tracker.latency_ema.update(latency_ms)
                self._latency_tracker.latency_ms = latency_ms
                self._tracker_ping_sent_time_ms = 0.0
            return

        if frame_msg_type == WSMsgType.PING:
            try:
                frame_payload_mv = frame.get_payload_as_memoryview()
                if self._state.state == ConnectionState.CONNECTED and self._transport is not None:
                    self._transport.send_pong(frame_payload_mv)
            except Exception:
                pass  # Ignore pong send errors
            return

        if frame_msg_type == WSMsgType.CLOSE:
            self._should_stop = True
            self._state.state = ConnectionState.DISCONNECTED
            self._cancel_latency_task()
            if self._transport is not None:
                try:
                    self._transport.disconnect(graceful=True)
                except Exception:
                    pass
            self._transport = None
            self._unfin_msg_buffer.clear()
            self._unfin_msg_size = 0
            return

        if not frame_is_data:
            return

        frame_size = frame.payload_size

        # Memory safety: prevent unbounded buffer growth
        if self._unfin_msg_size + frame_size > max_frame_size:
            self._unfin_msg_buffer.clear()
            self._unfin_msg_size = 0
            return

        if (
            not frame_unfinished
            and self._unfin_msg_size == 0
        ):
            self._ringbuffer.insert(frame.get_payload_as_bytes())
            self._seq_id += 1
            return

        if frame_size > 0:
            self._unfin_msg_buffer.extend(frame.get_payload_as_memoryview())
            self._unfin_msg_size += frame_size
        
        if frame_unfinished:
            return

        self._ringbuffer.insert(bytes(self._unfin_msg_buffer))
        self._unfin_msg_buffer.clear()
        self._unfin_msg_size = 0
        self._seq_id += 1

    cpdef on_ws_disconnected(self, WSTransport transport):
        """Called when the Websocket connection is closed."""
        # In the future, maybe add some default bytes message sent 
        # downstream to indicate the connection is closed. For now,
        # just close the stream without any downstream signal.
        self._should_stop = True
        self._state.state = ConnectionState.DISCONNECTED
        self._cancel_latency_task()
        self._transport = None  # Clear transport reference
        self._loop = None

    # ---------- Connection Management ---------- #

    @classmethod
    async def new(
        cls,
        ringbuffer: BytesRingBuffer,
        config: WsConnectionConfig,
    ) -> Self:
        """Opens a Websocket connection to the specified URL."""
        wst, wsl = await ws_connect(
            ws_listener_factory=lambda: cls(ringbuffer, config),
            url=config.wss_url,
            max_frame_size=config.max_frame_size,
        )
        return wsl

    @classmethod
    async def new_with_reconnect(
        cls,
        ringbuffer: BytesRingBuffer,
        config: WsConnectionConfig,
    ) -> AsyncIterable[Self]:
        """Opens a Websocket connection to the specified URL with reconnect."""
        while True:
            yield await cls.new(ringbuffer, config)
            await asyncio.sleep(1.0)

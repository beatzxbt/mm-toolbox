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
DEFAULT_LATENCY_PING_INTERVAL_MS = 1000


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

cdef class WsConnection(WSListener):
    """Abstract Websocket connection class, wrapping PicoWs."""

    def __cinit__(
        self, 
        BytesRingBuffer ringbuffer,
        object config,
    ):
        """Initializes a new Websocket connection."""
        self._config = cast('WsConnectionConfig', config)

        # Flatten hot-path state into cdef primitives
        self._conn_state = ConnectionState.DISCONNECTED
        self._seq_id = 0
        self._ringbuffer = ringbuffer
        self._latency_ms = 1000.0
        self._latency_ema = Ema(window=60, is_fast=True)
        self._max_frame_size = self._config.max_frame_size
        self._latency_ping_interval_s = self._config.latency_ping_interval_ms / 1000.0
        self._on_connect = self._config.on_connect

        # Use atomic-like operations for ping/pong tracking (single writes)
        self._tracker_ping_sent_time_ms = 0.0  # 0.0 means no ping sent
        self._tracker_pong_recv_time_ms = 0.0  # 0.0 means no pong received
        
        self._unfin_msg_buffer = bytearray()
        self._unfin_msg_size = 0  # Track buffer size for memory safety

        self._transport = None
        self._should_stop = False  # Lightweight stop signal
        self._loop = None

        self._latency_task = None

    cpdef int get_seq_id(self):
        """Returns the current sequence ID."""
        return self._seq_id

    cpdef double get_latency_ms(self):
        """Returns the current latency in milliseconds."""
        return self._latency_ms

    cpdef bint is_connected(self):
        """Returns whether the connection is currently connected."""
        return self._conn_state == ConnectionState.CONNECTED

    cpdef object get_ringbuffer(self):
        """Returns the ringbuffer for message storage."""
        return self._ringbuffer

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
        cdef double interval_s = self._latency_ping_interval_s
        cdef double ping_timeout_ms = interval_s * 3000.0

        try:
            while not self._should_stop:
                await asyncio.sleep(interval_s)

                if (
                    self._should_stop
                    or self._conn_state != ConnectionState.CONNECTED
                    or self._transport is None
                ):
                    continue

                if self._tracker_ping_sent_time_ms > 0.0:
                    # Reset if pong was lost (prevents stalling forever)
                    if time_ms() - self._tracker_ping_sent_time_ms > ping_timeout_ms:
                        self._tracker_ping_sent_time_ms = 0.0
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
        self._on_connect = on_connect

    cdef void _dispatch_on_loop(self, object func, tuple args):
        """Dispatch a callable onto the connection's event loop thread-safely."""
        cdef object loop = self._loop
        if loop is None:
            return
        try:
            if asyncio.get_running_loop() is loop:
                if self._conn_state == ConnectionState.CONNECTED and self._transport is not None:
                    func(*args)
                return
        except RuntimeError:
            pass
        loop.call_soon_threadsafe(
            lambda: self._exec_if_connected(func, args)
        )

    cpdef void _exec_if_connected(self, object func, tuple args):
        """Execute func only if still connected (called on event loop)."""
        if self._conn_state == ConnectionState.CONNECTED and self._transport is not None:
            func(*args)

    cpdef void send_ping(self, bytes msg=b""):
        """
        Sends a PING frame to the remote endpoint.

        Args:
            msg (bytes, optional): Optional payload for the PING frame.
        """
        self._dispatch_on_loop(getattr(self._transport, "send_ping", None), (msg,))

    cpdef void send_pong(self, bytes msg=b""):
        """
        Sends a PONG frame to the remote endpoint.

        Args:
            msg (bytes, optional): Optional payload for the PONG frame.
        """
        self._dispatch_on_loop(getattr(self._transport, "send_pong", None), (msg,))

    cpdef void send_data(self, bytes msg):
        """
        Sends data as a TEXT frame over the Websocket connection.

        Args:
            msg (bytes): The data to send as TEXT.
        """
        self._dispatch_on_loop(getattr(self._transport, "send", None), (WSMsgType.TEXT, msg))

    cpdef void send_data_bytearray(self, bytearray msg):
        """
        Sends a bytearray as a TEXT frame over the Websocket connection.

        Args:
            msg (bytearray): The data to send as TEXT.
        """
        cdef:
            bytearray transport_buffer
            Py_ssize_t msg_len

        msg_len = len(msg)
        transport_buffer = bytearray(14 + msg_len)
        transport_buffer[14:] = msg
        self._dispatch_on_loop(
            getattr(self._transport, "send_reuse_external_bytearray", None),
            (WSMsgType.TEXT, transport_buffer, 14)
        )

    cpdef void close(self):
        """Closes the Websocket connection."""
        # Signal task to stop first (cheapest operation)
        self._should_stop = True
        self._conn_state = ConnectionState.DISCONNECTED
        self._cancel_latency_task()
        
        # Clear any incomplete message state
        self._unfin_msg_buffer.clear()
        self._unfin_msg_size = 0
        
        # Disconnect transport if available
        if self._transport is not None:
            self._transport.disconnect(graceful=True)

    cpdef object get_config(self):
        """Returns the current connection config."""
        return self._config

    cpdef object get_state(self):
        """Returns the current connection state."""
        return self._conn_state

    # ---------- WSListener Callbacks ---------- #

    cpdef on_ws_connected(self, WSTransport transport):
        """Called when the handshake completes successfully."""
        self._should_stop = False
        self._seq_id = 0
        self._transport = transport
        self._conn_state = ConnectionState.CONNECTED
        self._loop = asyncio.get_running_loop()
        self._tracker_ping_sent_time_ms = 0.0
        self._tracker_pong_recv_time_ms = 0.0
        
        # Clear any stale fragmented message state
        self._unfin_msg_buffer.clear()
        self._unfin_msg_size = 0
        
        self._start_latency_task()

        # Send on_connect payloads directly via transport (skip redundant state check)
        for payload in self._on_connect:
            transport.send(msg_type=WSMsgType.TEXT, message=payload)

    cpdef on_ws_frame(self, WSTransport transport, WSFrame frame):
        """Called upon receiving a new frame."""
        # Guard against processing frames after close/disconnect
        if self._should_stop or self._conn_state != ConnectionState.CONNECTED:
            return

        cdef: 
            WSMsgType  frame_msg_type = frame.msg_type
            bint       frame_unfinished = frame.fin == 0
            bint       frame_is_data = _is_data_frame_type(frame_msg_type)
            Py_ssize_t frame_size = 0
            Py_ssize_t max_frame_size = self._max_frame_size
            double     pong_recv_time_ms = 0.0
            double     ping_sent_time_ms = 0.0
            double     latency_ms = 0.0
            object     frame_payload_mv

        if frame_msg_type == WSMsgType.PONG:
            pong_recv_time_ms = time_ms()
            ping_sent_time_ms = self._tracker_ping_sent_time_ms
            self._tracker_pong_recv_time_ms = pong_recv_time_ms
            if ping_sent_time_ms > 0.0 and pong_recv_time_ms >= ping_sent_time_ms:
                latency_ms = pong_recv_time_ms - ping_sent_time_ms
                self._latency_ema.update(latency_ms)
                self._latency_ms = latency_ms
                self._tracker_ping_sent_time_ms = 0.0
            return

        if frame_msg_type == WSMsgType.PING:
            if frame.payload_size > 0:
                frame_payload_mv = frame.get_payload_as_memoryview()
                if self._conn_state == ConnectionState.CONNECTED and self._transport is not None:
                    self._transport.send_pong(frame_payload_mv)
            else:
                if self._conn_state == ConnectionState.CONNECTED and self._transport is not None:
                    self._transport.send_pong(b"")
            return

        if frame_msg_type == WSMsgType.CLOSE:
            self._should_stop = True
            self._conn_state = ConnectionState.DISCONNECTED
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
            # Fast path: single complete frame — avoid picows internal copy
            frame_payload_mv = frame.get_payload_as_memoryview()
            self._ringbuffer.insert(bytes(frame_payload_mv))
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
        self._conn_state = ConnectionState.DISCONNECTED
        self._cancel_latency_task()
        self._transport = None  # Clear transport reference
        self._loop = None
        
        # Clear any incomplete message state
        self._unfin_msg_buffer.clear()
        self._unfin_msg_size = 0

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

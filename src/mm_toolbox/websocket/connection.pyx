import os
import random
import asyncio
import logging
import sys
from typing import AsyncIterable, Self, Optional, cast

from msgspec import Struct
from picows import ws_connect
from picows.picows cimport (
    WSFrame, 
    WSTransport, 
    WSListener, 
    WSMsgType, 
)

from mm_toolbox.time.time cimport time_ns, time_monotonic_ms
from mm_toolbox.websocket.connection cimport ConnectionState
from mm_toolbox.ringbuffer.bytes cimport BytesRingBuffer
from mm_toolbox.moving_average.ema cimport ExponentialMovingAverage as Ema

DEFAULT_MAX_FRAME_SIZE = 1_048_576
DEFAULT_LATENCY_PING_INTERVAL_MS = 1000
DEFAULT_RECEIVE_BUFFER_CAPACITY = 4096
DEFAULT_RECEIVE_BUFFER_MAX_BYTES = 16 * 1024 * 1024
_logger = logging.getLogger(__name__)


def _report_disconnect(future):
    """Retrieve and report transport errors even without a disconnect waiter."""
    if not future.cancelled():
        error = future.exception()
        if error is not None:
            _logger.warning("WebSocket transport disconnected with error: %s", error)


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
    receive_buffer_capacity: int = DEFAULT_RECEIVE_BUFFER_CAPACITY
    receive_buffer_max_bytes: int = DEFAULT_RECEIVE_BUFFER_MAX_BYTES

    def __post_init__(self):
        if not (self.wss_url.startswith("wss://") or self.wss_url.startswith("ws://")):
            raise ValueError("Invalid wss_url; must start with 'ws://' or 'wss://'")
        if self.max_frame_size <= 0:
            raise ValueError(
                f"Invalid max_frame_size; expected >0 but got {self.max_frame_size}"
            )
        if self.latency_ping_interval_ms <= 0:
            raise ValueError(
                "Invalid latency_ping_interval_ms; expected >0 but got "
                f"{self.latency_ping_interval_ms}"
            )
        if not isinstance(self.receive_buffer_capacity, int) or not 0 < self.receive_buffer_capacity <= 0x7FFFFFFF:
            raise ValueError("receive_buffer_capacity must be a positive 32-bit integer")
        if not isinstance(self.receive_buffer_max_bytes, int) or not self.max_frame_size <= self.receive_buffer_max_bytes <= sys.maxsize:
            raise ValueError("receive_buffer_max_bytes must be an integer between max_frame_size and sys.maxsize")
    
    @classmethod
    def default(
        cls, 
        wss_url: str, 
        conn_id: Optional[int]=None, 
        on_connect: Optional[list[bytes]]=None, 
        auto_reconnect: Optional[bool]=None,
        max_frame_size: Optional[int]=None,
        latency_ping_interval_ms: Optional[int]=None,
        receive_buffer_capacity=None,
        receive_buffer_max_bytes=None,
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
            receive_buffer_capacity=(receive_buffer_capacity if receive_buffer_capacity is not None else DEFAULT_RECEIVE_BUFFER_CAPACITY),
            receive_buffer_max_bytes=(receive_buffer_max_bytes if receive_buffer_max_bytes is not None else DEFAULT_RECEIVE_BUFFER_MAX_BYTES),
        )

cdef class _ReceiveBuffer(BytesRingBuffer):
    """Bounded wrapper FIFO, transport flow control and a single readiness event.

    Bare private buffers retain count-only overwrite behavior for fault tests.
    Wrappers configure a byte limit to enable shared high/low-water backpressure.
    All access runs on the owning event loop; no hot-path locks are introduced.
    """

    def __cinit__(self, int max_capacity):
        self._receive_loss_count = [0]
        self._read_transports = []
        self._queued_bytes = 0
        self._max_bytes = 0
        self._paused = False
        self._stopped = False
        self._generation = 0

    cpdef object get_dropped_messages(self):
        return self._receive_loss_count[0]

    cpdef Py_ssize_t get_buffered_bytes(self):
        return self._queued_bytes

    cpdef bint is_reading_paused(self):
        return self._paused

    cpdef void configure(self, Py_ssize_t max_bytes):
        if max_bytes <= 0:
            raise ValueError("Receive byte limit must be positive")
        if self._size or self._read_transports:
            raise RuntimeError("Configure the receive buffer before connecting")
        self._max_bytes = max_bytes
        self._high_count = max(1, self._max_capacity * 3 // 4)
        self._low_count = self._max_capacity // 4
        self._high_bytes = max_bytes - max_bytes // 4
        self._low_bytes = max_bytes // 4

    cdef void _register(self, object transport) except *:
        if self._max_bytes and transport is not None:
            self._read_transports.append(transport)
            if self._paused:
                transport.pause_reading()

    cdef void _unregister(self, object transport) except *:
        if transport is not None and transport in self._read_transports:
            self._read_transports.remove(transport)

    cdef void _prepare(self, Py_ssize_t tail_bytes) except *:
        if not self._max_bytes or self._paused:
            return
        # pause_reading cannot interrupt Picows's current synchronous parser
        # batch. Every remaining frame needs at least its two-byte header;
        # tail_bytes bounds both remaining frame count and payload bytes.
        if (
            self._size + <unsigned long long>(max(0, tail_bytes) // 2) >= self._high_count
            or self._queued_bytes >= self._high_bytes
            or max(0, tail_bytes) >= self._high_bytes - self._queued_bytes
        ):
            self._paused = True
            for transport in self._read_transports:
                transport.pause_reading()

    cdef void _maybe_resume(self) except *:
        if self._paused and not self._stopped and self._size <= self._low_count and self._queued_bytes <= self._low_bytes:
            self._paused = False
            for transport in self._read_transports:
                transport.resume_reading()

    cdef object _append(self, bytes item):
        cdef Py_ssize_t item_size = len(item)
        cdef object dropped = 0
        cdef unsigned long long tail
        if self._max_bytes and item_size > self._max_bytes:
            raise ValueError("Payload exceeds the receive byte limit")
        while self._size and self._max_bytes and item_size > self._max_bytes - self._queued_bytes:
            tail = self._tail
            self._queued_bytes -= len(self._buffer[tail])
            BytesRingBuffer.consume(self)
            self._buffer[tail] = b""
            dropped += 1
        if self._size == self._max_capacity:
            self._queued_bytes -= len(self._buffer[self._tail])
            dropped += 1
        BytesRingBuffer.insert(self, item)
        self._queued_bytes += item_size
        if dropped:
            self._receive_loss_count[0] += dropped
        return dropped

    cpdef bint insert(self, bytes item):
        self._append(item)
        self._prepare(0)
        return True

    cpdef bint insert_batch(self, list items):
        for item in items:
            self.insert(item)
        return True

    cpdef bytes consume(self):
        cdef unsigned long long tail = self._tail
        cdef bytes item = BytesRingBuffer.consume(self)
        self._buffer[tail] = b""
        self._queued_bytes -= len(item)
        self._maybe_resume()
        return item

    cpdef int consume_into(self, bytearray dst):
        cdef unsigned long long tail = self._tail
        cdef int size = BytesRingBuffer.consume_into(self, dst)
        self._buffer[tail] = b""
        self._queued_bytes -= size
        self._maybe_resume()
        return size

    cpdef int consume_all_into(self, list buffers):
        cdef unsigned long long tail = self._tail
        cdef int count = BytesRingBuffer.consume_all_into(self, buffers)
        for _ in range(count):
            self._queued_bytes -= len(self._buffer[tail])
            self._buffer[tail] = b""
            tail = (tail + 1) & self._mask
        self._maybe_resume()
        return count

    cpdef list consume_all(self):
        cdef list items = self.unwrapped()
        self.clear()
        return items

    cpdef void clear(self):
        BytesRingBuffer.clear(self)
        # The general ring retains old slot references. This wrapper releases
        # them so retained payload memory obeys its byte budget after draining.
        self._buffer = [b""] * self._max_capacity
        self._queued_bytes = 0
        self._maybe_resume()

    cpdef void stop(self):
        self._stopped = True
        self._buffer_not_empty_event.set()

    cpdef void reopen(self):
        if self._stopped:
            self._generation += 1
        self._stopped = False
        if not self._size:
            self._buffer_not_empty_event.clear()
        self._maybe_resume()

    async def aconsume(self):
        cdef unsigned long long generation = self._generation
        while True:
            if generation != self._generation:
                raise StopAsyncIteration
            if self._size:
                return self.consume()
            if self._stopped:
                raise StopAsyncIteration
            await self._buffer_not_empty_event.wait()


cdef class WsConnection(WSListener):
    """Abstract Websocket connection class, wrapping PicoWs.

    Manages a WebSocket connection with automatic latency tracking,
    message buffering via BytesRingBuffer, and configurable reconnection.
    """

    def __cinit__(
        self, 
        BytesRingBuffer ringbuffer,
        object config,
    ):
        """Initialize a new WebSocket connection.

        Args:
            ringbuffer: BytesRingBuffer for incoming message storage.
            config: WsConnectionConfig with connection parameters.
        """
        self._config = cast('WsConnectionConfig', config)

        # Flatten hot-path state into cdef primitives
        self._conn_state = ConnectionState.DISCONNECTED
        self._seq_id = 0
        self._dropped_messages = 0
        self._next_drop_warning = 1
        self._ringbuffer = ringbuffer
        self._receive_buffer = <_ReceiveBuffer>ringbuffer if isinstance(ringbuffer, _ReceiveBuffer) else None
        self._read_transport = None
        self._receive_loss_count = (
            (<_ReceiveBuffer>ringbuffer)._receive_loss_count
            if isinstance(ringbuffer, _ReceiveBuffer) else None
        )
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
        """Return the current sequence ID.

        Returns:
            Monotonically increasing sequence number for received messages.
        """
        return self._seq_id

    cpdef i64 get_dropped_messages(self):
        """Return receive-buffer overwrite count for this connection."""
        return self._dropped_messages

    cpdef double get_latency_ms(self):
        """Return the current latency in milliseconds.

        Returns:
            Estimated round-trip latency based on ping/pong timing.
        """
        return self._latency_ms

    cpdef bint is_connected(self):
        """Return whether the connection is currently connected.

        Returns:
            True if the WebSocket handshake is complete and active.
        """
        return self._conn_state == ConnectionState.CONNECTED

    cpdef object get_ringbuffer(self):
        """Return the ringbuffer for message storage.

        Returns:
            The BytesRingBuffer instance used for incoming messages.
        """
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

                if self._receive_buffer is not None and self._receive_buffer._paused:
                    # A deliberately paused socket cannot read its PONG. Clear
                    # the old probe and start a new one after reading resumes.
                    self._tracker_ping_sent_time_ms = 0.0
                    continue

                if self._tracker_ping_sent_time_ms > 0.0:
                    # A half-open transport must be replaced, not kept alive
                    # indefinitely by resetting its outstanding ping.
                    if time_monotonic_ms() - self._tracker_ping_sent_time_ms > ping_timeout_ms:
                        _logger.warning("WebSocket %s PONG timeout; disconnecting", self._config.conn_id)
                        self.close()
                        return
                    continue

                try:
                    self._transport.send_ping()
                    self._tracker_ping_sent_time_ms = time_monotonic_ms()
                except Exception:
                    self.close()
                    return
        except asyncio.CancelledError:
            return
  
    cpdef void set_on_connect(self, list[bytes] on_connect):
        """
        Sets the on_connect list.
        """
        self._config.on_connect = on_connect
        self._on_connect = on_connect

    cpdef void send_ping(self, bytes msg=b""):
        """
        Sends a PING frame to the remote endpoint.

        Args:
            msg (bytes, optional): Optional payload for the PING frame.
        """
        if self._conn_state == ConnectionState.CONNECTED and self._transport is not None:
            self._transport.send_ping(msg)

    cpdef void send_pong(self, bytes msg=b""):
        """
        Sends a PONG frame to the remote endpoint.

        Args:
            msg (bytes, optional): Optional payload for the PONG frame.
        """
        if self._conn_state == ConnectionState.CONNECTED and self._transport is not None:
            self._transport.send_pong(msg)

    cpdef void send_data(self, bytes msg):
        """
        Sends data as a TEXT frame over the Websocket connection.

        Args:
            msg (bytes): The data to send as TEXT.
        """
        if self._conn_state != ConnectionState.CONNECTED or self._transport is None:
            raise RuntimeError("WebSocket connection is disconnected")
        self._transport.send(WSMsgType.TEXT, msg)

    cpdef void send_data_bytearray(self, bytearray msg):
        """
        Sends a bytearray as a TEXT frame over the Websocket connection.

        Args:
            msg (bytearray): The data to send as TEXT.
        """
        cdef:
            bytearray transport_buffer
            Py_ssize_t msg_len

        if self._conn_state != ConnectionState.CONNECTED or self._transport is None:
            raise RuntimeError("WebSocket connection is disconnected")
        msg_len = len(msg)
        transport_buffer = bytearray(14 + msg_len)
        transport_buffer[14:] = msg
        self._transport.send_reuse_external_bytearray(WSMsgType.TEXT, transport_buffer, 14)

    cpdef void close(self):
        """Close the Websocket connection gracefully.

        Signals stop, cancels latency task, clears message state, and
        disconnects the transport.
        """
        # Signal task to stop first (cheapest operation)
        self._should_stop = True
        self._conn_state = ConnectionState.DISCONNECTED
        self._cancel_latency_task()
        if self._receive_buffer is not None:
            self._receive_buffer._unregister(self._read_transport)
        self._read_transport = None
        
        # Clear any incomplete message state
        self._unfin_msg_buffer.clear()
        self._unfin_msg_size = 0
        
        # Disconnect transport if available
        if self._transport is not None:
            self._transport.disconnect(graceful=False)
            self._transport = None

    cpdef object get_config(self):
        """Return the current connection config.

        Returns:
            WsConnectionConfig instance.
        """
        return self._config

    cpdef object get_state(self):
        """Return the current connection state.

        Returns:
            ConnectionState enum value.
        """
        return self._conn_state

    # ---------- WSListener Callbacks ---------- #

    cpdef on_ws_connected(self, WSTransport transport):
        """Handle successful WebSocket handshake.

        Args:
            transport: PicoWs WSTransport instance.
        """
        if self._should_stop:
            transport.disconnect(graceful=False)
            return
        self._seq_id = 0
        self._transport = transport
        self._read_transport = transport.underlying_transport
        if self._receive_buffer is not None:
            self._receive_buffer._register(self._read_transport)
        transport.disconnected_future.add_done_callback(_report_disconnect)
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
        """Handle an incoming WebSocket frame.

        Args:
            transport: PicoWs WSTransport instance.
            frame: Incoming WSFrame.
        """
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
            pong_recv_time_ms = time_monotonic_ms()
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
            if self._receive_buffer is not None:
                self._receive_buffer._unregister(self._read_transport)
            self._read_transport = None
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
            self._buffer_message(bytes(frame_payload_mv))
            if self._receive_buffer is not None:
                self._receive_buffer._prepare(frame.tail_size)
            return

        if frame_size > 0:
            self._unfin_msg_buffer.extend(frame.get_payload_as_memoryview())
            self._unfin_msg_size += frame_size
        
        if frame_unfinished:
            return

        self._buffer_message(bytes(self._unfin_msg_buffer))
        if self._receive_buffer is not None:
            self._receive_buffer._prepare(frame.tail_size)
        self._unfin_msg_buffer.clear()
        self._unfin_msg_size = 0

    cdef void _buffer_message(self, bytes payload) except *:
        cdef object dropped = 0
        if self._receive_buffer is not None:
            dropped = self._receive_buffer._append(payload)
        else:
            if self._ringbuffer.is_full():
                dropped = 1
            self._ringbuffer.insert(payload)
        if dropped:
            self._dropped_messages += dropped
            # Report the first loss and powers of two thereafter. Exact counts
            # remain available without synchronous logging on every overwrite.
            if self._dropped_messages >= self._next_drop_warning:
                self._next_drop_warning = 1 << int(self._dropped_messages).bit_length()
                _logger.warning(
                    "WebSocket %s receive buffer overflow: discarded oldest payload "
                    "(dropped=%d; warnings sampled); sequence-dependent consumers must resynchronize",
                    self._config.conn_id, self._dropped_messages,
                )
        self._seq_id += 1

    cpdef on_ws_disconnected(self, WSTransport transport):
        """Handle WebSocket disconnection.

        Args:
            transport: PicoWs WSTransport instance.
        """
        # In the future, maybe add some default bytes message sent
        # downstream to indicate the connection is closed. For now,
        # just close the stream without any downstream signal.
        self._should_stop = True
        self._conn_state = ConnectionState.DISCONNECTED
        self._cancel_latency_task()
        if self._receive_buffer is not None:
            self._receive_buffer._unregister(self._read_transport)
        self._read_transport = None
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
        listeners = []
        def listener_factory():
            listener = cls(ringbuffer, config)
            listeners.append(listener)
            return listener
        try:
            wst, wsl = await ws_connect(
                ws_listener_factory=listener_factory,
                url=config.wss_url,
                max_frame_size=config.max_frame_size,
            )
            if not wsl.is_connected():
                raise RuntimeError("WebSocket disconnected during connection establishment")
            return wsl
        except BaseException:
            for listener in listeners:
                listener.close()
            raise

    @classmethod
    async def new_with_reconnect(
        cls,
        ringbuffer: BytesRingBuffer,
        config: WsConnectionConfig,
    ) -> AsyncIterable[Self]:
        """Yield live connections, retrying failures with 1–30 second backoff.

        Closing the generator closes its current connection. With reconnect
        disabled, the first error propagates and only one connection is yielded.
        """
        delay = 1.0
        while True:
            try:
                conn = await cls.new(ringbuffer, config)
            except Exception as exc:
                if not config.auto_reconnect:
                    raise
                _logger.warning("WebSocket connection failed; retrying: %s", exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 30.0)
                continue
            delay = 1.0
            try:
                yield conn
                while conn.is_connected():
                    await asyncio.sleep(0.1)
            finally:
                conn.close()
            if not config.auto_reconnect:
                return
            await asyncio.sleep(delay)

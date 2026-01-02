import os
import random
import asyncio
import threading
from typing import AsyncIterable, Self, Optional, cast

from time import sleep
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

class WsConnectionConfig(Struct):
    conn_id: int
    wss_url: str
    on_connect: list[bytes]
    auto_reconnect: bool

    def __post_init__(self):
        if not self.wss_url.startswith("wss://"):
            raise ValueError("Invalid wss_url; must start with 'wss://'")
    
    @classmethod
    def default(
        cls, 
        wss_url: str, 
        conn_id: Optional[int]=None, 
        on_connect: Optional[list[bytes]]=None, 
        auto_reconnect: Optional[bool]=None
    ) -> WsConnectionConfig:
        return WsConnectionConfig(
            conn_id=conn_id if conn_id is not None else (time_ns() + os.getpid() + random.randint(1, 10000)),
            wss_url=wss_url,
            on_connect=on_connect if on_connect is not None else [],
            auto_reconnect=auto_reconnect if auto_reconnect is not None else True,
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
        
        self._unfin_msg_buffer = b""
        self._unfin_msg_size = 0  # Track buffer size for memory safety

        self._transport: Optional[WSTransport] = None
        self._reconnect_attempts: int = 0
        self._should_stop = False  # Lightweight stop signal

        self._timed_operations_thread = threading.Thread(
            target=self._timed_operations,
            daemon=True,
        )
        self._timed_operations_thread.start()
        
    cpdef void _timed_operations(self):
        """Performs timed operations for the connection."""
        cdef double ping_sent_time, pong_recv_time, latency_ms
        
        while not self._should_stop:
            if self._state.state != ConnectionState.CONNECTED:
                sleep(1.0)
                continue

            sleep(0.1)

            ping_sent_time = self._tracker_ping_sent_time_ms
            pong_recv_time = self._tracker_pong_recv_time_ms
            
            if ping_sent_time == 0.0:  # No ping sent yet
                self.send_ping()
                self._tracker_ping_sent_time_ms = time_ms()
            elif pong_recv_time > ping_sent_time:  # Pong received after ping
                latency_ms = pong_recv_time - ping_sent_time
                self._latency_tracker.latency_ema.update(latency_ms)
                self._latency_tracker.latency_ms = latency_ms
                
                # Reset for next ping (atomic-like single writes)
                self._tracker_ping_sent_time_ms = 0.0
                self._tracker_pong_recv_time_ms = 0.0
  
    cpdef void set_on_connect(self, list[bytes] on_connect):
        """
        Sets the on_connect list.
        """
        self._on_connect = on_connect

    cpdef void send_ping(self, bytes msg=b""):
        """
        Sends a PING frame to the remote endpoint.

        Args:
            msg (bytes, optional): Optional payload for the PING frame.
        """
        if self._state.is_connected and self._transport is not None:
            self._transport.send_ping(msg)

    cpdef void send_pong(self, bytes msg=b""):
        """
        Sends a PONG frame to the remote endpoint.

        Args:
            msg (bytes, optional): Optional payload for the PONG frame.
        """
        if self._state.is_connected and self._transport is not None:
            self._transport.send_pong(msg)

    cpdef void send_data(self, bytes msg):
        """
        Sends data as a TEXT frame over the Websocket connection.

        Args:
            msg (bytes): The data to send as TEXT.
        """
        if self._state.is_connected and self._transport is not None:
            self._transport.send(
                msg_type=WSMsgType.TEXT, 
                message=msg,
            )

    cpdef void close(self):
        """Closes the Websocket connection."""
        # Signal thread to stop first (cheapest operation)
        self._should_stop = True
        self._state.state = ConnectionState.DISCONNECTED
        
        # Disconnect transport if available
        if self._transport is not None:
            self._transport.disconnect(graceful=True)
        
        # Don't join thread here - let it stop naturally to avoid blocking

    cpdef object get_config(self):
        """Returns the current connection state."""
        return self._config

    cpdef object get_state(self):
        """Returns the current connection state."""
        return self._state

    # ---------- WSListener Callbacks ---------- #

    cpdef on_ws_connected(self, WSTransport transport):
        """Called when the handshake completes successfully."""
        self._seq_id = 0
        self._transport = transport
        self._reconnect_attempts = 0
        self._state.state = ConnectionState.CONNECTED

        for payload in self._config.on_connect:
            self.send_data(payload)

    cpdef on_ws_frame(self, WSTransport transport, WSFrame frame):
        """Called upon receiving a new frame."""
        cdef: 
            bytes      frame_bytes = frame.get_payload_as_bytes()
            WSMsgType  frame_msg_type = frame.msg_type
            bint       frame_unfinished = frame.fin == 0
            int        frame_size = len(frame_bytes)

        # Memory safety: prevent unbounded buffer growth
        if self._unfin_msg_size + frame_size > 1048576:  # 1MB limit
            self._unfin_msg_buffer = b""
            self._unfin_msg_size = 0
            return

        self._unfin_msg_buffer += frame_bytes
        self._unfin_msg_size += frame_size
        
        if frame_unfinished:
            return

        if frame_msg_type == WSMsgType.TEXT:
            self._ringbuffer.insert(self._unfin_msg_buffer)
            self._unfin_msg_buffer = b""
            self._unfin_msg_size = 0
            self._seq_id += 1

        elif frame_msg_type == WSMsgType.PONG:
            self._tracker_pong_recv_time_ms = time_ms()
            return

        elif frame_msg_type == WSMsgType.PING:
            try:
                self.send_pong(frame_bytes)
            except Exception:
                pass  # Ignore pong send errors
            return

    cpdef on_ws_disconnected(self, WSTransport transport):
        """Called when the Websocket connection is closed."""
        # In the future, maybe add some default bytes message sent 
        # downstream to indicate the connection is closed. For now,
        # just close the stream without any downstream signal.
        self._should_stop = True
        self._state.state = ConnectionState.DISCONNECTED
        self._transport = None  # Clear transport reference

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

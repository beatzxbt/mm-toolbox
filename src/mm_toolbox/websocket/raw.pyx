import time
import threading
import asyncio

from picows import ws_connect
from picows.picows cimport (
    WSFrame, 
    WSTransport, 
    WSListener, 
    WSMsgType, 
)

from mm_toolbox.time.time cimport time_ns, time_ms
from mm_toolbox.moving_average.ema cimport ExponentialMovingAverage as Ema
from mm_toolbox.websocket.raw cimport ConnectionState


cdef class WsConnection(WSListener):
    """
    Websocket connection class that manages opening, sending, receiving,
    and closing a Websocket connection.

    Attributes:
        conn_id (int): Unique identifier for this connection.
        seq_id (int): Sequence ID for received frames, incremented for
            each inbound frame.
    """

    def __init__(self, int conn_id):
        """
        Initializes a new Websocket connection.

        Args:
            conn_id (int): Unique connection identifier (e.g., an incrementing ID).
        """
        super().__init__()

        self._ev_loop = asyncio.get_event_loop()

        self._conn_id = conn_id
        self._seq_id = 0
        self._state = ConnectionState.DISCONNECTED

        self._transport = None
        self._url = ''
        self._on_connect = []
        self._process_ws_frame = None

        self._reconnect_attempts = 0
        self._reconnect_task = None
        
        # Latency tracking is only for the past 10s of data, 
        # being updated every second by the auto ping service.
        self._tracker_ping_sent = False
        self._tracker_ping_sent_time_ms = 0.0
        self._tracker_pong_recv = False
        self._tracker_pong_recv_time_ms = 0.0
        self._ema_latency_ms = Ema(window=10, is_fast=True)

        self._timed_operations_thread = threading.Thread(
            target=self._timed_operations,
            daemon=True,
        )
        self._timed_operations_thread.start()
        
    cpdef void _timed_operations(self):
        """
        Performs timed operations for the connection.
        """
        while True:
            if self._state != ConnectionState.CONNECTED:
                time.sleep(1.0)
                continue

            time.sleep(0.1)

            if not self._tracker_ping_sent:
                self.send_ping()
                self._tracker_ping_sent_time_ms = time_ms()
                self._tracker_ping_sent = True
            elif self._tracker_pong_recv:
                half_latency_ms = (self._tracker_pong_recv_time_ms - self._tracker_ping_sent_time_ms) / 2.0
                self._ema_latency_ms.update(half_latency_ms)
                
                # Reset state for the tracker
                self._tracker_ping_sent = False
                self._tracker_ping_sent_time_ms = 0.0
                self._tracker_pong_recv = False
                self._tracker_pong_recv_time_ms = 0.0
  
    cdef void set_on_connect(self, list[bytes] on_connect):
        """
        Sets the on_connect list.
        """
        self._on_connect = on_connect

    cdef void send_ping(self, bytes msg=b""):
        """
        Sends a PING frame to the remote endpoint.

        Args:
            msg (bytes, optional): Optional payload for the PING frame.
        """
        if self._state == ConnectionState.CONNECTED:
            self._transport.send_ping(msg)

    cdef void send_pong(self, bytes msg=b""):
        """
        Sends a PONG frame to the remote endpoint.

        Args:
            msg (bytes, optional): Optional payload for the PONG frame.
        """
        if self._state == ConnectionState.CONNECTED:
            self._transport.send_pong(msg)

    cdef void send_data(self, bytes msg):
        """
        Sends data as a TEXT frame over the Websocket connection.

        Args:
            msg (bytes): The data to send as TEXT.
        """
        if self._state == ConnectionState.CONNECTED:
            self._transport.send(
                msg_type=WSMsgType.TEXT, 
                message=msg,
            )

    # ---------- WSListener Callbacks ---------- #

    cdef on_ws_connected(self, WSTransport transport):
        """
        Called when the Websocket handshake completes successfully.

        Args:
            transport (WSTransport): Transport representing the live
                Websocket connection.
        """
        self._transport = transport
        self._state = ConnectionState.CONNECTED
        self._reconnect_attempts = 0

    cdef on_ws_frame(self, WSTransport transport, WSFrame frame):
        """
        Called upon receiving a new Websocket frame.

        Args:
            transport (WSTransport): The transport handling this connection.
            frame (WSFrame): The received Websocket frame, which may be
                TEXT, BINARY, PING, PONG, etc.
        """
        cdef double     recv_time_ns = time_ns()
        cdef bytes      frame_bytes = frame.get_payload_as_bytes()
        cdef bytes      frame_msg_type = frame.msg_type

        self._seq_id += 1

        if frame_msg_type == WSMsgType.TEXT:
            self._process_ws_frame(
                conn_id=self._conn_id,
                seq_id=self._seq_id,
                recv_time_ns=recv_time_ns,
                data=frame_bytes,
            )
            return

        elif frame_msg_type == WSMsgType.PONG:
            self._tracker_pong_recv_time_ms = time_ms()
            self._tracker_pong_recv = True
            return

        elif frame_msg_type == WSMsgType.PING:
            self.send_pong(frame_bytes)
            return

    cdef on_ws_disconnected(self, WSTransport transport):
        """
        Called when the Websocket connection is closed.

        Args:
            transport (WSTransport): The now-closed transport object.
        """
        # Connecting state infers it is already in a reconnect
        # attempt. Dont start new overlapping tasks
        if self._state != ConnectionState.CONNECTING:
            self._state = ConnectionState.DISCONNECTED
            self._reconnect_task = self._ev_loop.create_task(self._attempt_reconnect())
        
        # Intentional disconnect, dont attempt to reconnect
        elif self._state == ConnectionState.DISCONNECTING:
            return

    async def _attempt_reconnect(self):
        """
        Attempts to reconnect to the websocket server with exponential backoff.
        """
        delay_s = 0.25

        while (self._reconnect_attempts < 3 and 
               self._state == ConnectionState.DISCONNECTED):
            
            self._state = ConnectionState.CONNECTING

            self._reconnect_attempts += 1
            
            # Wait before attempting to reconnect
            await asyncio.sleep(delay_s)
            
            try:
                self._transport, _listener = await ws_connect(
                    ws_listener_factory=lambda: self,
                    url=self._url,
                    disconnect_on_exception=True,   
                )

                self._state = ConnectionState.CONNECTED

                for payload in self._on_connect:
                    self.send_data(payload)
                
            except Exception:
                self._state = ConnectionState.DISCONNECTED
                
                delay_s *= 2

                if self._reconnect_attempts >= 3:
                    raise ConnectionError(f"Websocket failed to reconnect; id: {self._conn_id}")

    # ---------- Connection Management ---------- #

    async def open(self, str url, list[bytes] on_connect, object process_ws_frame):
        """
        Opens a Websocket connection to the specified URL.

        Args:
            url (str): The Websocket URL to connect to.
            on_connect (list of bytes, optional): Messages to send immediately
                after connection is established.
            process_ws_frame (callable): A callback to handle received frames.
                The signature should be:
                `process_ws_frame(conn_id: int, seq_id: int, recv_time_ns: float, data: bytes)`
        """
        if self._state != ConnectionState.DISCONNECTED:
            raise RuntimeError(f"Websocket already running; id: {self._conn_id}")

        self._state = ConnectionState.CONNECTING
        self._url = url
        self._on_connect = on_connect
        self._process_ws_frame = process_ws_frame
        self._reconnect_attempts = 0

        try:
            self._transport, _listener = await ws_connect(
                ws_listener_factory=lambda: self,
                url=self._url,
                disconnect_on_exception=True,   
            )

            self._state = ConnectionState.CONNECTED

            for payload in self._on_connect:
                self.send_data(payload)

        except Exception:
            self._state = ConnectionState.DISCONNECTED
            raise ConnectionError(f"Websocket failed to start; id: {self._conn_id}")

    cdef close(self):
        """
        Closes the Websocket connection.
        """
        if self._state in [ConnectionState.DISCONNECTING, ConnectionState.DISCONNECTED]:
            return
            
        self._state = ConnectionState.DISCONNECTING

        if isinstance(self._reconnect_task, asyncio.Task) and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            self._reconnect_task = None

        if self._transport is not None:
            self._transport.send_close()
            self._transport.disconnect(graceful=True)

        self._state = ConnectionState.DISCONNECTED

    # ---------- Helper Methods ---------- #

    cdef double get_mean_latency_ms(self):
        """
        Retrieves the mean latency measured by the latency monitor.

        Returns:
            float: Mean latency in milliseconds.

        Raises:
            RuntimeError: If latency tracking is disabled.
        """
        return self._ema_latency_ms.get_value()

    cdef ConnectionState get_state(self):  
        """
        Returns the current connection state.
        """
        return self._state

    async def __aenter__(self):
        """
        Async context manager entry point.

        Opens the Websocket connection if it is not already open.
        """
        await self.open(self._url, self._on_connect, self._process_ws_frame)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit point.

        Closes the Websocket connection if it is open.
        """
        self.close()

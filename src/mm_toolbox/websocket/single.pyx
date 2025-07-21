from libc.stdint cimport (
    int64_t as i64,
    uint64_t as u64,
)

from mm_toolbox.time.time cimport time_ns

from .raw cimport WsConnection, ConnectionState

cdef class WsSingle:
    """
    A convenience wrapper around WsConnection for single-connection usage.

    This class simplifies the usage of a single WsConnection instance by configuring
    and opening a connection to a given URL, sending data, retrieving latency stats,
    and providing an optional callback for processing frames.
    """
    def __init__(
        self,
        str url,
        int conn_id=None,
        object logger=None,
        object process_ws_frame=None,
        list[bytes] on_connect=None,
    ):
        """Initializes the single connection wrapper.

        Args:
            url (str): The WebSocket URL to connect to.
            conn_id (int, optional): Connection identifier (default is 1).
            logger (object, optional): Logger for debugging or informational messages.
            process_ws_frame (callable, optional): Callback to handle received frames.
                The signature should be: process_ws_frame(conn_id: int, seq_id: int, recv_time_ns: float, data: bytes).
                If not provided, a default callback is used.
            on_connect (list[bytes], optional): Messages to send immediately after connection is established.
                If not provided, defaults to an empty list.
        """
        self._url = url
        self._conn_id = conn_id
        if self._conn_id is None:
            self._conn_id = time_ns()
        self._logger = logger

        # NOTE: Eventually there needs to be some sort of signature verification
        # for this function, but this currently breaks for cython functions
        # as it doesnt have the same pickling rules as python functions.
        # This is a known issue and will be fixed in the future.
        self._process_ws_frame = process_ws_frame    
        if self._process_ws_frame is None:
            self._process_ws_frame = self.__default_fallback_process_cb 

        self._on_connect = on_connect
        if self._on_connect is None:
            self._on_connect = []

        self._ws_conn = WsConnection(
            conn_id=self._conn_id,
        )

    cpdef void __default_fallback_process_cb(self, i64 conn_id, u64 seq_id, double recv_time_ns, bytes frame_data):
        """Default fallback callback for processing WebSocket frames.

        This callback logs the frame data.

        Args:
            seq_id (Py_ssize_t): The sequence identifier for the frame.
            recv_time (double): The timestamp when the frame was received.
            frame_data (bytes): The raw frame data.
        """
        print(frame_data.decode())

    def set_on_connect(self, list[bytes] on_connect):
        """Sets the messages to send upon connection establishment.

        Args:
            on_connect (list[bytes]): List of messages (as bytes) to send immediately after connection.
        """
        self._on_connect = on_connect
        self._ws_conn.set_on_connect(on_connect)

    async def open(self):
        """Opens the WebSocket connection and sends any on_connect messages.

        Args:
            timeout (float, optional): Connection timeout in seconds. Defaults to 5.0.

        Raises:
            Exception: Propagates exceptions from the underlying ws_conn.open call.
        """
        await self._ws_conn.open(
            url=self._url,
            on_connect=self._on_connect,
            process_ws_frame=self._process_ws_frame,
        )

    cpdef void send_data(self, bytes msg):
        """Sends data (as a TEXT frame) over the WebSocket connection.

        Args:
            msg (bytes): The message to send.
        """
        self._ws_conn.send_data(msg)

    cpdef close(self):
        """Closes the WebSocket connection gracefully."""
        self._ws_conn.close()

    # ---------- Helper Methods ---------- #

    cpdef double get_mean_latency_ms(self):
        """Retrieves the mean latency from the connection's latency monitor.

        Returns:
            float: The mean latency in milliseconds.
        """
        return self._ws_conn.get_mean_latency_ms()
        
    cpdef ConnectionState get_state(self):
        """Retrieves the current connection state.

        Returns:
            ConnectionState: The current connection state.
        """
        return self._ws_conn.get_state()
        
    cpdef bint is_connected(self):
        """Checks if the connection is in the CONNECTED state.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self._ws_conn.get_state() == ConnectionState.CONNECTED

from libc.stdint cimport (
    int64_t as i64,
)

from picows.picows cimport (
    WSTransport, 
    WSFrame, 
    WSListener,
)

from mm_toolbox.ringbuffer.bytes cimport BytesRingBuffer

cpdef enum ConnectionState:
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2

cdef class WsConnection(WSListener):
    cdef:
        # Flattened hot-path state (avoids Python attribute lookups)
        ConnectionState   _conn_state
        i64               _seq_id
        BytesRingBuffer   _ringbuffer
        double            _latency_ms
        object            _latency_ema

        # Cached config values for hot path
        int               _max_frame_size
        double            _latency_ping_interval_s
        list              _on_connect

        # Original tracking fields
        double            _tracker_ping_sent_time_ms
        double            _tracker_pong_recv_time_ms

        bytearray         _unfin_msg_buffer
        Py_ssize_t        _unfin_msg_size

        WSTransport       _transport
        bint              _should_stop

        public object     _latency_task
        object            _loop

        object            _config

    cpdef int           get_seq_id(self)
    cpdef double        get_latency_ms(self)
    cpdef bint          is_connected(self)
    cpdef object        get_ringbuffer(self)
    cpdef void          set_on_connect(self, list[bytes] on_connect)
    cpdef void          send_ping(self, bytes msg=*)
    cpdef void          send_pong(self, bytes msg=*)
    cpdef void          send_data(self, bytes msg)
    cpdef void          send_data_bytearray(self, bytearray msg)
    cpdef void          close(self)
    cpdef object        get_config(self)
    cpdef object        get_state(self)

    # Internal thread-safe send helpers
    cpdef void          _send_ping_safe(self, bytes msg)
    cpdef void          _send_pong_safe(self, bytes msg)
    cpdef void          _send_data_safe(self, bytes msg)
    cpdef void          _send_data_bytearray_safe(self, bytearray msg)
    cdef void           _dispatch_on_loop(self, object func, tuple args)

    # PicoWs should add void returns to these methods, but since they didnt
    # we cannot add them here as then it won't compile.
    cpdef               on_ws_connected(self, WSTransport transport) 
    cpdef               on_ws_frame(self, WSTransport transport, WSFrame frame)
    cpdef               on_ws_disconnected(self, WSTransport transport)

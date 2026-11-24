from libc.stdint cimport (
    int64_t as i64,
)

from picows.picows cimport (
    WSTransport, 
    WSFrame, 
    WSListener,
)

from mm_toolbox.ringbuffer.bytes cimport BytesRingBuffer

cdef class _ReceiveBuffer(BytesRingBuffer):
    cdef object _receive_loss_count
    cdef Py_ssize_t _queued_bytes
    cdef Py_ssize_t _max_bytes
    cdef Py_ssize_t _high_bytes
    cdef Py_ssize_t _low_bytes
    cdef unsigned long long _high_count
    cdef unsigned long long _low_count
    cdef unsigned long long _generation
    cdef unsigned long long _pause_generation
    cdef i64 _pause_started_time_ns
    cdef i64 _paused_time_ns
    cdef bint _paused
    cdef bint _stopped
    cdef list _read_transports
    cpdef object get_dropped_messages(self)
    cpdef Py_ssize_t get_buffered_bytes(self)
    cpdef bint is_reading_paused(self)
    cpdef void configure(self, Py_ssize_t max_bytes)
    cpdef void stop(self)
    cpdef void reopen(self)
    cpdef bint insert(self, bytes item)
    cpdef bint insert_batch(self, list items)
    cpdef bytes consume(self)
    cpdef int consume_into(self, bytearray dst)
    cpdef int consume_all_into(self, list buffers)
    cpdef list consume_all(self)
    cpdef void clear(self)
    cdef object _append(self, bytes item)
    cdef void _prepare(self, Py_ssize_t tail_bytes) except *
    cdef void _register(self, object transport) except *
    cdef void _unregister(self, object transport) except *
    cdef void _maybe_resume(self) except *
    cdef i64 _pause_time_ns(self)

cpdef enum ConnectionState:
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2

cdef class WsConnection(WSListener):
    cdef:
        # Flattened hot-path state (avoids Python attribute lookups)
        ConnectionState   _conn_state
        i64               _seq_id
        i64               _dropped_messages
        object            _next_drop_warning
        object            _receive_loss_count
        _ReceiveBuffer    _receive_buffer
        object            _read_transport
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
        double            _last_ping_sent_time_ms
        double            _heartbeat_loop_lag_ms
        i64               _heartbeat_pings_sent
        i64               _heartbeat_pongs_received
        Py_ssize_t        _last_pong_payload_size
        unsigned long long _probe_pause_generation
        bytes             _probe_payload
        i64               _probe_paused_time_ns
        i64               _heartbeat_pause_affected_samples
        i64               _heartbeat_pongs_matched

        bytearray         _unfin_msg_buffer
        Py_ssize_t        _unfin_msg_size

        WSTransport       _transport
        bint              _should_stop

        public object     _latency_task
        object            _loop

        object            _config

    cpdef int           get_seq_id(self)
    cpdef i64           get_dropped_messages(self)
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
    cpdef dict          _heartbeat_snapshot(self)
    cdef double         _readable_ping_age_ms(self, double now_ms)
    cdef void           _buffer_message(self, bytes payload) except *

    # PicoWs should add void returns to these methods, but since they didnt
    # we cannot add them here as then it won't compile.
    cpdef               on_ws_connected(self, WSTransport transport) 
    cpdef               on_ws_frame(self, WSTransport transport, WSFrame frame)
    cpdef               on_ws_disconnected(self, WSTransport transport)

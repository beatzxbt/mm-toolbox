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
        i64             _seq_id
        BytesRingBuffer _ringbuffer
        object          _latency_tracker
        object          _state
        object          _config

        double          _tracker_ping_sent_time_ms
        double          _tracker_pong_recv_time_ms

        bytes           _raw_unfin_msg_buffer
        int             _unfin_msg_size

        WSTransport     _transport
        int             _reconnect_attempts
        bint            _should_stop

        object          _timed_operations_thread

    cpdef void          _timed_operations(self)
    cpdef void          set_on_connect(self, list[bytes] on_connect)
    cpdef void          send_ping(self, bytes msg=*)
    cpdef void          send_pong(self, bytes msg=*)
    cpdef void          send_data(self, bytes msg)
    cpdef void          close(self)
    cpdef object        get_config(self)
    cpdef object        get_state(self)

    # PicoWs should add void returns to these methods, but since they didnt
    # we cannot add them here as then it won't compile.
    cpdef               on_ws_connected(self, WSTransport transport) 
    cpdef               on_ws_frame(self, WSTransport transport, WSFrame frame)
    cpdef               on_ws_disconnected(self, WSTransport transport)
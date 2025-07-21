from libc.stdint cimport (
    int64_t as i64,
    uint64_t as u64,
)

from picows.picows cimport WSTransport, WSFrame, WSListener

from mm_toolbox.moving_average.ema cimport ExponentialMovingAverage as Ema

cpdef enum ConnectionState:
    DISCONNECTED = 0x0
    CONNECTING = 0x1
    CONNECTED = 0x2
    DISCONNECTING = 0x3

cdef class WsConnection(WSListener):
    cdef:
        i64 _conn_id
        u64 _seq_id
        ConnectionState _state

        WSTransport _transport
        str _url
        list _on_connect
        object _process_ws_frame

        int _reconnect_attempts
        object _reconnect_task

        bint _tracker_ping_sent
        double _tracker_ping_sent_time_ms
        bint _tracker_pong_recv
        double _tracker_pong_recv_time_ms
        Ema _ema_latency_ms

        object _ev_loop
        object _timed_operations_thread

    cdef void _timed_operations(self)

    cdef void set_on_connect(self, list on_connect)
    cdef void send_data(self, bytes msg)
    cdef void send_ping(self, bytes msg=*)
    cdef void send_pong(self, bytes msg=*)

    cdef void on_ws_connected(self, WSTransport transport)
    cdef void on_ws_frame(self, WSTransport transport, WSFrame frame)
    cdef void on_ws_disconnected(self, WSTransport transport)

    cdef void close(self)

    cdef double get_mean_latency_ms(self)
    cdef ConnectionState get_state(self)


from libc.stdint cimport (
    uint16_t as u16,
    uint64_t as u64,
)

from .raw cimport WsConnection, WsConnectionConfig

cdef class PoolQueue:
    cdef:
        u64             _max_size            
        u64             _current_size       
        object          _queue     
        set             _latest_hashes   

    cdef u64           generate_hash(self, bytes msg)
    cdef bint          is_unique(self, u64 hash_value)
    cdef bint          is_empty(self)
    cdef void          put_item(self, bytes item, u64 hash_value)
    cdef void          put_item_with_overwrite(self, bytes item, u64 hash_value)
    cdef bytes         take_item(self)
    cdef list[bytes]   take_all(self)

cdef class WsPoolConfig:
    cdef public u16 evict_interval_s

cdef class WsPool:
    cdef:
        u64             _size
        object          _logger
        WsPoolConfig    _config

        PoolQueue       _queue
        dict            _conns
        set             _fast_conns

        object          _msg_ingress_task
        object          _conn_eviction_task

        bint            _is_running

    cdef inline u64     _generate_conn_id(self)
    cdef inline void    _process_ws_frame(self, u64 seq_id, double time, memoryview frame)
    cpdef void          send_data(self, msg)
    cpdef void          close(self)

    # async def _ingest_queue_data(self)    
    # async def _enforce_eviction(self)
    # async def _spawn_new_conn(self, str url, list[dict] on_connect)
    # async def open(self, str url, list[dict] on_connect, process)

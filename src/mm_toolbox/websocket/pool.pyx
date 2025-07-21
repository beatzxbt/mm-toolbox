import time
import threading
import asyncio
from collections import deque
from warnings import warn as warning
from xxhash import xxh3_64_intdigest

from libc.stdint cimport (
    uint16_t as u16,
    uint32_t as u32,
    uint64_t as u64,
)

from mm_toolbox.time.time cimport time_ns, time_s

from mm_toolbox.websocket.raw cimport (
    ConnectionState,
    WsConnection
)


cdef class PoolQueue:
    def __cinit__(self, u16 max_size=1000):
        """Initializes the PoolQueue with a specified maximum size.

        Args:
            max_size: The maximum number of items the queue can hold.

        Raises:
            ValueError: If max_size is less than or equal to 0.
        """
        if max_size <= 0:
            raise ValueError(f"Invalid max_size; expected >0, got {max_size}")
        self._max_size = max_size
        self._current_size = 0
        self._queue: deque = deque(maxlen=max_size)

        # We can settle with a deque here as lookups go from 
        # newest -> oldest, meaning the vast majority of msgs
        # will be the 1st result. Sets are far less efficient 
        # for disguarding old messages, so for now we settle with this.
        self._latest_hashes: deque = deque(maxlen=max_size)

    cdef u64 generate_hash(self, bytes msg):
        """Generates a 64-bit hash for the given message using xxhash.

        Args:
            msg: The message bytes to hash.

        Returns:
            A 64-bit hash value.
        """
        return xxh3_64_intdigest(msg)

    cdef bint is_unique(self, u64 hash_value):
        """Checks if the hash is unique (not already in the queue).

        Args:
            hash_value: The hash value to check.

        Returns:
            True if the hash is unique, False otherwise.
        """
        return hash_value not in self._latest_hashes

    cdef bint is_empty(self):
        """Checks if the queue is empty.

        Returns:
            True if the queue is empty, False otherwise.
        """
        return self._current_size == 0

    cdef void put_item(self, bytes item, u64 hash_value):
        """Adds an item to the queue if there is space.

        Args:
            item: The item to add to the queue.
            hash_value: The hash value of the item.

        Raises:
            IndexError: If the queue is full.
        """
        if self._current_size < self._max_size:
            self._queue.append((item, hash_value))
            self._latest_hashes.add(hash_value)
            self._current_size += 1
        else:
            raise IndexError("PoolQueue is full. Cannot add new item.")

    cdef void put_item_with_overwrite(self, bytes item, u64 hash_value):
        """Adds an item to the queue, overwriting the oldest item if the queue is full.

        Args:
            item: The item to add to the queue.
            hash_value: The hash value of the item.
        """
        cdef:
            bytes old_item
            u64 old_hash_value

        if self._current_size >= self._max_size:
            old_item, old_hash_value = self._queue.popleft()
            self._latest_hashes.remove(old_hash_value)
            self._current_size -= 1

        self.put_item(item, hash_value)

    cdef bytes take_item(self):
        """Removes and returns the oldest item from the queue.

        Returns:
            The oldest item from the queue.

        Raises:
            IndexError: If the queue is empty.
        """
        if self._current_size <= 0:
            raise IndexError("PoolQueue is empty. Cannot take an item.")

        cdef:
            bytes item
            u64 hash_value

        item, hash_value = self._queue.popleft()
        self._latest_hashes.remove(hash_value)
        self._current_size -= 1
        return item

    cdef list[bytes] take_all(self):
        """Empties the queue into a list and restarts all internal states.

        Returns:
            A list containing all items from the queue.
        """
        if self._current_size <= 0:
            return []

        cdef:
            u64 i
            bytes item
            u64 hash_value
            list[bytes] all_items = [None] * self._current_size

        for i in range(0, self._current_size):
            all_items[i] = self._queue.popleft()

        self._current_size = 0
        self._latest_hashes.clear()
        return all_items

cdef class WsPoolConfig:
    """Configuration for WebSocket connection eviction and latency measurement policies.

    Attributes:
        evict_interval_s: The interval in seconds at which eviction conditions are checked.
    """
    def __init__(
        self, 
        u16 evict_interval_s=60, 
    ):
        self.evict_interval_s = evict_interval_s 

cdef class WsPool:
    """Manages a pool of fast WebSocket connections."""
    def __init__(
        self,
        u16 size,
        function on_message,
        list[dict] on_connect,
        object logger,
        WsPoolConfig config=None,
    ) -> None:
        """Initializes the WebSocket pool.

        Args:
            size: The number of WebSocket connections to maintain in the pool.
            on_message: Callback function to handle incoming messages.
            on_connect: List of payloads to send upon connecting.
            logger: Logger instance for logging operations.
            config: Configuration for the pool (optional).

        Raises:
            ValueError: If size is less than or equal to 1.
        """
        self._size = size
        if self._size <= 1:
            raise ValueError(f"Invalid pool size; expected >1 but got {self._size}.")

        self._user_on_message = on_message
        self._on_connect = on_connect
        self._logger = logger

        self._config = config
        if self._config is None:
            self._config = WsPoolConfig()

        self._queue = PoolQueue() 
        self._conns: dict[u64, WsConnection] = {}
        self._conn_speed_tracker: dict[u64, int] = {}
        self._fast_conns: set[u64] = set()

        self._last_conn_eviction_time = 0.0

        self._msg_ingress_task: asyncio.Task = None

        self._timed_operations_thread = threading.Thread(
            target=self._timed_operations,
            daemon=True,
        )
        self._timed_operations_thread.start()

        self._state = ConnectionState.DISCONNECTED

    cdef inline u64 _generate_conn_id(self):
        """Generates a unique connection ID using nanosecond timestamps.

        Returns:
            A unique 64-bit connection ID.
        """
        return time_ns()

    cdef inline void _process_ws_frame(self, u64 conn_id, u64 seq_id, double time, bytes frame):
        """Feeds data from WebSocket connections into the queue if it is unique.

        Args:
            conn_id: The connection ID.
            seq_id: The sequence ID.
            time: The timestamp.
            frame: The frame data as a memoryview.
        """
        cdef:
            bytes   msg = frame
            u64     msg_hash = self._queue.generate_hash(msg)
            bint    hash_is_unique = self._queue.is_unique(msg_hash)
    
        if hash_is_unique:
            self._queue.put_item(msg, msg_hash)
            self._conn_speed_tracker[conn_id] += 1

    async def _msg_ingress(self) -> None:
        """Ingests queue data and feeds it to a user-provided processing function."""
        while self._is_running:
            try:
                if (msg := self._queue.take_item() != b""):
                    self._user_on_message(msg)
                else:
                    await asyncio.sleep(0)
                    continue
                    
            except asyncio.CancelledError:
                return

            except Exception:
                # We can guarantee that the queue is not the source of error here. 
                # Any errors here are due to the user's processing function, and 
                # hopefully should be handled/logged there.
                pass

    cpdef void _timed_operations(self):
        """Enforces the configuration to evict slow connections."""
        cdef double over_time_limit
        cdef double time_now_s = time_s()
        cdef double next_eviction_time = time_now_s + self._config.evict_interval_s

        while self._state == ConnectionState.CONNECTED:
            time_now_s = time_s()

            over_time_limit = time_now_s >= next_eviction_time

            if not over_time_limit:
                time.sleep(1)
                continue
            
            # Sort connections by speed (higher values mean faster connections)
            # self._conn_speed_tracker contains {conn_id: speed_value} pairs
            # Convert to list of (conn_id, speed) tuples and sort by speed in descending order
            conn_speeds = [(conn_id, speed) for conn_id, speed in self._conn_speed_tracker.items()]
            conn_speeds.sort(key=lambda x: x[1], reverse=True) 
            
            # Take the top 25% as fast connections and the bottom 25% as slow connections.
            fast_count = max(1, self._size // 4)
            fast_conns = conn_speeds[:fast_count]
            slow_conns = conn_speeds[self._size - fast_count:]
            
            for conn_id, _ in slow_conns:
                self._conns[conn_id].close()

                if conn_id in self._fast_conns:
                    self._fast_conns.remove(conn_id)

            for conn_id, _ in fast_conns:
                self._fast_conns.add(conn_id)

            # Reset eviction timer and active queue size. We manually
            # filter through the queue and process each message before
            # clearing to prevent data loss. This purposely blocks,
            # letting the websockets back up data before resuming.
            #
            # If you get backpressure issues, speed up your processing!
            next_eviction_time = time_s() + self._config.evict_interval_s

            for conn_id, conn in self._conns.items():
                conn.reset_seq_id()

            unprocessed_msgs = self._queue.take_all()
            for msg in unprocessed_msgs:
                self._user_on_message(msg)
            
    async def _open_new_conn(
        self,
        str url,
        list[dict] on_connect=None,
    ):
        """Establishes a new WebSocket connection and adds it to the connection pool.

        Args:
            url: The WebSocket URL to connect to.
            on_connect: List of payloads to send upon connecting. Defaults to None.

        Returns:
            None
        """
        new_conn_id = self._generate_conn_id()
        
        new_conn = WsConnection(conn_id=new_conn_id)

        await new_conn.start(
            url=url,
            on_connect=on_connect,
        )

        self._conns[new_conn_id] = new_conn

    cpdef void set_on_connect(self, list[dict] on_connect):
        """Sets the on_connect callback for all connections in the pool.

        Args:
            on_connect: List of payloads to send upon connecting.
        """
        self._on_connect = on_connect
        for conn in self._conns.values():
            conn.set_on_connect(on_connect)

    async def open(
        self,
        str url,
        list[dict] on_connect=None,
    ):
        """Starts all WebSocket connections in the pool.

        Args:
            url: The WebSocket URL to connect to.
            on_connect: List of payloads to send upon connecting. Defaults to None.

        Returns:
            None

        Raises:
            RuntimeError: If the connection is already running.
        """
        if self._state != ConnectionState.DISCONNECTED:
            raise RuntimeError("Connection already running; cannot open new connection.")

        self._state = ConnectionState.CONNECTING

        await asyncio.gather(
            *[
                self._open_new_conn(url, on_connect)
                for _ in range(self._size)
            ]
        )

        self._state = ConnectionState.CONNECTED

    cpdef void send_data(self, msg):
        """Sends a payload through all WebSocket connections in the pool.

        Args:
            msg: The message payload to send.

        Returns:
            None
        """
        if self._state != ConnectionState.CONNECTED:
            raise RuntimeError("Connection not running; cannot send data.")

        if len(self._fast_conns) > 0:
            for conn_id in self._fast_conns:
                self._conns[conn_id].send_data(msg)
        else:
            for conn in self._conns.values():
                conn.send_data(msg)
 
    cpdef void close(self):
        """Shuts down all WebSocket connections and stops the eviction task.

        Returns:
            None
        """
        self._state = ConnectionState.DISCONNECTING

        if self._timed_operations_thread.is_alive():
            self._timed_operations_thread.join()

        for conn in self._conns.values():
            conn.close()

        self._state = ConnectionState.DISCONNECTED
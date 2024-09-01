import asyncio
from abc import ABC, abstractmethod
from warnings import warn as warning
from dataclasses import dataclass
from typing import List, Dict, Optional

from mm_toolbox.logging import Logger

from .conn import PayloadData, SingleWsConnection


@dataclass
class WsPoolEvictionPolicy:
    """
    Defines the eviction policy for WebSocket connections.

    Attributes
    ----------
    interval_s : float
        Interval (in seconds) to check the eviction conditions.
    """

    interval_s: float = 300.0


class WsPool(ABC):
    """
    Manages a pool of WebSocket connections.

    Design
    ------
    Creates some N initial connections (conns), and measures speed performance of each
    by tracking their sequence id (seq id) arrivals.

    For example, a message {data: {}} is recieved from some API and sent to all N conns.
    All these conns forward their formatted payload through their respective queues, and is
    recieved within the self._ingest_queue() function. This then checks it against a last
    known seq id within the pool. The penalty applies in the following situation.

        * Conn 2 sends the data first, so its seq id is now the last known seq id.
        * Conn 1 sends the data second, seq id is not new so it is penalized by 1 point.
        * Conn 3 sends the data second, seq id is not new so it is penalized by 1 point.

    Following this penality mechanism over multiple messages, it clearly shows, in order,
    which connections are the fastest (less penalties) and which are the slowest (more penalties).
    
    The eviction policy collects these scores every X seconds (defined in WsPoolEvictionPolicy),
    and replaces the slowest connection with a new one.

    Consequently, it also penalizes websockets dropping frames as seq id's will not match.
    It is highly unlikely that the websocket with a higher message count will be sending bogus data.

    All scores are reset across the board and the measuring process starts again.
    """

    def __init__(
        self,
        size: int,
        logger: Logger,
        eviction_policy: Optional[WsPoolEvictionPolicy] = None,
    ) -> None:
        """
        Initializes the WebSocket connection pool.

        Parameters
        ----------
        size : int
            Number of WebSocket connections in the pool.

        logger : Logger
            Logger instance for logging activities.

        eviction_policy : WsPoolEvictionPolicy, optional
            Eviction policy for the pool (default is None).
        """
        self.size = size
        if self.size <= 1:
            warning("Pool size cannot be less than 2, defaulting to 2.", RuntimeWarning)
            self.size = 2

        self.logger = logger
        self.eviction_policy = (
            eviction_policy if eviction_policy else WsPoolEvictionPolicy()
        )

        self._started: bool = False
        self._latest_seq_id: int = 0
        self._conn_pool: List[SingleWsConnection] = []
        self._conn_speed_penalties: Dict[int, int] = {}
        self._conn_ingress_tasks: Dict[int, asyncio.Task] = {}
        self._eviction_task: asyncio.Task = None

    @abstractmethod
    def data_handler(self, recv: PayloadData) -> None:
        """
        Processes received WebSocket data.

        Parameters
        ----------
        recv : PayloadData
            The data received from the WebSocket connection.
        """
        pass

    async def _ingest_data(self, conn: SingleWsConnection) -> None:
        """
        Ingests data from a WebSocket connection.

        Parameters
        ----------
        conn : SingleWsConnection
            The WebSocket connection to ingest data from.
        """
        conn_id = conn.conn_id

        while self._started and conn.running:
            try:
                (seq_id, time, payload) = conn.queue.get_nowait()

                if seq_id > self._latest_seq_id:
                    self._latest_seq_id = seq_id
                    self.data_handler(payload)
                else:
                    self._conn_speed_penalties[conn_id] += 1

            except asyncio.QueueEmpty:
                await asyncio.sleep(0)
                continue

            except asyncio.CancelledError:
                # Flush all remaining objects and shutdown.
                await conn.queue.join()
                return

            except Exception as e:
                self.logger.warning(f"Conn {conn_id} queue ingress - {e}")

    async def _enforce_eviction_policy(
        self, url: str, on_connect: Optional[List[Dict]] = None
    ) -> None:
        """
        Enforces the eviction policy to remove slow connections.

        Parameters
        ----------
        url : str
            The WebSocket URL to reconnect to.

        on_connect : Optional[List[Dict]], optional
            List of payloads to send upon connecting (default is None).
        """
        while True:
            await asyncio.sleep(self.eviction_policy.interval_s)

            if not self._conn_speed_penalties:
                continue
            
            # Find the connection with the highest penalty.
            slowest_conn_id = max(
                self._conn_speed_penalties, key=self._conn_speed_penalties.get
            )
            slowest_conn: SingleWsConnection = None

            for conn in self._conn_pool:
                if conn.conn_id == slowest_conn_id:
                    slowest_conn = conn
                    break
        
            if slowest_conn:
                # Remove slow connection.
                slowest_conn.close()
                self._conn_pool.remove(slowest_conn)
                del self._conn_speed_penalties[slowest_conn_id]
                ingress_task = self._conn_ingress_tasks.pop(slowest_conn_id)
                ingress_task.cancel()
                self.logger.info(f"Conn '{slowest_conn_id}' evicted.")

                # Reset all the seq ids & penalties.
                self._latest_seq_id = 0
                self._conn_speed_penalties = {key: 0 for key in self._conn_speed_penalties}
                for conn in self._conn_pool:
                    conn.reset_seq_id()

                # Start a new connection.
                await self._start_new_conn(url, on_connect)

    async def _start_new_conn(self, url: str, on_connect: Optional[List[Dict]] = None) -> None:
        new_conn = SingleWsConnection(logger=self.logger)
        await new_conn.start(
            url=url,
            on_connect=on_connect,
        )
        self._conn_pool.append(new_conn)
        self._conn_speed_penalties[new_conn.conn_id] = new_conn.seq_id
        self._conn_ingress_tasks[new_conn.conn_id] = asyncio.create_task(
            self._ingest_data(new_conn)
        )
        
    async def start(
        self,
        url: str,
        on_connect: Optional[List[Dict]] = None,
    ) -> None:
        """
        Starts all WebSocket connections in the pool.

        Parameters
        ----------
        url : str
            The WebSocket URL to connect to.

        on_connect : Optional[List[Dict]], optional
            List of payloads to send upon connecting (default is None).
        """
        self._started = True

        await asyncio.gather(*[self._start_new_conn(url, on_connect) for _ in range(self.size)])

        self._eviction_task = asyncio.create_task(
            self._enforce_eviction_policy(url, on_connect)
        )

    def send_data(self, payload: Dict) -> None:
        """
        Sends a payload through all WebSocket connections in the pool.

        Parameters
        ----------
        payload : Dict
            The payload to send.
        """
        for conn in self._conn_pool:
            conn.send_data(payload)

    def shutdown(self) -> None:
        """
        Shuts down all WebSocket connections and stops the eviction task.
        """
        self._started = False

        if self._eviction_task:
            self._eviction_task.cancel()

        for conn in self._conn_pool:
            conn.close()

        for _, ingress_task in self._conn_ingress_tasks.items():
            ingress_task.cancel()

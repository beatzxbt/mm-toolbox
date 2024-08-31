import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional

from mm_toolbox.logging import Logger

from .conn import (
    PayloadData, 
    SingleWsConnection 
)

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

    Attributes
    ----------
    size : int
        Number of WebSocket connections in the pool.

    logger : Logger
        Providing extensive logging functionalities, strongly recommended. 

    eviction_policy : WsPoolEvictionPolicy
        Policy for evicting connections.
    """

    def __init__(
        self,
        size: int,
        logger: Logger,
        eviction_policy: Optional[WsPoolEvictionPolicy]=None,
    ) -> None:
        self._size = size
        self.logger = logger
        self._eviction_policy = eviction_policy if eviction_policy else WsPoolEvictionPolicy()
        
        self._started: bool = False
        self._latest_seq_id: int = 0
        self._conn_pool: List[SingleWsConnection] = []
        self._conn_speed_penalties: Dict[int, int] = {}
        self._conn_ingress_tasks: Dict[int, asyncio.Task] = {}
        self._eviction_task: asyncio.Task = None
    
    @abstractmethod
    def data_handler(self, recv: PayloadData) -> None:
        """
        Processes the recieved data from the websocket and 
        maps it to its respective handlers, or directly handles
        it. Example implementation found in /websocket/stream/.
        """
        pass
    
    async def _ingest_data(self, conn: SingleWsConnection) -> None:
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
                pass
            
            except asyncio.CancelledError:
                # Flush all remaining objects and shutdown.
                await conn.queue.join()
                return

            except Exception as e:
                self.logger.warning(f"Conn {conn_id} queue ingress - {e}")

    async def _enforce_eviction_policy(self, url: str, on_connect: Optional[List[Dict]] = None) -> None:
        """
        Periodically checks connections in the pool. Kicks out the slowest connection
        and replaces it with a fresh one. 
        """
        while True:
            await asyncio.sleep(self._eviction_policy.interval_s)

            if not self._conn_speed_penalties:
                continue

            # Find the connection with the highest penalty.
            slowest_conn_id = max(self._conn_speed_penalties, key=self._conn_speed_penalties.get)
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
                self.logger.info(f"Connection evicted: {slowest_conn_id}")
                
                # Reset all the connection's sequence ids.
                for conn in self._conn_pool:
                    conn.reset_seq_id()

                # Create and add a new connection.
                new_conn = SingleWsConnection(logger=self.logger)
                await new_conn.start(url, on_connect)
                self._conn_pool.append(new_conn)
                self._conn_speed_penalties[new_conn.conn_id] = 0
                await self.logger.info(f"New connection added: {new_conn.conn_id}")
    
    async def start(
        self,
        url: str,
        on_connect: Optional[List[Dict]] = None,
    ) -> None:
        """
        Start all WebSocket connections in the pool.

        Parameters
        ----------
        url : str
            The websocket url.

        on_connect : Optional[List[Dict]], optional
            A list of payloads to send upon connecting (default is None).
        """
        async def start_new_conn():
            new_conn = SingleWsConnection(logger=self.logger)
            await new_conn.start(
                url=url,
                on_connect=on_connect,
            )
            self._conn_pool.append(new_conn)
            self._conn_speed_penalties[new_conn.conn_id] = new_conn.seq_id
            self._conn_ingress_tasks[new_conn.conn_id] = asyncio.create_task(self._ingest_data(new_conn))
        
        self._started = True

        await asyncio.gather(*[
            start_new_conn() 
            for _ in range(self._size)
        ])

        self._eviction_task = asyncio.create_task(self._enforce_eviction_policy(url, on_connect))

    def send_data(self, payload: Dict) -> None:
        """
        Send a payload through all WebSocket conn's in the pool.

        Parameters
        ----------
        payload : Dict
            The payload to send.
        """
        for conn in self._conn_pool:
            conn.send_data(payload)

    def shutdown(self) -> None:
        """
        Stop the eviction task and shut down all WebSocket conn's in the pool.
        """
        self._started = False

        if self._eviction_task:
            self._eviction_task.cancel()

        for conn in self._conn_pool:
            conn.close()

        for _, ingress_task in self._conn_ingress_tasks.items():
            ingress_task.cancel()
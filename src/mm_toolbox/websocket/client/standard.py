import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from mm_toolbox.logging import Logger

from .conn import PayloadData, SingleWsConnection


class WsStandard(ABC):
    """
    Small wrapper around /client/conn.py for ease of use.
    """

    def __init__(
        self,
        logger: Logger,
    ) -> None:
        self.logger = logger

        self._started: bool = False
        self._conn: SingleWsConnection = None
        self._conn_ingress_task: asyncio.Task = None

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
                # Ignore seq_id field for solo streams.
                (_, time, payload) = await conn.queue.get()

                self.data_handler(payload)

            except asyncio.CancelledError:
                return

            except Exception as e:
                self.logger.warning(f"Conn {conn_id} queue ingress - {e}")

    async def _start_new_conn(
        self, url: str, on_connect: Optional[List[Dict]] = None
    ) -> None:
        """
        Establishes a new WebSocket connection, adds it to the connection pool,
        and begins data ingestion.

        Parameters
        ----------
        url : str
            The WebSocket URL to connect to.

        on_connect : Optional[List[Dict]], optional
            List of payloads to send upon connecting (default is None).
        """
        new_conn = SingleWsConnection(logger=self.logger)
        await new_conn.start(
            url=url,
            on_connect=on_connect,
        )
        self._conn = new_conn
        self._conn_ingress_task = asyncio.create_task(self._ingest_data(new_conn))

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
        await self._start_new_conn(url, on_connect)

    def send_data(self, payload: Dict) -> None:
        """
        Sends a payload through all WebSocket connections in the pool.

        Parameters
        ----------
        payload : Dict
            The payload to send.
        """
        if self._started:
            self._conn.send_data(payload)

    def shutdown(self) -> None:
        """
        Shuts down all WebSocket connections and stops the eviction task.
        """
        if not self._started:
            return

        self._conn.close()
        self._conn_ingress_task.cancel()

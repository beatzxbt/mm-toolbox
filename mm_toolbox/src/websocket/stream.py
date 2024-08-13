import orjson
import asyncio
from warnings import warn as warning
from dataclasses import dataclass
from aiohttp import ClientSession, ClientWebSocketResponse, WSMsgType
from typing import List, Dict, Callable, Any, Optional

from mm_toolbox.src.logging import Logger
from mm_toolbox.src.time import time_ms
from mm_toolbox.src.ringbuffer import RingBufferSingleDimFloat

@dataclass
class WsConnectionEvictionPolicy:
    """
    Defines the eviction policy for WebSocket connections.

    Attributes
    ----------
    max_errors : int
        Maximum number of errors allowed before eviction.
        
    max_avg_latency_ms : float
        Maximum average latency (in milliseconds) allowed before eviction.
        
    interval_s : float
        Interval (in seconds) to check the eviction conditions.
    """
    max_errors: int = 20
    max_avg_latency_ms: float = 1000.0
    interval_s: float = 60.0


class SingleWsConnection:
    """
    Manages a single WebSocket connection.

    Attributes
    ----------
    logger : Logger
        Logger instance for logging messages.

    session : ClientSession
        aiohttp ClientSession used for WebSocket connections.

    task : Optional[asyncio.Task]
        Asynchronous task managing the connection.

    ws_conn : Optional[ClientWebSocketResponse]
        The active WebSocket connection.

    latencies : RingBufferSingleDimFloat
        Buffer storing the most recent latencies.

    error_count : int
        Counter for tracking errors.
    """
    successful_msg = {WSMsgType.TEXT, WSMsgType.BINARY}
    failed_msg = {WSMsgType.CLOSED, WSMsgType.ERROR}
    fallback_timestamp_keys = ["time", "timestamp", "T", "ts"]

    def __init__(self, logger: Logger) -> None:
        self.logging = logger
        self.session = ClientSession()
        self.task: Optional[asyncio.Task] = None
        self.ws_conn: Optional[ClientWebSocketResponse] = None
        self.latencies = RingBufferSingleDimFloat(capacity=100_000)
        self.error_count = 0

    def get_mean_latency(self) -> float:
        """
        Calculate the mean latency over the recent period.

        Returns
        -------
        float
            The mean latency.
        """
        return sum(self.latencies) / len(self.latencies)

    def create_latency_handler_map(self, timestamp_keys: List[str]) -> Dict[str, Callable]:
        """
        Create a handler map for latency calculation.

        Parameters
        ----------
        timestamp_keys : List[str]
            List of timestamp keys to search for in the message data.

        Returns
        -------
        Dict[str, Callable]
            A mapping of timestamp keys to their corresponding handlers.
        """
        return {
            key: self.latencies.appendright
            for key in timestamp_keys + self.fallback_timestamp_keys
        }
    
    def record_latency(self, all_timestamp_keys: List[str], data: Dict[str, Any]) -> None:
        """
        Record the latency for the WebSocket message.

        Parameters
        ----------
        all_timestamp_keys : List[str]
            A list of possible keys for the timestamp.

        data : Dict[str, Any]
            The WebSocket message data.

        Raises
        ------
        KeyError
            If no valid timestamp key is found in the data.
        """
        for key in all_timestamp_keys:
            if key in data:
                self.latencies.appendright(time_ms() - float(data[key]))
                return
        
        raise KeyError(f"Timestamp key not found in data - {data}")

    async def send(self, payload: Dict) -> None:
        """
        Send a payload through the WebSocket connection.

        Parameters
        ----------
        payload : Dict
            The payload to send.
        """
        try:
            await self.ws_conn.send_bytes(orjson.dumps(payload))
        
        except orjson.JSONEncodeError as e:
            await self.logging.warning(
                topic="WS",
                msg=f"Failed to encode payload: {payload} - {e}"
            )

        except Exception as e:
            await self.logging.error(
                topic="WS",
                msg=f"Failed to send WebSocket payload: {payload} - {e}",
            )

    async def start(self, url: str, data_handler: Callable, timestamp_keys: List[str], on_connect: Optional[List[Dict]] = None) -> None:
        """
        Start the WebSocket connection and handle incoming messages.

        Parameters
        ----------
        url : str
            The URL of the WebSocket endpoint.

        data_handler : Callable
            A function to handle incoming messages.

        timestamp_keys : List[str]
            A list of possible keys for indexing the timestamp.

        on_connect : Optional[List[Dict]], optional
            A list of payloads to send upon connecting (default is None).
        """
        try:
            await self.logging.debug(
                topic="WS", msg=f"Starting stream on '{url}'."
            )
            
            # Combine primary and fallback timestamp keys
            all_timestamp_keys = timestamp_keys + list(self.fallback_timestamp_keys)

            async with self.session.ws_connect(url) as ws:
                self.ws_conn = ws  

                if on_connect:
                    for payload in on_connect:
                        await self.send(payload)

                async for msg in ws:
                    if msg.type in self.successful_msg:
                        try:
                            data = orjson.loads(msg.data)
                            data_handler(data)
                            self.record_latency(all_timestamp_keys, data)

                        except orjson.JSONDecodeError:
                            await self.logging.warning(
                                topic="WS", msg=f"Failed to decode payload: {msg.data}"
                            )

        except asyncio.CancelledError:
            # Session killed outside of context
            return
        
        except Exception as e:
            await self.logging.error(topic="WS", msg=f"'{url}': {e}")
    
    async def close(self) -> None:
        """
        Close the WebSocket connection and cancel any ongoing tasks.
        """
        if self.task:
            self.task.cancel()
        if self.ws_conn:
            await self.ws_conn.close()
        await self.session.close()


class WsPool:
    """
    Manages a pool of WebSocket connections.

    Attributes
    ----------
    size : int
        Number of WebSocket connections in the pool.

    url : str
        The URL of the WebSocket endpoint.

    eviction_policy : WsConnectionEvictionPolicy
        Policy for evicting connections based on errors or latency.

    conn_pool : List[SingleWsConnection]
        List of active WebSocket connections.
    """
    def __init__(
        self, 
        size: int, 
        url: str, 
        eviction_policy: WsConnectionEvictionPolicy,
        logger: Logger
    ) -> None:
        self.size = size
        self.url = url
        self.eviction_policy = eviction_policy
        self.logging = logger
        self.conn_pool = [SingleWsConnection(logger=self.logging) for _ in range(size)]
        self.eviction_task: Optional[asyncio.Task] = None

    async def start(self, data_handler: Callable, timestamp_keys: List[str], on_connect: Optional[List[Dict]] = None) -> None:
        """
        Start all WebSocket connections in the pool.

        Parameters
        ----------
        data_handler : Callable
            A function to handle incoming messages.

        timestamp_keys : List[str]
            A list of possible keys for indexing the timestamp.

        on_connect : Optional[List[Dict]], optional
            A list of payloads to send upon connecting (default is None).
        """
        for conn in self.conn_pool:
            await conn.start(
                url=self.url,
                data_handler=data_handler,
                timestamp_keys=timestamp_keys,
                on_connect=on_connect
            )

        self.eviction_task = asyncio.create_task(self.enforce_eviction_policy())

    async def enforce_eviction_policy(self) -> None:
        """
        Periodically checks connections in the pool and enforces the eviction policy.
        """
        while True:
            await asyncio.sleep(self.eviction_policy.interval_s)

            for conn in list(self.conn_pool):
                if self.should_evict(conn):
                    await self.evict_connection(conn)
                    await self.replace_connection(conn)

    def should_evict(self, conn: SingleWsConnection) -> bool:
        """
        Determine if a connection should be evicted based on the eviction policy.

        Parameters
        ----------
        conn : SingleWsConnection
            The WebSocket connection to check.

        Returns
        -------
        bool
            True if the connection should be evicted, False otherwise.
        """
        avg_latency = conn.get_mean_latency()
        if conn.error_count > self.eviction_policy.max_errors:
            return True
        if avg_latency > self.eviction_policy.max_avg_latency_ms:
            return True
        return False

    async def evict_connection(self, conn: SingleWsConnection) -> None:
        """
        Evict a connection from the pool.

        Parameters
        ----------
        conn : SingleWsConnection
            The WebSocket connection to evict.
        """
        await conn.close()
        self.conn_pool.remove(conn)
        await self.logging.info(
            topic="WS",
            msg=f"Connection evicted: AvgLatency={conn.get_mean_latency()}, Errors={conn.error_count}"
        )

    async def replace_connection(self, conn: SingleWsConnection) -> None:
        """
        Replace an evicted connection with a new one.

        Parameters
        ----------
        conn : SingleWsConnection
            The evicted WebSocket connection.
        """
        new_conn = SingleWsConnection(logger=self.logging)
        await new_conn.start(
            url=self.url,
            data_handler=conn.data_handler,
            timestamp_keys=conn.timestamp_keys,
            on_connect=conn.on_connect
        )
        self.conn_pool.append(new_conn)

    async def send(self, payload: Dict) -> None:
        """
        Send a payload through all WebSocket connections in the pool.

        Parameters
        ----------
        payload : Dict
            The payload to send.
        """
        for conn in self.conn_pool:
            await conn.send(payload)

    async def shutdown(self) -> None:
        """
        Shut down all WebSocket connections in the pool and stop the eviction task.
        """
        if self.eviction_task:
            self.eviction_task.cancel()

        for conn in self.conn_pool:
            await conn.close()


class FastWebsocketStream:
    """
    High-performance WebSocket stream management using a connection pool.

    Attributes
    ----------
    url : str
        The URL of the WebSocket endpoint.

    on_connect : List[Dict], optional
        A list of payloads to send upon connecting (default is None).

    num_streams : int
        Number of WebSocket streams to manage in the pool.

    eviction_policy : WsConnectionEvictionPolicy
        Policy for evicting connections based on errors or latency.

    logging : Logger
        Logger instance for logging messages.

    ws_pool : WsPool
        The pool of WebSocket connections.
    """
    def __init__(
        self, 
        url: str, 
        on_connect: Optional[List[Dict]] = None, 
        num_streams: int = 5, 
        eviction_policy: Optional[WsConnectionEvictionPolicy] = None,
        logger: Optional[Logger] = None
    ) -> None:
        self.url = url
        self.on_connect = on_connect if on_connect else []
        self.eviction_policy = eviction_policy if eviction_policy else WsConnectionEvictionPolicy()
        self.num_streams = max(num_streams, 2)

        if logger is None:
            warning(
                message="Custom loggers are strongly recommended for full functionality.",
                category=RuntimeWarning
            )
            logger = Logger()
        self.logging = logger

        self.ws_pool = WsPool(
            size=self.num_streams,
            url=self.url,
            eviction_policy=self.eviction_policy,
            logger=self.logging
        )
    
    async def start(self, data_handler: Callable, timestamp_keys: List[str]) -> None:
        """
        Start the WebSocket streams within the pool.

        Parameters
        ----------
        data_handler : Callable
            A function to handle incoming messages.

        timestamp_keys : List[str]
            A list of possible keys for indexing the timestamp.
        """
        await self.ws_pool.start(
            data_handler=data_handler,
            timestamp_keys=timestamp_keys,
            on_connect=self.on_connect
        )

    async def send(self, payload: Dict) -> None:
        """
        Send a payload through all WebSocket connections in the pool.

        Parameters
        ----------
        payload : Dict
            The payload to send.
        """
        await self.ws_pool.send(payload)

    async def shutdown(self) -> None:
        """
        Shut down all WebSocket connections in the pool.
        """
        await self.ws_pool.shutdown()

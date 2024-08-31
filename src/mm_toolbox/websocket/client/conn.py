import asyncio
import orjson
import numpy as np
from picows import ws_connect, WSFrame, WSTransport, WSListener, WSMsgType, WSCloseCode
from typing import Tuple, List, Dict, Union, Callable, Optional

from mm_toolbox.logging import Logger
from mm_toolbox.time import time_ns

RawWsPayload = bytearray
PayloadData = Dict[str, Union[int, float, str, Dict, List]]
QueuePayload = Tuple[int, int, PayloadData]

class RawWsConnection(WSListener):
    """
    Wrapper of WSListener class for PicoWs use.

    Roughly following https://github.com/tarasko/picows/blob/master/examples/echo_client_cython.pyx.

    Attributes
    ----------
    transport : WSTransport
        Link to underlying API for sending/recieving messages from buffer.

    process_frame: Callable
        Ingesting final bytearray payload. Must take 3 args, process_frame(seq_id, time, payload).
        Payload's memory is shared with the underlying buffer and must be copied in the process.

    final_frame: RawWsPayload
        A bytearray holding the final websocket frame. May hold partial frames if underlying 
        buffer yields one.
    """
    def __init__(self, process_frame):
        super().__init__()
        self.transport: WSTransport = None # For external access
        self.process_frame: Callable = process_frame
        
        self.final_frame: RawWsPayload = bytearray()

    def on_ws_connected(self, transport: WSTransport):
        self.transport = transport 

    def on_ws_frame(self, transport: WSTransport, frame: WSFrame):
        self.final_frame += frame.get_payload_as_memoryview()

        if frame.fin:
            self.process_frame(time_ns(), self.final_frame)
            self.final_frame.clear()

    # Not used, but available for future impl
    def on_ws_disconnected(self, transport: WSTransport): pass 
    def pause_writing(self): pass
    def resume_writing(self): pass


class SingleWsConnection:
    """
    Manages a single WebSocket connection.
    """

    def __init__(self, logger: Logger) -> None:
        self.logger = logger
        self.running: bool = False

        self._ws_client: WSListener = None
        self._seq_id: int = 0
        self._conn_id = np.random.randint(low=np.iinfo(np.uint16).min, high=np.iinfo(np.uint16).max)
        self._queue = asyncio.Queue(maxsize=1_000_000)

    @property
    def seq_id(self) -> int:
        return self._seq_id
    
    @property
    def conn_id(self) -> int:
        return self._conn_id
    
    @property
    def queue(self) -> asyncio.Queue:
        return self._queue

    def _process_frame(self, time: int, final_frame: RawWsPayload) -> None:
        """
        Take a frame object, deserialize it into a dict object.
        Then package it into a known format and insert into queue.
        """
        try:
            self._seq_id += 1
            q_payload: QueuePayload = (self._seq_id, time, orjson.loads(final_frame))
            self._queue.put_nowait(q_payload)
        
        except orjson.JSONDecodeError:
            self.logger.warning(f"Unable to decode msg {self._seq_id}: {final_frame.decode()}") 
            pass

        except asyncio.QueueFull:
            self.logger.warning(f"Conn '{self._conn_id}' queue full, skipping msg {self._seq_id}.")
            pass
        
    def reset_seq_id(self) -> None:
        """Set sequence id back to 0"""
        self._seq_id = 0
        
    def send_data(self, payload: Dict) -> None:
        """
        Send a payload through the WebSocket connection.

        Parameters
        ----------
        payload : Dict
            The payload to send.
        """
        try:
            if self._ws_client.transport is not None:
                self._ws_client.transport.send(WSMsgType.TEXT, orjson.dumps(payload))
            else:
                raise ConnectionError(f"Conn '{self._conn_id}' not started yet.")
            
        except orjson.JSONEncodeError as e:
            self.logger.warning(f"Conn '{self._conn_id}' failed to encode: {e}")

        except Exception as e:
            self.logger.warning(f"Conn '{self._conn_id}' failed to send: {e}")

    async def start(
        self,
        url: str,
        on_connect: Optional[List[Dict]] = None,
    ) -> None:
        """
        Start the WebSocket connection and handle incoming messages.

        Parameters
        ----------
        url : str
            The URL of the WebSocket endpoint.

        on_connect : Optional[List[Dict]], optional
            A list of payloads to send upon connecting (default is None).
        """
        try:
            self.running = True

            self.logger.info(f"Conn '{self._conn_id}' starting on '{url}'.")

            # (WSTransport, WSListener)
            (_, self._ws_client) = await ws_connect(lambda: RawWsConnection(self._process_frame), url)

            # Dummy empty list incase none provided, never iterates.
            on_connect = on_connect if on_connect else []

            for payload in on_connect:
                self.send_data(payload)

            self._conn_task = asyncio.create_task(self._ws_client.transport.wait_disconnected())

        except Exception as e:
            self.logger.error(f"Conn '{self._conn_id}': {e}")

    def close(self) -> None:
        """
        Close the WebSocket connection and cancel any ongoing tasks.
        """
        self.running = False

        if self._ws_client is not None:
            self._ws_client.transport.send_close(WSCloseCode.OK)
            self._ws_client.transport.disconnect()

        if self._conn_task is not None:
            self._conn_task.cancel()
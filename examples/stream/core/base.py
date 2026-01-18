"""Abstract stream processor for multi-venue streaming.

Provides shared WebSocket connection management, IPC output, and
template methods for decoding and normalizing venue-specific payloads.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import msgspec

from mm_toolbox.logging.advanced import LogLevel, LoggerConfig, WorkerLogger
from mm_toolbox.ringbuffer.ipc import IPCRingBufferConfig, IPCRingBufferProducer
from mm_toolbox.websocket import WsConnectionConfig, WsPool, WsPoolConfig

from .models import BBOUpdate, MsgType, OrderbookMsg, StreamMessage, TradeMsg, Venue


class BaseStreamProcessor(ABC):
    """Abstract base class for venue stream processors."""

    def __init__(self, venue: Venue, symbol: str, ipc_path: str, logger_path: str) -> None:
        """Initialize the stream processor.

        Args:
            venue: Venue name.
            symbol: Venue symbol.
            ipc_path: IPC path for outgoing normalized messages.
            logger_path: IPC path for worker logs.
        """
        self.venue = venue
        self.symbol = symbol
        self._encoder = msgspec.json.Encoder()

        logger_config = LoggerConfig(
            base_level=LogLevel.INFO,
            str_format="%(asctime)s [%(levelname)s] %(message)s",
            path=logger_path,
            flush_interval_s=0.5,
            emit_internal=False,
        )
        self._logger = WorkerLogger(config=logger_config, name="stream")

        self._ipc_producer = IPCRingBufferProducer(
            IPCRingBufferConfig(
                path=ipc_path,
                backlog=10000,
                num_producers=1,
                num_consumers=1,
                linger_ms=0,
            )
        )

        self._bbo_pool: WsPool | None = None
        self._trade_pool: WsPool | None = None
        self._orderbook_pool: WsPool | None = None

    @abstractmethod
    def get_stream_url(self, msg_type: MsgType) -> str:
        """Return the WebSocket URL for a message type.

        Args:
            msg_type: Message type for the stream.

        Returns:
            str: WebSocket URL or empty string if unsupported.
        """

    def get_subscribe_messages(self, msg_type: MsgType) -> list[bytes]:
        """Return subscription messages for a message type.

        Args:
            msg_type: Message type for the stream.

        Returns:
            list[bytes]: Subscription payloads.
        """
        return []

    @abstractmethod
    def parse_bbo(self, msg: bytes) -> BBOUpdate | None:
        """Parse a raw BBO message into a normalized update.

        Args:
            msg: Raw WebSocket message.

        Returns:
            BBOUpdate | None: Normalized update when available.
        """

    @abstractmethod
    def parse_trade(self, msg: bytes) -> TradeMsg | None:
        """Parse a raw trade message into a normalized trade message.

        Args:
            msg: Raw WebSocket message.

        Returns:
            TradeMsg | None: Normalized trade message when available.
        """

    @abstractmethod
    def parse_orderbook(self, msg: bytes) -> OrderbookMsg | None:
        """Parse a raw orderbook message into a normalized update.

        Args:
            msg: Raw WebSocket message.

        Returns:
            OrderbookMsg | None: Normalized update when available.
        """

    def _noop(self, msg: bytes) -> None:
        """Ignore callback-based messages from the WebSocket pool.

        Args:
            msg: Raw WebSocket message.
        """
        return

    def _send_message(self, msg: StreamMessage) -> None:
        """Send a normalized message via IPC.

        Args:
            msg: Normalized stream message.
        """
        self._ipc_producer.insert(self._encoder.encode(msg), copy=False)

    def _send_parsed_message(
        self, msg_type: MsgType, payload: BBOUpdate | TradeMsg | OrderbookMsg
    ) -> None:
        """Wrap and send a normalized payload.

        Args:
            msg_type: Message type enum.
            payload: Normalized payload.
        """
        stream_msg = StreamMessage(msg_type=msg_type, venue=self.venue, data=payload)
        self._send_message(stream_msg)

    def _send_parsed_messages(
        self,
        msg_type: MsgType,
        payloads: BBOUpdate | TradeMsg | OrderbookMsg | None,
    ) -> None:
        """Send normalized payloads, handling optional lists.

        Args:
            msg_type: Message type enum.
        payloads: Normalized payload.
        """
        if payloads is None:
            return
        self._send_parsed_message(msg_type, payloads)

    async def _run_stream(self, msg_type: MsgType) -> None:
        """Run a WebSocket stream for the requested message type.

        Args:
            msg_type: Message type for the stream.
        """
        url = self.get_stream_url(msg_type)
        if not url:
            return
        on_connect = self.get_subscribe_messages(msg_type)
        config = WsConnectionConfig.default(
            wss_url=url, on_connect=on_connect if on_connect else None
        )
        pool = await WsPool.new(
            config=config,
            on_message=self._noop,
            pool_config=WsPoolConfig.default(),
        )
        if msg_type == MsgType.BBO:
            self._bbo_pool = pool
        elif msg_type == MsgType.TRADE:
            self._trade_pool = pool
        else:
            self._orderbook_pool = pool

        async with pool:
            self._logger.info(f"[{self.venue}] {msg_type.name} stream connected")
            async for msg in pool:
                try:
                    match msg_type:
                        case MsgType.BBO:
                            parsed = self.parse_bbo(msg)
                            self._send_parsed_messages(MsgType.BBO, parsed)
                        case MsgType.TRADE:
                            parsed = self.parse_trade(msg)
                            self._send_parsed_messages(MsgType.TRADE, parsed)
                        case MsgType.ORDERBOOK:
                            parsed = self.parse_orderbook(msg)
                            if parsed is not None:
                                self._send_parsed_messages(MsgType.ORDERBOOK, parsed)
                except Exception as exc:
                    self._logger.error(
                        f"[{self.venue}] {msg_type.name} parse error: {exc}"
                    )

    async def _run_streams(self) -> None:
        """Run all configured streams for this venue."""
        tasks = []
        for msg_type in (MsgType.ORDERBOOK, MsgType.BBO, MsgType.TRADE):
            if self.get_stream_url(msg_type):
                tasks.append(asyncio.create_task(self._run_stream(msg_type)))
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)

    def run(self) -> None:
        """Run the stream processor."""
        try:
            asyncio.run(self._run_streams())
        except KeyboardInterrupt:
            self._logger.info(f"[{self.venue}] Stream interrupted")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Shutdown the stream processor."""
        if self._bbo_pool is not None:
            self._bbo_pool.close()
        if self._trade_pool is not None:
            self._trade_pool.close()
        if self._orderbook_pool is not None:
            self._orderbook_pool.close()
        self._ipc_producer.stop()
        self._logger.shutdown()

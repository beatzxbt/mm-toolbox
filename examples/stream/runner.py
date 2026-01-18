"""Process entry points for the multi-venue streaming example.

Provides importable stream process targets for multiprocessing spawn
and a factory to select per-venue stream processors.
"""

from __future__ import annotations

from examples.stream.binance import BinanceStreamProcessor
from examples.stream.bybit import BybitStreamProcessor
from examples.stream.hyperliquid import HyperliquidStreamProcessor
from examples.stream.lighter import LighterStreamProcessor
from examples.stream.okx import OkxStreamProcessor
from examples.stream.core.base import BaseStreamProcessor


def get_stream_processor(
    venue: str, symbol: str, ipc_path: str, logger_path: str
) -> BaseStreamProcessor:
    """Build the appropriate stream processor for a venue.

    Args:
        venue: Venue name.
        symbol: Venue symbol.
        ipc_path: IPC path for outgoing normalized messages.
        logger_path: IPC path for worker logs.

    Returns:
        BaseStreamProcessor: Stream processor instance.
    """
    if venue == "binance":
        return BinanceStreamProcessor(symbol=symbol, ipc_path=ipc_path, logger_path=logger_path)
    if venue == "bybit":
        return BybitStreamProcessor(symbol=symbol, ipc_path=ipc_path, logger_path=logger_path)
    if venue == "okx":
        return OkxStreamProcessor(symbol=symbol, ipc_path=ipc_path, logger_path=logger_path)
    if venue == "hyperliquid":
        return HyperliquidStreamProcessor(symbol=symbol, ipc_path=ipc_path, logger_path=logger_path)
    if venue == "lighter":
        return LighterStreamProcessor(symbol=symbol, ipc_path=ipc_path, logger_path=logger_path)
    raise ValueError(f"Unsupported venue: {venue}")


def stream_process_entry(venue: str, symbol: str, ipc_path: str, logger_path: str) -> None:
    """Entry point for a venue stream process.

    Args:
        venue: Venue name.
        symbol: Venue symbol.
        ipc_path: IPC path for outgoing normalized messages.
        logger_path: IPC path for worker logs.
    """
    processor = get_stream_processor(venue, symbol, ipc_path, logger_path)
    processor.run()

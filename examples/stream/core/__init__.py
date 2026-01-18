"""Core primitives for the multi-venue streaming example.

Includes normalized message models, base stream processor abstractions,
and statistics helpers for processing pipelines.
"""

from __future__ import annotations

__all__ = [
    "BaseStreamProcessor",
    "BBOUpdate",
    "CoreMsg",
    "MsgType",
    "OrderbookLevel",
    "OrderbookMsg",
    "StreamMessage",
    "Trade",
    "TradeMsg",
    "Venue",
]

from .base import BaseStreamProcessor
from .models import (
    BBOUpdate,
    CoreMsg,
    MsgType,
    OrderbookLevel,
    OrderbookMsg,
    StreamMessage,
    Trade,
    TradeMsg,
    Venue,
)

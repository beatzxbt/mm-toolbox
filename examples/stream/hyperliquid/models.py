"""Wire-format models for Hyperliquid WebSocket payloads.

Provides a flexible msgspec structure for decoding Hyperliquid channel
messages used by the stream processor.
"""

from __future__ import annotations

from typing import Any

import msgspec


class HyperliquidMessage(msgspec.Struct):
    """Generic Hyperliquid WebSocket message.

    Attributes:
        channel: Channel name when present.
        data: Payload data for the channel.
        type: Optional message type indicator.
    """

    channel: str | None = None
    data: dict[str, Any] | list[Any] | None = None
    type: str | None = None

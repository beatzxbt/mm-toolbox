"""Wire-format models for Lighter WebSocket payloads.

Provides a flexible msgspec structure for decoding Lighter channel
messages used by the stream processor.
"""

from __future__ import annotations

from typing import Any

import msgspec


class LighterMessage(msgspec.Struct):
    """Generic Lighter WebSocket message.

    Attributes:
        channel: Channel name when present.
        type: Message type when present.
        data: Payload data for the channel.
    """

    channel: str | None = None
    type: str | None = None
    data: dict[str, Any] | list[Any] | None = None

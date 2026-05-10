"""Configuration forwarding tests for WsConnection.

Layer-1 tests verifying that WsConnection.new correctly passes
config fields (e.g., max_frame_size) into the underlying ws_connect call.
This ensures user-supplied limits are respected by the transport layer.
"""

from __future__ import annotations

import pytest

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.websocket.connection import WsConnection, WsConnectionConfig


class TestWsConnectionConfigForwarding:
    """Layer-1 tests for config field forwarding from WsConnection to ws_connect."""

    async def test_new_forwards_max_frame_size(self, monkeypatch) -> None:
        """Given a custom max_frame_size in config, When WsConnection.new is called, Then that value is passed to ws_connect."""
        captured: dict[str, object] = {}

        async def fake_ws_connect(
            *,
            ws_listener_factory,
            url: str,
            max_frame_size: int,
        ):
            captured["ws_listener_factory"] = ws_listener_factory
            captured["url"] = url
            captured["max_frame_size"] = max_frame_size
            return object(), "listener-sentinel"

        monkeypatch.setattr(
            "mm_toolbox.websocket.connection.ws_connect", fake_ws_connect
        )

        ringbuffer = BytesRingBuffer(max_capacity=8, only_insert_unique=False)
        config = WsConnectionConfig.default(
            "wss://test.com/ws",
            max_frame_size=2048,
        )
        listener = await WsConnection.new(ringbuffer, config)

        assert listener == "listener-sentinel"
        assert callable(captured["ws_listener_factory"])
        assert captured["url"] == "wss://test.com/ws"
        assert captured["max_frame_size"] == 2048

"""Configuration forwarding tests for WsConnection.

Layer-1 tests verifying that WsConnection.new correctly passes
config fields (e.g., max_frame_size) into the underlying ws_connect call.
This ensures user-supplied limits are respected by the transport layer.
"""

from __future__ import annotations

import pytest
import sys

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.websocket.connection import WsConnection, WsConnectionConfig
from mm_toolbox.websocket import WsSingle


def test_default_receive_limits_are_bounded_and_configurable():
    config = WsConnectionConfig.default("ws://localhost")
    assert config.receive_buffer_capacity == 4096
    assert config.receive_buffer_max_bytes == 16 * 1024 * 1024
    custom = WsConnectionConfig.default(
        "ws://localhost",
        receive_buffer_capacity=512,
        receive_buffer_max_bytes=2 * 1024 * 1024,
    )
    assert custom.receive_buffer_capacity == 512
    assert custom.receive_buffer_max_bytes == 2 * 1024 * 1024


@pytest.mark.parametrize(
    "options",
    [
        {"receive_buffer_capacity": 0},
        {"receive_buffer_capacity": -1},
        {"receive_buffer_capacity": 1.5},
        {"receive_buffer_capacity": True},
        {"receive_buffer_capacity": False},
        {"receive_buffer_capacity": 2**31},
        {"receive_buffer_max_bytes": 0},
        {"receive_buffer_max_bytes": 1024},
        {"receive_buffer_max_bytes": sys.maxsize + 1},
        {"max_frame_size": 1, "receive_buffer_max_bytes": True},
        {"max_frame_size": 1, "receive_buffer_max_bytes": False},
    ],
)
def test_invalid_receive_limits_are_rejected(options):
    with pytest.raises(ValueError, match="receive_buffer"):
        WsConnectionConfig.default("ws://localhost", **options)


def test_huge_valid_byte_budget_does_not_overflow_watermarks():
    config = WsConnectionConfig.default(
        "ws://localhost", receive_buffer_max_bytes=sys.maxsize
    )
    client = WsSingle(config)
    try:
        client._ringbuffer.insert(b"small")
        assert not client._ringbuffer.is_reading_paused()
        assert client._ringbuffer.consume() == b"small"
    finally:
        client.close()


class TestWsConnectionConfigForwarding:
    """Layer-1 tests for config field forwarding from WsConnection to ws_connect."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("url", ["ws://test.com/ws", "wss://test.com/ws"])
    async def test_new_forwards_max_frame_size(self, monkeypatch, url) -> None:
        """Given a custom max_frame_size in config, When WsConnection.new is called, Then that value is passed to ws_connect."""
        captured: dict[str, object] = {}

        async def fake_ws_connect(
            *,
            ws_listener_factory,
            url: str,
            max_frame_size: int,
            use_aiofastnet: bool,
        ):
            captured["ws_listener_factory"] = ws_listener_factory
            captured["url"] = url
            captured["max_frame_size"] = max_frame_size
            captured["use_aiofastnet"] = use_aiofastnet

            class Listener:
                def is_connected(self):
                    return True

            listener = Listener()
            captured["listener"] = listener
            return object(), listener

        monkeypatch.setattr(
            "mm_toolbox.websocket.connection.ws_connect", fake_ws_connect
        )

        ringbuffer = BytesRingBuffer(max_capacity=8, only_insert_unique=False)
        config = WsConnectionConfig.default(
            url,
            max_frame_size=2048,
        )
        listener = await WsConnection.new(ringbuffer, config)

        assert listener is captured["listener"]
        assert callable(captured["ws_listener_factory"])
        assert captured["url"] == url
        assert captured["max_frame_size"] == 2048
        assert captured["use_aiofastnet"] is False

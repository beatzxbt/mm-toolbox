"""Wrapper to expose native Cython WsConnection tests to pytest."""

from __future__ import annotations

import sys
import os

# Go up to tests/ directory to find the compiled .so files
test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

try:
    import cython_test_websocket_connection as _native
except ImportError as e:
    import pytest

    pytest.skip(
        f"Native Cython test module not built: {e}. Run `make build-test`",
        allow_module_level=True,
    )


# Ping/Pong tracker tests


def test_ping_tracker_starts_at_zero():
    _native.test_ping_tracker_starts_at_zero()


def test_ping_timeout_resets_tracker():
    _native.test_ping_timeout_resets_tracker()


def test_spurious_pong_ignored():
    _native.test_spurious_pong_ignored()


# Fragment buffer tests


def test_unfin_buffer_cleared_on_close():
    _native.test_unfin_buffer_cleared_on_close()


def test_unfin_buffer_cleared_on_disconnect():
    _native.test_unfin_buffer_cleared_on_disconnect()


def test_unfin_buffer_cleared_on_connect():
    _native.test_unfin_buffer_cleared_on_connect()


# Transport reference tests


def test_transport_starts_none():
    _native.test_transport_starts_none()


def test_disconnect_sets_transport_none():
    _native.test_disconnect_sets_transport_none()


# Should-stop guard tests


def test_should_stop_blocks_frame_processing():
    _native.test_should_stop_blocks_frame_processing()


def test_should_stop_initially_false():
    _native.test_should_stop_initially_false()


# Flattened state tests


def test_flattened_state_initialized():
    _native.test_flattened_state_initialized()


def test_get_state_returns_enum():
    _native.test_get_state_returns_enum()


# Config cache tests


def test_config_cache_reflects_updates():
    _native.test_config_cache_reflects_updates()

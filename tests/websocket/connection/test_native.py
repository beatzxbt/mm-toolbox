"""Wrapper to expose native Cython WsConnection tests to pytest.

Layer-1 tests executed against the compiled Cython extension. These cover
internal state that is inaccessible from Python (cdef fields) including:
- Ping/pong tracker lifecycle and spurious-pong handling.
- Unfinished-fragment buffer clearing on close, disconnect, and connect.
- Transport reference management (starts None, set on connect, cleared on disconnect).
- should_stop flag semantics.
- Flattened state initialization and enum return types.
- Config cache consistency after updates.

Skipped gracefully when the native test module has not been built.
"""

from __future__ import annotations

import os
import sys

import pytest

# Go up to tests/ directory to find the compiled .so files
test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

try:
    import cython_test_websocket_connection as _native
except ImportError as e:
    pytest.skip(
        f"Native Cython test module not built: {e}. Run `make build-test`",
        allow_module_level=True,
    )


# --------------------------------------------------------------------------- #
# Ping/Pong tracker tests
# --------------------------------------------------------------------------- #


def test_ping_tracker_starts_at_zero():
    """Given a fresh connection, Then the ping tracker is initialised to zero."""
    _native.test_ping_tracker_starts_at_zero()


def test_ping_timeout_resets_tracker():
    """Given a ping timeout, Then the tracker is reset so stale pings do not accumulate."""
    _native.test_ping_timeout_resets_tracker()


def test_spurious_pong_ignored():
    """Given a PONG without a preceding PING, Then it is ignored and the tracker is unaffected."""
    _native.test_spurious_pong_ignored()


# --------------------------------------------------------------------------- #
# Fragment buffer tests
# --------------------------------------------------------------------------- #


def test_unfin_buffer_cleared_on_close():
    """Given an incomplete fragment buffer, When close() is called, Then the buffer is cleared."""
    _native.test_unfin_buffer_cleared_on_close()


def test_unfin_buffer_cleared_on_disconnect():
    """Given an incomplete fragment buffer, When disconnect occurs, Then the buffer is cleared."""
    _native.test_unfin_buffer_cleared_on_disconnect()


def test_unfin_buffer_cleared_on_connect():
    """Given an incomplete fragment buffer from a prior session, When a new connection starts, Then the buffer is cleared."""
    _native.test_unfin_buffer_cleared_on_connect()


# --------------------------------------------------------------------------- #
# Transport reference tests
# --------------------------------------------------------------------------- #


def test_transport_starts_none():
    """Given a fresh connection, Then the transport reference is None."""
    _native.test_transport_starts_none()


def test_disconnect_sets_transport_none():
    """Given a connected transport, When disconnect occurs, Then the transport reference becomes None."""
    _native.test_disconnect_sets_transport_none()


# --------------------------------------------------------------------------- #
# Should-stop guard tests
# --------------------------------------------------------------------------- #


def test_should_stop_blocks_frame_processing():
    """Given should_stop=True, When a frame arrives, Then it is not processed."""
    _native.test_should_stop_blocks_frame_processing()


def test_should_stop_initially_false():
    """Given a fresh connection, Then should_stop is False so frames are accepted."""
    _native.test_should_stop_initially_false()


# --------------------------------------------------------------------------- #
# Flattened state tests
# --------------------------------------------------------------------------- #


def test_flattened_state_initialized():
    """Given a fresh connection, Then the flattened state is initialised to DISCONNECTED."""
    _native.test_flattened_state_initialized()


def test_get_state_returns_enum():
    """Given any state, When get_state is called, Then a ConnectionState enum is returned."""
    _native.test_get_state_returns_enum()


# --------------------------------------------------------------------------- #
# Config cache tests
# --------------------------------------------------------------------------- #


def test_config_cache_reflects_updates():
    """Given a config update, When the cache is read, Then it reflects the new values."""
    _native.test_config_cache_reflects_updates()

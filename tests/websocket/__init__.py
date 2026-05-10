"""WebSocket test package.

Provides the test suite for the websocket module, organized by layer:
- connection: Low-level WsConnection primitives and state management.
- single: WsSingle wrapper for one-shot/async-iterable usage.
- pool: WsPool for multi-connection management, eviction, and deduplication.
- integration: End-to-end, realistic workflow, and live exchange tests.

Key coverage areas include connection lifecycle, frame handling,
reconnection semantics, backpressure, hash deduplication, and
graceful degradation under protocol errors or server rejection.
"""

from __future__ import annotations

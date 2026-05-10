"""WebSocket integration test package.

Layer-3 (mini-integration) tests combining WsConnection, WsSingle,
and WsPool with real local servers and optional live exchange feeds.

Covers end-to-end workflows, realistic error recovery, Binance Futures
smoke tests, and stress benchmarks (connection churn, sustained throughput).
"""

from __future__ import annotations

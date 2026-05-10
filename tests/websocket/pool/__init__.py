"""WsPool test package.

Layer-2 and Layer-3 tests for the WsPool multi-connection manager.
Covers config validation, connection lifecycle, async iteration,
hash-history deduplication, eviction logic, send routing (fastest vs
multicast), error resilience, and context-manager cleanup.

Edge cases include partial replacement failures, zero-connection pools,
ringbuffer overflow in hash history, and concurrent send/iterate paths.
"""

from __future__ import annotations

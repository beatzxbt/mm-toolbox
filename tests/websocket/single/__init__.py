"""WsSingle test package.

Layer-2 and Layer-3 tests for the WsSingle high-level wrapper.
Covers config validation, async iteration, context-manager semantics,
message callbacks, and error scenarios (refusal, timeout, mid-stream
disconnect, protocol errors, and rapid churn).

Edge cases include callback exceptions, empty payloads, fragmented
frames, concurrent callback + iteration, and auto-reconnect recovery.
"""

from __future__ import annotations

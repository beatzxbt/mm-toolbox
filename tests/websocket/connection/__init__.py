"""WsConnection test package.

Layer-1 and Layer-2 tests for the low-level WsConnection component.
Covers state machine correctness, config forwarding, send operations,
callback-driven frame handling, reconnection backoff, and native Cython
wrapper exposure.

Edge cases include fragmented frames, compressed frames, oversized
payloads, empty payloads, mid-stream disconnects, and connection rejection.
"""

from __future__ import annotations

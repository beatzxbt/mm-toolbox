"""Delivery policy shared by standard and advanced logging handlers."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from contextlib import AbstractAsyncContextManager
from email.utils import parsedate_to_datetime
import math
from typing import Final, Protocol

import aiohttp
import msgspec

from mm_toolbox.rate_limiter import (
    RateLimiter,
    RateLimiterConfig,
    RateLimitBurstConfig,
    RateLimitStateConfig,
    SubBucketStrategy,
)
from mm_toolbox.time import time_monotonic_ns, time_ns

DELIVERY_BUDGET_S: Final = 10.0
HTTP_TIMEOUT_S: Final = 3.0
SHUTDOWN_TIMEOUT_S: Final = 2.0
CLEANUP_TIMEOUT_S: Final = 0.25
MAX_PENDING_DELIVERIES: Final = 256
MAX_ATTEMPTS: Final = 3
DISCORD_CHUNK_CHARS: Final = 1800
TELEGRAM_CHUNK_CHARS: Final = 3500


class HTTPResponse(Protocol):
    @property
    def status(self) -> int: ...
    @property
    def headers(self) -> Mapping[str, str]: ...
    async def read(self) -> bytes: ...


class HTTPSession(Protocol):
    @property
    def closed(self) -> bool: ...
    def post(
        self, url: str, *, headers: dict[str, str], data: bytes
    ) -> AbstractAsyncContextManager[HTTPResponse]: ...
    async def close(self) -> None: ...


def make_rate_limiter(capacity: int, window_s: int) -> RateLimiter:
    """Use the full fixed-window allowance without warning/burst policies.

    Handlers consume only on their owned event loop. A synchronous reservation
    has no suspension point, so competing coroutines cannot spend the same slot.
    """
    return RateLimiter(
        RateLimiterConfig(
            capacity=capacity,
            window_s=window_s,
            state_config=RateLimitStateConfig(False, 0.75, 0.95),
            burst_config=RateLimitBurstConfig.default(),
            sub_bucket_strategy=SubBucketStrategy.DISABLED,
        )
    )


def chunk_messages(messages: Iterable[str], max_chars: int) -> list[str]:
    """Pack lines in order, splitting long messages without dropping characters."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for message in messages:
        for start in range(0, max(1, len(message)), max_chars):
            piece = message[start : start + max_chars]
            if current and current_len + 1 + len(piece) > max_chars:
                if packed := "\n".join(current):
                    chunks.append(packed)
                current = []
                current_len = 0
            current_len += (1 if current else 0) + len(piece)
            current.append(piece)
    if current:
        if packed := "\n".join(current):
            chunks.append(packed)
    return chunks


def _nonnegative_seconds(value: object) -> float | None:
    # Decode protocol data at this boundary; application types are checked by ty.
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        try:
            seconds = float(value)
        except (ValueError, OverflowError):
            return None
        if math.isfinite(seconds):
            return max(0.0, seconds)
    return None


def retry_after_seconds(value: str | None, body: bytes = b"") -> float | None:
    """Parse seconds or HTTP dates, retaining subsecond wall-clock precision."""
    if value:
        seconds = _nonnegative_seconds(value)
        if seconds is not None:
            return seconds
        try:
            retry_at_ns = int(parsedate_to_datetime(value).timestamp()) * 1_000_000_000
            return max(0.0, (retry_at_ns - time_ns()) / 1e9)
        except (TypeError, ValueError, OverflowError):
            pass
    try:
        payload = msgspec.json.decode(body)
    except msgspec.DecodeError:
        return None
    if isinstance(payload, dict):
        return _nonnegative_seconds(payload.get("retry_after"))
    return None


async def post_json_with_retries(
    session: HTTPSession,
    url: str,
    payload: Mapping[str, object],
    *,
    headers: dict[str, str],
    limiter: RateLimiter | None,
    service: str,
) -> None:
    """Deliver within one budget, including rate waits, requests and backoff."""
    encoded = msgspec.json.encode(payload)
    deadline = time_monotonic_ns() + int(DELIVERY_BUDGET_S * 1e9)
    for attempt in range(MAX_ATTEMPTS):
        remaining = (deadline - time_monotonic_ns()) / 1e9
        if remaining <= 0:
            raise TimeoutError(f"{service} delivery exceeded its 10-second budget")
        try:
            async with asyncio.timeout(remaining):
                if limiter is not None:
                    while not limiter.try_consume().allowed:
                        await asyncio.sleep(0.01)
                async with session.post(url, headers=headers, data=encoded) as response:
                    if 200 <= response.status < 300:
                        return
                    body = await response.read()
                    error = RuntimeError(f"{service} returned {response.status}")
                    retryable = response.status == 429 or 500 <= response.status < 600
                    retry_after = retry_after_seconds(
                        response.headers.get("Retry-After"), body
                    )
        except (aiohttp.ClientError, TimeoutError, OSError) as exc:
            error = exc
            retryable = True
            retry_after = None
        if not retryable or attempt == MAX_ATTEMPTS - 1:
            raise error
        remaining = (deadline - time_monotonic_ns()) / 1e9
        delay = retry_after if retry_after is not None else 0.1 * (2**attempt)
        if delay >= remaining:
            if retry_after is not None:
                raise RuntimeError(
                    f"{service} Retry-After of {delay:.3f}s exceeds the remaining delivery budget"
                ) from error
            raise TimeoutError(
                f"{service} delivery exceeded its 10-second budget"
            ) from error
        await asyncio.sleep(delay)

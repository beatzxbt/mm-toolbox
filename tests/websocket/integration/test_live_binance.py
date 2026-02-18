"""Live Binance Futures integration tests for websocket components.

Run with: pytest tests/websocket/integration/test_live_binance.py --run-live
"""

from __future__ import annotations

import asyncio
from decimal import Decimal, InvalidOperation
from time import time_ns
from typing import Any, Callable

import msgspec
import pytest

from mm_toolbox.websocket.connection import ConnectionState, WsConnectionConfig
from mm_toolbox.websocket.pool import WsPool, WsPoolConfig
from mm_toolbox.websocket.single import WsSingle


class BinanceFuturesBookTicker(msgspec.Struct):
    """Strict schema for Binance Futures @bookTicker events."""

    e: str
    u: int
    E: int
    T: int
    s: str
    b: str
    B: str
    a: str
    A: str


class _LiveBinanceHarness:
    """Shared helpers for live Binance websocket integration tests."""

    BOOK_TICKER_DECODER = msgspec.json.Decoder(
        type=BinanceFuturesBookTicker,
        strict=True,
    )

    @staticmethod
    def _noop_message_handler(msg: bytes) -> None:
        """No-op callback required by WsPool constructor."""
        return None

    @staticmethod
    def _to_decimal(value: str, field: str) -> Decimal:
        """Convert a numeric string to Decimal for strict numeric checks."""
        try:
            return Decimal(value)
        except InvalidOperation as exc:
            raise AssertionError(f"Invalid decimal in field '{field}': {value}") from exc

    @staticmethod
    def _now_ms() -> int:
        """Current unix timestamp in milliseconds."""
        return time_ns() // 1_000_000

    @classmethod
    def _validate_timestamps(
        cls,
        event_time_ms: int,
        transaction_time_ms: int,
        *,
        freshness_budget_ms: int,
        clock_future_drift_ms: int,
    ) -> None:
        """Validate event freshness and basic clock sanity."""
        recv_ms = cls._now_ms()
        assert event_time_ms > 0
        assert transaction_time_ms > 0
        assert event_time_ms <= recv_ms + clock_future_drift_ms
        assert transaction_time_ms <= recv_ms + clock_future_drift_ms
        assert recv_ms - event_time_ms <= freshness_budget_ms
        assert recv_ms - transaction_time_ms <= freshness_budget_ms

    @classmethod
    def _validate_book_ticker(
        cls,
        data: BinanceFuturesBookTicker,
        *,
        expected_symbols: set[str],
        freshness_budget_ms: int,
        clock_future_drift_ms: int,
    ) -> None:
        """Validate strict semantics for book ticker events."""
        assert data.e == "bookTicker"
        assert data.s in expected_symbols
        assert data.u > 0
        cls._validate_timestamps(
            data.E,
            data.T,
            freshness_budget_ms=freshness_budget_ms,
            clock_future_drift_ms=clock_future_drift_ms,
        )

        bid = cls._to_decimal(data.b, "b")
        bid_qty = cls._to_decimal(data.B, "B")
        ask = cls._to_decimal(data.a, "a")
        ask_qty = cls._to_decimal(data.A, "A")
        assert bid > 0
        assert ask > 0
        assert bid <= ask
        assert bid_qty >= 0
        assert ask_qty >= 0

    @staticmethod
    async def _wait_for_state(
        get_state: Callable[[], ConnectionState],
        expected: ConnectionState,
        *,
        timeout_s: float,
    ) -> None:
        """Wait for a websocket wrapper to reach an expected state."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            if get_state() == expected:
                return
            await asyncio.sleep(0.05)
        raise AssertionError(f"Timed out waiting for state={expected}")

    @staticmethod
    async def _wait_for_pool_connections(pool: WsPool, *, timeout_s: float) -> None:
        """Wait until a pool reports at least one connected websocket."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            if pool.get_connection_count() > 0:
                return
            await asyncio.sleep(0.05)
        raise AssertionError("Timed out waiting for active pool connections")

    @staticmethod
    async def _next_message(stream: WsSingle | WsPool, *, timeout_s: float) -> bytes:
        """Receive one websocket payload from an async iterator wrapper."""
        return await asyncio.wait_for(stream.__anext__(), timeout=timeout_s)


@pytest.mark.asyncio
@pytest.mark.live
class TestLiveBinanceFutures(_LiveBinanceHarness):
    """Live Binance futures smoke tests for WsSingle and WsPool."""

    async def test_single_btc_bookticker_smoke(
        self,
        live_test_config: dict[str, Any],
        live_timeout_s: float,
    ) -> None:
        """Validate WsSingle on real BTC futures @bookTicker feed."""
        config = WsConnectionConfig.default(
            f"{live_test_config['binance_futures_base']}/btcusdt@bookTicker"
        )
        ws = WsSingle(config)
        timeout_s = max(6.0, min(live_timeout_s, 20.0))

        valid_count = 0
        async with ws:
            await self._wait_for_state(
                ws.get_state,
                ConnectionState.CONNECTED,
                timeout_s=10.0,
            )
            deadline = asyncio.get_running_loop().time() + timeout_s
            while asyncio.get_running_loop().time() < deadline and valid_count < 3:
                payload = await self._next_message(ws, timeout_s=1.0)
                event = self.BOOK_TICKER_DECODER.decode(payload)
                self._validate_book_ticker(
                    event,
                    expected_symbols={"BTCUSDT"},
                    freshness_budget_ms=live_test_config["freshness_budget_ms"],
                    clock_future_drift_ms=live_test_config["clock_future_drift_ms"],
                )
                valid_count += 1

        assert valid_count >= 3

    async def test_pool_btc_bookticker_smoke(
        self,
        live_test_config: dict[str, Any],
        live_timeout_s: float,
    ) -> None:
        """Validate WsPool on real BTC futures @bookTicker feed."""
        config = WsConnectionConfig.default(
            f"{live_test_config['binance_futures_base']}/btcusdt@bookTicker"
        )
        pool = await WsPool.new(
            config=config,
            on_message=self._noop_message_handler,
            pool_config=WsPoolConfig(num_connections=2, evict_interval_s=60),
        )
        timeout_s = max(6.0, min(live_timeout_s, 20.0))

        valid_count = 0
        async with pool:
            await self._wait_for_state(
                pool.get_state,
                ConnectionState.CONNECTED,
                timeout_s=10.0,
            )
            await self._wait_for_pool_connections(pool, timeout_s=10.0)
            deadline = asyncio.get_running_loop().time() + timeout_s
            while asyncio.get_running_loop().time() < deadline and valid_count < 3:
                payload = await self._next_message(pool, timeout_s=1.0)
                event = self.BOOK_TICKER_DECODER.decode(payload)
                self._validate_book_ticker(
                    event,
                    expected_symbols={"BTCUSDT"},
                    freshness_budget_ms=live_test_config["freshness_budget_ms"],
                    clock_future_drift_ms=live_test_config["clock_future_drift_ms"],
                )
                valid_count += 1

        assert valid_count >= 3


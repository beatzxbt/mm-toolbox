"""Example demonstrating four moving averages side-by-side on Binance BBO mid-price.

This example shows:
- Using WsPool for Binance futures @bookTicker stream
- Computing mid-price from best bid/ask
- Feeding mid-price to EMA, SMA, WMA, and TEMA simultaneously
- Periodic formatted output of all MA values and spread in basis points
- Previewing hypothetical updates with `.next()` without mutating state
- Clean shutdown on Ctrl+C
"""

from __future__ import annotations

import asyncio
import signal
import sys

import msgspec
import numpy as np

from mm_toolbox.moving_average import (
    ExponentialMovingAverage,
    SimpleMovingAverage,
    TimeExponentialMovingAverage,
    WeightedMovingAverage,
)
from mm_toolbox.time.time import time_ms
from mm_toolbox.websocket import WsConnectionConfig, WsPool, WsPoolConfig


class MovingAverageSuite:
    """Holds and updates four moving average instances from BBO mid-prices."""

    def __init__(self, window: int = 20, half_life_s: float = 1.0) -> None:
        """Initialize the suite with four moving average instances.

        Args:
            window: Lookback window for SMA, WMA, and EMA.
            half_life_s: Half-life in seconds for TEMA.
        """
        self.sma = SimpleMovingAverage(window=window)
        self.wma = WeightedMovingAverage(window=window)
        self.ema = ExponentialMovingAverage(window=window)
        self.tema = TimeExponentialMovingAverage(window=2, half_life_s=half_life_s)

        self._window = window
        self._initialized = False
        self._buffer: list[float] = []

        self.best_bid: float = 0.0
        self.best_ask: float = 0.0

    def _initialize_all(self, values: list[float]) -> None:
        """Initialize all MAs with historical values.

        Args:
            values: List of mid-prices to seed the moving averages.
        """
        arr = np.array(values, dtype=np.float64)
        self.sma.initialize(arr)
        self.wma.initialize(arr)
        self.ema.initialize(arr)
        self.tema.initialize(arr)
        self._initialized = True

    def process_bbo(self, bid: float, ask: float) -> None:
        """Process a new best bid/offer update.

        Args:
            bid: Best bid price.
            ask: Best ask price.
        """
        self.best_bid = bid
        self.best_ask = ask
        mid = (bid + ask) / 2.0

        if not self._initialized:
            self._buffer.append(mid)
            if len(self._buffer) >= self._window:
                self._initialize_all(self._buffer)
            return

        self.sma.update(mid)
        self.wma.update(mid)
        self.ema.update(mid)
        self.tema.update(mid)

    def get_spread_bps(self) -> float:
        """Return the current spread in basis points.

        Returns:
            Spread in basis points, or 0.0 if bid/ask are not set.
        """
        if self.best_bid <= 0.0 or self.best_ask <= 0.0:
            return 0.0
        mid = (self.best_bid + self.best_ask) / 2.0
        return (self.best_ask - self.best_bid) / mid * 10_000.0

    def get_values(self) -> dict[str, float]:
        """Return current values of all moving averages.

        Returns:
            Dictionary mapping MA name to its current value.
        """
        return {
            "SMA": self.sma.get_value(),
            "WMA": self.wma.get_value(),
            "EMA": self.ema.get_value(),
            "TEMA": self.tema.get_value(),
        }

    def preview_jump(self, jump_pct: float = 0.001) -> dict[str, float]:
        """Preview MA values if the current mid jumps by jump_pct.

        Args:
            jump_pct: Percentage jump to preview (e.g., 0.001 for 0.1%).

        Returns:
            Dictionary mapping MA name to the hypothetical next value.
        """
        if not self._initialized:
            return {"SMA": 0.0, "WMA": 0.0, "EMA": 0.0, "TEMA": 0.0}

        mid = (self.best_bid + self.best_ask) / 2.0
        hypothetical = mid * (1.0 + jump_pct)
        return {
            "SMA": self.sma.next(hypothetical),
            "WMA": self.wma.next(hypothetical),
            "EMA": self.ema.next(hypothetical),
            "TEMA": self.tema.next(hypothetical),
        }


async def _consume_bbo(
    suite: MovingAverageSuite,
    pool: WsPool,
    shutdown_event: asyncio.Event,
) -> None:
    """Consume BBO messages and update the moving average suite.

    Args:
        suite: The MovingAverageSuite to update.
        pool: The WsPool connected to the Binance bookTicker stream.
        shutdown_event: Event to signal shutdown.
    """
    async for msg in pool:
        if shutdown_event.is_set():
            break
        try:
            data = msgspec.json.decode(msg, type=dict)
            bid = float(data["b"])
            ask = float(data["a"])
            suite.process_bbo(bid, ask)
        except (KeyError, ValueError) as e:
            print(f"Error processing message: {e}", file=sys.stderr)


async def _print_loop(
    suite: MovingAverageSuite,
    shutdown_event: asyncio.Event,
) -> None:
    """Print formatted MA values every second.

    Args:
        suite: The MovingAverageSuite to read values from.
        shutdown_event: Event to signal shutdown.
    """
    preview_counter = 0
    while not shutdown_event.is_set():
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=1.0)
            break
        except asyncio.TimeoutError:
            pass

        ts = time_ms()

        if not suite._initialized:
            print(
                f"[{ts}] Buffering initial {suite._window} samples... "
                f"({len(suite._buffer)} collected)"
            )
            continue

        spread_bps = suite.get_spread_bps()
        values = suite.get_values()

        row = (
            f"[{ts}] "
            f"Spread: {spread_bps:6.2f} bps | "
            f"SMA: {values['SMA']:10.2f} | "
            f"WMA: {values['WMA']:10.2f} | "
            f"EMA: {values['EMA']:10.2f} | "
            f"TEMA: {values['TEMA']:10.2f}"
        )
        print(row, flush=True)

        preview_counter += 1
        if preview_counter % 10 == 0:
            preview = suite.preview_jump(jump_pct=0.001)
            preview_row = (
                f"[{ts}] PREVIEW (+0.1% jump): "
                f"SMA: {preview['SMA']:10.2f} | "
                f"WMA: {preview['WMA']:10.2f} | "
                f"EMA: {preview['EMA']:10.2f} | "
                f"TEMA: {preview['TEMA']:10.2f}"
            )
            print(preview_row, flush=True)


async def main() -> None:
    """Set up WsPool and run the event loop."""
    symbol = "BTCUSDT"
    suite = MovingAverageSuite(window=20, half_life_s=1.0)
    shutdown_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    config = WsConnectionConfig.default(
        wss_url=f"wss://fstream.binance.com/ws/{symbol.lower()}@bookTicker"
    )
    pool = await WsPool.new(
        config=config,
        pool_config=WsPoolConfig.default(),
    )

    try:
        async with pool:
            print(f"Connected to Binance futures {symbol} @bookTicker")
            print("Press Ctrl+C to stop...")

            bbo_task = asyncio.create_task(_consume_bbo(suite, pool, shutdown_event))
            print_task = asyncio.create_task(_print_loop(suite, shutdown_event))
            await asyncio.gather(bbo_task, print_task)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        pool.close()


if __name__ == "__main__":
    asyncio.run(main())

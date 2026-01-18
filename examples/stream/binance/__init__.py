"""Binance-specific stream models and processor entry points.

Exposes the BinanceStreamProcessor used by the multi-venue runner.
"""

from __future__ import annotations

__all__ = ["BinanceStreamProcessor"]

from .stream import BinanceStreamProcessor

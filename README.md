# MM Toolbox
**MM Toolbox** is a Python library designed to provide high-performance tools for market making strategies.

## Contents
```plaintext
mm-toolbox/
├── src/
│   └── mm_toolbox/
│       ├── candles/            # Tools for handling and aggregating candlestick data
│       ├── logging/            # Lightweight logger + Discord/Telegram support
│       │   ├── standard/       # Standard logger implementation
│       │   └── advanced/       # Distributed HFT logger (worker/master)
│       ├── misc/               # Filtering helpers
│       │   └── filter/         # Bounds-based change filter
│       ├── moving_average/     # Various moving averages (EMA/SMA/WMA/TEMA)
│       ├── orderbook/          # Multiple orderbook implementations & tools
│       │   ├── standard/       # Python-based orderbook
│       │   └── advanced/       # High-performance Cython orderbook
│       ├── rate_limiter/       # Token bucket rate limiter
│       ├── ringbuffer/         # Efficient fixed-size circular buffers
│       ├── rounding/           # Fast price/size rounding utilities
│       ├── time/               # Time utilities
│       ├── websocket/          # WebSocket clients + verification tools
│       └── weights/            # Weight generators (EMA/geometric/logarithmic)
├── tests/                      # Unit tests for all the modules
├── pyproject.toml              # Project configuration and dependencies
├── LICENSE                     # License information
├── README.md                   # Main documentation file
└── setup.py                    # Setup script for building Cython extensions
```

## Installation

MM Toolbox is available on PyPI and can be installed using pip:

```bash
pip install mm_toolbox
```

To try the beta without replacing a stable install, use a separate virtual environment and install the pre-release:
```bash
python -m venv mm_toolbox_beta
source mm_toolbox_beta/bin/activate
pip install mm-toolbox==1.0.0b3
```

To always pull the latest pre-release:
```bash
pip install --pre mm-toolbox
```

To install directly from the source, clone the repository and install the dependencies:
```bash
git clone https://github.com/beatzxbt/mm-toolbox.git
cd mm-toolbox
# Install uv if you haven't already: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-groups
make build  # Compile Cython extensions
```

## v1.1 Roadmap Note

Parser modules are being introduced in **v1.1**. They are intentionally not
included in the current `v1.0b5` release branch.

## Usage

After installation, you can start using MM Toolbox by importing the necessary modules:
```python
from mm_toolbox.moving_average import ExponentialMovingAverage as EMA
from mm_toolbox.orderbook import Orderbook, OrderbookLevel
from mm_toolbox.logging.standard import Logger, LogLevel, LoggerConfig

# Example usage:
ema = EMA(window=10, is_fast=True)
tick_size = 0.01
lot_size = 0.001
orderbook = Orderbook(tick_size=tick_size, lot_size=lot_size, size=100)
orderbook.consume_bbo(
    ask=OrderbookLevel.from_values(100.01, 1.2, 1, tick_size, lot_size),
    bid=OrderbookLevel.from_values(100.00, 1.0, 1, tick_size, lot_size),
)
logger = Logger(
    name="Example",
    config=LoggerConfig(base_level=LogLevel.INFO, do_stdout=True),
)
```

## Latest release notes (v1.0.0, feature complete)

### Major Architecture Shift: Numba → Cython/C

MM Toolbox v1.0.0 represents a fundamental shift from Numba-accelerated code to Cython/C implementations. This transition brings significant benefits:

**Performance Improvements**: Core components now see speed improvements of 5–30x compared to previous Numba implementations, with some components achieving even greater gains.

**Better Interoperability**: Cython/C extensions integrate seamlessly with the Python ecosystem. Unlike Numba's JIT compilation, Cython extensions are pre-compiled, eliminating warm-up times and providing consistent performance from the first call. This makes MM Toolbox more suitable for production HFT systems where predictable latency is critical.

**Type Safety & Tooling**: Full type stub support (`.pyi` files) enables better IDE integration, static type checking with Pyright, and improved developer experience. Cython's explicit typing model also catches more errors at compile time.

**Zero-Allocation Designs**: Many components have been redesigned with zero-allocation patterns, reducing GC pressure and improving performance in tight loops.

The v1.0 feature set is complete. Each component ships with a focused README
that covers API details, architecture notes, and usage examples.

### Component Highlights

**Candles** (`mm_toolbox.candles`): High-performance candle aggregation with time, tick, volume, price, and multi-trigger buckets. Maintains a live `latest_candle`, stores completed candles in a ring buffer, and supports async iteration for stream processing.

**Misc** (`mm_toolbox.misc`): Utility helpers including `DataBoundsFilter` for bounds-based change detection. Parser modules are introduced in `v1.1` (not in this `v1.0b5` branch).

**Rate Limiter** (`mm_toolbox.rate_limiter`): Token-bucket rate limiting with optional burst policies and per-second sub-buckets, plus explicit state tracking via `RateLimitState`.

**Ringbuffer** (`mm_toolbox.ringbuffer`): Efficient circular buffers with multiple implementations:
- `NumericRingBuffer`: Fast numeric data handling
- `BytesRingBuffer`: Optimized for byte arrays
- `BytesRingBufferFast`: Pre-allocated slots for predictable byte workloads
- `GenericRingBuffer`: Flexible support for any Python type
- `IPCRingBuffer`: PUSH/PULL transport for SPSC/MPSC/SPMC topologies
- `SharedMemoryRingBuffer`: SPSC shared-memory ring buffer (POSIX-only)

All ring buffers share consistent insert/consume semantics and overwrite oldest
entries on overflow for bounded memory usage.

**Moving Average** (`mm_toolbox.moving_average`): Comprehensive moving average implementations including EMA, SMA, WMA, and TEMA (Triple Exponential Moving Average). All implementations support `.next()` for previewing future values without state mutation.

**Orderbook** (`mm_toolbox.orderbook`): Dual implementation approach with aligned APIs:
- `standard`: Pure Python implementation for flexibility
- `advanced`: Zero-allocation Cython implementation achieving >4x faster BBO updates and >5x faster per-level batch updates

**Websocket** (`mm_toolbox.websocket`): WebSocket connection management built on PicoWs with latency tracking, ring-buffered message ingestion, and pool routing to the fastest connection.

**Logging** (`mm_toolbox.logging`): Two-tier logging system:
- `standard`: Lightweight logger with Discord/Telegram support
- `advanced`: Distributed HFT logger with worker/master architecture, batching, and customizable handlers

**Rounding** (`mm_toolbox.rounding`): Fast, directional price/size rounding with scalar and vectorized paths.

**Time** (`mm_toolbox.time`): High-performance time utilities for timestamp operations.

**Weights** (`mm_toolbox.weights`): Weight generators for EMA, geometric, and logarithmic weighting schemes.

### Breaking Changes

These notes compare this branch (`v1.0b`) against `master`.

- **Top-level imports removed**: `mm_toolbox` no longer re-exports classes/functions; import from submodules instead (e.g., `mm_toolbox.orderbook`, `mm_toolbox.time`, `mm_toolbox.logging.standard`).
- **Numba stack removed**: `mm_toolbox.numba` and all Numba-based implementations are gone (old orderbook, ringbuffers, rounding, and array helpers).
- **Orderbook rewrite**: the Numba `Orderbook(size)` (arrays + `refresh`/`update_*`/`seq_id`) is replaced by standard/advanced orderbooks that require `tick_size` + `lot_size` and ingest `OrderbookLevel` objects via `consume_snapshot`, `consume_deltas`, and `consume_bbo(ask, bid)`.
- **Candles redesign**: candle aggregation now uses `Trade`/`Candle` objects, async iteration, and a generic ringbuffer; `MultiTriggerCandles` is renamed to `MultiCandles` with `max_size`, and `PriceCandles` was added.
- **Logging restructure**: `mm_toolbox.logging.Logger` and `FileLogConfig/DiscordLogConfig/TelegramLogConfig` were removed; use `mm_toolbox.logging.standard` or `mm_toolbox.logging.advanced` and pass handler objects directly.
- **Ringbuffer API replaced**: `RingBufferSingleDim*`, `RingBufferTwoDim*`, and `RingBufferMultiDim` were removed; use `NumericRingBuffer`, `GenericRingBuffer`, `BytesRingBuffer`, `BytesRingBufferFast`, and IPC/SHM variants.
- **Rounding API replaced**: `Round` was removed; use `Rounder` + `RounderConfig` (directional rounding is configurable).
- **Websocket rewrite**: `SingleWsConnection`, `WsStandard`, `WsFast`, `WsPoolEvictionPolicy`, and `VerifyWsPayload` were removed; use `WsConnection`, `WsSingle`, `WsPool`, and their config/state types.
- **Moving averages/time changes**: `HullMovingAverage` was removed; `SimpleMovingAverage` and `TimeExponentialMovingAverage` were added. Time helpers now return integers and `time_iso8601()` accepts an optional timestamp.

### Migration Guide

Follow these steps when moving from `master` to `v1.0b`.

1. **Install/build changes (source installs)**:
   - Poetry/requirements-based installs from `master` are replaced by `uv` + Cython builds.
   ```bash
   uv sync --all-groups
   make build
   ```

2. **Update imports (top-level exports removed)**:
   ```python
   # master
   from mm_toolbox import Orderbook, ExponentialMovingAverage, Round, time_s

   # v1.0b
   from mm_toolbox.orderbook import Orderbook
   from mm_toolbox.moving_average import ExponentialMovingAverage
   from mm_toolbox.rounding import Rounder, RounderConfig
   from mm_toolbox.time import time_s
   ```

3. **Orderbook migration**:
   - Old API used NumPy arrays + sequence IDs; new API uses `OrderbookLevel` objects and does not track `seq_id`.
   - `refresh`/`update_bids`/`update_asks` -> `consume_snapshot`/`consume_deltas`; `update_bbo` -> `consume_bbo(ask, bid)`.
   ```python
   # master
   ob = Orderbook(size=100)
   ob.refresh(asks_np, bids_np, new_seq_id=42)
   ob.update_bbo(bid_price, bid_size, ask_price, ask_size, new_seq_id=43)

   # v1.0b
   from mm_toolbox.orderbook import Orderbook, OrderbookLevel

   ob = Orderbook(tick_size=0.01, lot_size=0.001, size=100)
   asks = [
       OrderbookLevel.from_values(p, s, norders=0, tick_size=0.01, lot_size=0.001)
       for p, s in asks_np
   ]
   bids = [
       OrderbookLevel.from_values(p, s, norders=0, tick_size=0.01, lot_size=0.001)
       for p, s in bids_np
   ]
   ob.consume_snapshot(asks=asks, bids=bids)
   ob.consume_bbo(
       ask=OrderbookLevel.from_values(ask_price, ask_size, 0, 0.01, 0.001),
       bid=OrderbookLevel.from_values(bid_price, bid_size, 0, 0.01, 0.001),
   )
   ```
   - If you need the Cython implementation, import `AdvancedOrderbook` from `mm_toolbox.orderbook.advanced`.

4. **Candles migration**:
   - Trades are now passed as `Trade` objects and candles are stored as `Candle` objects.
   - `MultiTriggerCandles` -> `MultiCandles` (`max_volume` -> `max_size`, `max_ticks` is now `int`).
   ```python
   from mm_toolbox.candles import TimeCandles, MultiCandles
   from mm_toolbox.candles.base import Trade

   candles = TimeCandles(secs_per_bucket=1.0, num_candles=1000)
   candles.process_trade(Trade(time_ms=1700000000000, is_buy=True, price=100.0, size=0.5))
   ```

5. **Logging migration**:
   - Standard logger lives in `mm_toolbox.logging.standard`, advanced logger in `mm_toolbox.logging.advanced`.
   ```python
   from mm_toolbox.logging.standard import Logger, LoggerConfig, LogLevel
   from mm_toolbox.logging.standard.handlers import FileLogHandler

   logger = Logger(
       name="example",
       config=LoggerConfig(base_level=LogLevel.INFO, do_stdout=True),
       handlers=[FileLogHandler("logs.txt", create=True)],
   )
   ```

6. **Ringbuffer migration**:
   - `RingBufferSingleDimFloat/Int` -> `NumericRingBuffer(max_capacity=..., dtype=...)`
   - `RingBufferTwoDim*`/`RingBufferMultiDim` -> `GenericRingBuffer` (store arrays/objects)
   - `BytesRingBufferFast` now rejects inserts larger than its slot size.

7. **Rounding migration**:
   ```python
   from mm_toolbox.rounding import Rounder, RounderConfig

   rounder = Rounder(RounderConfig.default(tick_size=0.01, lot_size=0.001))
   price = rounder.bid(100.1234)
   ```

8. **Websocket migration**:
   ```python
   from mm_toolbox.websocket import WsConnectionConfig, WsSingle

   config = WsConnectionConfig.default("wss://example", on_connect=[b"SUBSCRIBE ..."])
   ws = WsSingle(config)
   await ws.start()
   ```

9. **Moving averages + time**:
   - `HullMovingAverage` was removed; use `SimpleMovingAverage` or `TimeExponentialMovingAverage`.
   - `time_s/time_ms/...` return integers now; `time_iso8601()` optionally formats a provided timestamp.

## Roadmap

### v1.1.0
- **Websocket**: Move `WsPool` and `WsSingle` into Cython classes to eliminate `call_soon_threadsafe` overhead in hot paths.
- **Logging**: Move more advanced logger components into C to unlock similar performance gains.
- **Orderbook**: Add Cython helpers to build/consume levels from string pair lists (e.g., `[[price, size], ...]`) to avoid Python loops in depth snapshots/deltas.

### v1.2.0
**Parsers**: Introduction of high-performance parsing utilities including JSON parsers and crypto exchange-specific parsers (e.g., Binance top-of-book parser).

## License
MM Toolbox is licensed under the MIT License. See the [LICENSE](/LICENSE) file for more information.

## Contributing
Contributions are welcome! Please read the [CONTRIBUTING.md](/CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Contact
For questions or support, please open an [issue](https://github.com/beatzxbt/mm-toolbox/issues).
I can also be reached on [Twitter](https://twitter.com/BeatzXBT) and [Discord](@gamingbeatz) :D

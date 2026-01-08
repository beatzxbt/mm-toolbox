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
pip install mm-toolbox==1.0.0b0
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

## Usage

After installation, you can start using MM Toolbox by importing the necessary modules:
```python
from mm_toolbox import Orderbook
from mm_toolbox import ExponentialMovingAverage as EMA
from mm_toolbox.logging.standard import Logger, LogLevel, LoggerConfig

# Example usage:
ema = EMA(window=10, is_fast=True)
logger = Logger(
    config=LoggerConfig(
        base_level=LogLevel.INFO,
        do_stdout=True
    ),
    name="Example",
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

**Misc** (`mm_toolbox.misc`): Utility helpers including `DataBoundsFilter` for bounds-based change detection.

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

- **Orderbook**: `consume_*` functions now take `asks` before `bids`; `update_bbo` is renamed to `consume_bbo`. Snapshots replace both ladders, and deltas that would wipe the opposite side without replacement levels are ignored.
- **Rate Limiter Module**: The limiter moved from `mm_toolbox.misc` to `mm_toolbox.rate_limiter` and exposes `RateLimitState` from the core module.
- **Ringbuffer**: Integer-specific ringbuffers removed; use `NumericRingBuffer` instead. `BytesRingBufferFast` now rejects oversized inserts instead of truncating silently.
- **API Unification**: Function signatures and parameter names were standardized across components.
- **Numba Deprecation**: Previous Numba implementations are no longer included; migrate to the Cython/C equivalents.

### Migration Guide

Most code should work with minimal changes. Update the following if you rely on affected components:
1. **Rate limiter imports**: Replace `mm_toolbox.misc.limiter` with `mm_toolbox.rate_limiter` and update any `RateLimiter`/`RateLimitState` imports accordingly.
2. **Orderbook ingestion**: Swap parameter order to `asks, bids` for `consume_*` calls, and rename `update_bbo` to `consume_bbo`. Remove any optional flags for tick/lot computation.
3. **Ringbuffer usage**: Replace integer-only ringbuffers with `NumericRingBuffer` and handle oversized inserts for `BytesRingBufferFast` explicitly.
4. **Numba users**: Migrate any Numba-specific paths to the Cython implementations and rebuild extensions with `make build`.

*Components not mentioned have either not incurred significant changes or maintain backward compatibility.*

## Roadmap

### v1.1.0
**Parsers**: Introduction of high-performance parsing utilities including JSON parsers and crypto exchange-specific parsers (e.g., Binance top-of-book parser).

## License
MM Toolbox is licensed under the MIT License. See the [LICENSE](/LICENSE) file for more information.

## Contributing
Contributions are welcome! Please read the [CONTRIBUTING.md](/CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Contact
For questions or support, please open an [issue](https://github.com/beatzxbt/mm-toolbox/issues).
I can also be reached on [Twitter](https://twitter.com/BeatzXBT) and [Discord](@gamingbeatz) :D

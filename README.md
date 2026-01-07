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
│       ├── moving_average/     # Various moving averages (EMA/SMA/WMA/TEMA)
│       ├── orderbook/          # Multiple orderbook implementations & tools
│       │   ├── standard/       # Python-based orderbook
│       │   └── advanced/       # High-performance Cython orderbook
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

## Latest release notes (v1.0.0)

### Major Architecture Shift: Numba → Cython/C

MM Toolbox v1.0.0 represents a fundamental shift from Numba-accelerated code to Cython/C implementations. This transition brings significant benefits:

**Performance Improvements**: Core components now see speed improvements of 5–30x compared to previous Numba implementations, with some components achieving even greater gains.

**Better Interoperability**: Cython/C extensions integrate seamlessly with the Python ecosystem. Unlike Numba's JIT compilation, Cython extensions are pre-compiled, eliminating warm-up times and providing consistent performance from the first call. This makes MM Toolbox more suitable for production HFT systems where predictable latency is critical.

**Type Safety & Tooling**: Full type stub support (`.pyi` files) enables better IDE integration, static type checking with Pyright, and improved developer experience. Cython's explicit typing model also catches more errors at compile time.

**Zero-Allocation Designs**: Many components have been redesigned with zero-allocation patterns, reducing GC pressure and improving performance in tight loops.

### Component Highlights

**Candles** (`mm_toolbox.candles`): High-performance candlestick aggregation supporting price, time, volume, and tick-based candles. New `.next()` method allows previewing next-state values without updating internal state, useful for backtesting and simulation scenarios.

**Ringbuffer** (`mm_toolbox.ringbuffer`): Efficient circular buffers with multiple implementations:
- `NumericRingBuffer`: Fast numeric data handling
- `BytesRingBuffer`: Optimized for byte arrays
- `GenericRingBuffer`: Flexible support for any Python type
- `SharedMemoryRingBuffer`: IPC-capable shared memory buffers (POSIX-only)

New unified API provides both safe and unsafe access patterns, enabling zero-copy operations for performance-critical code paths.

**Moving Average** (`mm_toolbox.moving_average`): Comprehensive moving average implementations including EMA, SMA, WMA, and TEMA (Triple Exponential Moving Average). All implementations support `.next()` for previewing future values without state mutation.

**Orderbook** (`mm_toolbox.orderbook`): Dual implementation approach:
- `standard`: Pure Python implementation for flexibility
- `advanced`: Zero-allocation Cython implementation achieving >4x faster BBO updates and >5x faster per-level batch updates

**Websocket** (`mm_toolbox.websocket`): Production-ready WebSocket client library:
- `SingleConnection`: Auto-pinging, latency tracking, and raw data access
- `ConnectionPool`: Smart load balancing, latency-aware routing, and modular design
- Built-in verification tools for message integrity

**Logging** (`mm_toolbox.logging`): Two-tier logging system:
- `standard`: Lightweight logger with Discord/Telegram support
- `advanced`: Distributed HFT logger with worker/master architecture, batching, and customizable handlers

**Rounding** (`mm_toolbox.rounding`): Fast price/size rounding utilities optimized for trading operations.

**Time** (`mm_toolbox.time`): High-performance time utilities for timestamp operations.

**Weights** (`mm_toolbox.weights`): Weight generators for EMA, geometric, and logarithmic weighting schemes.

### Breaking Changes

- **API Unification**: Some function signatures and parameter names have been standardized across components for consistency.
- **Ringbuffer**: Integer-specific ringbuffers removed; use `NumericRingBuffer` instead. Multi-ringbuffer now supports any numeric type, strings, and bytes.
- **Numba Deprecation**: Previous Numba implementations are no longer included. If you were using Numba-specific features, you'll need to migrate to the Cython equivalents.

### Migration Guide

Most code should work with minimal changes. The primary differences are:
1. Import paths remain the same (e.g., `from mm_toolbox import ExponentialMovingAverage`)
2. Performance characteristics are improved, so you may be able to simplify your code
3. Type hints are now fully supported and recommended for better IDE support

*Components not mentioned have either not incurred significant changes or maintain backward compatibility.*

## Planned additions/upgrades

### v1.1.0 (no earlier than Jun 25')
**Misc**: Introduction of `filter` and `limiter` modules for data filtering and rate limiting utilities.

**Orderbook**: Further performance optimizations for orderbook operations.

**Logger**: AdvancedLogger performance improvements & two-way communication between Master<>Worker loggers, enabling health checks and remote shutdown capabilities.

**Websocket**: VerifyWsPayload performance improvements, FastWsPool optimizations.

### v1.2.0
**Parsers**: Introduction of high-performance parsing utilities including JSON parsers and crypto exchange-specific parsers (e.g., Binance top-of-book parser).

**Ringbuffer**: Introduction of `SharedMemoryRingBuffer` for inter-process communication (IPC) scenarios, enabling efficient data sharing between processes.

## License
MM Toolbox is licensed under the MIT License. See the [LICENSE](/LICENSE) file for more information.

## Contributing
Contributions are welcome! Please read the [CONTRIBUTING.md](/CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Contact
For questions or support, please open an [issue](https://github.com/beatzxbt/mm-toolbox/issues).
I can also be reached on [Twitter](https://twitter.com/BeatzXBT) and [Discord](@gamingbeatz) :D

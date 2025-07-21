# MM Toolbox
**MM Toolbox** is a Python library designed to provide high-performance tools for market making strategies.

## Contents
```plaintext
mm-toolbox/
├── src/
│   ├── mm_toolbox/
│   │   ├── candles/            # Tools for handling and aggregating candlestick data
│   │   ├── logging/            # Lightweight logger + Discord/Telegram support
│   │   ├── moving_average/     # Various moving averages (EMA/SMA/WMA etc)
│   │   ├── numba/              # Numba-optimized functions
│   │   ├── orderbook/          # Multiple orderbook implementations & tools
│   │   ├── ringbuffer/         # Efficient fixed-size circular buffers
│   │   ├── rounding/           # Fast price/size rounding utilities
│   │   ├── time/               # Time utilities
│   │   ├── websocket/          # WebSocket clients + verification tools
│   │   ├── weights/            # Weight generators 
│   ├── __init__.py             # Package initialization
├── tests/                      # Unit tests for all the modules
├── .gitignore                  # Git ignore file
├── LICENSE                     # License information
├── README.md                   # Main documentation file
├── requirements.txt            # Python dependencies
└── setup.py                    # Setup script for pip installation
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
pip install poetry
poetry install
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
        do_stout=True
    ),
    name="Example",
)
```

## Latest release notes (v1.0.0)

### Migration from Numba to Cython
- Performance: Core library components see speed improvements of 5–30x.
- API Parity: The new Cython classes provide nearly the same interfaces as before, but now with more consistent design and naming.

### Common breaking changes 
- Reimplemented Components: Most are now written fully in Cython/C for maximum performance.
- Numba: Existing Numba code remains accessible under `mm_toolbox.numba.<component>`, but is slated for deprecation in v2.0.0.
- Unified APIs: Some function signatures or parameter names may have changed for consistency.

**Candles**: Added functionality to get next-state candle values without updating internal states (mirrors moving average's `.next()`).

**Ringbuffer**: Integer support dropped for 1/2D ringbuffers. Multi now supports *any* number type, strings and bytes (datetime compatibility may be added at a later date). New unified API, opening up safe and unsafe access to underlying buffer for faster custom operations.

**Moving Average**: Introduced SMA, TEMA. Deprecated HMA (lack of use).

**Orderbook**: New zero-alloc design Orderbook, >4x faster for BBO updates and >5x faster per-level for batch updates. 

**Websocket**: New single connection design, supporting auto-pinging and latency tracking natively. Optionally exposes raw unprocessed data for custom processing. New pool design, with smarter latency policies, load balancing features and better modularity. 

**Logging**: New AdvancedLogger, purpose built for distributed HFT systems. Two parts, a very lightweight worker logger to collect and batch send logs to a master logger. Master logger then distributes it to external handlers (file/discord/telegram/etc) in various customizable formats. Importable through `mm_toolbox.logger.advanced`.

**Numba**: Migration of existing components into sub folders in this directory. *These will be kept feature compatible with the Cython API until `v2.0.0`, where they will likely be deprecated*.

*Components not mentioned have either not incurred any changes, or are not significant enough to affect a vast majority of code bases.*

## Planned additions/upgrades

### v1.1.0 (no earlier than Jun 25')
**Orderbook**: Orderbook ++Performance.

**Logger**: AdvancedLogger ++Performance & two-way comms between Master<>Worker loggers, opening up ability to pulse health checks and trigger remote shutdowns.

**Websocket**: VerifyWsPayload ++Performance, FastWsPool ++Performance.

## License
MM Toolbox is licensed under the MIT License. See the [LICENSE](/LICENSE) file for more information.

## Contributing
Contributions are welcome! Please read the [CONTRIBUTING.md](/CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Contact
For questions or support, please open an [issue](https://github.com/beatzxbt/mm-toolbox/issues). 
I can also be reached on [Twitter](https://twitter.com/BeatzXBT) and [Discord](@gamingbeatz) :D

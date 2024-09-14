# MM Toolbox
**MM Toolbox** is a Python library designed to provide high-performance tools for market making strategies.

## Contents
```plaintext
mm-toolbox/
├── src/
│   ├── mm_toolbox/
│   │   ├── candles/            # Tools for handling and aggregating candlestick data
│   │   ├── logging/            # Lightweight logger + Discord/Telegram support
│   │   ├── moving_average/     # Various moving averages (EMA/HMA/WMA etc)
│   │   ├── numba/              # Numba-optimized functions
│   │   ├── orderbook/          # Multiple orderbook implementations & tools
│   │   ├── ringbuffer/         # Efficient fixed-size circular buffers
│   │   ├── rounding/           # Fast price/size rounding utilities
│   │   ├── time/               # Time utilities
│   │   ├── websocket/          # WebSocket clients + payload verification
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
from mm_toolbox import ExponentialMovingAverage
from mm_toolbox import time_iso8601

# Example usage:
orderbook = Orderbook(size=100)
```

## Planned additions/upgrades

### v0.2.0
**Numba**: Complete coverage of [Numba's top-level functions](https://numba.readthedocs.io/en/stable/reference/numpysupported.html#other-functions) (with custom implementation if faster).

**Moving Average**: Weighted Moving Average (WMA).

**Orderbook**: Directly update BBA, ++Performance.

**Candles**: Multi-trigger candle (time/tick/volume), ++Performance.

**Logger**: High performance logger.

**Websocket**: Standard websocket, Fast websocket pool + auto latency swapping mechanism.

### v0.3.0
**Numba**: Coverage of [Numba's reduction functions.](https://numba.readthedocs.io/en/stable/reference/numpysupported.html#reductions) (with custom implementation if faster).

**Moving Average**: Simple Moving Average (SMA).

**Orderbook**: [HFT Orderbook](/mm_toolbox/src/orderbook/hft.py), aiming to be fastest Python orderbook on GitHub.

### v0.4.0
**Weights**: Logarithmic.

**Orderbook**: HFTOrderbook ++Performance.

**Websocket**: FastWsPool ++Stability ++Performance, VerifyWsPayload ++Performance.

### v0.5.0
**Orderbook**: L3 Orderbook.

## License
MM Toolbox is licensed under the MIT License. See the [LICENSE](/LICENSE) file for more information.

## Contributing
Contributions are welcome! Please read the [CONTRIBUTING.md](/CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Contact
For questions or support, please open an [issue](https://github.com/beatzxbt/mm-toolbox/issues). 
I can also be reached on [Twitter](https://twitter.com/BeatzXBT) and [Discord](@gamingbeatz) :D

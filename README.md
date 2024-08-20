# MM Toolbox

**MM Toolbox** is a Python library designed to provide high-performance tools for market making strategies.

## Contents

```plaintext
mm-toolbox/
├── mm_toolbox/
│   ├── src/
│   │   ├── candles/            # Tools for handling and aggregating candlestick data
│   │   ├── logging/            # Lightweight logging utilities
│   │   ├── moving_average/     # Implementations of various moving averages
│   │   ├── numba/              # Numba-optimized functions and utilities
│   │   ├── orderbook/          # Multiple orderbook implementations & tools
│   │   ├── ringbuffer/         # Efficient fixed-size circular buffers
│   │   ├── rounding/           # Utilities for rounding prices and sizes
│   │   ├── time/               # High-performance time utilities
│   │   ├── websocket/          # WebSocket handling utilities
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
pip install -r requirements.txt
python setup.py install
```

## Usage

After installation, you can start using MM Toolbox by importing the necessary modules:
```python
from mm_toolbox.src.orderbook import Orderbook
from mm_toolbox.src.moving_average import ExponentialMovingAverage
from mm_toolbox.src.time import time_iso8601

# Example usage:
ob = Orderbook(size=100)
```

## Planned additions/upgrades

### v0.2.0
**Numba**: Complete coverage of [Numba's top-level functions](https://numba.readthedocs.io/en/stable/reference/numpysupported.html#other-functions)

**Orderbook**: Directly update BBA, Imbalance Feature, ++Performance

**Candles**: ++Performance

**Websocket**: Fast websocket pool + auto swapping latency mechanism.

### v0.3.0
**Candles**: Multi-trigger candle (time/tick/volume).

**Logger**: High performance logger w/Database support integrated.

**Moving Average**: Weighted Moving Average (WMA).

### v0.4.0
**Orderbook**: [HFT Orderbook](/mm_toolbox/src/orderbook/hft.py), aiming to be fastest Python orderbook on GitHub.

## License

MM Toolbox is licensed under the MIT License. See the [LICENSE](/LICENSE) file for more information.

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](/CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Contact

For questions or support, please open an [issue](https://github.com/beatzxbt/mm-toolbox/issues). 
I can also be reached on [Twitter](https://twitter.com/BeatzXBT) and [Discord](@gamingbeatz) :D
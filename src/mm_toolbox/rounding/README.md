# Rounding

Price and size rounding utilities.

## Core concepts

- `RounderConfig`: tick/lot sizes plus rounding direction controls.
- `Rounder`: fast scalar and vector rounding methods.

## Basic usage

```python
from mm_toolbox.rounding import Rounder, RounderConfig

config = RounderConfig.default(tick_size=0.01, lot_size=0.001)
rounder = Rounder(config)

bid = rounder.bid(100.1234)
ask = rounder.ask(100.1234)
size = rounder.size(0.9876)
```

## Operations

- `bid(price)`: round to the nearest tick (direction controlled by config).
- `ask(price)`: round to the nearest tick (direction controlled by config).
- `size(size)`: round to the nearest lot (direction controlled by config).
- `bids(prices)`, `asks(prices)`, `sizes(sizes)`: vectorized numpy variants.

## Behavior notes

- `RounderConfig.default()` rounds bids down, asks up, and sizes up.
- Directional behavior can be flipped in the config for maker/taker strategies.
- Scalar and vector paths share the same rounding logic and precision handling.

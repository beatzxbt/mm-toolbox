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

# Vectorized example
prices = np.array([100.1234, 100.5678, 101.0012])
bids = rounder.bids(prices)
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

## Supported grid

Tick and lot sizes must be canonical finite powers of ten from `1e-308` to
`1e308`, such as `0.001`, `0.01`, `0.1`, `1`, or `10`. Other increments are
rejected at construction. This deliberately excludes instruments with grids
such as `0.05`, `0.25`, or `0.5`. Configuration is captured at construction;
the rounder's grid sizes are read-only.

Scalar and array paths preserve canonical grid points and distinguish adjacent
floating-point values without a blanket epsilon. Inputs must be finite and
within `2**52 - 1` ticks of zero; values outside that precision range, and
nonfinite output, raise `ValueError`. Validation and decimal scale selection
happen once; the rounding path uses cached scales and native floor/ceiling
with an adjacent-grid correction.

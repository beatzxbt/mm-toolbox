# Moving averages

Streaming moving average implementations with a shared base interface and
optimized ring buffer storage.

## Core concepts

- `MovingAverage`: common interface with `initialize`, `next`, `update`, and accessors.
- `window`: number of observations that define the smoothing horizon.
- `is_fast`: disables history storage to reduce memory and overhead.

## Basic usage

```python
import numpy as np
from mm_toolbox.moving_average import SimpleMovingAverage

sma = SimpleMovingAverage(window=5)
sma.initialize(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

value = sma.update(6.0)
history = sma.get_values()
```

`next()` computes the next value without mutating state, which is useful for
what-if evaluation or plotting.

## Moving average families

### SimpleMovingAverage (SMA)
- Equal weights across the window.
- Good for stable smoothing with minimal bias.
- Requires `initialize()` with a full window.

### WeightedMovingAverage (WMA)
- Linear weights from 1..N (recent values matter more).
- Good for moderate responsiveness without heavy weighting.
- Requires `initialize()` with a full window.

### ExponentialMovingAverage (EMA)
- Exponential decay with `alpha` (default `2/(window+1)`).
- Good for responsive smoothing and trend tracking.
- Can start immediately; `initialize()` is optional.

### TimeExponentialMovingAverage (TEMA)
- Exponential decay based on wall-clock delta and `half_life_s`.
- Good for irregularly spaced data or variable update cadence.
- Can start immediately; `initialize()` is optional.

## Behavior notes

- `is_fast=True` disables history; `get_values`, iteration, and indexing raise.
- `initialize(values)` sets the baseline and warms the average.
- `update(value)` mutates state; `next(value)` does not.

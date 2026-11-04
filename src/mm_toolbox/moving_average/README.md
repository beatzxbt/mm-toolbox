# Moving averages

Streaming moving average implementations with a shared base interface and
optimized ring buffer storage.

## Core concepts

- `MovingAverageProtocol`: shared interface implemented by all moving averages.
- `MovingAverage`: common interface with `initialize`, `next`, `update`, and accessors.
- `window`: number of observations that define the smoothing horizon.
- `is_fast`: disables history storage to reduce memory and overhead.
- `get_value()`: primary accessor that returns the current average value.

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
- `alpha=0` selects that default; custom alpha must be finite and in `(0, 1]`.
- Good for responsive smoothing and trend tracking.
- Can start immediately; `initialize()` is optional.

### TimeExponentialMovingAverage (TEMA)
- Exponential decay based on precise monotonic elapsed time and `half_life_s`.
- Good for irregularly spaced data or variable update cadence.
- Can start immediately; `initialize()` is optional.

TEMA warm-up arrays have no timestamps. Their samples share one clock origin,
so the first sample seeds the value and elapsed-time decay begins with live
updates. Reinitialization resets that origin. Initialization results therefore
do not depend on how long the warm-up loop takes to execute. A single sample is
enough to seed either EMA or TEMA. Before initialization, `next(value)` previews
that seed without changing state, and the first `update(value)` records it in
history.

## Behavior notes

- `initialize(values)` accepts a nonempty, one-dimensional `float64` NumPy
  array. Read-only and strided arrays are supported. SMA and WMA require exactly
  `window` samples; EMA and TEMA accept any nonempty length.
- Samples must be finite. Invalid initialization arrays or scalar samples raise
  before changing the current value, history, rolling sums, or TEMA clock.
- Reinitialization discards previous samples and output history.
- `is_fast=True` skips allocating output history; `get_values()` and `len()`
  raise. SMA and WMA still retain the raw window needed for their calculations.
- Output history uses a ring buffer whose capacity rounds `window` up to the
  next power of two. Access it with `get_values()`; moving averages do not
  implement iteration or indexing.
- `update(value)` mutates state; `next(value)` does not.
- `get_value()` is the primary accessor for the current average value.

## Benchmark results (2026-11-04)

The table shows p50 nanoseconds per call from 100,000 samples per operation,
after 1,000 warmups. Configuration: window 100, history enabled (`--no-fast`),
and TEMA half-life 1 second. Initialization uses 100 values.

| Type | Initialize (ns) | Update (ns) | Next (ns) |
| --- | ---: | ---: | ---: |
| EMA | 11,000 | 292 | 167 |
| SMA | 10,625 | 500 | 209 |
| TEMA | 10,584 | 375 | 250 |
| WMA | 10,709 | 500 | 209 |

Measured on CPython 3.14.4, NumPy 2.5.4, Cython 3.3.0, Apple clang 21.0.0,
macOS 26.6.2 arm64, Apple M2 (8 CPU cores, 16 GiB), at revision `c5f9dba`.
These are local measurements and do not establish performance on other machines.

Reproduce from the repository root:

```sh
.venv/bin/python -u benchmarks/moving_average/benchmark_moving_average.py --operations 100000 --warmup 1000 --window 100 --half-life-s 1.0 --no-fast
```

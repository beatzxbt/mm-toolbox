# Weights

Weight generators for moving averages and signal smoothing.

## Functions

### ema_weights
- Exponential decay with `alpha` (default `2/(window+1)`).
- `normalized=True` returns weights that sum to 1.

### geometric_weights
- Geometric series weights with ratio `r` (default 0.75).
- `normalized=True` returns weights that sum to 1.

### logarithmic_weights
- Logarithmic growth weights based on `log(1..N)`.
- `normalized=True` returns weights that sum to 1.

## Basic usage

```python
from mm_toolbox.weights import ema_weights, geometric_weights, logarithmic_weights

ema = ema_weights(window=10)
geo = geometric_weights(num=10, r=0.8)
log = logarithmic_weights(num=10)
```

## Behavior notes

- All functions return `numpy.ndarray[float64]`.
- Set `normalized=False` to keep raw weight magnitudes.

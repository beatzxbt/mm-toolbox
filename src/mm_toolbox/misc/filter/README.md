# Filter

Streaming change filter for numeric values.

## Core concepts

- `DataBoundsFilter`: maintains dynamic lower/upper bounds around a reference
  value, defined by a percentage threshold.

## Basic usage

```python
from mm_toolbox.misc.filter import DataBoundsFilter

filter = DataBoundsFilter(threshold_pct=1.0)
filter.reset(value=100.0)

# Returns True only when value moves outside ±1% bounds
changed = filter.check_and_update(101.5)
```

## Operations

- `reset(value)`: reset bounds centered on value
- `check_and_update(value, reset=False)`: return True if bounds were (re)initialized

## Behavior notes

- `threshold_pct` must be in (0, 100).
- `check_and_update` with `reset=True` unconditionally resets bounds.

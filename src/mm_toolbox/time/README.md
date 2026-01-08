# Time

High-resolution time helpers and ISO-8601 conversions.

## Available functions

- `time_s`, `time_ms`, `time_us`, `time_ns`: wall-clock time.
- `time_monotonic_s`, `time_monotonic_ms`, `time_monotonic_us`, `time_monotonic_ns`:
  monotonic time for duration measurement.
- `iso8601_to_unix(timestamp)`: parse ISO-8601 string to unix seconds.
- `time_iso8601(timestamp=0.0)`: format unix seconds to ISO-8601 (uses now if 0).

## Basic usage

```python
from mm_toolbox.time import (
    time_ms,
    time_monotonic_ms,
    iso8601_to_unix,
    time_iso8601,
)

now_ms = time_ms()
elapsed_ms = time_monotonic_ms()

ts = iso8601_to_unix("2025-01-02T12:34:56Z")
stamp = time_iso8601(ts)
```

## Behavior notes

- Monotonic functions are preferred for durations and timeouts.
- Wall-clock functions are appropriate for timestamps and logging.
- ISO helpers expect/return seconds (float) and ISO-8601 strings.

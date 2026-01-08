# Rate Limiter

Token bucket rate limiter with optional per-second sub-buckets.

## Core concepts

- `RateLimiterConfig`: capacity, window size, sub-bucket strategy, and policies.
- `RateLimitStateConfig`: warning/block thresholds for utilization states.
- `RateLimitBurstConfig`: limited overage policy within the active second.
- `RateLimiter`: token bucket implementation with consumption helpers.

## Basic usage

```python
from mm_toolbox.rate_limiter import RateLimiter, RateLimiterConfig

config = RateLimiterConfig.default(capacity=10, window_s=1)
limiter = RateLimiter(config)

result = limiter.try_consume()
if result.allowed:
    remaining = result.remaining
```

## Factories

```python
from mm_toolbox.rate_limiter import RateLimiter

per_second = RateLimiter.per_second(5)
per_minute = RateLimiter.per_minute(120)
per_window = RateLimiter.per_window(20, 3)
```

## Behavior notes

- `SubBucketStrategy.PER_SECOND` caps per-second usage within a larger window.
- `RateLimiterConfig.default()` enables state thresholds and disables burst.
- `force=True` bypasses checks and yields `RateLimitState.OVERRIDE`.

## How it behaves

### Window refill model

- The limiter has a fixed `capacity` that refills every `window_s` seconds.
- Tokens are deducted on each consume call, and refilled only when the window elapses.
- Refill is aligned to the wall clock at creation/refill time (per-second buckets align
  to second boundaries).

### Sub-bucket strategy

- `DISABLED`: a single bucket for the full window.
- `PER_SECOND`: splits the window into `window_s` one-second buckets so each second
  receives an even share of the capacity.
- This prevents large bursts at the start of a window while still honoring the
  overall capacity.

### State thresholds

- `RateLimitState.NORMAL`: below `warning_threshold`.
- `RateLimitState.WARNING`: above `warning_threshold` but not past `block_threshold`.
- `RateLimitState.BLOCKED`: over `block_threshold`; consumption is denied.

### Burst policy

- When enabled, a limited number of over-capacity attempts can be allowed within
  the active second.
- This is a controlled escape hatch for spiky traffic; it does not increase
  the long-term capacity.

## Scenario examples

### Simple per-second limiter

```python
from mm_toolbox.rate_limiter import RateLimiter

limiter = RateLimiter.per_second(5)
for _ in range(5):
    assert limiter.try_consume().allowed is True
assert limiter.try_consume().allowed is False
```

### Per-second sub-buckets within a 2-second window

```python
from mm_toolbox.rate_limiter import RateLimiterConfig, RateLimiter, SubBucketStrategy

config = RateLimiterConfig.default(
    capacity=4,
    window_s=2,
    sub_bucket_strategy=SubBucketStrategy.PER_SECOND,
)
limiter = RateLimiter(config)

# In a single second, only 2 tokens are available (4 total / 2 seconds).
assert limiter.try_consume_multiple(2).allowed is True
assert limiter.try_consume().allowed is False
```

### Warning/block thresholds

```python
from mm_toolbox.rate_limiter import RateLimiterConfig, RateLimiter

config = RateLimiterConfig.default(capacity=4, window_s=1)
limiter = RateLimiter(config)

limiter.try_consume_multiple(2)  # usage 50%
state = limiter.try_consume().state  # usage 75%
```

### Burst allowance for spike handling

```python
from mm_toolbox.rate_limiter import (
    RateLimiterConfig,
    RateLimitBurstConfig,
    RateLimiter,
)

burst = RateLimitBurstConfig(is_enabled=True, max_tokens=2, max_burst_attempts=1)
config = RateLimiterConfig.default(capacity=2, window_s=2).replace(burst_config=burst)
limiter = RateLimiter(config)
```

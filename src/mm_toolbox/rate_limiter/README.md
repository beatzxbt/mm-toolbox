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

## Internal diagrams

### Bucket layout (overall + optional per-second)

Overall window bucket:

```text
window_s seconds
|<------------------------------->|
capacity = C tokens (refill on window boundary)
```

Optional per-second sub-buckets (`SubBucketStrategy.PER_SECOND`):

```text
capacity = C, window_s = W
q = C // W, r = C % W

sec 0   sec 1   ...  sec (r-1)  sec r   ...  sec (W-1)
 +----+  +----+       +----+     +----+       +----+
 |q+1|  |q+1|  ...   |q+1|     | q |  ...   | q |
 +----+  +----+       +----+     +----+       +----+
```

Notes:
- The first `r` seconds get one extra token each.
- If `W == 1` or sub-buckets are disabled, there is only a single bucket.

### Window and sub-bucket alignment

```text
time (ms)
t0             t0+1s          t0+2s          ...        t0+W
|---------------|---------------|-----------------------|
^ prev_refill   ^ per-second    ^ per-second            ^ next_refill

overall window: refills at t0+W
sub-buckets: refresh on the current second boundary
```

### Consumption decision flow

```text
+-----------------------+
| try_consume_multiple  |
+-----------------------+
            |
            v
+-----------------------+
| window expired?       |
+-----------------------+
   | yes            | no
   v                v
+-------------+  +----------------------+
| reset_state |  | num_tokens <= 0?     |
+-------------+  +----------------------+
                     | yes        | no
                     v            v
              +--------------+  +-----------+
              | allow, NORMAL|  | force?    |
              +--------------+  +-----------+
                                   | yes  | no
                                   v      v
                        +-------------------+  +---------------------------+
                        | apply usage,      |  | compute overall + sub usage|
                        | OVERRIDE          |  +---------------------------+
                        +-------------------+              |
                                                           v
                                              +------------------------+
                                              | overall_ok & sub_ok?   |
                                              +------------------------+
                                                | yes           | no
                                                v              v
                                       +------------------+   +-------------------------------+
                                       | usage > block?   |   | burst enabled & sub_enabled? |
                                       +------------------+   +-------------------------------+
                                        | yes       | no        | no           | yes
                                        v          v           v              v
                                +--------------+  +--------------------+  +-----------------+
                                | deny, BLOCKED|  | apply usage        |  | num_tokens >    |
                                +--------------+  +--------------------+  | max_tokens?     |
                                                     |                    +-----------------+
                                                     v                     | yes      | no
                                             +----------------------+      v          v
                                             | usage > warning?     | +--------------+ +--------------------------------+
                                             +----------------------+ | deny, BLOCKED| | burst_attempts_used < max?      |
                                              | yes        | no      +--------------+ +--------------------------------+
                                              v           v                              | yes               | no
                                       +--------------+ +--------------+                  v                  v
                                       | allow, WARNING| | allow, NORMAL|          +------------------+  +---------------+
                                       +--------------+ +--------------+          | allow, NORMAL     |  | deny, WARNING |
                                                                                | (cap usage)      |  +---------------+
                                                                                +------------------+
```

### Threshold bands

```text
usage (used / capacity)
0%            warning_threshold          block_threshold              100%+
|--------------------|--------------------------|------------------------|
NORMAL               WARNING                    BLOCKED
```

### Burst allowance within the active second

```text
per-second bucket capacity = 5, burst max_tokens = 2, max_burst_attempts = 1

second boundary
|--------------------------------------------------------------->
consume 5 tokens (normal) -> at capacity
1 burst attempt of up to 2 tokens -> allowed once
further attempts in same second -> denied
```

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

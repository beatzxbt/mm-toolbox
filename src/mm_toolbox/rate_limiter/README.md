# Rate Limiter

Token bucket rate limiter with optional per-second sub-buckets.

## Core concepts

- `RateLimiterConfig`: capacity, window size, sub-bucket strategy, and policies.
- `RateLimitStateConfig`: warning/block thresholds for utilization states.
- `RateLimitBurstConfig`: limited overage policy within the full window.
- `RateLimiter`: token bucket implementation with consumption helpers.

## Basic usage

```python
from mm_toolbox.rate_limiter import RateLimiter, RateLimiterConfig

config = RateLimiterConfig.default(capacity=10, window_s=1)
limiter = RateLimiter(config)

result = limiter.try_consume()
if result.allowed:
    remaining = result.remaining

# Consume multiple tokens at once
multi = limiter.try_consume_multiple(3)
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
- `RateLimiterConfig.default()` enables state thresholds, disables burst, and defaults to `SubBucketStrategy.PER_SECOND`.
- `force=True` bypasses policy checks and yields `RateLimitState.OVERRIDE`.
  Forced usage is charged in full, so remaining tokens can be negative and usage
  can exceed `1.0`. Counter overflow is rejected without changing accounting.
- Factory methods use the default 95% block threshold. Disable
  `config.state_config.is_enabled` before construction when the entire capacity
  should be available to normal requests.
- Zero or negative token counts are accounting queries: they consume nothing
  and return `allowed=True`, `state=NORMAL`, and the current remaining/usage.

## How it behaves

### Window refill model

- The limiter has a fixed `capacity` that refills every `window_s` seconds.
- Tokens are deducted on each consume call, and refilled only when the window elapses.
- Refill and sub-buckets use monotonic time relative to the current window
  start. They do not align to wall-clock second boundaries.
- Refill is lazy: the first operation after expiry starts a new window at that
  operation's timestamp. Idle time does not accumulate extra capacity.

### Sub-bucket strategy

- `DISABLED`: a single bucket for the full window.
- `PER_SECOND`: splits the window into `window_s` one-second buckets so each second
  receives an even share of the capacity.
- This prevents large bursts at the start of a window while still honoring the
  overall capacity.

### State thresholds

- `RateLimitState.NORMAL`: at or below `warning_threshold`.
- `RateLimitState.WARNING`: above `warning_threshold` but not past `block_threshold`.
- `RateLimitState.BLOCKED`: projected normal usage exceeds `block_threshold`;
  consumption is denied unless the request qualifies for a burst.

### Burst policy

- A burst budget lasts for the entire `window_s` window, including when
  per-second sub-buckets are enabled. Crossing a second boundary does not renew
  it; automatic or explicit window refill does.
- Burst handling applies when normal consumption would exceed overall capacity,
  the current sub-bucket allocation, or the block threshold.
- `max_tokens` limits the size of the entire burst request, and
  `max_burst_attempts` limits admitted burst requests per window. Normal requests
  and rejected requests do not spend an attempt.
- Accepted bursts return `NORMAL`. New charges stop at the bucket allocation;
  bursts preserve any higher usage already charged by `force=True`. The extra
  admitted work can exceed the configured normal capacity.
- A request larger than the burst request limit returns `BLOCKED`. An eligible
  request with no attempts left returns `WARNING` with `allowed=False`.

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
sub-buckets: each one-second boundary is relative to the window start
```

### Consumption decision order

1. Read monotonic time and refill if the full window has expired.
2. Return an unchanged accounting snapshot for a nonpositive token count.
3. For `force=True`, charge usage and return `OVERRIDE`, unless a counter would
   overflow.
4. If capacity, sub-bucket and block-threshold checks all pass, charge usage and
   return `NORMAL` or `WARNING` according to the warning threshold.
5. Otherwise, apply the shared burst request-size and per-window attempt limits.
   An accepted burst returns `NORMAL`; a rejection preserves accounting.

### Threshold bands

```text
usage (used / capacity)
0%            warning_threshold          block_threshold              100%+
|--------------------|--------------------------|------------------------|
NORMAL               WARNING                    BLOCKED
```

### Burst allowance within the full window

```text
overall capacity = 10, window = 2 seconds, sub-buckets = [5, 5]
burst max_tokens = 2, max_burst_attempts = 1, state thresholds disabled

window start                       second 1                 window end
|----------------------------------|------------------------|
consume 5 normal + 1 burst token    consume up to 4 normal    all budgets reset
one burst attempt is spent         no new burst attempt

The burst token counts toward the overall window budget. Its sub-bucket
charge stops at 5. Once normal capacity is exhausted, an unused burst attempt
can admit extra work while keeping normal counters capped.
```

## Scenario examples

### Simple per-second limiter

```python
from mm_toolbox.rate_limiter import RateLimiter, RateLimiterConfig

config = RateLimiterConfig.default(capacity=5, window_s=1)
config.state_config.is_enabled = False
limiter = RateLimiter(config)
for _ in range(5):
    assert limiter.try_consume().allowed is True
assert limiter.try_consume().allowed is False
```

### Per-second sub-buckets within a 2-second window

```python
from mm_toolbox.rate_limiter import RateLimiterConfig, RateLimiter

config = RateLimiterConfig.default(capacity=4, window_s=2)
limiter = RateLimiter(config)

# In a single second, only 2 tokens are available (4 total / 2 seconds).
assert limiter.try_consume_multiple(2).allowed is True
assert limiter.try_consume().allowed is False
```

### Warning/block thresholds

```python
from mm_toolbox.rate_limiter import RateLimiterConfig, RateLimiter

config = RateLimiterConfig.default(capacity=4, window_s=1)
config.state_config.warning_threshold = 0.5
config.state_config.block_threshold = 0.75
limiter = RateLimiter(config)

limiter.try_consume_multiple(2)  # 50%: allowed, NORMAL
limiter.try_consume()  # 75%: allowed, WARNING
limiter.try_consume()  # projected 100%: denied, BLOCKED; usage remains 75%
```

### Burst allowance for spike handling

```python
from mm_toolbox.rate_limiter import (
    RateLimiterConfig,
    RateLimitBurstConfig,
    RateLimiter,
)

burst = RateLimitBurstConfig(is_enabled=True, max_tokens=2, max_burst_attempts=1)
config = RateLimiterConfig.default(capacity=2, window_s=2)
config.burst_config = burst
limiter = RateLimiter(config)
```

## Benchmark results (2026-11-04)

The p50 values below are nanoseconds per call from 100,000 measured samples
per operation after 1,000 warmups. The configured bucket had capacity 1,000,
a 60-second window, per-second sub-buckets, state tracking and burst handling
enabled (10 burst tokens, five attempts).
The timed burst call is asserted to be admitted; setup uses a forced charge to
reach the over-capacity case without spending a burst attempt.

| Operation | p50 (ns/call) |
| --- | ---: |
| tokens_remaining | 125 |
| try_consume_allowed | 208 |
| try_consume_burst | 167 |
| try_consume_denied | 208 |
| try_consume_multiple | 208 |

Measured on CPython 3.14.4, NumPy 2.5.4, Cython 3.3.0, Apple clang 21.0.0,
macOS 26.6.2 arm64, Apple M2 (8 CPU cores, 16 GiB), at revision `19bbaf9`.
These are local measurements and do not establish performance on other machines.

Reproduce from the repository root:

```sh
PYTHONPATH=src .venv/bin/python -u benchmarks/rate_limiter/benchmark_rate_limiter.py --operations 100000 --warmup 1000 --capacity 1000 --window-s 60 --sub-bucket per_second --state --burst --max-burst-tokens 10 --max-burst-attempts 5 --consume-tokens 1
```

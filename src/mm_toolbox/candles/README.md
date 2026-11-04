# Candles

High-performance candle aggregation with multiple bucket styles. The module
ingests trades, maintains a rolling ring buffer of completed candles, and
exposes a live, in-progress `latest_candle`.

## Core concepts

- `Trade`: time_ms, side, price, size (a `msgspec.Struct`)
- `Candle`: OHLC, buy/sell size + volume, VWAP (`vwap`), trade count, trade list (a `msgspec.Struct`)
- `BaseCandles`: shared ring buffer, initialization, async notification

## Basic usage

```python
from mm_toolbox.candles import Trade, TimeCandles

candles = TimeCandles(secs_per_bucket=1.0, num_candles=1000, store_trades=True)
candles.process_trade(
    Trade(time_ms=1712150000000, is_buy=True, price=100.25, size=0.1)
)

latest = candles.latest_candle
history = list(candles)
```

All candle implementations also support async iteration to wait for completed candles:

```python
async for candle in candles:
    handle(candle)
```

## Architecture overview

Trades flow into a candle aggregator, which updates the live candle and pushes
completed candles into a ring buffer.

```
┌──────────────┐
│ Trade Stream │
└──────┬───────┘
       │ Trade(time_ms, side, price, size)
       ▼
┌──────────────────────┐
│ Candle Aggregator    │
│ Time/Tick/Volume/... │
└──────┬───────────────┘
       │ update + boundary check
       ├───────────────────────────────┐
       ▼                               ▼
┌──────────────────────┐     ┌──────────────────┐
│ latest_candle (live) │     │ Ring Buffer      │
│ in-progress candle   │     │ completed candles│
└──────────────────────┘     └──────────────────┘
```

## Candle families

### TimeCandles
- Trigger: fixed time windows (`secs_per_bucket`)
- Use for: time-aligned analytics and time-series indicators
- Notes: no empty candles are emitted when no trades occur

### TickCandles
- Trigger: fixed trade count (`ticks_per_bucket`)
- Use for: smoothing bursty markets by trade activity
- Notes: candle duration varies with market activity

### VolumeCandles
- Trigger: fixed total size (`volume_per_bucket`)
- Use for: volume-normalized analysis and flow studies
- Notes: trades that cross a boundary carry remainder into the next candle

### PriceCandles
- Trigger: price moves by `price_bucket` away from the candle open
- Use for: volatility-aware sampling and regime shifts
- Notes: candle duration is variable and tied to price movement

### MultiCandles
- Trigger: first of time, ticks, or size thresholds
  (`max_duration_secs`, `max_ticks`, `max_size`)
- Use for: bounding candle size and duration at once
- Notes: more knobs to tune, but keeps extremes in check

## Behavior notes

- Stale trades (time_ms older than the last accepted trade) are ignored.
- `num_candles` requests history capacity, rounded up to a power of two by the
  ring buffer; older candles are dropped once that capacity is reached.
- Closed candles support constant-time positive and negative indexing and
  iteration from oldest to newest, without copying the full history per access.
- `store_trades=True` keeps the trade list in each candle; tune `num_candles` for memory.
- `initialize(trades)` explicitly sets the baseline from a non-empty list of `Trade` objects.
- `Trade` and `Candle` are lightweight `msgspec.Struct` instances.
- Their constructors do not validate field types or ranges. Ingestion rejects
  nonfinite timestamps, prices, sizes or notionals, and negative sizes before
  advancing stream time. Equal timestamps, zero sizes and negative prices are
  allowed. Tick, time and price candles count zero-size trades; volume and multi
  candles advance stream time but omit those trades from candle statistics.

## Delivery and replay guarantees

Time buckets are anchored to the first accepted trade, so historical replay is
independent of construction time. Reinitializing resets that anchor and cancels
pending waits. Trades older than the last accepted stream timestamp are ignored,
including immediately after a bucket closes; equal timestamps are accepted.

Async iteration broadcasts a new closure to all currently waiting consumers.
Otherwise closures wait in FIFO order, separately from retained history. Pending
delivery is bounded by `num_candles`; an overrun raises `RuntimeError` on the next
await before retained candles are delivered. Consumers requiring continuity must
handle that error. Cancelling a wait does not consume a future candle.

Volume and multi-trigger thresholds measure trade size, not price-times-size.
Splits preserve size and notional, including quantities below `1e-12`. Nonfinite
limits and invalid quantities fail explicitly; a threshold too small to make
floating-point progress for an input trade raises rather than looping forever.

## Benchmark results (2026-11-04)

Each p50 is over 10 measured batches after two warmups. Each batch creates a
fresh candle and processes 100,000 seeded synthetic trades with 1,000 candles
and trade storage enabled. Throughput is derived from the median batch time.

| Candle type | p50 (ns/trade) | Throughput at p50 (trades/s) |
| --- | ---: | ---: |
| TimeCandles | 263.9 | 3,789,009 |
| TickCandles | 281.4 | 3,553,687 |
| VolumeCandles | 331.7 | 3,015,041 |
| PriceCandles | 284.3 | 3,517,731 |
| MultiCandles | 401.7 | 2,489,505 |

Measured on CPython 3.14.4, NumPy 2.5.4, Cython 3.3.0, Apple clang 21.0.0,
macOS 26.6.2 arm64, Apple M2 (8 CPU cores, 16 GiB), at revision `c5f9dba`.
These are local measurements and do not establish performance on other machines.

Reproduce from the repository root:

```sh
.venv/bin/python -u benchmarks/candles/benchmark_candles.py --operations 10 --warmup 2 --trade-count 100000 --num-candles 1000 --store-trades
```

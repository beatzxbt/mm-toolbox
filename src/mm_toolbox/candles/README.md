# Candles

High-performance candle aggregation with multiple bucket styles. The module
ingests trades, maintains a rolling ring buffer of completed candles, and
exposes a live, in-progress `latest_candle`.

## Core concepts

- `Trade`: time_ms, side, price, size
- `Candle`: OHLC, buy/sell size + volume, VWAP (`vwap`), trade count, trade list
- `BaseCandles`: shared ring buffer, initialization, async notification

## Basic usage

```python
from mm_toolbox.candles import Trade, TimeCandles

candles = TimeCandles(secs_per_bucket=1.0, num_candles=1000)
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

- Stale trades (time_ms older than the last candle close) are ignored.
- `num_candles` caps the ring buffer; older candles are dropped.
- Each candle stores a list of trades; tune `num_candles` for memory.
- `initialize(trades)` expects a non-empty list of `Trade` objects.

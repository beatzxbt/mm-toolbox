# WebSocket

WebSocket connection management built on PicoWs with byte ring buffers and
latency tracking.

## Core components

- `WsConnection`: low-level connection wrapper with ping/pong latency tracking.
- `WsSingle`: convenience wrapper for a single connection.
- `WsPool`: manages multiple connections and selects the fastest for sends.
- `ConnectionState`: connection state enum (`DISCONNECTED`, `CONNECTING`, `CONNECTED`).

## Architecture overview

Connections ingest frames, push text messages into a ring buffer, and expose
bytes via async iteration or callbacks.

```
┌──────────────┐
│ WSS Endpoint │
└──────┬───────┘
       │ frames
       ▼
┌──────────────────────┐
│ WsConnection         │
│ ping/pong tracker    │
│ latency EMA (ms)     │
└──────┬───────────────┘
       │ TEXT frames -> bytes
       ▼
┌──────────────────────┐
│ BytesRingBuffer      │
└──────┬───────────────┘
       │ async consume
       ▼
     ┌───────────────────────────┬───────────────────────────┐
     │                           │                           │
     ▼                           ▼                           ▼
┌──────────────────┐    ┌──────────────────┐       ┌──────────────────┐
│ WsSingle         │    │ WsPool           │       │ Latency Tracker  │
│ one connection   │    │ many connections │       │ per connection   │
│ no hash filter   │    │ hash-history filter│     │ choose fastest   │
└──────────────────┘    └──────────────────┘       └──────────────────┘
```

Pool connections share a payload ring buffer and apply duplicate filtering
through a separate bounded hash-history window (`xxhash64`). This keeps hash
state independent from payload consumption, so duplicates are suppressed even
if the first copy has already been consumed by downstream code.

## Configuration

```python
from mm_toolbox.websocket import WsConnectionConfig

config = WsConnectionConfig.default(
    wss_url="wss://fstream.binance.com/ws/btcusdt@bookTicker",
    on_connect=[b'{"method":"SUBSCRIBE","params":["btcusdt@bookTicker"],"id":1}'],
    auto_reconnect=True,
    max_frame_size=1_048_576,
    latency_ping_interval_ms=1000,
)
```

- `max_frame_size`: maximum allowed frame size in bytes (default 1 MB).
- `latency_ping_interval_ms`: interval between internal latency pings in milliseconds (default 1000).

## Quick start

### Single connection

```python
import asyncio
import msgspec
from mm_toolbox.websocket import WsSingle, WsConnectionConfig

def on_message(msg: bytes) -> None:
    print(msgspec.json.decode(msg))

async def main() -> None:
    config = WsConnectionConfig.default(
        wss_url="wss://fstream.binance.com/ws/btcusdt@bookTicker"
    )
    ws = WsSingle(config, on_message=on_message)
    await ws.start()

asyncio.run(main())
```

### Connection pool

```python
import asyncio
import msgspec
from mm_toolbox.websocket import WsPool, WsConnectionConfig, WsPoolConfig

def on_message(msg: bytes) -> None:
    print(msgspec.json.decode(msg))

async def main() -> None:
    config = WsConnectionConfig.default(
        wss_url="wss://fstream.binance.com/ws/btcusdt@bookTicker"
    )
    pool_config = WsPoolConfig.default()
    pool = await WsPool.new(config, on_message=on_message, pool_config=pool_config)

    async with pool:
        async for msg in pool:
            on_message(msg)

asyncio.run(main())
```

## When to use which

- `WsSingle` is best when you only need one connection and want a simple
  callback or async-iterator interface.
- `WsPool` is best when you want latency hedging, multiple subscriptions,
  or faster sends by routing through the lowest-latency connection. Sends default
  to the fastest connection; broadcast to all connections is also available.

## Behavior notes

- `wss_url` must start with `ws://` or `wss://`.
- Callbacks must accept a single `bytes` argument and annotate it as `bytes`.
- Complete TEXT and BINARY messages are inserted into the ring buffer.
- Pool startup is all-or-nothing: every configured socket must be live before
  entry succeeds. A failed open cancels pending opens, closes successful opens,
  and raises RuntimeError with the connection error as its cause.
- After successful startup, `auto_reconnect=True` restores missing sockets,
  including after a complete outage. At most one retry task runs per missing
  socket, with exponential backoff from 1 to 30 seconds. With reconnect disabled,
  lost sockets stay disconnected and periodic eviction is disabled.
- Pool state is `CONNECTING` during startup or a complete reconnecting outage,
  `CONNECTED` while any successfully started socket is live, and `DISCONNECTED`
  after close or a complete outage with reconnect disabled.
- The fastest socket is checked again before sending. Disconnected payload sends
  raise `RuntimeError`; transport errors propagate. Sends are never automatically
  retried because a failed write may already have reached the peer. A successful
  send only establishes transport acceptance, not application acknowledgement.
- `WsSingle.start()` retries initial failures when reconnect is enabled. Context
  entry makes one initial attempt and raises promptly on failure; after entry,
  both callback and iterator sessions monitor disconnects for recovery.
- `WsSingle.close()` is permanent and idempotent. It cancels connection, reconnect,
  and callback work, and wakes idle readers. Iteration drains already buffered
  messages before raising `StopAsyncIteration`. Create a new single client for a
  new session. Context exit also waits for owned tasks to terminate.
- Missing PONGs for three ping intervals disconnect the socket so the configured
  reconnect path can replace a half-open connection.
- Pool eviction periodically replaces slower connections based on latency.
- Receive buffers retain the newest 128 messages. Overflow discards the oldest
  payload. `WsConnection.get_dropped_messages()` counts exact overwrites by that
  listener; `WsSingle.get_dropped_messages()` and `WsPool.get_dropped_messages()`
  count exact overwrites throughout the wrapper's lifetime, including startup,
  failed handshakes, reconnects and retired pool sockets. Close does not reset
  these counters. Pool counts describe the shared buffer before deduplication;
  an overwritten redundant copy still counts, even if another copy arrived.
- Overwrite warnings through `mm_toolbox.websocket.connection` include the
  listener ID and count, sampled at counts 1, 2, 4, 8, ... to limit logging under
  overload. Read the counters for exact totals; warning lines are not loss counts.
- Consumers that require continuity must check loss from the initial zero
  baseline before applying buffered startup messages and monitor disconnects
  even when no messages arrive. Hash deduplication and reconnection do not
  establish exchange sequence continuity. A fresh orderbook snapshot cannot
  repair already missing trades, candles or moving-average inputs.
- The standalone market-data examples stop on receive/SHM loss, rejected SHM
  writes, sequence gaps and disconnects. They disable automatic reconnect and
  propagate child failure to the parent, which exits nonzero. Restart the whole
  example to rebuild its state. Their bounded buffers keep overwrite-oldest
  semantics; they do not promise loss-free throughput.

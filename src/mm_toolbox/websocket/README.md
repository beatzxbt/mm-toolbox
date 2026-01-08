# WebSocket

WebSocket connection management built on PicoWs with byte ring buffers and
latency tracking.

## Core components

- `WsConnection`: low-level connection wrapper with ping/pong latency tracking.
- `WsSingle`: convenience wrapper for a single connection.
- `WsPool`: manages multiple connections and selects the fastest for sends.

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
│ no dedup         │    │ ringbuffer dedup │       │ choose fastest   │
└──────────────────┘    └──────────────────┘       └──────────────────┘
```

Pool connections share a ring buffer configured with `only_insert_unique=True`
to avoid duplicate frames from overlapping subscriptions.

## Configuration

```python
from mm_toolbox.websocket import WsConnectionConfig

config = WsConnectionConfig.default(
    wss_url="wss://fstream.binance.com/ws/btcusdt@bookTicker",
    on_connect=[b'{"method":"SUBSCRIBE","params":["btcusdt@bookTicker"],"id":1}'],
    auto_reconnect=True,
)
```

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
  or faster sends by routing through the lowest-latency connection.

## Behavior notes

- `wss_url` must start with `wss://`.
- Callbacks must accept a single `bytes` argument and annotate it as `bytes`.
- Only TEXT frames are inserted into the ring buffer.
- Auto-reconnect loops recreate connections on disconnects.
- Pool eviction periodically replaces slower connections based on latency.

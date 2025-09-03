"""High-performance trading and market data toolkit."""
# Temporarily commented out due to Python 3.13 compilation issues
# from .candles.price import PriceCandles as PriceCandles
# from .candles.volume import VolumeCandles as VolumeCandles
# from .candles.tick import TickCandles as TickCandles
# from .candles.time import TimeCandles as TimeCandles
# from .candles.multi import MultiCandles as MultiCandles

from .candles import (
    MultiCandles as MultiCandles,
)
from .candles import (
    PriceCandles as PriceCandles,
)
from .candles import (
    TickCandles as TickCandles,
)
from .candles import (
    TimeCandles as TimeCandles,
)
from .candles import (
    VolumeCandles as VolumeCandles,
)
from .logging.standard import (
    BaseLogHandler as BaseLogHandler,
)
from .logging.standard import (
    DiscordLogHandler as DiscordLogHandler,
)
from .logging.standard import (
    FileLogHandler as FileLogHandler,
)
from .logging.standard import (
    Logger as Logger,
)
from .logging.standard import (
    LoggerConfig as LoggerConfig,
)
from .logging.standard import (
    LogLevel as LogLevel,
)
from .logging.standard import (
    TelegramLogHandler as TelegramLogHandler,
)
from .moving_average import (
    ExponentialMovingAverage as ExponentialMovingAverage,
)

# NOTE: For accessing advanced logging, it is only accessible by
#       doing '.logging.advanced'. This prevents any
#       naming clashes with the standard logger.
#
# from .logging.advanced import (
#     LogLevel as LogLevel,
#     LoggerConfig as LoggerConfig,
#     WorkerLogger as WorkerLogger,
#     MasterLogger as MasterLogger,
#     FileLogHandler as FileLogHandler,
#     DiscordLogHandler as DiscordLogHandler,
#     TelegramLogHandler as TelegramLogHandler,
#     BaseLogHandler as BaseLogHandler,
#     ZMQLogHandler as ZMQLogHandler,
#     TestLogHandler as TestLogHandler,
# )
from .moving_average import (
    SimpleMovingAverage as SimpleMovingAverage,
)
from .moving_average import (
    TimeExponentialMovingAverage as TimeExponentialMovingAverage,
)
from .moving_average import (
    WeightedMovingAverage as WeightedMovingAverage,
)
from .orderbook import (
    Orderbook as Orderbook,
)
from .orderbook import (
    OrderbookLevel as OrderbookLevel,
)
from .ringbuffer import (
    BytesRingBuffer as BytesRingBuffer,
)
from .ringbuffer import (
    GenericRingBuffer as GenericRingBuffer,
)
from .ringbuffer import (
    NumericRingBuffer as NumericRingBuffer,
)
from .rounding import (
    Rounder as Rounder,
)
from .rounding import (
    RounderConfig as RounderConfig,
)
from .time import (
    iso8601_to_unix as iso8601_to_unix,
)
from .time import (
    time_iso8601 as time_iso8601,
)
from .time import (
    time_ms as time_ms,
)
from .time import (
    time_ns as time_ns,
)
from .time import (
    time_s as time_s,
)
from .time import (
    time_us as time_us,
)
from .weights import (
    ema_weights as ema_weights,
)
from .weights import (
    geometric_weights as geometric_weights,
)
from .weights import (
    logarithmic_weights as logarithmic_weights,
)

# from .websocket import (
#     WsConnectionConfig as WsConnectionConfig,
#     ConnectionState as ConnectionState,
#     LatencyTrackerState as LatencyTrackerState,
#     WsConnectionState as WsConnectionState,
#     WsConnection as WsConnection,
#     WsSingle as WsSingle,
#     WsPool as WsPool,
#     WsPoolConfig as WsPoolConfig,
# )


__all__ = [
    # Logging
    "Logger",
    "LoggerConfig",
    "FileLogHandler",
    "DiscordLogHandler",
    "TelegramLogHandler",
    # Orderbook
    "Orderbook",
    "OrderbookLevel",
    # Weights
    "ema_weights",
    "geometric_weights",
    "logarithmic_weights",
    # Working modules (compiled for Python 3.13):
    "BytesRingBuffer",
    "GenericRingBuffer",
    "NumericRingBuffer",
    "Rounder",
    "RounderConfig",
    "time_s",
    "time_ms",
    "time_us",
    "time_ns",
    "time_iso8601",
    "iso8601_to_unix",
    # Working modules (compiled for Python 3.13):
    "PriceCandles",
    "VolumeCandles",
    "TickCandles",
    "TimeCandles",
    "MultiCandles",
    "SimpleMovingAverage",
    "ExponentialMovingAverage",
    "WeightedMovingAverage",
    "TimeExponentialMovingAverage",
    # "WsConnectionConfig", "ConnectionState", "LatencyTrackerState", "WsConnectionState",
    # "WsConnection", "WsSingle", "WsPool", "WsPoolConfig",
]

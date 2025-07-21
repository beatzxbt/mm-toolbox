from mm_toolbox.candles import (
    PriceCandles as PriceCandles,
    VolumeCandles as VolumeCandles,
    TickCandles as TickCandles,
    TimeCandles as TimeCandles,
    MultiCandles as MultiCandles,
)

from mm_toolbox.logging.standard import (
    Logger as Logger,
    LoggerConfig as LoggerConfig,
    BaseLogHandler as BaseLogHandler,
    FileLogHandler as FileLogHandler,
    DiscordLogHandler as DiscordLogHandler,
    TelegramLogHandler as TelegramLogHandler,
)

# NOTE: For accessing advanced logging, it is only accessible by 
#       doing 'mm_toolbox.logging.advanced'. This prevents any 
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
    ExponentialMovingAverage as ExponentialMovingAverage,
    WeightedMovingAverage as WeightedMovingAverage,
    TimeExponentialMovingAverage as TimeExponentialMovingAverage,
)

# NOTE: For accessing Numba compatible methods, it is only accessible by 
#       doing 'mm_toolbox.numba.*'. This prevents any naming clashes with 
#       the rest of the toolbox's API.
#
# from .numba import (
#     TickCandles as TickCandles,
#     VolumeCandles as VolumeCandles,
#     TimeCandles as TimeCandles,
#     MultiCandles as MultiCandles,
#     PriceCandles as PriceCandles,
#     SimpleMovingAverage as SimpleMovingAverage,
#     ExponentialMovingAverage as ExponentialMovingAverage,
#     WeightedMovingAverage as WeightedMovingAverage,
#     # TimeExponentialMovingAverage as TimeExponentialMovingAverage,
#     Orderbook as Orderbook,
#     RingBufferOneDim as RingBufferOneDim,
#     RingBufferTwoDim as RingBufferTwoDim,
#     # RingBufferMulti as RingBufferMulti,
#     ema_weights as ema_weights,
#     geometric_weights as geometric_weights,
#     logarithmic_weights as logarithmic_weights,
# )

from .ringbuffer import (
    RingBufferOneDim as RingBufferOneDim,
    RingBufferTwoDim as RingBufferTwoDim,
    # RingBufferMulti as RingBufferMulti,
)
from .rounding import (
    Round as Round,
)
from .weights import (
    ema_weights as ema_weights,
    geometric_weights as geometric_weights,
    logarithmic_weights as logarithmic_weights,
)

from .orderbook import Orderbook as Orderbook

from .ringbuffer import (
    RingBufferOneDim as RingBufferOneDim,
    RingBufferTwoDim as RingBufferTwoDim,
    RingBufferMulti as RingBufferMulti,
)

from .rounding import Round as Round

from src.mm_toolbox.time.time import (
    time_s as time_s,
    time_ms as time_ms,
    time_us as time_us,
    time_ns as time_ns,
    time_iso8601 as time_iso8601,
    iso8601_to_unix as iso8601_to_unix,
    unix_to_iso8601 as unix_to_iso8601,
)

# from .websocket import (
#     VerifyWsPayload as VerifyWsPayload,
#     parse_raw_orderbook_data as parse_raw_orderbook_data,
#     WsSingle as WsSingle,
#     WsPool as WsPool,
#     WsPoolEvictionPolicy as WsPoolEvictionPolicy,
# )

from .weights import (
    ema_weights as ema_weights,
    geometric_weights as geometric_weights,
    logarithmic_weights as logarithmic_weights,
)

__all__ = [
    #     "TickCandles",
    #     "VolumeCandles",
    #     "TimeCandles",
    #     "MultiTriggerCandles",
    #     "Logger",
    #     "LoggerConfig",
    #     "FileLogConfig",
    #     "DiscordLogConfig",
    #     "TelegramLogConfig",
    "SimpleMovingAverage",
    "ExponentialMovingAverage",
    "WeightedMovingAverage",
    #     "Orderbook",
    "RingBufferOneDim",
    "RingBufferTwoDim",
    "RingBufferMulti",
    "Round",
    "time_s",
    "time_ms",
    "time_us",
    "time_ns",
    "time_iso8601",
    "iso8601_to_unix",
    #     "SingleWsConnection",
    #     "WsPoolEvictionPolicy",
    #     "WsStandard",
    #     "WsFast",
    #     "VerifyWsPayload",
    "ema_weights",
    "geometric_weights",
    "logarithmic_weights",
]

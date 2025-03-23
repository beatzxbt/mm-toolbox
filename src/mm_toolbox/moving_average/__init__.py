from .sma import SimpleMovingAverage as SimpleMovingAverage
from .ema import ExponentialMovingAverage as ExponentialMovingAverage
from .wma import WeightedMovingAverage as WeightedMovingAverage
from .tema import TimeExponentialMovingAverage as TimeExponentialMovingAverage

__all__ = [
    "SimpleMovingAverage", 
    "ExponentialMovingAverage", 
    "WeightedMovingAverage", 
    "TimeExponentialMovingAverage"
]
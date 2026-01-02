"""Core benchmark framework for mm_toolbox.

Provides reusable components for benchmark implementation:
- stats: Statistical analysis and percentile computation
- reporting: Report formatting and output
- runner: Base classes for benchmark execution
- cli: Command-line interface utilities
"""

from __future__ import annotations

__all__ = [
    "OperationMetrics",
    "BenchmarkStatistics",
    "BenchmarkReporter",
    "ComparativeReporter",
    "BaseBenchmarkConfig",
    "BenchmarkRunner",
    "MultiSizeBenchmarkRunner",
    "BenchmarkCLI",
]

from .cli import BenchmarkCLI
from .reporting import BenchmarkReporter, ComparativeReporter
from .runner import BaseBenchmarkConfig, BenchmarkRunner, MultiSizeBenchmarkRunner
from .stats import BenchmarkStatistics, OperationMetrics

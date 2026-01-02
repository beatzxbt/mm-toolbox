"""Statistical analysis utilities for benchmarks.

Provides unified statistics collection and percentile computation for all benchmarks.
Replaces duplicated OperationStats/OperationMetric classes across benchmark files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class OperationMetrics:
    """Metrics for a single operation type.

    Tracks latencies and optional metadata (e.g., batch sizes, level counts).
    Provides percentile computation for performance analysis.

    Args:
        name: Operation name for identification.
        latencies_ns: List of latency measurements in nanoseconds.
        metadata: Flexible metadata tracking (e.g., num_levels, batch_size).
    """

    name: str
    latencies_ns: list[int] = field(default_factory=list)
    metadata: dict[str, list[Any]] = field(default_factory=dict)

    def add_latency(self, latency_ns: int, **meta: int | float) -> None:
        """Add a latency measurement with optional metadata.

        Args:
            latency_ns: Latency in nanoseconds.
            **meta: Optional metadata (e.g., num_levels=10, batch_size=100).
        """
        self.latencies_ns.append(latency_ns)
        for key, value in meta.items():
            if key not in self.metadata:
                self.metadata[key] = []
            self.metadata[key].append(value)

    def compute_percentiles(self) -> dict[str, float]:
        """Compute percentile statistics.

        Returns:
            Dictionary with count, mean, p10, p25, p50, p95, p99, p99_9, ops_per_sec.
        """
        if not self.latencies_ns:
            return {
                "count": 0,
                "mean": 0.0,
                "p10": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "p99_9": 0.0,
                "ops_per_sec": 0.0,
            }

        arr = np.array(self.latencies_ns, dtype=np.float64)
        mean_ns = float(np.mean(arr))
        ops_per_sec = 1e9 / mean_ns if mean_ns > 0 else 0

        return {
            "count": len(self.latencies_ns),
            "mean": mean_ns,
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "p99_9": float(np.percentile(arr, 99.9)),
            "ops_per_sec": ops_per_sec,
        }


@dataclass
class BenchmarkStatistics:
    """Aggregated benchmark statistics container.

    Tracks all operation metrics and provides aggregate statistics.
    Replaces BenchmarkResults and BenchmarkStats classes.

    Args:
        operations: Dictionary mapping operation names to their metrics.
        total_time_ns: Total benchmark execution time in nanoseconds.
    """

    operations: dict[str, OperationMetrics] = field(default_factory=dict)
    total_time_ns: int = 0

    def add_operation(self, name: str) -> OperationMetrics:
        """Add a new operation for tracking.

        Args:
            name: Operation name.

        Returns:
            The newly created OperationMetrics instance.
        """
        metrics = OperationMetrics(name=name)
        self.operations[name] = metrics
        return metrics

    def get_operation(self, name: str) -> OperationMetrics | None:
        """Get existing operation metrics.

        Args:
            name: Operation name.

        Returns:
            OperationMetrics if found, None otherwise.
        """
        return self.operations.get(name)

    @property
    def total_operations(self) -> int:
        """Total number of measured operations."""
        return sum(len(op.latencies_ns) for op in self.operations.values())

    @property
    def throughput(self) -> float:
        """Overall throughput in ops/sec."""
        total_time_s = self.total_time_ns / 1e9
        return self.total_operations / total_time_s if total_time_s > 0 else 0

"""Logging benchmark failures cannot be reported as successful delivery."""

from queue import Queue
from types import SimpleNamespace

import pytest

from benchmarks.logging import benchmark_advanced_logging as benchmark


def test_worker_reports_constructor_failure_and_aborts_startup(monkeypatch):
    aborted = []

    def fail_construct(**kwargs):
        raise RuntimeError("injected constructor failure")

    monkeypatch.setattr(benchmark, "WorkerLogger", fail_construct)
    results = Queue()
    benchmark._worker_emit_burst(
        "path",
        "W0",
        32,
        0,
        1,
        SimpleNamespace(abort=lambda: aborted.append(True)),
        None,
        1,
        results,
    )
    result = results.get_nowait()
    assert result.name == "W0"
    assert "constructor failure" in result.error
    assert result.produced_count == 0
    assert aborted == [True]


@pytest.mark.parametrize("shutdown_error", [False, True])
def test_worker_status_is_emitted_after_shutdown_with_exact_transport_stats(
    monkeypatch, shutdown_error
):
    events, results = [], Queue()

    class Worker:
        def __init__(self, **kwargs):
            pass

        def shutdown(self):
            events.append("shutdown")
            assert results.empty()
            if shutdown_error:
                raise RuntimeError("injected final flush failure")

        def get_transport_stats(self):
            return {
                "rejected_messages": 2,
                "transport_errors": 1,
                "overwritten_batches": 3,
            }

    monkeypatch.setattr(benchmark, "WorkerLogger", Worker)
    benchmark._worker_emit_burst(
        "path",
        "W0",
        32,
        0,
        1,
        SimpleNamespace(wait=lambda **kwargs: None),
        SimpleNamespace(wait=lambda **kwargs: True),
        1,
        results,
    )
    result = results.get_nowait()
    assert events == ["shutdown"]
    assert bool(result.error) == shutdown_error
    assert result.rejected_messages == 2
    assert result.transport_errors == 1
    assert result.overwritten_batches == 3


@pytest.mark.parametrize(
    "result,exitcode",
    [
        (None, 0),
        (benchmark.WorkerResult("W0", 100, 10), 1),
        (benchmark.WorkerResult("W0", 100, 10, error="failed shutdown"), 0),
    ],
)
def test_parent_counts_missing_failed_and_nonzero_after_success_results(
    result, exitcode
):
    results = Queue()
    if result is not None:
        results.put(result)
    received, failed = benchmark._collect_worker_results(
        [SimpleNamespace(name="W0", exitcode=exitcode)],
        results,
        0 if result is None else 1,
    )
    assert failed == 1
    assert len(received) == (result is not None)


@pytest.mark.parametrize(
    "invalid", [("ok", 100, 10), benchmark.WorkerResult("unknown", 100, 10)]
)
def test_parent_rejects_malformed_or_unowned_results(invalid):
    results = Queue()
    results.put(invalid)
    with pytest.raises(RuntimeError, match="Invalid logging worker result"):
        benchmark._collect_worker_results(
            [SimpleNamespace(name="W0", exitcode=0)], results, 1
        )


@pytest.mark.parametrize(
    "failure",
    [
        "worker_errors",
        "forced_terminations",
        "drain_timeouts",
        "transport_errors",
        "rejected_messages",
    ],
)
def test_invalid_delivery_fails_validation_while_overwrite_alone_is_allowed(failure):
    sample = benchmark.ScenarioSample(
        1, 256, 10, 100, 70, 0, 0, 0, overwritten_batches=3
    )
    benchmark._require_valid_delivery(sample)
    setattr(sample, failure, 1)
    with pytest.raises(RuntimeError, match="Invalid logging benchmark"):
        benchmark._require_valid_delivery(sample)


def test_missing_delivery_without_recorded_overwrite_fails_validation():
    sample = benchmark.ScenarioSample(1, 256, 10, 100, 70, 0, 0, 0)
    with pytest.raises(RuntimeError, match="unexplained_messages=30"):
        benchmark._require_valid_delivery(sample)


def test_aggregation_cannot_hide_unexplained_loss_behind_another_runs_overwrite():
    overwritten = benchmark.ScenarioSample(
        1, 256, 10, 100, 70, 0, 0, 0, overwritten_batches=3
    )
    unexplained = benchmark.ScenarioSample(1, 256, 10, 100, 90, 0, 0, 0)
    summary = benchmark._summarize_samples(256, [overwritten, unexplained])
    assert summary.overwritten_batches == 3
    assert summary.unexplained_messages == 10
    with pytest.raises(RuntimeError, match="unexplained_messages=10"):
        benchmark._require_valid_delivery(summary)


def test_parent_rejects_duplicate_status_before_all_workers_report():
    results = Queue()
    results.put(benchmark.WorkerResult("W0", 100, 10))
    results.put(benchmark.WorkerResult("W0", 100, 10))
    processes = [SimpleNamespace(name=name, exitcode=0) for name in ("W0", "W1")]
    with pytest.raises(RuntimeError, match="Duplicate logging worker result"):
        benchmark._collect_worker_results(processes, results, 1)


def test_parent_rejects_extra_status_after_every_worker_reported():
    results = Queue()
    results.put(benchmark.WorkerResult("W0", 100, 10))
    results.put(benchmark.WorkerResult("W0", 100, 10))
    with pytest.raises(RuntimeError, match="Duplicate or extra"):
        benchmark._collect_worker_results(
            [SimpleNamespace(name="W0", exitcode=0)], results, 1
        )


@pytest.mark.parametrize(
    "field,value",
    [("produced_count", -1), ("transport_errors", "unknown"), ("error", None)],
)
def test_parent_rejects_malformed_typed_accounting(field, value):
    result = benchmark.WorkerResult("W0", 100, 10)
    setattr(result, field, value)
    results = Queue()
    results.put(result)
    with pytest.raises(RuntimeError, match="Invalid logging worker accounting"):
        benchmark._collect_worker_results(
            [SimpleNamespace(name="W0", exitcode=0)], results, 1
        )


@pytest.mark.parametrize(
    "worker_errors,overwritten_batches,error",
    [(1, 3, "worker_errors=1"), (0, 0, "unexplained_messages=30")],
)
def test_cli_prints_results_then_rejects_invalid_lifecycle(
    monkeypatch, capsys, worker_errors, overwritten_batches, error
):
    sample = benchmark.ScenarioSample(
        1,
        256,
        10,
        100,
        70,
        worker_errors,
        0,
        0,
        overwritten_batches=overwritten_batches,
    )

    class Benchmark:
        def __init__(self, config):
            self.message_size_axis_rows = [benchmark._summarize_samples(256, [sample])]
            self.worker_count_axis_rows = []
            self.message_size_axis_stats = []

        def run(self):
            return None

    reporter = SimpleNamespace(
        print_operation_table=lambda stats: None,
        print_comparative_table=lambda *args: None,
    )
    monkeypatch.setattr(benchmark, "AdvancedLoggingBenchmark", Benchmark)
    monkeypatch.setattr(
        benchmark, "BenchmarkReporter", lambda *args, **kwargs: reporter
    )
    monkeypatch.setattr(
        benchmark, "ComparativeReporter", lambda *args, **kwargs: reporter
    )
    monkeypatch.setattr(benchmark.sys, "argv", ["benchmark_advanced_logging.py"])
    with pytest.raises(RuntimeError, match=error):
        benchmark.main()
    output = capsys.readouterr().out
    assert "Advanced Logging Benchmark Results" in output
    assert "WorkerErr" in output
    assert "Attempted" in output

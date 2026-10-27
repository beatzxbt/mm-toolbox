"""Exact worker transport accounting survives sampled diagnostics and cleanup."""

import pytest

import mm_toolbox.logging.advanced.worker as worker_module
from mm_toolbox.logging.advanced import LoggerConfig, WorkerLogger


class FakeTransport:
    num_rings = 2
    last_insert_dropped = 0
    failed_inserts = 0

    def __init__(self, mode):
        self.mode = mode
        self.closed = False

    def insert(self, payload):
        if self.mode == "error":
            raise OSError("injected insert error")
        if self.mode == "rejected":
            self.failed_inserts += 1
            return False
        self.last_insert_dropped = 3 if self.mode == "overwrite" else 0
        return True

    def close(self):
        self.closed = True
        if self.mode == "close_error":
            raise OSError("injected close error")


@pytest.mark.parametrize(
    "mode,rejected,errors",
    [("rejected", 1, 0), ("error", 1, 1), ("close_error", 0, 1), ("oversized", 1, 0)],
)
def test_worker_rejections_and_errors_remain_visible_after_shutdown(
    tmp_path, monkeypatch, mode, rejected, errors
):
    transport = FakeTransport(mode)
    monkeypatch.setattr(worker_module, "ShmMpscProducer", lambda **kwargs: transport)
    config = LoggerConfig(
        path=str(tmp_path / "worker.shm"),
        flush_interval_s=60,
        max_batch_messages=1,
        max_batch_bytes=16 if mode == "oversized" else 1024,
    )
    worker = WorkerLogger(config=config, name="fault-worker")
    try:
        worker.info(b"x" * 16)
    finally:
        worker.shutdown()
    expected = {
        "rejected_messages": rejected,
        "transport_errors": errors,
        "overwritten_batches": 0,
    }
    assert worker.get_transport_stats() == expected
    worker.shutdown()
    assert worker.get_transport_stats() == expected
    assert transport.closed


def test_sampled_overwrite_diagnostics_preserve_exact_batch_counts(
    tmp_path, monkeypatch, capsys
):
    transport = FakeTransport("overwrite")
    monkeypatch.setattr(worker_module, "ShmMpscProducer", lambda **kwargs: transport)
    config = LoggerConfig(
        path=str(tmp_path / "worker.shm"), flush_interval_s=60, max_batch_messages=1
    )
    worker = WorkerLogger(config=config, name="saturated-worker")
    try:
        for _ in range(16):
            worker.info(b"message")
    finally:
        worker.shutdown()
    assert worker.get_transport_stats() == {
        "rejected_messages": 0,
        "transport_errors": 0,
        "overwritten_batches": 48,
    }
    warnings = [
        line for line in capsys.readouterr().err.splitlines() if "SHM overwrite" in line
    ]
    assert len(warnings) == 5
    assert "dropped 48 older log batches" in warnings[-1]

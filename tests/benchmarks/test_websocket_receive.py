"""Driver contracts and normal local receive validation for WebSocket bursts."""

import asyncio

import pytest

from benchmarks.websocket import benchmark_local, inbound as benchmark


def _healthy_result(receiver="pool"):
    connections = 1 if receiver == "single" else 2
    logical = 2
    wire_per_connection = (logical,) * connections
    wire_total = sum(wire_per_connection)
    unique = logical
    return benchmark.InboundScenarioResult(
        receiver=receiver,
        scenario="startup-burst",
        logical_payloads=logical,
        message_size=256,
        connection_count=connections,
        source_sent_per_connection=wire_per_connection,
        native_received_per_connection=wire_per_connection,
        receive_buffer_drops=0,
        peak_buffered_frames=0,
        peak_buffered_bytes=0,
        read_paused_samples=0,
        raw_frames_consumed=wire_total,
        deduplicated_frames=wire_total - unique,
        app_received_frames=unique,
        unique_payloads_received=unique,
        missing_unique_payloads=0,
        missing_id_samples=(),
        reordered_unique_payloads=0,
        duplicate_app_payloads=0,
        malformed_payloads=0,
        delivery_elapsed_ns=1_000_000,
        delivery_completed=True,
        latency_samples_ns=(100_000, 200_000),
        latency_observations=unique,
        receive_timed_out=False,
    )


def test_inbound_payload_has_fixed_size_sequence_and_monotonic_timestamp():
    payload = benchmark._make_inbound_payload(17, 123456, 256)

    assert len(payload) == 256
    assert benchmark._decode_inbound_payload(payload) == (17, 123456)
    assert payload[16:] == bytes([17]) * 240


def test_pool_accounting_separates_filtered_duplicates_from_frame_loss():
    result = _healthy_result()

    assert result.wire_frames_sent == 4
    assert result.native_frames_received == 4
    assert result.deduplicated_frames == 2
    assert result.lost_wire_frames == 0
    assert result.pending_frames == 0
    assert result.unaccounted_frames == 0
    benchmark._require_inbound_no_loss([result])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("missing_unique_payloads", 1),
        ("reordered_unique_payloads", 1),
        ("receive_buffer_drops", 1),
        ("malformed_payloads", 1),
    ],
)
def test_strict_inbound_acceptance_rejects_delivery_errors(field, value):
    result = _healthy_result()
    setattr(result, field, value)

    with pytest.raises(RuntimeError, match="no-loss acceptance failed"):
        benchmark._require_inbound_no_loss([result])


@pytest.mark.asyncio
async def test_receive_wait_propagates_collector_failure_without_waiting_for_timeout():
    accounting = benchmark.InboundAccounting(
        sequence_start=0,
        expected_count=129,
        message_size=256,
    )

    async def fail_collector():
        raise ValueError("injected collector failure")

    consumer_task = asyncio.create_task(fail_collector())
    with pytest.raises(RuntimeError, match="Inbound collector failed") as error:
        await benchmark._wait_for_inbound_accounting(
            accounting,
            expected_raw_frames=129,
            timeout_s=10,
            consumer_task=consumer_task,
        )

    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("receiver", "scenario"),
    [
        ("single", "paced"),
        ("single", "startup-burst"),
        ("single", "post-idle-burst"),
        ("pool", "paced"),
        ("pool", "startup-burst"),
        ("pool", "post-idle-burst"),
    ],
)
@pytest.mark.asyncio
async def test_local_256_byte_burst_reaches_every_unique_payload(receiver, scenario):
    """Exercise actual receiver components with a burst above the old 128 limit."""
    config = benchmark.InboundBenchmarkConfig(
        pool_connections=2,
        message_size=256,
        num_messages=256,
        warmup_messages=8,
        paced_interval_ms=0.25,
        idle_interval_ms=5,
        timeout_s=3,
    )

    result = await benchmark._run_inbound_scenario(receiver, scenario, config)
    benchmark._require_inbound_no_loss([result])

    expected_connections = 1 if receiver == "single" else 2
    assert result.source_sent_per_connection == (256,) * expected_connections
    assert result.native_received_per_connection == (256,) * expected_connections
    assert result.unique_payloads_received == 256


@pytest.mark.asyncio
async def test_inbound_scenario_closes_source_when_client_exit_raises(monkeypatch):
    sources = []
    original_start = benchmark.InboundSourceProcess.start

    async def track_start(host, timeout_s):
        source = await original_start(host, timeout_s)
        sources.append(source)
        return source

    class ExitFailingSingle(benchmark.WsSingle):
        async def __aexit__(self, exc_type, exc_value, traceback):
            await super().__aexit__(exc_type, exc_value, traceback)
            raise RuntimeError("injected client exit failure")

    monkeypatch.setattr(
        benchmark.InboundSourceProcess, "start", staticmethod(track_start)
    )
    monkeypatch.setattr(benchmark, "WsSingle", ExitFailingSingle)
    config = benchmark.InboundBenchmarkConfig(
        pool_connections=2,
        num_messages=129,
        warmup_messages=0,
        timeout_s=1,
    )

    with pytest.raises(RuntimeError, match="injected client exit failure"):
        await benchmark._run_inbound_scenario("single", "startup-burst", config)

    assert sources
    assert sources[0].start_method == "spawn"
    assert sources[0].exitcode == 0


@pytest.mark.asyncio
async def test_local_source_wait_and_shutdown_have_finite_deadlines():
    server = benchmark.LocalSourceServer()
    await server.start()
    try:
        with pytest.raises(TimeoutError, match="accepted 0 of 1"):
            await server.wait_for_connections(1, timeout_s=0.01)
    finally:
        await asyncio.wait_for(server.stop(timeout_s=1), timeout=1.5)

    assert server._server is None


@pytest.mark.asyncio
async def test_source_process_exits_when_receiver_control_channel_closes():
    source = await benchmark.InboundSourceProcess.start("127.0.0.1", timeout_s=2)
    source._parent_connection.close()

    await source.stop(timeout_s=2)

    assert source.exitcode == 0


class _FakeProcess:
    def __init__(self, *, alive=False, exitcode=0, terminate_stops=True):
        self.alive = alive
        self.exitcode = exitcode
        self.terminate_stops = terminate_stops
        self.join_timeouts = []
        self.terminated = False
        self.killed = False
        self.closed = False

    def join(self, timeout):
        self.join_timeouts.append(timeout)

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        if self.terminate_stops:
            self.alive = False

    def kill(self):
        self.killed = True
        self.alive = False

    def close(self):
        self.closed = True


def test_benchmark_child_failure_is_propagated():
    process = _FakeProcess(exitcode=7)

    with pytest.raises(RuntimeError, match="exited with status 7"):
        benchmark_local._join_benchmark_process(process, "receive child")

    assert process.closed


def test_hung_benchmark_child_is_terminated_and_reported():
    process = _FakeProcess(alive=True)

    with pytest.raises(TimeoutError, match="did not finish within 0.0 seconds"):
        benchmark_local._join_benchmark_process(
            process, "receive child", timeout_s=0, cleanup_timeout_s=0
        )

    assert process.terminated
    assert process.closed
    assert process.join_timeouts == [0, 0]


def test_hung_benchmark_child_escalates_to_kill_and_reaps():
    process = _FakeProcess(alive=True, terminate_stops=False)

    with pytest.raises(TimeoutError, match="did not finish within 0.0 seconds"):
        benchmark_local._join_benchmark_process(
            process, "receive child", timeout_s=0, cleanup_timeout_s=0
        )

    assert process.terminated
    assert process.killed
    assert process.closed
    assert process.join_timeouts == [0, 0, 0]

"""Local WebSocket benchmark using an asyncio sink server.

Measures throughput and latency for WsSingle and WsPool in isolation
using separate processes to avoid thread/core interference.

Usage:
    uv run python benchmarks/websocket/benchmark_local.py
    uv run python benchmarks/websocket/benchmark_local.py --mode inbound
    uv run python benchmarks/websocket/benchmark_local.py --mode inbound --allow-loss
"""

from __future__ import annotations

import asyncio
import multiprocessing
import time
from dataclasses import dataclass

import numpy as np
import websockets

from mm_toolbox.websocket.connection import ConnectionState, WsConnectionConfig
from mm_toolbox.websocket.pool import WsPool, WsPoolConfig
from mm_toolbox.websocket.single import WsSingle


try:
    from benchmarks.core import BenchmarkCLI
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.core import BenchmarkCLI

from benchmarks.websocket.inbound import (
    InboundBenchmarkConfig,
    _MAX_LATENCY_SAMPLES,
    _run_pool_inbound_benchmark,
    _run_single_inbound_benchmark,
)


# ---------------------------------------------------------------------------
# Local sink server
# ---------------------------------------------------------------------------


class LocalSinkServer:
    """Read and count outbound messages without adding echo traffic."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._server: websockets.Server | None = None
        self.received_messages = 0
        self._received_event = asyncio.Event()

    async def _handle_connection(self, websocket) -> None:
        """Drain one client connection and count every accepted message."""
        try:
            async for _ in websocket:
                self.received_messages += 1
                self._received_event.set()
        except websockets.exceptions.ConnectionClosed:
            pass

    async def start(self) -> None:
        """Start the local sink server."""
        self._server = await websockets.serve(
            self._handle_connection, self.host, self.port, ping_interval=None
        )

    async def stop(self) -> None:
        """Stop the echo server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def wait_for_messages(self, target: int, timeout_s: float = 30.0) -> None:
        """Wait until the sink has received the expected number of messages."""
        deadline = time.monotonic() + timeout_s
        while self.received_messages < target:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise RuntimeError(
                    f"Local sink received {self.received_messages} of {target} messages"
                )
            try:
                await asyncio.wait_for(self._received_event.wait(), timeout=remaining_s)
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    f"Local sink received {self.received_messages} of {target} messages"
                ) from exc
            self._received_event.clear()

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Configuration for local websocket benchmark."""

    server_host: str = "127.0.0.1"
    server_port: int = 8765
    message_size: int = 128
    pool_connections: int = 3
    num_messages: int = 10_000
    warmup_messages: int = 1_000


# ---------------------------------------------------------------------------
# Single benchmark
# ---------------------------------------------------------------------------


async def _benchmark_single(config: BenchmarkConfig) -> dict:
    """Benchmark WsSingle operations."""
    server = LocalSinkServer(config.server_host, config.server_port)
    await server.start()

    try:
        url = server.url
        msg = b"x" * config.message_size

        # Benchmark: connect
        connect_latencies = []
        for _ in range(100):
            ws = WsSingle(
                WsConnectionConfig(
                    conn_id=1,
                    wss_url=url,
                    on_connect=[],
                    auto_reconnect=False,
                )
            )
            start = time.perf_counter_ns()
            async with ws:
                if ws.get_state() != ConnectionState.CONNECTED:
                    raise RuntimeError("WsSingle did not connect to the local sink")
                connect_latencies.append(time.perf_counter_ns() - start)

        # Benchmark: send throughput
        ws = WsSingle(
            WsConnectionConfig(
                conn_id=1,
                wss_url=url,
                on_connect=[],
                auto_reconnect=False,
            )
        )
        async with ws:
            if ws.get_state() != ConnectionState.CONNECTED:
                raise RuntimeError("WsSingle did not connect to the local sink")

            for _ in range(config.warmup_messages):
                ws.send_data(msg)

            start = time.perf_counter_ns()
            for _ in range(config.num_messages):
                ws.send_data(msg)
            throughput_ns = time.perf_counter_ns() - start

            send_latencies = []
            for _ in range(config.num_messages):
                op_start = time.perf_counter_ns()
                ws.send_data(msg)
                send_latencies.append(time.perf_counter_ns() - op_start)
            await server.wait_for_messages(
                config.warmup_messages + 2 * config.num_messages
            )

        return {
            "connect": connect_latencies,
            "send_throughput": [throughput_ns, config.num_messages],
            "send_latency": send_latencies,
            "server_received": [server.received_messages],
        }
    finally:
        await server.stop()


def _run_single_benchmark(config_dict: dict) -> dict:
    """Run single benchmark in separate process."""
    config = BenchmarkConfig(**config_dict)
    return asyncio.run(_benchmark_single(config))


# ---------------------------------------------------------------------------
# Pool benchmark
# ---------------------------------------------------------------------------


async def _benchmark_pool(config: BenchmarkConfig) -> dict:
    """Benchmark WsPool operations."""
    server = LocalSinkServer(config.server_host, config.server_port)
    await server.start()

    try:
        url = server.url
        msg = b"x" * config.message_size

        # Benchmark: connect
        connect_latencies = []
        for _ in range(100):
            pool = await WsPool.new(
                config=WsConnectionConfig(
                    conn_id=1,
                    wss_url=url,
                    on_connect=[],
                    auto_reconnect=False,
                ),
                pool_config=WsPoolConfig(
                    num_connections=config.pool_connections,
                    evict_interval_s=15,
                ),
            )
            start = time.perf_counter_ns()
            async with pool:
                if pool.get_connection_count() != config.pool_connections:
                    raise RuntimeError("WsPool did not connect every local socket")
                connect_latencies.append(time.perf_counter_ns() - start)

        # Benchmark: send throughput (fastest)
        pool = await WsPool.new(
            config=WsConnectionConfig(
                conn_id=1,
                wss_url=url,
                on_connect=[],
                auto_reconnect=False,
            ),
            pool_config=WsPoolConfig(
                num_connections=config.pool_connections,
                evict_interval_s=15,
            ),
        )
        async with pool:
            if pool.get_connection_count() != config.pool_connections:
                raise RuntimeError("WsPool did not connect every local socket")
            for _ in range(config.warmup_messages):
                pool.send_data(msg, only_fastest=True)

            start = time.perf_counter_ns()
            for _ in range(config.num_messages):
                pool.send_data(msg, only_fastest=True)
            fastest_throughput_ns = time.perf_counter_ns() - start

            fastest_latencies = []
            for _ in range(config.num_messages):
                op_start = time.perf_counter_ns()
                pool.send_data(msg, only_fastest=True)
                fastest_latencies.append(time.perf_counter_ns() - op_start)
            await server.wait_for_messages(
                config.warmup_messages + 2 * config.num_messages
            )

        # Benchmark: broadcast throughput
        pool = await WsPool.new(
            config=WsConnectionConfig(
                conn_id=1,
                wss_url=url,
                on_connect=[],
                auto_reconnect=False,
            ),
            pool_config=WsPoolConfig(
                num_connections=config.pool_connections,
                evict_interval_s=15,
            ),
        )
        async with pool:
            if pool.get_connection_count() != config.pool_connections:
                raise RuntimeError("WsPool did not connect every local socket")
            for _ in range(config.warmup_messages):
                pool.send_data(msg, only_fastest=False)

            start = time.perf_counter_ns()
            for _ in range(config.num_messages):
                pool.send_data(msg, only_fastest=False)
            broadcast_throughput_ns = time.perf_counter_ns() - start

            broadcast_latencies = []
            for _ in range(config.num_messages):
                op_start = time.perf_counter_ns()
                pool.send_data(msg, only_fastest=False)
                broadcast_latencies.append(time.perf_counter_ns() - op_start)
            messages_before_broadcast = server.received_messages
            await server.wait_for_messages(
                messages_before_broadcast
                + config.pool_connections
                * (config.warmup_messages + 2 * config.num_messages)
            )

        return {
            "connect": connect_latencies,
            "send_fastest": [fastest_throughput_ns],
            "send_latency_fastest": fastest_latencies,
            "send_broadcast": [
                broadcast_throughput_ns,
                config.num_messages * config.pool_connections,
            ],
            "send_latency_broadcast": broadcast_latencies,
            "server_received": [server.received_messages],
        }
    finally:
        await server.stop()


def _run_pool_benchmark(config_dict: dict) -> dict:
    """Run pool benchmark in separate process."""
    config = BenchmarkConfig(**config_dict)
    return asyncio.run(_benchmark_pool(config))


# ---------------------------------------------------------------------------
# Results printing
# ---------------------------------------------------------------------------


def _print_results(title: str, results: dict, num_messages: int) -> None:
    """Print benchmark results."""
    print(f"\n{title}")
    print("-" * 80)

    for op_name, latencies in results.items():
        if not latencies:
            continue

        if op_name == "server_received":
            print(f"  server received       : {latencies[0]:,} messages")
        elif op_name == "connect" or op_name.startswith("send_latency"):
            # Percentiles use NumPy's linearly interpolated definition.
            samples = np.asarray(latencies, dtype=np.float64)
            scale = 1_000_000 if op_name == "connect" else 1
            unit = "ms/op" if op_name == "connect" else "ns/op"
            print(
                f"  {op_name:24s}: mean={np.mean(samples) / scale:9.3f} "
                f"p50={np.percentile(samples, 50) / scale:9.3f} "
                f"p95={np.percentile(samples, 95) / scale:9.3f} "
                f"p99={np.percentile(samples, 99) / scale:9.3f} {unit} "
                f"({len(latencies):,} samples)"
            )
        else:
            # Throughput measurement (single timing for N messages)
            total_ns = latencies[0]
            total_s = total_ns / 1_000_000_000
            message_count = latencies[1] if len(latencies) > 1 else num_messages
            msg_per_s = message_count / total_s
            print(
                f"  {op_name:20s}: {msg_per_s:10,.0f} msg/s  "
                f"({total_s:.6f}s for {message_count:,} msgs; "
                f"{num_messages:,} send_data calls)"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point."""
    cli = BenchmarkCLI("Benchmark local WebSocket performance")
    cli.parser.set_defaults(operations=10_000, warmup=1_000)
    cli.parser.add_argument(
        "--mode",
        choices=("outbound", "inbound", "both"),
        default="outbound",
        help="Run existing outbound benchmarks, inbound delivery, or both",
    )
    cli.parser.add_argument(
        "--message-size",
        type=int,
        default=128,
        help="Size of test messages in bytes (default: 128)",
    )
    cli.parser.add_argument(
        "--pool-connections",
        type=int,
        default=3,
        help="Number of pool connections (default: 3)",
    )
    cli.parser.add_argument(
        "--receive-messages",
        type=int,
        default=512,
        help="Unique inbound messages per scenario; at least 129 for burst coverage",
    )
    cli.parser.add_argument(
        "--receive-warmup",
        type=int,
        default=32,
        help="Drained inbound warmup messages before paced/post-idle scenarios",
    )
    cli.parser.add_argument(
        "--receive-message-size",
        type=int,
        default=256,
        help="Inbound frame size in bytes (default: realistic 256-byte frame)",
    )
    cli.parser.add_argument(
        "--receive-interval-ms",
        type=float,
        default=1.0,
        help="Inter-message interval for the paced inbound scenario",
    )
    cli.parser.add_argument(
        "--receive-idle-ms",
        type=float,
        default=20.0,
        help="Quiet period before the post-idle burst scenario",
    )
    cli.parser.add_argument(
        "--receive-timeout",
        type=float,
        default=3.0,
        help="Finite deadline for connection setup and inbound drain",
    )
    cli.parser.add_argument(
        "--allow-loss",
        action="store_true",
        help="Report receive loss without strict no-loss failure",
    )

    args = cli.parse()

    config = BenchmarkConfig(
        num_messages=args.operations,
        warmup_messages=args.warmup,
        message_size=args.message_size,
        pool_connections=args.pool_connections,
    )

    if args.mode in {"outbound", "both"}:
        print("=" * 80)
        print("Local WebSocket Benchmark")
        print("=" * 80)
        print(f"Message size: {config.message_size} bytes")
        print(f"Pool connections: {config.pool_connections}")
        print(f"Send calls per mode: {config.num_messages:,}")
        print(f"Warmup send calls per mode: {config.warmup_messages:,}")
        print("Send latency is the synchronous send_data() call, not network RTT")
        print()

        config_dict = {
            "server_host": config.server_host,
            "server_port": config.server_port,
            "message_size": config.message_size,
            "pool_connections": config.pool_connections,
            "num_messages": config.num_messages,
            "warmup_messages": config.warmup_messages,
        }

        print("Running single-connection benchmark...")
        single_proc = multiprocessing.Process(
            target=_run_single_and_print,
            args=(config_dict,),
        )
        single_proc.start()
        _join_benchmark_process(single_proc, "single-connection")

        print("\nRunning pool benchmark...")
        pool_proc = multiprocessing.Process(
            target=_run_pool_and_print,
            args=(config_dict,),
        )
        pool_proc.start()
        _join_benchmark_process(pool_proc, "pool")

    if args.mode in {"inbound", "both"}:
        try:
            inbound_config = InboundBenchmarkConfig(
                server_host=config.server_host,
                pool_connections=config.pool_connections,
                message_size=args.receive_message_size,
                num_messages=args.receive_messages,
                warmup_messages=args.receive_warmup,
                paced_interval_ms=args.receive_interval_ms,
                idle_interval_ms=args.receive_idle_ms,
                timeout_s=args.receive_timeout,
                require_no_loss=not args.allow_loss,
            )
        except ValueError as exc:
            cli.parser.error(str(exc))
        print("\n" + "=" * 80)
        print("Inbound WebSocket Delivery Benchmark")
        print("=" * 80)
        print(
            f"Each receiver runs paced, startup-burst, and post-idle-burst; "
            f"{inbound_config.num_messages:,} unique frames per scenario "
            f"({inbound_config.message_size} bytes), "
            f"{inbound_config.pool_connections} pool sockets."
        )
        print(
            "The source runs in a separate process from the receiver. Rate is "
            "unique app deliveries divided by the interval from first source "
            "send to last unique app delivery. Latency is per-ID source-send to "
            "app receipt using the same-host monotonic clock; "
            f"percentiles use at most {_MAX_LATENCY_SAMPLES:,} reservoir samples "
            "(all deliveries below the cap)."
        )
        print(
            "Pool wire copies are intentional duplicates; their hash-filter count "
            "is separate from receive-buffer losses."
        )
        print(
            "No-loss acceptance: "
            f"{'required' if inbound_config.require_no_loss else 'disabled by --allow-loss'}"
        )
        inbound_dict = {
            "server_host": inbound_config.server_host,
            "pool_connections": inbound_config.pool_connections,
            "message_size": inbound_config.message_size,
            "num_messages": inbound_config.num_messages,
            "warmup_messages": inbound_config.warmup_messages,
            "paced_interval_ms": inbound_config.paced_interval_ms,
            "idle_interval_ms": inbound_config.idle_interval_ms,
            "timeout_s": inbound_config.timeout_s,
            "require_no_loss": inbound_config.require_no_loss,
        }
        inbound_child_timeout_s = max(
            60.0,
            2.0
            * (
                inbound_config.num_messages * inbound_config.paced_interval_ms / 1_000
                + 3 * inbound_config.timeout_s
                + inbound_config.idle_interval_ms / 1_000
                + 10
            ),
        )
        print("Running inbound single-connection scenarios...")
        single_proc = multiprocessing.Process(
            target=_run_single_inbound_benchmark,
            args=(inbound_dict,),
        )
        single_proc.start()
        _join_benchmark_process(
            single_proc, "inbound single-connection", inbound_child_timeout_s
        )

        print("\nRunning inbound redundant-pool scenarios...")
        pool_proc = multiprocessing.Process(
            target=_run_pool_inbound_benchmark,
            args=(inbound_dict,),
        )
        pool_proc.start()
        _join_benchmark_process(pool_proc, "inbound pool", inbound_child_timeout_s)

    print("\n" + "=" * 80)
    print("Benchmark complete")
    print("=" * 80)


def _join_benchmark_process(
    process: multiprocessing.Process,
    name: str,
    timeout_s: float = 60.0,
    cleanup_timeout_s: float = 5.0,
) -> None:
    """Wait for a child, stop it if it hangs, and propagate child failures."""
    process.join(timeout=timeout_s)
    if process.is_alive():
        cleanup_deadline = time.monotonic() + cleanup_timeout_s
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        process.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
        if process.is_alive():
            try:
                process.kill()
            except ProcessLookupError:
                pass
            process.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
        if process.is_alive():
            raise TimeoutError(
                f"{name} benchmark child remained alive after terminate and kill"
            )
        process.close()
        raise TimeoutError(
            f"{name} benchmark did not finish within {timeout_s:.1f} seconds"
        )
    exitcode = process.exitcode
    process.close()
    if exitcode != 0:
        raise RuntimeError(f"{name} benchmark exited with status {exitcode}")
    print(f"{name} benchmark child exit status: {exitcode}")


def _run_single_and_print(config_dict: dict) -> None:
    """Run single benchmark and print results."""
    results = _run_single_benchmark(config_dict)
    _print_results("Single Connection", results, config_dict["num_messages"])


def _run_pool_and_print(config_dict: dict) -> None:
    """Run pool benchmark and print results."""
    results = _run_pool_benchmark(config_dict)
    _print_results("Pool", results, config_dict["num_messages"])


if __name__ == "__main__":
    main()

"""Local WebSocket benchmark using asyncio echo server.

Measures throughput and latency for WsSingle and WsPool in isolation
using separate processes to avoid thread/core interference.

Usage:
    uv run python benchmarks/websocket/benchmark_local.py
"""

from __future__ import annotations

import asyncio
import multiprocessing
import time
from dataclasses import dataclass

import websockets

from mm_toolbox.websocket.connection import ConnectionState, WsConnectionConfig
from mm_toolbox.websocket.pool import WsPool, WsPoolConfig
from mm_toolbox.websocket.single import WsSingle


def _noop_handler(msg: bytes) -> None:
    """No-op message handler for benchmarks."""
    pass


try:
    from benchmarks.core import BenchmarkCLI
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.core import BenchmarkCLI


# ---------------------------------------------------------------------------
# Echo server
# ---------------------------------------------------------------------------


async def _echo_handler(websocket: websockets.WebSocketServerProtocol) -> None:
    """Echo all messages back to client."""
    try:
        async for message in websocket:
            await websocket.send(message)
    except websockets.exceptions.ConnectionClosed:
        pass


class EchoServer:
    """WebSocket echo server for benchmarking."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._server: websockets.Server | None = None

    async def start(self) -> None:
        """Start the echo server."""
        self._server = await websockets.serve(
            _echo_handler, self.host, self.port, ping_interval=None
        )

    async def stop(self) -> None:
        """Stop the echo server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

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
    server = EchoServer(config.server_host, config.server_port)
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
            task = asyncio.create_task(ws.start())
            # Wait for connection
            for _ in range(500):
                if ws.get_state() == ConnectionState.CONNECTED:
                    break
                await asyncio.sleep(0.01)
            elapsed = time.perf_counter_ns() - start
            connect_latencies.append(elapsed)
            ws.close()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Benchmark: send throughput
        ws = WsSingle(
            WsConnectionConfig(
                conn_id=1,
                wss_url=url,
                on_connect=[],
                auto_reconnect=False,
            )
        )
        task = asyncio.create_task(ws.start())
        # Wait for connection
        for _ in range(500):
            if ws.get_state() == ConnectionState.CONNECTED:
                break
            await asyncio.sleep(0.01)

        # Warmup
        for _ in range(config.warmup_messages):
            ws.send_data(msg)

        # Measure
        send_latencies = []
        start = time.perf_counter_ns()
        for _ in range(config.num_messages):
            ws.send_data(msg)
        elapsed = time.perf_counter_ns() - start
        send_latencies.append(elapsed)

        ws.close()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        return {
            "connect": connect_latencies,
            "send_throughput": send_latencies,
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
    server = EchoServer(config.server_host, config.server_port)
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
                on_message=_noop_handler,
                pool_config=WsPoolConfig(
                    num_connections=config.pool_connections,
                    evict_interval_s=15,
                ),
            )
            start = time.perf_counter_ns()
            async with pool:
                elapsed = time.perf_counter_ns() - start
                connect_latencies.append(elapsed)

        # Benchmark: send throughput (fastest)
        pool = await WsPool.new(
            config=WsConnectionConfig(
                conn_id=1,
                wss_url=url,
                on_connect=[],
                auto_reconnect=False,
            ),
            on_message=_noop_handler,
            pool_config=WsPoolConfig(
                num_connections=config.pool_connections,
                evict_interval_s=15,
            ),
        )
        async with pool:
            # Warmup
            for _ in range(config.warmup_messages):
                pool.send_data(msg, only_fastest=True)

            # Measure
            send_latencies = []
            start = time.perf_counter_ns()
            for _ in range(config.num_messages):
                pool.send_data(msg, only_fastest=True)
            elapsed = time.perf_counter_ns() - start
            send_latencies.append(elapsed)

        # Benchmark: broadcast throughput
        pool = await WsPool.new(
            config=WsConnectionConfig(
                conn_id=1,
                wss_url=url,
                on_connect=[],
                auto_reconnect=False,
            ),
            on_message=_noop_handler,
            pool_config=WsPoolConfig(
                num_connections=config.pool_connections,
                evict_interval_s=15,
            ),
        )
        async with pool:
            # Warmup
            for _ in range(config.warmup_messages):
                pool.send_data(msg, only_fastest=False)

            # Measure
            broadcast_latencies = []
            start = time.perf_counter_ns()
            for _ in range(config.num_messages):
                pool.send_data(msg, only_fastest=False)
            elapsed = time.perf_counter_ns() - start
            broadcast_latencies.append(elapsed)

        return {
            "connect": connect_latencies,
            "send_fastest": send_latencies,
            "send_broadcast": broadcast_latencies,
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

        if op_name == "connect":
            # Per-operation latencies
            avg_ns = sum(latencies) / len(latencies)
            avg_ms = avg_ns / 1_000_000
            print(f"  {op_name:20s}: {avg_ms:8.3f} ms/op  ({len(latencies)} ops)")
        else:
            # Throughput measurement (single timing for N messages)
            total_ns = latencies[0]
            total_s = total_ns / 1_000_000_000
            msg_per_s = num_messages / total_s
            print(
                f"  {op_name:20s}: {msg_per_s:10,.0f} msg/s  ({total_s:.3f}s for {num_messages:,} msgs)"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point."""
    cli = BenchmarkCLI("Benchmark local WebSocket performance")
    cli.parser.set_defaults(operations=10_000, warmup=1_000)
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

    args = cli.parse()

    config = BenchmarkConfig(
        num_messages=args.operations,
        warmup_messages=args.warmup,
        message_size=args.message_size,
        pool_connections=args.pool_connections,
    )

    print("=" * 80)
    print("Local WebSocket Benchmark")
    print("=" * 80)
    print(f"Message size: {config.message_size} bytes")
    print(f"Pool connections: {config.pool_connections}")
    print(f"Messages per test: {config.num_messages:,}")
    print()

    config_dict = {
        "server_host": config.server_host,
        "server_port": config.server_port,
        "message_size": config.message_size,
        "pool_connections": config.pool_connections,
        "num_messages": config.num_messages,
        "warmup_messages": config.warmup_messages,
    }

    # Run benchmarks in separate processes
    print("Running single-connection benchmark...")
    single_proc = multiprocessing.Process(
        target=_run_single_and_print,
        args=(config_dict,),
    )
    single_proc.start()
    single_proc.join()

    print("\nRunning pool benchmark...")
    pool_proc = multiprocessing.Process(
        target=_run_pool_and_print,
        args=(config_dict,),
    )
    pool_proc.start()
    pool_proc.join()

    print("\n" + "=" * 80)
    print("Benchmark complete")
    print("=" * 80)


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

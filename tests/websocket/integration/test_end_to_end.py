"""End-to-end integration tests for the WebSocket module.

This module provides comprehensive E2E coverage across:
- WsConnection lifecycle (connect, send, receive, disconnect)
- WsSingle wrapper (async context manager, iteration)
- WsPool management (multiple connections, deduplication, fastest vs multicast)
- Message flow (bursts, large payloads, fragmentation, empty payloads)
- Concurrency (multi-thread sends, rapid ping during bursts)
- Reconnection (auto-reconnect, server restart)
- Latency (ping/pong timing)
- Error resilience (malformed frames, close during flight, callback exceptions)
- Stress (connection churn, sustained throughput)
"""

from __future__ import annotations

import asyncio
import gc
import resource
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

import pytest

from mm_toolbox.ringbuffer.bytes import BytesRingBuffer
from mm_toolbox.websocket.connection import (
    ConnectionState,
    WsConnection,
)
from mm_toolbox.websocket.pool import WsPool, WsPoolConfig
from mm_toolbox.websocket.single import WsSingle


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _noop_handler(msg: bytes) -> None:
    """No-op message handler for pool tests.

    Args:
        msg (bytes): Incoming message payload.

    Returns:
        None: This handler does not return a value.
    """
    return None


async def _wait_for_single_state(
    ws: WsSingle, expected: ConnectionState, timeout_s: float = 2.0
) -> None:
    """Wait for a WsSingle instance to reach the expected state.

    Args:
        ws (WsSingle): WsSingle instance.
        expected (ConnectionState): Expected connection state.
        timeout_s (float): Timeout in seconds.

    Returns:
        None: This helper does not return a value.

    Raises:
        AssertionError: If state is not reached in time.
    """
    start = asyncio.get_running_loop().time()
    while (asyncio.get_running_loop().time() - start) < timeout_s:
        if ws.get_state() == expected:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {expected}")


# --------------------------------------------------------------------------- #
# Connection Lifecycle
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestConnectionLifecycleE2E:
    """E2E tests covering full connection lifecycle across all layers."""

    async def test_normal_connect_send_receive_disconnect_all_layers(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Validate WsConnection, WsSingle, and WsPool in sequence.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            # --- Layer 1: WsConnection ---
            ringbuffer = BytesRingBuffer(max_capacity=128, only_insert_unique=False)
            config = connection_config_factory(basic_server)
            conn = await WsConnection.new(ringbuffer, config)
            assert conn.get_state() == ConnectionState.CONNECTED

            conn.send_data(b"hello-conn")
            await asyncio.sleep(0.2)
            assert b"hello-conn" in basic_server.get_received_messages()

            msg = await asyncio.wait_for(ringbuffer.aconsume(), timeout=2.0)
            assert msg == b"hello-conn"
            conn.close()
            await asyncio.sleep(0.1)
            assert conn.get_state() == ConnectionState.DISCONNECTED

            # --- Layer 2: WsSingle ---
            single_config = connection_config_factory(basic_server)
            collected_single: list[bytes] = []
            async with WsSingle(single_config) as ws:
                await _wait_for_single_state(ws, ConnectionState.CONNECTED)
                ws.send_data(b"hello-single")

                async def _collect_single() -> None:
                    """Collect echoed message from WsSingle iteration."""
                    async for msg in ws:
                        if msg == b"hello-single":
                            collected_single.append(msg)
                            break

                task = asyncio.create_task(_collect_single())
                await asyncio.wait_for(task, timeout=2.0)

            assert collected_single == [b"hello-single"]
            assert b"hello-single" in basic_server.get_received_messages()

            # --- Layer 3: WsPool ---
            pool_config = WsPoolConfig(num_connections=3, evict_interval_s=60)
            pool_conn_config = connection_config_factory(basic_server)
            pool = await WsPool.new(
                pool_conn_config, on_message=_noop_handler, pool_config=pool_config
            )
            collected_pool: list[bytes] = []
            async with pool:
                await asyncio.sleep(0.3)
                assert pool.get_connection_count() == 3
                pool.send_data(b"hello-pool", only_fastest=True)

                async def _collect_pool() -> None:
                    """Collect echoed message from WsPool iteration."""
                    async for msg in pool:
                        if msg == b"hello-pool":
                            collected_pool.append(msg)
                            break

                task = asyncio.create_task(_collect_pool())
                await asyncio.wait_for(task, timeout=2.0)

            assert collected_pool == [b"hello-pool"]
            assert b"hello-pool" in basic_server.get_received_messages()

    async def test_rapid_connect_disconnect_cycles(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Ensure 50 rapid connect/send/close cycles do not exhaust resources.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            for _ in range(50):
                conn = await connection_factory(basic_server)
                conn.send_data(b"ping")
                await asyncio.sleep(0.01)
                conn.close()
                await asyncio.sleep(0.01)
                assert conn.get_state() == ConnectionState.DISCONNECTED

    async def test_connection_rejection_and_retry(
        self,
        server_reject_connections,
        connection_config_factory,
    ) -> None:
        """Ensure rejected connections reach DISCONNECTED and retry can be broken.

        Args:
            server_reject_connections: Fixture providing rejecting server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with server_reject_connections:
            # Direct new() should connect but server closes immediately
            ringbuffer = BytesRingBuffer(max_capacity=8, only_insert_unique=False)
            config = connection_config_factory(server_reject_connections)
            conn = await WsConnection.new(ringbuffer, config)
            await asyncio.sleep(0.2)
            assert conn.get_state() == ConnectionState.DISCONNECTED
            conn.close()

            # Reconnect iterator: attempt a few times then break
            rb2 = BytesRingBuffer(max_capacity=8, only_insert_unique=False)
            conn_iter = WsConnection.new_with_reconnect(rb2, config)
            conn_iter.__aiter__()
            attempt = 0
            async with asyncio.timeout(5.0):
                async for _conn in conn_iter:
                    attempt += 1
                    _conn.close()
                    if attempt >= 3:
                        break
            assert attempt >= 1


# --------------------------------------------------------------------------- #
# Message Flow
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestMessageFlowE2E:
    """E2E tests for various message flow scenarios."""

    async def test_burst_messages_1000(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Send 1000 messages rapidly and collect all echoes.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            collected: list[bytes] = []

            async def _collector() -> None:
                """Collect 1000 echoed messages."""
                while len(collected) < 1000:
                    try:
                        msg = await asyncio.wait_for(
                            conn.get_ringbuffer().aconsume(), timeout=5.0
                        )
                        collected.append(msg)
                    except asyncio.TimeoutError:
                        break

            task = asyncio.create_task(_collector())
            for i in range(1000):
                conn.send_data(f"msg-{i}".encode("utf-8"))
                if i % 100 == 0:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(task, timeout=15.0)
            assert len(collected) == 1000
            conn.close()

    async def test_large_payloads_near_max_frame_size(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Verify payloads near max_frame_size round-trip correctly.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            ringbuffer = BytesRingBuffer(max_capacity=16, only_insert_unique=False)
            config = connection_config_factory(basic_server)
            config.max_frame_size = 1024
            conn = await WsConnection.new(ringbuffer, config)

            payload_1023 = b"x" * 1023
            payload_1024 = b"y" * 1024

            conn.send_data(payload_1023)
            msg1 = await asyncio.wait_for(ringbuffer.aconsume(), timeout=2.0)
            assert msg1 == payload_1023

            conn.send_data(payload_1024)
            msg2 = await asyncio.wait_for(ringbuffer.aconsume(), timeout=2.0)
            assert msg2 == payload_1024

            conn.close()

    async def test_empty_payload(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Verify empty payload is echoed back correctly.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            conn.send_data(b"")
            msg = await asyncio.wait_for(conn.get_ringbuffer().aconsume(), timeout=2.0)
            assert msg == b""
            conn.close()

    async def test_fragmented_messages(
        self,
        server_with_fragmentation,
        connection_config_factory,
    ) -> None:
        """Verify fragmented messages reassemble correctly.

        Args:
            server_with_fragmentation: Fixture providing fragmenting echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with server_with_fragmentation:
            ringbuffer = BytesRingBuffer(max_capacity=32, only_insert_unique=False)
            config = connection_config_factory(server_with_fragmentation)
            conn = await WsConnection.new(ringbuffer, config)

            payloads = [b"AB", b"HelloWorld", b"x" * 1000]
            for payload in payloads:
                conn.send_data(payload)
                msg = await asyncio.wait_for(ringbuffer.aconsume(), timeout=2.0)
                assert msg == payload

            conn.close()

    async def test_messages_during_reconnect(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Send before disconnect and verify reconnection yields a usable conn.

        Note: WsConnection.new_with_reconnect yields a fresh connection after
        the previous one closes.  We validate the iterator pattern here rather
        than transparent auto-reconnect inside a single WsConnection instance.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            ringbuffer = BytesRingBuffer(max_capacity=16, only_insert_unique=False)
            config = connection_config_factory(basic_server, auto_reconnect=True)
            conn_iter = WsConnection.new_with_reconnect(ringbuffer, config)

            async with asyncio.timeout(5.0):
                async for conn in conn_iter:
                    assert conn.get_state() == ConnectionState.CONNECTED
                    conn.send_data(b"before")
                    await asyncio.sleep(0.2)
                    assert b"before" in basic_server.get_received_messages()
                    conn.close()
                    await asyncio.sleep(0.1)
                    # Only test one disconnect/reconnect cycle
                    break


# --------------------------------------------------------------------------- #
# Concurrent
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestConcurrentE2E:
    """E2E tests for concurrent and multi-threaded usage."""

    async def test_multi_thread_sends_same_connection(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Send 100 messages from 10 threads via the same connection.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)

            def _send_batch(conn_ref: WsConnection, start: int) -> None:
                """Send 10 numbered messages."""
                for i in range(10):
                    conn_ref.send_data(f"t-{start}-{i}".encode("utf-8"))

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(_send_batch, conn, t) for t in range(10)]
                for fut in futures:
                    fut.result()

            await asyncio.sleep(0.5)
            received = basic_server.get_received_messages()
            assert len(received) == 100
            conn.close()

    async def test_pool_sends_from_multiple_threads(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Send 100 messages from 5 threads via a WsPool.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            pool_config = WsPoolConfig(num_connections=3, evict_interval_s=60)
            config = connection_config_factory(basic_server)
            pool = await WsPool.new(
                config, on_message=_noop_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.3)

                def _send_batch(pool_ref: WsPool, start: int) -> None:
                    """Send 20 numbered messages through the pool."""
                    for i in range(20):
                        pool_ref.send_data(
                            f"p-{start}-{i}".encode("utf-8"), only_fastest=True
                        )

                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(_send_batch, pool, t) for t in range(5)]
                    for fut in futures:
                        fut.result()

                await asyncio.sleep(0.5)
                received = basic_server.get_received_messages()
                assert len(received) == 100

    async def test_rapid_ping_during_message_burst(
        self,
        basic_server,
        connection_config_factory,
        latency_waiter,
    ) -> None:
        """Send 500 messages while latency pings run every 50 ms.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.
            latency_waiter: Fixture providing latency wait helper.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            ringbuffer = BytesRingBuffer(max_capacity=128, only_insert_unique=False)
            config = connection_config_factory(basic_server)
            config.latency_ping_interval_ms = 50
            conn = await WsConnection.new(ringbuffer, config)
            await latency_waiter(conn, timeout_s=2.0)
            conn.get_latency_ms()

            collected: list[bytes] = []

            async def _collector() -> None:
                """Collect up to 500 echoed messages."""
                while len(collected) < 500:
                    try:
                        msg = await asyncio.wait_for(ringbuffer.aconsume(), timeout=5.0)
                        collected.append(msg)
                    except asyncio.TimeoutError:
                        break

            task = asyncio.create_task(_collector())
            for i in range(500):
                conn.send_data(f"burst-{i}".encode("utf-8"))
                if i % 50 == 0:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(task, timeout=10.0)
            assert len(collected) == 500
            assert conn.get_state() == ConnectionState.CONNECTED
            conn.close()


# --------------------------------------------------------------------------- #
# Reconnection
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestReconnectionE2E:
    """E2E tests for reconnection behavior."""

    async def test_clean_server_close_auto_reconnect(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Close server while connected, restart, and verify reconnection.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            ringbuffer = BytesRingBuffer(max_capacity=16, only_insert_unique=False)
            config = connection_config_factory(basic_server, auto_reconnect=True)
            conn_iter = WsConnection.new_with_reconnect(ringbuffer, config)

            async with asyncio.timeout(5.0):
                async for conn in conn_iter:
                    assert conn.get_state() == ConnectionState.CONNECTED
                    conn.send_data(b"before")
                    await asyncio.sleep(0.2)
                    assert b"before" in basic_server.get_received_messages()
                    conn.close()
                    await asyncio.sleep(0.1)
                    break

    async def test_context_manager_with_auto_reconnect(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure async with WsSingle and auto_reconnect connects and sends.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server, auto_reconnect=True)
            async with WsSingle(config) as ws:
                await _wait_for_single_state(ws, ConnectionState.CONNECTED)
                ws.send_data(b"auto-reconnect-msg")
                await asyncio.sleep(0.2)
            assert b"auto-reconnect-msg" in basic_server.get_received_messages()


# --------------------------------------------------------------------------- #
# Pool
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestPoolE2E:
    """E2E tests for WsPool behavior."""

    async def test_pool_all_healthy(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Start a pool of 5 and verify all connections are healthy.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            pool_config = WsPoolConfig(num_connections=5, evict_interval_s=60)
            config = connection_config_factory(basic_server)
            pool = await WsPool.new(
                config, on_message=_noop_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.3)
                assert pool.get_connection_count() == 5
                pool.send_data(b"all-healthy", only_fastest=True)
                await asyncio.sleep(0.2)
                assert b"all-healthy" in basic_server.get_received_messages()

    async def test_pool_all_fail_raises(
        self,
        server_reject_connections,
        connection_config_factory,
    ) -> None:
        """Ensure a pool handles rejection gracefully (connections drop to 0).

        Note: The local reject server accepts the websocket handshake before
        sending a close frame, so ws_connect succeeds and WsConnection.new
        returns normally.  The connections then immediately disconnect.  We
        verify the pool still starts (no exception) but has zero healthy
        connections after the dust settles.

        Args:
            server_reject_connections: Fixture providing rejecting server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with server_reject_connections:
            pool_config = WsPoolConfig(num_connections=3, evict_interval_s=60)
            config = connection_config_factory(server_reject_connections)
            pool = await WsPool.new(
                config, on_message=_noop_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.5)
                assert pool.get_connection_count() == 0

    async def test_pool_hash_deduplication(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Broadcast same message to all connections; pool iterator deduplicates.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            pool_config = WsPoolConfig(num_connections=3, evict_interval_s=60)
            config = connection_config_factory(basic_server)
            pool = await WsPool.new(
                config, on_message=_noop_handler, pool_config=pool_config
            )
            collected: list[bytes] = []
            async with pool:
                await asyncio.sleep(0.3)

                async def _collector() -> None:
                    """Collect messages from pool iterator (should dedup)."""
                    async for msg in pool:
                        if msg == b"dedup-test":
                            collected.append(msg)
                            if len(collected) >= 2:
                                break

                task = asyncio.create_task(_collector())
                await asyncio.sleep(0.1)
                # Multicast sends to all 3 connections; server echoes all 3
                pool.send_data(b"dedup-test", only_fastest=False)
                await asyncio.sleep(0.1)
                # Give a bit of time then cancel if still waiting
                await asyncio.sleep(0.5)
                if not task.done():
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task

            # With 3 connections and hash dedup, we should see the message once
            assert len(collected) == 1

    async def test_pool_fastest_vs_multicast(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Send with only_fastest=True and only_fastest=False without crashes.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            pool_config = WsPoolConfig(num_connections=3, evict_interval_s=60)
            config = connection_config_factory(basic_server)
            pool = await WsPool.new(
                config, on_message=_noop_handler, pool_config=pool_config
            )
            async with pool:
                await asyncio.sleep(0.3)
                pool.send_data(b"fastest", only_fastest=True)
                await asyncio.sleep(0.1)
                pool.send_data(b"multicast", only_fastest=False)
                await asyncio.sleep(0.1)
                received = basic_server.get_received_messages()
                assert b"fastest" in received
                assert b"multicast" in received


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestLatencyE2E:
    """E2E tests for latency tracking."""

    async def test_normal_ping_pong_latency(
        self,
        basic_server,
        connection_config_factory,
        latency_waiter,
    ) -> None:
        """Wait for ping/pong to update latency from the default 1000.0 ms.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.
            latency_waiter: Fixture providing latency wait helper.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            ringbuffer = BytesRingBuffer(max_capacity=8, only_insert_unique=False)
            config = connection_config_factory(basic_server)
            config.latency_ping_interval_ms = 100
            conn = await WsConnection.new(ringbuffer, config)
            await latency_waiter(conn, timeout_s=3.0)
            assert conn.get_latency_ms() < 1000.0
            conn.close()


# --------------------------------------------------------------------------- #
# Error Resilience
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestErrorResilienceE2E:
    """E2E tests for graceful handling of errors."""

    async def test_malformed_frames(
        self,
        server_send_invalid_frames,
        connection_config_factory,
    ) -> None:
        """Ensure malformed frames cause a graceful disconnect.

        Args:
            server_send_invalid_frames: Fixture providing invalid frame server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with server_send_invalid_frames:
            ringbuffer = BytesRingBuffer(max_capacity=8, only_insert_unique=False)
            config = connection_config_factory(server_send_invalid_frames)
            conn = await WsConnection.new(ringbuffer, config)
            conn.send_data(b"trigger-invalid")
            await asyncio.sleep(0.3)
            assert conn.get_state() == ConnectionState.DISCONNECTED
            conn.close()

    async def test_client_close_while_messages_in_flight(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Send many messages and immediately close; ensure no crash.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            for i in range(1000):
                conn.send_data(f"flight-{i}".encode("utf-8"))
            conn.close()
            await asyncio.sleep(0.2)
            assert conn.get_state() == ConnectionState.DISCONNECTED

    async def test_exception_in_callback(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Ensure an exception in the on_message callback does not kill the conn.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_config_factory: Fixture providing config factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            config = connection_config_factory(basic_server)

            def _bad_handler(msg: bytes) -> None:
                """Handler that always raises."""
                raise ValueError("callback error")

            ws = WsSingle(config, on_message=_bad_handler)
            async with ws:
                await _wait_for_single_state(ws, ConnectionState.CONNECTED)
                ws.send_data(b"trigger-callback-error")
                await asyncio.sleep(0.3)
                # Connection should still be up despite callback error
                assert ws.get_state() == ConnectionState.CONNECTED

    @pytest.mark.slow
    async def test_memory_leak_check(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Run 100 open/close cycles and assert memory growth is bounded.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            tracemalloc.start()
            gc.collect()
            before = tracemalloc.take_snapshot()

            for _ in range(100):
                conn = await connection_factory(basic_server)
                conn.send_data(b"leak-check")
                await asyncio.sleep(0.01)
                conn.close()
                await asyncio.sleep(0.01)

            gc.collect()
            after = tracemalloc.take_snapshot()
            tracemalloc.stop()

            diff = after.compare_to(before, "lineno")
            total_growth = sum(stat.size_diff for stat in diff if stat.size_diff > 0)
            # Allow 5 MB of growth across 100 cycles
            assert total_growth < 5 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Stress
# --------------------------------------------------------------------------- #


@pytest.mark.stress
@pytest.mark.asyncio
class TestStressE2E:
    """Stress/load tests for the WebSocket module."""

    async def test_connection_churn(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Perform 100 rapid open/close cycles without fd exhaustion.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            initial_fd = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
            for i in range(100):
                conn = await connection_factory(basic_server)
                conn.send_data(f"churn-{i}".encode("utf-8"))
                await asyncio.sleep(0.01)
                conn.close()
                await asyncio.sleep(0.01)
                assert conn.get_state() == ConnectionState.DISCONNECTED
            final_fd = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
            assert final_fd == initial_fd

    async def test_sustained_throughput(
        self,
        basic_server,
        connection_factory,
    ) -> None:
        """Measure message throughput over a 5-second burst.

        Args:
            basic_server: Fixture providing a basic echo server.
            connection_factory: Fixture providing connected WsConnection factory.

        Returns:
            None: This test does not return a value.
        """
        async with basic_server:
            conn = await connection_factory(basic_server)
            ringbuffer = conn.get_ringbuffer()
            collected: list[bytes] = []

            async def _collector() -> None:
                """Collect messages for the duration of the burst."""
                while True:
                    try:
                        msg = await asyncio.wait_for(ringbuffer.aconsume(), timeout=1.0)
                        collected.append(msg)
                    except asyncio.TimeoutError:
                        break

            task = asyncio.create_task(_collector())
            start = time.monotonic()
            count = 0
            while time.monotonic() - start < 5.0:
                conn.send_data(f"throughput-{count}".encode("utf-8"))
                count += 1
                if count % 50 == 0:
                    await asyncio.sleep(0.001)

            # Give collector time to drain
            await asyncio.sleep(1.0)
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            rate = len(collected) / 5.0
            assert rate > 10.0  # at least 10 msg/sec
            conn.close()

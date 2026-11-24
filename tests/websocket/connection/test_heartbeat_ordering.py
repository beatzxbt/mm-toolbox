"""Deterministic heartbeat ordering at deadline and across local read pauses."""

import asyncio
import queue
import select
import threading
from types import SimpleNamespace

import pytest
import picows
import websockets

import mm_toolbox.websocket.connection as connection_module
from mm_toolbox.websocket import WsSingle
from mm_toolbox.websocket import WsConnectionConfig
from tests.websocket.connection.test_native import _native
from tests.websocket.tls_support import tls_contexts


async def _stop_automatic_probe(conn):
    task = conn._latency_task
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_manual_empty_and_counter_pongs_cannot_ack_internal_probe(
    basic_server, connection_config_factory, monkeypatch
):
    async with basic_server:
        async with WsSingle(connection_config_factory(basic_server)) as client:
            conn = client._ws_conn
            await _stop_automatic_probe(conn)
            ticks = 0

            async def tick(interval):
                nonlocal ticks
                ticks += 1
                if ticks > 1:
                    raise asyncio.CancelledError

            with monkeypatch.context() as patched:
                patched.setattr(
                    connection_module,
                    "asyncio",
                    SimpleNamespace(sleep=tick, CancelledError=asyncio.CancelledError),
                )
                await conn._latency_loop()
            payload = _native.pending_probe_payload(conn)
            assert payload.startswith(b"mm-toolbox\x00")
            assert len(payload) < 125
            _native.deliver_pong(conn, b"")
            _native.deliver_pong(conn, (1).to_bytes(8, "big"))
            assert _native.ping_sent_time(conn) > 0
            assert conn._heartbeat_snapshot()["pongs_matched"] == 0
            _native.deliver_pong(conn, payload)
            assert _native.ping_sent_time(conn) == 0
            assert conn._heartbeat_snapshot()["pongs_matched"] == 1


@pytest.mark.asyncio
async def test_deadline_grace_polls_an_already_readable_tls_socket(
    tmp_path, monkeypatch
):
    server_context, client_context = tls_contexts(tmp_path)
    address = queue.Queue()
    sent = threading.Event()
    stop = threading.Event()

    async def peer(socket):
        await socket.recv()
        await socket.pong(b"ready")
        sent.set()
        await socket.wait_closed()

    def serve():
        async def run():
            async with websockets.serve(
                peer, "127.0.0.1", 0, ssl=server_context, close_timeout=1
            ) as server:
                address.put(server.sockets[0].getsockname()[1])
                await asyncio.to_thread(stop.wait)

        asyncio.run(run())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    port = address.get(timeout=5)

    async def trusted_connect(**kwargs):
        kwargs["ssl_context"] = client_context
        return await picows.ws_connect(**kwargs)

    monkeypatch.setattr(connection_module, "ws_connect", trusted_connect)
    try:
        async with WsSingle(
            WsConnectionConfig.default(f"wss://127.0.0.1:{port}")
        ) as client:
            conn = client._ws_conn
            await _stop_automatic_probe(conn)
            _native.arm_probe(conn, b"ready")
            client.send_data(b"reply")
            assert sent.wait(5)
            assert select.select([_native.read_socket(conn)], [], [], 5)[0]
            assert conn._heartbeat_snapshot()["pongs_received"] == 0
            ticks = 0

            async def tick(interval):
                nonlocal ticks
                ticks += 1
                if ticks > 1:
                    raise asyncio.CancelledError

            with monkeypatch.context() as patched:
                patched.setattr(
                    connection_module,
                    "asyncio",
                    SimpleNamespace(sleep=tick, CancelledError=asyncio.CancelledError),
                )
                await conn._latency_loop()
            assert conn.is_connected()
            assert conn._heartbeat_snapshot()["pongs_matched"] == 1
    finally:
        stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive()


@pytest.mark.asyncio
async def test_already_queued_pong_is_serviced_before_expired_watchdog(
    basic_server, connection_config_factory, monkeypatch
):
    async with basic_server:
        async with WsSingle(connection_config_factory(basic_server)) as client:
            conn = client._ws_conn
            await _stop_automatic_probe(conn)
            _native.expire_ping(conn)
            loop = asyncio.get_running_loop()
            ticks = 0

            async def tick(interval):
                nonlocal ticks
                if interval == 0:
                    await asyncio.sleep(0)
                    return
                ticks += 1
                if ticks > 1:
                    raise asyncio.CancelledError
                # A delayed watchdog task can precede an already queued read
                # callback after a busy event-loop turn. No wall-clock sleeps.
                loop.call_soon(_native.deliver_pong, conn)

            with monkeypatch.context() as patched:
                patched.setattr(
                    connection_module,
                    "asyncio",
                    SimpleNamespace(
                        sleep=tick,
                        CancelledError=asyncio.CancelledError,
                        get_running_loop=asyncio.get_running_loop,
                    ),
                )
                await conn._latency_loop()
            await asyncio.sleep(0)
            assert conn.is_connected()
            assert _native.ping_sent_time(conn) == 0.0


@pytest.mark.asyncio
async def test_completed_pause_is_excluded_from_pending_probe_age(
    basic_server, connection_config_factory, monkeypatch
):
    async with basic_server:
        async with WsSingle(connection_config_factory(basic_server)) as client:
            conn = client._ws_conn
            await _stop_automatic_probe(conn)
            _native.arm_probe(conn, b"before pause")
            _native.simulate_completed_read_pause(conn, 10000)
            pending = _native.ping_sent_time(conn)
            ticks = 0

            async def tick(interval):
                nonlocal ticks
                if interval == 0:
                    await asyncio.sleep(0)
                    return
                ticks += 1
                if ticks > 1:
                    raise asyncio.CancelledError

            with monkeypatch.context() as patched:
                patched.setattr(
                    connection_module,
                    "asyncio",
                    SimpleNamespace(
                        sleep=tick,
                        CancelledError=asyncio.CancelledError,
                        get_running_loop=asyncio.get_running_loop,
                    ),
                )
                await conn._latency_loop()
            assert conn.is_connected()
            assert _native.ping_sent_time(conn) == pending
            stats = conn._heartbeat_snapshot()
            assert stats["pending_readable_age_ms"] < stats["pending_ping_age_ms"]
            assert stats["paused_ms_since_probe"] >= 10000
            _native.deliver_pong(conn, b"before pause")
            assert conn._heartbeat_snapshot()["pongs_matched"] == 1
            assert conn._heartbeat_snapshot()["latency_samples_affected_by_pause"] == 1


@pytest.mark.asyncio
async def test_repeated_short_pauses_do_not_reset_missing_pong_deadline(
    basic_server, connection_config_factory, monkeypatch
):
    async with basic_server:
        async with WsSingle(connection_config_factory(basic_server)) as client:
            conn = client._ws_conn
            await _stop_automatic_probe(conn)
            _native.arm_probe(conn, b"missing")

            async def tick(interval):
                for _ in range(20):
                    _native.pause_client_reads(conn)
                    _native.resume_client_reads(conn)

            with monkeypatch.context() as patched:
                patched.setattr(
                    connection_module,
                    "asyncio",
                    SimpleNamespace(
                        sleep=tick,
                        CancelledError=asyncio.CancelledError,
                        get_running_loop=asyncio.get_running_loop,
                    ),
                )
                await conn._latency_loop()
            assert not conn.is_connected()
            assert conn._heartbeat_snapshot()["pending_readable_age_ms"] > 3000


@pytest.mark.asyncio
@pytest.mark.parametrize("during_grace", ["cancel", "close", "new_probe"])
async def test_grace_rechecks_probe_and_is_safe_to_cancel_or_close(
    basic_server, connection_config_factory, monkeypatch, during_grace
):
    async with basic_server:
        async with WsSingle(connection_config_factory(basic_server)) as client:
            conn = client._ws_conn
            await _stop_automatic_probe(conn)
            _native.arm_probe(conn, b"old")
            ticks = 0

            async def tick(interval):
                nonlocal ticks
                ticks += 1
                if ticks > 1:
                    raise asyncio.CancelledError

            original_wake = connection_module._wake_pending_future
            callbacks = []
            loop = asyncio.get_running_loop()
            old_handler = loop.get_exception_handler()
            loop.set_exception_handler(lambda loop, context: callbacks.append(context))
            task = None

            def mutate_before_wake(future):
                if during_grace == "cancel":
                    task.cancel()
                elif during_grace == "close":
                    client.close()
                else:
                    _native.replace_probe_payload(conn, b"new")
                original_wake(future)

            try:
                with monkeypatch.context() as patched:
                    patched.setattr(
                        connection_module,
                        "asyncio",
                        SimpleNamespace(
                            sleep=tick,
                            CancelledError=asyncio.CancelledError,
                            get_running_loop=asyncio.get_running_loop,
                        ),
                    )
                    patched.setattr(
                        connection_module, "_wake_pending_future", mutate_before_wake
                    )
                    task = asyncio.create_task(conn._latency_loop())
                    conn._latency_task = task
                    await task
                await asyncio.sleep(0)
            finally:
                loop.set_exception_handler(old_handler)
            assert callbacks == []
            assert conn.is_connected() == (during_grace != "close")
            assert conn._heartbeat_snapshot()["pings_sent"] == 0

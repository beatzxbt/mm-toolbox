"""Verified TLS must survive actual pause/resume without transport callbacks."""

import asyncio
import ssl

import picows
import pytest
import websockets

import mm_toolbox.websocket.connection as connection_module
from mm_toolbox.websocket import WsConnectionConfig, WsPool, WsPoolConfig, WsSingle
from tests.websocket.tls_support import tls_contexts


@pytest.mark.asyncio
@pytest.mark.parametrize("connections", [1, 3])
async def test_verified_tls_pause_resume_has_no_callback_errors(
    tmp_path, monkeypatch, connections
):
    server_context, client_context = tls_contexts(tmp_path)
    per_phase, phases = 512, 2
    sent = [0] * phases
    completed = [asyncio.Event() for _ in range(phases)]
    pause_calls, resume_calls = [], []
    callbacks = []
    loop = asyncio.get_running_loop()
    original_handler = loop.get_exception_handler()

    def unexpected_callback(loop, context):
        callbacks.append(context)
        loop.default_exception_handler(context)

    async def trusted_connect(**kwargs):
        assert kwargs["use_aiofastnet"] is False
        assert "ssl_context" not in kwargs
        kwargs["ssl_context"] = client_context
        transport, listener = await picows.ws_connect(**kwargs)
        underlying = transport.underlying_transport
        assert type(underlying).__module__.startswith("asyncio.")
        pause, resume = underlying.pause_reading, underlying.resume_reading

        def pause_reading():
            pause_calls.append(True)
            pause()

        def resume_reading():
            resume_calls.append(True)
            resume()

        monkeypatch.setattr(underlying, "pause_reading", pause_reading)
        monkeypatch.setattr(underlying, "resume_reading", resume_reading)
        return transport, listener

    monkeypatch.setattr(connection_module, "ws_connect", trusted_connect)

    async def source(socket):
        for phase in range(phases):
            await socket.recv()
            for index in range(phase * per_phase, (phase + 1) * per_phase):
                await socket.send(index.to_bytes(8, "big") + b"x" * 248)
                if (index + 1) % 256 == 0:
                    await asyncio.sleep(0)
            sent[phase] += 1
            if sent[phase] == connections:
                completed[phase].set()
        await socket.wait_closed()

    loop.set_exception_handler(unexpected_callback)
    try:
        async with websockets.serve(
            source, "127.0.0.1", 0, ssl=server_context, compression=None
        ) as server:
            uri = f"wss://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            config = WsConnectionConfig.default(uri, auto_reconnect=False)
            client = (
                WsSingle(config)
                if connections == 1
                else WsPool(
                    config,
                    pool_config=WsPoolConfig(connections, 60, hash_capacity=2048),
                )
            )
            received = []
            async with client:

                async def consume():
                    async for payload in client:
                        received.append(int.from_bytes(payload[:8], "big"))

                reader = asyncio.create_task(consume())
                try:
                    conns = (
                        [client._ws_conn]
                        if connections == 1
                        else list(client._conns.values())
                    )
                    for phase in range(phases):
                        client.send_data(b"next phase", only_fastest=False)
                        await asyncio.wait_for(completed[phase].wait(), 5)
                        expected = (phase + 1) * per_phase
                        async with asyncio.timeout(5):
                            while (
                                len(received) != expected
                                or len(client._ringbuffer)
                                or any(conn.get_seq_id() != expected for conn in conns)
                            ):
                                assert client.get_dropped_messages() == 0
                                if reader.done():
                                    reader.result()
                                    pytest.fail(
                                        "TLS receive stream ended before all messages arrived"
                                    )
                                await asyncio.sleep(0)
                        assert client.get_dropped_messages() == 0
                        assert client._ringbuffer.get_buffered_bytes() == 0
                    assert received == list(range(per_phase * phases))
                finally:
                    reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(original_handler)
    assert pause_calls
    assert resume_calls
    assert callbacks == []


@pytest.mark.asyncio
@pytest.mark.parametrize("trusted_wrong_hostname", [False, True])
async def test_tls_certificate_failure_propagates_without_backend_retry(
    tmp_path, monkeypatch, trusted_wrong_hostname
):
    server_context, client_context = tls_contexts(tmp_path)
    attempts = []

    async def connect(**kwargs):
        attempts.append(kwargs["use_aiofastnet"])
        assert "ssl_context" not in kwargs
        if trusted_wrong_hostname:
            kwargs["ssl_context"] = client_context
            kwargs["server_hostname"] = "not-localhost.invalid"
        return await picows.ws_connect(**kwargs)

    monkeypatch.setattr(connection_module, "ws_connect", connect)

    async def source(socket):
        pytest.fail(
            "An untrusted or mismatched certificate must not reach WebSocket startup"
        )

    async with websockets.serve(source, "127.0.0.1", 0, ssl=server_context) as server:
        uri = f"wss://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        client = WsSingle(WsConnectionConfig.default(uri))
        with pytest.raises(ssl.SSLCertVerificationError) as rejected:
            await client.__aenter__()
        if trusted_wrong_hostname:
            assert "hostname mismatch" in rejected.value.verify_message.lower()
        else:
            assert "self-signed" in rejected.value.verify_message.lower()
        assert client.get_connection_count() == 0
    assert attempts == [False]

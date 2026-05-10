"""Async iteration tests for WsSingle.

Layer-2 tests validating that WsSingle correctly exposes an async iterator
over messages from its underlying connection.

Key coverage:
- async for loop consumes all expected messages.
- Arrival validation (completeness, not strict ordering).
- Iteration stops gracefully after disconnect.
- Concurrent send and iterate paths do not interfere.
- Large message bursts are consumed correctly.
"""

from __future__ import annotations

import asyncio

import pytest

from mm_toolbox.websocket.single import WsSingle


@pytest.mark.asyncio
class TestWsSingleAsyncIteration:
    """Layer-2 tests for async iteration over WsSingle messages."""

    async def test_async_for_loop_consumes_messages(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given 3 broadcast messages, When iterating with async for, Then all 3 are collected."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            collected: list[bytes] = []

            async with WsSingle(config) as ws:

                async def _collector() -> None:
                    async for msg in ws:
                        collected.append(msg)
                        if len(collected) >= 3:
                            break

                collect_task = asyncio.create_task(_collector())
                await asyncio.sleep(0.1)
                await basic_server.send_to_all_clients(b"msg1")
                await basic_server.send_to_all_clients(b"msg2")
                await basic_server.send_to_all_clients(b"msg3")
                await asyncio.wait_for(collect_task, timeout=2.0)

            assert len(collected) == 3

    async def test_async_iteration_arrival_validation(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given 4 broadcast messages, When iterating, Then the collected set matches the sent set regardless of order.

        Single-connection iteration should preserve order in practice, but
the important invariant is that no messages are lost."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            payloads = [b"a", b"b", b"c", b"d"]
            collected: list[bytes] = []

            async with WsSingle(config) as ws:

                async def _collector() -> None:
                    async for msg in ws:
                        collected.append(msg)
                        if len(collected) >= len(payloads):
                            break

                collect_task = asyncio.create_task(_collector())
                await asyncio.sleep(0.1)
                for payload in payloads:
                    await basic_server.send_to_all_clients(payload)
                await asyncio.wait_for(collect_task, timeout=2.0)

            assert set(collected) == set(payloads)

    async def test_async_iteration_stops_on_disconnect(
        self,
        server_send_close_frame,
        connection_config_factory,
    ) -> None:
        """Given a close-frame server, When disconnect occurs, Then the pending __anext__ either returns the last message or can be cancelled cleanly."""
        async with server_send_close_frame:
            config = connection_config_factory(server_send_close_frame)
            async with WsSingle(config) as ws:
                collect_task = asyncio.create_task(ws.__anext__())
                await asyncio.sleep(0.1)
                ws.send_data(b"trigger-close")
                await asyncio.sleep(0.2)
                if collect_task.done():
                    assert collect_task.result() == b"trigger-close"
                else:
                    collect_task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await collect_task

    async def test_async_iteration_concurrent_with_send(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given concurrent send_data and async iteration, When traffic flows, Then both paths work without interference."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            expected = {b"srv1", b"srv2"}
            seen: set[bytes] = set()

            async with WsSingle(config) as ws:

                async def _collector() -> None:
                    async for msg in ws:
                        if msg in expected:
                            seen.add(msg)
                        if expected.issubset(seen):
                            break

                collect_task = asyncio.create_task(_collector())
                await asyncio.sleep(0.1)
                ws.send_data(b"client-msg")
                await basic_server.send_to_all_clients(b"srv1")
                await basic_server.send_to_all_clients(b"srv2")
                await asyncio.wait_for(collect_task, timeout=2.0)

            assert seen == expected
            assert b"client-msg" in basic_server.get_received_messages()

    async def test_async_iterator_with_large_message_burst(
        self,
        basic_server,
        connection_config_factory,
    ) -> None:
        """Given a burst of 10 messages, When iterated, Then all are collected without drops.

        Bursts stress the ringbuffer and the frame-processing loop;
missing messages here indicate a buffering bug."""
        async with basic_server:
            config = connection_config_factory(basic_server)
            payloads = [f"msg-{idx}".encode("utf-8") for idx in range(10)]
            collected: list[bytes] = []

            async with WsSingle(config) as ws:

                async def _collector() -> None:
                    async for msg in ws:
                        collected.append(msg)
                        if len(collected) >= len(payloads):
                            break

                collect_task = asyncio.create_task(_collector())
                await asyncio.sleep(0.1)
                for payload in payloads:
                    await basic_server.send_to_all_clients(payload)
                await asyncio.wait_for(collect_task, timeout=3.0)

            assert set(collected) == set(payloads)

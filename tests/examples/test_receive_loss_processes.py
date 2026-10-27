"""A local native burst makes the real multi-process example fail and clean up."""

import asyncio
import os
from pathlib import Path
import sys
import textwrap

import pytest
import websockets


@pytest.mark.asyncio
async def test_native_receive_overwrite_fails_parent_and_removes_owned_files(tmp_path):
    # A one-slot buffer is an explicit test fault. Production examples retain
    # their normal bounded capacity; a local burst deterministically overwrites.
    script = tmp_path / "local_receive_fault.py"
    script.write_text(
        textwrap.dedent(
            """\
            import sys
            import tempfile
            from examples import binance_stream as demo
            from mm_toolbox.websocket import WsSingle
            from mm_toolbox.websocket.connection import _ReceiveBuffer

            class LocalSingle(WsSingle):
                def __init__(self, config):
                    config.wss_url = sys.argv[1]
                    super().__init__(config)
                    self._ringbuffer = _ReceiveBuffer(1)

            demo.WsSingle = LocalSingle
            if __name__ == "__main__":
                with tempfile.TemporaryDirectory(prefix="fault-", dir=sys.argv[2]) as runtime:
                    demo._run_processes(4, runtime)
            """
        ),
        encoding="utf-8",
    )

    async def burst(socket):
        for _ in range(300):
            await socket.send(b"{}")
        await socket.wait_closed()

    project = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project) + os.pathsep + env.get("PYTHONPATH", "")
    async with websockets.serve(burst, "127.0.0.1", 0) as server:
        uri = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script),
            uri,
            str(tmp_path),
            cwd=project,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), 20)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
    assert proc.returncode != 0
    assert b"receive buffer overwrote" in stderr
    assert b"Example child process failed" in stderr
    assert b"Orderbook snapshot received" not in stdout
    assert b"1s Candle:" not in stdout
    assert list(tmp_path.glob("fault-*")) == []

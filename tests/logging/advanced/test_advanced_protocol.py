"""Layer 3 — Integration tests for the advanced logging binary protocol.

Because ``BinaryWriter`` and ``BinaryReader`` are cdef-only Cython classes,
the protocol is validated indirectly through worker/master integration.
Covers round-trip correctness, batching, binary payloads (null/high bytes),
multiprocess delivery, empty messages, 1 MB messages, and unicode worker names.
"""

from __future__ import annotations

import multiprocessing
import time

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.master import MasterLogger
from mm_toolbox.logging.advanced.pylog import PyLog, PyLogLevel
from mm_toolbox.logging.advanced.worker import WorkerLogger


class CaptureHandler(BaseLogHandler):
    """Test-double handler that captures every received log for inspection."""

    def __init__(self):
        super().__init__()
        self.logs = []

    def push(self, logs: list[PyLog]) -> None:
        """Append the entire batch to an internal list.

        Args:
            logs: Batch of log entries received from the master.
        """
        self.logs.extend(logs)


def _protocol_worker(path: str, name: str, messages: list[bytes]) -> None:
    """Target function for a multiprocessing worker that sends binary messages.

    Args:
        path: IPC shared-memory path.
        name: Worker name tag.
        messages: List of raw byte payloads to log.
    """
    config = LoggerConfig(path=path)
    logger = WorkerLogger(config=config, name=name)
    for msg in messages:
        logger.info(msg_bytes=msg)
    logger.shutdown()


class TestProtocolIndirect:
    """Layer 3 — Indirect protocol validation via worker/master integration."""

    def test_protocol_roundtrip_basic(self, tmp_path):
        """Given a single INFO message, the master receives the exact name, level, and payload."""
        path = str(tmp_path / "test_protocol_basic.shm")
        config = LoggerConfig(path=path)
        handler = CaptureHandler()
        master = MasterLogger(config=config, log_handlers=[handler])
        handler.add_primary_config(config)

        logger = WorkerLogger(config=config, name="TEST")
        logger.info(msg_bytes=b"hello world")
        logger.shutdown()

        time.sleep(0.1)
        master.shutdown()

        assert len(handler.logs) >= 1
        assert handler.logs[0].name == b"TEST"
        assert handler.logs[0].message == b"hello world"
        assert handler.logs[0].level == PyLogLevel.INFO

    def test_protocol_multiple_messages(self, tmp_path):
        """Given 25 messages, all are decoded and delivered despite batching boundaries."""
        path = str(tmp_path / "test_protocol_multi.shm")
        config = LoggerConfig(path=path, max_batch_messages=10)
        handler = CaptureHandler()
        master = MasterLogger(config=config, log_handlers=[handler])
        handler.add_primary_config(config)

        logger = WorkerLogger(config=config, name="BATCH")
        messages = [f"message_{i}".encode() for i in range(25)]
        for msg in messages:
            logger.info(msg_bytes=msg)
        logger.shutdown()

        time.sleep(0.2)
        master.shutdown()

        received_messages = [log.message for log in handler.logs]
        for msg in messages:
            assert msg in received_messages

    def test_protocol_binary_payload(self, tmp_path):
        """Given payloads with null bytes, high bytes, and mixed content, they survive round-trip intact."""
        path = str(tmp_path / "test_protocol_binary.shm")
        config = LoggerConfig(path=path)
        handler = CaptureHandler()
        master = MasterLogger(config=config, log_handlers=[handler])
        handler.add_primary_config(config)

        logger = WorkerLogger(config=config, name="BINARY")

        payloads = [
            b"\x00\x01\x02\x03",
            b"\xff\xfe\xfd\xfc",
            b"\x00" * 100,
            b"\xff" * 100,
            b"mixed\x00null\xffhigh",
        ]

        for payload in payloads:
            logger.info(msg_bytes=payload)
        logger.shutdown()

        time.sleep(0.1)
        master.shutdown()

        received = {log.message for log in handler.logs}
        for payload in payloads:
            assert payload in received

    def test_protocol_multiprocess(self, tmp_path):
        """Given a separate process emitting 10 messages, the master receives all of them."""
        path = str(tmp_path / "test_protocol_mp.shm")
        config = LoggerConfig(path=path)
        handler = CaptureHandler()
        master = MasterLogger(config=config, log_handlers=[handler])
        handler.add_primary_config(config)

        messages = [f"proc_msg_{i}".encode() for i in range(10)]
        p = multiprocessing.Process(
            target=_protocol_worker,
            args=(path, "MPWORKER", messages),
        )
        p.start()
        p.join(timeout=10.0)

        assert p.exitcode == 0

        time.sleep(0.1)
        master.shutdown()

        received = [log.message for log in handler.logs]
        for msg in messages:
            assert msg in received

    def test_protocol_empty_message(self, tmp_path):
        """Given an empty byte string, the master receives a log with an empty message."""
        path = str(tmp_path / "test_protocol_empty.shm")
        config = LoggerConfig(path=path)
        handler = CaptureHandler()
        master = MasterLogger(config=config, log_handlers=[handler])
        handler.add_primary_config(config)

        logger = WorkerLogger(config=config, name="EMPTY")
        logger.info(msg_bytes=b"")
        logger.shutdown()

        time.sleep(0.1)
        master.shutdown()

        assert len(handler.logs) >= 1
        assert handler.logs[0].message == b""

    def test_protocol_large_message(self, tmp_path):
        """Given a 1 MB payload, the master receives it without truncation or corruption."""
        path = str(tmp_path / "test_protocol_large.shm")
        config = LoggerConfig(path=path)
        handler = CaptureHandler()
        master = MasterLogger(config=config, log_handlers=[handler])
        handler.add_primary_config(config)

        logger = WorkerLogger(config=config, name="LARGE")
        large_msg = b"X" * (1024 * 1024)
        logger.info(msg_bytes=large_msg)
        logger.shutdown()

        time.sleep(0.5)
        master.shutdown()

        assert len(handler.logs) >= 1
        assert handler.logs[0].message == large_msg

    def test_protocol_unicode_name(self, tmp_path):
        """Given a unicode worker name, it is UTF-8 encoded and decoded correctly."""
        path = str(tmp_path / "test_protocol_unicode.shm")
        config = LoggerConfig(path=path)
        handler = CaptureHandler()
        master = MasterLogger(config=config, log_handlers=[handler])
        handler.add_primary_config(config)

        name = "\u6d4b\u8bd5\u5de5\u4eba"  # "Test worker" in Chinese
        logger = WorkerLogger(config=config, name=name)
        logger.info(msg_bytes=b"test")
        logger.shutdown()

        time.sleep(0.1)
        master.shutdown()

        assert len(handler.logs) >= 1
        assert handler.logs[0].name == name.encode("utf-8")

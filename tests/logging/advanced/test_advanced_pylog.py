"""Tests for Python-accessible log types."""

from __future__ import annotations

import pytest

from mm_toolbox.logging.advanced.pylog import PyLog, PyLogLevel


class TestPyLogLevel:
    def test_level_values(self):
        assert PyLogLevel.TRACE == 0
        assert PyLogLevel.DEBUG == 1
        assert PyLogLevel.INFO == 2
        assert PyLogLevel.WARNING == 3
        assert PyLogLevel.ERROR == 4

    def test_is_lower(self):
        assert PyLogLevel.DEBUG.is_lower(PyLogLevel.INFO) is True
        assert PyLogLevel.INFO.is_lower(PyLogLevel.DEBUG) is False
        assert PyLogLevel.TRACE.is_lower(PyLogLevel.ERROR) is True

    def test_is_higher(self):
        assert PyLogLevel.WARNING.is_higher(PyLogLevel.INFO) is True
        assert PyLogLevel.INFO.is_higher(PyLogLevel.WARNING) is False
        assert PyLogLevel.ERROR.is_higher(PyLogLevel.TRACE) is True


class TestPyLog:
    def test_creation(self):
        log = PyLog(
            timestamp_ns=1234567890,
            name=b"test_logger",
            level=PyLogLevel.INFO,
            message=b"test message",
        )
        assert log.timestamp_ns == 1234567890
        assert log.name == b"test_logger"
        assert log.level == PyLogLevel.INFO
        assert log.message == b"test message"

    def test_iteration(self):
        log = PyLog(1, b"name", PyLogLevel.DEBUG, b"msg")
        items = list(log)
        assert items == [1, b"name", PyLogLevel.DEBUG, b"msg"]

    def test_getitem_valid_indices(self):
        log = PyLog(1, b"name", PyLogLevel.DEBUG, b"msg")
        assert log[0] == 1
        assert log[1] == b"name"
        assert log[2] == PyLogLevel.DEBUG
        assert log[3] == b"msg"

    def test_getitem_index_error(self):
        log = PyLog(1, b"name", PyLogLevel.DEBUG, b"msg")
        with pytest.raises(IndexError, match="out of range"):
            _ = log[4]
        with pytest.raises(IndexError, match="out of range"):
            _ = log[-1]

    def test_equality(self):
        log1 = PyLog(1, b"name", PyLogLevel.INFO, b"msg")
        log2 = PyLog(1, b"name", PyLogLevel.INFO, b"msg")
        assert log1 == log2

    def test_inequality(self):
        log1 = PyLog(1, b"name", PyLogLevel.INFO, b"msg")
        log2 = PyLog(2, b"name", PyLogLevel.INFO, b"msg")
        assert log1 != log2

import pytest

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.pylog import PyLogLevel


class TestLoggerConfig:
    @pytest.mark.parametrize(
        "base_level",
        [
            PyLogLevel.TRACE,
            PyLogLevel.DEBUG,
            PyLogLevel.INFO,
            PyLogLevel.WARNING,
            PyLogLevel.ERROR,
        ],
    )
    def test_valid_base_level(self, base_level):
        config = LoggerConfig(base_level=base_level)
        assert config.base_level == base_level

    def test_default_base_level(self):
        config = LoggerConfig()
        assert config.base_level == PyLogLevel.INFO

    @pytest.mark.parametrize("fallback_level", ["INFO", 2.5, None])
    def test_base_level_invalid_inputs_fallback(self, fallback_level):
        # Invalid inputs should not raise; they should fallback to INFO
        config = LoggerConfig(base_level=fallback_level)
        assert int(config.base_level) == int(PyLogLevel.INFO)

    @pytest.mark.parametrize("do_stdout", [True, False])
    def test_valid_do_stdout(self, do_stdout):
        config = LoggerConfig(do_stdout=do_stdout)
        assert config.do_stdout == do_stdout

    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            ("True", True),
            (1, True),
            (None, False),
        ],
    )
    def test_do_stdout_coercion(self, input_val, expected):
        config = LoggerConfig(do_stdout=input_val)
        assert config.do_stdout is expected

    @pytest.mark.parametrize(
        "str_format",
        [
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            "%(message)s",
            "%(asctime)s %(message)s",
        ],
    )
    def test_valid_str_format(self, str_format):
        config = LoggerConfig(str_format=str_format)
        assert config.str_format == str_format

    @pytest.mark.parametrize(
        "invalid_format",
        [
            "%(asctime)s [%(levelname)s] %(name)s",  # missing %(message)s
            "",  # empty
            123,  # not str
            None,
        ],
    )
    def test_invalid_str_format(self, invalid_format):
        with pytest.raises((ValueError, TypeError)):
            LoggerConfig(str_format=invalid_format)

    @pytest.mark.parametrize("path", ["/tmp/test.shm", "/tmp/hft_logger.shm"])
    def test_valid_path(self, path):
        config = LoggerConfig(path=path)
        assert config.path == path

    @pytest.mark.parametrize("invalid_path", [123, None, b"ipc://test"])
    def test_invalid_path_type(self, invalid_path):
        with pytest.raises(TypeError):
            LoggerConfig(path=invalid_path)

    @pytest.mark.parametrize("flush_interval", [0.1, 1.0, 5.0, 10.0])
    def test_valid_flush_interval(self, flush_interval):
        config = LoggerConfig(flush_interval_s=flush_interval)
        assert config.flush_interval_s == flush_interval

    @pytest.mark.parametrize("invalid_interval", [0.0, -1.0, "1.0", None])
    def test_invalid_flush_interval(self, invalid_interval):
        with pytest.raises((ValueError, TypeError)):
            LoggerConfig(flush_interval_s=invalid_interval)

    @pytest.mark.parametrize("shm_capacity_bytes", [1, 1024, 67108864])
    def test_valid_shm_capacity_bytes(self, shm_capacity_bytes):
        config = LoggerConfig(shm_capacity_bytes=shm_capacity_bytes)
        assert config.shm_capacity_bytes == shm_capacity_bytes

    @pytest.mark.parametrize("invalid_shm_capacity_bytes", [0, -1, -1024])
    def test_invalid_shm_capacity_bytes(self, invalid_shm_capacity_bytes):
        with pytest.raises(ValueError):
            LoggerConfig(shm_capacity_bytes=invalid_shm_capacity_bytes)

    @pytest.mark.parametrize("max_batch_messages", [1, 100, 10000])
    def test_valid_max_batch_messages(self, max_batch_messages):
        config = LoggerConfig(max_batch_messages=max_batch_messages)
        assert config.max_batch_messages == max_batch_messages

    @pytest.mark.parametrize("invalid_max_batch_messages", [0, -1, -100])
    def test_invalid_max_batch_messages(self, invalid_max_batch_messages):
        with pytest.raises(ValueError):
            LoggerConfig(max_batch_messages=invalid_max_batch_messages)

    @pytest.mark.parametrize("max_batch_bytes", [1, 1024, 1048576])
    def test_valid_max_batch_bytes(self, max_batch_bytes):
        config = LoggerConfig(max_batch_bytes=max_batch_bytes)
        assert config.max_batch_bytes == max_batch_bytes

    @pytest.mark.parametrize("invalid_max_batch_bytes", [0, -1, -1024])
    def test_invalid_max_batch_bytes(self, invalid_max_batch_bytes):
        with pytest.raises(ValueError):
            LoggerConfig(max_batch_bytes=invalid_max_batch_bytes)

    @pytest.mark.parametrize("emit_internal", [True, False])
    def test_valid_emit_internal(self, emit_internal):
        config = LoggerConfig(emit_internal=emit_internal)
        assert config.emit_internal == emit_internal

    def test_multiple_params(self):
        config = LoggerConfig(
            base_level=PyLogLevel.DEBUG,
            do_stdout=True,
            str_format="%(message)s",
            path="/tmp/multi.shm",
            flush_interval_s=2.0,
            emit_internal=True,
            shm_capacity_bytes=134217728,
            max_batch_messages=5000,
            max_batch_bytes=2097152,
        )
        assert config.base_level == PyLogLevel.DEBUG
        assert config.do_stdout is True
        assert config.str_format == "%(message)s"
        assert config.path == "/tmp/multi.shm"
        assert config.flush_interval_s == 2.0
        assert config.emit_internal is True
        assert config.shm_capacity_bytes == 134217728
        assert config.max_batch_messages == 5000
        assert config.max_batch_bytes == 2097152

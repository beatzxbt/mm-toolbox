"""Tests covering standard logger handler infrastructure."""

import pytest

from mm_toolbox.logging.standard.config import LoggerConfig
from mm_toolbox.logging.standard.handlers import BaseLogHandler


class DummyPayloadHandler(BaseLogHandler):
    def __init__(self) -> None:
        super().__init__()
        self.received = []
        self.closed = False

    async def push(self, buffer: list[str]) -> None:  # pragma: no cover
        self.received.append(tuple(buffer))

    async def aclose(self) -> None:
        self.closed = True
        await super().aclose()


class TestBaseLogHandler:
    """Basic behaviour of BaseLogHandler."""

    @pytest.mark.asyncio
    async def test_primary_config_added(self):
        handler = DummyPayloadHandler()
        cfg = LoggerConfig()
        handler.add_primary_config(cfg)
        assert handler.primary_config is cfg

    @pytest.mark.asyncio
    async def test_aclose_sets_flag(self):
        handler = DummyPayloadHandler()
        await handler.aclose()
        assert handler.closed

import asyncio
import unittest
from unittest.mock import patch, AsyncMock

from mm_toolbox.src.logging import Logger

# Two commented out tests need working on, i'm finding difficultly to properly
# ensure their function. Any help is much appreciated!


class TestLogger(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.logger = Logger(debug_mode=True)

    # @patch("mm_toolbox.src.logging.logger.aiofiles.open", new_callable=AsyncMock)
    # async def test_write_logs_to_file(self, mock_open):
    #     self.logger.msgs = ["test log message"]

    #     # Mock the context manager methods for async file operations
    #     mock_file = AsyncMock()
    #     mock_open.return_value.__aenter__.return_value = mock_file

    #     await self.logger._write_logs_to_file_()

    #     expected_lines = ["test log message\n"]
    #     mock_file.writelines.assert_called_once_with(expected_lines)
    #     self.assertEqual(self.logger.msgs, [])

    @patch("mm_toolbox.src.logging.logger.Logger._message_", new_callable=AsyncMock)
    async def test_critical(self, mock_message):
        await self.logger.critical("test_topic", "test_message")
        mock_message.assert_called_once_with(
            "CRITICAL", "TEST_TOPIC", "test_message", flush_buffer=True
        )

    @patch("mm_toolbox.src.logging.logger.Logger._message_", new_callable=AsyncMock)
    async def test_debug(self, mock_message):
        await self.logger.debug("test_topic", "test_message")
        mock_message.assert_called_once_with("DEBUG", "TEST_TOPIC", "test_message")

    @patch("mm_toolbox.src.logging.logger.Logger._message_", new_callable=AsyncMock)
    async def test_error(self, mock_message):
        await self.logger.error("test_topic", "test_message")
        mock_message.assert_called_once_with("ERROR", "TEST_TOPIC", "test_message")

    @patch("mm_toolbox.src.logging.logger.Logger._message_", new_callable=AsyncMock)
    async def test_info(self, mock_message):
        await self.logger.info("test_topic", "test_message")
        mock_message.assert_called_once_with("INFO", "TEST_TOPIC", "test_message")

    @patch("mm_toolbox.src.logging.logger.Logger._message_", new_callable=AsyncMock)
    async def test_success(self, mock_message):
        await self.logger.success("test_topic", "test_message")
        mock_message.assert_called_once_with("SUCCESS", "TEST_TOPIC", "test_message")

    @patch("mm_toolbox.src.logging.logger.Logger._message_", new_callable=AsyncMock)
    async def test_warning(self, mock_message):
        await self.logger.warning("test_topic", "test_message")
        mock_message.assert_called_once_with("WARNING", "TEST_TOPIC", "test_message")

    # @patch("mm_toolbox.src.logging.logger.DiscordClient", autospec=True)
    # @patch("mm_toolbox.src.logging.logger.TelegramClient", autospec=True)
    # @patch("mm_toolbox.src.logging.logger.time_iso8601", return_value="2024-08-23T12:00:00Z")
    # async def test_message(self, mock_time_iso8601, MockTelegramClient, MockDiscordClient):
    #     self.logger.discord_client = MockDiscordClient()
    #     self.logger.telegram_client = MockTelegramClient()

    #     await self.logger._message_("INFO", "test_topic", "test_message")

    #     expected_message = "2024-08-23T12:00:00Z | INFO | TEST_TOPIC | test_message"

    #     self.logger.discord_client.send.assert_called_once_with(expected_message, False)
    #     self.logger.telegram_client.send.assert_called_once_with(expected_message, False)
    #     self.assertIn(expected_message, self.logger.msgs)

    @patch("mm_toolbox.src.logging.logger.DiscordClient", autospec=True)
    @patch("mm_toolbox.src.logging.logger.TelegramClient", autospec=True)
    async def test_shutdown(self, MockTelegramClient, MockDiscordClient):
        self.logger.discord_client = MockDiscordClient()
        self.logger.telegram_client = MockTelegramClient()

        # Import asyncio and create an asyncio task
        self.logger.tasks = [asyncio.create_task(asyncio.sleep(0))]
        self.logger.msgs = ["log message"]

        with patch.object(
            self.logger, "_write_logs_to_file_", new_callable=AsyncMock
        ) as mock_write_logs:
            await self.logger.shutdown()

            self.logger.discord_client.shutdown.assert_called_once()
            self.logger.telegram_client.shutdown.assert_called_once()
            mock_write_logs.assert_called_once()

    async def test_shutdown_no_clients(self):
        with patch.object(
            self.logger, "_write_logs_to_file_", new_callable=AsyncMock
        ) as mock_write_logs:
            await self.logger.shutdown()
            mock_write_logs.assert_not_called()


if __name__ == "__main__":
    unittest.main()

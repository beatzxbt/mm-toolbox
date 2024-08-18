import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from mm_toolbox.src.logging import Logger 


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger = Logger(debug_mode=True)

    @patch("logger.time_iso8601", return_value="2024-08-17T12:00:00Z")
    def test_message_formatting_and_storage(self, mock_time):
        asyncio.run(self.logger.info("test_topic", "This is a test message"))

        expected_message = "2024-08-17T12:00:00Z | INFO | TEST_TOPIC | This is a test message"
        self.assertIn(expected_message, self.logger.msgs)
        self.assertEqual(len(self.logger.msgs), 1)

    @patch("aiofiles.open", new_callable=AsyncMock)
    def test_write_logs_to_file(self, mock_open):
        self.logger.msgs = ["Test log message"]
        asyncio.run(self.logger._write_logs_to_file_())

        mock_open.assert_called_once_with("logs.txt", "a")
        mock_open.return_value.writelines.assert_called_once_with(["Test log message\n"])
        self.assertEqual(self.logger.msgs, [])

    @patch("logger.DiscordClient", new_callable=MagicMock)
    @patch("logger.TelegramClient", new_callable=MagicMock)
    @patch.dict('os.environ', {'DISCORD_WEBHOOK': 'fake_webhook', 'TELEGRAM_BOT_TOKEN': 'fake_token', 'TELEGRAM_CHAT_ID': 'fake_chat_id'})
    @patch("logger.time_iso8601", return_value="2024-08-17T12:00:00Z")
    def test_integration_with_discord_and_telegram(self, mock_time, mock_telegram, mock_discord):
        logger = Logger(debug_mode=True)

        asyncio.run(logger.info("test_topic", "This is a test message"))

        expected_message = "2024-08-17T12:00:00Z | INFO | TEST_TOPIC | This is a test message"
        mock_discord.return_value.send.assert_called_once_with(expected_message, False)
        mock_telegram.return_value.send.assert_called_once_with(expected_message, False)

    @patch("logger.time_iso8601", return_value="2024-08-17T12:00:00Z")
    def test_debug_mode(self, mock_time):
        logger = Logger(debug_mode=False)
        asyncio.run(logger.debug("test_topic", "This should not be logged"))
        self.assertEqual(len(logger.msgs), 0)

        logger = Logger(debug_mode=True)
        asyncio.run(logger.debug("test_topic", "This should be logged"))
        expected_message = "2024-08-17T12:00:00Z | DEBUG | TEST_TOPIC | This should be logged"
        self.assertIn(expected_message, logger.msgs)

    @patch("logger.DiscordClient", new_callable=MagicMock)
    @patch("logger.TelegramClient", new_callable=MagicMock)
    @patch("aiofiles.open", new_callable=AsyncMock)
    def test_shutdown_procedure(self, mock_open, mock_telegram, mock_discord):
        logger = Logger(debug_mode=True)

        # Log some messages
        asyncio.run(logger.info("test_topic", "This is a test message"))
        asyncio.run(logger.error("test_topic", "This is an error message"))

        # Call shutdown
        asyncio.run(logger.shutdown())

        # Ensure clients were shut down
        mock_discord.return_value.shutdown.assert_called_once()
        mock_telegram.return_value.shutdown.assert_called_once()

        # Ensure all tasks were awaited
        self.assertEqual(len(logger.tasks), 0)

        # Ensure messages were written to file
        mock_open.assert_called_once_with("logs.txt", "a")
        mock_open.return_value.writelines.assert_called_once()

if __name__ == '__main__':
    unittest.main()

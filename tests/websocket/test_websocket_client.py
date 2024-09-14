import unittest
import asyncio
from unittest.mock import patch
from mm_toolbox import Logger, LoggerConfig

from mm_toolbox.websocket.client import SingleWsConnection


class TestSingleWsConnection(unittest.TestCase):
    def setUp(self):
        self.logger = Logger(LoggerConfig(base_level="INFO", stout=False))
        self.test_server_url = "wss://fstream.binance.com/ws/btcusdt@aggTrade"
        self.conn = SingleWsConnection(self.logger)

    def tearDown(self):
        self.conn.close()

    def test_initialization(self):
        self.assertFalse(self.conn.running)
        self.assertEqual(self.conn.seq_id, 0)
        self.assertEqual(self.conn.conn_id, 0)
        self.assertIsInstance(self.conn.queue, asyncio.Queue)

    async def test_start_connection(self):
        await self.conn.start(self.test_server_url)
        await asyncio.sleep(1.0)  # Spare time for response.

        self.assertEqual(self.conn.url, self.test_server_url)
        self.assertIsInstance(self.conn.on_connect, list)
        self.assertTrue(self.conn.running)
        self.assertGreater(self.conn.seq_id, 0)

    async def test_invalid_start_connection(self):
        await self.conn.start(self.test_server_url)
        await asyncio.sleep(1.0)

        # Attempt to start it again and check if it logs a warning
        with patch.object(self.logger, "warning") as mock_warning:
            await self.conn.start(self.test_server_url)
            mock_warning.assert_called_once_with(
                "Conn '0' already started, use restart to reconnect."
            )

    async def test_invalid_restart_connection(self):
        with patch.object(self.logger, "warning") as mock_warning:
            await self.conn.restart()
            mock_warning.assert_called_once_with("Conn '0' not started yet.")

    async def test_restart_connection(self):
        # Start and then restart connection
        await self.conn.start(self.test_server_url)
        await asyncio.sleep(1.0)  # Spare time for response.

        await self.conn.restart()

        self.assertEqual(self.conn.url, self.test_server_url)
        self.assertIsInstance(self.conn.on_connect, list)
        self.assertTrue(self.conn.running)
        self.assertGreater(self.conn.seq_id, 0)

    async def test_send_data(self):
        await self.conn.start(
            url=self.test_server_url,
            on_connect=[
                {
                    "method": "SUBSCRIBE",
                    "params": ["btcusdt@depth@100ms"],
                    "id": 1,
                }
            ],
        )
        await asyncio.sleep(1.0)  # Spare time for response.

        # Simulate receiving messages in the queue.
        await self.conn.queue.put((1, 123456789, {"id": 1}))

        # Now search the queue for subscription confirmation.
        found = False
        try:
            for _ in range(50):
                message = await asyncio.wait_for(self.conn.queue.get(), timeout=2.0)
                _, _, payload_data = message

                if isinstance(payload_data, dict) and payload_data.get("id") == 1:
                    found = True
                    break
        except asyncio.TimeoutError:
            self.fail("Timeout waiting for message with id=1.")

        self.assertTrue(found, "Subscription confirmation message with id=1 not found.")

    async def test_queue_full(self):
        self.conn.queue = asyncio.Queue(maxsize=1)
        await self.conn.queue.put((1, 123456789, {"id": 1}))

        with patch.object(self.logger, "warning") as mock_warning:
            # Queue is full, this should trigger a warning
            self.conn._process_frame(123456789, b'{"id": 2}')
            mock_warning.assert_called_once_with("Conn '0' queue full, skipping msg 2.")

    async def test_send_data_no_connection(self):
        # Ensure connection is not started
        self.conn._ws_client = None

        with patch.object(self.logger, "warning") as mock_warning:
            self.conn.send_data({"key": "value"})
            mock_warning.assert_called_once_with(
                "Conn '0' failed to send: Connection not started yet."
            )

    async def test_close_connection(self):
        # Start connection
        await self.conn.start(self.test_server_url)
        await asyncio.sleep(1.0)  # Spare time for response.

        self.conn.close()

        self.assertFalse(self.conn.running)


if __name__ == "__main__":
    unittest.main()

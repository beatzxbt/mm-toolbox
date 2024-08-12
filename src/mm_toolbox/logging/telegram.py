import orjson
import aiosonic
import asyncio

class TelegramClient:
    """
    A client for sending messages to a Telegram channel using a bot token and chat ID with buffering capabilities.
    """

    def __init__(self, buffer_size: int = 5) -> None:
        self.max_buffer_size = buffer_size
        self.buffer = []

        self.client = None
        self.bot_token = None
        self.chat_id = None
        self.data = {"text": ""}
        self.headers = {"Content-Type": "application/json"}

        self.tasks = []

    def start(self, bot_token: str, chat_id: str) -> None:
        """
        Initialize the Telegram client with the provided bot token and chat ID.

        Parameters
        ----------
        bot_token : str
            The bot token for the Telegram bot.

        chat_id : str
            The chat ID of the Telegram channel.

        Returns
        -------
        None
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.client = aiosonic.HTTPClient()

    async def send(self, content: str, flush_buffer: bool = False) -> None:
        """
        Send a message to a Telegram channel.

        Parameters
        ----------
        content : str
            The formatted message to send.

        flush_buffer : bool, optional
            Whether to flush the buffer and send all buffered messages immediately (default is False).

        Returns
        -------
        None
        """
        try:
            if not self.client or not self.bot_token or not self.chat_id:
                raise RuntimeError(
                    "Client not initialized, bot token, or chat ID not set."
                )

            self.data["text"] = content
            self.buffer.append(self.data.copy())

            if len(self.buffer) >= self.max_buffer_size or flush_buffer:
                tasks = []

                for message in self.buffer:
                    tasks.append(
                        asyncio.create_task(
                            self.client.post(
                                url=f"https://api.telegram.org/bot{self.bot_token}/sendMessage?chat_id={self.chat_id}",
                                data=orjson.dumps(message).decode(),
                                headers=self.headers,
                            )
                        )
                    )

                _ = await asyncio.gather(*tasks)
                tasks.clear()

                self.buffer.clear()

        except Exception as e:
            print(f"Failed to send message to Telegram: {e}")

    async def shutdown(self) -> None:
        """
        Close the async client, if existing, and ensure all tasks are complete.

        Returns
        -------
        None
        """
        if self.tasks:
            await asyncio.gather(*self.tasks)

        if self.client:
            await self.send("Shutting down logger.", flush_buffer=True)
            await asyncio.sleep(1.0)
            await self.client.connector.cleanup()
            await self.client.__aexit__(None, None, None)
            del self.client

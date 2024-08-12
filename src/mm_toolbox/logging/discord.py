import os
import orjson
import aiofiles
import aiosonic
import asyncio

class DiscordClient:
    """
    A client for sending messages to a Discord channel using a webhook URL with buffering capabilities.
    """

    def __init__(self, buffer_size: int = 5) -> None:
        self.max_buffer_size = buffer_size
        self.buffer = []

        self.client = None
        self.webhook = None
        self.data = {"content": ""}
        self.headers = {"Content-Type": "application/json"}

        self.tasks = []

    def start(self, webhook: str) -> None:
        """
        Initialize the Discord client with the provided webhook URL.

        Parameters
        ----------
        webhook : str
            The webhook URL for the Discord channel.

        Returns
        -------
        None
        """
        self.webhook = webhook
        self.client = aiosonic.HTTPClient()

    async def send(self, content: str, flush_buffer: bool = False) -> None:
        """
        Send a message to a Discord channel.

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
            if not self.client or not self.webhook:
                raise RuntimeError("Client not initialized or webhook URL not set.")

            self.data["content"] = content
            self.buffer.append(self.data.copy())

            if len(self.buffer) >= self.max_buffer_size or flush_buffer:
                tasks = []

                for message in self.buffer:
                    tasks.append(
                        asyncio.create_task(
                            self.client.post(
                                url=self.webhook,
                                data=orjson.dumps(message).decode(),
                                headers=self.headers,
                            )
                        )
                    )

                _ = await asyncio.gather(*tasks)

                self.buffer.clear()

        except Exception as e:
            print(f"Failed to send message to Discord: {e}")

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
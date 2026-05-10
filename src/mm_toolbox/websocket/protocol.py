"""Protocol definition for WebSocket client interfaces.

Unifies ``WsPool`` and ``WsSingle`` behind a common interface so that
consumer code can work with either a single connection or a pool without
changing call sites.

Methods include lifecycle management (``__aenter__`` / ``__aexit__``),
message iteration (``__aiter__`` / ``__anext__``), sending, querying
state/latency, and updating connect-time messages.
"""

from typing import Any, Protocol, Self, runtime_checkable

from mm_toolbox.websocket.connection import ConnectionState, WsConnectionConfig


@runtime_checkable
class WebsocketClient(Protocol):
    """Protocol unifying WebSocket client implementations.

        Both ``WsPool`` and ``WsSingle`` satisfy this interface, enabling
    generic consumers that do not need to know which variant is in use.
    """

    async def __aenter__(self) -> Self:
        """Enter the async context and open the connection(s)."""
        ...

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the async context and close the connection(s)."""
        ...

    def __aiter__(self) -> Self:
        """Return an async iterator over incoming messages."""
        ...

    async def __anext__(self) -> bytes:
        """Return the next message from the iterator."""
        ...

    def send_data(self, msg: bytes, only_fastest: bool = True) -> None:
        """Send a payload through the connection(s)."""
        ...

    def close(self) -> None:
        """Close all underlying connections."""
        ...

    def get_state(self) -> ConnectionState:
        """Return the current aggregate connection state."""
        ...

    def get_config(self) -> WsConnectionConfig:
        """Return the base connection configuration."""
        ...

    def get_connection_count(self) -> int:
        """Return the number of active connections."""
        ...

    def get_latency_ms(self) -> float:
        """Return the best available latency in milliseconds."""
        ...

    def get_seq_id(self) -> int:
        """Return the highest sequence ID seen."""
        ...

    def set_on_connect(self, on_connect: list[bytes]) -> None:
        """Update the messages sent when a connection opens."""
        ...

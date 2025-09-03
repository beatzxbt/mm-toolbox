import asyncio
import time

import msgspec

from mm_toolbox.websocket.connection import WsConnectionConfig
from mm_toolbox.websocket.single import WsSingle


class BinanceDataFeedExample:
    msg_count: int = 0

    def __init__(self, runtime_s: int = 10):
        self.runtime_s = runtime_s
        self.start_time = time.time()
        self.end_time = self.start_time + self.runtime_s

    async def book_ticker_stream(self, symbol: str):
        config = WsConnectionConfig.default(
            wss_url=f"wss://fstream.binance.com/ws/{symbol.lower()}@bookTicker",
        )

        class BinanceBookTickerMsg(msgspec.Struct):
            event_type: str = msgspec.field(name="e")
            event_time: int = msgspec.field(name="E")
            update_id: int = msgspec.field(name="u")
            transaction_time: int = msgspec.field(name="T")
            symbol: str = msgspec.field(name="s")
            best_bid_price: float = msgspec.field(name="b")
            best_bid_qty: float = msgspec.field(name="B")
            best_ask_price: float = msgspec.field(name="a")
            best_ask_qty: float = msgspec.field(name="A")

        decoder = msgspec.json.Decoder(type=BinanceBookTickerMsg)

        async with WsSingle(config) as ws:
            async for msg in ws:
                print(decoder.decode(msg))
                self.msg_count += 1

                if time.time() > self.end_time:
                    return

    async def full_orderbook_stream(self, symbol: str):
        config = WsConnectionConfig.default(
            wss_url=f"wss://fstream.binance.com/ws/{symbol.lower()}@depth",
        )

        # orderbook = Orderbook(max_num_levels=10)
        async with WsSingle(config) as ws:
            async for msg in ws:
                print(msg)
                self.msg_count += 1

                if time.time() > self.end_time:
                    return

    async def raw_trade_stream(self, symbol: str):
        config = WsConnectionConfig.default(
            wss_url=f"wss://fstream.binance.com/ws/{symbol.lower()}@trade",
        )

        class BinanceRawTradeMsg(msgspec.Struct):
            event_type: str = msgspec.field(name="e")
            event_time: int = msgspec.field(name="E")
            transaction_time: int = msgspec.field(name="T")
            symbol: str = msgspec.field(name="s")
            trade_id: int = msgspec.field(name="t")
            price: float = msgspec.field(name="p")
            size: float = msgspec.field(name="q")
            order_type: str = msgspec.field(name="X")
            buyer_is_maker: bool = msgspec.field(name="m")

        decoder = msgspec.json.Decoder(type=BinanceRawTradeMsg)

        async with WsSingle(config) as ws:
            async for msg in ws:
                decoded_msg = decoder.decode(msg)

                print(decoded_msg)

                self.msg_count += 1

                if time.time() > self.end_time:
                    return

    async def run(self, symbol: str):
        tasks = [
            self.book_ticker_stream(symbol),
            self.raw_trade_stream(symbol),
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    SYMBOL = "btcusdt"
    feed = BinanceDataFeedExample(runtime_s=10)
    asyncio.run(feed.run(SYMBOL))

"""Small, deliberately unoptimized orderbook contract, independent of production.

Only integer ticks/lots and dictionaries are stored. Decimal input conversion,
whole-batch replacement and sorting are intentionally unrelated to the ladder's
ring buffers, insertion paths, caches, or production precision helpers.
"""

from decimal import Decimal, ROUND_FLOOR
from math import inf
from fractions import Fraction

Row = tuple[float, float, int]


class ReferenceBook:
    def __init__(self, tick_size=1.0, lot_size=1.0, capacity=None):
        self.tick = Decimal(str(tick_size))
        self.lot = Decimal(str(lot_size))
        self.capacity = capacity
        self.clear()

    @staticmethod
    def units(value, quantum):
        value = Decimal(str(value))
        if not value.is_finite() or value < 0:
            raise ValueError("price and size must be finite and nonnegative")
        units = int((value / quantum).to_integral_value(rounding=ROUND_FLOOR))
        if units >= 2**64:
            raise ValueError("normalized price/size must fit unsigned 64-bit units")
        return units

    def normalize(self, rows):
        result = {}
        for price, size, orders in rows:
            tick = self.units(price, self.tick)
            lots = self.units(size, self.lot)
            if not isinstance(orders, int) or not 0 <= orders < 2**64:
                raise ValueError("norders must be an unsigned 64-bit integer")
            result[tick] = (lots, orders)
        return result

    def clear(self):
        self.asks, self.bids = {}, {}
        self.initialized = False

    def _finish(self, asks, bids):
        self.asks = {p: asks[p] for p in sorted(asks)[: self.capacity]}
        self.bids = {p: bids[p] for p in sorted(bids, reverse=True)[: self.capacity]}
        self.initialized = True

    def snapshot(self, asks, bids):
        asks, bids = self.normalize(asks), self.normalize(bids)
        asks = {p: q for p, q in asks.items() if q[0]}
        bids = {p: q for p, q in bids.items() if q[0]}
        if asks and bids and max(bids) >= min(asks):
            raise ValueError("locked or crossed snapshot")
        self._finish(asks, bids)

    def deltas(self, asks, bids):
        ask_updates, bid_updates = self.normalize(asks), self.normalize(bids)
        # A one-sided feed cannot authoritatively erase the entire opposite side.
        if self.asks and self.bids:
            live_asks = [p for p, q in ask_updates.items() if q[0]]
            live_bids = [p for p, q in bid_updates.items() if q[0]]
            if live_asks and min(live_asks) <= min(self.bids) and not live_bids:
                return
            if live_bids and max(live_bids) >= max(self.asks) and not live_asks:
                return
        new_asks, new_bids = self.asks.copy(), self.bids.copy()
        for side, updates in ((new_asks, ask_updates), (new_bids, bid_updates)):
            for price, quantity in updates.items():
                if quantity[0]:
                    side[price] = quantity
                else:
                    side.pop(price, None)
        # Completed ask batches have priority when both final sides cross.
        if new_asks and ask_updates:
            new_bids = {p: q for p, q in new_bids.items() if p < min(new_asks)}
        if new_bids and bid_updates:
            new_asks = {p: q for p, q in new_asks.items() if p > max(new_bids)}
        self._finish(new_asks, new_bids)

    def bbo(self, ask, bid):
        ap, (aq, ao) = next(iter(self.normalize([ask]).items()))
        bp, (bq, bo) = next(iter(self.normalize([bid]).items()))
        if bp >= ap:
            raise ValueError("locked or crossed BBO")
        if (not aq and (not self.asks or ap != min(self.asks))) or (
            not bq and (not self.bids or bp != max(self.bids))
        ):
            raise ValueError("zero BBO must identify the current best")
        asks = (
            {p: q for p, q in self.asks.items() if p >= ap} if aq else self.asks.copy()
        )
        bids = (
            {p: q for p, q in self.bids.items() if p <= bp} if bq else self.bids.copy()
        )
        if aq:
            asks[ap] = (aq, ao)
        else:
            asks.pop(ap)
        if bq:
            bids[bp] = (bq, bo)
        else:
            bids.pop(bp)
        self._finish(asks, bids)

    def integer_rows(self, side):
        values = getattr(self, side)
        return [(p, *values[p]) for p in sorted(values, reverse=side == "bids")]

    def rows(self, side):
        return [
            (float(p * self.tick), float(q * self.lot), n)
            for p, q, n in self.integer_rows(side)
        ]

    def query(self, name, *args):
        if not self.asks or not self.bids:
            raise ValueError("both sides required")
        ask, bid = min(self.asks), max(self.bids)
        if name == "get_mid_price":
            return float(((ask + bid) // 2) * self.tick)
        if name == "get_bbo_spread":
            return float((ask - bid) * self.tick)
        if name == "get_wmid_price":
            aq, bq = self.asks[ask][0], self.bids[bid][0]
            return float(((ask * aq + bid * bq) // (aq + bq)) * self.tick)
        if name == "does_bbo_price_change":
            bp, ap = (self.units(p, self.tick) for p in args)
            return bp != bid or ap != ask
        if name == "does_bbo_cross":
            bp, ap = (self.units(p, self.tick) for p in args)
            return bp > ask or ap < bid
        if name == "get_size_for_price_impact_bps":
            bps, buy, base = args
            if bps <= 0:
                return 0.0
            anchor = ask if buy else bid
            boundary = Fraction(anchor) * (
                1 + (1 if buy else -1) * Fraction(str(bps)) / 10000
            )
            selected = [
                (p, q)
                for p, q, _ in self.integer_rows("asks" if buy else "bids")
                if (p <= boundary if buy else p >= boundary)
            ]
            return float(
                sum(
                    (
                        Decimal(q) * self.lot * (1 if base else p * self.tick)
                        for p, q in selected
                    ),
                    Decimal(0),
                )
            )
        if name == "get_volume_weighted_mid_price":
            size, base = args
            if size <= 0:
                return self.query("get_mid_price")
            quantity = Fraction(str(size))
            if not base:
                quantity /= Fraction((ask + bid) // 2) * Fraction(self.tick)
            notionals = []
            for side in ("asks", "bids"):
                remaining, notional = quantity, Fraction(0)
                for p, q, _ in self.integer_rows(side):
                    take = min(remaining, q * Fraction(self.lot))
                    notional += take * p * Fraction(self.tick)
                    remaining -= take
                if remaining > 0:
                    return inf
                notionals.append(notional)
            return float(sum(notionals) / (2 * quantity))
        if name == "get_price_impact":
            size, buy, base = args
            if size <= 0:
                return 0.0
            anchor = ask if buy else bid
            remaining = Fraction(str(size))
            if not base:
                remaining /= anchor * Fraction(self.tick)
            last = anchor
            for p, q, _ in self.integer_rows("asks" if buy else "bids"):
                take = min(remaining, q * Fraction(self.lot))
                if take:
                    last = p
                remaining -= take
                if remaining <= 0:
                    return float(abs(last - anchor) * self.tick)
            return inf
        raise KeyError(name)

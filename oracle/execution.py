"""Stage 9 - Execution with nautilus_trader.

`RebalanceStrategy` trades a schedule of target weights. The same class runs
in nautilus' BacktestEngine (replay, used here to cross-check fills) and in a
live TradingNode wired to a paper account (e.g. Interactive Brokers paper,
see nautilus docs: integrations/ib). Paper first, always.
"""
from __future__ import annotations

from decimal import Decimal

import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

VENUE = "XNAS"


class RebalanceConfig(StrategyConfig, frozen=True):
    symbols: tuple[str, ...]
    schedule: dict          # {"YYYY-MM-DD": {symbol: weight}}
    cash_buffer: float = 0.02


class RebalanceStrategy(Strategy):
    def __init__(self, config: RebalanceConfig):
        super().__init__(config)
        self.ids = {s: InstrumentId.from_str(f"{s}.{VENUE}") for s in config.symbols}
        self.last: dict[str, float] = {}
        self.pending = sorted(config.schedule)
        self.day = None

    def on_start(self):
        for s in self.ids:
            self.subscribe_bars(BarType.from_str(f"{s}.{VENUE}-1-DAY-LAST-EXTERNAL"))

    def on_bar(self, bar: Bar):
        sym = bar.bar_type.instrument_id.symbol.value
        self.last[sym] = float(bar.close)
        day = pd.Timestamp(bar.ts_event, tz="UTC").strftime("%Y-%m-%d")
        self.day = day
        # rebalance once every symbol has printed today's bar
        if self.pending and day >= self.pending[0] and len(self.last) == len(self.ids) and sym == self._last_symbol():
            target = self.config.schedule[self.pending.pop(0)]
            while self.pending and self.pending[0] <= day:
                target = self.config.schedule[self.pending.pop(0)]
            self._rebalance(target)

    def _last_symbol(self):
        return list(self.ids)[-1]

    def _equity(self) -> float:
        acct = self.portfolio.account(self.ids[next(iter(self.ids))].venue)
        cash = float(acct.balance_total().as_double())
        pos = sum(float(self.portfolio.net_position(i)) * self.last[s] for s, i in self.ids.items())
        return cash + pos

    def _rebalance(self, target: dict):
        eq = self._equity() * (1 - self.config.cash_buffer)
        deltas = {}
        for s, iid in self.ids.items():
            want = int(eq * target.get(s, 0.0) / self.last[s])
            have = int(float(self.portfolio.net_position(iid)))
            if want != have:
                deltas[s] = want - have
        for s, d in sorted(deltas.items(), key=lambda kv: kv[1]):   # sells first, free up cash
            order = self.order_factory.market(
                instrument_id=self.ids[s],
                order_side=OrderSide.BUY if d > 0 else OrderSide.SELL,
                quantity=Quantity.from_int(abs(d)),
            )
            self.submit_order(order)


def replay(prices: pd.DataFrame, weights: pd.DataFrame, start_cash: float = 1_000_000.0,
           fee_rate: float = 0.001) -> pd.Series:
    """Replay a weight schedule through nautilus' BacktestEngine; returns daily equity."""
    from nautilus_trader.backtest.config import BacktestEngineConfig
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import Symbol, Venue
    from nautilus_trader.model.instruments import Equity
    from nautilus_trader.model.objects import Money, Price

    engine = BacktestEngine(BacktestEngineConfig(trader_id="ORACLE-001", logging=LoggingConfig(log_level="ERROR")))
    engine.add_venue(Venue(VENUE), OmsType.NETTING, AccountType.CASH, [Money(start_cash, USD)], base_currency=USD)
    symbols = tuple(prices.columns)
    for s in symbols:
        inst = Equity(InstrumentId.from_str(f"{s}.{VENUE}"), Symbol(s), USD, 2, Price.from_str("0.01"),
                      Quantity.from_int(1), 0, 0, maker_fee=Decimal(str(fee_rate)), taker_fee=Decimal(str(fee_rate)))
        engine.add_instrument(inst)
        bar_type = BarType.from_str(f"{s}.{VENUE}-1-DAY-LAST-EXTERNAL")
        vol = Quantity.from_int(1_000_000_000)
        bars = []
        for ts, px in prices[s].items():
            p = Price(round(float(px), 2), 2)
            ns = (pd.Timestamp(ts).tz_localize("UTC") + pd.Timedelta(hours=21)).value   # US close
            bars.append(Bar(bar_type, p, p, p, p, vol, ns, ns))
        engine.add_data(bars)

    sched = {d.strftime("%Y-%m-%d"): {s: float(w) for s, w in row.items() if w > 0} for d, row in weights.iterrows()}
    strat = RebalanceStrategy(RebalanceConfig(symbols=symbols, schedule=sched))
    engine.add_strategy(strat)
    equity = {}
    _on_bar = strat.on_bar

    def record(bar):
        _on_bar(bar)
        if bar.bar_type.instrument_id.symbol.value == symbols[-1]:
            equity[pd.Timestamp(bar.ts_event, tz="UTC").normalize()] = strat._equity()
    strat.on_bar = record
    engine.run()
    engine.dispose()
    s = pd.Series(equity).sort_index()
    s.index = s.index.tz_localize(None)
    return s

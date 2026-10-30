# Trading Oracle

A 10-stage quant research pipeline that wires together the open-source stack
(OpenBB, Chronos, qlib/RD-Agent, TradingAgents, QuantLib, gs-quant, skfolio,
vectorbt, nautilus_trader). It is built to answer one question honestly:
**is there an edge, or did the sweep just find luck?**

```
data ─► forecast ─► factors ─► committee ─► risk ─► portfolio ─► sweep ─► holdout ─► paper
OpenBB   Chronos    Alpha158/   bull/bear/   QuantLib  skfolio      numpy +   vectorbt +  nautilus
yfinance EWMA       RD-Agent    trader/risk  gs-quant  (min CVaR,   10k vars  nautilus    RebalanceStrategy
                                (+TradingAgents)       maxDD cap)            replay
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                                    # 8 tests, ~10 s
python -m oracle.pipeline --synthetic --variants 300   # smoke test, no network needed
python -m oracle.pipeline --variants 10000             # real data: 30 large caps, 5 years
python -m oracle.pipeline --sp500 --variants 10000     # full S&P 500 (see caveats)
```

Output goes to `runs/latest/`: `REPORT.md`, `summary.json`, `sweep.csv`
(every variant, train *and* test stats), `factor_ic_train.csv`,
`target_weights.csv`, `hedges.csv`, `risk.csv`.

## What each stage actually does

| Stage | Library | Used for | Fallback when unavailable |
|---|---|---|---|
| Data | OpenBB → yfinance | Adjusted closes, FRED macro, CBOE option chains | yfinance; `--synthetic` only on request |
| Forecast | Chronos-Bolt | Zero-shot 20-day median & 10th-pct log-return, per rebalance date, only past data | EWMA drift ± vol |
| Factors | qlib Alpha158 ideas, RD-Agent | 7 built-in factors; `--rdagent-factors file.parquet` adds mined ones; signs set by **train-window** IC | — |
| Committee | (TradingAgents) | Bull = forecast+factors, bear = downside quantile & vol; trader picks top-k; risk team vetoes | Deterministic committee (always used in backtests) |
| Risk | QuantLib, gs-quant | BSM protective-put pricing + Greeks per holding; vol / max DD / beta via `gs_quant.timeseries` | Realised vol as IV proxy (flagged); pandas maths |
| Portfolio | skfolio | `MeanRisk` min-CVaR with `max_max_drawdown` and per-name caps | Inverse-vol |
| Testing | numpy, vectorbt | Thousands of variants with fees + slippage, 1-day execution lag; vectorbt re-runs the winner | — |
| Execution | nautilus_trader | `RebalanceStrategy` replayed in `BacktestEngine`; same class plugs into a live paper `TradingNode` | — |

## Rules that keep it honest

* **Holdout is sacred.** The last 30% of history is never used for factor
  signs, parameter choice, or "survives fees" filtering. The chosen variant
  is evaluated on it once.
* **Deflated Sharpe Ratio.** Testing 10,000 versions and keeping the best
  guarantees an impressive in-sample Sharpe even on random data. The report
  gives the Bailey and López de Prado probability that the winner beats the
  best result you'd expect from luck, given the number and spread of trials.
  Below 0.95 the verdict is **NO EDGE YET**, whatever the equity curve looks like.
* **Benchmarks, not zero.** The strategy is compared with SPY buy-and-hold,
  daily-rebalanced equal weight, and the *median* variant on the holdout.
* **Three engines must agree.** The numpy sweep, vectorbt and nautilus replay
  of the chosen variant are reported side by side. On the synthetic smoke test
  they agree to within 0.1 pp of total return. nautilus differs slightly
  because of whole shares and a 2% cash buffer.
* **No look-ahead.** Signals use data through close *t*, and orders fill at
  close *t+1*. Tests shock future prices and assert past factors and
  forecasts don't move.

## Caveats you should know before trusting any output

* **LLM agents cannot be backtested honestly.** TradingAgents and RD-Agent
  use models trained on text written after most of your backtest window. That
  is look-ahead you can't remove. TradingAgents is therefore only used
  (`--tradingagents`) to review *today's* shortlist, never inside the sweep.
  RD-Agent factors are imported as data. Judge them only on the holdout, and
  prefer factors mined on data that ends before the holdout starts.
* **`--sp500` uses today's members**, which builds in survivorship bias:
  companies that fell out of the index are missing. That flatters every
  long-only result. Use point-in-time membership for anything serious.
* **gs-quant**: only the open `gs_quant.timeseries` module is used. Pricing
  and risk models need Goldman Sachs Marquee credentials.
* **qlib** has no wheel for Python 3.13. Run RD-Agent and qlib in their own
  Python 3.10–3.12 + Docker environment, and export factors to parquet with
  a `(datetime, instrument)` index.
* **Chronos** downloads weights from HuggingFace (needs `torch`). Without
  network or GPU, use `--forecast ewma`.
* **Two weeks of paper trading proves almost nothing statistically.** Its
  value is catching execution bugs, not confirming edge. Use the holdout
  and DSR for that.

## Paper trading

`oracle/execution.py::RebalanceStrategy` takes a `{date: {symbol: weight}}`
schedule. To paper-trade, build a nautilus `TradingNode` with your broker's
paper adapter (e.g. Interactive Brokers paper account), load
`runs/latest/target_weights.csv` as the schedule, and add the strategy. Use
the same code that ran in the replay.

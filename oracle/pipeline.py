"""End-to-end run: python -m oracle.pipeline [--synthetic] [--variants N] ..."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest, committee, data, factors, forecast, portfolio, risk
from .config import Config

log = logging.getLogger("oracle")


def run(cfg: Config) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = data.load(cfg)
    prices, bench = md.prices, md.benchmark
    log.info("data: %s, %d tickers x %d days", md.source, prices.shape[1], prices.shape[0])

    warmup = 252
    split = prices.index[int(len(prices) * (1 - cfg.holdout_frac))]
    grid = prices.index[warmup::5]                       # finest rebalance grid: weekly

    fc = forecast.forecast(prices, grid, cfg)
    fz = factors.builtin(prices, bench)
    if cfg.rdagent_factors:
        fz.update(factors.load_rdagent(cfg.rdagent_factors, prices))
    ic = factors.ic_report(fz, prices, cfg.horizon, split)   # train window only
    comp = factors.composite(fz, ic)
    sig = committee.build_signals(prices, fc, comp, grid)

    table, daily = backtest.sweep(prices, grid, sig, cfg, split)
    rets = prices.pct_change().fillna(0)
    bench_r = bench.pct_change().fillna(0)
    ew_r = rets.mean(axis=1)                              # daily-rebalanced equal weight
    live = rets.index >= grid[0]
    train, test = live & (rets.index < split), rets.index >= split
    pick = backtest.select(table, daily, split, backtest.stats(bench_r[train])["sharpe"])
    best = table.loc[pick["best_id"]]
    params = committee.Params(**{k: best[k] for k in committee.Params.__dataclass_fields__})
    params = committee.Params(float(params.alpha), int(params.top_k), int(params.rebalance), float(params.veto_vol_pct),
                              float(params.veto_q10), None if pd.isna(params.target_vol) else float(params.target_vol))

    holdout = pd.DataFrame({
        "strategy": backtest.stats(daily.loc[test, pick["best_id"]]),
        "buy_and_hold_" + cfg.benchmark: backtest.stats(bench_r[test]),
        "equal_weight_universe": backtest.stats(ew_r[test]),
        "median_variant": backtest.stats(daily.loc[test].median(axis=1)),
    }).T

    # independent engine cross-checks of the chosen variant on the holdout
    base = int(np.median(np.diff(rets.index.get_indexer(grid))))
    sel = np.arange(0, len(grid), max(1, round(params.rebalance / base)))
    sub = committee.Signals(sig.fc_z[sel], sig.factor_z[sel], sig.q10[sel], sig.vol_pct[sel], sig.vol[sel],
                            [sig.cov[s] for s in sel])
    W = pd.DataFrame(committee.decide(sub, params, cfg.max_weight), index=grid[sel], columns=prices.columns)
    W.index = prices.index[prices.index.get_indexer(W.index) + 1].where(
        prices.index.get_indexer(W.index) + 1 < len(prices), prices.index[-1])   # trade next day, like the sweep
    W_test = W[W.index >= split]
    p_test = prices[prices.index >= split]
    r_np = backtest.simulate(p_test.pct_change().fillna(0).values, p_test.index.get_indexer(W_test.index),
                             W_test.values, cfg.fees + cfg.slippage)
    checks = {"numpy_same_window": {"total_return": float(np.prod(1 + r_np) - 1)}}
    try:
        checks["vectorbt"] = backtest.vectorbt_check(p_test, W_test, cfg)
    except Exception as e:  # noqa: BLE001
        checks["vectorbt"] = {"error": repr(e)}
    try:
        from .execution import replay

        eq = replay(p_test, W_test, fee_rate=cfg.fees + cfg.slippage)
        checks["nautilus_replay"] = {**backtest.stats(eq.pct_change()), "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1)}
    except Exception as e:  # noqa: BLE001
        checks["nautilus_replay"] = {"error": repr(e)}

    # today's book: committee shortlist -> TradingAgents (optional) -> skfolio -> risk
    today = W.iloc[-1]
    shortlist = list(today[today > 0].index)
    review = committee.tradingagents_review(shortlist, str(prices.index[-1].date())) if cfg.use_tradingagents else {}
    shortlist = [t for t in shortlist if review.get(t, "BUY").find("SELL") < 0]
    weights, method = portfolio.optimise(rets[shortlist].iloc[-504:], cfg.max_drawdown, cfg.max_weight)
    weights = weights * today[shortlist].sum() / weights.sum()     # keep the variant's cash level
    hedges = risk.hedge_table(prices, weights, cfg.horizon, chain_fn=None if cfg.synthetic else data.option_chain)
    rr = risk.risk_report(prices, bench, weights / weights.sum())

    summary = {
        "data_source": md.source, "forecast_backend": fc.backend, "as_of": str(prices.index[-1].date()),
        "train": [str(grid[0].date()), str(split.date())], "holdout": [str(split.date()), str(prices.index[-1].date())],
        "selection": pick, "chosen_params": asdict(params), "holdout_stats": holdout.round(4).to_dict("index"),
        "engine_checks": checks, "portfolio_method": method, "tradingagents_review": review,
        "risk_engine": rr.attrs.get("engine"), "weights": weights.round(4).to_dict(),
        "verdict": verdict(holdout, pick, cfg.benchmark),
    }
    table.to_csv(out / "sweep.csv")
    ic.to_csv(out / "factor_ic_train.csv")
    hedges.to_csv(out / "hedges.csv")
    rr.to_csv(out / "risk.csv")
    W.to_csv(out / "target_weights.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "REPORT.md").write_text(report(summary, ic, hedges, rr))
    return summary


def verdict(holdout: pd.DataFrame, pick: dict, bench: str) -> str:
    s, b = holdout.loc["strategy"], holdout.loc[f"buy_and_hold_{bench}"]
    beats = s.sharpe > b.sharpe and s.cagr > b.cagr
    if pick["deflated_sharpe_prob"] < 0.95:
        return (f"NO EDGE YET: deflated Sharpe probability {pick['deflated_sharpe_prob']:.2f} < 0.95 across "
                f"{pick['n_trials']} trials; the in-sample winner is indistinguishable from luck. "
                + ("It did beat buy-and-hold on the holdout, which is encouraging but not proof." if beats else
                   "It also failed to beat buy-and-hold on the holdout."))
    if not beats:
        return "NO EDGE YET: survives the multiple-testing check in-sample but loses to buy-and-hold out of sample."
    return "CANDIDATE: beats buy-and-hold out of sample and survives deflation. Paper trade it next; don't fund it yet."


def report(s: dict, ic, hedges, rr) -> str:
    h = pd.DataFrame(s["holdout_stats"]).T
    lines = [
        "# Trading Oracle run", "",
        f"**Data:** {s['data_source']}  ·  **Forecast:** {s['forecast_backend']}  ·  **As of:** {s['as_of']}",
        f"**Train:** {s['train'][0]} → {s['train'][1]}  ·  **Holdout:** {s['holdout'][0]} → {s['holdout'][1]}", "",
        f"## Verdict\n\n{s['verdict']}", "",
        f"## Holdout (never seen during selection)\n\n{h.to_markdown(floatfmt='.3f')}", "",
        f"Selected variant {s['selection']['best_id']} of {s['selection']['n_trials']} "
        f"({s['selection']['n_survivors']} survived fees and beat the benchmark in-sample). "
        f"Deflated Sharpe probability: **{s['selection']['deflated_sharpe_prob']:.3f}**.", "",
        f"Params: `{s['chosen_params']}`", "",
        f"## Engine cross-checks (holdout)\n\n```json\n{json.dumps(s['engine_checks'], indent=2)}\n```", "",
        f"## Factor IC (train window)\n\n{ic.to_markdown(floatfmt='.3f')}", "",
        f"## Today's book ({s['portfolio_method']})\n\n{hedges.to_markdown(floatfmt='.4f')}", "",
        f"## Risk ({s['risk_engine']})\n\n{rr.to_markdown(floatfmt='.3f')}", "",
    ]
    if s["data_source"] == "SYNTHETIC":
        lines.insert(2, "> **SYNTHETIC DATA. This run only checks that the code works. None of these numbers say anything about markets.**\n")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true", help="random-walk data; smoke test only")
    ap.add_argument("--variants", type=int, default=Config.n_variants)
    ap.add_argument("--years", type=int, default=Config.years)
    ap.add_argument("--tickers", help="comma-separated universe (default: 30 large caps)")
    ap.add_argument("--sp500", action="store_true", help="current S&P 500 members (survivorship-biased!)")
    ap.add_argument("--forecast", default="auto", choices=["auto", "chronos", "ewma"])
    ap.add_argument("--rdagent-factors", help="parquet/csv of RD-Agent factors (datetime, instrument)")
    ap.add_argument("--tradingagents", action="store_true", help="LLM review of today's shortlist")
    ap.add_argument("--out", default=Config.out_dir)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = Config(synthetic=a.synthetic, n_variants=a.variants, years=a.years, forecast_backend=a.forecast,
                 rdagent_factors=a.rdagent_factors, use_tradingagents=a.tradingagents, out_dir=a.out)
    if a.tickers:
        cfg.universe = [t.strip().upper() for t in a.tickers.split(",")]
    elif a.sp500:
        cfg.universe = sp500_members()
    s = run(cfg)
    print(json.dumps({k: s[k] for k in ("data_source", "forecast_backend", "selection", "verdict")}, indent=2))
    print(f"\nfull report: {Path(cfg.out_dir) / 'REPORT.md'}")


def sp500_members() -> list[str]:
    t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return [s.replace(".", "-") for s in t["Symbol"]]


if __name__ == "__main__":
    main()

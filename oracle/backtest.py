"""Stage 8 - Testing: fast numpy sweep over thousands of variants, vectorbt
cross-check of finalists, and the statistics that keep the sweep honest.

Running 10,000 versions and keeping the best is a multiple-testing machine:
the best in-sample Sharpe is inflated by luck. So we (1) select on a train
window only, (2) report the Deflated Sharpe Ratio for the number of trials,
and (3) evaluate the single chosen variant once on an untouched holdout.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict

import numpy as np
import pandas as pd

from .committee import Params, Signals, decide


def param_grid(n_assets: int, n_variants: int, seed: int) -> list[Params]:
    ks = sorted({k for k in (3, 5, 8, 10, 15, 20, 30, 50) if k <= n_assets} | {min(n_assets, 10)})
    grid = [
        Params(a, k, rb, vv, vq, tv)
        for a, k, rb, vv, vq, tv in itertools.product(
            (0.0, 0.25, 0.5, 0.75, 1.0),
            ks,
            (5, 10, 21, 42, 63),
            (0.7, 0.8, 0.9, 1.0),
            (-0.10, -0.15, -0.25, -np.inf),
            (0.10, 0.15, 0.20, None),
        )
    ]
    if len(grid) <= n_variants:
        return grid
    rng = np.random.default_rng(seed)
    return [grid[i] for i in sorted(rng.choice(len(grid), n_variants, replace=False))]


def simulate(returns: np.ndarray, reb_idx: np.ndarray, W: np.ndarray, cost: float) -> np.ndarray:
    """Daily net returns with drifting weights.

    returns: (D, N) daily simple returns. reb_idx: (T,) day index at which
    target W[t] is traded (at that day's close). cost per unit turnover.
    """
    D, N = returns.shape
    out = np.zeros(D)
    w = np.zeros(N)
    j = 0
    for d in range(D):
        if j < len(reb_idx) and d == reb_idx[j]:
            tgt = W[j]
            out[d] -= cost * np.abs(tgt - w).sum()
            w = tgt.copy()
            j += 1
        if d + 1 < D:
            r = returns[d + 1]
            g = w * (1 + r)
            port = 1 - w.sum() + g.sum()           # cash earns 0
            out[d + 1] += g.sum() - w.sum()
            w = g / port
    return out


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return {"cagr": 0.0, "vol": 0.0, "sharpe": 0.0, "max_dd": 0.0, "n_days": len(r)}
    eq = (1 + r).cumprod()
    years = len(r) / 252
    return {
        "cagr": eq.iloc[-1] ** (1 / years) - 1,
        "vol": r.std() * math.sqrt(252),
        "sharpe": r.mean() / r.std() * math.sqrt(252),
        "max_dd": (eq / eq.cummax() - 1).min(),
        "n_days": len(r),
    }


def deflated_sharpe(sr_daily: float, n_obs: int, sr_trials_daily: np.ndarray, skew: float, kurt: float) -> float:
    """Bailey & Lopez de Prado (2014). Probability the best SR beats luck."""
    from scipy.stats import norm

    n = max(len(sr_trials_daily), 2)
    var = np.var(sr_trials_daily, ddof=1)
    emc = 0.5772156649
    sr0 = math.sqrt(var) * ((1 - emc) * norm.ppf(1 - 1 / n) + emc * norm.ppf(1 - 1 / (n * math.e)))
    denom = math.sqrt(max(1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily**2, 1e-12))
    return float(norm.cdf((sr_daily - sr0) * math.sqrt(n_obs - 1) / denom))


def sweep(prices: pd.DataFrame, sig_dates: pd.DatetimeIndex, sig: Signals, cfg, split: pd.Timestamp):
    """Run every variant over the full period; rank on train window only."""
    rets = prices.pct_change().fillna(0)
    day_pos = rets.index.get_indexer(sig_dates)
    base_step = int(np.median(np.diff(day_pos)))
    cost = cfg.fees + cfg.slippage
    params = param_grid(prices.shape[1], cfg.n_variants, cfg.seed)
    train = rets.index < split
    rows, daily = [], {}
    for i, p in enumerate(params):
        stride = max(1, round(p.rebalance / base_step))
        sel = np.arange(0, len(sig_dates), stride)
        sub = Signals(sig.fc_z[sel], sig.factor_z[sel], sig.q10[sel], sig.vol_pct[sel], sig.vol[sel], [sig.cov[s] for s in sel])
        W = decide(sub, p, cfg.max_weight)
        # decided on close of day t, traded on close of t+1 (no same-bar look-ahead)
        r = pd.Series(simulate(rets.values, day_pos[sel] + 1, W, cost), index=rets.index)
        daily[i] = r
        st_tr, st_te = stats(r[train]), stats(r[~train])
        rows.append({"id": i, **asdict(p), **{f"train_{k}": v for k, v in st_tr.items()},
                     **{f"test_{k}": v for k, v in st_te.items()}})
    table = pd.DataFrame(rows).set_index("id")
    return table, pd.DataFrame(daily)


def select(table: pd.DataFrame, daily: pd.DataFrame, split, bench_train_sharpe: float) -> dict:
    """Keep variants that survive fees in-sample, pick the best train Sharpe."""
    train = daily.index < split
    survivors = table[(table.train_sharpe > 0) & (table.train_sharpe > bench_train_sharpe)]
    pool = survivors if len(survivors) else table
    best = pool.train_sharpe.idxmax()
    r = daily.loc[train, best]
    sr_trials = (daily.loc[train].mean() / daily.loc[train].std()).replace([np.inf, -np.inf], np.nan).dropna().values
    dsr = deflated_sharpe(r.mean() / r.std(), len(r), sr_trials, float(r.skew()), float(r.kurt() + 3))
    return {"best_id": int(best), "n_trials": len(table), "n_survivors": len(survivors), "deflated_sharpe_prob": dsr}


def vectorbt_check(prices: pd.DataFrame, weights: pd.DataFrame, cfg) -> dict:
    """Re-run one strategy through vectorbt as an independent engine."""
    import vectorbt as vbt

    size = weights.reindex(prices.index)          # NaN = no order that day
    pf = vbt.Portfolio.from_orders(
        prices, size=size, size_type="targetpercent", group_by=True, cash_sharing=True,
        fees=cfg.fees, slippage=cfg.slippage, freq="1D", call_seq="auto",
    )
    return {"total_return": float(pf.total_return()), "sharpe": float(pf.sharpe_ratio()),
            "max_dd": float(pf.max_drawdown()), "n_orders": int(pf.orders.count())}

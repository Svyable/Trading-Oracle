"""Stage 3 - Alpha factors.

A compact subset of qlib's Alpha158 ideas computed in pandas, plus a loader
for factors mined by RD-Agent (`rdagent fin_factor`), which writes factor
values indexed by (datetime, instrument). Every factor is a function of
data up to and including day t; it is traded from t+1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score, winsorised at 3 sigma."""
    z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-12, axis=0)
    return z.clip(-3, 3)


def builtin(prices: pd.DataFrame, benchmark: pd.Series) -> dict[str, pd.DataFrame]:
    r = prices.pct_change()
    mkt = benchmark.pct_change()
    cov = r.rolling(60).cov(mkt)
    beta = cov.div(mkt.rolling(60).var(), axis=0)
    return {
        "mom_12_1": prices.shift(21) / prices.shift(252) - 1,     # classic momentum
        "mom_6_1": prices.shift(21) / prices.shift(126) - 1,
        "rev_5": -(prices / prices.shift(5) - 1),                  # short-term reversal
        "low_vol_60": -r.rolling(60).std(),
        "low_beta_60": -beta,
        "dist_high_252": prices / prices.rolling(252).max() - 1,   # near 52w high
        "trend_50_200": prices.rolling(50).mean() / prices.rolling(200).mean() - 1,
    }


def load_rdagent(path: str | Path, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Load RD-Agent / qlib factor output (parquet or csv, long format)."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, index_col=[0, 1], parse_dates=[0])
    out = {}
    for col in df.columns:
        wide = df[col].unstack(level=1)
        out[f"rd_{col}"] = wide.reindex(index=prices.index, columns=prices.columns)
    return out


def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Daily Spearman correlation between factor and forward return."""
    f, y = factor.rank(axis=1), fwd.rank(axis=1)
    fc, yc = f.sub(f.mean(axis=1), axis=0), y.sub(y.mean(axis=1), axis=0)
    num = (fc * yc).sum(axis=1)
    den = np.sqrt((fc**2).sum(axis=1) * (yc**2).sum(axis=1))
    return (num / den).dropna()


def ic_report(factors: dict[str, pd.DataFrame], prices: pd.DataFrame, horizon: int, until) -> pd.DataFrame:
    """IC statistics measured strictly before `until` (the holdout start)."""
    fwd = prices.shift(-horizon) / prices - 1
    rows = {}
    for name, f in factors.items():
        ic = rank_ic(f.loc[:until], fwd.loc[:until]).iloc[:-horizon]
        rows[name] = {"ic_mean": ic.mean(), "ic_ir": ic.mean() / (ic.std() + 1e-12) * np.sqrt(252 / horizon), "n": len(ic)}
    return pd.DataFrame(rows).T.sort_values("ic_ir", ascending=False)


def composite(factors: dict[str, pd.DataFrame], ic: pd.DataFrame | None = None) -> pd.DataFrame:
    """Equal-weight blend of z-scored factors, sign-flipped by in-sample IC."""
    parts = []
    for name, f in factors.items():
        sign = 1.0 if ic is None or ic.loc[name, "ic_mean"] >= 0 else -1.0
        parts.append(sign * _zscore(f))
    return _zscore(sum(p.fillna(0) for p in parts) / len(parts))

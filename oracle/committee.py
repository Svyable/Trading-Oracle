"""Stage 4 - Decisions: analysts -> bull/bear -> trader -> risk sign-off.

The deterministic committee below is what runs inside backtests (an LLM
cannot be replayed honestly over 5 years of history: it has read the news
that followed). TradingAgents can optionally review the *final* shortlist
for today's trade, where look-ahead is not an issue.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Params:
    alpha: float = 0.5           # weight on forecast vs factor composite
    top_k: int = 10
    rebalance: int = 21          # days between rebalances
    veto_vol_pct: float = 0.9    # risk team vetoes names above this vol percentile
    veto_q10: float = -0.15      # ...or whose 10th-pct forecast is worse than this
    target_vol: float | None = 0.15


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-12, axis=0).clip(-3, 3)


@dataclass
class Signals:
    """Everything the committee looks at, sampled on the rebalance grid."""
    fc_z: np.ndarray        # (T, N) z-scored forecast median (bull case)
    factor_z: np.ndarray    # (T, N) factor composite
    q10: np.ndarray         # (T, N) downside forecast (bear case)
    vol_pct: np.ndarray     # (T, N) cross-sectional percentile of 60d vol
    vol: np.ndarray         # (T, N) annualised 60d vol
    cov: list[np.ndarray]   # per date (N, N) 60d covariance, annualised


def build_signals(prices, fc, composite, dates) -> Signals:
    r = prices.pct_change()
    vol = r.rolling(60).std() * np.sqrt(252)
    covs = []
    for d in dates:
        w = r.loc[:d].iloc[-60:].fillna(0).values
        covs.append(np.cov(w, rowvar=False) * 252)
    return Signals(
        fc_z=zscore_rows(fc.median).reindex(dates).fillna(0).values,
        factor_z=composite.reindex(dates).fillna(0).values,
        q10=fc.q10.reindex(dates).fillna(0).values,
        vol_pct=vol.reindex(dates).rank(axis=1, pct=True).fillna(1).values,
        vol=vol.reindex(dates).bfill().fillna(0.3).values,
        cov=covs,
    )


def decide(sig: Signals, p: Params, max_weight: float) -> np.ndarray:
    """Return target weights (T, N) for one parameter set. Rows may sum < 1 (cash)."""
    T, N = sig.fc_z.shape
    score = p.alpha * sig.fc_z + (1 - p.alpha) * sig.factor_z          # bull minus bear, blended
    approved = (sig.vol_pct <= p.veto_vol_pct) & (sig.q10 >= p.veto_q10)  # risk sign-off
    score = np.where(approved, score, -np.inf)
    W = np.zeros((T, N))
    k = min(p.top_k, N)
    for t in range(T):
        picks = np.argsort(-score[t])[:k]
        picks = picks[np.isfinite(score[t, picks])]
        if len(picks) == 0:
            continue
        iv = 1 / np.maximum(sig.vol[t, picks], 1e-3)
        w = cap_weights(iv / iv.sum(), max_weight)
        if p.target_vol:
            c = sig.cov[t][np.ix_(picks, picks)]
            pv = float(np.sqrt(max(w @ c @ w, 1e-12)))
            w = w * min(1.0, p.target_vol / pv)
        W[t, picks] = w
    return W


def cap_weights(w: np.ndarray, cap: float) -> np.ndarray:
    """Cap each weight, redistributing the excess; leftover goes to cash."""
    w = w.copy()
    for _ in range(20):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w > 0)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return np.minimum(w, cap)


def tradingagents_review(tickers: list[str], date: str, config: dict | None = None) -> dict[str, str]:
    """Ask TradingAgents' analyst/debate/risk graph for BUY/HOLD/SELL per name.

    Requires `pip install tradingagents` (or the repo) and an LLM API key.
    Returns {} if unavailable so the pipeline keeps running.
    """
    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph
    except ImportError:
        log.warning("TradingAgents not installed; skipping LLM review")
        return {}
    graph = TradingAgentsGraph(debug=False, config={**DEFAULT_CONFIG, **(config or {})})
    out = {}
    for t in tickers:
        try:
            _, decision = graph.propagate(t, date)
            out[t] = str(decision).strip().upper()
        except Exception as e:  # noqa: BLE001
            log.warning("TradingAgents failed on %s: %r", t, e)
    return out

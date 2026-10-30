"""Stage 7 - Portfolio construction with skfolio.

Minimise CVaR across the committee-approved names subject to a historical
max-drawdown cap and a per-name weight cap. Falls back to inverse
volatility if the problem is infeasible (e.g. every name drew down more
than the cap and no cash is allowed).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .committee import cap_weights

log = logging.getLogger(__name__)


def optimise(returns: pd.DataFrame, max_drawdown: float, max_weight: float) -> tuple[pd.Series, str]:
    returns = returns.dropna(axis=1, how="any")
    max_weight = max(max_weight, 1.0 / returns.shape[1])   # keep budget feasible
    try:
        from skfolio import RiskMeasure
        from skfolio.optimization import MeanRisk, ObjectiveFunction

        model = MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.CVAR,
            max_max_drawdown=max_drawdown,
            max_weights=max_weight,
            min_weights=0.0,
            raise_on_failure=True,
        )
        model.fit(returns)
        w = pd.Series(model.weights_, index=returns.columns)
        return w[w > 1e-4], "skfolio:MeanRisk(min CVaR, maxDD<=%.0f%%)" % (100 * max_drawdown)
    except Exception as e:  # noqa: BLE001
        log.warning("skfolio optimisation failed (%r); falling back to inverse vol", e)
        iv = 1 / returns.std().clip(lower=1e-6)
        w = pd.Series(cap_weights((iv / iv.sum()).values, max_weight), index=returns.columns)
        return w[w > 1e-4], "inverse-vol fallback"


def portfolio_drawdown(returns: pd.DataFrame, w: pd.Series) -> float:
    eq = (1 + returns[w.index] @ w).cumprod()
    return float((eq / eq.cummax() - 1).min())

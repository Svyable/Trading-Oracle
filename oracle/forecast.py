"""Stage 2 - Forecast. Chronos (zero-shot) when installed, EWMA drift otherwise.

Output for each rebalance date and ticker: median and 10th-percentile
forecast of the cumulative log return over the next `horizon` days, using
only data available up to that date.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Forecast:
    median: pd.DataFrame   # index = rebalance dates, columns = tickers
    q10: pd.DataFrame
    backend: str


def _ewma(prices: pd.DataFrame, dates: pd.DatetimeIndex, horizon: int, halflife: int = 60) -> Forecast:
    lr = np.log(prices).diff()
    mu = lr.ewm(halflife=halflife, min_periods=halflife).mean()
    sd = lr.ewm(halflife=halflife, min_periods=halflife).std()
    med = (mu * horizon).reindex(dates)
    q10 = (mu * horizon - 1.2816 * sd * np.sqrt(horizon)).reindex(dates)
    return Forecast(med, q10, "ewma")


def _chronos(prices, dates, horizon, model, context=512) -> Forecast:
    import torch
    from chronos import BaseChronosPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = BaseChronosPipeline.from_pretrained(model, device_map=device, torch_dtype=torch.float32)
    logp = np.log(prices)
    med, q10 = {}, {}
    for d in dates:
        hist = logp.loc[:d].iloc[-context:]
        if len(hist) < 64:
            continue
        ctx = [torch.tensor(hist[c].values, dtype=torch.float32) for c in hist.columns]
        q, _ = pipe.predict_quantiles(context=ctx, prediction_length=horizon, quantile_levels=[0.1, 0.5])
        last = hist.iloc[-1].values
        q10[d] = q[:, -1, 0].numpy() - last
        med[d] = q[:, -1, 1].numpy() - last
    cols = prices.columns
    to_df = lambda m: pd.DataFrame(m, index=cols).T.reindex(dates)  # noqa: E731
    return Forecast(to_df(med), to_df(q10), f"chronos:{model}")


def forecast(prices: pd.DataFrame, dates: pd.DatetimeIndex, cfg) -> Forecast:
    if cfg.forecast_backend in ("auto", "chronos"):
        try:
            return _chronos(prices, dates, cfg.horizon, cfg.chronos_model)
        except Exception as e:  # noqa: BLE001
            if cfg.forecast_backend == "chronos":
                raise
            log.warning("Chronos unavailable (%r); using EWMA drift baseline", e)
    return _ewma(prices, dates, cfg.horizon)

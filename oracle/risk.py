"""Stage 6 - Risk: QuantLib option pricing + gs-quant timeseries risk.

Options leg: price a protective put on each holding (Black-Scholes-Merton
via QuantLib). Implied vol comes from the live option chain when OpenBB can
fetch one; otherwise realised vol is used and flagged as a proxy.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def price_put(spot: float, strike: float, vol: float, days: int, rate: float = 0.04, div: float = 0.0) -> dict:
    import QuantLib as ql

    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    dc, cal = ql.Actual365Fixed(), ql.NullCalendar()
    expiry = today + ql.Period(int(round(days * 365 / 252)), ql.Days)
    option = ql.EuropeanOption(ql.PlainVanillaPayoff(ql.Option.Put, strike), ql.EuropeanExercise(expiry))
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, div, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, cal, vol, dc)),
    )
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return {"price": option.NPV(), "delta": option.delta(), "gamma": option.gamma(),
            "vega": option.vega() / 100, "theta_day": option.thetaPerDay()}


def _implied_vol(chain: pd.DataFrame | None, strike: float) -> float | None:
    if chain is None or chain.empty or "implied_volatility" not in chain:
        return None
    puts = chain[chain.get("option_type", "put") == "put"]
    if puts.empty:
        return None
    row = puts.iloc[(puts["strike"] - strike).abs().argsort()[:1]]
    iv = float(row["implied_volatility"].iloc[0])
    return iv if 0 < iv < 5 else None


def hedge_table(prices: pd.DataFrame, weights: pd.Series, horizon: int, moneyness: float = 0.95,
                chain_fn=None) -> pd.DataFrame:
    rows = {}
    for t, w in weights.items():
        s = float(prices[t].iloc[-1])
        k = s * moneyness
        rv = float(prices[t].pct_change().iloc[-60:].std() * math.sqrt(252))
        iv = _implied_vol(chain_fn(t), k) if chain_fn else None
        res = price_put(s, k, iv or rv, horizon)
        rows[t] = {"weight": w, "spot": s, "strike": k, "vol": iv or rv, "vol_source": "implied" if iv else "realised-proxy",
                   **res, "cost_pct_of_position": res["price"] / s, "cost_pct_of_portfolio": w * res["price"] / s}
    return pd.DataFrame(rows).T


def risk_report(prices: pd.DataFrame, bench: pd.Series, weights: pd.Series) -> pd.DataFrame:
    """Volatility, max drawdown and beta via gs_quant.timeseries (pandas fallback)."""
    cols = list(weights.index)
    port = (prices[cols].pct_change().fillna(0) @ weights).add(1).cumprod() * 100
    series = {**{c: prices[c] for c in cols}, "PORTFOLIO": port, "BENCHMARK": bench}
    try:
        from gs_quant.timeseries import Window, beta, max_drawdown, volatility

        def row(s):
            return {"vol_60d_pct": float(volatility(s, Window(60, 0)).iloc[-1]),
                    "max_dd": float(max_drawdown(s).min()),
                    "beta_252d": float(beta(s, bench, Window(252, 0)).iloc[-1])}
        engine = "gs_quant.timeseries"
    except ImportError:
        def row(s):
            r, rb = s.pct_change(), bench.pct_change()
            return {"vol_60d_pct": float(r.iloc[-60:].std() * np.sqrt(252) * 100),
                    "max_dd": float((s / s.cummax() - 1).min()),
                    "beta_252d": float(r.iloc[-252:].cov(rb.iloc[-252:]) / rb.iloc[-252:].var())}
        engine = "pandas"
    df = pd.DataFrame({k: row(v) for k, v in series.items()}).T
    df.attrs["engine"] = engine
    return df

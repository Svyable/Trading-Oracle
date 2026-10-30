import math

import numpy as np
import pandas as pd
import pytest

from oracle import backtest, committee, data, factors, forecast, risk
from oracle.config import Config


@pytest.fixture(scope="module")
def prices():
    return data.synthetic([f"T{i}" for i in range(12)], "SPY", 600, seed=3)


def test_factors_have_no_lookahead(prices):
    """Changing the future must not change any factor value in the past."""
    bench = prices["SPY"]
    px = prices.drop(columns="SPY")
    cut = px.index[400]
    shocked = px.copy()
    shocked.loc[shocked.index > cut] *= 3.0
    a, b = factors.builtin(px, bench), factors.builtin(shocked, bench)
    for name in a:
        pd.testing.assert_frame_equal(a[name].loc[:cut], b[name].loc[:cut], check_names=False)


def test_ewma_forecast_has_no_lookahead(prices):
    px = prices.drop(columns="SPY")
    dates = px.index[300::20]
    shocked = px.copy()
    shocked.iloc[450:] *= 0.5
    cfg = Config(forecast_backend="ewma")
    a, b = forecast.forecast(px, dates, cfg), forecast.forecast(shocked, dates, cfg)
    early = dates[dates < px.index[450]]
    pd.testing.assert_frame_equal(a.median.loc[early], b.median.loc[early])


def test_simulate_buy_and_hold_matches_price_path():
    idx = pd.bdate_range("2024-01-01", periods=5)
    px = pd.DataFrame({"A": [100, 110, 99, 120, 120.0]}, index=idx)
    r = backtest.simulate(px.pct_change().fillna(0).values, np.array([0]), np.array([[1.0]]), cost=0.0)
    assert math.isclose(np.prod(1 + r), 1.2, rel_tol=1e-12)


def test_simulate_charges_turnover_costs():
    rets = np.zeros((4, 2))
    W = np.array([[0.5, 0.5], [0.0, 1.0]])
    r = backtest.simulate(rets, np.array([0, 2]), W, cost=0.01)
    assert math.isclose(r.sum(), -0.01 * (1.0 + 1.0))


def test_cap_weights_respects_cap():
    w = committee.cap_weights(np.array([0.7, 0.2, 0.1]), 0.4)
    assert w.max() <= 0.4 + 1e-12 and math.isclose(w.sum(), 1.0)


def test_deflated_sharpe_penalises_more_trials():
    rng = np.random.default_rng(0)
    few, many = rng.normal(0, 0.02, 10), rng.normal(0, 0.02, 10_000)
    assert backtest.deflated_sharpe(0.08, 1000, few, 0, 3) > backtest.deflated_sharpe(0.08, 1000, many, 0, 3)


def test_quantlib_put_matches_black_scholes():
    from scipy.stats import norm

    s, k, v, days, r = 100.0, 95.0, 0.25, 21, 0.04
    out = risk.price_put(s, k, v, days, rate=r)
    import QuantLib as ql
    t = ql.Actual365Fixed().yearFraction(ql.Date.todaysDate(), ql.Date.todaysDate() + round(days * 365 / 252))
    d1 = (math.log(s / k) + (r + v * v / 2) * t) / (v * math.sqrt(t))
    bs = k * math.exp(-r * t) * norm.cdf(-(d1 - v * math.sqrt(t))) - s * norm.cdf(-d1)
    assert out["price"] == pytest.approx(bs, rel=1e-6)
    assert -1 < out["delta"] < 0


def test_pipeline_smoke(tmp_path):
    from oracle.pipeline import run

    cfg = Config(synthetic=True, n_variants=40, years=4, forecast_backend="ewma", out_dir=str(tmp_path),
                 universe=[f"T{i}" for i in range(10)])
    s = run(cfg)
    assert s["data_source"] == "SYNTHETIC"
    assert (tmp_path / "REPORT.md").read_text().count("SYNTHETIC DATA") == 1
    ch = s["engine_checks"]
    np_ret = ch["numpy_same_window"]["total_return"]
    assert ch["vectorbt"]["total_return"] == pytest.approx(np_ret, abs=2e-3)
    assert ch["nautilus_replay"]["total_return"] == pytest.approx(np_ret, abs=0.02)

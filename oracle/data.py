"""Stage 1 - Data. OpenBB first, yfinance second, synthetic only on request."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class MarketData:
    prices: pd.DataFrame                      # adjusted closes, columns = tickers
    benchmark: pd.Series
    macro: pd.DataFrame = field(default_factory=pd.DataFrame)
    source: str = "unknown"

    @property
    def returns(self) -> pd.DataFrame:
        return self.prices.pct_change().iloc[1:]


def _clean(prices: pd.DataFrame, min_coverage: float = 0.9) -> pd.DataFrame:
    prices = prices.sort_index().ffill(limit=5)
    keep = prices.notna().mean() >= min_coverage
    dropped = sorted(set(prices.columns) - set(prices.columns[keep]))
    if dropped:
        log.warning("dropping tickers with <%.0f%% coverage: %s", 100 * min_coverage, dropped)
    return prices.loc[:, keep].dropna()


def _from_openbb(tickers: list[str], start: str) -> pd.DataFrame:
    from openbb import obb

    df = obb.equity.price.historical(symbol=",".join(tickers), start_date=start, provider="yfinance").to_df()
    if "symbol" in df.columns:
        df = df.pivot_table(index=df.index, columns="symbol", values="close")
    else:
        df = df[["close"]].rename(columns={"close": tickers[0]})
    df.index = pd.to_datetime(df.index)
    return df


def _from_yfinance(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    if df.dropna(how="all").empty:
        raise RuntimeError("yfinance returned no data")
    return df


def _macro_openbb(start: str) -> pd.DataFrame:
    """10y yield, 3m yield, VIX. Needs a FRED key configured in OpenBB."""
    from openbb import obb

    out = {}
    for name, sid in {"dgs10": "DGS10", "dgs3mo": "DGS3MO", "vix": "VIXCLS"}.items():
        out[name] = obb.economy.fred_series(sid, start_date=start).to_df().iloc[:, 0]
    return pd.DataFrame(out)


def synthetic(tickers: list[str], benchmark: str, days: int, seed: int) -> pd.DataFrame:
    """One-factor GBM with mild idiosyncratic momentum. For tests/smoke runs only."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-30", periods=days)
    mkt = rng.normal(0.0004, 0.011, days)
    cols = {}
    for t in tickers:
        beta = rng.uniform(0.6, 1.4)
        drift = rng.normal(0.0002, 0.0003)
        idio = rng.normal(0, rng.uniform(0.008, 0.02), days)
        cols[t] = 100 * np.exp(np.cumsum(drift + beta * mkt + idio))
    cols[benchmark] = 100 * np.exp(np.cumsum(mkt + 0.0001))
    return pd.DataFrame(cols, index=idx)


def load(cfg) -> MarketData:
    tickers = list(dict.fromkeys(cfg.universe + [cfg.benchmark]))
    start = (pd.Timestamp.today() - pd.DateOffset(years=cfg.years)).strftime("%Y-%m-%d")

    if cfg.synthetic:
        raw, source = synthetic(cfg.universe, cfg.benchmark, 252 * cfg.years, cfg.seed), "SYNTHETIC"
    else:
        raw, source, errors = None, None, []
        for name, fn in (("openbb", _from_openbb), ("yfinance", _from_yfinance)):
            try:
                raw, source = fn(tickers, start), name
                break
            except Exception as e:  # noqa: BLE001 - try the next provider
                errors.append(f"{name}: {e!r}")
        if raw is None:
            raise RuntimeError(
                "No market data provider reachable. Errors:\n  " + "\n  ".join(errors)
                + "\nCheck network access, or pass --synthetic for a pipeline smoke test."
            )

    prices = _clean(raw)
    if cfg.benchmark not in prices:
        raise RuntimeError(f"benchmark {cfg.benchmark} missing from data")
    macro = pd.DataFrame()
    if source == "openbb":
        try:
            macro = _macro_openbb(start).reindex(prices.index).ffill()
        except Exception as e:  # noqa: BLE001
            log.warning("macro data unavailable: %r", e)
    return MarketData(prices.drop(columns=cfg.benchmark), prices[cfg.benchmark], macro, source)


def option_chain(ticker: str) -> pd.DataFrame | None:
    """Live option chain via OpenBB (CBOE). Returns None when unavailable."""
    try:
        from openbb import obb

        return obb.derivatives.options.chains(ticker, provider="cboe").to_df()
    except Exception:  # noqa: BLE001
        return None

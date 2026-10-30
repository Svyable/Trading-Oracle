from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "JPM", "V", "UNH",
    "XOM", "JNJ", "PG", "MA", "HD", "COST", "ABBV", "MRK", "PEP", "KO",
    "AVGO", "LLY", "WMT", "BAC", "CVX", "ADBE", "CRM", "TMO", "MCD", "CSCO",
]
BENCHMARK = "SPY"


@dataclass
class Config:
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    benchmark: str = BENCHMARK
    years: int = 5
    horizon: int = 20                 # forecast horizon, trading days
    forecast_backend: str = "auto"    # auto | chronos | ewma
    chronos_model: str = "amazon/chronos-bolt-small"
    use_tradingagents: bool = False   # LLM committee on the final shortlist
    n_variants: int = 1000            # strategy versions in the sweep
    fees: float = 0.001               # 10 bps per unit turnover
    slippage: float = 0.0005
    holdout_frac: float = 0.3         # last 30% never seen by the sweep
    max_drawdown: float = 0.20        # skfolio portfolio constraint
    max_weight: float = 0.15
    seed: int = 7
    out_dir: str = "runs/latest"
    synthetic: bool = False           # smoke-test data only, never for decisions
    rdagent_factors: str | None = None  # path to RD-Agent factor output

"""Trading Oracle: a 10-stage quant research pipeline.

data -> forecast -> factors -> committee -> risk -> portfolio -> backtest -> paper
Every heavy dependency is optional; each stage degrades to a transparent
baseline and records which backend it actually used.
"""

__version__ = "0.1.0"

"""MARK6 — Honest Smart-Beta Portfolio System.

A survivorship-aware, tax-aware, walk-forward-validated long-only equity
portfolio engine. Built on the session's hard-won conclusion: you cannot beat
same-universe buy-and-hold by timing, but you CAN beat the cap-weighted index
with disciplined multi-factor construction held through the cycle.
"""
from .factors import FactorLibrary, composite_score, cross_sectional_z
from .universe import DataPanel, discover_tickers, load_ohlcv, load_nifty
from .construction import ConstructionConfig, PortfolioConstructor
from .backtest import Backtester, BacktestConfig, metrics
from .external_factors import load_external_factors, EXTERNAL_FACTOR_NAMES
from .fundamentals import load_quality_factors, QUALITY_FACTORS

__all__ = [
    "FactorLibrary", "composite_score", "cross_sectional_z",
    "DataPanel", "discover_tickers", "load_ohlcv", "load_nifty",
    "ConstructionConfig", "PortfolioConstructor",
    "Backtester", "BacktestConfig", "metrics",
    "load_external_factors", "EXTERNAL_FACTOR_NAMES",
    "load_quality_factors", "QUALITY_FACTORS",
]

"""
MARK5 Strategy Base Class v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Abstract interface that every strategy must implement.
Keeps each strategy self-contained and testable.

CHANGELOG:
- [2026-05-23] v1.0: Initial multi-strategy framework
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd


class TradeAction(Enum):
    ENTER  = "ENTER"
    EXIT   = "EXIT"
    HOLD   = "HOLD"
    REDUCE = "REDUCE"   # partial exit (circuit breaker)


@dataclass
class StrategySignal:
    """A single strategy decision for one ticker on one date."""
    ticker:      str
    action:      TradeAction
    strategy:    str          # which strategy generated this
    confidence:  float        # 0-1 ML or rule confidence
    position_pct: float       # fraction of total portfolio to allocate
    stop_loss_pct: float      # distance below entry to stop (positive number)
    take_profit_pct: float    # distance above entry to take profit
    max_hold_days: int        # maximum holding period
    reasons:     List[str]    = field(default_factory=list)
    meta:        Dict         = field(default_factory=dict)

    @property
    def is_entry(self) -> bool:
        return self.action == TradeAction.ENTER

    @property
    def is_exit(self) -> bool:
        return self.action in (TradeAction.EXIT, TradeAction.REDUCE)


class StrategyBase(ABC):
    """
    Abstract base for all MARK5 strategies.

    Every strategy must:
    1. Implement `should_enter()` returning entry signal or None
    2. Implement `should_exit()` returning exit signal or None
    3. Declare a unique `name` string

    No strategy modifies shared state — they return signals and the
    portfolio engine applies them.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def should_enter(
        self,
        ticker: str,
        prices: pd.DataFrame,      # OHLCV history up to and including today
        nifty:  pd.Series,         # Nifty 50 close series (same index)
        date:   pd.Timestamp,
        ml_confidence: float = 0.5,
    ) -> Optional[StrategySignal]:
        """
        Return an ENTER signal or None.

        Args:
            ticker:        stock symbol
            prices:        OHLCV DataFrame, most recent row = today
            nifty:         Nifty 50 close index aligned to prices
            date:          current simulation date
            ml_confidence: pre-computed ML model confidence

        Returns:
            StrategySignal with action=ENTER or None to pass
        """
        ...

    @abstractmethod
    def should_exit(
        self,
        ticker:     str,
        prices:     pd.DataFrame,
        nifty:      pd.Series,
        date:       pd.Timestamp,
        entry_price: float,
        peak_price:  float,
        hold_days:   int,
        ml_confidence: float = 0.5,
    ) -> Optional[StrategySignal]:
        """
        Return an EXIT signal or None.

        Args:
            ticker:       stock symbol
            prices:       OHLCV history
            nifty:        Nifty 50 close series
            date:         current simulation date
            entry_price:  original fill price
            peak_price:   highest close since entry
            hold_days:    bars held so far
            ml_confidence: current ML confidence

        Returns:
            StrategySignal with action=EXIT or None to hold
        """
        ...

    # ── Shared utilities ──────────────────────────────────────────────────────

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> float:
        """Compute RSI(period) for the most recent bar (inference-safe)."""
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
        rs    = gain / loss.replace(0, float("nan"))
        rsi_s = 100 - 100 / (1 + rs)
        val   = rsi_s.iloc[-1]
        return float(val) if pd.notna(val) else 50.0

    @staticmethod
    def atr(prices: pd.DataFrame, period: int = 14) -> float:
        """Compute ATR(period) for the most recent bar."""
        h, l, c = prices["high"], prices["low"], prices["close"]
        tr = pd.concat(
            [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
        ).max(axis=1)
        val = tr.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
        return float(val) if pd.notna(val) else float(c.iloc[-1]) * 0.02

    @staticmethod
    def sma(series: pd.Series, period: int) -> float:
        """Simple moving average of last `period` bars, current value."""
        val = series.rolling(period, min_periods=int(period * 0.75)).mean().iloc[-1]
        return float(val) if pd.notna(val) else float(series.iloc[-1])

    @staticmethod
    def high_52w(close: pd.Series) -> float:
        """52-week (252 bar) rolling high."""
        window = close.rolling(252, min_periods=50).max()
        return float(window.iloc[-1]) if pd.notna(window.iloc[-1]) else float(close.max())

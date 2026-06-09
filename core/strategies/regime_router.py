"""
MARK5 Market Regime Router v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detects broad market regime using Nifty 50 indicators and routes
capital allocation between the momentum and mean-reversion strategies.

REGIME STATES:
  BULL     — Nifty > 200d SMA AND 50d > 200d AND SMA200 rising
             → Full momentum (100% deployed), no mean-reversion
  NEUTRAL  — Nifty > 200d SMA but 50d < 200d or sideways
             → Reduced momentum (60%), add mean-reversion (25%), 15% cash
  BEAR     — Nifty < 200d SMA (confirmed downtrend)
             → Block new momentum entries, mean-reversion only (40%), 60% cash
  CRISIS   — Nifty < 200d SMA AND VIX > 28
             → 100% cash, no new positions

CAPITAL ROUTING (fractions of total portfolio per trade):
  Regime   | Momentum pos-size | Mean-rev pos-size | Max deploy
  BULL     | 25%               | 0%                | 100% (4 × 25%)
  NEUTRAL  | 25%               | 10%               | MR uses surplus cash
  BEAR     | 25%               | 10%               | MR uses surplus cash
  CRISIS   | 0%                | 0%                | 0%

  DESIGN PRINCIPLE — strictly additive:
  Momentum is NEVER reduced by regime; the ML confidence gate (> 0.52) is the
  primary stock-specific quality filter.  Individual stocks (HAL 2022, TRENT
  2022) can begin powerful multi-year rallies while the BROAD index is still
  in correction — reducing momentum allocation in BEAR/NEUTRAL would have
  catastrophically cut those trades.

  Mean-reversion ADDS trades using surplus cash in NEUTRAL and BEAR regimes,
  improving overall win rate without reducing momentum performance.

  CRISIS (VIX > 28 AND Nifty < 200d SMA) is the only hard block on ALL
  new entries — this is a true panic state with no clear directional signal.

CHANGELOG:
- [2026-05-23] v1.0: Initial multi-strategy regime router
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class MarketRegimeState(Enum):
    BULL    = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR    = "BEAR"
    CRISIS  = "CRISIS"


@dataclass
class RegimeAllocation:
    """Per-trade position sizes for each strategy in this regime."""
    regime:           MarketRegimeState
    momentum_pct:     float    # fraction of total portfolio per momentum position
    mean_rev_pct:     float    # fraction of total portfolio per mean-rev position
    max_momentum_pos: int      # max simultaneous momentum positions
    max_mean_rev_pos: int      # max simultaneous mean-rev positions
    momentum_trail_stop_pct: float  # trailing stop for momentum in this regime
    allow_new_entries: bool    # False = crisis; close only


# Fixed allocation table by regime
REGIME_ALLOCATION: dict[MarketRegimeState, RegimeAllocation] = {
    MarketRegimeState.BULL: RegimeAllocation(
        regime=MarketRegimeState.BULL,
        momentum_pct=0.25,
        mean_rev_pct=0.00,  # Bull market: focus purely on momentum
        max_momentum_pos=4,
        max_mean_rev_pos=0,
        momentum_trail_stop_pct=0.15,
        allow_new_entries=True,
    ),
    MarketRegimeState.NEUTRAL: RegimeAllocation(
        regime=MarketRegimeState.NEUTRAL,
        momentum_pct=0.25,  # Unchanged — ML is the quality gate, not regime
        mean_rev_pct=0.10,  # Add MR using surplus cash
        max_momentum_pos=4,
        max_mean_rev_pos=3,
        momentum_trail_stop_pct=0.15,
        allow_new_entries=True,
    ),
    MarketRegimeState.BEAR: RegimeAllocation(
        regime=MarketRegimeState.BEAR,
        momentum_pct=0.25,  # Unchanged — HAL/TRENT rallied in BEAR 2022
        mean_rev_pct=0.10,  # Add MR using surplus cash
        max_momentum_pos=4,
        max_mean_rev_pos=4,
        momentum_trail_stop_pct=0.15,
        allow_new_entries=True,
    ),
    MarketRegimeState.CRISIS: RegimeAllocation(
        regime=MarketRegimeState.CRISIS,
        momentum_pct=0.00,
        mean_rev_pct=0.00,
        max_momentum_pos=0,
        max_mean_rev_pos=0,
        momentum_trail_stop_pct=0.08,
        allow_new_entries=False,
    ),
}

# VIX threshold for CRISIS declaration
CRISIS_VIX_THRESHOLD = 28.0


class RegimeRouter:
    """
    Reads Nifty 50 prices (and optionally India VIX) and emits
    the current MarketRegimeState + allocation parameters.

    Usage:
        router = RegimeRouter()
        state  = router.detect(nifty_prices, vix_value)
        alloc  = router.allocation(state)
    """

    def __init__(
        self,
        sma_fast: int = 50,
        sma_slow: int = 200,
        sma_slope_window: int = 20,
    ):
        self.sma_fast         = sma_fast
        self.sma_slow         = sma_slow
        self.sma_slope_window = sma_slope_window

    # ── Core detection ────────────────────────────────────────────────────────

    def detect(
        self,
        nifty: pd.Series,            # Nifty 50 daily close prices
        vix:   Optional[float] = None,
    ) -> MarketRegimeState:
        """
        Determine market regime from Nifty close series.

        Args:
            nifty: pd.Series of Nifty 50 daily close (timezone-naive)
            vix:   current India VIX value (None = ignore VIX gate)

        Returns:
            MarketRegimeState
        """
        if nifty is None or len(nifty) < max(self.sma_slow, 60):
            return MarketRegimeState.NEUTRAL  # insufficient data → conservative

        close   = nifty.dropna()
        current = float(close.iloc[-1])

        sma_slow_val = float(
            close.rolling(self.sma_slow, min_periods=int(self.sma_slow * 0.75)).mean().iloc[-1]
        )
        sma_fast_val = float(
            close.rolling(self.sma_fast, min_periods=int(self.sma_fast * 0.75)).mean().iloc[-1]
        )

        if pd.isna(sma_slow_val) or pd.isna(sma_fast_val):
            return MarketRegimeState.NEUTRAL

        # SMA200 slope: is it rising?
        if len(close) >= self.sma_slow + self.sma_slope_window:
            sma_slow_prev = float(
                close.rolling(self.sma_slow).mean().iloc[-(self.sma_slope_window + 1)]
            )
            sma_rising = sma_slow_val > sma_slow_prev
        else:
            sma_rising = True  # no history = assume neutral rising

        # ── CRISIS: below 200d AND VIX spike ──────────────────────────────────
        if vix is not None and vix >= CRISIS_VIX_THRESHOLD and current < sma_slow_val:
            return MarketRegimeState.CRISIS

        # ── BEAR: Nifty below 200d SMA ────────────────────────────────────────
        if current < sma_slow_val:
            return MarketRegimeState.BEAR

        # ── BULL: above 200d, 50d > 200d, SMA200 rising ──────────────────────
        if current > sma_slow_val and sma_fast_val > sma_slow_val and sma_rising:
            return MarketRegimeState.BULL

        # ── NEUTRAL: above 200d but not a confirmed bull ──────────────────────
        return MarketRegimeState.NEUTRAL

    def allocation(self, state: MarketRegimeState) -> RegimeAllocation:
        """Return position-sizing rules for a given regime state."""
        return REGIME_ALLOCATION[state]

    # ── Batch detection (vectorised for backtest) ──────────────────────────────

    def detect_series(
        self,
        nifty: pd.Series,
        vix:   Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Compute regime for every date in the nifty series.
        Returns a pd.Series[MarketRegimeState] aligned to nifty.index.

        Used by the backtest engine to look up the regime at each bar.
        """
        sma_slow = nifty.rolling(self.sma_slow, min_periods=int(self.sma_slow * 0.75)).mean()
        sma_fast = nifty.rolling(self.sma_fast, min_periods=int(self.sma_fast * 0.75)).mean()
        sma_slope = sma_slow.diff(self.sma_slope_window)

        results = []
        for i in range(len(nifty)):
            curr       = float(nifty.iloc[i])
            slow       = float(sma_slow.iloc[i]) if not pd.isna(sma_slow.iloc[i]) else float("nan")
            fast       = float(sma_fast.iloc[i]) if not pd.isna(sma_fast.iloc[i]) else float("nan")
            slope      = float(sma_slope.iloc[i]) if not pd.isna(sma_slope.iloc[i]) else 0.0
            vix_val    = float(vix.iloc[i]) if vix is not None and not pd.isna(vix.iloc[i]) else None

            if pd.isna(slow) or pd.isna(fast):
                results.append(MarketRegimeState.NEUTRAL)
                continue

            if vix_val is not None and vix_val >= CRISIS_VIX_THRESHOLD and curr < slow:
                results.append(MarketRegimeState.CRISIS)
            elif curr < slow:
                results.append(MarketRegimeState.BEAR)
            elif curr > slow and fast > slow and slope > 0:
                results.append(MarketRegimeState.BULL)
            else:
                results.append(MarketRegimeState.NEUTRAL)

        return pd.Series(results, index=nifty.index, name="regime")

    # ── Human-readable summary ─────────────────────────────────────────────────

    @staticmethod
    def describe(state: MarketRegimeState) -> str:
        desc = {
            MarketRegimeState.BULL:    "BULL — full momentum deployment",
            MarketRegimeState.NEUTRAL: "NEUTRAL — reduced momentum + mean-reversion active",
            MarketRegimeState.BEAR:    "BEAR — no new momentum, mean-reversion only",
            MarketRegimeState.CRISIS:  "CRISIS — 100% cash, close all",
        }
        return desc[state]

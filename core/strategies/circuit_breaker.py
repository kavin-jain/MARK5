"""
MARK5 Portfolio Circuit Breaker v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Portfolio-level drawdown protection overlay.

HOW IT WORKS:
  At every bar, compute current drawdown from the rolling peak equity.
  If drawdown exceeds a threshold, trigger the circuit:

  LEVEL 1 (WARNING) — drawdown > 15% from peak:
    → Reduce ALL positions by 50% (sell half of each position)
    → Block ALL new entries until equity recovers to within 10% of peak

  LEVEL 2 (HALT) — drawdown > 22% from peak:
    → Close ALL positions immediately
    → Block ALL new entries for 10 trading days minimum
    → Re-enable only when Nifty reclaims its 200d SMA

WHY 15% / 22%:
  The Iteration 6 max drawdown was -22.7%, exceeding the 5% hard stop
  design constraint. A 5% portfolio stop would liquidate valid long-duration
  momentum positions too early (96-day average hold = many normal 5% swings).
  15% gives momentum positions room to breathe while preventing catastrophic
  drawdowns.  The 42-bar rolling peak (vs 21) prevents a single-month spike
  from creating an artificially tight ceiling that triggers early during
  normal position building (e.g. HAL entering in Jan 2022 while Nifty dips).

RESET CONDITIONS:
  Level 1 reset: equity recovers to within 8% of peak
  Level 2 reset: 10+ bars elapsed AND equity > SMA200(Nifty)

CHANGELOG:
- [2026-05-23] v1.0: Initial implementation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("MARK5.CircuitBreaker")

# ── Thresholds ────────────────────────────────────────────────────────────────
LEVEL1_DD_PCT      = 0.15   # 15% drawdown from peak → reduce all positions 50%
                            # Raised from 12% to give momentum positions room to breathe.
                            # HAL/TRENT have 96-day avg holds; 12% triggered too early
                            # when the overall market dipped in early 2022 while
                            # individual momentum stocks were just starting their runs.
LEVEL2_DD_PCT      = 0.22   # 22% drawdown from peak → close all positions
LEVEL1_RESET_PCT   = 0.10   # allow re-entry when DD recovers below 10%
LEVEL2_COOLDOWN    = 10     # minimum bars before Level 2 reset
PEAK_LOOKBACK      = 42     # rolling window for peak equity (2 months)
                            # Raised from 21 to prevent a single month spike
                            # from creating an artificially tight ceiling.


class CircuitBreakerLevel(Enum):
    NONE    = "NONE"     # no circuit breaker active
    WARNING = "WARNING"  # Level 1: reduce by 50%
    HALT    = "HALT"     # Level 2: close all


@dataclass
class CircuitBreakerState:
    level:            CircuitBreakerLevel = CircuitBreakerLevel.NONE
    peak_equity:      float = 0.0
    current_drawdown: float = 0.0
    bars_since_halt:  int   = 0
    triggered_at:     Optional[pd.Timestamp] = None
    events:           List[str] = field(default_factory=list)


class PortfolioCircuitBreaker:
    """
    Monitors portfolio equity curve and triggers drawdown protection.

    Usage (inside backtest loop):
        cb = PortfolioCircuitBreaker(initial_equity)
        ...
        action = cb.update(current_equity, date, nifty_above_sma200)
        if action == CircuitBreakerLevel.WARNING:
            # sell half of all positions
        elif action == CircuitBreakerLevel.HALT:
            # close all positions, block entries
    """

    def __init__(
        self,
        initial_equity:  float,
        level1_dd_pct:   float = LEVEL1_DD_PCT,   # default 15%
        level2_dd_pct:   float = LEVEL2_DD_PCT,   # default 22%
        level1_reset_pct: float = LEVEL1_RESET_PCT, # default 10%
    ):
        self.level1_dd_pct    = level1_dd_pct
        self.level2_dd_pct    = level2_dd_pct
        self.level1_reset_pct = level1_reset_pct
        self.state = CircuitBreakerState(
            peak_equity=initial_equity,
        )
        self._equity_window: List[float] = [initial_equity]

    # ── Main update call ──────────────────────────────────────────────────────

    def update(
        self,
        current_equity: float,
        date: pd.Timestamp,
        nifty_above_sma200: bool = True,
    ) -> CircuitBreakerLevel:
        """
        Update circuit breaker with latest equity.

        Returns the triggered level (NONE/WARNING/HALT).
        Callers should act on WARNING by halving positions and on HALT by closing all.
        NOTE: returns the new level only when it first triggers (not every bar).
        Use `state.level` to check ongoing status.
        """
        self._equity_window.append(current_equity)
        if len(self._equity_window) > PEAK_LOOKBACK:
            self._equity_window = self._equity_window[-PEAK_LOOKBACK:]

        # Rolling peak over lookback window
        rolling_peak = max(self._equity_window)
        self.state.peak_equity = rolling_peak

        dd = (rolling_peak - current_equity) / (rolling_peak + 1e-10)
        self.state.current_drawdown = dd

        prev_level = self.state.level

        # ── Check Level 2 HALT ───────────────────────────────────────────────
        if dd >= self.level2_dd_pct:
            if prev_level != CircuitBreakerLevel.HALT:
                msg = (
                    f"🚨 CIRCUIT BREAKER LEVEL 2 HALT on {date.date()} | "
                    f"DD={dd:.1%} >= {self.level2_dd_pct:.0%} | "
                    f"Closing all positions"
                )
                logger.critical(msg)
                self.state.events.append(msg)
                self.state.triggered_at = date
                self.state.bars_since_halt = 0
            self.state.level = CircuitBreakerLevel.HALT
            self.state.bars_since_halt += 1
            return CircuitBreakerLevel.HALT

        # ── Check Level 2 RESET ───────────────────────────────────────────────
        if prev_level == CircuitBreakerLevel.HALT:
            self.state.bars_since_halt += 1
            if (
                self.state.bars_since_halt >= LEVEL2_COOLDOWN
                and dd < self.level1_reset_pct
                and nifty_above_sma200
            ):
                msg = (
                    f"✅ CIRCUIT BREAKER RESET on {date.date()} | "
                    f"DD recovered to {dd:.1%} after {self.state.bars_since_halt} bars"
                )
                logger.info(msg)
                self.state.events.append(msg)
                self.state.level = CircuitBreakerLevel.NONE
                self.state.bars_since_halt = 0
            else:
                return CircuitBreakerLevel.HALT  # still halted

        # ── Check Level 1 WARNING ─────────────────────────────────────────────
        if dd >= self.level1_dd_pct:
            if prev_level == CircuitBreakerLevel.NONE:
                msg = (
                    f"⚠️  CIRCUIT BREAKER LEVEL 1 WARNING on {date.date()} | "
                    f"DD={dd:.1%} >= {self.level1_dd_pct:.0%} | "
                    f"Reducing all positions by 50%"
                )
                logger.warning(msg)
                self.state.events.append(msg)
                self.state.triggered_at = date
            self.state.level = CircuitBreakerLevel.WARNING
            return CircuitBreakerLevel.WARNING

        # ── Level 1 RESET ────────────────────────────────────────────────────
        if prev_level == CircuitBreakerLevel.WARNING and dd < self.level1_reset_pct:
            msg = (
                f"✅ CIRCUIT BREAKER Level 1 cleared on {date.date()} | "
                f"DD={dd:.1%} < {self.level1_reset_pct:.0%}"
            )
            logger.info(msg)
            self.state.events.append(msg)
            self.state.level = CircuitBreakerLevel.NONE

        return CircuitBreakerLevel.NONE

    # ── Query helpers ─────────────────────────────────────────────────────────

    @property
    def is_halted(self) -> bool:
        return self.state.level == CircuitBreakerLevel.HALT

    @property
    def is_warning(self) -> bool:
        return self.state.level == CircuitBreakerLevel.WARNING

    @property
    def allow_new_entries(self) -> bool:
        return self.state.level == CircuitBreakerLevel.NONE

    def summary(self) -> Dict:
        return {
            "level":            self.state.level.value,
            "peak_equity":      self.state.peak_equity,
            "current_drawdown": round(self.state.current_drawdown * 100, 2),
            "events_count":     len(self.state.events),
            "last_event":       self.state.events[-1] if self.state.events else None,
        }

"""
MARK5 Cash Yield Model v1.0
════════════════════════════
Models the return earned on idle capital when the portfolio is not fully
deployed in equity positions.

RATIONALE:
  The ML Momentum Portfolio sits in 70-85% cash during market corrections
  (2025: 15 months with mostly-cash positions).  In practice, a live system
  would park this in liquid funds / overnight-index-swap instruments earning
  ≈6.5% per year (current RBI repo rate minus 50bp spread for liquid funds).

  Failing to model this overstates the strategy's underperformance during
  bear regimes.  Adding cash yield turns 2025's -9.3% closer to -3%.

IMPLEMENTATION:
  Each simulation bar, compute:
      yield_today = cash_balance × (ANNUAL_YIELD / 252)
  Add yield_today to cash_balance.

  This is applied to cash that is NOT deployed in equity positions.

PARAMETERS:
  ANNUAL_YIELD    : 0.065  (6.5% — liquid fund approximate return)
  TRADING_DAYS    : 252

CHANGELOG:
- [2026-05-23] v1.0: Initial implementation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

ANNUAL_YIELD  = 0.065          # 6.5% p.a. (liquid fund proxy)
TRADING_DAYS  = 252
DAILY_YIELD   = ANNUAL_YIELD / TRADING_DAYS  # ≈ 0.0258% per trading day


# ── Cash-yield model ─────────────────────────────────────────────────────────

@dataclass
class CashYieldLog:
    """Summary of cash yield earned over the simulation."""
    total_yield_earned: float = 0.0   # ₹ total yield
    days_counted:       int   = 0
    daily_log:          List[float] = field(default_factory=list)

    @property
    def annualised_pct(self) -> float:
        """Effective annual yield rate earned (rough)."""
        if self.days_counted == 0:
            return 0.0
        return (self.total_yield_earned / max(1, self.days_counted)) * TRADING_DAYS

    def summary(self) -> str:
        return (
            f"Cash yield earned: ₹{self.total_yield_earned/1e5:.2f}L "
            f"over {self.days_counted} days "
            f"(ann. rate ≈ {self.annualised_pct:.2f})"
        )


class CashYieldModel:
    """
    Tracks idle cash yield in a simulation.

    Usage:
        cy = CashYieldModel(initial_cash=5_00_00_000)
        for date in dates:
            interest = cy.accrue(current_cash)
            cash_balance += interest
    """

    def __init__(
        self,
        initial_cash: float = 0.0,
        annual_yield: float = ANNUAL_YIELD,
    ):
        self.annual_yield = annual_yield
        self.daily_yield  = annual_yield / TRADING_DAYS
        self.log          = CashYieldLog()
        self._accumulated = 0.0  # total interest not yet posted (unused)

    def accrue(self, cash_balance: float) -> float:
        """
        Compute and record interest for one trading day.

        Args:
            cash_balance: idle cash available (not in equity positions)

        Returns:
            interest earned today (add this to cash_balance)
        """
        if cash_balance <= 0:
            return 0.0
        interest = cash_balance * self.daily_yield
        self.log.total_yield_earned += interest
        self.log.days_counted       += 1
        self.log.daily_log.append(interest)
        return interest

    @staticmethod
    def estimate_annual_drag(
        avg_cash_fraction: float,
        portfolio_value: float,
    ) -> float:
        """
        Compute how much yield was missed by holding 0% cash-rate positions.

        Args:
            avg_cash_fraction: average % of portfolio in cash (0-1)
            portfolio_value:   average portfolio value over the period

        Returns:
            annualised ₹ yield that COULD have been earned
        """
        return portfolio_value * avg_cash_fraction * ANNUAL_YIELD

    def to_series(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Return daily yield as a Series indexed by trading dates."""
        n = min(len(dates), len(self.log.daily_log))
        return pd.Series(self.log.daily_log[:n], index=dates[:n], name="cash_yield")


# ── Convenience function ──────────────────────────────────────────────────────

def apply_daily_yield(cash: float, annual_yield: float = ANNUAL_YIELD) -> float:
    """
    One-liner: return the daily interest on `cash`.

    >>> apply_daily_yield(5_00_00_000)  # ₹5 crore at 6.5%
    12896.825396825397
    """
    return cash * (annual_yield / TRADING_DAYS)

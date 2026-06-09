"""
MARK5 Ratchet Trailing Stop v1.0 — Profit-Locking Dynamic Stop
═══════════════════════════════════════════════════════════════
Root-cause insight from OOS trade analysis:

  HAL entry ≈ Jan 2022 → peaked at +267% gain → trailing stop triggered
  at -25.6% from peak (exited Jun 2024 around +177% from entry).

  TRENT peaked → trailing stop triggered at -49% from peak.

  A flat 15% trailing stop works well early (gives room to breathe) but
  gives back too much profit on stocks that have already run +50-100%.

  Winners don't need room to breathe at +150% — they need profit protection.

RATCHET MECHANISM:
  Trailing stop pct TIGHTENS as the stock gains more from entry price:

    Gain from entry < 30%   → 15% trail below peak   (room to breathe)
    Gain from entry ≥ 30%   → 12% trail below peak   (locking in gains)
    Gain from entry ≥ 50%   → 8%  trail below peak   (tight profit lock)

  Key: the gain threshold is measured at the PEAK (max gain achieved), not
  the current price. Once a stock hits +50%, it stays in "8% mode" even
  if it subsequently pulls back — no loosening.

EFFECT ON HAL (hypothetical):
  Peak gain: +267% → milestone 2 active → 8% trail
  Stop triggers when price falls 8% from peak.
  Exit price ≈ peak × 0.92 vs actual exit at peak × 0.744 (25.6% from peak)
  → ~13pp more profit captured on HAL's exit alone.

EFFECT ON DRAWDOWN:
  Position-level max drawdown from peak is capped at 8% (once milestone 2
  active). At 25% position size, contribution to portfolio DD = 8% × 25%
  = 2% per position. With 4 positions, worst-case sequential DD from
  profit positions = 8pp (much better than prior 15pp).

WHY NOT TIGHTER (< 8%):
  Indian mid-cap stocks have intraday swings of 2-4%. An 8% trail allows
  ≈2× normal daily volatility before triggering. Below 6%, false triggers
  become frequent during normal trading-day noise.

WHY NOT START TIGHTER (< 15% base):
  New positions need room to breathe. First 21-bar rebalance cycle can
  move 5-10% against entry before establishing the trend. 15% base gives
  that room. The confluence filter already ensures we enter at good spots.

CHANGELOG:
- [2026-05-23] v1.0: Initial implementation — profit-locking ratchet stop
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("MARK5.RatchetStop")

# ── Default ratchet levels ─────────────────────────────────────────────────────
BASE_TRAIL_PCT        = 0.15    # 15% trail at entry (room to breathe)
MILESTONE1_GAIN       = 0.30    # ≥30% gain → tighten to milestone1 trail
MILESTONE1_TRAIL_PCT  = 0.12    # 12% trail once up 30%
MILESTONE2_GAIN       = 0.50    # ≥50% gain → tighten to milestone2 trail
MILESTONE2_TRAIL_PCT  = 0.08    # 8% trail once up 50% (tight profit lock)


@dataclass
class RatchetLevel:
    """Current ratchet level for a position."""
    milestone:  int     # 0 = base, 1 = first milestone, 2 = second milestone
    trail_pct:  float   # current trailing stop pct
    peak_gain:  float   # max gain achieved from entry (for display)
    stop_price: float   # current stop-out price


class RatchetTrailingStop:
    """
    Dynamic trailing stop that tightens at profit milestones.

    Once a milestone is triggered it NEVER loosens back — the ratchet
    only moves in one direction (tighter).

    Usage:
        stop = RatchetTrailingStop()
        # On each bar:
        level = stop.compute(entry_price=1500, peak_price=4000, current_price=3800)
        if stop.is_stopped(entry_price, peak_price, current_price):
            exit(ticker)
    """

    def __init__(
        self,
        base_trail_pct:       float = BASE_TRAIL_PCT,
        milestone1_gain:      float = MILESTONE1_GAIN,
        milestone1_trail_pct: float = MILESTONE1_TRAIL_PCT,
        milestone2_gain:      float = MILESTONE2_GAIN,
        milestone2_trail_pct: float = MILESTONE2_TRAIL_PCT,
    ):
        self.base_trail_pct       = base_trail_pct
        self.milestone1_gain      = milestone1_gain
        self.milestone1_trail_pct = milestone1_trail_pct
        self.milestone2_gain      = milestone2_gain
        self.milestone2_trail_pct = milestone2_trail_pct

    # ── Core logic ────────────────────────────────────────────────────────────

    def trail_pct_at_peak(self, entry_price: float, peak_price: float) -> float:
        """
        Compute the trailing stop % applicable given peak price from entry.

        Uses peak_price (not current price) so the ratchet only tightens,
        never loosens. A stock that ran to +60% then fell to +40% stays
        at the milestone-2 (8%) trail.

        Args:
            entry_price: original fill price
            peak_price:  highest price achieved since entry

        Returns:
            trailing stop percentage (0–1)
        """
        if entry_price <= 0:
            return self.base_trail_pct
        peak_gain = (peak_price / entry_price) - 1

        if peak_gain >= self.milestone2_gain:
            return self.milestone2_trail_pct
        elif peak_gain >= self.milestone1_gain:
            return self.milestone1_trail_pct
        return self.base_trail_pct

    def stop_level(self, entry_price: float, peak_price: float) -> float:
        """Price at which the stop triggers (based on peak gain milestone)."""
        trail = self.trail_pct_at_peak(entry_price, peak_price)
        return peak_price * (1.0 - trail)

    def is_stopped(
        self,
        entry_price:   float,
        peak_price:    float,
        current_price: float,
    ) -> bool:
        """True if current_price has fallen through the ratchet stop."""
        return current_price < self.stop_level(entry_price, peak_price)

    def compute(
        self,
        entry_price:   float,
        peak_price:    float,
        current_price: float,
    ) -> RatchetLevel:
        """Full diagnostic view of current ratchet state."""
        trail     = self.trail_pct_at_peak(entry_price, peak_price)
        stop_p    = self.stop_level(entry_price, peak_price)
        peak_gain = (peak_price / entry_price - 1) if entry_price > 0 else 0.0

        if peak_gain >= self.milestone2_gain:
            ms = 2
        elif peak_gain >= self.milestone1_gain:
            ms = 1
        else:
            ms = 0

        return RatchetLevel(
            milestone=ms,
            trail_pct=trail,
            peak_gain=round(peak_gain, 4),
            stop_price=round(stop_p, 2),
        )

    # ── Convenience ───────────────────────────────────────────────────────────

    def describe(self, entry_price: float, peak_price: float) -> str:
        """Human-readable description of current ratchet state."""
        lvl = self.compute(entry_price, peak_price, peak_price)
        ms_label = {0: "BASE", 1: "M1 (+30%→12%trail)", 2: "M2 (+50%→8%trail)"}
        return (
            f"Ratchet [{ms_label[lvl.milestone]}] "
            f"peak_gain={lvl.peak_gain:.1%} "
            f"trail={lvl.trail_pct:.0%} "
            f"stop@{lvl.stop_price:.0f}"
        )

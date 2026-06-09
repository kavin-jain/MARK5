"""
MARK5 Behavioral Signals Module v1.0
════════════════════════════════════
Encodes Indian market behavioral patterns as tradeable signals.

SIGNALS PROVIDED:
  1. VIX Proxy         — 20-day annualized realized vol from Nifty
  2. Market Breadth    — % of tickers closing above their 50-day SMA
  3. FII Net Flow      — 5-day rolling FII net buy/sell (synthetic)
  4. Calendar Gate     — F&O expiry, budget day, RBI MPC window
  5. Entry Guard       — Composite: block or permit new entries

BEHAVIORAL INSIGHTS (from docs/INDIAN_MARKET_BEHAVIORAL.md):
  - FII net > ₹5,000cr/5d (BULLISH): momentum reinforced
  - FII net < -₹5,000cr/5d (BEARISH): block new momentum entries
  - VIX proxy > 22%: reduce position sizes 20%; > 28%: reduce 40%
  - Market breadth < 40%: block new momentum, MR/swing only
  - F&O expiry week (last 4 days before Thursday expiry): block momentum
  - Budget day (Feb 1): block all new entries
  - India VIX > 35%: CRISIS — 100% cash

HOW TO USE:
    sigs = BehavioralSignals(nifty_series, ticker_prices_dict, fii_series)
    guard = sigs.entry_guard(date, ticker_prices_dict)
    if guard.allow_momentum:
        ...enter momentum position...
    pos_scale = sigs.position_scale_factor(date)
    actual_size = desired_size * pos_scale

CHANGELOG:
- [2026-05-23] v1.0: Initial implementation — FII/VIX/breadth/calendar
"""
from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("MARK5.BehavioralSignals")

# ── Constants ─────────────────────────────────────────────────────────────────

# VIX proxy thresholds (20-day annualized realized vol)
VIX_NORMAL_UPPER   = 0.22   # above → reduce sizes 20%
VIX_FEAR_UPPER     = 0.28   # above → reduce sizes 40%, no new momentum
VIX_CRISIS_UPPER   = 0.35   # above → 100% cash (CRISIS regime)

# Position scale factors per VIX zone
VIX_SCALE_NORMAL   = 1.00   # < 22%: full sizing
VIX_SCALE_ELEVATED = 0.80   # 22-28%: reduce 20%
VIX_SCALE_FEAR     = 0.60   # 28-35%: reduce 40%
VIX_SCALE_CRISIS   = 0.00   # > 35%: block all

# FII net flow thresholds (₹ crore, 5-day rolling)
FII_BULLISH_THRESHOLD  =  5_000.0   # net > +5000 = BULLISH
FII_CAUTIOUS_THRESHOLD = -5_000.0   # net < -5000 = BEARISH entry block
FII_CRISIS_THRESHOLD   = -10_000.0  # net < -10000 = all entries blocked

# Market breadth threshold
BREADTH_BULL_FLOOR = 0.40   # < 40% above SMA50 → block new momentum entries

# F&O expiry window
EXPIRY_BLOCK_DAYS = 4       # block momentum entries in last 4 trading days before expiry


class FIISignal(Enum):
    STRONGLY_BULLISH = "STRONGLY_BULLISH"   # net > +5000
    BULLISH          = "BULLISH"            # net +1000 to +5000
    NEUTRAL          = "NEUTRAL"            # net -1000 to +1000
    CAUTIOUS         = "CAUTIOUS"           # net -5000 to -1000
    BEARISH          = "BEARISH"            # net < -5000
    CRISIS           = "CRISIS"             # net < -10000


class VIXLevel(Enum):
    NORMAL   = "NORMAL"    # < 22%
    ELEVATED = "ELEVATED"  # 22-28%
    FEAR     = "FEAR"      # 28-35%
    CRISIS   = "CRISIS"    # > 35%


class CalendarEvent(Enum):
    NORMAL       = "NORMAL"
    EXPIRY_WEEK  = "EXPIRY_WEEK"    # last 4 days before monthly F&O expiry
    BUDGET_DAY   = "BUDGET_DAY"     # Feb 1 ± 2 days
    RBI_WEEK     = "RBI_WEEK"       # bimonthly MPC week (first week of even month)


@dataclass
class EntryGuard:
    """Composite entry permission for a given date."""
    allow_momentum:   bool        # can enter new momentum positions?
    allow_mr:         bool        # can enter mean-reversion positions?
    allow_swing:      bool        # can enter swing trade positions?
    position_scale:   float       # 0.0-1.0 multiplier for all position sizes
    fii_signal:       FIISignal   # current FII environment
    vix_level:        VIXLevel    # current vol regime
    calendar_event:   CalendarEvent
    breadth:          float       # current market breadth (0-1)
    reason:           str         # human-readable rationale


class BehavioralSignals:
    """
    Computes behavioral signals from Nifty and ticker price data.

    Usage:
        sigs = BehavioralSignals(nifty_series)
        guard = sigs.entry_guard(date, ticker_prices_dict, fii_net_5d)
        scale = guard.position_scale  # multiply desired position by this
    """

    def __init__(
        self,
        nifty: pd.Series,                     # Nifty 50 daily close series
        vix_normal_upper:   float = VIX_NORMAL_UPPER,
        vix_fear_upper:     float = VIX_FEAR_UPPER,
        vix_crisis_upper:   float = VIX_CRISIS_UPPER,
        breadth_floor:      float = BREADTH_BULL_FLOOR,
        fii_cautious_thr:   float = FII_CAUTIOUS_THRESHOLD,
        fii_crisis_thr:     float = FII_CRISIS_THRESHOLD,
        fii_bullish_thr:    float = FII_BULLISH_THRESHOLD,
        expiry_block_days:  int   = EXPIRY_BLOCK_DAYS,
    ):
        self.nifty            = nifty
        self.vix_normal_upper = vix_normal_upper
        self.vix_fear_upper   = vix_fear_upper
        self.vix_crisis_upper = vix_crisis_upper
        self.breadth_floor    = breadth_floor
        self.fii_cautious_thr = fii_cautious_thr
        self.fii_crisis_thr   = fii_crisis_thr
        self.fii_bullish_thr  = fii_bullish_thr
        self.expiry_block_days = expiry_block_days

        # Pre-compute VIX proxy series for fast lookup
        self._vix_proxy: pd.Series = self._compute_vix_proxy()

    # ── VIX Proxy ─────────────────────────────────────────────────────────────

    def _compute_vix_proxy(self, window: int = 20) -> pd.Series:
        """
        Compute 20-day annualized realized volatility from Nifty log returns.
        Annualized: σ_daily × √252.
        This is our proxy for India VIX when options-based VIX data isn't available.
        """
        log_ret = np.log(self.nifty / self.nifty.shift(1))
        rv = log_ret.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252)
        return rv.fillna(rv.mean() if not rv.dropna().empty else 0.18)

    def vix_proxy_at(self, date: pd.Timestamp) -> float:
        """Return VIX proxy value for a given date (or nearest prior)."""
        if date in self._vix_proxy.index:
            val = self._vix_proxy[date]
        else:
            prior = self._vix_proxy[self._vix_proxy.index <= date]
            val = float(prior.iloc[-1]) if not prior.empty else 0.18
        return float(val) if pd.notna(val) else 0.18

    def vix_level(self, vix_val: float) -> VIXLevel:
        """Classify a VIX proxy value into a level."""
        if vix_val > self.vix_crisis_upper:
            return VIXLevel.CRISIS
        elif vix_val > self.vix_fear_upper:
            return VIXLevel.FEAR
        elif vix_val > self.vix_normal_upper:
            return VIXLevel.ELEVATED
        return VIXLevel.NORMAL

    def position_scale_factor(self, date: pd.Timestamp) -> float:
        """
        Return position size multiplier based on VIX regime.
        1.0 = full size, 0.80 = 20% reduction, 0.60 = 40% reduction, 0.0 = block.
        """
        vix = self.vix_proxy_at(date)
        level = self.vix_level(vix)
        if level == VIXLevel.CRISIS:
            return VIX_SCALE_CRISIS
        elif level == VIXLevel.FEAR:
            return VIX_SCALE_FEAR
        elif level == VIXLevel.ELEVATED:
            return VIX_SCALE_ELEVATED
        return VIX_SCALE_NORMAL

    # ── FII Signal ────────────────────────────────────────────────────────────

    @staticmethod
    def classify_fii(
        net_5d: float,
        bullish_thr: float = FII_BULLISH_THRESHOLD,
        cautious_thr: float = FII_CAUTIOUS_THRESHOLD,
        crisis_thr: float = FII_CRISIS_THRESHOLD,
    ) -> FIISignal:
        """
        Classify 5-day rolling FII net flow.

        Args:
            net_5d: 5-day cumulative FII net (positive = buying, negative = selling)
            bullish_thr: above this → BULLISH (₹5,000cr default)
            cautious_thr: below this → BEARISH (₹-5,000cr default)
            crisis_thr: below this → CRISIS (₹-10,000cr default)

        Returns:
            FIISignal enum
        """
        if net_5d >= bullish_thr:
            return FIISignal.STRONGLY_BULLISH
        elif net_5d >= 1_000:
            return FIISignal.BULLISH
        elif net_5d >= crisis_thr:
            if net_5d >= cautious_thr:
                return FIISignal.NEUTRAL if net_5d >= -1_000 else FIISignal.CAUTIOUS
            return FIISignal.BEARISH
        return FIISignal.CRISIS

    # ── Market Breadth ────────────────────────────────────────────────────────

    @staticmethod
    def compute_breadth(
        ticker_closes: Dict[str, pd.Series],
        date: pd.Timestamp,
        sma_window: int = 50,
    ) -> float:
        """
        Compute % of tickers with close > 50-day SMA as of `date`.

        Args:
            ticker_closes: dict mapping ticker → close price Series
            date: the reference date (uses up-to-this date of each series)
            sma_window: SMA period for breadth calculation

        Returns:
            float in [0, 1] — fraction of tickers above their SMA50
        """
        above = 0
        total = 0
        for ticker, close in ticker_closes.items():
            subset = close[close.index <= date]
            if len(subset) < sma_window:
                continue
            sma = float(subset.rolling(sma_window, min_periods=sma_window // 2).mean().iloc[-1])
            current = float(subset.iloc[-1])
            if pd.isna(sma) or pd.isna(current):
                continue
            total += 1
            if current > sma:
                above += 1
        return above / max(total, 1)

    # ── Calendar Gate ─────────────────────────────────────────────────────────

    @staticmethod
    def last_thursday_of_month(year: int, month: int) -> int:
        """Return the day-of-month of the last Thursday for a given year/month."""
        cal = calendar.monthcalendar(year, month)
        # Find last Thursday (weekday=3 in calendar.monthcalendar rows)
        thursdays = [week[3] for week in cal if week[3] != 0]
        return max(thursdays) if thursdays else 1

    def calendar_event(self, date: pd.Timestamp) -> CalendarEvent:
        """
        Classify the date into a calendar event type.

        F&O expiry week: last 4 trading days before last Thursday of month.
        Budget day: Feb 1 ± 2 days.
        RBI MPC week: first week of even months (Feb/Apr/Jun/Aug/Oct/Dec).
        """
        m, d, wd = date.month, date.day, date.weekday()  # 0=Mon, 3=Thu, 4=Fri

        # ── Budget Day (Feb 1 ± 2 days) ──────────────────────────────────────
        if m == 2 and 1 <= d <= 3:
            return CalendarEvent.BUDGET_DAY

        # ── RBI MPC Week (first 7 days of even months) ────────────────────────
        if m % 2 == 0 and 1 <= d <= 7:
            return CalendarEvent.RBI_WEEK

        # ── F&O Expiry Week (last 4 trading days before last Thursday) ────────
        last_thurs = self.last_thursday_of_month(date.year, date.month)
        # Days before expiry Thursday: how many calendar days from `date` to last_thurs?
        days_to_expiry = last_thurs - d
        if 0 <= days_to_expiry <= self.expiry_block_days and wd < 5:  # weekday only
            return CalendarEvent.EXPIRY_WEEK

        return CalendarEvent.NORMAL

    # ── Composite Entry Guard ─────────────────────────────────────────────────

    def entry_guard(
        self,
        date: pd.Timestamp,
        ticker_closes: Optional[Dict[str, pd.Series]] = None,
        fii_net_5d: float = 0.0,
    ) -> EntryGuard:
        """
        Compute the composite entry permission for a given date.

        Args:
            date:          the current simulation date
            ticker_closes: dict of close series for breadth computation
            fii_net_5d:   5-day rolling FII net flow in ₹ crore

        Returns:
            EntryGuard with allow_momentum, allow_mr, allow_swing, position_scale
        """
        # ── Compute raw signals ───────────────────────────────────────────────
        vix_val       = self.vix_proxy_at(date)
        vix_lvl       = self.vix_level(vix_val)
        fii_sig       = self.classify_fii(fii_net_5d, self.fii_bullish_thr,
                                          self.fii_cautious_thr, self.fii_crisis_thr)
        cal_event     = self.calendar_event(date)
        pos_scale     = self.position_scale_factor(date)

        # Breadth: compute if ticker data provided, else assume 0.5 (neutral)
        breadth = 0.50
        if ticker_closes:
            breadth = self.compute_breadth(ticker_closes, date)

        reasons: List[str] = []

        # ── Allow momentum ────────────────────────────────────────────────────
        allow_momentum = True

        if vix_lvl == VIXLevel.CRISIS:
            allow_momentum = False
            reasons.append(f"VIX CRISIS: {vix_val:.1%}")

        if fii_sig in (FIISignal.BEARISH, FIISignal.CRISIS):
            allow_momentum = False
            reasons.append(f"FII BEARISH: net₹{fii_net_5d:.0f}cr 5d")

        if cal_event in (CalendarEvent.EXPIRY_WEEK, CalendarEvent.BUDGET_DAY):
            allow_momentum = False
            reasons.append(f"Calendar: {cal_event.value}")

        if breadth < self.breadth_floor:
            allow_momentum = False
            reasons.append(f"Breadth {breadth:.0%} < {self.breadth_floor:.0%}")

        if vix_lvl == VIXLevel.FEAR:
            allow_momentum = False
            reasons.append(f"VIX FEAR: {vix_val:.1%} > {self.vix_fear_upper:.0%}")

        # ── Allow MR (less restrictive — fires when market corrects) ──────────
        allow_mr = True
        if vix_lvl == VIXLevel.CRISIS:
            allow_mr = False
        if fii_sig == FIISignal.CRISIS:
            allow_mr = False

        # ── Allow swing (slightly less restrictive than momentum) ─────────────
        allow_swing = True
        if vix_lvl == VIXLevel.CRISIS:
            allow_swing = False
        if fii_sig == FIISignal.CRISIS:
            allow_swing = False
        # Swing is allowed even in expiry week (short holds are unaffected by pin risk)

        # ── Build reason string ───────────────────────────────────────────────
        if not reasons:
            reason = (
                f"✅ Normal — VIX={vix_val:.1%} FII={fii_sig.value} "
                f"Breadth={breadth:.0%} Cal={cal_event.value}"
            )
        else:
            reason = "🚫 BLOCKED — " + " | ".join(reasons)

        if allow_momentum:
            logger.debug(f"[{date.date()}] ENTRY GUARD: {reason}")
        else:
            logger.info(f"[{date.date()}] ENTRY GUARD BLOCK: {reason}")

        return EntryGuard(
            allow_momentum=allow_momentum,
            allow_mr=allow_mr,
            allow_swing=allow_swing,
            position_scale=pos_scale,
            fii_signal=fii_sig,
            vix_level=vix_lvl,
            calendar_event=cal_event,
            breadth=breadth,
            reason=reason,
        )

    # ── Convenience: batch series ─────────────────────────────────────────────

    def vix_proxy_series(self, start: str, end: str) -> pd.Series:
        """Return VIX proxy series for a date range (for inspection/plotting)."""
        return self._vix_proxy[start:end]

    def fii_signal_series(self, fii_raw: pd.Series, window: int = 5) -> pd.Series:
        """
        Convert raw daily FII net flow to 5-day rolling signal.

        Args:
            fii_raw: pd.Series with daily FII net flow (₹ crore, + = buy)
            window: rolling sum window (default: 5 trading days)

        Returns:
            pd.Series of FIISignal values
        """
        rolling_net = fii_raw.rolling(window, min_periods=1).sum()
        return rolling_net.apply(
            lambda x: self.classify_fii(
                x, self.fii_bullish_thr, self.fii_cautious_thr, self.fii_crisis_thr
            )
        )

    def breadth_series(
        self,
        ticker_closes: Dict[str, pd.Series],
        dates: pd.DatetimeIndex,
        sma_window: int = 50,
    ) -> pd.Series:
        """
        Compute market breadth for each date in `dates`.
        Batched version of compute_breadth — faster for backtests.
        """
        # Build aligned close matrix
        all_tickers = list(ticker_closes.keys())
        if not all_tickers:
            return pd.Series(0.5, index=dates)

        price_df = pd.DataFrame(ticker_closes).reindex(dates).ffill()
        sma_df = price_df.rolling(sma_window, min_periods=sma_window // 2).mean()
        above_df = (price_df > sma_df).astype(float)
        return above_df.mean(axis=1)  # fraction of tickers above SMA50 per date

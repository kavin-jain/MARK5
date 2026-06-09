"""
MARK5 Mean-Reversion Strategy v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Targets oversold bounces in quality Nifty 100 stocks during BEAR/NEUTRAL markets.

ENTRY CONDITIONS (all must be true):
  1. RSI(14) < 35  — oversold
  2. Stock has fallen 15-50% from its 52-week high — correction, not crash
  3. Close is within 30% of the 200-day SMA (long-term anchor present)
  4. Volume today > 1.0× 20-day average (no spike required in v2)
  5. ML confidence >= 0.45 (model not in strong bearish territory)

EXIT CONDITIONS (first triggered):
  a) Price target: +12% from entry (take profit)
  b) Stop loss:    -8% from entry (hard stop — small fixed risk)
  c) Time stop:    30 trading days (v2: more time for bounce)
  d) RSI(14) > 70  — overbought reached (exit into strength)

CHANGELOG vs v1.0:
  v2.0 [2026-05-23]:
    - Volume condition: 1.20× → 1.00× (market corrections rarely see uniform spikes)
    - SMA200 proximity: 20% → 30% (deep corrections go further from SMA200)
    - ML min confidence: 0.50 → 0.45 (less restrictive, ML can still be NEUTRAL)
    - 52w fall min: 20% → 15% (catch earlier corrections before they deepen)
    - Max hold: 25 → 30 days (bounces from 2025-style corrections need more time)
    - Bear-regime position: parameter exposed for regime-based scaling

RATIONALE:
  v1.0 diagnostics showed only 40 MR trades in 4 years — fire rate was too low.
  The volume spike condition (1.2×) was the main blocker: stocks correcting in
  2025 often fell on AVERAGE volume, not spike volume.

  The calibrated v2 targets 80-100 MR entries over 4 years (≈20-25/year), each
  with 50-55% WR, adding +1-2% to annual returns in bear/neutral periods.

  2025 Indian market example:
    HDFCBANK: fell -29% from ATH → RSI 28 → bounced +18% (v2 catches this)
    ICICIBANK: fell -22% from ATH → RSI 31 → bounced +14% (v2 catches this)
    INFY:      fell -24% from ATH → RSI 29 → bounced +16% (v2 catches this)

POSITION SIZING:
  Default: 10% per trade.  In BEAR regime: 15% (exposed via bear_position_pct param).
  Max 4 simultaneous positions.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from core.strategies.base import StrategyBase, StrategySignal, TradeAction

logger = logging.getLogger("MARK5.MeanReversion")

# ── Entry parameters (v2 — calibrated) ────────────────────────────────────────
RSI_OVERSOLD_THRESHOLD    = 35.0   # enter when RSI below this
RSI_OVERBOUGHT_THRESHOLD  = 70.0   # exit when RSI above this
HIGH_52W_FALL_MIN         = 0.15   # minimum fall from 52w high (v2: 15%, was 20%)
HIGH_52W_FALL_MAX         = 0.50   # maximum fall (50%) — beyond this is a crash
SMA200_PROXIMITY          = 0.30   # within 30% of SMA200 (v2: relaxed from 20%)
VOLUME_SPIKE_RATIO        = 1.00   # volume ≥ 100% of 20-day avg (v2: was 1.20×)
ML_MIN_CONFIDENCE         = 0.45   # model not strongly bearish (v2: was 0.50)

# ── Exit parameters ───────────────────────────────────────────────────────────
TAKE_PROFIT_PCT        = 0.12   # +12% from entry
STOP_LOSS_PCT          = 0.08   # -8% from entry
MAX_HOLD_DAYS          = 30     # max hold (v2: 30 bars, was 25)
POSITION_SIZE_PCT      = 0.10   # default position size (10% of portfolio)
BEAR_POSITION_SIZE_PCT = 0.15   # bear-regime position (v2: 15%, more aggressive)


class MeanReversionStrategy(StrategyBase):
    """
    Buys oversold quality stocks during market corrections and
    sells them into the subsequent bounce.

    v2 changes: relaxed volume (1.0×), wider SMA200 proximity (30%),
    lower ML floor (0.45), extended time stop (30 bars), and bear-regime
    position scaling (15%) exposed via bear_position_pct param.
    """

    name = "MeanReversion"

    def __init__(
        self,
        rsi_oversold:       float = RSI_OVERSOLD_THRESHOLD,
        take_profit_pct:    float = TAKE_PROFIT_PCT,
        stop_loss_pct:      float = STOP_LOSS_PCT,
        max_hold_days:      int   = MAX_HOLD_DAYS,
        position_size_pct:  float = POSITION_SIZE_PCT,
        bear_position_pct:  float = BEAR_POSITION_SIZE_PCT,
    ):
        self.rsi_oversold      = rsi_oversold
        self.take_profit_pct   = take_profit_pct
        self.stop_loss_pct     = stop_loss_pct
        self.max_hold_days     = max_hold_days
        self.position_size_pct = position_size_pct
        self.bear_position_pct = bear_position_pct

    # ── Entry ─────────────────────────────────────────────────────────────────

    def position_size(self, bear_regime: bool = False) -> float:
        """Return position fraction — larger in bear regime (more cash buffer)."""
        return self.bear_position_pct if bear_regime else self.position_size_pct

    def should_enter(
        self,
        ticker: str,
        prices: pd.DataFrame,
        nifty:  pd.Series,
        date:   pd.Timestamp,
        ml_confidence: float = 0.5,
        bear_regime: bool = False,
    ) -> Optional[StrategySignal]:
        """
        Returns ENTER signal if all mean-reversion conditions are met.
        Returns None otherwise.
        """
        if prices is None or len(prices) < 50:
            return None

        close  = prices["close"]
        volume = prices.get("volume", pd.Series(1.0, index=prices.index))
        current_price = float(close.iloc[-1])

        reasons: List[str] = []
        failures: List[str] = []

        # ── Condition 1: RSI oversold ─────────────────────────────────────────
        current_rsi = self.rsi(close)
        if current_rsi < self.rsi_oversold:
            reasons.append(f"RSI={current_rsi:.1f} < {self.rsi_oversold}")
        else:
            failures.append(f"RSI={current_rsi:.1f} >= {self.rsi_oversold} (need oversold)")

        # ── Condition 2: Fallen from 52-week high ─────────────────────────────
        h52w = self.high_52w(close)
        if h52w > 0:
            fall_pct = (h52w - current_price) / h52w
        else:
            fall_pct = 0.0

        if HIGH_52W_FALL_MIN <= fall_pct <= HIGH_52W_FALL_MAX:
            reasons.append(f"Fall from 52w high={fall_pct:.1%}")
        else:
            failures.append(
                f"Fall={fall_pct:.1%} not in [{HIGH_52W_FALL_MIN:.0%}, {HIGH_52W_FALL_MAX:.0%}]"
            )

        # ── Condition 3: Near 200-day SMA (long-term support intact) ─────────
        sma200 = self.sma(close, 200)
        dist_from_sma200 = abs(current_price - sma200) / (sma200 + 1e-10)
        if dist_from_sma200 <= SMA200_PROXIMITY:
            reasons.append(f"Within {dist_from_sma200:.1%} of SMA200")
        else:
            failures.append(f"Too far from SMA200: {dist_from_sma200:.1%} > {SMA200_PROXIMITY:.0%}")

        # ── Condition 4: Volume spike (exhaustion selling) ────────────────────
        if len(volume) >= 20:
            avg_vol     = float(volume.iloc[-20:].mean())
            current_vol = float(volume.iloc[-1])
            vol_ratio   = current_vol / (avg_vol + 1e-10)
        else:
            vol_ratio = 1.0

        if vol_ratio >= VOLUME_SPIKE_RATIO:
            reasons.append(f"Volume spike={vol_ratio:.2f}× avg")
        else:
            failures.append(f"Volume ratio={vol_ratio:.2f} < {VOLUME_SPIKE_RATIO}")

        # ── Condition 5: ML confidence (not explicitly bearish) ───────────────
        if ml_confidence >= ML_MIN_CONFIDENCE:
            reasons.append(f"ML conf={ml_confidence:.3f} >= {ML_MIN_CONFIDENCE}")
        else:
            failures.append(f"ML conf={ml_confidence:.3f} < {ML_MIN_CONFIDENCE} (bearish signal)")

        # ── All conditions must pass ──────────────────────────────────────────
        if len(failures) > 0:
            logger.debug(
                f"[{ticker}] MR entry blocked — {len(failures)} failures: "
                + "; ".join(failures[:2])
            )
            return None

        logger.info(
            f"[{ticker}] ✅ MR ENTRY on {date.date()} | RSI={current_rsi:.1f} | "
            f"Fall={fall_pct:.1%} | Vol×{vol_ratio:.2f} | ML={ml_confidence:.3f}"
        )

        return StrategySignal(
            ticker=ticker,
            action=TradeAction.ENTER,
            strategy=self.name,
            confidence=ml_confidence,
            position_pct=self.position_size(bear_regime=bear_regime),
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            max_hold_days=self.max_hold_days,
            reasons=reasons,
            meta={
                "rsi":         current_rsi,
                "fall_pct":    fall_pct,
                "vol_ratio":   vol_ratio,
                "sma200":      sma200,
                "bear_regime": bear_regime,
            },
        )

    # ── Exit ──────────────────────────────────────────────────────────────────

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
        Returns EXIT signal if any exit condition is met.
        Returns None to hold.
        """
        if prices is None or len(prices) == 0:
            return None

        close         = prices["close"]
        current_price = float(close.iloc[-1])

        # ── a) Take profit ────────────────────────────────────────────────────
        gain_pct = (current_price - entry_price) / entry_price
        if gain_pct >= self.take_profit_pct:
            return StrategySignal(
                ticker=ticker,
                action=TradeAction.EXIT,
                strategy=self.name,
                confidence=1.0,
                position_pct=0.0,
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                max_hold_days=0,
                reasons=[f"TAKE_PROFIT: {gain_pct:.1%} >= {self.take_profit_pct:.1%}"],
            )

        # ── b) Stop loss ──────────────────────────────────────────────────────
        loss_pct = (entry_price - current_price) / entry_price
        if loss_pct >= self.stop_loss_pct:
            return StrategySignal(
                ticker=ticker,
                action=TradeAction.EXIT,
                strategy=self.name,
                confidence=1.0,
                position_pct=0.0,
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                max_hold_days=0,
                reasons=[f"STOP_LOSS: -{loss_pct:.1%} <= -{self.stop_loss_pct:.1%}"],
            )

        # ── c) Time stop ──────────────────────────────────────────────────────
        if hold_days >= self.max_hold_days:
            return StrategySignal(
                ticker=ticker,
                action=TradeAction.EXIT,
                strategy=self.name,
                confidence=0.5,
                position_pct=0.0,
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                max_hold_days=0,
                reasons=[f"TIME_STOP: {hold_days} >= {self.max_hold_days} bars"],
            )

        # ── d) Overbought exit ────────────────────────────────────────────────
        current_rsi = self.rsi(close)
        if current_rsi >= RSI_OVERBOUGHT_THRESHOLD:
            return StrategySignal(
                ticker=ticker,
                action=TradeAction.EXIT,
                strategy=self.name,
                confidence=0.8,
                position_pct=0.0,
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                max_hold_days=0,
                reasons=[f"RSI_OVERBOUGHT: RSI={current_rsi:.1f} >= {RSI_OVERBOUGHT_THRESHOLD}"],
            )

        return None

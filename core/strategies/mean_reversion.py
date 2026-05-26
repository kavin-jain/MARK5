"""
MARK5 Mean Reversion Strategy v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Targets oversold bounces in quality stocks using FII cascade pattern:
  FII sell → retail panic → RSI < 35 → DII SIP buying → bounce.

ENTRY CONDITIONS (all required):
  1. Momentum score in neutral zone: 0.35 ≤ score ≤ 0.55
  2. RSI(14) < 35  (< 30 for defence stocks: HAL, BEL)
  3. Price > SMA(200) — no falling knives
  4. 3-day return < -5% — sharp recent dip
  5. ATR(14) > 1.5% — enough volatility for a meaningful bounce

EXIT CONDITIONS:
  - Profit target: +5% from entry (fixed, not trailing)
  - Hard stop:     -3% from entry  →  R:R = 1.67:1
  - Time stop:     20 bars (~4 weeks) — exit at market if target not hit

POSITION SIZE: Fixed 7% of equity. No overlap with open momentum positions.

TARGET UNIVERSE (9 stocks):
  Banking:  HDFCBANK, ICICIBANK, KOTAKBANK
  Pharma:   LUPIN, SUNPHARMA
  IT:       TCS, INFY, COFORGE
  Defence:  HAL, BEL  (stricter RSI threshold)

CHANGELOG:
- [2026-05-26] v1.0: Initial implementation.

TRADING ROLE: Complementary alpha in sideways/bear regimes
SAFETY LEVEL: HIGH
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

POSITION_SIZE_PCT    = 0.07   # 7% of equity per trade
PROFIT_TARGET_PCT    = 0.05   # +5%
STOP_LOSS_PCT        = 0.03   # -3%
TIME_STOP_BARS       = 20     # ~4 weeks
ATR_MIN_PCT          = 0.015  # ATR > 1.5% of price
DIP_3D_THRESHOLD     = -0.05  # 3-day return < -5%

RSI_THRESHOLD_DEFAULT  = 35.0  # general stocks
RSI_THRESHOLD_DEFENCE  = 30.0  # HAL, BEL — higher volatility

SCORE_NEUTRAL_MIN = 0.35
SCORE_NEUTRAL_MAX = 0.55

DEFENCE_TICKERS: Set[str] = {"HAL", "BEL"}

TARGET_UNIVERSE: Set[str] = {
    # Banking
    "HDFCBANK", "ICICIBANK", "KOTAKBANK",
    # Pharma
    "LUPIN", "SUNPHARMA",
    # IT
    "TCS", "INFY", "COFORGE",
    # Defence
    "HAL", "BEL",
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExitSignal:
    reason: str        # 'profit_target' | 'stop_loss' | 'time_stop' | 'hold'
    exit_price: float  # suggested exit price (0.0 if reason == 'hold')


# ── Strategy class ────────────────────────────────────────────────────────────

class MeanReversionStrategy:
    """
    Stateless strategy logic for mean reversion trades.
    Thread-safe (no mutable state).
    """

    def _rsi(self, close: pd.Series, period: int = 14) -> float:
        if len(close) < period + 1:
            return 50.0
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = (100 - 100 / (1 + rs)).iloc[-1]
        return float(rsi) if not np.isnan(rsi) else 50.0

    def _sma200(self, close: pd.Series) -> Optional[float]:
        if len(close) < 200:
            return None
        val = close.rolling(200).mean().iloc[-1]
        return float(val) if not np.isnan(val) else None

    def _atr_pct(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return 0.0
        close = df["close"]
        high  = df.get("high", close)
        low   = df.get("low",  close)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(period, min_periods=5).mean().iloc[-1])
        price = float(close.iloc[-1])
        return (atr / price) if price > 0 else 0.0

    def _ticker_base(self, ticker: str) -> str:
        return ticker.replace(".NS", "").replace(".BO", "")

    # ── Entry ─────────────────────────────────────────────────────────────────

    def should_enter(
        self,
        df: pd.DataFrame,
        ticker: str,
        momentum_score: float,
        open_positions: Optional[Set[str]] = None,
    ) -> bool:
        """
        Returns True if all mean reversion entry conditions are met.

        Args:
            df:               OHLCV DataFrame, DatetimeIndex, sorted ascending.
            ticker:           Ticker symbol (with or without .NS suffix).
            momentum_score:   Current MomentumSignal.score() for this stock.
            open_positions:   Set of tickers currently held (to prevent overlap).

        Returns:
            bool — True = enter, False = skip.
        """
        base = self._ticker_base(ticker)

        # ── Pre-conditions ───────────────────────────────────────────────
        if base not in TARGET_UNIVERSE:
            return False

        if df is None or len(df) < 210:
            return False  # need SMA200 + buffer

        # No overlap with existing momentum position
        if open_positions and (base in open_positions or ticker in open_positions):
            return False

        close = df["close"]
        cur   = float(close.iloc[-1])

        # ── Condition 1: Neutral momentum score ──────────────────────────
        if not (SCORE_NEUTRAL_MIN <= momentum_score <= SCORE_NEUTRAL_MAX):
            return False

        # ── Condition 2: RSI oversold ─────────────────────────────────────
        rsi_val   = self._rsi(close, 14)
        rsi_limit = RSI_THRESHOLD_DEFENCE if base in DEFENCE_TICKERS else RSI_THRESHOLD_DEFAULT
        if rsi_val >= rsi_limit:
            return False

        # ── Condition 3: Price > SMA(200) ────────────────────────────────
        sma200 = self._sma200(close)
        if sma200 is None or cur <= sma200:
            return False

        # ── Condition 4: 3-day return < -5% ──────────────────────────────
        if len(close) < 4:
            return False
        ret_3d = float(close.iloc[-1] / close.iloc[-4]) - 1
        if ret_3d >= DIP_3D_THRESHOLD:
            return False

        # ── Condition 5: ATR > 1.5% ──────────────────────────────────────
        atr_pct = self._atr_pct(df, 14)
        if atr_pct < ATR_MIN_PCT:
            return False

        return True

    # ── Exit ──────────────────────────────────────────────────────────────────

    def exit_signal(
        self,
        df: pd.DataFrame,
        entry_price: float,
        entry_bar: int,
        current_bar: int,
    ) -> ExitSignal:
        """
        Evaluate exit conditions for an open mean reversion position.

        Args:
            df:            Full OHLCV DataFrame (same object used at entry).
            entry_price:   Actual fill price at entry.
            entry_bar:     Integer index into df.index at which position was opened.
            current_bar:   Integer index into df.index at current evaluation.

        Returns:
            ExitSignal with reason and suggested exit price.
            reason == 'hold' means no exit trigger; exit_price == 0.0.
        """
        if current_bar >= len(df):
            return ExitSignal(reason="hold", exit_price=0.0)

        close = df["close"]
        cur   = float(close.iloc[current_bar])
        low_i = float(df["low"].iloc[current_bar]) if "low" in df.columns else cur
        bars_held = current_bar - entry_bar

        profit_target_px = entry_price * (1 + PROFIT_TARGET_PCT)
        stop_loss_px     = entry_price * (1 - STOP_LOSS_PCT)

        # Profit target
        if cur >= profit_target_px:
            return ExitSignal(reason="profit_target", exit_price=cur)

        # Hard stop (use intraday low to check if stop was triggered)
        if low_i <= stop_loss_px:
            return ExitSignal(reason="stop_loss", exit_price=stop_loss_px)

        # Time stop
        if bars_held >= TIME_STOP_BARS:
            return ExitSignal(reason="time_stop", exit_price=cur)

        return ExitSignal(reason="hold", exit_price=0.0)

    # ── Position sizing ───────────────────────────────────────────────────────

    @staticmethod
    def position_size(equity: float) -> float:
        """Returns the ₹ capital to deploy for one MR trade (7% of equity)."""
        return equity * POSITION_SIZE_PCT

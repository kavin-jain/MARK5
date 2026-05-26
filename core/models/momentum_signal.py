"""
MARK5 Momentum Signal v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━
Composite 7-component momentum scorer for portfolio entry selection.

CHANGELOG:
- [2026-05-26] v2.0: Added sector RS, FII flow component, weekly_aligned().
  Weights rebalanced. All components fall back to 0.5 on missing data.
- [2026-05-10] v1.0: Initial scorer (extracted from portfolio_backtest.py).

Components and weights (sum = 1.0):
  Trend Alignment          0.28  SMA 20/50/200 stack
  Price Momentum           0.28  20d + 63d RS vs NIFTY50 (blended)
  Relative Strength        0.18  20d excess return vs NIFTY50
  Vol-Adjusted Sharpe      0.13  60-day rolling Sharpe proxy
  Sector RS                0.08  stock RS minus median of sector peers
  FII Flow Momentum        0.03  sigmoid(5d net FII / 10000)
  Volume Quality           0.02  5d/20d volume ratio

TRADING ROLE: Entry ranking signal for momentum portfolio
SAFETY LEVEL: HIGH — incorrect scores → wrong entries/exits
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sigmoid(x: float, scale: float = 1.0) -> float:
    """Sigmoid squashing to [0, 1]."""
    return 1.0 / (1.0 + np.exp(-x * scale))


def _safe_rsi(close: pd.Series, period: int = 14) -> float:
    """
    RSI via Wilder's EMA. Returns value in [0, 100].
    Returns 50.0 (neutral) if insufficient data.
    """
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).iloc[-1]
    return float(rsi) if not np.isnan(rsi) else 50.0


def _sma(series: pd.Series, window: int) -> Optional[float]:
    """Last value of simple moving average, or None if insufficient data."""
    if len(series) < window:
        return None
    val = series.rolling(window).mean().iloc[-1]
    return float(val) if not np.isnan(val) else None


# ── Main class ────────────────────────────────────────────────────────────────

class MomentumSignal:
    """
    Stateless composite scorer.  Instantiate once and call score() per bar.
    Thread-safe (no mutable state after init).
    """

    # ── component weights ──────────────────────────────────────────────────
    W_TREND_ALIGN   = 0.28
    W_PRICE_MOM     = 0.28
    W_REL_STR       = 0.18
    W_SHARPE        = 0.13
    W_SECTOR_RS     = 0.08
    W_FII_FLOW      = 0.03
    W_VOLUME        = 0.02

    # sanity check
    _TOTAL_W = (W_TREND_ALIGN + W_PRICE_MOM + W_REL_STR + W_SHARPE
                + W_SECTOR_RS + W_FII_FLOW + W_VOLUME)
    assert abs(_TOTAL_W - 1.0) < 1e-9, f"Weights sum to {_TOTAL_W}, not 1.0"

    def score(
        self,
        df: pd.DataFrame,
        nifty: Optional[pd.Series] = None,
        fii_5d: float = 0.0,
        sector_peers: Optional[List[pd.Series]] = None,
    ) -> float:
        """
        Compute composite momentum score.

        Args:
            df:           OHLCV DataFrame with columns [open, high, low, close, volume].
                          DatetimeIndex, sorted ascending.
            nifty:        NIFTY50 close price series (optional).
            fii_5d:       5-day sum of FII net flow in ₹cr (positive = buying).
            sector_peers: List of close price pd.Series for sector peer stocks.

        Returns:
            float in [0, 1].  Higher = stronger momentum.
            Returns 0.5 (neutral) if df is too short (<25 bars).
        """
        if df is None or len(df) < 25:
            return 0.5

        close  = df["close"]
        volume = df.get("volume", pd.Series(1.0, index=df.index)) if isinstance(df, pd.DataFrame) else pd.Series(1.0)

        # ── 1. Trend Alignment: SMA 20/50/200 stack ───────────────────────
        sma20  = _sma(close, 20)
        sma50  = _sma(close, 50)
        sma200 = _sma(close, 200)
        cur    = float(close.iloc[-1])

        if sma20 is None or sma50 is None:
            trend_score = 0.5
        else:
            alignment = 0
            if cur > sma20:        alignment += 1
            if sma20 > sma50:      alignment += 1
            if sma200 is not None and sma50 > sma200:
                alignment += 1
            trend_score = alignment / 3.0  # 0, 0.33, 0.67, or 1.0

        # ── 2. Price Momentum: 20d + 63d RS vs NIFTY (blended) ────────────
        look20 = min(20, len(close) - 1)
        look63 = min(63, len(close) - 1)

        stock_r20 = (cur / float(close.iloc[-look20]) - 1) if look20 >= 2 else 0.0
        stock_r63 = (cur / float(close.iloc[-look63]) - 1) if look63 >= 10 else 0.0

        nifty_r20 = nifty_r63 = 0.0
        if nifty is not None and len(nifty) >= 2:
            n20 = min(look20, len(nifty) - 1)
            n63 = min(look63, len(nifty) - 1)
            nifty_r20 = float(nifty.iloc[-1]) / float(nifty.iloc[-n20]) - 1 if n20 >= 1 else 0.0
            nifty_r63 = float(nifty.iloc[-1]) / float(nifty.iloc[-n63]) - 1 if n63 >= 1 else 0.0

        excess_20 = stock_r20 - nifty_r20
        excess_63 = stock_r63 - nifty_r63
        # blend: 60% short-term, 40% sustained
        blended_excess = 0.6 * excess_20 + 0.4 * excess_63
        # sigmoid-scale: ±10% excess → score ≈ 0.73 / 0.27
        price_mom_score = _sigmoid(blended_excess * 10.0)

        # ── 3. Relative Strength vs NIFTY50 (20d only) ────────────────────
        rel_str_score = _sigmoid(excess_20 * 10.0)

        # ── 4. Vol-Adjusted Sharpe (60d rolling proxy) ────────────────────
        if len(close) >= 61:
            rets_60 = close.pct_change().dropna().tail(60)
            mu  = float(rets_60.mean())
            sig = float(rets_60.std())
            sharpe_ann = (mu / sig * np.sqrt(252)) if sig > 1e-10 else 0.0
            # sigmoid: Sharpe 1.0 → ~0.73, Sharpe 2.0 → ~0.88, Sharpe -1.0 → ~0.27
            sharpe_score = _sigmoid(sharpe_ann / 2.0)
        else:
            sharpe_score = 0.5

        # ── 5. Sector Relative Strength ────────────────────────────────────
        # Defined as: stock 20d return minus median 20d return of sector peers.
        # Falls back to 0.5 if no peers provided.
        if sector_peers:
            peer_r20s = []
            for peer_close in sector_peers:
                if peer_close is not None and len(peer_close) >= look20:
                    pr = float(peer_close.iloc[-1]) / float(peer_close.iloc[-look20]) - 1
                    peer_r20s.append(pr)
            if peer_r20s:
                sector_median_r20 = float(np.median(peer_r20s))
                sector_excess = stock_r20 - sector_median_r20
                sector_rs_score = _sigmoid(sector_excess * 10.0)
            else:
                sector_rs_score = 0.5
        else:
            sector_rs_score = 0.5

        # ── 6. FII Flow Momentum ───────────────────────────────────────────
        # sigmoid(fii_5d_net / 10000): +₹10,000cr → ~0.73, -₹10,000cr → ~0.27
        fii_score = _sigmoid(float(fii_5d) / 10_000.0)

        # ── 7. Volume Quality (5d/20d ratio) ──────────────────────────────
        if len(volume) >= 20:
            vol_5d  = float(volume.iloc[-5:].mean())
            vol_20d = float(volume.iloc[-20:].mean())
            vol_ratio = vol_5d / (vol_20d + 1e-10)
            # Normalise: ratio 1.0 → 0.5 (neutral), 2.0 → ~0.73, 0.5 → ~0.27
            vol_score = _sigmoid((vol_ratio - 1.0) * 2.0)
        else:
            vol_score = 0.5

        # ── Composite ─────────────────────────────────────────────────────
        composite = (
            self.W_TREND_ALIGN * trend_score
            + self.W_PRICE_MOM   * price_mom_score
            + self.W_REL_STR     * rel_str_score
            + self.W_SHARPE      * sharpe_score
            + self.W_SECTOR_RS   * sector_rs_score
            + self.W_FII_FLOW    * fii_score
            + self.W_VOLUME      * vol_score
        )
        return float(np.clip(composite, 0.0, 1.0))

    # ── weekly_aligned ────────────────────────────────────────────────────────

    def weekly_aligned(self, df: pd.DataFrame) -> bool:
        """
        Resample daily OHLCV to 5-bar synthetic weekly bars.
        Returns True if at least 2 of 3 weekly SMA conditions hold:
          (1) price > weekly SMA4  (≈ SMA20 daily)
          (2) weekly SMA4 > weekly SMA10 (≈ SMA50 daily)
          (3) weekly SMA10 > weekly SMA40 (≈ SMA200 daily)

        Returns True (neutral pass-through) if fewer than 50 bars of data.
        """
        if df is None or len(df) < 50:
            return True  # not enough history to filter — let daily gate decide

        close = df["close"]

        # 5-bar resample: take last close of each 5-bar chunk
        # (avoids calendar dependency — works on already-filtered trading days)
        n = len(close)
        weekly_closes = close.iloc[[(i + 1) * 5 - 1 for i in range(n // 5)]]

        if len(weekly_closes) < 12:
            return True  # insufficient weekly bars

        wc = weekly_closes.reset_index(drop=True)
        w4  = _sma(wc, 4)
        w10 = _sma(wc, 10)
        w40 = _sma(wc, 40)

        if w4 is None or w10 is None:
            return True

        cur_w = float(wc.iloc[-1])
        alignment = 0
        if cur_w > w4:                                        alignment += 1
        if w4 > w10:                                          alignment += 1
        if w40 is not None and w10 > w40:                     alignment += 1

        return alignment >= 2

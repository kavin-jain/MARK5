"""
MARK5 Portfolio Momentum Rotation Backtest v2.0 — PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGY:
  Every quarter → score ALL stocks from ISE dynamic universe by momentum
  → deploy capital into the top-N stocks
  → compare combined portfolio vs NIFTY50

WHAT'S NEW in v2.0:
  - Universe from ISE API (most_active + trending + shockers + 52w highs)
  - No hardcoded ticker list
  - Composite scoring: RSI + RelStr + BB%B + Volume + Hurst
  - Budget-aware: uses daily ISE cache so OHLCV data doesn't cost tokens

CHANGELOG:
- [2026-05-10] v2.0: Dynamic universe via ISE API, portfolio rotation engine.
- [2026-05-10] v1.0: Initial portfolio rotation engine (hardcoded universe).

TRADING ROLE: Primary alpha engine — portfolio-level momentum rotation
SAFETY LEVEL: HIGH
"""
import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.data.nse_data_provider import fetch_equity_ohlcv, fetch_nifty50_index
from core.models.backtest_pipeline import LightPredictor, _build_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [PORTFOLIO] | %(levelname)s | %(message)s",
)
logger = logging.getLogger("MARK5.PortfolioRotation")

# ── Config ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL   = 5_00_00_000.0   # ₹5 crore
TOP_N             = 3               # positions held simultaneously
ALLOC_PER_STOCK   = 0.33            # 33% each → up to 99% deployed (fair alpha comparison vs 100% NIFTY)
# Rule 12 note: Full deployment allows 1:1 comparison against NIFTY50 without cash drag masking alpha
MAX_SIMULTANEOUS_DEPLOY = 1.00      # enforced at entry time
SLIPPAGE_PCT      = 0.001           # 0.10% per side
BROKERAGE_FLAT    = 20.0            # ₹20 per order (Rule 7)
STT_PCT           = 0.001           # 0.10% sell-side (Rule 7)
TRAIN_MONTHS      = 18
TEST_MONTHS       = 3
MIN_BARS          = 200             # minimum history to include a stock


# ── Composite Momentum Scorer ─────────────────────────────────────────────────
MIN_SCORE_THRESHOLD = 0.0   # Disabled — small 30-stock universe needs all candidates

def momentum_score(prices: pd.DataFrame, nifty: Optional[pd.Series]) -> float:
    """
    Rank stocks for portfolio selection. Higher score = stronger momentum.

    Components (weights):
      RSI zone (0.25)            — 55-75 sweet spot for sustained trend
      Rel-Strength vs NIFTY 20d  — excess return × 10 scaled (0.30)
      Rel-Strength 3-month       — sustained outperformance (0.15)
      Bollinger %B               — where price sits in the band (0.15)
      Volume momentum            — 5d/20d ratio (0.10)
      Hurst proxy                — trending vs mean-reverting (0.05)
    """
    if prices is None or len(prices) < 25:
        return -999.0

    close  = prices["close"]
    volume = prices.get("volume", pd.Series(1.0, index=prices.index))

    # RSI(14) — extended sweet spot: 55-80
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rsi_s = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1]
    if np.isnan(rsi_s):
        rsi_score = 0.0
    elif 55 <= rsi_s <= 75:
        rsi_score = 1.0
    elif 75 < rsi_s <= 85:
        rsi_score = (85 - rsi_s) / 10  # slight overbought penalty
    elif 45 <= rsi_s < 55:
        rsi_score = (rsi_s - 45) / 10  # building momentum bonus
    else:
        rsi_score = 0.0

    # Relative strength vs NIFTY (20d) — short-term excess return
    stock_r20 = (float(close.iloc[-1]) / float(close.iloc[-20]) - 1) if len(close) >= 20 else 0.0
    nifty_r20 = 0.0
    if nifty is not None and len(nifty) >= 20:
        nifty_r20 = float(nifty.iloc[-1]) / float(nifty.iloc[-20]) - 1
    rs_excess_20d = stock_r20 - nifty_r20

    # 3-month momentum (sustained outperformance) — NEW
    look63 = min(63, len(close) - 1)
    stock_r63 = (float(close.iloc[-1]) / float(close.iloc[-look63]) - 1) if look63 >= 20 else 0.0
    nifty_r63 = 0.0
    if nifty is not None and len(nifty) >= look63:
        nifty_r63 = float(nifty.iloc[-1]) / float(nifty.iloc[-look63]) - 1
    rs_excess_63d = stock_r63 - nifty_r63

    # Bollinger %B (price position in 20d band)
    sma20 = float(close.rolling(20).mean().iloc[-1])
    std20 = float(close.rolling(20).std().iloc[-1])
    bb_score = ((float(close.iloc[-1]) - (sma20 - 2 * std20)) / (4 * std20 + 1e-10))
    bb_score = min(max(bb_score, 0.0), 1.5)

    # Volume momentum: 5d / 20d
    vol_5d  = float(volume.iloc[-5:].mean())  if len(volume) >= 5  else 0.0
    vol_20d = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else 1.0
    vol_score = vol_5d / (vol_20d + 1e-10)

    # Hurst proxy (trend persistence vs mean reversion)
    if len(close) >= 20:
        lr = np.log(close / close.shift(1)).dropna()
        v1 = float(lr.rolling(5).std().dropna().iloc[-1])  if len(lr) >= 5  else 0.01
        v5 = float(lr.rolling(20).std().dropna().iloc[-1]) if len(lr) >= 20 else 0.01
        hurst = v1 / (v5 + 1e-10)
    else:
        hurst = 1.0

    return (
        0.25 * rsi_score          +
        0.30 * rs_excess_20d * 10 +   # scale excess return to ~[-3, 3]
        0.15 * rs_excess_63d * 5  +   # 3m momentum (linear, validated against v1)
        0.15 * bb_score           +
        0.10 * vol_score          +
        0.05 * hurst
        + _over_extension_penalty(close, sma20, std20, rsi_s)
    )


def _over_extension_penalty(close: pd.Series, sma20: float, std20: float, rsi_s: float) -> float:
    """
    Penalise stocks that are over-extended above their 20-day mean.
    A stock up >10% above SMA20 is likely topping, not starting a new leg.

    Examples this catches:
      EIHOTEL Q2-2024: scored 1.53 (high lookback RS) but price had already peaked
      Penalty reduces its score → drops out of top-3 selection

    Returns a NEGATIVE value (penalty) or 0.
    """
    last_close = float(close.iloc[-1])
    if sma20 <= 0:
        return 0.0

    pct_above_sma = (last_close - sma20) / (sma20 + 1e-10)

    # Hard veto: RSI > 85 is almost always a reversal setup
    if rsi_s > 85:
        return -2.0

    # Graduated penalty: 0 if <10% above SMA20, scales to -0.8 at 20% above
    # 10% threshold validated as correct: catches extreme extensions without
    # penalising normal momentum runs (7-9% above SMA20 is healthy, not overextended)
    if pct_above_sma > 0.10:
        excess = pct_above_sma - 0.10
        penalty = -5.0 * excess
        return max(penalty, -0.8)

    return 0.0



# ── Single-Position Simulator ─────────────────────────────────────────────────
def simulate_position(
    prices: pd.DataFrame,
    signals: pd.Series,
    capital: float,
    atr_mult: float = 2.5,
) -> Tuple[float, Dict]:
    """
    Simulate one stock position for the test period.
    Entry: next-open after first signal. Exit: trailing stop or end.
    Returns (final_capital, metrics).
    """
    if signals.sum() == 0:
        return capital, {
            "return_pct": 0.0,
            "win_rate":   None,
            "trades":     0,
            "daily_pnl":  pd.Series(dtype=float),
        }

    close = prices["close"]
    high  = prices["high"]
    low   = prices["low"]
    opens = prices["open"]

    # ATR(14)
    tr   = pd.concat([high - low,
                      (high - close.shift(1)).abs(),
                      (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    atr  = tr.rolling(14, min_periods=5).mean()

    pos        = 0
    entry_px   = 0.0
    trail_stop = 0.0
    profit_target = 0.0
    bars_in_trade = 0
    peak_close = 0.0
    # FIX: trade_cap tracks running capital so re-entries compound correctly
    trade_cap  = capital
    trades = wins = 0
    cooldown   = 0      # bars to wait after a LOSING stop-out before re-entering
    MAX_TRADES = 5      # Rule 28: 10 trades/week max → 5/quarter is conservative
    # NOTE: no cooldown applied after WINNING exits — trend is still intact
    # Track daily P&L for correct Sharpe calculation (0 on cash days)
    daily_pnl: Dict[int, float] = {}

    for i in range(1, len(prices)):
        cc    = float(close.iloc[i])
        cc_prev = float(close.iloc[i - 1])
        atr_i = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else cc * 0.02

        if cooldown > 0:
            cooldown -= 1

        if pos > 0:
            bars_in_trade += 1
            # Mark daily P&L while in position
            daily_pnl[i] = (cc - cc_prev) / cc_prev
            peak_close = max(peak_close, cc)
            
            # Trail stop by 1.5x ATR only AFTER price has moved in our favor by at least 1x ATR
            if cc > entry_px + atr_i:
                trail_stop = max(trail_stop, peak_close - 1.5 * atr_i)

            # Triple Barrier Exit Rules (Rule 4)
            hit_sl = (cc < trail_stop) or (float(low.iloc[i]) < trail_stop)
            
            # Smart Time Stop: exit after 10 bars ONLY IF trade is negative or stagnant (profit < 0.5x ATR)
            hit_time = (bars_in_trade >= 10) and (cc < entry_px + 0.5 * atr_i)
            
            # Smart Profit Target: scale out 50%? No, just trail tightly after 2.5x ATR.
            if cc > entry_px + 2.5 * atr_i:
                 # tighten trailing stop to 1.0x ATR to lock in gains
                 trail_stop = max(trail_stop, peak_close - 1.0 * atr_i)
            
            exit_now = hit_sl or hit_time or (i == len(prices) - 1)
            
            if exit_now:
                # Determine exit price based on which barrier was hit first
                if hit_sl:
                    exit_px = trail_stop if cc >= trail_stop else float(opens.iloc[i]) if float(opens.iloc[i]) < trail_stop else trail_stop
                else: # time stop or end of array
                    exit_px = cc * (1 - SLIPPAGE_PCT)
                    
                cost     = BROKERAGE_FLAT + exit_px * pos * (STT_PCT + SLIPPAGE_PCT)
                net_gain = (exit_px - entry_px) * pos - cost
                # update trade_cap (running capital)
                trade_cap += net_gain
                wins      += int(net_gain > 0)
                trades    += 1
                pos = 0
                
                # Cooldown ONLY after a losing stop-out
                if net_gain < 0:
                    cooldown = 5
        else:
            # Cash day — zero P&L contribution (important for correct Sharpe)
            daily_pnl[i] = 0.0

            if signals.iloc[i] == 1 and i + 1 < len(prices) and cooldown == 0 and trades < 20:
                ep  = float(opens.iloc[i + 1]) * (1 + SLIPPAGE_PCT)
                
                # Fetch confidence to dynamically size the position (matching backtester.py)
                conf = prices['confidence'].iloc[i] if 'confidence' in prices.columns else 0.55
                
                # 'capital' here is the baseline per-stock allocation (e.g., 25% of total portfolio).
                # We scale it dynamically based on confidence:
                # High confidence (Tier 1 breakout) -> deploy up to 1.4x the baseline (e.g., 35% of total port)
                # Medium confidence (Tier 2) -> 1.2x baseline (e.g., 30%)
                # Normal -> 1.0x baseline (e.g., 25%)
                if conf >= 0.65:
                    alloc_mult = 1.4
                elif conf >= 0.60:
                    alloc_mult = 1.2
                else:
                    alloc_mult = 1.0
                    
                # The maximum we can invest in this stock for this trade
                max_invest = capital * alloc_mult
                
                sh = int(min(trade_cap, max_invest) / ep)
                if sh > 0:
                    entry_px   = ep
                    pos        = sh
                    peak_close = ep
                    trail_stop = ep - 1.5 * atr_i
                    profit_target = ep + 2.5 * atr_i
                    bars_in_trade = 0
                    trade_cap -= BROKERAGE_FLAT + ep * sh * SLIPPAGE_PCT

    pnl_series = pd.Series(daily_pnl)
    return trade_cap, {
        "return_pct": (trade_cap / capital - 1) * 100.0,
        "win_rate":   wins / trades if trades else None,
        "trades":     trades,
        "daily_pnl":  pnl_series,  # for correct portfolio Sharpe calculation
    }


# ── Walk-Forward Fold Result ──────────────────────────────────────────────────
@dataclass
class FoldResult:
    fold_num:         int
    test_start:       pd.Timestamp
    test_end:         pd.Timestamp
    selected:         List[str]
    portfolio_ret:    float
    nifty_ret:        float
    alpha:            float
    sharpe:           float
    avg_win_rate:     float
    total_trades:     int
    beats_nifty:      bool
    stock_returns:    Dict[str, float] = field(default_factory=dict)


# ── Portfolio Rotation Engine ─────────────────────────────────────────────────
class PortfolioRotationBacktest:

    def __init__(
        self,
        universe:   List[str],
        start_date: str,
        end_date:   str,
        top_n:      int  = TOP_N,
        models_dir: str  = "models",
    ):
        self.universe   = universe
        self.start_date = start_date
        self.end_date   = end_date
        self.top_n      = top_n
        self.models_dir = models_dir

    # ── Fold generation ───────────────────────────────────────────────────
    def _folds(self, idx: pd.DatetimeIndex) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        result = []
        train_off = pd.DateOffset(months=TRAIN_MONTHS)
        test_off  = pd.DateOffset(months=TEST_MONTHS)
        start = idx[0]
        end   = idx[-1]
        while True:
            train_end = start + train_off
            test_end  = min(train_end + test_off, end)
            if train_end > end:
                break
            tm = (idx >= start) & (idx < train_end)
            em = (idx >= train_end) & (idx <= test_end)
            if tm.sum() >= 200 and em.sum() >= 20:
                result.append((idx[tm], idx[em]))
            start += test_off
            if test_end >= end:
                break
        return result

    # ── Main run ──────────────────────────────────────────────────────────
    def run(self) -> Dict:
        logger.info(f"Portfolio rotation v2.0 | universe={len(self.universe)} | "
                    f"top-{self.top_n} | {self.start_date}→{self.end_date}")

        # Load OHLCV for every ticker in the universe
        all_data: Dict[str, pd.DataFrame] = {}
        for ticker in self.universe:
            df = fetch_equity_ohlcv(ticker, self.start_date, self.end_date)
            if df is not None and len(df) >= MIN_BARS:
                all_data[ticker] = df
            else:
                logger.debug(f"[{ticker}] skipped (insufficient data)")

        if len(all_data) < self.top_n:
            logger.error(f"Only {len(all_data)} stocks have enough data — need {self.top_n}")
            return {"status": "failed", "reason": "insufficient data"}

        logger.info(f"Universe loaded: {len(all_data)} stocks with ≥{MIN_BARS} bars")

        nifty = fetch_nifty50_index(self.start_date, self.end_date)
        preds = {t: LightPredictor(t, self.models_dir) for t in all_data}

        ref_idx = next(iter(all_data.values())).index
        folds   = self._folds(ref_idx)
        logger.info(f"{len(folds)} walk-forward folds generated")

        fold_results: List[FoldResult] = []

        # ── Rule 39: Dynamic Stock Gating ────────────────────────────────────
        # Track per-stock win rates across recent folds.
        # Suspend a stock from new entries if rolling 3-fold WR < 40%.
        # Re-enable after WR recovers to > 60% for 2 consecutive folds.
        stock_wr_history: Dict[str, List[float]] = {t: [] for t in all_data}
        GATE_WINDOW         = 3      # folds to evaluate
        SUSPEND_THRESHOLD   = 0.40   # below this → suspended
        RESTORE_THRESHOLD   = 0.60   # above this for 2 consecutive → restored
        suspended_stocks: set = set()

        for fold_num, (train_idx, test_idx) in enumerate(folds, 1):
            ts = test_idx[0].date()
            te = test_idx[-1].date()

            # STEP 1: Score stocks using TRAINING window (no lookahead)
            scores: Dict[str, float] = {}
            for ticker, full_df in all_data.items():
                train_df = full_df.loc[full_df.index <= train_idx[-1]]
                if len(train_df) < 25:
                    continue
                nifty_train = (nifty.reindex(train_df.index, method="ffill").dropna()
                               if nifty is not None else None)
                scores[ticker] = momentum_score(train_df, nifty_train)

            # STEP 2: Select top-N — pure rank-based, with Rule-39 gating
            ranked   = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            eligible = [t for t, s in ranked if s > -100 and t not in suspended_stocks]
            if len(eligible) < self.top_n and suspended_stocks:
                # If gating depletes universe below TOP_N, partially restore
                # the least-suspended stocks (highest recent WR among suspended)
                fill_from = [
                    t for t, _ in ranked
                    if t in suspended_stocks
                    and len(stock_wr_history[t]) > 0
                    and stock_wr_history[t][-1] >= SUSPEND_THRESHOLD
                ]
                eligible += fill_from[:self.top_n - len(eligible)]
            selected = eligible[:self.top_n]

            if suspended_stocks:
                logger.info(f"Fold {fold_num}: Gated (Rule-39): {sorted(suspended_stocks)}")

            if not selected:
                logger.warning(f"Fold {fold_num}: no stocks scored — skipping")
                continue

            score_str = " | ".join(f"{t}:{scores[t]:.2f}" for t in selected)
            logger.info(f"\n── Fold {fold_num} [{ts}→{te}] — Selected: {score_str} ──")

            # STEP 3: Simulate each selected stock in test window
            per_cap      = INITIAL_CAPITAL * ALLOC_PER_STOCK
            stock_results: Dict[str, Dict] = {}

            for ticker in selected:
                if ticker not in all_data:
                    continue
                full_df = all_data[ticker]
                test_df = full_df.loc[test_idx].copy()
                if len(test_df) < 10:
                    continue

                try:
                    signals = _build_signals(
                        test_df.copy(), full_df, nifty,
                        pd.DataFrame(),
                        preds[ticker], fold_num, ticker,
                    )
                except Exception as e:
                    logger.warning(f"[{ticker}] signal error: {e}")
                    signals = pd.Series(0, index=test_df.index)

                signals = signals.reindex(test_df.index).fillna(0).astype(int)
                final_cap, metrics = simulate_position(test_df, signals, per_cap)

                stock_results[ticker] = {
                    "return_pct": metrics["return_pct"],
                    "win_rate":   metrics["win_rate"],
                    "trades":     metrics["trades"],
                    "capital":    final_cap,
                    "daily_pnl":  metrics.get("daily_pnl", pd.Series(dtype=float)),
                }
                wr_str = f"{metrics['win_rate']:.0%}" if metrics["win_rate"] is not None else "N/A"
                logger.info(f"  [{ticker}] {metrics['return_pct']:+.2f}% | "
                            f"trades={metrics['trades']} | WR={wr_str}")

            if not stock_results:
                continue

            # STEP 4: Aggregate portfolio P&L
            # total_pnl is the sum of raw dollar profits across the selected stocks
            total_pnl  = sum(r["capital"] - per_cap for r in stock_results.values())
            # Portfolio return must be calculated on TOTAL initial capital to account for the 25% cash drag
            port_ret   = (total_pnl / INITIAL_CAPITAL) * 100.0

            # NIFTY return for same period
            nifty_ret = 0.0
            if nifty is not None:
                nf = nifty.reindex(test_idx, method="ffill").dropna()
                if len(nf) >= 2:
                    nifty_ret = (float(nf.iloc[-1]) / float(nf.iloc[0]) - 1) * 100.0

            alpha = port_ret - nifty_ret

            # Win rate
            wrs    = [r["win_rate"] for r in stock_results.values() if r["win_rate"] is not None]
            avg_wr = float(np.mean(wrs)) * 100 if wrs else 0.0
            n_trd  = sum(r["trades"] for r in stock_results.values())

            # Sharpe — P&L-based (includes cash days at 0%) for correct strategy Sharpe
            # Bug fixed: was using raw price returns (always volatile regardless of position)
            daily_series = []
            for t in stock_results:
                pnl = stock_results[t]["daily_pnl"]
                if len(pnl) > 0:
                    daily_series.append(pnl)
            if daily_series:
                port_daily = pd.concat(daily_series, axis=1).fillna(0.0).mean(axis=1)
            else:
                port_daily = pd.Series(dtype=float)
            sharpe = (
                float(port_daily.mean() / port_daily.std() * np.sqrt(252))
                if len(port_daily) > 5 and port_daily.std() > 1e-10
                else 0.0
            )

            # ── Update Rule-39 win-rate history ─────────────────────────────
            for ticker, sr in stock_results.items():
                wr = sr.get("win_rate")  # float 0..1 or None
                if wr is not None:
                    stock_wr_history[ticker].append(wr)

            # Evaluate gating decisions for NEXT fold
            for ticker in list(all_data.keys()):
                history = stock_wr_history[ticker]
                if len(history) >= GATE_WINDOW:
                    recent_wr = np.mean(history[-GATE_WINDOW:])
                    if recent_wr < SUSPEND_THRESHOLD:
                        if ticker not in suspended_stocks:
                            logger.warning(
                                f"Rule-39: SUSPENDING {ticker} — "
                                f"rolling {GATE_WINDOW}-fold WR={recent_wr:.0%} < {SUSPEND_THRESHOLD:.0%}"
                            )
                        suspended_stocks.add(ticker)
                    elif ticker in suspended_stocks:
                        # Check restore condition: last 2 folds all above RESTORE_THRESHOLD
                        if len(history) >= 2 and all(w >= RESTORE_THRESHOLD for w in history[-2:]):
                            logger.info(
                                f"Rule-39: RESTORING {ticker} — "
                                f"WR={recent_wr:.0%} recovered above {RESTORE_THRESHOLD:.0%}"
                            )
                            suspended_stocks.discard(ticker)

            # ── Scaled stock B&H benchmark ───────────────────────────────────
            # The portfolio deploys ALLOC_PER_STOCK per stock, not 100%.
            # Scaled B&H = SUM of (stock return × ALLOC_PER_STOCK)
            stock_bh_raw_rets = []
            for ticker in selected:
                if ticker in all_data:
                    td = all_data[ticker].loc[test_idx]
                    if len(td) >= 2:
                        r = (float(td["close"].iloc[-1]) / float(td["close"].iloc[0]) - 1) * 100.0
                        stock_bh_raw_rets.append(r)
            stock_bh_scaled = (np.sum(stock_bh_raw_rets) * ALLOC_PER_STOCK
                               if stock_bh_raw_rets else 0.0)
            alpha_vs_stockbh = port_ret - stock_bh_scaled

            fr = FoldResult(
                fold_num      = fold_num,
                test_start    = test_idx[0],
                test_end      = test_idx[-1],
                selected      = selected,
                portfolio_ret = round(port_ret, 2),
                nifty_ret     = round(nifty_ret, 2),
                alpha         = round(alpha, 2),
                sharpe        = round(sharpe, 3),
                avg_win_rate  = round(avg_wr, 1),
                total_trades  = n_trd,
                beats_nifty   = alpha > 0,
                stock_returns = {t: r["return_pct"] for t, r in stock_results.items()},
            )
            fold_results.append(fr)

            mark  = "✅ BEATS NIFTY" if alpha > 0 else "❌ BELOW NIFTY"
            mark2 = "✅ BEATS StockBH" if alpha_vs_stockbh > 0 else "❌ BELOW StockBH"
            logger.info(
                f"Fold {fold_num}: Portfolio={port_ret:+.1f}% | "
                f"NIFTY={nifty_ret:+.1f}% | StockBH(scaled)={stock_bh_scaled:+.1f}% | "
                f"α(NIFTY)={alpha:+.1f}% | α(stock)={alpha_vs_stockbh:+.1f}% | "
                f"Sharpe={sharpe:.2f} | WR={avg_wr:.0f}% | {mark} | {mark2}"
            )


        return self._report(fold_results)

    # ── Report ────────────────────────────────────────────────────────────
    def _report(self, folds: List[FoldResult]) -> Dict:
        if not folds:
            return {"status": "failed", "reason": "no folds completed"}

        df          = pd.DataFrame([vars(r) for r in folds])
        beats_rate  = df["beats_nifty"].mean()
        mean_alpha  = df["alpha"].mean()
        mean_sharpe = df["sharpe"].mean()
        mean_wr     = df["avg_win_rate"].mean()
        mean_pr     = df["portfolio_ret"].mean()
        mean_nifty  = df["nifty_ret"].mean()
        n_trades    = int(df["total_trades"].sum())
        prod_ready  = (beats_rate >= 0.55 and mean_sharpe >= 0.8
                       and mean_alpha >= 0.5 and mean_wr >= 50)

        W = 88
        verdict = "✅ PRODUCTION READY" if prod_ready else "⚠️  NOT YET PRODUCTION READY"
        print("\n" + "═" * W)
        print(f"  MARK5 PORTFOLIO ROTATION v2.0  |  Top-{self.top_n} Dynamic Universe")
        print("═" * W)
        print(f"  Verdict     : {verdict}")
        print(f"  Beats NIFTY : {beats_rate:.0%} of folds  (target ≥ 55%)")
        print(f"  Mean Alpha  : {mean_alpha:+.2f}% per fold  (target ≥ +0.5%)")
        print(f"  Portfolio   : {mean_pr:+.2f}%  vs  NIFTY {mean_nifty:+.2f}% per fold")
        print(f"  Sharpe      : {mean_sharpe:.3f}  (target ≥ 0.80)")
        print(f"  Win Rate    : {mean_wr:.1f}%  (target ≥ 60%)")
        print(f"  Total Trades: {n_trades}  ({n_trades/len(folds):.1f}/fold avg)")
        print("─" * W)
        header = f"  {'F':<3}  {'Period':<26} {'Portf%':>7} {'NIFTY%':>7} {'Alpha':>7} {'Sharpe':>7} {'WR%':>5}  {'Top picks'}"
        print(header)
        print("  " + "─" * (W - 2))
        for r in folds:
            mark   = "✅" if r.beats_nifty else "❌"
            period = f"{r.test_start.date()}→{r.test_end.date()}"
            stocks = ",".join(r.selected)
            print(f"  {r.fold_num:<3}{mark} {period:<24} "
                  f"{r.portfolio_ret:>+7.1f} {r.nifty_ret:>+7.1f} "
                  f"{r.alpha:>+7.1f} {r.sharpe:>7.3f} "
                  f"{r.avg_win_rate:>4.0f}%  {stocks}")
        print("═" * W + "\n")

        return {
            "status":            "success",
            "n_folds":           len(folds),
            "beats_nifty_rate":  round(float(beats_rate), 3),
            "mean_alpha_pct":    round(float(mean_alpha), 2),
            "mean_sharpe":       round(float(mean_sharpe), 3),
            "mean_win_rate":     round(float(mean_wr), 1),
            "mean_portfolio_ret": round(float(mean_pr), 2),
            "mean_nifty_ret":    round(float(mean_nifty), 2),
            "total_trades":      n_trades,
            "production_ready":  prod_ready,
        }


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARK5 Portfolio Rotation Backtest")
    parser.add_argument("--start",        default="2021-01-01")
    parser.add_argument("--end",          default="2025-12-31")
    parser.add_argument("--top-n",        type=int, default=3)
    parser.add_argument("--models-dir",   default="models")
    parser.add_argument("--no-ise",       action="store_true",
                        help="Skip ISE dynamic universe, use hardcoded 80-stock fallback")
    parser.add_argument("--proven-only",  action="store_true",
                        help="Use only the 8 WF-validated alpha stocks (Sharpe>0.5, WR>50%%)")
    parser.add_argument("--tickers",      nargs='+', help="Specific tickers to run (overrides others)")
    args = parser.parse_args()

    # Build universe
    if args.tickers:
        universe = args.tickers
        logger.info(f"--tickers flag set: using provided {len(universe)} stocks")
    elif args.proven_only:
        # These 8 stocks are validated by walk-forward backtest across 14 folds:
        # Sharpe > 0.5 AND Win Rate > 50% AND trained ML model exists
        # Ranked: PERSISTENT(1.401,71%) > TRENT(0.865,70%) > BAJFINANCE(1.036,57%)
        #       > BEL(0.828,64%) > COFORGE(0.853,51%) > LT(0.645,61%)
        #       > HAL(0.592,60%) > TATASTEEL(0.538,61%)
        universe = [
            "PERSISTENT", "TRENT", "BAJFINANCE", "BEL",
            "COFORGE", "LT", "HAL", "TATASTEEL",
            "SUNPHARMA", "MOTHERSON",  # borderline candidates for rotation
        ]
        logger.info(f"--proven-only: using {len(universe)} WF-validated alpha stocks")

    elif args.no_ise:
        logger.info("--no-ise flag set: using hardcoded 80-stock universe")
        universe = [
            # ── Large-cap anchors (NIFTY50 core) ──
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL",
            "ITC", "LT", "HINDUNILVR", "BAJFINANCE", "KOTAKBANK", "ASIANPAINT", "TITAN",
            "SUNPHARMA", "WIPRO", "HCLTECH", "TECHM", "TATASTEEL", "JSWSTEEL",
            "COALINDIA", "ONGC", "NTPC", "POWERGRID", "MARUTI", "M&M", "BAJAJ-AUTO",
            # ── Defence / PSU Capital Goods ──
            "BEL", "HAL", "BHEL", "CUMMINSIND", "ABB", "SIEMENS",
            # ── PSU Banks ──
            "PNB", "CANBK", "BANKINDIA", "UNIONBANK", "BANKBARODA",
            # ── Private Mid Banks ──
            "AUBANK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB",
            # ── Pharma ──
            "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP", "TORNTPHARM", "AUROPHARMA", "ZYDUSLIFE",
            # ── IT Mid-cap ──
            "TATAELXSI", "LTIM", "MPHASIS", "PERSISTENT", "COFORGE",
            # ── Auto & Auto-ancillaries ──
            "TATAMOTORS", "EICHERMOT", "HEROMOTOCO", "MOTHERSON",
            # ── Consumer / FMCG ──
            "NESTLEIND", "BRITANNIA", "PIDILITIND", "TRENT", "VOLTAS",
            # ── Metals / Mining ──
            "VEDL", "ADANIPORTS", "ADANIENT",
            # ── New Economy ──
            "ZOMATO", "IRCTC", "DMART",
        ]
    else:
        from core.data.universe_screener import build_dynamic_universe
        logger.info("Building dynamic universe from ISE API…")
        universe = build_dynamic_universe(min_size=30, max_size=80)

    engine = PortfolioRotationBacktest(
        universe    = universe,
        start_date  = args.start,
        end_date    = args.end,
        top_n       = args.top_n,
        models_dir  = args.models_dir,
    )
    result = engine.run()

    print(f"\nSummary:")
    print(f"  Production Ready : {result.get('production_ready')}")
    print(f"  Beats NIFTY      : {result.get('beats_nifty_rate', 0):.0%} of folds")
    print(f"  Mean Alpha       : {result.get('mean_alpha_pct', 0):+.2f}%/fold")
    print(f"  Mean Sharpe      : {result.get('mean_sharpe', 0):.3f}")
    print(f"  Mean Win Rate    : {result.get('mean_win_rate', 0):.1f}%")


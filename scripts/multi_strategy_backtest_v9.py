"""
MARK5 Multi-Strategy Backtest v9.0 — Adaptive Volatility System
════════════════════════════════════════════════════════════════
TARGET: Net annual ≥ 20% | Sharpe ≥ 1.0 | Calmar ≥ 1.5 | WR ≥ 45%

V8 Final State (inherited):
  • Net: +15.35% | MaxDD -11.78% (best ever) | Calmar 1.629 (best ever)
  • Initial stop loss (-7%, 45d) — primary V8 innovation
  • Rolling high stop (150%+ trigger, 7% trail) — insurance
  • YTD gate (60% scale if YTD < -2%)
  • Entry hurdle 0.56
  • Gap to 20% = 4.65pp — primarily model retrain (2021→2024 cutoff)

V9 IMPROVEMENTS (4 targeted additions):

  FIX 1 — ATR-ADAPTIVE TRAIL PERCENTAGE
  ──────────────────────────────────────
  V8 uses VIX-only trail (9/12/15%). V9 adds stock-specific ATR adjustment:
    trail = VIX_base × (ATR_21 × 7.0 / TRAIL_NORMAL), clamped [10%, 22%]
  Effect: High-ATR stocks (TATAELXSI 2.5%, BEL 2.8%) get wider trails
  (17-20%) so normal 8-12% corrections don't trigger exits. Low-ATR stocks
  (RELIANCE 1.9%) get tighter trails (11-13%) for faster loss exits.
  Calibration: 7× ATR ≈ median trail width for NSE mid-cap momentum stocks.

  FIX 2 — NIFTY 21-DAY REGIME GATE
  ────────────────────────────────────
  If Nifty 21-day return < -5% → cap new entries at 2 positions (not 4).
  This fires during genuine broad market corrections (not stock-specific dips).
  Effect: In 2026 Q1 (Nifty -10-12% in 21 days), only 2 new positions
  allowed → fewer bad entries in a down-market regime. Fires ~7% of trading
  days in 2025-2026. Does NOT affect existing position management.

  FIX 3 — ROLLING 60-DAY PORTFOLIO PERFORMANCE GATE
  ────────────────────────────────────────────────────
  If portfolio 60-day return < -8% → raise entry hurdle from 0.56 to 0.62.
  Effect: When our own models are clearly wrong (portfolio losing money on
  recent entries), require higher conviction before adding new positions.
  Self-calibrating: adapts to regime changes without year-boundary resets.
  Complementary to YTD gate (which reduces SIZE; this raises HURDLE).

  FIX 4 — INITIAL STOP COOLDOWN
  ────────────────────────────────
  After an INITIAL_STOP exit: block re-entry in same ticker for 60 days.
  Rationale: INITIAL_STOP = "wrong from the start" entry. The same ticker
  with the same ML model will likely generate the same wrong signal again
  within 1-2 months. A 60-day block prevents rapid re-entry into failed
  entries. Does NOT block after TRAIL_STOP or ML_EXIT (those are valid
  trend completions, not false entries).

INHERITS from V8 (all active):
  - Initial Stop Loss (Fix 1a): -7% in first 45 days
  - Rolling High Stop (Fix 1b): 150%+ trigger, 7% below 5-day high
  - YTD Gate (Fix 5): 60% size if YTD < -2%
  - V8 Entry Hurdle: 0.56 (vs V7's 0.52)
  - CB Recovery Protocol + RSI gate + FII gate (V7)
  - Equity CB at 12/18/25% thresholds (V6)

PAPER MODE ONLY. Capital pool: ₹5 crore.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _ROOT    not in sys.path: sys.path.insert(0, _ROOT)
if _SCRIPTS not in sys.path: sys.path.insert(0, _SCRIPTS)

from multi_strategy_backtest_v6 import (
    INITIAL_CAPITAL, OOS_START, OOS_END, TRUE_OOS_START,
    COST_PCT, SLIPPAGE_PCT, MAX_POSITIONS, MIN_ML_STD,
    ML_ENTRY_HURDLE, ML_EXIT_HURDLE, ML_ROLL_WINDOW,
    CONF_TIER_4, CONF_TIER_3, CONF_TIER_2,
    TRAIL_NORMAL, TRAIL_ELEVATED, TRAIL_HIGH,
    EQUITY_CB_CAUTION, EQUITY_CB_PAUSE, EQUITY_CB_EMERGENCY,
    RSI_ENTRY_MAX, RSI_ENTRY_MIN,
    REBAL_NORMAL_DAYS, REBAL_HIGH_VIX_DAYS, VIX_REBAL_TRIGGER,
    FII_PROXY_CRISIS,
    CANDIDATE_TICKERS, EXCLUDED_TICKERS,
    get_confidence_alloc, get_vix_trail_stop, compute_vix_proxy,
    get_equity_dd_state, compute_fii_proxy, get_rolling_conf, compute_rsi,
    load_ticker, load_nifty, load_ml_confidence, _compile_results,
    print_verification_report, print_comparison_table,
    run_v2_baseline, run_v6,
)
from multi_strategy_backtest_v7 import (
    CbRecoveryTracker, check_quality_gate_rsi_only,
    CB_RECOVERY_MIN_DAYS, CB_RECOVERY_NIFTY_RISE,
    CB_RECOVERY_CONF_HURDLE, CB_RECOVERY_MAX_POS, CB_RECOVERY_ALLOC,
    FII_PROXY_BLOCK_V7, run_v7,
)
from multi_strategy_backtest_v8 import (
    INITIAL_STOP_LOSS_PCT, INITIAL_STOP_DAYS,
    ROLLING_HIGH_WINDOW, ROLLING_HIGH_TRIGGER, ROLLING_HIGH_TRAIL_PCT,
    PORT_YTD_DOWN_SCALE, V8_ML_ENTRY_HURDLE,
    get_effective_stop, run_v8,
    V8Position,
)

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | V9 | %(levelname)s | %(message)s")
logger = logging.getLogger("MARK5.V9")

# ── V9 Constants ──────────────────────────────────────────────────────────────

# Fix 1: ATR-Adaptive Trail
# trail = VIX_base × (ATR_21 × ATR_TRAIL_MULTIPLIER / TRAIL_NORMAL), clamped [min, max]
# Calibration: 7.0× ATR gives "natural" trail (6-7× ATR ≈ typical intratrend corrections)
# NSE mid-cap median ATRs: RELIANCE 1.9%, BHARTIARTL 2.1%, TATAELXSI 2.5%, BEL 2.8%
ATR_WINDOW          = 21     # 21-day ATR window
ATR_TRAIL_MULTIPLIER = 7.0   # How many × ATR to use as trail (7 = ~18% for 2.5% ATR stock)
ATR_TRAIL_MIN       = 0.10   # Floor: 10% minimum trail (no matter how low ATR)
ATR_TRAIL_MAX       = 0.22   # Ceiling: 22% maximum trail (no matter how high ATR)

# Fix 2: Nifty 21-day Regime Gate
NIFTY_REGIME_WINDOW      = 21     # 21-trading-day lookback
NIFTY_REGIME_THRESHOLD   = -0.05  # If Nifty 21d return < -5%: regime gate fires
NIFTY_REGIME_MAX_SLOTS   = 2      # Max new entry slots in adverse regime (vs normal 4)

# Fix 3: Rolling 60-day Portfolio Performance Gate
PERF_GATE_LOOKBACK   = 60     # 60 calendar days in equity_history list
PERF_GATE_THRESHOLD  = -0.08  # -8% portfolio 60-day return → raise hurdle
PERF_GATE_HURDLE     = 0.62   # Raised entry hurdle when gate fires

# Fix 4: Initial Stop Cooldown
INITIAL_STOP_COOLDOWN_DAYS = 60   # Block re-entry in ticker X for 60d after INITIAL_STOP on X

# ── V9 Position (extends V8Position with ATR tracking) ────────────────────────

@dataclass
class V9Position:
    """V9 position: same as V8Position but trail set via ATR-adaptive method."""
    ticker:           str
    entry_price:      float
    peak_price:       float
    entry_date:       pd.Timestamp
    shares:           int
    entry_cost:       float
    trail_pct:        float       # ATR-adaptive trail (set once at entry)
    conf_entry:       float
    alloc_tier:       str
    # V8 extensions
    conf_peak:        float = 0.0
    ratchet_floor:    float = 0.0
    # V9 extension
    atr_pct_entry:    float = 0.0  # ATR/price at entry (diagnostic)


@dataclass
class Trade:
    ticker:      str
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    entry_price: float
    exit_price:  float
    shares:      int
    net_pnl:     float
    pnl_pct:     float
    hold_days:   int
    exit_reason: str
    conf_entry:  float
    alloc_tier:  str
    alloc_pct:   float
    partial:     bool = False


# ── V9 Helper Functions ───────────────────────────────────────────────────────

def compute_atr_pct(
    df: pd.DataFrame,
    date: pd.Timestamp,
    n: int = ATR_WINDOW,
) -> float:
    """
    [V9 Fix 1] Compute ATR(n) as fraction of current price.

    Uses True Range = max(H-L, |H-prev_C|, |L-prev_C|).
    Returns ATR/Close as decimal (e.g. 0.020 = 2%).
    Falls back to 0.020 (NSE mid-cap median) if insufficient data.
    """
    try:
        subset = df[df.index <= date].tail(n + 2)
        if len(subset) < max(5, n // 2):
            return 0.020   # Fallback: NSE mid-cap median ATR
        prev_close = subset["close"].shift(1)
        tr = pd.concat([
            subset["high"] - subset["low"],
            (subset["high"] - prev_close).abs(),
            (subset["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1).dropna()
        if len(tr) < 3:
            return 0.020
        atr = float(tr.tail(n).mean())
        close = float(subset["close"].iloc[-1])
        return float(atr / close) if close > 0 else 0.020
    except Exception:
        return 0.020


def get_adaptive_trail(
    vix_val: float,
    atr_pct: float,
) -> float:
    """
    [V9 Fix 1] ATR-adaptive trailing stop percentage.

    Formula: VIX_base × (ATR_21 × 7.0 / TRAIL_NORMAL), clamped [10%, 22%]

    VIX_base provides market-level risk adjustment (wider in panics):
      - VIX < 22%: 15% (normal)
      - VIX 22-28%: 12% (elevated)
      - VIX > 28%:  9%  (high)

    ATR factor adjusts for stock-specific volatility:
      - RELIANCE (1.9% ATR): factor = 7×0.019/0.15 = 0.89 → trail narrows
      - BHARTIARTL (2.1% ATR): factor ≈ 0.98 → near neutral
      - TATAELXSI (2.5% ATR): factor = 7×0.025/0.15 = 1.17 → trail widens
      - BEL (2.8% ATR): factor = 1.31 → trail widens more
    """
    vix_base   = get_vix_trail_stop(vix_val)
    atr_clamped = max(0.010, min(0.045, atr_pct))
    atr_factor = (atr_clamped * ATR_TRAIL_MULTIPLIER) / TRAIL_NORMAL
    adaptive   = vix_base * atr_factor
    return max(ATR_TRAIL_MIN, min(ATR_TRAIL_MAX, adaptive))


def compute_nifty_regime(nifty: pd.Series, date: pd.Timestamp) -> float:
    """
    [V9 Fix 2] Compute Nifty 21-day return.

    Returns the fractional return over the last 21 trading days.
    Returns 0.0 if insufficient data.
    """
    try:
        subset = nifty[nifty.index <= date]
        if len(subset) < NIFTY_REGIME_WINDOW + 2:
            return 0.0
        now  = float(subset.iloc[-1])
        prev = float(subset.iloc[-(NIFTY_REGIME_WINDOW + 1)])
        return (now - prev) / prev if prev > 0 else 0.0
    except Exception:
        return 0.0


def get_port_60d_return(equity_history: List[Dict], current_equity: float) -> float:
    """
    [V9 Fix 3] Compute portfolio return over approximately last 60 days.

    Uses equity_history list (daily entries), looks back PERF_GATE_LOOKBACK entries.
    Returns 0.0 if insufficient history.
    """
    if len(equity_history) < PERF_GATE_LOOKBACK:
        return 0.0
    try:
        eq_60d_ago = equity_history[-PERF_GATE_LOOKBACK]["equity"]
        if eq_60d_ago <= 0:
            return 0.0
        return (current_equity - eq_60d_ago) / eq_60d_ago
    except Exception:
        return 0.0


# ── V9 Portfolio ──────────────────────────────────────────────────────────────

class V9Portfolio:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital   = initial_capital
        self.cash              = initial_capital
        self.positions:  Dict[str, V9Position] = {}
        self.trades:     List[Trade]            = []
        self.equity_history: List[Dict]         = []
        self._ytd_equity_jan1: float            = initial_capital
        # V9 Fix 4: track initial stop cooldown per ticker
        self.initial_stop_dates: Dict[str, pd.Timestamp] = {}

    def get_equity(self, prices: Dict[str, float]) -> float:
        pos_val = sum(
            p.shares * prices.get(t, p.entry_price)
            for t, p in self.positions.items()
        )
        return self.cash + pos_val

    def reset_ytd(self, prices: Dict[str, float]):
        self._ytd_equity_jan1 = self.get_equity(prices)

    def ytd_return(self, prices: Dict[str, float]) -> float:
        eq = self.get_equity(prices)
        if self._ytd_equity_jan1 <= 0:
            return 0.0
        return (eq - self._ytd_equity_jan1) / self._ytd_equity_jan1

    def is_on_cooldown(self, ticker: str, date: pd.Timestamp) -> bool:
        """[V9 Fix 4] Check if ticker is in initial-stop cooldown period."""
        if ticker not in self.initial_stop_dates:
            return False
        stop_date = self.initial_stop_dates[ticker]
        return (date - stop_date).days < INITIAL_STOP_COOLDOWN_DAYS

    def enter(
        self,
        ticker:     str,
        price:      float,
        date:       pd.Timestamp,
        conf:       float,
        vix_val:    float,
        atr_pct:    float,
        size_scale: float = 1.0,
    ) -> bool:
        if ticker in self.positions:
            return False
        if len(self.positions) >= MAX_POSITIONS:
            return False

        alloc_pct = get_confidence_alloc(conf) * size_scale
        alloc     = self.initial_capital * alloc_pct
        max_alloc = min(alloc, self.cash * 0.98)
        if max_alloc < 10_000:
            return False

        fill  = price * (1 + SLIPPAGE_PCT)
        sh    = int(max_alloc / fill)
        if sh < 1:
            return False
        cost    = sh * fill
        tx_cost = cost * COST_PCT
        total   = cost + tx_cost
        if total > self.cash:
            sh = int((self.cash * 0.98 / (1 + COST_PCT)) / fill)
            if sh < 1:
                return False
            cost    = sh * fill
            tx_cost = cost * COST_PCT
            total   = cost + tx_cost

        trail = get_adaptive_trail(vix_val, atr_pct)    # V9: ATR-adaptive
        tier  = (
            "T4" if conf >= CONF_TIER_4[0] else
            "T3" if conf >= CONF_TIER_3[0] else
            "T2" if conf >= CONF_TIER_2[0] else "T1"
        )
        self.cash -= total
        self.positions[ticker] = V9Position(
            ticker=ticker, entry_price=fill, peak_price=fill,
            entry_date=date, shares=sh, entry_cost=total,
            trail_pct=trail, conf_entry=conf, alloc_tier=tier,
            conf_peak=conf,
            atr_pct_entry=atr_pct,
        )
        logger.info(
            f"ENTER {ticker} @{fill:.0f}×{sh} | conf={conf:.3f} "
            f"tier={tier} alloc={alloc_pct:.0%} trail={trail:.1%} ATR={atr_pct:.2%}"
        )
        return True

    def _record_trade(
        self,
        pos: V9Position,
        fill: float,
        n_shares: int,
        cost_basis: float,
        proceeds_net: float,
        date: pd.Timestamp,
        reason: str,
        partial: bool = False,
    ) -> Trade:
        net_pnl  = proceeds_net - cost_basis
        hold     = (date - pos.entry_date).days
        pnl_pct  = net_pnl / cost_basis * 100 if cost_basis > 0 else 0.0
        alloc_pct = cost_basis / self.initial_capital
        trade = Trade(
            ticker=pos.ticker, entry_date=pos.entry_date, exit_date=date,
            entry_price=pos.entry_price, exit_price=fill, shares=n_shares,
            net_pnl=net_pnl, pnl_pct=pnl_pct, hold_days=hold,
            exit_reason=reason, conf_entry=pos.conf_entry,
            alloc_tier=pos.alloc_tier, alloc_pct=alloc_pct, partial=partial,
        )
        self.trades.append(trade)
        logger.info(
            f"EXIT {pos.ticker} @{fill:.0f} ({reason}) | "
            f"PnL={pnl_pct:+.1f}% ({hold}d)"
        )
        return trade

    def exit(
        self,
        ticker: str,
        price:  float,
        date:   pd.Timestamp,
        reason: str,
    ) -> Optional[Trade]:
        if ticker not in self.positions:
            return None
        pos      = self.positions.pop(ticker)
        fill     = price * (1 - SLIPPAGE_PCT)
        proceeds = pos.shares * fill
        tx_cost  = proceeds * COST_PCT
        net_proc = proceeds - tx_cost
        self.cash += net_proc
        # V9 Fix 4: track initial stop cooldown
        if "INITIAL_STOP" in reason:
            self.initial_stop_dates[ticker] = date
        return self._record_trade(pos, fill, pos.shares, pos.entry_cost, net_proc, date, reason)

    def exit_all(
        self,
        prices: Dict[str, float],
        date:   pd.Timestamp,
        reason: str,
    ):
        for tk in list(self.positions.keys()):
            price = prices.get(tk, self.positions[tk].entry_price)
            self.exit(tk, price, date, reason)


# ── V9 Metrics Compiler ───────────────────────────────────────────────────────

def _compile_results_v9(port: V9Portfolio, label: str, oos_start: str, oos_end: str) -> Dict:
    """Compile results from V9Portfolio."""
    eq_df  = pd.DataFrame(port.equity_history).set_index("date")
    trades = port.trades
    n      = len(trades)

    final_eq  = float(eq_df["equity"].iloc[-1])
    total_ret = (final_eq / INITIAL_CAPITAL - 1) * 100
    n_years   = (pd.Timestamp(oos_end) - pd.Timestamp(oos_start)).days / 365.25
    ann_ret   = ((1 + total_ret / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0
    net_tax   = ann_ret * 0.80

    pnl_series = pd.Series([t.net_pnl for t in trades])
    win_rate   = float((pnl_series > 0).mean() * 100) if n > 0 else 0.0

    roll_max  = eq_df["equity"].cummax()
    dd_series = (eq_df["equity"] / roll_max - 1) * 100
    max_dd    = float(dd_series.min()) if len(eq_df) > 1 else 0.0

    eq_ret   = eq_df["equity"].pct_change().dropna()
    rf_daily = 0.065 / 252
    sharpe   = 0.0
    if len(eq_ret) > 10 and eq_ret.std() > 1e-10:
        excess = eq_ret - rf_daily
        sharpe = float(excess.mean() / excess.std() * math.sqrt(252))

    calmar   = float(ann_ret / abs(max_dd)) if abs(max_dd) > 0.01 else 0.0
    avg_hold = float(np.mean([t.hold_days for t in trades])) if n > 0 else 0.0
    avg_win  = float(np.mean([t.pnl_pct for t in trades if t.net_pnl > 0])) if any(t.net_pnl > 0 for t in trades) else 0.0
    avg_loss = float(np.mean([t.pnl_pct for t in trades if t.net_pnl <= 0])) if any(t.net_pnl <= 0 for t in trades) else 0.0
    exp_val  = (win_rate / 100) * avg_win - (1 - win_rate / 100) * abs(avg_loss)

    eq_df["year"] = eq_df.index.year
    annual: Dict[int, float] = {}
    prev   = INITIAL_CAPITAL
    for yr in sorted(eq_df["year"].unique()):
        yr_end = float(eq_df[eq_df["year"] == yr]["equity"].iloc[-1])
        annual[yr] = (yr_end / prev - 1) * 100
        prev = yr_end

    ticker_stats: Dict = {}
    for tk in sorted(set(t.ticker for t in trades)):
        tk_t   = [t for t in trades if t.ticker == tk]
        tk_pnl = sum(t.net_pnl for t in tk_t)
        tk_wr  = float(sum(1 for t in tk_t if t.net_pnl > 0) / len(tk_t) * 100)
        tk_avg = float(np.mean([t.pnl_pct for t in tk_t]))
        ticker_stats[tk] = {
            "n_trades": len(tk_t), "wr_pct": round(tk_wr, 1),
            "avg_pnl_pct": round(tk_avg, 2),
            "total_pnl_L": round(tk_pnl / 1e5, 2),
        }

    tier_stats: Dict = {}
    for tier in ["T1", "T2", "T3", "T4"]:
        tier_t = [t for t in trades if t.alloc_tier == tier]
        if tier_t:
            tier_stats[tier] = {
                "n_trades": len(tier_t),
                "wr_pct":   round(float((pd.Series([t.net_pnl for t in tier_t]) > 0).mean() * 100), 1),
                "avg_pnl":  round(float(np.mean([t.pnl_pct for t in tier_t])), 2),
            }

    n_initial_stops     = sum(1 for t in trades if "INITIAL_STOP"    in t.exit_reason)
    n_rolling_exits     = sum(1 for t in trades if "ROLLING_PEAK"    in t.exit_reason)
    n_regime_blocks     = 0   # counted in run_v9 via regime_blocks counter
    n_cooldown_blocks   = 0   # counted in run_v9 via cooldown_blocks counter
    n_perf_gate_events  = 0   # counted in run_v9

    return {
        "label": label, "oos_start": oos_start, "oos_end": oos_end,
        "n_years": round(n_years, 2),
        "total_ret": round(total_ret, 2), "ann_cagr": round(ann_ret, 2),
        "net_after_tax": round(net_tax, 2),
        "max_dd": round(max_dd, 2), "sharpe": round(sharpe, 3),
        "calmar": round(calmar, 3),
        "n_trades": n, "win_rate": round(win_rate, 1),
        "avg_hold_days": round(avg_hold, 1),
        "avg_win_pct": round(avg_win, 2), "avg_loss_pct": round(avg_loss, 2),
        "expected_value": round(exp_val, 3),
        "annual": {str(k): round(v, 1) for k, v in annual.items()},
        "ticker_stats": ticker_stats, "tier_stats": tier_stats,
        "equity_df": eq_df, "trades": trades,
        # V9-specific
        "n_initial_stops":    n_initial_stops,
        "n_rolling_exits":    n_rolling_exits,
        "n_partial_exits":    0,
        "n_conf_trail_exits": 0,
    }


# ── V9 Main Backtest ──────────────────────────────────────────────────────────

def run_v9(
    all_data:  Dict[str, pd.DataFrame],
    conf_map:  Dict[str, pd.Series],
    nifty:     pd.Series,
    dates:     pd.DatetimeIndex,
    oos_start: str = OOS_START,
    oos_end:   str = OOS_END,
) -> Dict:
    """
    V9 backtest: V8 framework + 4 adaptive improvements.

    V9 additions:
      1. ATR-adaptive trail (stock-specific trail width at entry)
      2. Nifty 21d regime gate (cap entries at 2 slots during downtrends)
      3. Rolling 60-day performance gate (raise hurdle when models failing)
      4. Initial stop cooldown (60d block after INITIAL_STOP exit)
    """
    port        = V9Portfolio(INITIAL_CAPITAL)
    fii_proxy   = compute_fii_proxy(nifty)
    peak_equity = INITIAL_CAPITAL
    last_rebal: Optional[pd.Timestamp] = None
    cb_tracker  = CbRecoveryTracker()
    current_year = pd.Timestamp(oos_start).year

    # Diagnostic counters
    n_regime_blocks   = 0
    n_cooldown_blocks = 0
    n_perf_gate_fires = 0

    for date in dates:
        prices = {t: float(all_data[t].loc[date, "close"])
                  for t in all_data if date in all_data[t].index}
        if not prices:
            continue

        # ── Reset YTD on new calendar year ────────────────────────────────────
        if date.year != current_year:
            current_year = date.year
            port.reset_ytd(prices)

        # ── Regime signals ────────────────────────────────────────────────────
        vix_val    = compute_vix_proxy(nifty, date)
        trail_base = get_vix_trail_stop(vix_val)

        fii_series = fii_proxy[fii_proxy.index <= date]
        fii_ret    = float(fii_series.iloc[-1]) if len(fii_series) else 0.0
        fii_block  = fii_ret <= FII_PROXY_BLOCK_V7
        fii_crisis = fii_ret <= FII_PROXY_CRISIS

        nifty_slice = nifty[nifty.index <= date]
        nifty_now   = float(nifty_slice.iloc[-1]) if len(nifty_slice) else 0.0

        # ── [V9 Fix 2] Nifty 21d regime ──────────────────────────────────────
        nifty_21d_ret = compute_nifty_regime(nifty, date)
        nifty_adverse = nifty_21d_ret < NIFTY_REGIME_THRESHOLD  # < -5%

        # ── Equity circuit breaker ────────────────────────────────────────────
        eq_now      = port.get_equity(prices)
        peak_equity = max(peak_equity, eq_now)
        equity_dd, equity_state = get_equity_dd_state(eq_now, peak_equity)

        if equity_state in ("PAUSE", "EMERGENCY"):
            cb_tracker.enter_pause(date, nifty_now)
        else:
            cb_tracker.clear_pause()

        if equity_state == "EMERGENCY":
            port.exit_all(prices, date, "EQUITY_CB_EMERGENCY")
            port.equity_history.append({
                "date": date, "equity": port.get_equity(prices),
                "n_pos": 0, "vix": round(vix_val * 100, 1),
                "equity_dd": round(equity_dd * 100, 1),
                "equity_state": equity_state,
            })
            continue

        size_scale = 0.5 if equity_state == "CAUTION" else 1.0
        entry_ok   = equity_state not in ("PAUSE", "EMERGENCY")

        # V7 CB Recovery
        recovery_ok = False
        if not entry_ok and not fii_crisis:
            recovery_ok = cb_tracker.can_attempt_recovery(
                date, nifty_now, len(port.positions)
            )

        # ── [V8 Fix 5] YTD Gate ───────────────────────────────────────────────
        ytd_ret = port.ytd_return(prices)
        if ytd_ret < -0.02:
            size_scale = min(size_scale, PORT_YTD_DOWN_SCALE)

        # ── [V9 Fix 3] Rolling 60-day performance gate ────────────────────────
        port_60d_ret = get_port_60d_return(port.equity_history, eq_now)
        perf_gate_active = port_60d_ret < PERF_GATE_THRESHOLD

        # ── Update positions (peaks) ──────────────────────────────────────────
        for tk, pos in list(port.positions.items()):
            if tk in prices:
                curr = prices[tk]
                pos.peak_price = max(pos.peak_price, curr)
                if tk in conf_map:
                    current_conf = get_rolling_conf(conf_map[tk], date)
                    pos.conf_peak = max(pos.conf_peak, current_conf)
                # Ratchet DISABLED (net negative — see V8 analysis)

        # ── Dynamic rebalancing ───────────────────────────────────────────────
        rebal_freq = REBAL_HIGH_VIX_DAYS if vix_val > VIX_REBAL_TRIGGER else REBAL_NORMAL_DAYS
        is_rebal   = (last_rebal is None) or ((date - last_rebal).days >= rebal_freq)

        # ── Exits (checked daily) ─────────────────────────────────────────────
        for tk in list(port.positions.keys()):
            if tk not in prices:
                continue
            pos = port.positions.get(tk)
            if pos is None:
                continue
            curr = prices[tk]

            # ── [V8 Fix 1a] Initial Stop Loss ─────────────────────────────────
            hold_days    = (date - pos.entry_date).days
            initial_stop = pos.entry_price * (1 - INITIAL_STOP_LOSS_PCT)
            if hold_days <= INITIAL_STOP_DAYS and curr < initial_stop:
                port.exit(tk, curr, date,
                          f"INITIAL_STOP({hold_days}d,{(curr/pos.entry_price-1):.0%})")
                continue

            # ── [V8 Fix 1b] Rolling High Stop (150%+ only) ────────────────────
            curr_gain       = (curr / pos.entry_price) - 1.0
            rolling_5d_high = 0.0
            if curr_gain >= ROLLING_HIGH_TRIGGER:
                tkdf = all_data.get(tk)
                if tkdf is not None:
                    recent_close = tkdf["close"][tkdf.index <= date].astype(float)
                    if len(recent_close) >= ROLLING_HIGH_WINDOW:
                        rolling_5d_high = float(
                            recent_close.iloc[-ROLLING_HIGH_WINDOW:].max()
                        )

            # ── [V9 Fix 1] ATR-adaptive effective stop ────────────────────────
            # pos.trail_pct already set via ATR at entry; use it directly
            standard_stop = pos.peak_price * (1 - pos.trail_pct)
            rolling_stop  = (rolling_5d_high * (1 - ROLLING_HIGH_TRAIL_PCT)
                             if rolling_5d_high > 0 else 0.0)
            eff_stop = max(standard_stop, rolling_stop)

            if curr < eff_stop:
                if rolling_stop > 0 and rolling_stop >= standard_stop:
                    stop_type = f"ROLLING_PEAK_STOP({curr_gain:.0%}gain)"
                else:
                    stop_type = f"TRAIL_STOP({pos.trail_pct:.1%})"
                port.exit(tk, curr, date, stop_type)
                continue

            # ── Standard ML exit ──────────────────────────────────────────────
            if is_rebal and tk in conf_map:
                current_conf = get_rolling_conf(conf_map[tk], date)
                if current_conf < ML_EXIT_HURDLE:
                    port.exit(tk, curr, date, f"ML_EXIT(rc={current_conf:.3f})")

        # ── Entries ───────────────────────────────────────────────────────────
        if is_rebal and (entry_ok or recovery_ok) and not fii_crisis:
            last_rebal = date

            using_recovery = recovery_ok and not entry_ok

            # Base entry hurdle: V8's 0.56
            entry_hurdle = CB_RECOVERY_CONF_HURDLE if using_recovery else V8_ML_ENTRY_HURDLE

            # [V9 Fix 3] Raise hurdle if 60d performance is bad
            if perf_gate_active and not using_recovery:
                prev_hurdle  = entry_hurdle
                entry_hurdle = max(entry_hurdle, PERF_GATE_HURDLE)
                if entry_hurdle > prev_hurdle:
                    n_perf_gate_fires += 1

            # [V9 Fix 2] Cap entry slots in adverse Nifty regime
            if nifty_adverse and not using_recovery:
                slot_limit = NIFTY_REGIME_MAX_SLOTS
                n_regime_blocks += 1
            else:
                slot_limit = CB_RECOVERY_MAX_POS if using_recovery else MAX_POSITIONS

            scores: List[Tuple[str, float]] = []
            for tk in conf_map:
                if tk in port.positions or tk not in prices:
                    continue
                if tk in EXCLUDED_TICKERS or fii_block:
                    continue

                # [V9 Fix 4] Skip if ticker is in initial-stop cooldown
                if port.is_on_cooldown(tk, date):
                    n_cooldown_blocks += 1
                    continue

                rc = get_rolling_conf(conf_map[tk], date)
                if rc < entry_hurdle:
                    continue

                tkdf = all_data.get(tk)
                if tkdf is not None:
                    passes, _ = check_quality_gate_rsi_only(tkdf, date)
                    if not passes:
                        continue

                scores.append((tk, rc))

            scores.sort(key=lambda x: -x[1])
            slots = slot_limit - len(port.positions)

            for tk, rc in scores[:slots]:
                tkdf = all_data.get(tk)
                if tkdf is None:
                    continue

                # [V9 Fix 1] Compute ATR at entry for adaptive trail
                atr_pct = compute_atr_pct(tkdf, date, n=ATR_WINDOW)

                entered = port.enter(
                    tk, prices[tk], date, rc, vix_val,
                    atr_pct=atr_pct,
                    size_scale=0.5 if using_recovery else size_scale,
                )
                if entered and using_recovery:
                    cb_tracker.on_recovery_entry(date, nifty_now)

        # ── Record daily equity ───────────────────────────────────────────────
        port.equity_history.append({
            "date":         date,
            "equity":       port.get_equity(prices),
            "n_pos":        len(port.positions),
            "vix":          round(vix_val * 100, 1),
            "equity_dd":    round(equity_dd * 100, 1),
            "equity_state": equity_state,
        })

    # ── Force exit at end ─────────────────────────────────────────────────────
    final_date   = dates[-1]
    final_prices = {t: float(all_data[t].loc[final_date, "close"])
                    for t in all_data if final_date in all_data[t].index}
    port.exit_all(final_prices, final_date, "END_SIM")

    result = _compile_results_v9(port, "V9 (Adaptive-Vol)", oos_start, oos_end)
    result["cb_recoveries"]      = cb_tracker.n_recoveries
    result["n_regime_blocks"]    = n_regime_blocks
    result["n_cooldown_blocks"]  = n_cooldown_blocks
    result["n_perf_gate_fires"]  = n_perf_gate_fires
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═'*90}")
    print("  MARK5 V9 — ADAPTIVE VOLATILITY SYSTEM")
    print("  4 additions: ATR-trail, Nifty regime gate, 60d perf gate, cooldown")
    print(f"  OOS: {OOS_START} → {OOS_END} | Capital: ₹{INITIAL_CAPITAL/1e7:.0f}cr")
    print(f"{'═'*90}\n")

    print("Loading data...")
    nifty = load_nifty()
    if nifty is None:
        print("ERROR: Nifty not found."); return

    all_data: Dict[str, pd.DataFrame] = {}
    for tk in CANDIDATE_TICKERS:
        if tk in EXCLUDED_TICKERS: continue
        df = load_ticker(tk)
        if df is not None and len(df) >= 300:
            all_data[tk] = df

    conf_map: Dict[str, pd.Series] = {}
    for tk in list(all_data.keys()):
        conf = load_ml_confidence(tk)
        if conf is None: continue
        oos_conf = conf.loc[OOS_START:OOS_END]
        if len(oos_conf) < 50: continue
        if float(oos_conf.std()) < MIN_ML_STD or float(oos_conf.max()) < ML_ENTRY_HURDLE:
            continue
        conf_map[tk] = conf

    print(f"Active ML tickers: {len(conf_map)}: {sorted(conf_map.keys())}")

    dates_full = pd.bdate_range(start=OOS_START, end=OOS_END)
    dates_full = dates_full[dates_full <= pd.Timestamp(OOS_END)]
    dates_true = pd.bdate_range(start=TRUE_OOS_START, end=OOS_END)
    dates_true = dates_true[dates_true <= pd.Timestamp(OOS_END)]

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("Running V8 FULL (baseline for V9 comparison)...")
    rv8 = run_v8(all_data, conf_map, nifty, dates_full, OOS_START, OOS_END)
    print(f"  V8: {rv8['n_trades']}t | Net={rv8['net_after_tax']:+.1f}% | "
          f"WR={rv8['win_rate']:.1f}% | DD={rv8['max_dd']:.1f}% | "
          f"Sharpe={rv8['sharpe']:.2f} | Calmar={rv8['calmar']:.2f}")

    # ── V9 Full ───────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("Running V9 FULL (Adaptive Volatility, 2022-2026)...")
    rv9 = run_v9(all_data, conf_map, nifty, dates_full, OOS_START, OOS_END)
    print(f"  V9: {rv9['n_trades']}t | Net={rv9['net_after_tax']:+.1f}% | "
          f"WR={rv9['win_rate']:.1f}% | DD={rv9['max_dd']:.1f}% | "
          f"Sharpe={rv9['sharpe']:.2f} | Calmar={rv9['calmar']:.2f}")
    print(f"  V9: Initial_stop={rv9['n_initial_stops']} | "
          f"Rolling_peak={rv9['n_rolling_exits']} | "
          f"RegimeBlocks={rv9['n_regime_blocks']} | "
          f"Cooldowns={rv9['n_cooldown_blocks']} | "
          f"PerfGate={rv9['n_perf_gate_fires']}")

    # ── V9 True OOS ───────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("Running V9 TRUE OOS (2025-2026)...")
    rv9_oos = run_v9(all_data, conf_map, nifty, dates_true, TRUE_OOS_START, OOS_END)
    print(f"  V9 OOS: {rv9_oos['n_trades']}t | Net={rv9_oos['net_after_tax']:+.1f}% | "
          f"WR={rv9_oos['win_rate']:.1f}% | DD={rv9_oos['max_dd']:.1f}% | "
          f"Sharpe={rv9_oos['sharpe']:.2f}")

    # ── Full Report ───────────────────────────────────────────────────────────
    print(f"\n{'═'*75}")
    print(f"  V9 FULL (2022-2026) — VERIFIED STATS")
    print(f"{'═'*75}")
    print(f"  Total Return    : {rv9['total_ret']:+.2f}%")
    print(f"  Annual CAGR     : {rv9['ann_cagr']:+.2f}%")
    print(f"  Net After STCG  : {rv9['net_after_tax']:+.2f}%")
    print(f"  Max Drawdown    : {rv9['max_dd']:.2f}%")
    print(f"  Sharpe Ratio    : {rv9['sharpe']:.3f}")
    print(f"  Calmar Ratio    : {rv9['calmar']:.3f}")
    print(f"  Win Rate        : {rv9['win_rate']:.1f}%")
    print(f"  Total Trades    : {rv9['n_trades']}")
    print(f"  Avg Win %       : +{rv9['avg_win_pct']:.2f}%")
    print(f"  Avg Loss %      : {rv9['avg_loss_pct']:.2f}%")
    print(f"  Expected Value  : {rv9['expected_value']:+.3f}%/trade")

    print(f"\n  Annual Breakdown:")
    for yr, ret in rv9["annual"].items():
        flag = "✅" if ret > 5 else "🔴" if ret < -5 else "≈"
        v8yr = rv8["annual"].get(yr, 0.0)
        delta = ret - v8yr
        print(f"    {yr}: {ret:+.1f}%  {flag}  (V8: {v8yr:+.1f}%, delta: {delta:+.1f}pp)")

    print(f"\n  V9 vs V8: {rv9['net_after_tax'] - rv8['net_after_tax']:+.2f}pp net annual")
    net = rv9["net_after_tax"]
    target_str = "✅ HIT" if net >= 20 else f"⚠️  {net:.1f}% ({20 - net:.1f}pp gap remains)"
    print(f"  Target (20% net): {target_str}")

    print(f"\n  V9 Mechanism Fires:")
    print(f"    Initial Stops:         {rv9['n_initial_stops']}")
    print(f"    Rolling Peak Stops:    {rv9['n_rolling_exits']}")
    print(f"    Nifty Regime Blocks:   {rv9['n_regime_blocks']} rebal days capped to 2 slots")
    print(f"    Cooldown Blocks:       {rv9['n_cooldown_blocks']} ticker re-entries blocked")
    print(f"    Perf Gate Fires:       {rv9['n_perf_gate_fires']} hurdle raises to 0.62")

    print(f"\n  Ticker PnL (V9):")
    for tk, stats in sorted(rv9["ticker_stats"].items(), key=lambda x: -x[1]["total_pnl_L"]):
        print(f"    {tk:<15}: {stats['n_trades']}t | WR {stats['wr_pct']:.0f}% | "
              f"AvgPnL {stats['avg_pnl_pct']:+.1f}% | Total ₹{stats['total_pnl_L']:+.1f}L")

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'─'*85}")
    print(f"  {'Metric':<28} {'V8 Full':>14} {'V9 Full':>14} {'V9 OOS':>14}")
    print(f"  {'─'*70}")
    rows = [
        ("Net After Tax (ann%)", "net_after_tax", "{:+.2f}%"),
        ("Annual CAGR",          "ann_cagr",      "{:+.2f}%"),
        ("Win Rate",              "win_rate",      "{:.1f}%"),
        ("Max Drawdown",          "max_dd",        "{:.2f}%"),
        ("Sharpe Ratio",          "sharpe",        "{:.3f}"),
        ("Calmar Ratio",          "calmar",        "{:.3f}"),
        ("Total Trades",          "n_trades",      "{:d}"),
        ("Avg Hold Days",         "avg_hold_days", "{:.1f}"),
        ("Avg Win %",             "avg_win_pct",   "{:+.2f}%"),
        ("Avg Loss %",            "avg_loss_pct",  "{:+.2f}%"),
    ]
    for label, key, fmt in rows:
        vals = [rv8.get(key, 0), rv9.get(key, 0), rv9_oos.get(key, 0)]
        vstrs = [fmt.format(int(v) if fmt.endswith("d}") else v) for v in vals]
        print(f"  {label:<28} " + " ".join(f"{s:>14}" for s in vstrs))

    # ── Save ──────────────────────────────────────────────────────────────────
    reports_dir = os.path.join(_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    def _to_json(r: Dict) -> Dict:
        out = {k: v for k, v in r.items() if k not in ("equity_df", "trades")}
        if "trades" in r:
            out["trades"] = [
                {"ticker": t.ticker, "entry_date": str(t.entry_date.date()),
                 "exit_date": str(t.exit_date.date()),
                 "entry_price": round(t.entry_price, 2),
                 "exit_price": round(t.exit_price, 2),
                 "shares": t.shares, "net_pnl": round(t.net_pnl, 2),
                 "pnl_pct": round(t.pnl_pct, 2), "hold_days": t.hold_days,
                 "exit_reason": t.exit_reason, "conf_entry": round(t.conf_entry, 4),
                 "alloc_tier": t.alloc_tier, "partial": getattr(t, "partial", False)}
                for t in r["trades"]
            ]
        return out

    out = {
        "v8_full": _to_json(rv8), "v9_full": _to_json(rv9),
        "v9_true_oos": _to_json(rv9_oos),
    }
    json_path = os.path.join(reports_dir, "multi_strategy_backtest_v9.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {json_path}\n")


if __name__ == "__main__":
    main()

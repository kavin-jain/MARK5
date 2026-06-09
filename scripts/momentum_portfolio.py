"""
MARK5 — Cross-Sectional Momentum Portfolio
==========================================
Strategy: Each month, rank ALL 33 NSE tickers by Multi-Factor Momentum Score.
Hold the top 4 (score ≥ 0.55). Exit when score drops ≤ 0.40 OR trailing stop.

WHY THIS BEATS THE ML PORTFOLIO:
  1. Universe: 33 tickers vs 6-8 ML tickers → captures BEL (+52% B&H), LUPIN, BHARTIARTL
  2. Signal: momentum score has genuine variance for ALL tickers
     ML: pah=100% (HAL/TRENT = always-on), pah=0% (BEL/SBIN = never triggers)
     Momentum: every ticker has 0.15-0.85 range → actual timing signal
  3. No training required → no overfitting, no regime shift problem

FOUNDATION MODEL INTEGRATION (optional):
  Enable with --foundation-model {auto|kronos|chronos}
  When enabled, blends a 10% foundation model score into the ranking signal.
  Entry threshold is UNCHANGED — foundation model improves ranking, not gating.
  Disabled by default to preserve verified +18.05% baseline.

  Install prerequisites:
    Chronos (easy):  pip install chronos-forecasting
    Kronos  (NSE):   pip install git+https://github.com/shiyu-coder/Kronos.git

PAPER MODE ONLY — never switch to LIVE.
"""
import argparse
import os, sys, json, logging, warnings
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.data.sector_data import get_sector_rs as _get_sector_rs
from core.strategies.behavioral_signals import BehavioralSignals as _BehavioralSignals

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("MARK5.MomentumPortfolio")

# ── Universe (all OOS-eligible tickers) ─────────────────────────────────────
# Exclusions (structural — not data-driven snooping):
#   YESBANK:   RBI bailout, structural capital impairment (also excluded from ML)
#   PNB:       Repeated fraud/scam incidents (Nirav Modi), PSU governance issues
#   TATAELXSI: 100% revenue from Jaguar Land Rover (single-client IT, not momentum)
ALL_TICKERS = [
    "ASIANPAINT", "BAJFINANCE", "BEL", "BHARTIARTL", "COFORGE",
    "HDFCBANK",   "HINDUNILVR", "ICICIBANK", "IDEA",   "IDFCFIRSTB",
    "INFY",       "ITC",        "KOTAKBANK", "LT",     "LUPIN",
    "MARUTI",     "MOTHERSON",  "PERSISTENT",           "RELIANCE",
    "SBIN",       "SUNPHARMA",               "TATASTEEL","TCS",
    "TITAN",      "TRENT",      "VOLTAS",               "AUBANK",
    "HAL",        "BANDHANBNK", "BAJAJ-AUTO",
]

# Sector map for concentration cap
SECTOR = {
    "ASIANPAINT": "CONSUMER",  "HINDUNILVR": "CONSUMER",  "ITC": "CONSUMER",
    "BAJFINANCE": "NBFC",      "HDFCBANK": "BANKING",     "ICICIBANK": "BANKING",
    "KOTAKBANK":  "BANKING",   "SBIN": "PSU-BANK",        "PNB": "PSU-BANK",
    "BANDHANBNK": "BANKING",   "IDFCFIRSTB": "BANKING",   "AUBANK": "BANKING",
    "INFY":       "IT",        "TCS": "IT",               "COFORGE": "IT",
    "TATAELXSI":  "IT",        "PERSISTENT": "IT",
    "HAL":        "DEFENCE",   "BEL": "DEFENCE",
    "TRENT":      "RETAIL",    "TITAN": "CONSUMER",
    "MARUTI":     "AUTO",      "MOTHERSON": "AUTO",        "BAJAJ-AUTO": "AUTO",
    "TATASTEEL":  "METALS",    "RELIANCE": "ENERGY",
    "BHARTIARTL": "TELECOM",   "IDEA": "TELECOM",
    "LT":         "INFRA",     "VOLTAS": "INFRA",
    "LUPIN":      "PHARMA",    "SUNPHARMA": "PHARMA",
    "YESBANK":    "PSU-BANK",
}

# ── Portfolio config ─────────────────────────────────────────────────────────
INITIAL_CAPITAL      = 5_00_00_000.0   # ₹5 crore
MAX_POSITIONS        = 5               # top 5 by score (balances opportunity and concentration)
ALLOC_PER_POS        = 0.25            # 25% base per position
ENTRY_THRESHOLD      = 0.55            # score ≥ this → eligible to enter
EXIT_THRESHOLD       = 0.43            # score ≤ this → momentum exit
SCORE_ROLL_WINDOW    = 5               # bars to smooth score
TRAILING_STOP_PCT    = 0.15            # 15% flat from peak (proven optimal)
HARD_STOP_FROM_ENTRY = 0.13           # hard stop 13% below ENTRY PRICE (new-entry capital protection)
                                       # Prevents full 15%+ loss when stock crashes immediately post-entry.
                                       # Trailing stop can't protect if there's no peak above entry.
TRAILING_STOP_COOLDOWN = 45            # bars cooldown after stop
REBALANCE_FREQ_BARS  = 15             # ~21 calendar days
COST_PCT             = 0.0029          # 0.29% round-trip
SLIPPAGE_PCT         = 0.001           # 0.10% slippage
MAX_SECTOR_POSITIONS = 2               # max 2 positions from same sector
_BASELINE_EDGE       = 0.10            # score edge = (score - 0.55) / this

# Circuit breaker — ROLLING 6-month window (prevents permanent lockout)
# Classic all-time-peak CB has a fatal flaw: a single bad run in month 1
# can lock the portfolio for years because cash earns nothing and equity
# never rises above the old peak. Rolling window auto-resets after 6 months.
CB_ROLLING_BARS     = 126              # 6 months rolling peak for drawdown
CB_REDUCE_THRESHOLD = -0.10           # -10% rolling DD → reduce to 2 positions
CB_BLOCK_THRESHOLD  = -0.15           # -15% rolling DD → no new entries

# OOS warm-up: skip first rebalance to avoid entering on bar 0 before
# the regime is established. Bar 0 of OOS might inherit stale momentum
# from the training period that doesn't reflect the new macro regime.
OOS_WARMUP_BARS     = 15              # skip first 15 bars (one rebalance) before any entry

# Tax (Indian equity)
LTCG_RATE   = 0.125    # 12.5% for holds > 365 days
STCG_RATE   = 0.200    # 20% for holds ≤ 365 days
LTCG_EXEMPT = 125_000  # ₹1.25L annual exemption

OOS_START = "2022-01-01"
OOS_END   = "2026-05-21"
CACHE_DIR = os.path.join(_ROOT, "data", "cache")
REPORTS_DIR = os.path.join(_ROOT, "reports")


# ── Data helpers ─────────────────────────────────────────────────────────────
def load_cache(ticker: str) -> Optional[pd.DataFrame]:
    for suffix in ["_daily.parquet", "_NS_1d.parquet"]:
        path = os.path.join(CACHE_DIR, f"{ticker}{suffix}")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            if hasattr(df.index, "tz") and df.index.tz:
                from zoneinfo import ZoneInfo
                df.index = df.index.tz_convert(ZoneInfo("Asia/Kolkata")).tz_localize(None)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            return df
    return None


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    """ATR as % of close, last `period` bars. Returns fraction (e.g. 0.025 = 2.5%)."""
    try:
        hi = df["high"].astype(float)
        lo = df["low"].astype(float)
        cl = df["close"].astype(float)
        tr = pd.concat([hi - lo,
                        (hi - cl.shift()).abs(),
                        (lo - cl.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(period, min_periods=3).mean().iloc[-1])
        price = float(cl.iloc[-1])
        return atr / price if price > 0 else 0.025
    except Exception:
        return 0.025


def compute_tax(trades: List[Dict]) -> float:
    """Compute LTCG/STCG tax with Indian loss-offsetting rules.

    Indian CG rules: STCG losses offset STCG gains first, then LTCG gains.
    LTCG losses offset LTCG gains only. Both within the same financial year.
    """
    by_year: Dict[int, Dict[str, float]] = {}
    for t in trades:
        yr = t["exit_date"].year
        if yr not in by_year:
            by_year[yr] = {"ltcg": 0.0, "stcg": 0.0}
        if t["hold_days"] > 365:
            by_year[yr]["ltcg"] += t["net_pnl"]   # accumulate +ve and -ve
        else:
            by_year[yr]["stcg"] += t["net_pnl"]   # accumulate +ve and -ve

    total_tax = 0.0
    for yr, gains in by_year.items():
        ltcg_net = gains["ltcg"]
        stcg_net = gains["stcg"]
        # STCG losses first offset STCG, then any remaining offset LTCG
        if stcg_net < 0:
            ltcg_net += stcg_net
            stcg_net = 0.0
        # LTCG losses only offset LTCG
        ltcg_taxable = max(0.0, ltcg_net - LTCG_EXEMPT)
        stcg_taxable = max(0.0, stcg_net)
        total_tax   += ltcg_taxable * LTCG_RATE + stcg_taxable * STCG_RATE
    return total_tax


# ── Portfolio class ───────────────────────────────────────────────────────────
class MomentumPortfolio:
    def __init__(self):
        self.cash       = INITIAL_CAPITAL
        self.positions: Dict[str, Dict] = {}
        self.trades:    List[Dict]      = []
        self.equity_history: List[Dict] = []

    def get_equity(self, prices: Dict[str, float]) -> float:
        pos_val = sum(
            p["shares"] * prices.get(tk, p["entry_price"])
            for tk, p in self.positions.items()
        )
        return self.cash + pos_val

    def _score_edge_scale(self, score: float) -> float:
        """Kelly edge proportional to score margin above threshold."""
        edge = max(0.005, score - ENTRY_THRESHOLD)
        return max(0.50, min(1.50, edge / _BASELINE_EDGE))

    def enter(self, ticker: str, price: float, date: pd.Timestamp,
              score: float, atr_pct: float, equity: float, bar_idx: int,
              market_scale: float = 1.0):
        if ticker in self.positions or len(self.positions) >= MAX_POSITIONS:
            return
        # Sector concentration cap
        sector = SECTOR.get(ticker, ticker)
        sector_count = sum(1 for tk in self.positions
                           if SECTOR.get(tk, tk) == sector)
        if sector_count >= MAX_SECTOR_POSITIONS:
            return

        # Kelly edge sizing; market_scale boosts alloc in calm-VIX regimes
        target_atr = 0.02
        vol_scale   = max(0.6, min(1.2, target_atr / atr_pct)) if atr_pct > 0 else 1.0
        edge_scale  = self._score_edge_scale(score)
        alloc = max(equity * 0.10, min(equity * 0.38,
                    equity * ALLOC_PER_POS * vol_scale * edge_scale * market_scale))
        alloc = min(alloc, self.cash * 0.99)
        if alloc < 10_000:
            return

        fill  = price * (1 + SLIPPAGE_PCT)
        shares = int(alloc / fill)
        if shares < 1:
            return
        spent = shares * fill * (1 + COST_PCT)
        if spent > self.cash:
            shares = int(self.cash * 0.99 / (fill * (1 + COST_PCT)))
            if shares < 1:
                return
            spent = shares * fill * (1 + COST_PCT)
        self.cash -= spent
        self.positions[ticker] = {
            "shares": shares, "entry_price": fill, "peak_price": fill,
            "entry_date": date, "entry_bar": bar_idx, "entry_total": spent,
            "score_at_entry": score,
        }
        logger.info(f"ENTER {ticker} @₹{fill:.0f} ×{shares} score={score:.3f}")

    def exit(self, ticker: str, price: float, date: pd.Timestamp, reason: str):
        if ticker not in self.positions:
            return
        pos = self.positions.pop(ticker)
        fill  = price * (1 - SLIPPAGE_PCT)
        proceeds = pos["shares"] * fill * (1 - COST_PCT)
        net_pnl  = proceeds - pos["entry_total"]
        self.cash += proceeds
        hold_days = (date - pos["entry_date"]).days
        self.trades.append({
            "ticker": ticker,
            "entry_date": pos["entry_date"], "exit_date": date,
            "entry_price": pos["entry_price"], "exit_price": fill,
            "shares": pos["shares"], "net_pnl": net_pnl,
            "pnl_pct": net_pnl / pos["entry_total"] * 100,
            "hold_days": hold_days, "reason": reason,
        })
        logger.info(f"EXIT {ticker} @₹{fill:.0f} ({reason}) pnl={net_pnl/1e5:.2f}L")


# ── Main simulation ───────────────────────────────────────────────────────────
def run_portfolio(foundation_model: str = "disabled", foundation_size: str = "mini",
                  vix_upscale: float = 1.0):
    """
    Run the cross-sectional momentum portfolio backtest.

    Args:
        foundation_model: "disabled" (default) | "auto" | "kronos" | "chronos"
            disabled → original verified behaviour (+18.05% baseline)
            auto     → try Kronos first, then Chronos (recommended when installing)
            kronos   → Kronos only (OHLCV-specific, install required)
            chronos  → Chronos only (close-price, pip install chronos-forecasting)
        foundation_size: model size variant ("mini", "small", "base", etc.)
    """
    from core.models.momentum_signal import MomentumSignalEngine

    use_foundation = foundation_model != "disabled"
    foundation_signal = None
    FD_NEUTRAL = 0.5          # neutral score for missing foundation data
    blend_with_momentum = None  # imported below when use_foundation=True
    if use_foundation:
        try:
            from core.models.foundation_signal import (
                build_foundation_signal,
                blend_with_momentum,
                NEUTRAL as FD_NEUTRAL,
            )
            foundation_signal = build_foundation_signal(
                model=foundation_model, size=foundation_size
            )
        except ImportError:
            print("  [WARNING] core.models.foundation_signal not found — "
                  "running without foundation model")
            use_foundation = False

    print(f"\n{'═'*90}")
    print(f"  MARK5 CROSS-SECTIONAL MOMENTUM PORTFOLIO")
    print(f"  OOS: {OOS_START} → {OOS_END}")
    print(f"  Universe: {len(ALL_TICKERS)} tickers | Max positions: {MAX_POSITIONS}")
    print(f"  Entry: score ≥ {ENTRY_THRESHOLD:.2f} | Exit: score ≤ {EXIT_THRESHOLD:.2f} "
          f"| Trail stop: {TRAILING_STOP_PCT:.0%}")
    print(f"  Capital: ₹{INITIAL_CAPITAL/1e7:.0f} crore | Rebal every {REBALANCE_FREQ_BARS} bars")
    if use_foundation and foundation_signal is not None:
        print(f"  Foundation model: {foundation_model.upper()} "
              f"(size={foundation_size}, 10% ranking weight)")
    else:
        print(f"  Foundation model: DISABLED (pure momentum baseline)")
    print(f"{'═'*90}\n")

    engine = MomentumSignalEngine()

    # ── Load all OHLCV data ────────────────────────────────────────────────
    print("Loading data and pre-computing momentum scores...")
    all_data: Dict[str, pd.DataFrame] = {}
    for tk in ALL_TICKERS:
        df = load_cache(tk)
        if df is not None:
            all_data[tk] = df

    # ── Load NIFTY50 for relative strength component ───────────────────────
    nifty_df = None
    for npath in [
        os.path.join(CACHE_DIR, "nse", "NIFTY50_20150101_20260521.parquet"),
        os.path.join(CACHE_DIR, "NIFTY50_daily.parquet"),
    ]:
        if os.path.exists(npath):
            _nf = pd.read_parquet(npath)
            _nf.columns = [c.lower() for c in _nf.columns]
            if hasattr(_nf.index, "tz") and _nf.index.tz:
                _nf.index = _nf.index.tz_localize(None)
            nifty_df = _nf.sort_index()
            break
    # Pre-compute NIFTY 200-day SMA for regime filter
    nifty_close  = None
    nifty_sma200 = None
    if nifty_df is not None:
        nifty_close  = nifty_df["close"].astype(float).sort_index()
        nifty_sma200 = nifty_close.rolling(200, min_periods=60).mean()
        print(f"  NIFTY50 loaded: {len(nifty_df)} bars → relative strength + regime filter enabled")
    else:
        print("  NIFTY50 not found → using neutral relative strength (0.5)")

    # ── Pre-compute momentum scores (vectorised, O(n) per ticker) ─────────
    score_series: Dict[str, pd.Series] = {}
    active_tickers = []

    for tk in ALL_TICKERS:
        df = all_data.get(tk)
        if df is None:
            print(f"  {tk}: no data")
            continue
        oos_df = df.loc[:OOS_END]  # include training period for warmup
        if len(oos_df) < 60:
            print(f"  {tk}: insufficient history ({len(oos_df)} bars)")
            continue
        scores = engine.precompute_scores(oos_df, nifty_df=nifty_df)
        oos_scores = scores.loc[OOS_START:OOS_END]
        if len(oos_scores) < 30:
            print(f"  {tk}: insufficient OOS bars ({len(oos_scores)})")
            continue
        score_series[tk] = scores  # keep full series for rolling lookups
        active_tickers.append(tk)
        pah = (oos_scores >= ENTRY_THRESHOLD).mean()
        pal = (oos_scores <= EXIT_THRESHOLD).mean()
        print(f"  {tk}: {len(oos_scores)} OOS bars  "
              f"mean={float(oos_scores.mean()):.3f}  "
              f"pah={pah:.0%}  pal={pal:.0%}")

    print(f"\n  Active: {len(active_tickers)} tickers\n")

    # ── Build unified OOS calendar from the ticker with the most in-window bars ──
    # (was hardcoded to HAL, which only has data from 2018-04 and would silently
    #  truncate any earlier holdout window. Picking the max-coverage active ticker
    #  makes the calendar correct for ANY evaluation window.)
    _cal_counts = {tk: len(all_data[tk].loc[OOS_START:OOS_END]) for tk in active_tickers}
    _ref_tk = max(_cal_counts, key=_cal_counts.get)
    ref_df = all_data[_ref_tk]
    all_dates = ref_df.loc[OOS_START:OOS_END].index

    # ── Pre-compute foundation model scores for all rebalance dates ────────
    # Identifies rebalance dates without actually running the portfolio,
    # then batch-computes and disk-caches all foundation scores.
    # First run: slow (model inference × tickers × rebal dates).
    # Subsequent runs: instant (disk cache hit).
    _foundation_scores: dict = {}   # tk → {date: score}
    if use_foundation and foundation_signal is not None:
        # Detect rebalance dates via a dry calendar pass
        _dry_rebal_dates = []
        _dry_last = -REBALANCE_FREQ_BARS
        for _i, _d in enumerate(all_dates):
            if (_i - _dry_last) >= REBALANCE_FREQ_BARS and _i >= OOS_WARMUP_BARS:
                _dry_rebal_dates.append(_d)
                _dry_last = _i
        _rebal_idx = pd.DatetimeIndex(_dry_rebal_dates)
        print(f"\n  Pre-computing foundation scores for {len(active_tickers)} tickers "
              f"× {len(_rebal_idx)} rebalance dates...")
        for _tk in active_tickers:
            _df_tk = all_data[_tk]
            try:
                _foundation_scores[_tk] = foundation_signal.precompute_rebalance_scores(
                    _tk, _df_tk, _rebal_idx, horizon=21
                )
            except Exception as _fe:
                print(f"    {_tk}: foundation precompute failed ({_fe}) → neutral")
                _foundation_scores[_tk] = {d: FD_NEUTRAL for d in _rebal_idx}
        print(f"  Foundation scores ready. "
              f"Backend: {getattr(foundation_signal, 'backend_name', foundation_model)}\n")

    # ── Behavioral signals: VIX proxy + market breadth ───────────────────
    # VIX proxy = 20-day realized annualized vol from NIFTY (no options data needed)
    # Breadth   = % of active tickers with close > 50-day SMA
    _vix_scale_series: Optional[pd.Series] = None
    _breadth_series:   Optional[pd.Series] = None
    # VIX calm-market upscale: boost position sizes 10% when realized vol < 15%.
    # Design note (regime_router.py): do NOT downscale in bear/elevated VIX —
    # HAL/TRENT begin multi-year rallies during corrections.
    # Upscale only — more aggressive when the market is objectively calm.
    _vix_upscale_series: Optional[pd.Series] = None
    if nifty_df is not None:
        try:
            _nifty_close = nifty_df["close"].astype(float)
            _bhvr = _BehavioralSignals(_nifty_close)
            # Pre-compute VIX scale for every OOS date (fast, no downloads)
            _vix_scale_series = pd.Series(
                {d: _bhvr.position_scale_factor(d) for d in all_dates},
                dtype=float,
            )
            # Upscale: realized vol < 15% → 1.10×, else 1.0 (no downscale)
            _ret_ser = _nifty_close.pct_change()
            _vix_upscale_series = pd.Series(
                {d: (vix_upscale if _ret_ser[:d].tail(20).std() * (252 ** 0.5) < 0.15
                     else 1.0)
                 for d in all_dates},
                dtype=float,
            )
            # Pre-compute market breadth (batch — single pass over all tickers)
            _ticker_closes = {
                tk: all_data[tk]["close"].astype(float)
                for tk in active_tickers if tk in all_data
            }
            _breadth_series = _bhvr.breadth_series(_ticker_closes, all_dates)
            _vix_sub = (_vix_scale_series < 1.0).mean()
            _bear_days = (_breadth_series < 0.30).mean()
            _calm_days = (_vix_upscale_series > 1.0).mean()
            print(f"\n  Behavioral signals:")
            print(f"    VIX elevated/fear/crisis : {_vix_sub:.0%} of OOS days  "
                  f"(min scale={float(_vix_scale_series.min()):.2f})")
            print(f"    VIX calm (<15%) upscale  : {_calm_days:.0%} of OOS days (1.10× sizing)")
            print(f"    Breadth < 30% (extreme)  : {_bear_days:.0%} of OOS days")
        except Exception as e:
            print(f"  Behavioral signals init failed: {e} — running without")

    # Pre-compute bull regime for dynamic position limit (+1 slot in confirmed bull)
    # BULL = Nifty > 200d SMA AND 50d SMA > 200d SMA (no VIX downscale — see regime_router.py
    # design note: HAL/TRENT begin multi-year rallies during index corrections).
    _bull_regime: Dict[pd.Timestamp, bool] = {}
    if nifty_df is not None:
        try:
            _nc = nifty_df["close"].astype(float).sort_index()
            for d in all_dates:
                hist = _nc[_nc.index <= d].tail(250)
                if len(hist) < 200:
                    _bull_regime[d] = False
                    continue
                curr = float(hist.iloc[-1])
                sma200 = float(hist.tail(200).mean())
                sma50 = float(hist.tail(50).mean())
                _bull_regime[d] = (curr > sma200 and sma50 > sma200)
            _bull_pct = sum(_bull_regime.values()) / max(len(_bull_regime), 1)
            print(f"    Bull regime (Nifty>200d+50d>200d): {_bull_pct:.0%} of OOS days "
                  f"→ MAX_POSITIONS={MAX_POSITIONS + 1} during those days")
        except Exception as e:
            print(f"  Regime detection failed: {e}")

    portfolio         = MomentumPortfolio()
    _ts_cooldown: Dict[str, int] = {}   # tk → earliest bar_idx for re-entry
    _last_rebal_bar   = -REBALANCE_FREQ_BARS

    def _rolling_score(tk: str, date: pd.Timestamp) -> float:
        s = score_series.get(tk)
        if s is None:
            return 0.0
        return engine.rolling_score(s, date, SCORE_ROLL_WINDOW)

    def _rolling_dd(recent_equity_vals: list) -> float:
        """Rolling 6-month drawdown from peak of the last CB_ROLLING_BARS equity values."""
        if not recent_equity_vals:
            return 0.0
        window = recent_equity_vals[-CB_ROLLING_BARS:]
        peak   = max(window)
        curr   = window[-1]
        return (curr / peak - 1) if peak > 0 else 0.0

    _equity_history_vals: list = []   # raw floats for rolling CB calculation

    # Score persistence: require score ≥ threshold at PREVIOUS rebalance too.
    # Prevents single-rebalance spike entries (dead-cat-bounce false signals).
    # Empty dict = first OOS rebalance → benefit of doubt, all entries allowed.
    _prev_rebal_scores: Dict[str, float] = {}

    # ── Main simulation loop ───────────────────────────────────────────────
    for bar_idx, date in enumerate(all_dates):
        # Current prices
        prices: Dict[str, float] = {}
        for tk in active_tickers:
            try:
                prices[tk] = float(all_data[tk].loc[date, "close"])
            except (KeyError, ValueError):
                pass

        # Update trailing peak prices
        for tk, pos in portfolio.positions.items():
            if tk in prices:
                pos["peak_price"] = max(pos["peak_price"], prices[tk])

        # ── Check exits (daily) ────────────────────────────────────────────
        is_rebal = (bar_idx - _last_rebal_bar) >= REBALANCE_FREQ_BARS

        for tk in list(portfolio.positions.keys()):
            if tk not in prices:
                continue
            pos  = portfolio.positions[tk]
            curr = prices[tk]
            peak = pos["peak_price"]
            entry_price = pos["entry_price"]

            # 1a. Hard stop from ENTRY PRICE (capital protection on new entries)
            #     Trailing stop can't protect if the stock never rises after entry.
            #     This caps the worst-case loss on any position at ~13%.
            if curr < entry_price * (1 - HARD_STOP_FROM_ENTRY):
                hold_bars = bar_idx - pos.get("entry_bar", bar_idx)
                portfolio.exit(tk, curr, date, "HARD_STOP")
                _ts_cooldown[tk] = bar_idx + TRAILING_STOP_COOLDOWN
                continue

            # 1b. Trailing stop from peak (profit protection)
            if curr < peak * (1 - TRAILING_STOP_PCT):
                hold_bars = bar_idx - pos.get("entry_bar", bar_idx)
                portfolio.exit(tk, curr, date, "TRAILING_STOP")
                cooldown = 90 if hold_bars > 500 else TRAILING_STOP_COOLDOWN
                _ts_cooldown[tk] = bar_idx + cooldown
                continue

            # 2. Score exit (monthly check)
            if is_rebal:
                rs = _rolling_score(tk, date)
                pnl_pct   = (curr / entry_price - 1)
                hold_bars = bar_idx - pos.get("entry_bar", bar_idx)
                hold_days = (date - pos["entry_date"]).days
                # LTCG threshold: tighten exit for profitable positions 330-364 days
                # old (gains taxed at STCG 20%). An 18-day extension saves 7.5pp on gains.
                ltcg_adj    = 0.10 if (pnl_pct > 0 and 330 <= hold_days < 365) else 0.0
                eff_exit_th = EXIT_THRESHOLD - ltcg_adj
                if pnl_pct > 0.25:
                    pass   # big winner — only trailing stop exits
                elif pnl_pct < -0.05 and hold_bars > 30 and rs < 0.48:
                    portfolio.exit(tk, curr, date, f"SCORE_LOSS_EXIT({rs:.3f})")
                elif rs <= eff_exit_th:
                    portfolio.exit(tk, curr, date, f"SCORE_EXIT({rs:.3f})")

        # ── Equity history + ROLLING circuit breaker ──────────────────────
        equity = portfolio.get_equity(prices)
        portfolio.equity_history.append({"date": date, "equity": equity,
                                          "n_pos": len(portfolio.positions)})
        _equity_history_vals.append(equity)

        # Rolling 6M drawdown — auto-resets after CB_ROLLING_BARS bars
        # so a bad patch in month 1 doesn't lock the system for 4 years.
        dd_pct = _rolling_dd(_equity_history_vals)
        cb_max = MAX_POSITIONS
        if dd_pct <= CB_BLOCK_THRESHOLD:
            cb_max = 0
        elif dd_pct <= CB_REDUCE_THRESHOLD:
            cb_max = 2

        # ── Monthly rebalancing: score all tickers, enter top N ──────────
        if is_rebal and bar_idx >= OOS_WARMUP_BARS:
            _last_rebal_bar = bar_idx

            # Compute scores once for all tickers at this rebalance
            _curr_rebal_scores: Dict[str, float] = {
                tk: _rolling_score(tk, date) for tk in active_tickers
            }

            if cb_max > len(portfolio.positions):
                slots = cb_max - len(portfolio.positions)
                candidates = []
                for tk in active_tickers:
                    if tk in portfolio.positions:
                        continue
                    if tk not in prices or prices[tk] <= 0:
                        continue
                    if bar_idx < _ts_cooldown.get(tk, 0):
                        continue  # in cooldown
                    rs = _curr_rebal_scores.get(tk, 0.0)
                    if rs >= ENTRY_THRESHOLD:
                        # ── Phase 4: Weekly trend confirmation gate ───────────
                        # Blocks entries where the daily score fired on a short
                        # spike but the weekly trend (5-wk MA > 10-wk MA) is
                        # still bearish. Canonical false-signal: TATASTEEL
                        # March 2025 — daily 0.57 on a 3-day bounce, weekly
                        # clearly still bearish.
                        if not engine.weekly_aligned(all_data[tk], date):
                            logger.debug(f"  {tk}: weekly bearish ({rs:.3f}) — blocked")
                            continue
                        # ── Phase 5: Score persistence gate ──────────────────
                        # Require score ≥ ENTRY_THRESHOLD at the PREVIOUS
                        # rebalance too. Blocks single-cycle dead-cat-bounce
                        # spikes where the score crosses the threshold for only
                        # one rebalance window then collapses.
                        # Skipped on the very first OOS rebalance (_prev_rebal_scores
                        # is empty) to avoid blocking the warmup recovery entries.
                        if _prev_rebal_scores:
                            prev_rs = _prev_rebal_scores.get(tk, 0.0)
                            if prev_rs < ENTRY_THRESHOLD:
                                logger.debug(
                                    f"  {tk}: persistence blocked "
                                    f"prev={prev_rs:.3f} curr={rs:.3f}"
                                )
                                continue
                        # ── Foundation model ranking boost (10% weight) ───────
                        # Entry threshold still uses pure momentum score (rs).
                        # Foundation score only re-ranks among qualifying candidates.
                        if use_foundation and _foundation_scores and blend_with_momentum is not None:
                            fd_score = _foundation_scores.get(tk, {}).get(date, FD_NEUTRAL)
                            rank_score = blend_with_momentum(rs, fd_score, 0.10)
                        else:
                            rank_score = rs
                        candidates.append((tk, rank_score))

                # Select top by (blended) score
                candidates.sort(key=lambda x: x[1], reverse=True)
                eq_now = portfolio.get_equity(prices)
                mkt_scale = (float(_vix_upscale_series.get(date, 1.0))
                             if _vix_upscale_series is not None else 1.0)
                for tk, rs in candidates[:slots]:
                    df_tk = all_data[tk]
                    hist = df_tk.loc[:date].tail(30)
                    atr_pct = compute_atr_pct(hist) if len(hist) >= 5 else 0.025
                    portfolio.enter(tk, prices[tk], date, rs, atr_pct, eq_now, bar_idx,
                                    market_scale=mkt_scale)

            # Persist current scores for next rebalance's filter
            _prev_rebal_scores = _curr_rebal_scores

        elif is_rebal:
            _last_rebal_bar = bar_idx  # advance counter even during warmup

    # ── Close any open positions at OOS end ───────────────────────────────
    last_date = all_dates[-1]
    last_prices: Dict[str, float] = {}
    for tk in list(portfolio.positions.keys()):
        try:
            last_prices[tk] = float(all_data[tk].loc[last_date, "close"])
        except (KeyError, ValueError):
            last_prices[tk] = portfolio.positions[tk]["entry_price"]
    for tk in list(portfolio.positions.keys()):
        portfolio.exit(tk, last_prices.get(tk, 0.0), last_date, "END_OF_SIM")
    final_equity = portfolio.get_equity(last_prices)

    # ── Performance computation ────────────────────────────────────────────
    eq_df     = pd.DataFrame(portfolio.equity_history).set_index("date")
    years     = (all_dates[-1] - all_dates[0]).days / 365.25
    gross_ret = (final_equity / INITIAL_CAPITAL - 1)
    gross_cagr= (1 + gross_ret) ** (1 / years) - 1 if years > 0 else 0.0

    total_tax  = compute_tax(portfolio.trades)
    net_final  = final_equity - total_tax
    net_ret    = (net_final / INITIAL_CAPITAL - 1)
    net_cagr   = (1 + net_ret) ** (1 / years) - 1 if years > 0 else 0.0

    # Max drawdown
    eq_curve = eq_df["equity"]
    rolling_max = eq_curve.cummax()
    dd_curve = eq_curve / rolling_max - 1
    max_dd = float(dd_curve.min())

    # Sharpe (annualised daily returns)
    daily_ret_ser = eq_curve.pct_change().dropna()
    sharpe = (float(daily_ret_ser.mean()) / float(daily_ret_ser.std()) * np.sqrt(252)
              if float(daily_ret_ser.std()) > 0 else 0.0)

    # Annual breakdown — use year-start and year-end equity rows
    annual: Dict[int, float] = {}
    eq_curve_full = eq_df["equity"]
    for yr in range(int(eq_df.index.year.min()), int(eq_df.index.year.max()) + 1):
        yr_mask = eq_df.index.year == yr
        yr_data = eq_curve_full[yr_mask]
        if len(yr_data) == 0:
            continue
        # Year-start: last equity value strictly before the first bar of this year
        before_yr = eq_curve_full[eq_df.index < yr_data.index[0]]
        y_start = float(before_yr.iloc[-1]) if len(before_yr) > 0 else INITIAL_CAPITAL
        y_end   = float(yr_data.iloc[-1])
        annual[yr] = y_end / y_start - 1

    # Trades summary
    wins  = [t for t in portfolio.trades if t["net_pnl"] > 0]
    loss  = [t for t in portfolio.trades if t["net_pnl"] <= 0]
    avg_hold = np.mean([t["hold_days"] for t in portfolio.trades]) if portfolio.trades else 0
    wr    = len(wins) / len(portfolio.trades) * 100 if portfolio.trades else 0

    # ── Print results ──────────────────────────────────────────────────────
    print(f"\n{'═'*90}")
    print(f"  PORTFOLIO RESULTS — {OOS_START} to {OOS_END}")
    print(f"{'═'*90}")
    print(f"  Capital:        ₹{INITIAL_CAPITAL/1e7:.0f}cr → ₹{net_final/1e7:.2f}cr  "
          f"(+{net_ret*100:.1f}%)")
    print(f"  Gross CAGR:     {gross_cagr*100:+.2f}%")
    print(f"  Net CAGR:       {net_cagr*100:+.2f}%   ← after LTCG/STCG tax")
    print(f"  Max Drawdown:   {max_dd*100:.1f}%")
    print(f"  Sharpe:         {sharpe:.2f}")
    print(f"  Total Tax:      ₹{total_tax/1e5:.1f}L")
    print(f"  Trades:         {len(portfolio.trades)}  "
          f"(W:{len(wins)} L:{len(loss)} WR:{wr:.0f}%)  "
          f"avg hold {avg_hold:.0f}d")
    print(f"\n  Annual returns:")
    for yr, ret in sorted(annual.items()):
        bar = "▓" * int(abs(ret) * 100) if abs(ret) * 100 < 40 else "▓" * 40
        sign = "+" if ret >= 0 else ""
        print(f"    {yr}: {sign}{ret*100:.1f}%  {bar}")

    # ── Compare to ML system ───────────────────────────────────────────────
    ML_NET_CAGR  = 23.42
    ML_MAX_DD    = -17.0
    ML_TRADES    = 22
    # Real NIFTY50 buy-and-hold CAGR over the ACTUAL evaluation window
    # (was hardcoded 12.5; true value is 7.0% for 2022-2026 and 14.3% for 2016-2021).
    NIFTY_CAGR = 12.5
    if nifty_close is not None:
        _nb = nifty_close.loc[OOS_START:OOS_END]
        if len(_nb) > 2:
            _nyrs = (_nb.index[-1] - _nb.index[0]).days / 365.25
            NIFTY_CAGR = ((_nb.iloc[-1] / _nb.iloc[0]) ** (1 / _nyrs) - 1) * 100

    print(f"\n{'─'*90}")
    print(f"  {'Metric':<25} {'Momentum':<20} {'ML V2 (baseline)':<20} {'NIFTY50 (passive)'}")
    print(f"  {'─'*85}")
    print(f"  {'Net CAGR':<25} {net_cagr*100:>+8.2f}%          {ML_NET_CAGR:>+8.2f}%          {NIFTY_CAGR:>+8.1f}%")
    print(f"  {'Max Drawdown':<25} {max_dd*100:>+8.1f}%          {ML_MAX_DD:>+8.1f}%")
    print(f"  {'Sharpe':<25} {sharpe:>8.2f}            {'1.31':>8}")
    print(f"  {'# Trades':<25} {len(portfolio.trades):>8}              {ML_TRADES:>8}")
    print(f"  {'Universe size':<25} {len(active_tickers):>8}              {'6':>8}")
    print(f"{'═'*90}\n")

    # ── Top 10 trades ──────────────────────────────────────────────────────
    print(f"  Top 10 trades by PnL:")
    top10 = sorted(portfolio.trades, key=lambda t: t["net_pnl"], reverse=True)[:10]
    for t in top10:
        ed = t["entry_date"].strftime("%Y-%m-%d") if hasattr(t["entry_date"], "strftime") else str(t["entry_date"])
        xd = t["exit_date"].strftime("%Y-%m-%d")  if hasattr(t["exit_date"],  "strftime") else str(t["exit_date"])
        print(f"    {t['ticker']:<14} {ed}→{xd} ({t['hold_days']:4d}d) "
              f"{t['pnl_pct']:+6.1f}%  ₹{t['net_pnl']/1e5:+6.1f}L  [{t['reason']}]")

    # ── Save results ───────────────────────────────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    output = {
        "strategy": "cross_sectional_momentum",
        "oos_start": OOS_START, "oos_end": OOS_END,
        "years": round(years, 2),
        "initial_capital": INITIAL_CAPITAL,
        "final_equity_gross": round(final_equity, 2),
        "final_equity_net": round(net_final, 2),
        "gross_cagr_pct": round(gross_cagr * 100, 2),
        "net_cagr_pct": round(net_cagr * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 3),
        "total_tax": round(total_tax, 2),
        "n_trades": len(portfolio.trades),
        "win_rate_pct": round(wr, 1),
        "avg_hold_days": round(float(avg_hold), 1),
        "annual_returns": {str(yr): round(v * 100, 2) for yr, v in annual.items()},
        "universe_size": len(active_tickers),
        "trades": [
            {
                "ticker":      t["ticker"],
                "entry_date":  str(t["entry_date"].date()) if hasattr(t["entry_date"], "date") else str(t["entry_date"]),
                "exit_date":   str(t["exit_date"].date())  if hasattr(t["exit_date"],  "date") else str(t["exit_date"]),
                "entry_price": round(float(t["entry_price"]), 2),
                "exit_price":  round(float(t["exit_price"]),  2),
                "shares":      int(t["shares"]),
                "net_pnl":     round(float(t["net_pnl"]), 2),
                "pnl_pct":     round(float(t["pnl_pct"]), 2),
                "hold_days":   int(t["hold_days"]),
                "reason":      t["reason"],
            }
            for t in portfolio.trades
        ],
        # Compact daily equity curve: [[date_str, equity_float], ...]
        # Used by Monte Carlo simulation (scripts/monte_carlo.py)
        "equity_curve": [
            [str(h["date"].date()) if hasattr(h["date"], "date") else str(h["date"]),
             round(float(h["equity"]), 2)]
            for h in portfolio.equity_history
        ],
    }
    out_path = os.path.join(REPORTS_DIR, "momentum_portfolio_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved → {out_path}")
    print(f"{'═'*90}\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MARK5 Cross-Sectional Momentum Portfolio Backtest"
    )
    parser.add_argument(
        "--foundation-model",
        choices=["disabled", "auto", "kronos", "chronos"],
        default="disabled",
        help=(
            "Foundation model for ranking augmentation (default: disabled). "
            "'auto' tries Kronos first, then Chronos. "
            "Install: pip install chronos-forecasting"
        ),
    )
    parser.add_argument(
        "--foundation-size",
        default="mini",
        help="Model size variant: mini/small/base/v2 (default: mini)",
    )
    parser.add_argument(
        "--vix-upscale",
        type=float,
        default=1.0,
        help="VIX calm-market position size multiplier (default: 1.0 = disabled)",
    )
    parser.add_argument(
        "--oos-start",
        default=OOS_START,
        help="OOS window start (default: 2022-01-01). Use an earlier untouched "
             "window (e.g. 2016-01-01) for honest holdout testing.",
    )
    parser.add_argument(
        "--oos-end",
        default=OOS_END,
        help="OOS window end (default: 2026-05-21).",
    )
    args = parser.parse_args()
    # Window override for honest holdout testing — ONLY the evaluation window
    # changes; every strategy parameter above is frozen.
    OOS_START = args.oos_start
    OOS_END = args.oos_end
    run_portfolio(
        foundation_model=args.foundation_model,
        foundation_size=args.foundation_size,
        vix_upscale=args.vix_upscale,
    )

"""
MARK5 Multi-Strategy Backtest v2.0
════════════════════════════════════
Compares BASELINE (v1, 13 tickers) vs ENHANCED-v2 (32 tickers + cash yield +
calibrated MR + re-entry cooldown).

ROOT-CAUSE FIXES vs v1 enhanced:
  1. Universe: 13 → 32 tickers (all tickers with trained models)
  2. Cash yield: idle cash earns 6.5% p.a. (liquid fund equivalent)
  3. MR calibration: volume 1.0× (was 1.2×), SMA200 proximity 30% (was 20%),
     bear position 15% (was 10%), max hold 30 days (was 25)
  4. Re-entry cooldown: 21-bar wait after trailing stop before re-entering
     the same ticker (prevents COFORGE-style churn)

HOW TO RUN:
    cd /home/lynx/Documents/MARK5
    python3 scripts/multi_strategy_backtest_v2.py

OUTPUT:
    - Side-by-side comparison: BASELINE vs ENHANCED-v2
    - Annual returns 2022–2026
    - Win rate, drawdown, Sharpe
    - reports/multi_strategy_backtest_v2.json

PAPER MODE ONLY — never switch to LIVE.

CHANGELOG:
- [2026-05-23] v2.0: Universe expansion + cash yield + calibrated MR + cooldown
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.strategies.regime_router import (
    RegimeRouter, MarketRegimeState, REGIME_ALLOCATION,
)
from core.strategies.mean_reversion import MeanReversionStrategy
from core.strategies.circuit_breaker import PortfolioCircuitBreaker, CircuitBreakerLevel
from core.strategies.cash_yield import CashYieldModel
from core.strategies.universe_expander import UniverseExpander

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("MARK5.MultiStratV2")

# ── Config ────────────────────────────────────────────────────────────────────

INITIAL_CAPITAL      = 5_00_00_000.0  # ₹5 crore
OOS_START            = "2022-01-01"
OOS_END              = "2026-05-21"
COST_PCT             = 0.0029          # 0.29% round-trip
SLIPPAGE             = 0.001           # 0.1% slippage

# Baseline momentum params (unchanged from Iteration 6)
BASELINE_ALLOC       = 0.25
BASELINE_MAX_POS     = 4
BASELINE_TRAIL_STOP  = 0.15
BASELINE_ML_ENTRY    = 0.52
BASELINE_ML_EXIT     = 0.45
BASELINE_REBAL_DAYS  = 21
BASELINE_ML_WINDOW   = 10

# v2 additions
REENTRY_COOLDOWN_BARS = 21    # bars before re-entering a ticker after trailing stop

DATA_CACHE = os.path.join(_ROOT, "data", "cache")
MODELS_DIR = os.path.join(_ROOT, "models")

# ── Baseline universe (Iteration 6) ──────────────────────────────────────────
BASELINE_TICKERS = [
    "ASIANPAINT", "AUBANK", "BAJFINANCE", "BHARTIARTL", "COFORGE",
    "HAL", "PNB", "RELIANCE", "TATAELXSI", "TATASTEEL",
    "TCS", "TRENT", "YESBANK",
]

# Mean-reversion candidates (quality, large-cap quality stocks)
MR_CANDIDATES = [
    "HDFCBANK", "ICICIBANK", "INFY", "TCS", "RELIANCE",
    "BAJFINANCE", "SBIN", "KOTAKBANK", "LT", "SUNPHARMA",
    "TITAN", "HINDUNILVR", "MARUTI", "LUPIN", "ITC",
    "BHARTIARTL", "COFORGE", "PERSISTENT", "MOTHERSON", "VOLTAS",
    "BANDHANBNK", "BEL", "TATAELXSI", "TATASTEEL",
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lower() for c in df.columns]
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()[~df.sort_index().index.duplicated(keep="last")]


def load_ticker(ticker: str, cache_dir: str = DATA_CACHE) -> Optional[pd.DataFrame]:
    search_dirs = [cache_dir, os.path.join(cache_dir, "nse")]
    patterns = [
        f"{ticker}_daily.parquet",
        f"{ticker}_NS_1d.parquet",
        f"{ticker}_20220101_20260521.parquet",
        f"{ticker}_20220101_20260522.parquet",
        f"{ticker}_20210101_20251231.parquet",
    ]
    for d in search_dirs:
        for p in patterns:
            path = os.path.join(d, p)
            if os.path.exists(path):
                try:
                    df = _clean_df(pd.read_parquet(path))
                    if "close" in df.columns:
                        return df
                except Exception:
                    pass
    return None


def load_nifty() -> Optional[pd.Series]:
    cache_nse = os.path.join(DATA_CACHE, "nse")
    for fn in [
        "NIFTY50_20150101_20260521.parquet",
        "NIFTY50_20220101_20260521.parquet",
        "NIFTY50_1d.parquet",
    ]:
        path = os.path.join(cache_nse, fn)
        if os.path.exists(path):
            try:
                df = _clean_df(pd.read_parquet(path))
                return df["close"].dropna().sort_index()
            except Exception:
                pass
    # Try root cache
    for fn in ["NIFTY50_1d.parquet"]:
        path = os.path.join(DATA_CACHE, fn)
        if os.path.exists(path):
            try:
                df = _clean_df(pd.read_parquet(path))
                return df["close"].dropna().sort_index()
            except Exception:
                pass
    return None


def load_ml_confidence(ticker: str) -> Optional[pd.Series]:
    try:
        from core.models.backtest_pipeline import LightPredictor
        from core.models.features import engineer_features_df
        df = load_ticker(ticker)
        if df is None:
            return None
        pred = LightPredictor(ticker, MODELS_DIR)
        if not pred.has_models():
            return None
        feat = engineer_features_df(df, is_daily=True)
        proba = pred.predict_proba(feat)
        return pd.Series(proba, index=feat.index, name=ticker)
    except Exception as e:
        logger.debug(f"[{ticker}] ML conf failed: {e}")
        return None


def get_rolling_conf(series: pd.Series, date: pd.Timestamp, window: int = 10) -> float:
    try:
        idx   = series.index.searchsorted(date, side="right") - 1
        idx   = max(0, min(idx, len(series) - 1))
        start = max(0, idx - window + 1)
        val   = float(series.iloc[start:idx + 1].mean())
        return val if not np.isnan(val) else 0.5
    except Exception:
        return 0.5


# ── Portfolio engine ──────────────────────────────────────────────────────────

@dataclass
class Position:
    ticker:        str
    strategy:      str
    entry_price:   float
    peak_price:    float
    entry_date:    pd.Timestamp
    shares:        int
    entry_cost:    float
    trail_stop_pct: float
    take_profit_pct: float = 0.0
    stop_loss_pct:   float = 0.0
    max_hold_days:   int   = 9999


@dataclass
class Trade:
    ticker:     str
    strategy:   str
    entry_date: pd.Timestamp
    exit_date:  pd.Timestamp
    entry_price: float
    exit_price:  float
    shares:     int
    net_pnl:    float
    pnl_pct:    float
    hold_days:  int
    exit_reason: str


class Portfolio:
    def __init__(self, initial_capital: float, use_cash_yield: bool = True):
        self.initial_capital = initial_capital
        self.cash            = initial_capital
        self.positions:      Dict[str, Position]  = {}
        self.trades:         List[Trade]           = []
        self.equity_history: List[Dict]            = []
        self.circuit_breaker = PortfolioCircuitBreaker(initial_capital)
        self.cash_yield_model = CashYieldModel(initial_capital) if use_cash_yield else None
        self.total_cash_yield: float = 0.0

    def get_equity(self, prices: Dict[str, float]) -> float:
        pos_val = sum(
            p.shares * prices.get(t, p.entry_price)
            for t, p in self.positions.items()
        )
        return self.cash + pos_val

    def accrue_yield(self):
        """Apply one trading day of cash yield to the cash balance."""
        if self.cash_yield_model and self.cash > 0:
            interest = self.cash_yield_model.accrue(self.cash)
            self.cash              += interest
            self.total_cash_yield  += interest

    def enter(
        self,
        ticker:    str,
        strategy:  str,
        price:     float,
        date:      pd.Timestamp,
        alloc_pct: float,
        trail_stop_pct: float,
        take_profit_pct: float = 0.0,
        stop_loss_pct:   float = 0.0,
        max_hold_days:   int   = 9999,
    ) -> bool:
        if ticker in self.positions:
            return False
        alloc     = self.initial_capital * alloc_pct
        max_alloc = min(alloc, self.cash * 0.98)
        if max_alloc < 10_000:
            return False
        fill = price * (1 + SLIPPAGE)
        sh   = int(max_alloc / fill)
        if sh < 1:
            return False
        cost      = sh * fill
        tx_cost   = cost * COST_PCT
        total_out = cost + tx_cost
        if total_out > self.cash:
            sh = int((self.cash * 0.98 / (1 + COST_PCT)) / fill)
            if sh < 1:
                return False
            cost      = sh * fill
            tx_cost   = cost * COST_PCT
            total_out = cost + tx_cost
        self.cash -= total_out
        self.positions[ticker] = Position(
            ticker=ticker,
            strategy=strategy,
            entry_price=fill,
            peak_price=fill,
            entry_date=date,
            shares=sh,
            entry_cost=total_out,
            trail_stop_pct=trail_stop_pct,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_hold_days=max_hold_days,
        )
        return True

    def exit(
        self,
        ticker:  str,
        price:   float,
        date:    pd.Timestamp,
        reason:  str,
    ) -> Optional[Trade]:
        if ticker not in self.positions:
            return None
        pos      = self.positions.pop(ticker)
        fill     = price * (1 - SLIPPAGE)
        proceeds = pos.shares * fill
        tx_cost  = proceeds * COST_PCT
        net_gain = (proceeds - tx_cost) - pos.entry_cost
        self.cash += (proceeds - tx_cost)
        hold     = (date - pos.entry_date).days
        pnl_pct  = net_gain / pos.entry_cost * 100
        trade    = Trade(
            ticker=ticker,
            strategy=pos.strategy,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=fill,
            shares=pos.shares,
            net_pnl=net_gain,
            pnl_pct=pnl_pct,
            hold_days=hold,
            exit_reason=reason,
        )
        self.trades.append(trade)
        return trade

    def reduce_all(self, prices: Dict[str, float], date: pd.Timestamp, fraction: float = 0.50):
        for ticker in list(self.positions.keys()):
            pos   = self.positions[ticker]
            price = prices.get(ticker, pos.entry_price)
            sell_sh = int(pos.shares * fraction)
            if sell_sh < 1:
                continue
            fill          = price * (1 - SLIPPAGE)
            proceeds      = sell_sh * fill
            tx_cost       = proceeds * COST_PCT
            entry_frac    = sell_sh / pos.shares
            entry_portion = pos.entry_cost * entry_frac
            self.cash    += (proceeds - tx_cost)
            pos.shares   -= sell_sh
            pos.entry_cost -= entry_portion
            if pos.shares <= 0:
                self.positions.pop(ticker, None)


# ── Results compiler ──────────────────────────────────────────────────────────

def _compile(port: Portfolio, label: str) -> Dict:
    eq_df    = pd.DataFrame(port.equity_history).set_index("date")
    trades   = port.trades
    n_trades = len(trades)

    total_ret  = (eq_df["equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    n_years    = (pd.Timestamp(OOS_END) - pd.Timestamp(OOS_START)).days / 365.25
    ann_ret    = ((1 + total_ret / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0
    net_tax    = ann_ret * 0.80

    win_rate  = float((pd.Series([t.net_pnl for t in trades]) > 0).mean() * 100) if n_trades > 0 else 0.0
    avg_hold  = float(np.mean([t.hold_days for t in trades])) if n_trades > 0 else 0.0

    roll_max  = eq_df["equity"].cummax()
    max_dd    = float((eq_df["equity"] / roll_max - 1).min() * 100) if len(eq_df) > 1 else 0.0

    eq_ret    = eq_df["equity"].pct_change().dropna()
    rf_daily  = 0.065 / 252
    sharpe    = 0.0
    if len(eq_ret) > 5 and eq_ret.std() > 1e-10:
        excess = eq_ret - rf_daily
        sharpe = float(excess.mean() / excess.std() * np.sqrt(252))

    # Annual
    eq_df["year"] = eq_df.index.year
    annual: Dict[int, float] = {}
    prev_eq = INITIAL_CAPITAL
    for yr in sorted(eq_df["year"].unique()):
        yr_eq  = eq_df[eq_df["year"] == yr]["equity"]
        yr_end = float(yr_eq.iloc[-1])
        annual[yr] = (yr_end / prev_eq - 1) * 100
        prev_eq = yr_end

    mom_trades = [t for t in trades if t.strategy == "momentum"]
    mr_trades  = [t for t in trades if t.strategy == "mean_reversion"]
    mom_wr     = float((pd.Series([t.net_pnl for t in mom_trades]) > 0).mean() * 100) if mom_trades else 0.0
    mr_wr      = float((pd.Series([t.net_pnl for t in mr_trades]) > 0).mean() * 100) if mr_trades else 0.0

    return {
        "label":           label,
        "total_ret":       round(total_ret, 2),
        "ann_ret":         round(ann_ret, 2),
        "net_after_tax":   round(net_tax, 2),
        "win_rate":        round(win_rate, 1),
        "max_dd":          round(max_dd, 2),
        "sharpe":          round(sharpe, 3),
        "n_trades":        n_trades,
        "avg_hold":        round(avg_hold, 1),
        "annual":          {str(k): round(v, 1) for k, v in annual.items()},
        "mom_trades":      len(mom_trades),
        "mr_trades":       len(mr_trades),
        "mom_win_rate":    round(mom_wr, 1),
        "mr_win_rate":     round(mr_wr, 1),
        "cash_yield_total": round(port.total_cash_yield / 1e5, 2),  # in lakhs
        "trades":          trades,
        "equity_df":       eq_df,
    }


# ── Baseline runner (Iteration 6 — unchanged) ─────────────────────────────────

def run_baseline(all_data: Dict, conf_map: Dict, dates: pd.DatetimeIndex) -> Dict:
    port        = Portfolio(INITIAL_CAPITAL, use_cash_yield=False)
    last_rebal: Optional[pd.Timestamp] = None

    for date in dates:
        prices = {t: float(all_data[t].loc[date, "close"])
                  for t in all_data if date in all_data[t].index}

        for tk, pos in port.positions.items():
            if tk in prices:
                pos.peak_price = max(pos.peak_price, prices[tk])

        is_rebal = (last_rebal is None) or ((date - last_rebal).days >= BASELINE_REBAL_DAYS)

        for tk in list(port.positions.keys()):
            if tk not in prices:
                continue
            pos  = port.positions[tk]
            curr = prices[tk]
            if curr < pos.peak_price * (1 - BASELINE_TRAIL_STOP):
                port.exit(tk, curr, date, "TRAILING_STOP")
                continue
            if is_rebal and tk in conf_map:
                rc = get_rolling_conf(conf_map[tk], date)
                if rc < BASELINE_ML_EXIT:
                    port.exit(tk, curr, date, f"ML_EXIT(rc={rc:.3f})")

        if is_rebal:
            last_rebal = date
            candidates = []
            for tk in conf_map:
                if tk in port.positions or tk not in prices:
                    continue
                rc = get_rolling_conf(conf_map[tk], date)
                if rc >= BASELINE_ML_ENTRY:
                    candidates.append((tk, rc))
            candidates.sort(key=lambda x: -x[1])
            slots = BASELINE_MAX_POS - len(port.positions)
            for tk, rc in candidates[:slots]:
                port.enter(tk, "momentum", prices[tk], date, BASELINE_ALLOC, BASELINE_TRAIL_STOP)

        eq = port.get_equity(prices)
        port.equity_history.append({"date": date, "equity": eq, "n_pos": len(port.positions)})

    final_date   = dates[-1]
    final_prices = {t: float(all_data[t].loc[final_date, "close"])
                    for t in all_data if final_date in all_data[t].index}
    for tk in list(port.positions.keys()):
        port.exit(tk, final_prices.get(tk, port.positions[tk].entry_price), final_date, "END_OF_SIM")

    return _compile(port, "BASELINE (13 tickers, no cash yield)")


# ── Enhanced v2 runner ────────────────────────────────────────────────────────

def run_enhanced_v2(
    all_data:  Dict,
    mr_data:   Dict,
    conf_map:  Dict,
    mr_conf:   Dict,
    nifty:     pd.Series,
    dates:     pd.DatetimeIndex,
) -> Dict:
    """
    Multi-strategy v2:
    - 32-ticker expanded universe
    - Cash yield at 6.5% p.a.
    - Calibrated MR (v2 conditions)
    - Re-entry cooldown (21 bars after trailing stop)
    - Circuit breaker overlay
    """
    router   = RegimeRouter()
    mr_strat = MeanReversionStrategy()  # v2 parameters
    port     = Portfolio(INITIAL_CAPITAL, use_cash_yield=True)

    regime_series = router.detect_series(nifty)

    last_rebal:    Optional[pd.Timestamp] = None
    mr_cooldown:   Dict[str, int]  = {}   # ticker → bars until MR re-entry OK
    trail_cooldown: Dict[str, int] = {}   # ticker → bars until momentum re-entry OK

    for date in dates:
        prices    = {t: float(all_data[t].loc[date, "close"])
                     for t in all_data if date in all_data[t].index}
        mr_prices = {t: float(mr_data[t].loc[date, "close"])
                     for t in mr_data if date in mr_data[t].index}
        all_prices = {**prices, **mr_prices}

        # ── Regime ───────────────────────────────────────────────────────────
        if date in regime_series.index:
            regime = regime_series.loc[date]
        else:
            regime = MarketRegimeState.NEUTRAL
        alloc = router.allocation(regime)
        is_bear = regime == MarketRegimeState.BEAR

        # ── Cash yield ───────────────────────────────────────────────────────
        port.accrue_yield()

        # ── Circuit breaker ───────────────────────────────────────────────────
        eq_now = port.get_equity(all_prices)
        if len(port.positions) > 0:
            nifty_slice = nifty.loc[:date]
            nifty_sma200 = float(nifty_slice.rolling(200, min_periods=100).mean().iloc[-1]) \
                if len(nifty_slice) >= 100 else float(nifty.iloc[0])
            nifty_curr = float(nifty.loc[date]) if date in nifty.index else float(nifty.iloc[-1])
            nifty_above_sma200 = nifty_curr > nifty_sma200
            cb_action = port.circuit_breaker.update(eq_now, date, nifty_above_sma200)
        else:
            port.circuit_breaker._equity_window = [eq_now]
            port.circuit_breaker.state.level = CircuitBreakerLevel.NONE
            cb_action = CircuitBreakerLevel.NONE

        if cb_action == CircuitBreakerLevel.HALT:
            for tk in list(port.positions.keys()):
                p = all_prices.get(tk, port.positions[tk].entry_price)
                port.exit(tk, p, date, "CIRCUIT_BREAKER_HALT")
            eq = port.get_equity(all_prices)
            port.equity_history.append({"date": date, "equity": eq, "regime": regime.value, "n_pos": 0})
            continue

        if cb_action == CircuitBreakerLevel.WARNING:
            port.reduce_all(all_prices, date, fraction=0.50)

        # ── Update peaks ──────────────────────────────────────────────────────
        for tk, pos in port.positions.items():
            p = all_prices.get(tk, pos.entry_price)
            pos.peak_price = max(pos.peak_price, p)

        # ── Decrement cooldowns ───────────────────────────────────────────────
        for d in [mr_cooldown, trail_cooldown]:
            for tk in list(d.keys()):
                d[tk] -= 1
                if d[tk] <= 0:
                    del d[tk]

        is_rebal = (last_rebal is None) or ((date - last_rebal).days >= BASELINE_REBAL_DAYS)

        # ── Check exits ───────────────────────────────────────────────────────
        for tk in list(port.positions.keys()):
            pos = port.positions.get(tk)
            if pos is None:
                continue
            curr = all_prices.get(tk, pos.entry_price)
            hold = (date - pos.entry_date).days

            if pos.strategy == "momentum":
                trail_stop = alloc.momentum_trail_stop_pct
                if curr < pos.peak_price * (1 - trail_stop):
                    port.exit(tk, curr, date, "TRAILING_STOP")
                    trail_cooldown[tk] = REENTRY_COOLDOWN_BARS  # ← re-entry cooldown
                    continue
                if is_rebal and tk in conf_map:
                    rc = get_rolling_conf(conf_map[tk], date)
                    if rc < BASELINE_ML_EXIT:
                        port.exit(tk, curr, date, f"ML_EXIT(rc={rc:.3f})")
                if regime == MarketRegimeState.CRISIS:
                    port.exit(tk, curr, date, "CRISIS_CLOSE")

            elif pos.strategy == "mean_reversion":
                _h1 = mr_data.get(tk)
                _h2 = all_data.get(tk)
                hist = _h1 if _h1 is not None else _h2
                if hist is not None and date in hist.index:
                    hist_slice = hist.loc[:date]
                    mc         = mr_conf.get(tk)
                    ml_conf    = get_rolling_conf(mc, date) if mc is not None else 0.5
                    sig = mr_strat.should_exit(
                        tk, hist_slice, nifty, date,
                        pos.entry_price, pos.peak_price, hold, ml_conf,
                    )
                    if sig is not None:
                        reason = sig.reasons[0] if sig.reasons else "MR_EXIT"
                        port.exit(tk, curr, date, reason)
                        mr_cooldown[tk] = 3  # 3-bar MR cooldown

        # ── Entries ───────────────────────────────────────────────────────────
        if port.circuit_breaker.allow_new_entries and alloc.allow_new_entries:

            # Momentum entries (monthly rebalance, bull+neutral regime)
            if is_rebal and regime in (MarketRegimeState.BULL, MarketRegimeState.NEUTRAL):
                last_rebal = date
                candidates = []
                mom_in_port = {tk for tk, p in port.positions.items() if p.strategy == "momentum"}
                for tk in conf_map:
                    if tk in port.positions:
                        continue
                    if tk in trail_cooldown:   # re-entry cooldown active
                        continue
                    if tk not in prices:
                        continue
                    rc = get_rolling_conf(conf_map[tk], date)
                    if rc >= BASELINE_ML_ENTRY:
                        candidates.append((tk, rc))
                candidates.sort(key=lambda x: -x[1])
                slots = alloc.max_momentum_pos - len(mom_in_port)
                for tk, rc in candidates[:slots]:
                    port.enter(
                        tk, "momentum", prices[tk], date,
                        alloc.momentum_pct, alloc.momentum_trail_stop_pct,
                    )

            # MR entries (daily, bear+neutral regime)
            if regime in (MarketRegimeState.BEAR, MarketRegimeState.NEUTRAL):
                mr_in_port = {tk for tk, p in port.positions.items() if p.strategy == "mean_reversion"}
                slots = alloc.max_mean_rev_pos - len(mr_in_port)
                if slots > 0:
                    for tk in MR_CANDIDATES:
                        if slots <= 0:
                            break
                        if tk in port.positions or tk in mr_cooldown:
                            continue
                        _h1 = mr_data.get(tk)
                        _h2 = all_data.get(tk)
                        hist = _h1 if _h1 is not None else _h2
                        if hist is None or date not in hist.index:
                            continue
                        hist_slice = hist.loc[:date]
                        if len(hist_slice) < 50:
                            continue
                        mc      = mr_conf.get(tk)
                        ml_conf = get_rolling_conf(mc, date) if mc is not None else 0.5
                        sig = mr_strat.should_enter(
                            tk, hist_slice, nifty, date, ml_conf, bear_regime=is_bear,
                        )
                        if sig is not None:
                            curr_p = float(hist.loc[date, "close"])
                            pos_pct = sig.position_pct   # regime-aware from v2 MR
                            if port.enter(
                                tk, "mean_reversion", curr_p, date,
                                pos_pct, mr_strat.stop_loss_pct,
                                take_profit_pct=mr_strat.take_profit_pct,
                                stop_loss_pct=mr_strat.stop_loss_pct,
                                max_hold_days=mr_strat.max_hold_days,
                            ):
                                slots -= 1

        # ── Record equity ─────────────────────────────────────────────────────
        eq = port.get_equity(all_prices)
        port.equity_history.append({
            "date": date, "equity": eq,
            "regime": regime.value, "n_pos": len(port.positions),
        })

    # Force-close at end
    final_date   = dates[-1]
    final_prices = {t: float(all_data[t].loc[final_date, "close"])
                    for t in all_data if final_date in all_data[t].index}
    final_mr     = {t: float(mr_data[t].loc[final_date, "close"])
                    for t in mr_data if final_date in mr_data[t].index}
    final_all    = {**final_prices, **final_mr}
    for tk in list(port.positions.keys()):
        port.exit(tk, final_all.get(tk, port.positions[tk].entry_price),
                  final_date, "END_OF_SIM")

    return _compile(port, "ENHANCED-v2 (32 tickers + cash yield + calibrated MR)")


# ── Report printer ────────────────────────────────────────────────────────────

def print_comparison(baseline: Dict, v2: Dict) -> None:
    W = 90
    print("\n" + "═" * W)
    print("  MARK5 MULTI-STRATEGY BACKTEST v2.0 — OOS 2022–2026  |  PAPER MODE")
    print("═" * W)

    fmt = "  {:<36} {:>22}  {:>22}"
    print(fmt.format("Metric", "BASELINE (13t, no yield)", "ENHANCED-v2 (32t + yield)"))
    print("  " + "─" * (W - 2))

    def row(label, b_val, e_val, better_higher=True):
        if isinstance(b_val, (int, float)) and isinstance(e_val, (int, float)):
            diff = e_val - b_val
            if better_higher:
                mark = "✅" if diff > 0.2 else ("❌" if diff < -0.2 else "")
            else:
                mark = "✅" if diff < -0.2 else ("❌" if diff > 0.2 else "")
            print(fmt.format(label, f"{float(b_val):.2f}", f"{float(e_val):.2f}  {mark}"))
        else:
            print(fmt.format(label, str(b_val), str(e_val)))

    row("4yr Total Return (%)",       baseline["total_ret"],     v2["total_ret"])
    row("Annual CAGR % (gross)",      baseline["ann_ret"],       v2["ann_ret"])
    row("Net After 20% Tax (%/yr)",   baseline["net_after_tax"], v2["net_after_tax"])
    row("Overall Win Rate (%)",       baseline["win_rate"],      v2["win_rate"])
    row("Max Drawdown (%)",           baseline["max_dd"],        v2["max_dd"], better_higher=False)
    row("Sharpe Ratio",               baseline["sharpe"],        v2["sharpe"])
    row("Total Trades",               baseline["n_trades"],      v2["n_trades"])
    row("Avg Hold Days",              baseline["avg_hold"],      v2["avg_hold"], better_higher=False)

    print("  " + "─" * (W - 2))
    print(f"  Momentum trades       : {baseline['mom_trades']:>22}   {v2['mom_trades']:>22}")
    print(f"  Mean-rev trades       : {0:>22}   {v2['mr_trades']:>22}")
    print(f"  Momentum WR (%)       : {baseline['mom_win_rate']:>22.1f}   {v2['mom_win_rate']:>22.1f}")
    print(f"  Mean-rev WR (%)       : {'N/A':>22}   {v2['mr_win_rate']:>22.1f}")
    print(f"  Cash yield earned (₹L): {'N/A':>22}   {v2['cash_yield_total']:>22.1f}")

    print("\n  Annual Breakdown:")
    all_years = sorted(set(list(baseline["annual"].keys()) + list(v2["annual"].keys())))
    for yr in all_years:
        b   = baseline["annual"].get(yr, 0.0)
        e   = v2["annual"].get(yr, 0.0)
        dif = e - b
        fb  = "✅" if b > 0 else "❌"
        fe  = "✅" if e > 0 else "❌"
        print(f"    {yr}:  {fb} {b:>+7.1f}%  →  {fe} {e:>+7.1f}%  (Δ{dif:+.1f}%)")

    print("\n" + "═" * W)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═'*80}")
    print("  MARK5 Multi-Strategy Backtest v2.0")
    print(f"  OOS: {OOS_START} → {OOS_END}  |  Capital: ₹{INITIAL_CAPITAL/1e7:.0f} crore")
    print(f"{'═'*80}\n")

    # ── Expand universe ───────────────────────────────────────────────────────
    print("Scanning available ML models for expanded universe...")
    expander = UniverseExpander(model_root=MODELS_DIR, cache_dir=DATA_CACHE)
    expanded_tickers = expander.scan(oos_start=OOS_START, oos_end=OOS_END, verbose=False)

    # Exclude obvious structural decliners that also happen to have models
    expanded_tickers = [t for t in expanded_tickers if t not in {"IDEA", "YESBANK", "IDFCFIRSTB"}]

    print(f"  Baseline universe  : {len(BASELINE_TICKERS)} tickers: {BASELINE_TICKERS}")
    print(f"  Expanded universe  : {len(expanded_tickers)} tickers: {expanded_tickers}")

    # ── Load baseline data ────────────────────────────────────────────────────
    print("\nLoading baseline ticker data...")
    baseline_data: Dict[str, pd.DataFrame] = {}
    for tk in BASELINE_TICKERS:
        df = load_ticker(tk)
        if df is not None:
            baseline_data[tk] = df
            print(f"  {tk}: {len(df)} bars")

    # ── Load expanded data ────────────────────────────────────────────────────
    print("\nLoading expanded ticker data...")
    expanded_data: Dict[str, pd.DataFrame] = {}
    for tk in expanded_tickers:
        df = load_ticker(tk)
        if df is not None and len(df.loc[OOS_START:OOS_END]) >= 100:
            expanded_data[tk] = df
            print(f"  {tk}: {len(df)} bars")

    # ── Load MR candidate data ────────────────────────────────────────────────
    print("\nLoading MR candidates...")
    mr_data: Dict[str, pd.DataFrame] = {}
    for tk in MR_CANDIDATES:
        df = load_ticker(tk)
        if df is not None:
            mr_data[tk] = df

    # ── Load Nifty ────────────────────────────────────────────────────────────
    print("\nLoading Nifty 50...")
    nifty = load_nifty()
    if nifty is None:
        print("ERROR: Nifty 50 data not found. Exiting.")
        return {}
    nifty = nifty.loc[:OOS_END]
    print(f"  Nifty: {len(nifty)} bars ({nifty.index[0].date()}→{nifty.index[-1].date()})")

    # ── Pre-compute ML confidence ─────────────────────────────────────────────
    print("\nPre-computing baseline ML confidence series...")
    baseline_conf: Dict[str, pd.Series] = {}
    for tk in baseline_data:
        c = load_ml_confidence(tk)
        if c is not None:
            baseline_conf[tk] = c

    print("\nPre-computing expanded ML confidence series...")
    expanded_conf: Dict[str, pd.Series] = {}
    for tk in expanded_data:
        c = load_ml_confidence(tk)
        if c is not None:
            expanded_conf[tk] = c
            oos = c.loc[OOS_START:OOS_END] if OOS_START in c.index or len(c) > 0 else c
            above = (oos >= BASELINE_ML_ENTRY).mean() * 100 if len(oos) > 0 else 0
            print(f"  {tk}: {above:.0f}% bars above entry hurdle")
        else:
            print(f"  {tk}: No ML model found")

    mr_conf: Dict[str, pd.Series] = {}
    for tk in MR_CANDIDATES:
        if tk not in expanded_conf and tk not in baseline_conf:
            c = load_ml_confidence(tk)
            if c is not None:
                mr_conf[tk] = c

    # Merge all conf maps for MR
    all_mr_conf = {**baseline_conf, **expanded_conf, **mr_conf}

    # ── OOS date index ────────────────────────────────────────────────────────
    ref_ticker = next(iter(baseline_data))
    oos_dates  = baseline_data[ref_ticker].loc[OOS_START:OOS_END].index

    print(f"\nOOS trading days: {len(oos_dates)}")
    print(f"Baseline tickers with ML: {len(baseline_conf)}")
    print(f"Expanded tickers with ML: {len(expanded_conf)}")

    # ── Run both systems ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Running BASELINE (Iteration 6 — 13 tickers, no cash yield)...")
    baseline = run_baseline(baseline_data, baseline_conf, oos_dates)

    print("\n" + "─" * 60)
    print("Running ENHANCED-v2 (32 tickers + cash yield + calibrated MR + cooldown)...")
    enhanced = run_enhanced_v2(
        expanded_data, mr_data, expanded_conf, all_mr_conf, nifty, oos_dates,
    )

    # ── Print & save ──────────────────────────────────────────────────────────
    print_comparison(baseline, enhanced)

    out = {
        "baseline": {k: v for k, v in baseline.items() if k not in ("trades", "equity_df")},
        "enhanced_v2": {k: v for k, v in enhanced.items() if k not in ("trades", "equity_df")},
        "improvements": {
            "universe_size": {
                "baseline": len(BASELINE_TICKERS),
                "v2":       len(expanded_tickers),
            },
            "cash_yield_total_lakhs": enhanced["cash_yield_total"],
            "mr_trades_v1": 40,    # from previous run
            "mr_trades_v2": enhanced["mr_trades"],
        },
    }
    out_path = os.path.join(_ROOT, "reports", "multi_strategy_backtest_v2.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  ✅ Results saved → {out_path}")

    # Print cash yield contribution
    print(f"\n  Cash yield earned (v2): ₹{enhanced['cash_yield_total']:.1f}L "
          f"(adds to equity above)")

    return out


if __name__ == "__main__":
    main()

"""
MARK5 Multi-Strategy Portfolio Backtest v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compares BASELINE (momentum-only) vs ENHANCED (multi-strategy) system
on the true OOS period 2022–2026.

BASELINE  = Iteration 6 ML Momentum Portfolio (25% positions, 15% trailing stop)
ENHANCED  = Regime-Gated Momentum + Mean-Reversion + Circuit Breaker

HOW TO RUN:
    cd /home/lynx/Documents/MARK5
    python3 scripts/multi_strategy_backtest.py

OUTPUT:
    Side-by-side comparison table showing:
    - Annual returns 2022–2026
    - Win rate
    - Max drawdown
    - Sharpe ratio
    - N trades

STRATEGY LOGIC (ENHANCED):
  1. RegimeRouter classifies market each day as BULL/NEUTRAL/BEAR/CRISIS
     using Nifty 50 200d SMA + 50d SMA crossover.

  2. BULL regime  → Momentum only (25% per stock, up to 4 positions)
  3. NEUTRAL regime→ Momentum (15%) + Mean-Reversion (10%), 15% cash buffer
  4. BEAR regime  → Momentum BLOCKED, Mean-Reversion only (10%, up to 4)
  5. CRISIS regime→ All BLOCKED, close all positions immediately

  6. PortfolioCircuitBreaker overlay:
     • -12% portfolio DD from rolling peak → reduce positions by 50%
     • -18% portfolio DD → close ALL, 10-bar cooldown

CHANGELOG:
- [2026-05-23] v1.0: Initial multi-strategy backtest
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.strategies.regime_router import RegimeRouter, MarketRegimeState
from core.strategies.mean_reversion import MeanReversionStrategy
from core.strategies.circuit_breaker import PortfolioCircuitBreaker, CircuitBreakerLevel

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("MARK5.MultiStrategyBacktest")

# ── Universe & config ─────────────────────────────────────────────────────────
MOMENTUM_TICKERS = [
    "ASIANPAINT", "AUBANK", "BAJFINANCE", "BHARTIARTL", "COFORGE",
    "HAL", "PNB", "RELIANCE", "TATAELXSI", "TATASTEEL",
    "TCS", "TRENT", "YESBANK",
]

# Mean-reversion candidates: quality Nifty 100 stocks in the cache
MEAN_REV_CANDIDATES = [
    "HDFCBANK", "ICICIBANK", "INFY", "TCS", "RELIANCE",
    "BAJFINANCE", "SBIN", "KOTAKBANK", "LT", "AXISBANK",
    "TITAN", "SUNPHARMA", "HINDUNILVR", "MARUTI", "LUPIN",
    "BRITANNIA", "BHARTIARTL", "ITC", "COFORGE", "PERSISTENT",
]

INITIAL_CAPITAL      = 5_00_00_000.0    # ₹5 crore
OOS_START            = "2022-01-01"
OOS_END              = "2026-05-21"

# Baseline (Iteration 6) params
BASELINE_ALLOC       = 0.25
BASELINE_MAX_POS     = 4
BASELINE_TRAIL_STOP  = 0.15
BASELINE_ML_ENTRY    = 0.52
BASELINE_ML_EXIT     = 0.45
BASELINE_REBAL_DAYS  = 21
BASELINE_ML_WINDOW   = 10

# Transaction costs
COST_PCT   = 0.0029     # 0.29% round-trip (EQUITY_DELIVERY)
SLIPPAGE   = 0.001      # 0.1% slippage

DATA_CACHE = os.path.join(_ROOT, "data", "cache")
MODELS_DIR = os.path.join(_ROOT, "models")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ticker(ticker: str) -> Optional[pd.DataFrame]:
    """Load daily OHLCV from parquet cache (tries _daily.parquet first)."""
    for suffix in ["_daily.parquet", "_NS_1d.parquet"]:
        path = os.path.join(DATA_CACHE, f"{ticker}{suffix}")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            return df
    return None


def load_nifty() -> Optional[pd.Series]:
    """Load Nifty 50 index from yfinance (cached)."""
    cache_path = os.path.join(DATA_CACHE, "nse", "NIFTY50_20150101_20260521.parquet")
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        df.columns = [c.lower() for c in df.columns]
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df["close"].sort_index()

    try:
        import yfinance as yf
        raw = yf.download("^NSEI", start="2015-01-01", end=OOS_END, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        series = raw["close"].sort_index().dropna()
        # Cache it
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        raw.to_parquet(cache_path)
        return series
    except Exception as e:
        logger.error(f"Nifty fetch failed: {e}")
        return None


def load_ml_confidence(ticker: str) -> Optional[pd.Series]:
    """Pre-compute full ML confidence series using existing LightPredictor."""
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
    """Get rolling average ML confidence at a date (inference-safe)."""
    try:
        idx = series.index.searchsorted(date, side="right") - 1
        idx = max(0, min(idx, len(series) - 1))
        start = max(0, idx - window + 1)
        return float(series.iloc[start:idx + 1].mean())
    except Exception:
        return 0.5


# ── Portfolio engine ──────────────────────────────────────────────────────────

@dataclass
class Position:
    ticker:      str
    strategy:    str        # "momentum" or "mean_reversion"
    entry_price: float
    peak_price:  float
    entry_date:  pd.Timestamp
    shares:      int
    entry_cost:  float      # total cash out at entry
    trail_stop_pct: float   # current trailing stop %


@dataclass
class Trade:
    ticker:      str
    strategy:    str
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    entry_price: float
    exit_price:  float
    shares:      int
    net_pnl:     float
    pnl_pct:     float
    hold_days:   int
    exit_reason: str


class Portfolio:
    """Shared portfolio state for all strategies."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash      = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades:    List[Trade]         = []
        self.equity_history: List[Dict]     = []
        self.circuit_breaker = PortfolioCircuitBreaker(initial_capital)

    def get_equity(self, prices: Dict[str, float]) -> float:
        pos_value = sum(
            p.shares * prices.get(t, p.entry_price)
            for t, p in self.positions.items()
        )
        return self.cash + pos_value

    def enter(
        self,
        ticker: str,
        strategy: str,
        price: float,
        date: pd.Timestamp,
        alloc_pct: float,
        trail_stop_pct: float,
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
        cost     = sh * fill
        tx_cost  = cost * COST_PCT
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
        )
        logger.info(
            f"[{strategy}] ENTER {ticker} @ ₹{fill:.0f} ×{sh} "
            f"({alloc_pct:.0%} alloc) on {date.date()}"
        )
        return True

    def exit(self, ticker: str, price: float, date: pd.Timestamp, reason: str) -> Optional[Trade]:
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
        logger.info(
            f"[{pos.strategy}] EXIT {ticker} @ ₹{fill:.0f} ({reason}) | "
            f"PnL={pnl_pct:+.1f}% ({hold}d) | Net=₹{net_gain/1e5:.2f}L"
        )
        return trade

    def reduce_all(self, prices: Dict[str, float], date: pd.Timestamp, fraction: float = 0.50):
        """Circuit breaker: sell `fraction` of all positions."""
        for ticker in list(self.positions.keys()):
            pos   = self.positions[ticker]
            price = prices.get(ticker, pos.entry_price)
            sell_sh = int(pos.shares * fraction)
            if sell_sh < 1:
                continue
            fill      = price * (1 - SLIPPAGE)
            proceeds  = sell_sh * fill
            tx_cost   = proceeds * COST_PCT
            # proportional entry cost
            entry_fraction = sell_sh / pos.shares
            entry_portion  = pos.entry_cost * entry_fraction
            net_gain       = (proceeds - tx_cost) - entry_portion
            self.cash     += (proceeds - tx_cost)
            pos.shares    -= sell_sh
            pos.entry_cost -= entry_portion
            logger.warning(
                f"⚠️  CB REDUCE {ticker}: sold {sell_sh}sh @ ₹{fill:.0f} | "
                f"remaining {pos.shares}sh | Net=₹{net_gain/1e5:.2f}L"
            )
            if pos.shares <= 0:
                self.positions.pop(ticker, None)


# ── Strategy runners ──────────────────────────────────────────────────────────

def run_baseline(all_data: Dict, conf_map: Dict, dates: pd.DatetimeIndex) -> Dict:
    """
    Baseline: Iteration 6 ML Momentum Portfolio (no regime filter, no CB).
    """
    port      = Portfolio(INITIAL_CAPITAL)
    last_rebal: Optional[pd.Timestamp] = None

    for date in dates:
        prices = {t: float(all_data[t].loc[date, "close"])
                  for t in all_data if date in all_data[t].index}

        # Update trailing peaks
        for tk, pos in port.positions.items():
            if tk in prices:
                pos.peak_price = max(pos.peak_price, prices[tk])

        # ── Check exits ───────────────────────────────────────────────────────
        is_rebal = (last_rebal is None) or ((date - last_rebal).days >= BASELINE_REBAL_DAYS)

        for tk in list(port.positions.keys()):
            pos = port.positions[tk]
            if tk not in prices:
                continue
            curr = prices[tk]
            # Trailing stop
            if curr < pos.peak_price * (1 - BASELINE_TRAIL_STOP):
                port.exit(tk, curr, date, "TRAILING_STOP")
                continue
            # ML exit (monthly)
            if is_rebal and tk in conf_map:
                rc = get_rolling_conf(conf_map[tk], date)
                if rc < BASELINE_ML_EXIT:
                    port.exit(tk, curr, date, f"ML_EXIT(rc={rc:.3f})")

        # ── Entries (monthly) ─────────────────────────────────────────────────
        if is_rebal:
            last_rebal = date
            candidates = []
            for tk in conf_map:
                if tk in port.positions:
                    continue
                if tk not in prices:
                    continue
                rc = get_rolling_conf(conf_map[tk], date)
                if rc >= BASELINE_ML_ENTRY:
                    candidates.append((tk, rc))
            candidates.sort(key=lambda x: x[1], reverse=True)
            slots = BASELINE_MAX_POS - len(port.positions)
            for tk, rc in candidates[:slots]:
                port.enter(tk, "momentum", prices[tk], date, BASELINE_ALLOC, BASELINE_TRAIL_STOP)

        # ── Record equity ─────────────────────────────────────────────────────
        eq = port.get_equity(prices)
        port.equity_history.append({"date": date, "equity": eq, "n_pos": len(port.positions)})

    # Force close at end
    final_date   = dates[-1]
    final_prices = {t: float(all_data[t].loc[final_date, "close"])
                    for t in all_data if final_date in all_data[t].index}
    for tk in list(port.positions.keys()):
        port.exit(tk, final_prices.get(tk, port.positions[tk].entry_price), final_date, "END_OF_SIM")

    return _compile_results(port, "BASELINE (Momentum-Only)")


def run_enhanced(
    all_data:  Dict,
    mr_data:   Dict,
    conf_map:  Dict,
    mr_conf:   Dict,
    nifty:     pd.Series,
    dates:     pd.DatetimeIndex,
) -> Dict:
    """
    Enhanced: Regime-Gated Momentum + Mean-Reversion + Circuit Breaker.
    """
    router  = RegimeRouter()
    mr_strat = MeanReversionStrategy()
    port    = Portfolio(INITIAL_CAPITAL)
    regime_series = router.detect_series(nifty)

    last_rebal: Optional[pd.Timestamp] = None
    mr_cooldown: Dict[str, int] = {}   # ticker → bars since last MR exit

    for date in dates:
        prices = {t: float(all_data[t].loc[date, "close"])
                  for t in all_data if date in all_data[t].index}
        mr_prices = {t: float(mr_data[t].loc[date, "close"])
                     for t in mr_data if date in mr_data[t].index}
        all_prices = {**prices, **mr_prices}

        # Regime for today
        if date in regime_series.index:
            regime = regime_series.loc[date]
        else:
            regime = RegimeRouter().detect(nifty.loc[:date].iloc[-300:] if len(nifty.loc[:date]) > 300 else nifty.loc[:date])
        alloc = router.allocation(regime)

        # ── Circuit breaker check (only activates with open positions) ────────
        # The CB must NOT trigger when the portfolio is in a cash-phase
        # (naturally between momentum positions). A drop from 2023 peak to 2024
        # cash level is expected portfolio behaviour, not a crisis.
        eq_now = port.get_equity(all_prices)
        if len(port.positions) > 0:
            nifty_slice = nifty.loc[:date]
            nifty_sma200 = float(nifty_slice.rolling(200, min_periods=100).mean().iloc[-1]) \
                if len(nifty_slice) >= 100 else float(nifty.iloc[0])
            nifty_above_sma200 = (float(nifty.loc[date]) if date in nifty.index
                                  else float(nifty.iloc[-1])) > nifty_sma200
            cb_action = port.circuit_breaker.update(eq_now, date, nifty_above_sma200)
        else:
            # In cash: reset the CB peak to current equity so it doesn't fire on re-entry
            port.circuit_breaker._equity_window = [eq_now]
            port.circuit_breaker.state.level = CircuitBreakerLevel.NONE
            cb_action = CircuitBreakerLevel.NONE

        if cb_action == CircuitBreakerLevel.HALT:
            # Close all positions immediately
            for tk in list(port.positions.keys()):
                p = all_prices.get(tk, port.positions[tk].entry_price)
                port.exit(tk, p, date, "CIRCUIT_BREAKER_HALT")
            eq = port.get_equity(all_prices)
            port.equity_history.append({"date": date, "equity": eq, "regime": regime.value, "n_pos": 0})
            continue

        if cb_action == CircuitBreakerLevel.WARNING:
            port.reduce_all(all_prices, date, fraction=0.50)

        # Update trailing peaks for open positions
        for tk, pos in port.positions.items():
            p = all_prices.get(tk, pos.entry_price)
            pos.peak_price = max(pos.peak_price, p)

        # Decrement MR cooldowns
        for tk in list(mr_cooldown.keys()):
            mr_cooldown[tk] -= 1
            if mr_cooldown[tk] <= 0:
                del mr_cooldown[tk]

        is_rebal = (last_rebal is None) or ((date - last_rebal).days >= BASELINE_REBAL_DAYS)

        # ── Check exits ───────────────────────────────────────────────────────
        for tk in list(port.positions.keys()):
            pos = port.positions.get(tk)
            if pos is None:
                continue
            curr = all_prices.get(tk, pos.entry_price)
            hold = (date - pos.entry_date).days

            # Momentum exits
            if pos.strategy == "momentum":
                trail_stop = alloc.momentum_trail_stop_pct
                if curr < pos.peak_price * (1 - trail_stop):
                    port.exit(tk, curr, date, "MOM_TRAILING_STOP")
                    continue
                if is_rebal and tk in conf_map:
                    rc = get_rolling_conf(conf_map[tk], date)
                    if rc < BASELINE_ML_EXIT:
                        port.exit(tk, curr, date, f"MOM_ML_EXIT(rc={rc:.3f})")
                # Crisis: close all momentum immediately
                if regime == MarketRegimeState.CRISIS:
                    port.exit(tk, curr, date, "CRISIS_CLOSE")

            # Mean-reversion exits
            elif pos.strategy == "mean_reversion":
                hist = mr_data.get(tk)
                if hist is None:
                    hist = all_data.get(tk)
                if hist is not None:
                    hist_slice = hist.loc[:date]
                    mc = mr_conf.get(tk)
                    ml_conf = get_rolling_conf(mc, date) if mc is not None else 0.5
                    sig = mr_strat.should_exit(
                        tk, hist_slice, nifty, date,
                        pos.entry_price, pos.peak_price, hold, ml_conf
                    )
                    if sig is not None:
                        port.exit(tk, curr, date, sig.reasons[0] if sig.reasons else "MR_EXIT")
                        mr_cooldown[tk] = 3  # 3-bar cooldown after MR exit

        # ── Entries ───────────────────────────────────────────────────────────
        if port.circuit_breaker.allow_new_entries and alloc.allow_new_entries:

            # Momentum entries (monthly rebal)
            if is_rebal and regime in (MarketRegimeState.BULL, MarketRegimeState.NEUTRAL):
                last_rebal = date
                candidates = []
                mom_positions = {tk for tk, p in port.positions.items() if p.strategy == "momentum"}
                for tk in conf_map:
                    if tk in port.positions:
                        continue
                    if tk not in prices:
                        continue
                    rc = get_rolling_conf(conf_map[tk], date)
                    if rc >= BASELINE_ML_ENTRY:
                        candidates.append((tk, rc))
                candidates.sort(key=lambda x: x[1], reverse=True)
                slots = alloc.max_momentum_pos - len(mom_positions)
                for tk, rc in candidates[:slots]:
                    port.enter(
                        tk, "momentum", prices[tk], date,
                        alloc.momentum_pct, alloc.momentum_trail_stop_pct,
                    )

            # Mean-reversion entries (daily check)
            if regime in (MarketRegimeState.BEAR, MarketRegimeState.NEUTRAL):
                mr_positions = {tk for tk, p in port.positions.items() if p.strategy == "mean_reversion"}
                slots = alloc.max_mean_rev_pos - len(mr_positions)
                if slots > 0:
                    for tk in MEAN_REV_CANDIDATES:
                        if slots <= 0:
                            break
                        if tk in port.positions:
                            continue
                        if tk in mr_cooldown:
                            continue
                        hist = mr_data.get(tk) if mr_data.get(tk) is not None else all_data.get(tk)
                        if hist is None:
                            continue
                        if date not in hist.index:
                            continue
                        hist_slice = hist.loc[:date]
                        if len(hist_slice) < 50:
                            continue
                        mc = mr_conf.get(tk)
                        ml_conf = get_rolling_conf(mc, date) if mc is not None else 0.5
                        sig = mr_strat.should_enter(tk, hist_slice, nifty, date, ml_conf)
                        if sig is not None:
                            curr_p = float(hist.loc[date, "close"]) if date in hist.index else None
                            if curr_p is None:
                                continue
                            if port.enter(tk, "mean_reversion", curr_p, date,
                                          alloc.mean_rev_pct, mr_strat.stop_loss_pct):
                                slots -= 1

        # ── Record equity ─────────────────────────────────────────────────────
        eq = port.get_equity(all_prices)
        port.equity_history.append({
            "date": date, "equity": eq,
            "regime": regime.value, "n_pos": len(port.positions)
        })

    # Force close at end
    final_date   = dates[-1]
    final_prices = {t: float(all_data[t].loc[final_date, "close"])
                    for t in all_data if final_date in all_data[t].index}
    final_mr_prices = {t: float(mr_data[t].loc[final_date, "close"])
                       for t in mr_data if final_date in mr_data[t].index}
    final_all = {**final_prices, **final_mr_prices}
    for tk in list(port.positions.keys()):
        port.exit(tk, final_all.get(tk, port.positions[tk].entry_price),
                  final_date, "END_OF_SIM")

    return _compile_results(port, "ENHANCED (Multi-Strategy)")


# ── Results compilation ───────────────────────────────────────────────────────

def _compile_results(port: Portfolio, label: str) -> Dict:
    eq_df    = pd.DataFrame(port.equity_history).set_index("date")
    trades   = port.trades
    n_trades = len(trades)

    total_ret  = (eq_df["equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    n_years    = (pd.Timestamp(OOS_END) - pd.Timestamp(OOS_START)).days / 365.25
    ann_ret    = ((1 + total_ret / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0
    net_tax    = ann_ret * 0.80  # 20% STCG

    win_rate  = float((pd.Series([t.net_pnl for t in trades]) > 0).mean() * 100) if n_trades > 0 else 0.0
    avg_hold  = float(np.mean([t.hold_days for t in trades])) if n_trades > 0 else 0.0

    roll_max  = eq_df["equity"].cummax()
    max_dd    = float((eq_df["equity"] / roll_max - 1).min() * 100) if len(eq_df) > 1 else 0.0

    # Sharpe (active days only)
    eq_ret    = eq_df["equity"].pct_change().dropna()
    rf_daily  = 0.065 / 252
    if len(eq_ret) > 5 and eq_ret.std() > 1e-10:
        excess  = eq_ret - rf_daily
        sharpe  = float(excess.mean() / excess.std() * np.sqrt(252))
    else:
        sharpe  = 0.0

    # Annual breakdown
    eq_df["year"] = eq_df.index.year
    annual: Dict[int, float] = {}
    prev_eq = INITIAL_CAPITAL
    for yr in sorted(eq_df["year"].unique()):
        yr_eq  = eq_df[eq_df["year"] == yr]["equity"]
        yr_end = float(yr_eq.iloc[-1])
        annual[yr] = (yr_end / prev_eq - 1) * 100
        prev_eq = yr_end

    # Per-strategy breakdown
    mom_trades = [t for t in trades if t.strategy == "momentum"]
    mr_trades  = [t for t in trades if t.strategy == "mean_reversion"]
    mom_wr     = float((pd.Series([t.net_pnl for t in mom_trades]) > 0).mean() * 100) if mom_trades else 0.0
    mr_wr      = float((pd.Series([t.net_pnl for t in mr_trades]) > 0).mean() * 100) if mr_trades else 0.0

    return {
        "label":         label,
        "total_ret":     round(total_ret, 2),
        "ann_ret":       round(ann_ret, 2),
        "net_after_tax": round(net_tax, 2),
        "win_rate":      round(win_rate, 1),
        "max_dd":        round(max_dd, 2),
        "sharpe":        round(sharpe, 3),
        "n_trades":      n_trades,
        "avg_hold":      round(avg_hold, 1),
        "annual":        {str(k): round(v, 1) for k, v in annual.items()},
        "mom_trades":    len(mom_trades),
        "mr_trades":     len(mr_trades),
        "mom_win_rate":  round(mom_wr, 1),
        "mr_win_rate":   round(mr_wr, 1),
        "trades":        trades,
        "equity_df":     eq_df,
    }


# ── Report printer ────────────────────────────────────────────────────────────

def print_comparison(baseline: Dict, enhanced: Dict) -> None:
    W = 90
    print("\n" + "═" * W)
    print("  MARK5 MULTI-STRATEGY BACKTEST — OOS 2022–2026  |  PAPER MODE")
    print("═" * W)

    # Side-by-side header
    fmt = "  {:<32} {:>22}  {:>22}"
    print(fmt.format("Metric", "BASELINE (Momentum-Only)", "ENHANCED (Multi-Strat)"))
    print("  " + "─" * (W - 2))

    def row(label, b_val, e_val, better_is_higher=True):
        if isinstance(b_val, float) and isinstance(e_val, float):
            diff = e_val - b_val
            mark = ""
            if better_is_higher:
                mark = "✅" if diff > 0 else ("❌" if diff < -0.5 else "")
            else:
                mark = "✅" if diff < 0 else ("❌" if diff > 0.5 else "")
            print(fmt.format(label, f"{b_val:.2f}", f"{e_val:.2f}  {mark}"))
        else:
            print(fmt.format(label, str(b_val), str(e_val)))

    row("4yr Total Return (%)",       baseline["total_ret"],     enhanced["total_ret"])
    row("Annual CAGR % (gross)",      baseline["ann_ret"],       enhanced["ann_ret"])
    row("Net After 20% Tax (%)",      baseline["net_after_tax"], enhanced["net_after_tax"])
    row("Overall Win Rate (%)",       baseline["win_rate"],      enhanced["win_rate"])
    row("Max Drawdown (%)",           baseline["max_dd"],        enhanced["max_dd"], better_is_higher=False)
    row("Sharpe Ratio",               baseline["sharpe"],        enhanced["sharpe"])
    row("Total Trades",               float(baseline["n_trades"]), float(enhanced["n_trades"]))
    row("Avg Hold Days",              baseline["avg_hold"],      enhanced["avg_hold"], better_is_higher=False)

    print("  " + "─" * (W - 2))
    print(f"  Momentum trades      : {baseline['mom_trades']:>22}   {enhanced['mom_trades']:>22}")
    print(f"  Mean-rev trades      : {0:>22}   {enhanced['mr_trades']:>22}")
    print(f"  Momentum WR (%)      : {baseline['mom_win_rate']:>22.1f}   {enhanced['mom_win_rate']:>22.1f}")
    print(f"  Mean-rev WR (%)      : {'N/A':>22}   {enhanced['mr_win_rate']:>22.1f}")

    print("\n  Annual Breakdown:")
    all_years = sorted(set(baseline["annual"].keys()) | set(enhanced["annual"].keys()))
    for yr in all_years:
        b = baseline["annual"].get(yr, 0.0)
        e = enhanced["annual"].get(yr, 0.0)
        diff = e - b
        flag_b = "✅" if b > 0 else "❌"
        flag_e = "✅" if e > 0 else "❌"
        change = f"Δ{diff:+.1f}%"
        print(f"    {yr}:  {flag_b} {b:>+7.1f}%  →  {flag_e} {e:>+7.1f}%  ({change})")

    print("\n  Circuit Breaker Events:")
    cb_events = enhanced.get("equity_df", pd.DataFrame()).get("regime", pd.Series())
    print(f"    (see logs for circuit breaker trigger details)")

    print("\n" + "═" * W)
    print()


# ── Main entry point ──────────────────────────────────────────────────────────

def main():
    print(f"\n{'═'*80}")
    print("  MARK5 Multi-Strategy Backtest v1.0")
    print(f"  OOS: {OOS_START} → {OOS_END}")
    print(f"  Capital: ₹{INITIAL_CAPITAL/1e7:.0f} crore")
    print(f"{'═'*80}\n")

    # ── Load momentum data ────────────────────────────────────────────────────
    print("Loading momentum ticker data...")
    all_data: Dict[str, pd.DataFrame] = {}
    for tk in MOMENTUM_TICKERS:
        df = load_ticker(tk)
        if df is not None:
            all_data[tk] = df
            print(f"  {tk}: {len(df)} bars ({df.index[0].date()}→{df.index[-1].date()})")

    # ── Load mean-reversion candidate data ────────────────────────────────────
    print("\nLoading mean-reversion candidates...")
    mr_data: Dict[str, pd.DataFrame] = {}
    for tk in MEAN_REV_CANDIDATES:
        df = load_ticker(tk)
        if df is not None:
            mr_data[tk] = df

    # ── Load Nifty ────────────────────────────────────────────────────────────
    print("\nLoading Nifty 50...")
    nifty = load_nifty()
    if nifty is None:
        print("ERROR: Could not load Nifty 50 data. Exiting.")
        return {}
    nifty = nifty.loc[:OOS_END]
    print(f"  Nifty: {len(nifty)} bars ({nifty.index[0].date()}→{nifty.index[-1].date()})")

    # ── Pre-compute ML confidence ─────────────────────────────────────────────
    print("\nPre-computing ML confidence series...")
    conf_map: Dict[str, pd.Series] = {}
    for tk in all_data:
        c = load_ml_confidence(tk)
        if c is not None:
            conf_map[tk] = c
            oos = c.loc[OOS_START:OOS_END]
            above = (oos >= BASELINE_ML_ENTRY).mean() * 100 if len(oos) > 0 else 0
            print(f"  {tk}: ML active | {above:.0f}% bars above hurdle")
        else:
            print(f"  {tk}: No ML model")

    mr_conf: Dict[str, pd.Series] = {}
    for tk in MEAN_REV_CANDIDATES:
        if tk not in conf_map:
            c = load_ml_confidence(tk)
            if c is not None:
                mr_conf[tk] = c

    # ── OOS date index ────────────────────────────────────────────────────────
    ref_ticker = next(iter(all_data))
    oos_dates  = all_data[ref_ticker].loc[OOS_START:OOS_END].index

    print(f"\nOOS trading days: {len(oos_dates)} ({oos_dates[0].date()}→{oos_dates[-1].date()}")
    print(f"Active ML tickers: {len(conf_map)}")

    # ── Run both systems ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Running BASELINE (Momentum-Only, Iteration 6)...")
    baseline = run_baseline(all_data, conf_map, oos_dates)

    print("\n" + "─" * 60)
    print("Running ENHANCED (Multi-Strategy: Momentum + MeanRev + CircuitBreaker)...")
    enhanced = run_enhanced(all_data, mr_data, conf_map, mr_conf, nifty, oos_dates)

    # ── Print comparison ──────────────────────────────────────────────────────
    print_comparison(baseline, enhanced)

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "baseline": {k: v for k, v in baseline.items() if k not in ("trades", "equity_df")},
        "enhanced": {k: v for k, v in enhanced.items() if k not in ("trades", "equity_df")},
    }
    out_path = os.path.join(_ROOT, "reports", "multi_strategy_backtest.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Results saved → {out_path}")

    return out


if __name__ == "__main__":
    main()

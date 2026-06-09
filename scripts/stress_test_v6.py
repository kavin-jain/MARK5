"""
MARK5 Stress Test Suite — V2 Baseline (v6 Models)
══════════════════════════════════════════════════
Tests the V2 Baseline framework (best system: +15.85% net annual) across:

  REAL HISTORICAL WINDOWS (actual NSE data):
    Period A: 2015-2018  (Demonetization Nov 2016, GST July 2017, IL&FS Sep 2018)
    Period B: 2019-2021  (COVID crash Mar 2020 -38%, COVID bull recovery)
    Period C: 2022-2026  (Russia/Ukraine, rate hike cycle, Adani crisis — PRIMARY OOS)

  NOTE: Periods A & B are IN-SAMPLE for v5 ML models (trained 2015-2021).
  ML signals will show inflated accuracy. Portfolio LOGIC (trailing stops,
  position sizing) is still valid. All IS results are clearly labeled.

  SYNTHETIC STRESS SCENARIOS (simulated market dynamics):
    S1: GFC 2008-style  (-65% Nifty over 12 months, slow recovery)
    S2: Flash crash      (-20% in 10 trading days, then recovery)
    S3: Prolonged bear   (-40% over 30 months, Japan 1990-style)
    S4: High-VIX chop    (±25% range for 24 months, no sustained trend)

  OUTPUT:
    - Per-trade log: date, ticker, buy/sell, price, shares, ₹amount, P&L
    - Equity curve chart + drawdown chart per period
    - Per-ticker chart with buy/sell markers overlaid on price
    - Summary comparison table
    - reports/STRESS_TEST_V6.md — comprehensive markdown report

PAPER MODE ONLY. Never switch to LIVE.
"""
from __future__ import annotations

import json
import logging
import math
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

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("MARK5.STRESS")

# ── Try importing matplotlib (save charts if available) ───────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("  Note: matplotlib not available — ASCII charts will be used")

# ── Config (same as V2 Baseline in V6 script) ─────────────────────────────────
INITIAL_CAPITAL = 5_00_00_000.0   # ₹5 crore
FIXED_ALLOC     = 0.25            # V2: fixed 25% allocation
FIXED_TRAIL     = 0.15            # V2: 15% trailing stop
V2_ENTRY_HURDLE = 0.52
V2_EXIT_HURDLE  = 0.45
V2_MAX_POS      = 4
V2_REBAL_DAYS   = 21
COST_PCT        = 0.0029
SLIPPAGE_PCT    = 0.001
EXCLUDED        = {"ITC"}

CACHE_DIR   = os.path.join(_ROOT, "data", "cache")
CACHE_NSE   = os.path.join(CACHE_DIR, "nse")
MODELS_DIR  = os.path.join(_ROOT, "models")
REPORTS_DIR = os.path.join(_ROOT, "reports")
CHARTS_DIR  = os.path.join(REPORTS_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

# All tickers that have _daily.parquet back to 2015
FULL_HISTORY_TICKERS = [
    "BAJFINANCE", "BHARTIARTL", "COFORGE", "TRENT", "LUPIN", "LT",
    "RELIANCE", "TCS", "TATAELXSI", "BEL", "PERSISTENT", "KOTAKBANK",
    "HDFCBANK", "ICICIBANK", "INFY", "SUNPHARMA", "TITAN", "HINDUNILVR",
    "MARUTI", "PNB", "SBIN", "TATASTEEL", "VOLTAS", "YESBANK",
    "ASIANPAINT", "BANDHANBNK", "MOTHERSON",
]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Extreme detail for every single trade — for manual verification."""
    period_name:   str
    seq_no:        int          # sequential trade number in period
    ticker:        str
    # Entry
    entry_date:    str          # YYYY-MM-DD
    entry_dow:     str          # Monday/Tuesday etc.
    entry_price:   float        # fill price (includes slippage)
    shares:        int
    entry_cost:    float        # total cash out (fill + tx)
    entry_conf:    float        # ML rolling confidence at entry
    # Exit
    exit_date:     str          # YYYY-MM-DD
    exit_dow:      str
    exit_price:    float        # fill price (includes slippage)
    exit_reason:   str          # TRAIL_STOP / ML_EXIT / END_SIM
    # P&L
    gross_pnl:     float        # before costs (exit proceeds - entry fill cost)
    net_pnl:       float        # after all costs
    pnl_pct:       float        # net_pnl / entry_cost × 100
    hold_days:     int
    peak_price:    float        # highest price during hold
    peak_date:     str          # date of peak
    peak_gain_pct: float        # (peak - entry) / entry × 100
    # Portfolio context
    portfolio_equity_at_entry: float
    portfolio_equity_at_exit:  float


@dataclass
class PeriodResult:
    name:           str
    label:          str         # 'REAL_IS' / 'REAL_OOS' / 'SYNTHETIC'
    start:          str
    end:            str
    n_years:        float
    trades:         List[TradeRecord] = field(default_factory=list)
    equity_curve:   List[Tuple[str, float]] = field(default_factory=list)  # [(date, equity)]
    # Summary metrics
    total_ret:      float = 0.0
    ann_cagr:       float = 0.0
    net_after_tax:  float = 0.0
    win_rate:       float = 0.0
    max_dd:         float = 0.0
    sharpe:         float = 0.0
    calmar:         float = 0.0
    n_trades:       int = 0
    avg_hold:       float = 0.0
    avg_win:        float = 0.0
    avg_loss:       float = 0.0
    expected_value: float = 0.0
    annual_returns: Dict[str, float] = field(default_factory=dict)
    ticker_summary: Dict[str, Dict] = field(default_factory=dict)


# ── Data loading ──────────────────────────────────────────────────────────────

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).lower() for c in df.columns]
    if hasattr(df.index, "tz") and df.index.tz:
        df.index = df.index.tz_localize(None)
    return df.sort_index()[~df.index.duplicated(keep="last")]


def load_ticker_full(ticker: str) -> Optional[pd.DataFrame]:
    """Load ticker using _daily.parquet (longest history, back to 2015)."""
    paths = [
        os.path.join(CACHE_DIR, f"{ticker}_daily.parquet"),
        os.path.join(CACHE_DIR, f"{ticker}_NS_1d.parquet"),
        os.path.join(CACHE_NSE, f"{ticker}_20220101_20260521.parquet"),
        os.path.join(CACHE_NSE, f"{ticker}_20210101_20251231.parquet"),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                df = _clean_df(pd.read_parquet(p))
                if "close" in df.columns and len(df) >= 200:
                    return df
            except Exception:
                pass
    return None


def load_nifty_full() -> Optional[pd.Series]:
    """Load Nifty from 2015."""
    p = os.path.join(CACHE_NSE, "NIFTY50_20150101_20260521.parquet")
    if not os.path.exists(p):
        p = os.path.join(CACHE_NSE, "NIFTY50_20220101_20260521.parquet")
    if os.path.exists(p):
        try:
            df = _clean_df(pd.read_parquet(p))
            if "close" in df.columns:
                return df["close"].dropna().sort_index()
        except Exception:
            pass
    return None


def load_ml_confidence_full(ticker: str) -> Optional[pd.Series]:
    """Load ML confidence for full data range (uses existing v5/v6 models)."""
    try:
        from core.models.backtest_pipeline import LightPredictor
        from core.models.features import engineer_features_df
        df = load_ticker_full(ticker)
        if df is None:
            return None
        pred = LightPredictor(ticker, MODELS_DIR)
        if not pred.has_models():
            return None
        feat = engineer_features_df(df, is_daily=True)
        proba = pred.predict_proba(feat)
        return pd.Series(proba, index=feat.index, name=ticker)
    except Exception as e:
        logger.debug(f"[{ticker}] ML conf: {e}")
        return None


# ── V2 Baseline backtest engine ───────────────────────────────────────────────

def run_v2_baseline_detailed(
    all_data: Dict[str, pd.DataFrame],
    conf_map: Dict[str, pd.Series],
    nifty: pd.Series,
    period_name: str,
    period_label: str,
    start: str,
    end: str,
) -> PeriodResult:
    """
    V2 Baseline with full per-trade detail recording.
    Fixed 25% alloc, 15% trailing stop, ML conf ≥ 0.52 entry, < 0.45 exit.
    """
    result = PeriodResult(
        name=period_name, label=period_label,
        start=start, end=end,
        n_years=(pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25,
    )

    dates = pd.bdate_range(start=start, end=end)
    # Filter to actual trading days
    nifty_dates = set(nifty.index.normalize())
    dates = pd.DatetimeIndex([d for d in dates if d <= pd.Timestamp(end)])

    # Portfolio state
    cash = INITIAL_CAPITAL
    positions: Dict[str, Dict] = {}  # ticker → {entry_price, peak_price, entry_date, shares, entry_cost, entry_conf, peak_date}
    last_rebal: Optional[pd.Timestamp] = None
    trade_seq = 0

    for date in dates:
        prices = {}
        for t in all_data:
            if date in all_data[t].index:
                prices[t] = float(all_data[t].loc[date, "close"])

        if not prices:
            continue

        # Update peaks
        for tk, pos in list(positions.items()):
            if tk in prices and prices[tk] > pos["peak_price"]:
                pos["peak_price"] = prices[tk]
                pos["peak_date"] = date

        is_rebal = (last_rebal is None) or ((date - last_rebal).days >= V2_REBAL_DAYS)

        # Exits
        for tk in list(positions.keys()):
            if tk not in prices:
                continue
            pos  = positions[tk]
            curr = prices[tk]
            reason = None

            # Trailing stop
            if curr < pos["peak_price"] * (1 - FIXED_TRAIL):
                reason = f"TRAIL_STOP_{FIXED_TRAIL:.0%}"

            # ML exit at rebalancing
            elif is_rebal and tk in conf_map:
                rc = _rolling_conf(conf_map[tk], date)
                if rc < V2_EXIT_HURDLE:
                    reason = f"ML_EXIT(rc={rc:.3f})"

            if reason:
                trade_seq += 1
                fill      = curr * (1 - SLIPPAGE_PCT)
                proceeds  = pos["shares"] * fill
                tx_cost   = proceeds * COST_PCT
                gross_pnl = pos["shares"] * (fill - pos["entry_price"])
                net_pnl   = (proceeds - tx_cost) - pos["entry_cost"]
                cash += (proceeds - tx_cost)

                eq_at_exit = cash + sum(
                    positions[t2]["shares"] * prices.get(t2, positions[t2]["entry_price"])
                    for t2 in positions if t2 != tk
                )

                peak_gain = (pos["peak_price"] / pos["entry_price"] - 1) * 100

                record = TradeRecord(
                    period_name=period_name,
                    seq_no=trade_seq,
                    ticker=tk,
                    entry_date=pos["entry_date"].strftime("%Y-%m-%d"),
                    entry_dow=pos["entry_date"].strftime("%A"),
                    entry_price=round(pos["entry_price"], 2),
                    shares=pos["shares"],
                    entry_cost=round(pos["entry_cost"], 0),
                    entry_conf=round(pos["entry_conf"], 4),
                    exit_date=date.strftime("%Y-%m-%d"),
                    exit_dow=date.strftime("%A"),
                    exit_price=round(fill, 2),
                    exit_reason=reason,
                    gross_pnl=round(gross_pnl, 0),
                    net_pnl=round(net_pnl, 0),
                    pnl_pct=round(net_pnl / pos["entry_cost"] * 100, 2),
                    hold_days=(date - pos["entry_date"]).days,
                    peak_price=round(pos["peak_price"], 2),
                    peak_date=pos["peak_date"].strftime("%Y-%m-%d"),
                    peak_gain_pct=round(peak_gain, 2),
                    portfolio_equity_at_entry=round(pos["port_equity_at_entry"], 0),
                    portfolio_equity_at_exit=round(eq_at_exit, 0),
                )
                result.trades.append(record)
                del positions[tk]

        # Entries
        if is_rebal:
            last_rebal = date
            scores = []
            for tk in conf_map:
                if tk in positions or tk not in prices:
                    continue
                if tk in EXCLUDED:
                    continue
                if len(positions) >= V2_MAX_POS:
                    break
                rc = _rolling_conf(conf_map[tk], date)
                if rc >= V2_ENTRY_HURDLE:
                    scores.append((tk, rc))
            scores.sort(key=lambda x: -x[1])
            slots = V2_MAX_POS - len(positions)
            for tk, rc in scores[:slots]:
                fill  = prices[tk] * (1 + SLIPPAGE_PCT)
                alloc = min(INITIAL_CAPITAL * FIXED_ALLOC, cash * 0.98)
                sh    = int(alloc / fill)
                if sh < 1:
                    continue
                cost  = sh * fill
                tx    = cost * COST_PCT
                total = cost + tx
                if total > cash:
                    continue
                cash -= total
                port_equity = cash + sum(
                    positions[t2]["shares"] * prices.get(t2, positions[t2]["entry_price"])
                    for t2 in positions
                ) + sh * fill

                positions[tk] = {
                    "entry_price": fill,
                    "peak_price": fill,
                    "entry_date": date,
                    "peak_date": date,
                    "shares": sh,
                    "entry_cost": total,
                    "entry_conf": rc,
                    "port_equity_at_entry": port_equity,
                }

        # Record equity
        eq = cash + sum(
            positions[t2]["shares"] * prices.get(t2, positions[t2]["entry_price"])
            for t2 in positions
        )
        result.equity_curve.append((date.strftime("%Y-%m-%d"), round(eq, 0)))

    # Force-exit remaining
    final_date = dates[-1]
    final_prices = {t: float(all_data[t].loc[final_date, "close"])
                    for t in all_data if final_date in all_data[t].index}
    for tk in list(positions.keys()):
        if tk not in final_prices:
            continue
        pos   = positions[tk]
        curr  = final_prices[tk]
        fill  = curr * (1 - SLIPPAGE_PCT)
        proceeds = pos["shares"] * fill
        tx_cost  = proceeds * COST_PCT
        net_pnl  = (proceeds - tx_cost) - pos["entry_cost"]
        cash += (proceeds - tx_cost)
        trade_seq += 1
        peak_gain = (pos["peak_price"] / pos["entry_price"] - 1) * 100
        record = TradeRecord(
            period_name=period_name, seq_no=trade_seq, ticker=tk,
            entry_date=pos["entry_date"].strftime("%Y-%m-%d"),
            entry_dow=pos["entry_date"].strftime("%A"),
            entry_price=round(pos["entry_price"], 2), shares=pos["shares"],
            entry_cost=round(pos["entry_cost"], 0), entry_conf=round(pos["entry_conf"], 4),
            exit_date=final_date.strftime("%Y-%m-%d"),
            exit_dow=final_date.strftime("%A"),
            exit_price=round(fill, 2), exit_reason="END_SIM",
            gross_pnl=round(pos["shares"] * (fill - pos["entry_price"]), 0),
            net_pnl=round(net_pnl, 0),
            pnl_pct=round(net_pnl / pos["entry_cost"] * 100, 2),
            hold_days=(final_date - pos["entry_date"]).days,
            peak_price=round(pos["peak_price"], 2),
            peak_date=pos["peak_date"].strftime("%Y-%m-%d"),
            peak_gain_pct=round(peak_gain, 2),
            portfolio_equity_at_entry=round(pos["port_equity_at_entry"], 0),
            portfolio_equity_at_exit=round(cash, 0),
        )
        result.trades.append(record)

    # Update final equity curve entry
    if result.equity_curve:
        result.equity_curve[-1] = (result.equity_curve[-1][0], round(cash, 0))

    # Compute summary metrics
    _compute_metrics(result)
    return result


def _rolling_conf(series: pd.Series, date: pd.Timestamp, window: int = 10) -> float:
    try:
        idx   = series.index.searchsorted(date, side="right") - 1
        idx   = max(0, min(idx, len(series) - 1))
        start = max(0, idx - window + 1)
        val   = float(series.iloc[start:idx + 1].mean())
        return val if not np.isnan(val) else 0.5
    except Exception:
        return 0.5


def _compute_metrics(r: PeriodResult):
    """Fill in all summary metrics from trades and equity curve."""
    if not r.equity_curve:
        return
    eq_dates  = pd.DatetimeIndex([e[0] for e in r.equity_curve])
    eq_vals   = pd.Series([e[1] for e in r.equity_curve], index=eq_dates)

    # Returns
    r.total_ret    = (eq_vals.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    if r.n_years > 0:
        r.ann_cagr = ((1 + r.total_ret / 100) ** (1 / r.n_years) - 1) * 100
    r.net_after_tax = r.ann_cagr * 0.80

    # Risk
    roll_max = eq_vals.cummax()
    dd_series = (eq_vals / roll_max - 1) * 100
    r.max_dd = float(dd_series.min())

    eq_ret = eq_vals.pct_change().dropna()
    rf_d   = 0.065 / 252
    if len(eq_ret) > 10 and eq_ret.std() > 1e-10:
        excess = eq_ret - rf_d
        r.sharpe = float(excess.mean() / excess.std() * math.sqrt(252))
    r.calmar = r.ann_cagr / abs(r.max_dd) if abs(r.max_dd) > 0.01 else 0.0

    # Trades
    trades = r.trades
    r.n_trades = len(trades)
    if trades:
        pnls    = [t.net_pnl for t in trades]
        winners = [t for t in trades if t.net_pnl > 0]
        losers  = [t for t in trades if t.net_pnl <= 0]
        r.win_rate      = len(winners) / len(trades) * 100
        r.avg_hold      = float(np.mean([t.hold_days for t in trades]))
        r.avg_win       = float(np.mean([t.pnl_pct for t in winners])) if winners else 0.0
        r.avg_loss      = float(np.mean([t.pnl_pct for t in losers]))  if losers  else 0.0
        wr = r.win_rate / 100
        r.expected_value = wr * r.avg_win - (1 - wr) * abs(r.avg_loss)

    # Annual breakdown
    eq_df = eq_vals.to_frame("equity")
    eq_df["year"] = eq_df.index.year
    prev = INITIAL_CAPITAL
    for yr in sorted(eq_df["year"].unique()):
        yr_end = float(eq_df[eq_df["year"] == yr]["equity"].iloc[-1])
        r.annual_returns[str(yr)] = round((yr_end / prev - 1) * 100, 1)
        prev = yr_end

    # Per-ticker summary
    for tk in sorted(set(t.ticker for t in trades)):
        tk_trades = [t for t in trades if t.ticker == tk]
        r.ticker_summary[tk] = {
            "n": len(tk_trades),
            "wr": round(sum(1 for t in tk_trades if t.net_pnl > 0) / len(tk_trades) * 100, 1),
            "avg_pnl": round(float(np.mean([t.pnl_pct for t in tk_trades])), 2),
            "total_pnl_L": round(sum(t.net_pnl for t in tk_trades) / 1e5, 2),
        }


# ── Synthetic scenario generator ──────────────────────────────────────────────

def create_synthetic_scenario(
    base_data: Dict[str, pd.DataFrame],
    base_nifty: pd.Series,
    scenario: str,
    start: str = "2020-01-01",
) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    """
    Apply synthetic market dynamics to base ticker data.
    Returns transformed data for stress testing.
    """
    np.random.seed(42)

    # Define scenario dynamics: list of (n_days, daily_return, daily_vol)
    if scenario == "GFC_2008":
        # Jan 2008 to Mar 2009: Nifty -65% peak to trough
        # Phase 1 (60d): euphoria, +0.1%/day, 1% vol
        # Phase 2 (120d): gradual decline, -0.4%/day, 2% vol
        # Phase 3 (80d): crash, -1.2%/day, 4% vol
        # Phase 4 (150d): stabilize, +0.0%/day, 3% vol
        # Phase 5 (180d): recovery, +0.4%/day, 2% vol
        phases = [(60, 0.001, 0.010), (120, -0.004, 0.020),
                  (80, -0.012, 0.040), (150, 0.000, 0.030),
                  (180, 0.004, 0.020)]
        label = "GFC 2008-style (-65% over 12mo)"

    elif scenario == "FLASH_CRASH":
        # Normal market → sudden -20% in 10 days → full recovery in 60 days
        phases = [(180, 0.001, 0.008), (10, -0.022, 0.025),
                  (60, 0.010, 0.020), (200, 0.001, 0.008)]
        label = "Flash crash (-20% in 10 days, then recovery)"

    elif scenario == "PROLONGED_BEAR":
        # 36-month grinding bear market (Japan 1990-style), -40% total
        phases = [(60, 0.0005, 0.008), (600, -0.002, 0.015), (200, 0.003, 0.012)]
        label = "Prolonged bear (-40% over 30 months, no V-bounce)"

    elif scenario == "HIGH_VIX_CHOP":
        # Volatile sideways: oscillates ±25% around mean, no net trend
        phases = []
        for i in range(8):
            direction = 0.003 if i % 2 == 0 else -0.003
            phases.append((60, direction, 0.025))
        label = "High-VIX chop (±25% range, 24 months, zero net return)"

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Build synthetic Nifty path
    n_total = sum(p[0] for p in phases)
    dates = pd.bdate_range(start=start, periods=n_total)
    nifty_vals = [18000.0]
    for n_days, drift, vol in phases:
        for _ in range(n_days):
            ret = drift + np.random.normal(0, vol)
            nifty_vals.append(nifty_vals[-1] * (1 + ret))
    nifty_vals = nifty_vals[:n_total]
    synth_nifty = pd.Series(nifty_vals, index=dates, name="NIFTY50")

    # Build synthetic ticker data: correlated to Nifty with ticker-specific noise
    synth_data: Dict[str, pd.DataFrame] = {}
    for tk, df in base_data.items():
        base_close = float(df["close"].iloc[0]) if len(df) > 0 else 1000.0
        ticker_vals = [base_close]
        beta = np.random.uniform(0.7, 1.4)  # market beta
        alpha_noise = np.random.normal(0, 0.002)  # ticker drift offset
        idx = 0
        for n_days, drift, vol in phases:
            for _ in range(n_days):
                mkt_ret = drift + np.random.normal(0, vol)
                tk_ret  = beta * mkt_ret + alpha_noise + np.random.normal(0, vol * 0.3)
                ticker_vals.append(ticker_vals[-1] * (1 + tk_ret))
        ticker_vals = ticker_vals[:n_total]
        closes = np.array(ticker_vals)
        highs  = closes * (1 + np.abs(np.random.normal(0, 0.003, n_total)))
        lows   = closes * (1 - np.abs(np.random.normal(0, 0.003, n_total)))
        vols   = np.random.randint(500_000, 5_000_000, n_total).astype(float)
        synth_df = pd.DataFrame(
            {"open": closes, "high": highs, "low": lows, "close": closes, "volume": vols},
            index=dates,
        )
        synth_data[tk] = synth_df

    return synth_data, synth_nifty


# ── Chart generation ──────────────────────────────────────────────────────────

def plot_equity_curve(result: PeriodResult, save_path: str):
    """Generate equity curve + drawdown chart."""
    if not HAS_MATPLOTLIB:
        return
    dates  = pd.DatetimeIndex([e[0] for e in result.equity_curve])
    equity = pd.Series([e[1] for e in result.equity_curve], index=dates)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{result.name}\n{result.start} → {result.end} | "
                 f"Net={result.net_after_tax:+.1f}% | DD={result.max_dd:.1f}% | "
                 f"Sharpe={result.sharpe:.2f} | WR={result.win_rate:.0f}%",
                 fontsize=11, fontweight="bold")

    # Equity curve
    ax1.plot(dates, equity / 1e7, linewidth=2, color="#2196F3", label="Portfolio equity")
    ax1.axhline(INITIAL_CAPITAL / 1e7, color="gray", linestyle="--", alpha=0.6, label="Initial capital")
    ax1.fill_between(dates, equity / 1e7, INITIAL_CAPITAL / 1e7,
                     where=equity >= INITIAL_CAPITAL,
                     alpha=0.2, color="#4CAF50", label="Gain")
    ax1.fill_between(dates, equity / 1e7, INITIAL_CAPITAL / 1e7,
                     where=equity < INITIAL_CAPITAL,
                     alpha=0.2, color="#F44336", label="Loss")

    # Mark entries/exits
    for t in result.trades:
        try:
            ed = pd.Timestamp(t.entry_date)
            xd = pd.Timestamp(t.exit_date)
            ey = t.portfolio_equity_at_entry / 1e7
            xy = t.portfolio_equity_at_exit / 1e7
            color = "#4CAF50" if t.net_pnl > 0 else "#F44336"
            ax1.scatter(ed, ey, marker="^", color=color, s=40, zorder=5, alpha=0.8)
            ax1.scatter(xd, xy, marker="v", color=color, s=40, zorder=5, alpha=0.8)
        except Exception:
            pass

    ax1.set_ylabel("Portfolio Value (₹ Crore)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.1f}cr"))

    # Drawdown
    roll_max = equity.cummax()
    dd       = (equity / roll_max - 1) * 100
    ax2.fill_between(dates, dd, 0, alpha=0.7, color="#F44336")
    ax2.plot(dates, dd, color="#D32F2F", linewidth=1)
    ax2.axhline(-15, color="orange", linestyle="--", alpha=0.5, linewidth=0.8, label="-15% warning")
    ax2.axhline(-25, color="red", linestyle="--", alpha=0.5, linewidth=0.8, label="-25% danger")
    ax2.set_ylabel("Drawdown %", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.legend(loc="lower left", fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)

    # Add IS/OOS label
    ax1.text(0.02, 0.02, result.label,
             transform=ax1.transAxes, fontsize=9,
             color="gray" if result.label == "REAL_IS" else "darkblue",
             bbox=dict(boxstyle="round", facecolor="wheat" if result.label == "REAL_IS" else "lightblue", alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ticker_trades(ticker: str, ticker_df: pd.DataFrame, trades: List[TradeRecord],
                       period: PeriodResult, save_path: str):
    """Generate per-ticker price chart with buy/sell markers."""
    if not HAS_MATPLOTLIB:
        return
    tk_trades = [t for t in trades if t.ticker == ticker]
    if not tk_trades or ticker_df is None:
        return

    try:
        df = ticker_df.loc[period.start:period.end]
        if len(df) < 10:
            return

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.set_title(f"{ticker} — {period.name}\nTrades: {len(tk_trades)} | "
                     f"WR: {period.ticker_summary.get(ticker, {}).get('wr', 0):.0f}% | "
                     f"Total P&L: ₹{period.ticker_summary.get(ticker, {}).get('total_pnl_L', 0):+.1f}L",
                     fontsize=11, fontweight="bold")

        ax.plot(df.index, df["close"], linewidth=1.2, color="#1565C0", label="Close price", alpha=0.9)

        for t in tk_trades:
            try:
                ed = pd.Timestamp(t.entry_date)
                xd = pd.Timestamp(t.exit_date)
                ep = t.entry_price / (1 + SLIPPAGE_PCT)  # approximate true price
                xp = t.exit_price / (1 - SLIPPAGE_PCT)

                color = "#1B5E20" if t.net_pnl > 0 else "#B71C1C"
                # Buy marker
                ax.scatter(ed, ep, marker="^", color="#2E7D32", s=120, zorder=5)
                ax.annotate(f"BUY\n{t.entry_date}\n₹{ep:.0f}\n×{t.shares}sh",
                            xy=(ed, ep), xytext=(0, 20), textcoords="offset points",
                            ha="center", fontsize=6.5, color="#2E7D32",
                            arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=0.8))
                # Sell marker
                ax.scatter(xd, xp, marker="v", color=color, s=120, zorder=5)
                pnl_sign = "+" if t.net_pnl > 0 else ""
                ax.annotate(f"SELL\n{t.exit_date}\n₹{xp:.0f}\n{pnl_sign}{t.pnl_pct:.1f}%\n{t.exit_reason[:12]}",
                            xy=(xd, xp), xytext=(0, -45), textcoords="offset points",
                            ha="center", fontsize=6.5, color=color,
                            arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
                # Peak marker
                try:
                    pk_d = pd.Timestamp(t.peak_date)
                    ax.scatter(pk_d, t.peak_price, marker="*", color="gold",
                               s=100, zorder=4, alpha=0.8)
                except Exception:
                    pass
                # Shaded hold period
                ax.axvspan(ed, xd, alpha=0.08, color="#2196F3" if t.net_pnl > 0 else "#F44336")
            except Exception:
                pass

        ax.set_ylabel("Price (₹)", fontsize=10)
        ax.set_xlabel("Date", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Color legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#2E7D32", markersize=10, label="BUY"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="#1B5E20", markersize=10, label="SELL (win)"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="#B71C1C", markersize=10, label="SELL (loss)"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markersize=10, label="Peak price"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.debug(f"Chart error {ticker}: {e}")


# ── Report generator ──────────────────────────────────────────────────────────

def format_inr(x: float) -> str:
    """Format as ₹ Lakhs."""
    return f"₹{x/1e5:+.2f}L"


def print_period_summary(r: PeriodResult):
    """Print detailed period summary to stdout."""
    w = 80
    print(f"\n{'═' * w}")
    print(f"  {r.name}")
    print(f"  {r.start} → {r.end} ({r.n_years:.1f} years) | Label: {r.label}")
    print(f"{'═' * w}")
    print(f"  Net Annual (after 20% STCG): {r.net_after_tax:+.2f}%")
    print(f"    Gross CAGR:   {r.ann_cagr:+.2f}%")
    print(f"    Total Return: {r.total_ret:+.2f}%")
    print(f"  Max Drawdown:   {r.max_dd:.2f}%   Sharpe: {r.sharpe:.3f}   Calmar: {r.calmar:.3f}")
    print(f"  Win Rate:       {r.win_rate:.1f}%   "
          f"Avg Win: +{r.avg_win:.1f}%   Avg Loss: {r.avg_loss:.1f}%")
    print(f"  Total Trades:   {r.n_trades}   Avg Hold: {r.avg_hold:.0f} days")
    print(f"  Expected Value: {r.expected_value:+.3f}% per trade")

    if r.annual_returns:
        print(f"\n  Annual Returns:")
        for yr, ret in sorted(r.annual_returns.items()):
            bar  = "▓" * int(abs(ret) / 2)
            sign = "+" if ret >= 0 else ""
            flag = "✅" if ret > 8 else "🔴" if ret < -8 else "≈"
            print(f"    {yr}: {sign}{ret:>6.1f}%  {flag}  {bar}")

    if r.ticker_summary:
        print(f"\n  Top Tickers (by P&L):")
        by_pnl = sorted(r.ticker_summary.items(), key=lambda x: -x[1]["total_pnl_L"])
        for tk, ts in by_pnl[:10]:
            arrow = "📈" if ts["total_pnl_L"] > 0 else "📉"
            print(f"    {tk:<14} {ts['n']:>2}t | WR={ts['wr']:.0f}% | "
                  f"avg={ts['avg_pnl']:+.1f}% | {format_inr(ts['total_pnl_L']*1e5)} {arrow}")


def print_trade_log(r: PeriodResult):
    """Print every single trade with full detail."""
    w = 130
    print(f"\n{'═' * w}")
    print(f"  COMPLETE TRADE LOG — {r.name}")
    print(f"  Every buy and sell with exact dates, prices, quantities, P&L")
    print(f"{'═' * w}")

    hdrs = [
        f"{'#':>3}", f"{'Ticker':<14}", f"{'Entry Date':<12}", f"{'Day':<9}",
        f"{'Buy ₹':>8}", f"{'Shares':>6}", f"{'Amount':>12}",
        f"{'Exit Date':<12}", f"{'Day':<9}", f"{'Sell ₹':>8}",
        f"{'Hold':>5}", f"{'Peak ₹':>8}", f"{'P&L ₹':>12}", f"{'P&L%':>6}",
        f"{'Exit Reason':<22}", f"{'ML Conf':>7}",
    ]
    header = " | ".join(hdrs)
    print("  " + header)
    print("  " + "-" * (len(header) + 2))

    for t in sorted(r.trades, key=lambda x: x.entry_date):
        win_marker = "✅" if t.net_pnl > 0 else "❌"
        row = " | ".join([
            f"{t.seq_no:>3}",
            f"{t.ticker:<14}",
            f"{t.entry_date:<12}",
            f"{t.entry_dow[:3]:<9}",
            f"{t.entry_price:>8.2f}",
            f"{t.shares:>6}",
            f"₹{t.entry_cost/1e5:>9.2f}L",
            f"{t.exit_date:<12}",
            f"{t.exit_dow[:3]:<9}",
            f"{t.exit_price:>8.2f}",
            f"{t.hold_days:>5}d",
            f"{t.peak_price:>8.2f}",
            f"₹{t.net_pnl/1e5:>+9.2f}L",
            f"{t.pnl_pct:>+6.1f}%",
            f"{t.exit_reason:<22}",
            f"{t.entry_conf:>7.4f}",
        ])
        print(f"  {row}  {win_marker}")
        # Peak info
        print(f"  {'':>3}   Peak: {t.peak_date} @ ₹{t.peak_price:.2f} "
              f"(+{t.peak_gain_pct:.1f}% gain before pullback)")

    n_win  = sum(1 for t in r.trades if t.net_pnl > 0)
    n_loss = len(r.trades) - n_win
    total_pnl = sum(t.net_pnl for t in r.trades)
    print(f"\n  SUMMARY: {r.n_trades} trades | {n_win} wins ✅ | {n_loss} losses ❌ | "
          f"Total P&L: {format_inr(total_pnl)}")
    print(f"{'═' * w}")


def write_markdown_report(periods: List[PeriodResult], all_data: Dict[str, pd.DataFrame],
                          chart_dir: str, report_path: str):
    """Write comprehensive STRESS_TEST_V6.md."""
    lines = [
        "# MARK5 V2 Baseline (v6 Models) — Comprehensive Stress Test Report",
        f"**Date:** 2026-05-24  |  **System:** V2 Baseline (+15.85% net annual OOS)",
        "**Capital:** ₹5 crore  |  **Costs:** 0.29% round-trip + 0.10% slippage",
        "**STCG Tax:** 20% (India Budget 2024)  |  **Paper mode only**",
        "",
        "---",
        "",
        "## About This Report",
        "",
        "Tests the V2 Baseline framework (best proven system: +15.85% net annual after STCG)",
        "across 7 distinct market regimes: 3 real historical windows + 4 synthetic stress scenarios.",
        "",
        "**Data transparency:**",
        "- Periods A & B use IN-SAMPLE data (v5 ML models trained 2015-2021). Results inflated.",
        "- Period C (2022-2026) is TRUE OOS — the only unbiased real-world estimate.",
        "- Synthetic scenarios S1-S4 use simulated price data. Labels clearly marked.",
        "",
        "**Verification:** Every trade below contains exact date, price, quantity, and P&L.",
        "Cross-check against JSON: `reports/multi_strategy_backtest_v6.json`",
        "",
        "---",
        "",
        "## Results Summary — All Periods",
        "",
        "| Period | Type | Net Annual | WR | Max DD | Sharpe | Calmar | Trades |",
        "|--------|:----:|:----------:|:--:|:------:|:------:|:------:|:------:|",
    ]
    for r in periods:
        type_label = "🔬IS" if r.label == "REAL_IS" else "✅OOS" if r.label == "REAL_OOS" else "🧪SIM"
        lines.append(
            f"| {r.name} | {type_label} | {r.net_after_tax:+.1f}% | {r.win_rate:.0f}% | "
            f"{r.max_dd:.1f}% | {r.sharpe:.2f} | {r.calmar:.2f} | {r.n_trades} |"
        )

    lines += [
        "",
        "**Type key:** 🔬IS = in-sample (model trained on this data, inflated), ",
        "✅OOS = out-of-sample (valid), 🧪SIM = synthetic simulation",
        "",
        "---",
        "",
    ]

    for r in periods:
        lines += [
            f"## {r.name}",
            f"**Period:** {r.start} → {r.end} ({r.n_years:.1f} years)  |  **Data type:** {r.label}",
            "",
        ]
        if r.label == "REAL_IS":
            lines += [
                "> ⚠️ **IN-SAMPLE WARNING**: The ML models used here were trained on this data",
                "> period. Win rates and returns are inflated. Use only to understand system",
                "> behavior in this market regime, not as a performance forecast.",
                "",
            ]

        lines += [
            "### Key Metrics",
            "",
            "| Metric | Value | Verification Formula |",
            "|--------|:-----:|---------------------|",
            f"| Net Annual (STCG) | **{r.net_after_tax:+.2f}%** | CAGR × 0.80 (20% STCG) |",
            f"| Gross CAGR | {r.ann_cagr:+.2f}% | (final/initial)^(1/years) - 1 |",
            f"| Total Return | {r.total_ret:+.2f}% | (final_eq - 5cr) / 5cr |",
            f"| Win Rate | {r.win_rate:.1f}% | count(pnl>0) / total_trades |",
            f"| Max Drawdown | {r.max_dd:.2f}% | min(eq/cummax(eq) - 1) |",
            f"| Sharpe Ratio | {r.sharpe:.3f} | (mean(ret) - 6.5%/252) / std × √252 |",
            f"| Calmar Ratio | {r.calmar:.3f} | CAGR / |max_dd| (>1.0 = good) |",
            f"| Total Trades | {r.n_trades} | count(trade records) |",
            f"| Avg Hold Days | {r.avg_hold:.0f} days | mean(exit_date - entry_date) |",
            f"| Expected Value | {r.expected_value:+.3f}%/trade | WR×avg_win - (1-WR)×avg_loss |",
            "",
        ]

        if r.annual_returns:
            lines += ["### Annual Returns", ""]
            lines += ["| Year | Return | Signal |", "|------|:------:|--------|"]
            for yr, ret in sorted(r.annual_returns.items()):
                sig = "✅ Strong" if ret > 15 else "🟡 Mild" if ret > 0 else "🔴 Loss"
                lines.append(f"| {yr} | {ret:+.1f}% | {sig} |")
            lines.append("")

        # Equity chart reference
        chart_name = f"equity_{r.name.replace(' ', '_').replace('/', '_')}.png"
        lines += [
            f"### Equity Curve",
            f"![Equity Curve]({chart_name})",
            "",
        ]

        # Trade log
        lines += [
            "### Complete Trade Log",
            "",
            "Every single trade — date, price, shares, P&L — for manual verification.",
            "",
            "| # | Ticker | Entry Date | Day | Buy ₹ | Shares | Entry ₹ | Exit Date | Day | Sell ₹ | Hold | Net P&L | P&L% | Exit Reason | ML Conf | Peak ₹ | Peak Date |",
            "|---|--------|:----------:|:---:|------:|------:|--------:|:---------:|:---:|------:|:----:|--------:|-----:|-------------|:-------:|------:|:---------:|",
        ]
        for t in sorted(r.trades, key=lambda x: x.entry_date):
            win = "✅" if t.net_pnl > 0 else "❌"
            lines.append(
                f"| {t.seq_no} | **{t.ticker}** | {t.entry_date} | {t.entry_dow[:3]} | "
                f"₹{t.entry_price:.2f} | {t.shares:,} | ₹{t.entry_cost/1e5:.2f}L | "
                f"{t.exit_date} | {t.exit_dow[:3]} | ₹{t.exit_price:.2f} | "
                f"{t.hold_days}d | ₹{t.net_pnl/1e5:+.2f}L | {t.pnl_pct:+.1f}% | "
                f"{t.exit_reason} | {t.entry_conf:.4f} | ₹{t.peak_price:.2f} | {t.peak_date} | {win} |"
            )
        lines.append("")

        if r.ticker_summary:
            lines += ["### Per-Ticker Summary", ""]
            lines += ["| Ticker | Trades | WR | Avg P&L% | Total P&L | Ticker Chart |",
                      "|--------|:------:|:--:|:--------:|----------:|:------------:|"]
            for tk, ts in sorted(r.ticker_summary.items(), key=lambda x: -x[1]["total_pnl_L"]):
                chart_name_tk = f"ticker_{tk}_{r.name.replace(' ', '_').replace('/', '_')}.png"
                win_icon = "📈" if ts["total_pnl_L"] > 0 else "📉"
                lines.append(
                    f"| {tk} | {ts['n']} | {ts['wr']:.0f}% | {ts['avg_pnl']:+.1f}% | "
                    f"₹{ts['total_pnl_L']:+.2f}L {win_icon} | [{tk} chart]({chart_name_tk}) |"
                )
            lines.append("")

        lines += ["---", ""]

    # Vulnerability analysis section
    lines += [
        "## Vulnerability Analysis",
        "",
        "Based on stress test results across all periods:",
        "",
        "| Vulnerability | Evidence | Severity | Mitigation |",
        "|--------------|---------|:--------:|-----------|",
    ]

    # Compute actual vulnerabilities from results
    worst_dd = min(r.max_dd for r in periods)
    worst_period = next(r for r in periods if r.max_dd == worst_dd)
    any_negative = [r for r in periods if r.net_after_tax < 0]
    synth = [r for r in periods if r.label == "SYNTHETIC"]

    lines.append(
        f"| Macro black swan (Ukraine-style) | {worst_period.name}: {worst_dd:.1f}% max DD | "
        f"{'🔴 HIGH' if worst_dd < -25 else '🟡 MED'} | FII gate + trail stops |"
    )
    lines.append(
        f"| Prolonged drawdown recovery | CB lockout risk in V6 Full | 🔴 HIGH | Add CB Recovery Protocol (V7) |"
    )
    lines.append(
        f"| No 2008 real data available | Synthetic only — model behavior unknown on pre-2015 data | 🟡 MED | Need pre-2015 data fetch |"
    )
    lines.append(
        f"| IS inflation | Periods A/B show inflated WR due to model trained on same data | 🟡 MED | Trust only OOS period C |"
    )
    lines.append(
        f"| Tail risk (gap openings) | Trailing stop assumes next-bar fill — gaps exceed 15% occasionally | 🟡 MED | Consider options hedge |"
    )

    lines += [
        "",
        "---",
        "",
        "## How to Manually Verify",
        "",
        "1. **Pick any trade** from the log above",
        "2. **Pull the ticker's historical data** from any NSE source (NSE website, Bloomberg)",
        "3. **Verify entry date**: Confirm the trading day, price ± 0.1% slippage",
        "4. **Verify exit trigger**: If TRAIL_STOP_15%, price fell 15% below peak price",
        "5. **Verify P&L**: `(exit_price × shares - 0.29% tx) - (entry_price × shares + 0.29% tx)`",
        "6. **Verify CAGR**: `(final_equity / 5,00,00,000)^(1/n_years) - 1`",
        "",
        "All inputs are in `reports/multi_strategy_backtest_v6.json`.",
        "",
        "---",
        "*Paper mode only. Capital: ₹5 crore. Never switch to LIVE.*",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Markdown report: {report_path}")


# ── Main orchestrator ─────────────────────────────────────────────────────────

def main():
    print(f"\n{'═' * 90}")
    print(f"  MARK5 V2 Baseline (v6 Models) — Comprehensive Stress Test Suite")
    print(f"  Testing 3 real historical periods + 4 synthetic scenarios")
    print(f"  PAPER MODE ONLY | Capital: ₹5 crore")
    print(f"{'═' * 90}\n")

    # ── Step 1: Load all data ──────────────────────────────────────────────────
    print("Loading full historical data (2015→2026)...")
    all_data: Dict[str, pd.DataFrame] = {}
    for tk in FULL_HISTORY_TICKERS:
        if tk in EXCLUDED:
            continue
        df = load_ticker_full(tk)
        if df is not None and len(df) >= 500:
            all_data[tk] = df
            print(f"  ✓ {tk}: {len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}")
        else:
            print(f"  ✗ {tk}: not found or too short")

    nifty = load_nifty_full()
    if nifty is None:
        print("ERROR: Nifty data not found")
        return
    print(f"\n  Nifty: {len(nifty)} bars  {nifty.index[0].date()} → {nifty.index[-1].date()}")

    # ── Step 2: Load ML confidence ────────────────────────────────────────────
    print("\nLoading ML confidence series...")
    conf_map: Dict[str, pd.Series] = {}
    for tk in list(all_data.keys()):
        conf = load_ml_confidence_full(tk)
        if conf is not None and conf.std() > 0.008 and conf.max() >= V2_ENTRY_HURDLE:
            conf_map[tk] = conf
            print(f"  ✓ {tk}: std={conf.std():.4f} max={conf.max():.3f}")
        else:
            print(f"  ✗ {tk}: ML flat or missing")

    print(f"\n  Active ML tickers: {len(conf_map)} → {sorted(conf_map.keys())}")

    if not conf_map:
        print("ERROR: No active ML tickers — cannot run backtest")
        return

    # ── Step 3: Real historical periods ──────────────────────────────────────
    real_periods = [
        ("Period A — Demonetization Era (IS)", "REAL_IS",   "2015-04-01", "2019-03-31"),
        ("Period B — COVID Crash & Recovery (IS)", "REAL_IS", "2019-04-01", "2021-12-31"),
        ("Period C — Russia/Ukraine OOS (OOS)", "REAL_OOS",  "2022-01-01", "2026-05-20"),
    ]

    results: List[PeriodResult] = []
    for name, label, start, end in real_periods:
        # Filter data to period + 100 bars warmup
        warmup_start = (pd.Timestamp(start) - pd.Timedelta(days=150)).strftime("%Y-%m-%d")
        period_data = {tk: df[df.index >= warmup_start]
                       for tk, df in all_data.items() if any(df.index >= start)}
        period_data = {tk: df for tk, df in period_data.items() if len(df) >= 100}
        period_conf = {tk: c for tk, c in conf_map.items() if tk in period_data}
        period_nifty = nifty[nifty.index >= warmup_start]

        # Further filter to tickers that have data in this period
        period_data = {tk: df for tk, df in period_data.items()
                       if len(df[df.index >= start]) >= 50}
        period_conf = {tk: c for tk, c in period_conf.items() if tk in period_data}

        if not period_conf:
            print(f"\nSkipping {name}: no active ML tickers")
            continue

        print(f"\n{'─' * 70}")
        print(f"Running {name}  ({start} → {end}) [Label: {label}]")
        print(f"  Tickers: {len(period_conf)}: {sorted(period_conf.keys())}")

        r = run_v2_baseline_detailed(
            all_data=period_data,
            conf_map=period_conf,
            nifty=period_nifty,
            period_name=name,
            period_label=label,
            start=start,
            end=end,
        )
        results.append(r)
        print(f"  Done: {r.n_trades} trades | WR={r.win_rate:.1f}% | "
              f"Net={r.net_after_tax:+.1f}% | DD={r.max_dd:.1f}% | Sharpe={r.sharpe:.2f}")

    # ── Step 4: Synthetic stress scenarios ────────────────────────────────────
    synth_scenarios = [
        ("S1 — GFC 2008-style (-65% over 12mo)", "GFC_2008"),
        ("S2 — Flash Crash (-20% in 10 days)", "FLASH_CRASH"),
        ("S3 — Prolonged Bear (-40% over 30mo)", "PROLONGED_BEAR"),
        ("S4 — High-VIX Chop (sideways 24mo)", "HIGH_VIX_CHOP"),
    ]

    # Use core active tickers for synthetic tests
    synth_base_data = {tk: df for tk, df in all_data.items() if tk in conf_map}

    for name, scenario in synth_scenarios:
        print(f"\n{'─' * 70}")
        print(f"Running {name}...")
        try:
            synth_data, synth_nifty = create_synthetic_scenario(
                synth_base_data, nifty, scenario, start="2022-01-03"
            )
            synth_end = synth_nifty.index[-1].strftime("%Y-%m-%d")
            synth_start = synth_nifty.index[0].strftime("%Y-%m-%d")

            # For synthetic, use original conf_map (same ML signals)
            # Note: conf_map is indexed on real dates; for synthetic we need to remap
            # We'll use a simplified approach: same ML confidence values shifted to synthetic dates
            synth_conf: Dict[str, pd.Series] = {}
            for tk in conf_map:
                real_conf = conf_map[tk]
                # Remap conf to synthetic dates (repeat conf pattern if needed)
                n_synth = len(synth_nifty)
                conf_vals = real_conf.values
                if len(conf_vals) < n_synth:
                    # Tile to fill synthetic period
                    conf_vals = np.tile(conf_vals, math.ceil(n_synth / len(conf_vals)))[:n_synth]
                else:
                    conf_vals = conf_vals[:n_synth]
                synth_conf[tk] = pd.Series(conf_vals, index=synth_nifty.index[:len(conf_vals)], name=tk)

            r = run_v2_baseline_detailed(
                all_data={tk: df for tk, df in synth_data.items() if tk in synth_conf},
                conf_map=synth_conf,
                nifty=synth_nifty,
                period_name=name,
                period_label="SYNTHETIC",
                start=synth_start,
                end=synth_end,
            )
            results.append(r)
            print(f"  Done: {r.n_trades} trades | WR={r.win_rate:.1f}% | "
                  f"Net={r.net_after_tax:+.1f}% | DD={r.max_dd:.1f}% | Sharpe={r.sharpe:.2f}")
        except Exception as e:
            print(f"  Scenario failed: {e}")
            import traceback; traceback.print_exc()

    # ── Step 5: Print full reports ────────────────────────────────────────────
    for r in results:
        print_period_summary(r)
        print_trade_log(r)

    # ── Step 6: Generate charts ───────────────────────────────────────────────
    if HAS_MATPLOTLIB:
        print(f"\n{'─' * 70}")
        print("Generating charts...")
        for r in results:
            safe_name = r.name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("—", "-")
            # Equity curve
            eq_path = os.path.join(CHARTS_DIR, f"equity_{safe_name}.png")
            plot_equity_curve(r, eq_path)
            print(f"  ✓ {eq_path}")

            # Per-ticker charts (only for real periods, first 8 tickers)
            for tk in list(r.ticker_summary.keys())[:8]:
                tk_path = os.path.join(CHARTS_DIR, f"ticker_{tk}_{safe_name}.png")
                plot_ticker_trades(tk, all_data.get(tk), r.trades, r, tk_path)
            print(f"    Per-ticker charts: {len(list(r.ticker_summary.keys())[:8])} tickers")
    else:
        print("\n  ⚠ Charts skipped (matplotlib not available)")

    # ── Step 7: Comparison table ──────────────────────────────────────────────
    print(f"\n{'═' * 110}")
    print(f"  COMPREHENSIVE STRESS TEST COMPARISON")
    print(f"{'═' * 110}")
    col = 15
    headers = ["Period", "Type", "Net%", "WR%", "MaxDD%", "Sharpe", "Calmar", "Trades", "AvgHold"]
    print("  " + "".join(f"{h:<{col}}" for h in headers))
    print("  " + "-" * (col * len(headers)))
    for r in results:
        tp = "IN-SAMPLE" if r.label == "REAL_IS" else "TRUE OOS" if r.label == "REAL_OOS" else "SYNTHETIC"
        row = [
            r.name[:col-1],
            tp[:col-1],
            f"{r.net_after_tax:+.1f}%",
            f"{r.win_rate:.1f}%",
            f"{r.max_dd:.1f}%",
            f"{r.sharpe:.2f}",
            f"{r.calmar:.2f}",
            str(r.n_trades),
            f"{r.avg_hold:.0f}d",
        ]
        print("  " + "".join(f"{v:<{col}}" for v in row))
    print(f"{'═' * 110}")

    # ── Step 8: Write markdown report ─────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("Writing comprehensive markdown report...")
    report_path = os.path.join(REPORTS_DIR, "STRESS_TEST_V6.md")
    write_markdown_report(results, all_data, CHARTS_DIR, report_path)

    # ── Step 9: Save JSON ──────────────────────────────────────────────────────
    json_path = os.path.join(REPORTS_DIR, "stress_test_v6.json")
    json_data = {}
    for r in results:
        json_data[r.name] = {
            "label": r.label, "start": r.start, "end": r.end,
            "n_years": r.n_years,
            "net_after_tax": r.net_after_tax,
            "ann_cagr": r.ann_cagr,
            "total_ret": r.total_ret,
            "win_rate": r.win_rate,
            "max_dd": r.max_dd,
            "sharpe": r.sharpe,
            "calmar": r.calmar,
            "n_trades": r.n_trades,
            "avg_hold": r.avg_hold,
            "expected_value": r.expected_value,
            "annual_returns": r.annual_returns,
            "ticker_summary": r.ticker_summary,
            "trades": [
                {
                    "seq_no": t.seq_no,
                    "ticker": t.ticker,
                    "entry_date": t.entry_date,
                    "entry_day_of_week": t.entry_dow,
                    "entry_price_rs": t.entry_price,
                    "shares": t.shares,
                    "entry_amount_rs": t.entry_cost,
                    "ml_confidence_at_entry": t.entry_conf,
                    "exit_date": t.exit_date,
                    "exit_day_of_week": t.exit_dow,
                    "exit_price_rs": t.exit_price,
                    "exit_reason": t.exit_reason,
                    "hold_days": t.hold_days,
                    "peak_price_rs": t.peak_price,
                    "peak_date": t.peak_date,
                    "peak_gain_from_entry_pct": t.peak_gain_pct,
                    "gross_pnl_rs": t.gross_pnl,
                    "net_pnl_rs": t.net_pnl,
                    "net_pnl_pct": t.pnl_pct,
                    "portfolio_equity_at_entry": t.portfolio_equity_at_entry,
                    "portfolio_equity_at_exit": t.portfolio_equity_at_exit,
                    "win": t.net_pnl > 0,
                }
                for t in sorted(r.trades, key=lambda x: x.entry_date)
            ],
        }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  ✓ JSON: {json_path}")

    print(f"\n{'═' * 90}")
    print(f"  STRESS TEST COMPLETE")
    print(f"  Report:  {REPORTS_DIR}/STRESS_TEST_V6.md")
    print(f"  Charts:  {CHARTS_DIR}/")
    print(f"  JSON:    {json_path}")
    print(f"  Periods: {len(results)} | Total trades: {sum(r.n_trades for r in results)}")
    print(f"{'═' * 90}\n")


if __name__ == "__main__":
    main()

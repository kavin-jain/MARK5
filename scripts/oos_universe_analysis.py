"""
MARK5 — Full Universe OOS Analysis
====================================
Runs a standalone per-ticker OOS backtest (2022-2026) for every trained ticker
in models_v2_oos.  Each ticker is evaluated in ISOLATION (no portfolio competition):
if the ML signal is on, 25% of capital is deployed in that ticker alone.

This gives the HONEST picture of how the system performs across the full
trained universe, not just the cherry-picked 6 PROD tickers.

Metrics per ticker:
  - Gross CAGR, Net CAGR (after Indian tax: LTCG 12.5%, STCG 20%)
  - Max Drawdown, Sharpe ratio
  - Win Rate, # trades, avg hold days
  - pct_above_hurdle (% of OOS bars where rolling ML conf ≥ ML_ENTRY_HURDLE)
  - CPCV mean_sharpe, p_sharpe (from training), passes_prod_gate

Usage:
    python3 scripts/oos_universe_analysis.py
    python3 scripts/oos_universe_analysis.py --models-dir models_v2_oos
    python3 scripts/oos_universe_analysis.py --tickers HAL TRENT BAJFINANCE
"""
import argparse
import json
import logging
import math
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("MARK5.OOSAnalysis")

# ── Backtest parameters (same as ml_momentum_portfolio.py) ───────────────────
OOS_START           = "2022-01-01"
OOS_END             = "2026-05-21"
ML_ENTRY_HURDLE     = 0.52
ML_EXIT_HURDLE      = 0.45
ML_ROLL_WINDOW      = 10
TRAILING_STOP_PCT   = 0.15
TRAILING_STOP_COOLDOWN = 45
REBALANCE_FREQ_DAYS = 15
COST_PCT            = 0.0029
SLIPPAGE_PCT        = 0.001
ALLOC_PCT           = 0.25          # 25% allocation per ticker (single-ticker mode)
INITIAL_CAPITAL     = 5_00_00_000.0  # ₹5 crore

CACHE_DIR   = os.path.join(_ROOT, "data", "cache")
REPORTS_DIR = os.path.join(_ROOT, "reports")

# ── Helpers ───────────────────────────────────────────────────────────────────

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


def get_trained_tickers(models_dir: str) -> List[str]:
    """Return tickers that have a trained model in models_dir."""
    out = []
    if not os.path.exists(models_dir):
        return out
    for entry in sorted(os.listdir(models_dir)):
        if entry in ("versions.json", "registry.json"):
            continue
        path = os.path.join(models_dir, entry)
        if not os.path.isdir(path):
            continue
        versions = sorted(
            [d for d in os.listdir(path) if d.startswith("v")],
            key=lambda x: int(x[1:]) if x[1:].isdigit() else 0,
        )
        if not versions:
            continue
        latest = os.path.join(path, versions[-1])
        if any(f.endswith(".pkl") for f in os.listdir(latest)):
            ticker = entry.replace(".NS", "")
            if ticker not in out:
                out.append(ticker)
    return out


def compute_features(ticker: str, df: pd.DataFrame, pred) -> Optional[pd.DataFrame]:
    if pred.is_v2:
        try:
            from core.models.features_v2 import engineer_features_v2, build_full_context
            ctx = build_full_context(
                ticker=ticker, stock_df=df,
                start_date=str(df.index[0].date()),
                end_date=str(df.index[-1].date()),
                include_sector=True, include_fno=True,
            )
            return engineer_features_v2(df, context=ctx)
        except Exception as e:
            logger.warning(f"{ticker}: V2 features failed ({e}), using V1")
    from core.models.features import engineer_features_df
    return engineer_features_df(df, is_daily=True)


def precompute_conf(ticker: str, pred, feat: pd.DataFrame) -> Optional[pd.Series]:
    try:
        proba = pred.predict_proba(feat)
        return pd.Series(proba, index=feat.index, name=ticker)
    except Exception as e:
        logger.warning(f"{ticker}: conf precompute failed: {e}")
        return None


def rolling_conf(series: pd.Series, date: pd.Timestamp, window: int = ML_ROLL_WINDOW) -> float:
    try:
        idx = series.index.searchsorted(date, side="right") - 1
        idx = max(0, min(idx, len(series) - 1))
        start = max(0, idx - window + 1)
        vals = series.iloc[start:idx + 1]
        return float(vals.mean()) if len(vals) > 0 else 0.0
    except Exception:
        return 0.0


def compute_atr_pct(df: pd.DataFrame, date: pd.Timestamp, n: int = 14) -> float:
    try:
        idx = df.index.searchsorted(date, side="right") - 1
        start = max(0, idx - n)
        sl = df.iloc[start:idx + 1]
        if len(sl) < 3:
            return 0.02
        hl = sl["high"] - sl["low"]
        hc = (sl["high"] - sl["close"].shift(1)).abs()
        lc = (sl["low"]  - sl["close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.mean()
        price = float(df.loc[df.index[idx], "close"])
        return float(atr / price) if price > 0 else 0.02
    except Exception:
        return 0.02


# ── Tax Calculation (Indian LTCG/STCG) ───────────────────────────────────────
LTCG_RATE = 0.125   # 12.5% after ₹1.25L exemption per year (LTCG)
STCG_RATE = 0.20    # 20%  (STCG ≤365d)
LTCG_EXEMPT = 1_25_000  # ₹1.25L annual exemption


def compute_net_after_tax(trades: List[Dict]) -> Tuple[float, float]:
    """
    Returns (total_gross_pnl, total_net_pnl) after Indian equity tax.
    Groups trades by calendar year and applies LTCG/STCG rules.
    """
    if not trades:
        return 0.0, 0.0

    gross_total = sum(t["gross_pnl"] for t in trades)
    net_total   = 0.0

    # Group by exit calendar year
    by_year: Dict[int, List[Dict]] = {}
    for t in trades:
        yr = t["exit_date"].year
        by_year.setdefault(yr, []).append(t)

    for yr, yr_trades in by_year.items():
        ltcg_gains  = sum(t["gross_pnl"] for t in yr_trades if t["hold_days"] > 365 and t["gross_pnl"] > 0)
        ltcg_losses = sum(t["gross_pnl"] for t in yr_trades if t["hold_days"] > 365 and t["gross_pnl"] < 0)
        stcg_gains  = sum(t["gross_pnl"] for t in yr_trades if t["hold_days"] <= 365 and t["gross_pnl"] > 0)
        stcg_losses = sum(t["gross_pnl"] for t in yr_trades if t["hold_days"] <= 365 and t["gross_pnl"] < 0)

        # STCG: losses offset STCG gains first, then LTCG gains
        net_stcg = stcg_gains + stcg_losses
        if net_stcg < 0:
            ltcg_gains = max(0, ltcg_gains + net_stcg)
            net_stcg   = 0.0

        # LTCG: losses offset LTCG gains only
        net_ltcg = max(0, ltcg_gains + ltcg_losses)

        # Apply LTCG exemption of ₹1.25L
        taxable_ltcg = max(0, net_ltcg - LTCG_EXEMPT)

        year_net = (
            stcg_losses + stcg_gains              # raw STCG
            + ltcg_losses + ltcg_gains             # raw LTCG
            - net_stcg * STCG_RATE                 # STCG tax (on net STCG gains)
            - taxable_ltcg * LTCG_RATE             # LTCG tax (on taxable LTCG)
        )
        net_total += year_net

    return gross_total, net_total


# ── Single-Ticker Backtest ────────────────────────────────────────────────────
def run_single_ticker(
    ticker: str,
    conf_series: pd.Series,
    price_df: pd.DataFrame,
    oos_start: str = OOS_START,
    oos_end:   str = OOS_END,
) -> Dict:
    """
    Simulate a single-ticker strategy over the OOS window.
    Returns a dict of statistics.
    """
    start_ts = pd.Timestamp(oos_start)
    end_ts   = pd.Timestamp(oos_end)

    df = price_df[(price_df.index >= start_ts) & (price_df.index <= end_ts)].copy()
    if len(df) < 50:
        return {"status": "insufficient_oos_data", "n_bars": len(df)}

    conf_oos = conf_series[(conf_series.index >= start_ts) & (conf_series.index <= end_ts)]
    if len(conf_oos) == 0:
        return {"status": "no_conf_data"}

    # pct_above_hurdle — key quality metric
    above_hurdle_rolling = []
    for date in conf_oos.index:
        rc = rolling_conf(conf_oos, date)
        above_hurdle_rolling.append(rc >= ML_ENTRY_HURDLE)
    pct_above_hurdle = sum(above_hurdle_rolling) / len(above_hurdle_rolling) * 100

    # OOS confidence stats
    global_std   = float(conf_oos.std())
    global_mean  = float(conf_oos.mean())
    max_conf     = float(conf_oos.max())

    # ── Simulation ────────────────────────────────────────────────────────────
    cash   = INITIAL_CAPITAL
    pos    = None
    trades: List[Dict] = []
    equity_curve: List[Tuple[pd.Timestamp, float]] = []
    cooldown_until = -999   # bar index after trailing stop

    # Pre-index daily bars
    dates = df.index.tolist()
    n = len(dates)

    last_rebal = -REBALANCE_FREQ_DAYS  # force immediate first check

    for bar_i, date in enumerate(dates):
        row = df.loc[date]
        close = float(row["close"])

        # Update peak if in position
        if pos is not None:
            pos["peak_price"] = max(pos["peak_price"], close)

        # Current equity
        if pos is not None:
            pos_value = pos["shares"] * close
            equity    = cash + pos_value
        else:
            equity = cash

        equity_curve.append((date, equity))

        # ── Exit checks (checked every bar) ──────────────────────────────────
        if pos is not None:
            # 1) Trailing stop
            trail_price = pos["peak_price"] * (1 - TRAILING_STOP_PCT)
            if close <= trail_price:
                fill = close * (1 - SLIPPAGE_PCT)
                proceeds = pos["shares"] * fill
                exit_cost = proceeds * COST_PCT
                gross_pnl = (proceeds - exit_cost) - pos["entry_total"]
                cash += (proceeds - exit_cost)
                hold = (date - pos["entry_date"]).days
                trades.append({
                    "entry_date": pos["entry_date"], "exit_date": date,
                    "entry_price": pos["entry_price"], "exit_price": fill,
                    "shares": pos["shares"],
                    "gross_pnl": gross_pnl, "hold_days": hold,
                    "reason": "trailing_stop",
                    "entry_capital": pos["entry_total"],
                })
                pos = None
                cooldown_until = bar_i + TRAILING_STOP_COOLDOWN
                last_rebal = bar_i
                continue

            # 2) ML exit signal (checked on rebalancing days)
            if bar_i - last_rebal >= REBALANCE_FREQ_DAYS:
                rc = rolling_conf(conf_oos, date)
                if rc < ML_EXIT_HURDLE:
                    fill = close * (1 - SLIPPAGE_PCT)
                    proceeds = pos["shares"] * fill
                    exit_cost = proceeds * COST_PCT
                    gross_pnl = (proceeds - exit_cost) - pos["entry_total"]
                    cash += (proceeds - exit_cost)
                    hold = (date - pos["entry_date"]).days
                    trades.append({
                        "entry_date": pos["entry_date"], "exit_date": date,
                        "entry_price": pos["entry_price"], "exit_price": fill,
                        "shares": pos["shares"],
                        "gross_pnl": gross_pnl, "hold_days": hold,
                        "reason": "ml_exit",
                        "entry_capital": pos["entry_total"],
                    })
                    pos = None
                    last_rebal = bar_i
                    continue

        # ── Entry check (on rebalancing days when flat) ───────────────────────
        if pos is None and (bar_i - last_rebal >= REBALANCE_FREQ_DAYS):
            last_rebal = bar_i
            if bar_i < cooldown_until:
                continue
            # NOTE: MIN_ML_STD_GLOBAL intentionally NOT applied here.
            # In portfolio mode that filter prevents "zombie" entries, but in
            # single-ticker analysis we WANT to see the raw ML signal output —
            # even pct_above_hurdle=100% tickers should show buy-and-trail behaviour.
            # The pct_above_hurdle metric already captures signal quality.
            rc = rolling_conf(conf_oos, date)
            if rc >= ML_ENTRY_HURDLE:
                atr_pct = compute_atr_pct(df, date)
                _vol_scale = max(0.6, min(1.2, 0.02 / atr_pct)) if atr_pct > 0 else 1.0
                _edge = max(0.005, rc - ML_ENTRY_HURDLE)
                _edge_scale = max(0.50, min(1.50, _edge / 0.10))
                alloc = max(equity * 0.10, min(equity * 0.35,
                            equity * ALLOC_PCT * _vol_scale * _edge_scale))
                alloc = min(alloc, cash * 0.99)
                fill  = close * (1 + SLIPPAGE_PCT)
                shares = int(alloc / fill)
                if shares < 1:
                    continue
                entry_total = shares * fill * (1 + COST_PCT)
                if entry_total > cash:
                    shares = int((cash * 0.99 / (1 + COST_PCT)) / fill)
                    if shares < 1:
                        continue
                    entry_total = shares * fill * (1 + COST_PCT)
                cash -= entry_total
                pos = {
                    "shares": shares, "entry_price": fill,
                    "peak_price": fill, "entry_date": date,
                    "entry_total": entry_total, "conf_at_entry": rc,
                }

    # Force-close any open position at OOS end
    if pos is not None:
        close = float(df.iloc[-1]["close"])
        fill  = close * (1 - SLIPPAGE_PCT)
        proceeds  = pos["shares"] * fill
        exit_cost = proceeds * COST_PCT
        gross_pnl = (proceeds - exit_cost) - pos["entry_total"]
        cash += (proceeds - exit_cost)
        hold = (dates[-1] - pos["entry_date"]).days
        trades.append({
            "entry_date": pos["entry_date"], "exit_date": dates[-1],
            "entry_price": pos["entry_price"], "exit_price": fill,
            "shares": pos["shares"],
            "gross_pnl": gross_pnl, "hold_days": hold,
            "reason": "end_of_backtest",
            "entry_capital": pos["entry_total"],
        })
        pos = None

    # Final equity
    final_equity = cash
    if len(equity_curve) == 0:
        return {"status": "no_data"}
    final_equity = equity_curve[-1][1]

    # ── Performance stats ─────────────────────────────────────────────────────
    n_years = (end_ts - start_ts).days / 365.25
    gross_total, net_total = compute_net_after_tax(trades)

    # CAGR (gross = based on final equity; net adjusts final equity by tax)
    gross_final = INITIAL_CAPITAL + gross_total
    net_final   = INITIAL_CAPITAL + net_total

    def cagr(final, init=INITIAL_CAPITAL, yrs=n_years):
        if final <= 0 or yrs <= 0:
            return 0.0
        return ((final / init) ** (1 / yrs) - 1) * 100

    gross_cagr = cagr(gross_final)
    net_cagr   = cagr(net_final)

    # Equity curve → returns
    eq_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    eq_df["ret"] = eq_df["equity"].pct_change().fillna(0)

    # Max drawdown
    roll_max = eq_df["equity"].cummax()
    dd       = (eq_df["equity"] - roll_max) / roll_max
    max_dd   = float(dd.min()) * 100

    # Sharpe (annualised, 252 trading days)
    if eq_df["ret"].std() > 1e-9:
        sharpe = float(eq_df["ret"].mean() / eq_df["ret"].std() * math.sqrt(252))
    else:
        sharpe = 0.0

    # Trade stats
    n_trades = len(trades)
    winners  = [t for t in trades if t["gross_pnl"] > 0]
    win_rate = len(winners) / n_trades * 100 if n_trades > 0 else 0.0
    avg_hold = sum(t["hold_days"] for t in trades) / n_trades if n_trades > 0 else 0.0
    avg_win  = sum(t["gross_pnl"] for t in winners) / len(winners) if winners else 0.0
    losers   = [t for t in trades if t["gross_pnl"] <= 0]
    avg_loss = sum(t["gross_pnl"] for t in losers) / len(losers) if losers else 0.0

    # Annual returns (calendar year breakdown)
    annual: Dict[int, float] = {}
    for yr in range(2022, 2027):
        y_start = pd.Timestamp(f"{yr}-01-01")
        y_end   = pd.Timestamp(f"{yr}-12-31")
        yr_eq   = eq_df[(eq_df.index >= y_start) & (eq_df.index <= y_end)]
        if len(yr_eq) < 5:
            continue
        yr_ret = (yr_eq["equity"].iloc[-1] / yr_eq["equity"].iloc[0] - 1) * 100
        annual[yr] = round(yr_ret, 1)

    # ── Buy-and-hold comparison ───────────────────────────────────────────────
    bh_entry = float(df.iloc[0]["close"])
    bh_exit  = float(df.iloc[-1]["close"])
    bh_return_pct = (bh_exit / bh_entry - 1) * 100
    bh_cagr  = cagr(INITIAL_CAPITAL * (1 + bh_return_pct / 100))
    # ML alpha = how much better (worse) than buy-and-hold
    ml_alpha = net_cagr - bh_cagr

    return {
        "status":           "success",
        "gross_cagr":       round(gross_cagr, 2),
        "net_cagr":         round(net_cagr, 2),
        "bh_cagr":          round(bh_cagr, 2),       # buy-and-hold CAGR for comparison
        "ml_alpha":         round(ml_alpha, 2),       # ML net CAGR - buy-and-hold CAGR
        "max_dd":           round(max_dd, 2),
        "sharpe":           round(sharpe, 2),
        "n_trades":         n_trades,
        "win_rate":         round(win_rate, 1),
        "avg_hold_days":    round(avg_hold, 0),
        "avg_win_inr":      round(avg_win / 1e5, 2),
        "avg_loss_inr":     round(avg_loss / 1e5, 2),
        "gross_total_inr":  round(gross_total / 1e5, 2),
        "net_total_inr":    round(net_total / 1e5, 2),
        "pct_above_hurdle": round(pct_above_hurdle, 1),
        "conf_mean":        round(global_mean, 4),
        "conf_std":         round(global_std, 4),
        "conf_max":         round(max_conf, 4),
        "annual":           annual,
        "n_oos_bars":       len(df),
    }


# ── Load CPCV training stats ──────────────────────────────────────────────────
def load_cpcv_stats(ticker: str, models_dir: str) -> Dict:
    """Load training CPCV stats from the model's features.json or a results JSON."""
    # Try the retrain results file
    results_paths = [
        os.path.join(_ROOT, "reports", "retrain_results_cutoff20211231_models_v2_oos.json"),
        os.path.join(_ROOT, "reports", "train_universe_oos_results.json"),
    ]
    for rp in results_paths:
        if not os.path.exists(rp):
            continue
        try:
            with open(rp) as f:
                data = json.load(f)
            if ticker in data:
                d = data[ticker]
                return {
                    "mean_sharpe":      float(d.get("mean_sharpe", 0) or 0),
                    "p_sharpe":         float(d.get("p_sharpe", 0) or 0),
                    "worst_5pct_sharpe":float(d.get("worst_5pct_sharpe", 0) or 0),
                    "passes_prod_gate": str(d.get("passes_prod_gate", "False")).lower() == "true",
                    "auc":              float(d.get("auc", 0) or 0),
                    "n_features":       int(d.get("n_features", 0) or 0),
                }
        except Exception:
            pass
    return {}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default=os.path.join(_ROOT, "models_v2_oos"))
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers (default: all trained in models-dir)")
    parser.add_argument("--oos-start", default=OOS_START)
    parser.add_argument("--oos-end",   default=OOS_END)
    args = parser.parse_args()

    from core.models.backtest_pipeline import LightPredictor

    models_dir = args.models_dir
    tickers = args.tickers or get_trained_tickers(models_dir)

    oos_start = args.oos_start
    oos_end   = args.oos_end

    print(f"\n{'═'*80}")
    print(f"  MARK5 FULL UNIVERSE OOS ANALYSIS")
    print(f"  Models dir : {models_dir}")
    print(f"  OOS window : {oos_start} → {oos_end}")
    print(f"  Tickers    : {len(tickers)}")
    print(f"  Entry hurdle: {ML_ENTRY_HURDLE} | Exit: {ML_EXIT_HURDLE} | Trail stop: {TRAILING_STOP_PCT*100:.0f}%")
    print(f"{'═'*80}\n")

    all_results: Dict = {}

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:2d}/{len(tickers)}] {ticker:<20s}", end=" ", flush=True)

        # Load model
        pred = LightPredictor(ticker, models_dir)
        if not pred.models:
            print("⏭  no model")
            all_results[ticker] = {"status": "no_model"}
            continue

        # Load data
        df = load_cache(ticker)
        if df is None:
            print("⏭  no cache data")
            all_results[ticker] = {"status": "no_data"}
            continue

        # Compute features on FULL history (before OOS split — needed for rolling windows)
        try:
            feat = compute_features(ticker, df, pred)
        except Exception as e:
            print(f"❌  feature error: {e}")
            all_results[ticker] = {"status": "feature_error", "error": str(e)}
            continue
        if feat is None or len(feat) == 0:
            print("❌  empty features")
            all_results[ticker] = {"status": "empty_features"}
            continue

        # Pre-compute ML confidence
        conf = precompute_conf(ticker, pred, feat)
        if conf is None:
            print("❌  conf precompute failed")
            all_results[ticker] = {"status": "conf_failed"}
            continue

        # Run per-ticker OOS simulation
        try:
            result = run_single_ticker(ticker, conf, df,
                                       oos_start=oos_start, oos_end=oos_end)
        except Exception as e:
            print(f"❌  backtest error: {e}")
            all_results[ticker] = {"status": "backtest_error", "error": str(e)}
            continue

        # Merge in CPCV training stats
        cpcv = load_cpcv_stats(ticker, models_dir)
        result.update(cpcv)
        all_results[ticker] = result

        if result["status"] == "success":
            nc  = result["net_cagr"]
            dd  = result["max_dd"]
            sr  = result["sharpe"]
            wr  = result["win_rate"]
            nt  = result["n_trades"]
            pah = result["pct_above_hurdle"]
            flag = "🟢" if nc >= 15 else ("🟡" if nc >= 5 else "🔴")
            print(f"{flag}  net={nc:+6.1f}% CAGR | DD={dd:+5.1f}% | SR={sr:.2f} | WR={wr:.0f}% | {nt}t | pah={pah:.0f}%")
        else:
            print(f"⚠️  {result.get('status')}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  FULL UNIVERSE OOS SUMMARY — {oos_start} → {oos_end}")
    print(f"{'═'*80}")

    success = {t: r for t, r in all_results.items() if r.get("status") == "success"}
    by_net_cagr = sorted(success.items(), key=lambda x: x[1]["net_cagr"], reverse=True)

    # Distribution stats
    cagrs   = [r["net_cagr"]  for _, r in by_net_cagr]
    dds     = [r["max_dd"]    for _, r in by_net_cagr]
    sharpes = [r["sharpe"]    for _, r in by_net_cagr]

    header = (f"  {'Ticker':<14s} {'NetCAGR':>8} {'B&H CAGR':>9} {'ML Alpha':>9} {'MaxDD':>7} "
              f"{'Sharpe':>7} {'WR':>5} {'Trades':>7} {'AvgHold':>8} {'PAH':>5} {'Gate':>5}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    profitable_tickers, gate_pass = [], []
    for ticker, r in by_net_cagr:
        nc   = r["net_cagr"]
        bh   = r.get("bh_cagr", 0)
        alp  = r.get("ml_alpha", 0)
        dd   = r["max_dd"]
        sr   = r["sharpe"]
        wr   = r["win_rate"]
        nt   = r["n_trades"]
        ah   = r["avg_hold_days"]
        pah  = r["pct_above_hurdle"]
        gate = "✅" if r.get("passes_prod_gate", False) else "  "
        flag = "🟢" if nc >= 15 else ("🟡" if nc >= 5 else "🔴")
        alpha_flag = "⬆" if alp > 2 else ("⬇" if alp < -2 else "≈")
        print(f"  {flag} {ticker:<12s} {nc:>+7.1f}% {bh:>+8.1f}% {alp:>+8.1f}%{alpha_flag} {dd:>+6.1f}% "
              f"{sr:>7.2f} {wr:>4.0f}% {nt:>7d} {ah:>7.0f}d {pah:>4.0f}% {gate}")
        if nc >= 5:
            profitable_tickers.append(ticker)
        if r.get("passes_prod_gate", False):
            gate_pass.append(ticker)

    print(f"\n  Summary:")
    print(f"    Total tickers analysed  : {len(success)}")
    print(f"    Profitable (>5% net)    : {len(profitable_tickers)}  ({len(profitable_tickers)/max(1,len(success))*100:.0f}%)")
    print(f"    Strong (>15% net)       : {len([t for t,r in success.items() if r['net_cagr']>=15])}  ({len([t for t,r in success.items() if r['net_cagr']>=15])/max(1,len(success))*100:.0f}%)")
    print(f"    Meets 20%+ target       : {len([t for t,r in success.items() if r['net_cagr']>=20])}")
    print(f"    Passes CPCV prod gate   : {len(gate_pass)}  {gate_pass}")
    if cagrs:
        print(f"\n    Distribution of Net CAGR:")
        print(f"      Median : {sorted(cagrs)[len(cagrs)//2]:+.1f}%")
        print(f"      Mean   : {sum(cagrs)/len(cagrs):+.1f}%")
        print(f"      Top 25%: {sorted(cagrs)[int(len(cagrs)*0.75)]:+.1f}%")
        print(f"      Bottom : {min(cagrs):+.1f}%  |  Top: {max(cagrs):+.1f}%")
    print(f"{'═'*80}")

    # ── Year-by-year breakdown for successful tickers ──────────────────────────
    print(f"\n  ANNUAL RETURNS BY TICKER:")
    print(f"  {'Ticker':<14s} {'2022':>7} {'2023':>7} {'2024':>7} {'2025':>7} {'2026':>7}")
    print("  " + "─" * 50)
    for ticker, r in by_net_cagr:
        ann = r.get("annual", {})
        row = f"  {ticker:<14s}"
        for yr in [2022, 2023, 2024, 2025, 2026]:
            val = ann.get(yr, None)
            if val is None:
                row += f"{'—':>7}"
            else:
                row += f"{val:>+6.1f}%"
        print(row)

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_json = os.path.join(REPORTS_DIR, "oos_universe_analysis.json")
    out_md   = os.path.join(REPORTS_DIR, "oos_universe_analysis.md")

    # Serialize
    serializable = {}
    for k, v in all_results.items():
        sd = {}
        for kk, vv in v.items():
            if isinstance(vv, dict):
                sd[kk] = {str(kkk): float(vvv) if isinstance(vvv, (int, float, np.floating)) else vvv
                           for kkk, vvv in vv.items()}
            elif isinstance(vv, (np.floating, np.integer)):
                sd[kk] = float(vv)
            else:
                sd[kk] = vv
        serializable[k] = sd

    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2)

    # Markdown report
    with open(out_md, "w") as f:
        f.write(f"# MARK5 Full Universe OOS Analysis\n\n")
        f.write(f"**Period**: {oos_start} → {oos_end}  \n")
        f.write(f"**Models**: {models_dir}  \n")
        f.write(f"**Generated**: {pd.Timestamp.now().date()}  \n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Ticker | Net CAGR | Gross CAGR | Max DD | Sharpe | WR | Trades | AvgHold | PAH | Gate |\n")
        f.write(f"|--------|----------|------------|--------|--------|----|---------|---------|----|------|\n")
        for ticker, r in by_net_cagr:
            gate = "✅" if r.get("passes_prod_gate", False) else ""
            f.write(f"| {ticker} | {r['net_cagr']:+.1f}% | {r['gross_cagr']:+.1f}% | {r['max_dd']:+.1f}% | "
                    f"{r['sharpe']:.2f} | {r['win_rate']:.0f}% | {r['n_trades']} | {r['avg_hold_days']:.0f}d | "
                    f"{r['pct_above_hurdle']:.0f}% | {gate} |\n")
        f.write(f"\n## Distribution\n\n")
        if cagrs:
            f.write(f"- **Profitable (>5% net)**: {len(profitable_tickers)}/{len(success)} ({len(profitable_tickers)/len(success)*100:.0f}%)\n")
            f.write(f"- **Strong (>15% net)**: {len([t for t,r in success.items() if r['net_cagr']>=15])}/{len(success)}\n")
            f.write(f"- **Meets 20%+ target**: {len([t for t,r in success.items() if r['net_cagr']>=20])}/{len(success)}\n")
            f.write(f"- **Median Net CAGR**: {sorted(cagrs)[len(cagrs)//2]:+.1f}%\n")
            f.write(f"- **Mean Net CAGR**: {sum(cagrs)/len(cagrs):+.1f}%\n")

    print(f"\n  Reports saved:")
    print(f"    {out_json}")
    print(f"    {out_md}")

    return all_results


if __name__ == "__main__":
    main()

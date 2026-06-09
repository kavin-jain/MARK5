"""
MARK5 — Universal Momentum Backtest
=====================================
Per-ticker OOS backtest using the Multi-Factor Momentum Signal.
No ML model required. Works for ANY stock in the local parquet cache.

Goal: given any random NSE stock, don't lose — and ideally make profit.

How it works:
  1. Compute MomentumSignalEngine scores for every bar in OOS window
  2. Enter when rolling 5-bar score ≥ 0.55 (bullish momentum confirmed)
  3. Exit when rolling 5-bar score ≤ 0.40 (trend deteriorating) OR
     ATR-based trailing stop (4×20-day ATR from peak, capped 10-25%)
  4. Kelly edge-proportional sizing (same as portfolio system)
  5. Indian tax: LTCG 12.5% / STCG 20%

Usage:
    # Single ticker
    python3 scripts/universal_backtest.py --ticker HAL

    # All OOS-eligible tickers (comparison vs ML system)
    python3 scripts/universal_backtest.py --all

    # Custom OOS window
    python3 scripts/universal_backtest.py --all --oos-start 2022-01-01 --oos-end 2026-05-21

    # Load from any parquet file
    python3 scripts/universal_backtest.py --ticker BEL --show-scores

PAPER MODE ONLY — never executes real trades.
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
logger = logging.getLogger("MARK5.UniversalBacktest")

from core.models.momentum_signal import MomentumSignalEngine, ENTRY_THRESHOLD, EXIT_THRESHOLD

# ── Parameters ────────────────────────────────────────────────────────────────
OOS_START        = "2022-01-01"
OOS_END          = "2026-05-21"
INITIAL_CAPITAL  = 5_00_00_000.0    # ₹5 crore
ALLOC_PCT        = 0.25             # 25% per position (single-ticker: 25% deployed)
REBAL_FREQ       = 15               # bars between entry/exit checks
COST_PCT         = 0.0029           # 0.29% round-trip (NSE equity delivery)
SLIPPAGE_PCT     = 0.001            # 0.10% slippage
ATR_MULTIPLIER   = 4.0              # trailing stop = 4×ATR from peak
ATR_STOP_MIN     = 0.10             # minimum 10% trail (protects against whipsawing)
ATR_STOP_MAX     = 0.25             # maximum 25% trail (caps per-trade loss)
COOLDOWN_BARS    = 30               # bars to wait after trailing stop
ROLL_WINDOW      = 5                # bars for rolling score average
ML_ALPHA_FILE    = os.path.join(_ROOT, "reports", "oos_universe_analysis.json")

LTCG_RATE    = 0.125
STCG_RATE    = 0.20
LTCG_EXEMPT  = 1_25_000
CACHE_DIR    = os.path.join(_ROOT, "data", "cache")
REPORTS_DIR  = os.path.join(_ROOT, "reports")

# All OOS-eligible tickers (≥900 bars before 2021-12-31 + data post-2022)
ALL_ELIGIBLE = [
    "ASIANPAINT", "AUBANK", "BAJFINANCE", "BAJAJ-AUTO", "BANDHANBNK",
    "BEL", "BHARTIARTL", "COFORGE", "HDFCBANK", "HINDUNILVR",
    "ICICIBANK", "IDEA", "IDFCFIRSTB", "INFY", "ITC",
    "KOTAKBANK", "LT", "LUPIN", "MARUTI", "MOTHERSON",
    "PERSISTENT", "PNB", "RELIANCE", "SBIN", "SUNPHARMA",
    "TATAELXSI", "TATASTEEL", "TCS", "TITAN", "TRENT",
    "VOLTAS", "YESBANK", "HAL",
]

# ── Data helpers ──────────────────────────────────────────────────────────────

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


def load_nifty50() -> Optional[pd.DataFrame]:
    path = os.path.join(CACHE_DIR, "NIFTY50_1d.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if hasattr(df.index, "tz") and df.index.tz:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def load_ml_results() -> Dict:
    """Load per-ticker ML results from previous OOS analysis."""
    if not os.path.exists(ML_ALPHA_FILE):
        return {}
    try:
        with open(ML_ALPHA_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


# ── Tax calculation ───────────────────────────────────────────────────────────

def compute_tax(trades: List[Dict]) -> Tuple[float, float]:
    """Returns (gross_pnl, net_pnl_after_tax)."""
    if not trades:
        return 0.0, 0.0
    gross = sum(t["gross_pnl"] for t in trades)
    net   = 0.0
    by_year: Dict[int, List] = {}
    for t in trades:
        by_year.setdefault(t["exit_date"].year, []).append(t)
    for _, yr_trades in by_year.items():
        ltcg_g = sum(t["gross_pnl"] for t in yr_trades if t["hold_days"] > 365 and t["gross_pnl"] > 0)
        ltcg_l = sum(t["gross_pnl"] for t in yr_trades if t["hold_days"] > 365 and t["gross_pnl"] < 0)
        stcg_g = sum(t["gross_pnl"] for t in yr_trades if t["hold_days"] <= 365 and t["gross_pnl"] > 0)
        stcg_l = sum(t["gross_pnl"] for t in yr_trades if t["hold_days"] <= 365 and t["gross_pnl"] < 0)
        net_stcg = stcg_g + stcg_l
        if net_stcg < 0:
            ltcg_g = max(0.0, ltcg_g + net_stcg)
            net_stcg = 0.0
        net_ltcg = max(0.0, ltcg_g + ltcg_l)
        taxable_ltcg = max(0.0, net_ltcg - LTCG_EXEMPT)
        net += (stcg_l + stcg_g + ltcg_l + ltcg_g
                - net_stcg * STCG_RATE
                - taxable_ltcg * LTCG_RATE)
    return gross, net


# ── Core simulation ───────────────────────────────────────────────────────────

def run_momentum_backtest(
    ticker: str,
    df: pd.DataFrame,
    nifty_df: Optional[pd.DataFrame],
    oos_start: str = OOS_START,
    oos_end:   str = OOS_END,
    show_scores: bool = False,
) -> Dict:
    """
    Simulate the momentum strategy on one ticker over the OOS window.

    Returns a dict with all performance metrics.
    """
    start_ts = pd.Timestamp(oos_start)
    end_ts   = pd.Timestamp(oos_end)

    # OOS slice for simulation
    df_oos = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if len(df_oos) < 60:
        return {"status": "insufficient_oos_data", "n_bars": len(df_oos)}

    engine = MomentumSignalEngine()

    # Compute scores on FULL history (prevents look-ahead bias: rolling windows need past data)
    full_scores = engine.precompute_scores(df, nifty_df=nifty_df)
    oos_scores  = full_scores[(full_scores.index >= start_ts) & (full_scores.index <= end_ts)]

    if show_scores:
        _print_score_sample(ticker, oos_scores, df_oos)

    # pct_above_hurdle (for comparison with ML system)
    roll_vals = []
    for date in oos_scores.index:
        rs = engine.rolling_score(oos_scores, date, window=ROLL_WINDOW)
        roll_vals.append(rs >= ENTRY_THRESHOLD)
    pct_above_hurdle = sum(roll_vals) / max(1, len(roll_vals)) * 100
    oos_std  = float(oos_scores.std())
    oos_mean = float(oos_scores.mean())

    # ── Simulate ─────────────────────────────────────────────────────────────
    dates = df_oos.index.tolist()
    n     = len(dates)
    cash  = INITIAL_CAPITAL
    pos   = None
    trades: List[Dict] = []
    equity_curve: List[Tuple[pd.Timestamp, float]] = []
    cooldown_until = -COOLDOWN_BARS
    last_rebal     = -REBAL_FREQ

    for bar_i, date in enumerate(dates):
        row   = df_oos.loc[date]
        close = float(row["close"])

        if pos is not None:
            pos["peak_price"] = max(pos["peak_price"], close)

        equity = cash + (pos["shares"] * close if pos else 0.0)
        equity_curve.append((date, equity))

        # ── Trailing stop (every bar) ─────────────────────────────────────
        if pos is not None:
            atr_pct    = engine.compute_atr_pct(df_oos, date)
            stop_price = engine.trailing_stop_price(
                pos["peak_price"], atr_pct, ATR_MULTIPLIER, ATR_STOP_MIN, ATR_STOP_MAX
            )
            if close <= stop_price:
                fill       = close * (1 - SLIPPAGE_PCT)
                proceeds   = pos["shares"] * fill
                exit_cost  = proceeds * COST_PCT
                gross_pnl  = (proceeds - exit_cost) - pos["entry_total"]
                cash      += (proceeds - exit_cost)
                hold       = (date - pos["entry_date"]).days
                trades.append({
                    "entry_date": pos["entry_date"], "exit_date": date,
                    "entry_price": pos["entry_price"], "exit_price": fill,
                    "shares": pos["shares"], "gross_pnl": gross_pnl,
                    "hold_days": hold, "reason": "trailing_stop",
                    "entry_total": pos["entry_total"],
                })
                pos = None
                cooldown_until = bar_i + COOLDOWN_BARS
                last_rebal     = bar_i
                continue

        # ── Rebalancing check (every REBAL_FREQ bars) ─────────────────────
        if bar_i - last_rebal < REBAL_FREQ:
            continue
        last_rebal = bar_i

        rs = engine.rolling_score(oos_scores, date, window=ROLL_WINDOW)

        # Exit: score turned bearish
        if pos is not None and rs <= EXIT_THRESHOLD:
            fill       = close * (1 - SLIPPAGE_PCT)
            proceeds   = pos["shares"] * fill
            exit_cost  = proceeds * COST_PCT
            gross_pnl  = (proceeds - exit_cost) - pos["entry_total"]
            cash      += (proceeds - exit_cost)
            hold       = (date - pos["entry_date"]).days
            trades.append({
                "entry_date": pos["entry_date"], "exit_date": date,
                "entry_price": pos["entry_price"], "exit_price": fill,
                "shares": pos["shares"], "gross_pnl": gross_pnl,
                "hold_days": hold, "reason": "momentum_exit",
                "entry_total": pos["entry_total"],
            })
            pos = None
            continue

        # Entry: score turned bullish
        if pos is None and bar_i >= cooldown_until and rs >= ENTRY_THRESHOLD:
            # Kelly edge-proportional sizing (same as portfolio)
            _edge  = max(0.005, rs - ENTRY_THRESHOLD)
            _scale = max(0.50, min(1.50, _edge / 0.10))
            alloc  = max(equity * 0.10, min(equity * 0.35,
                         equity * ALLOC_PCT * _scale))
            alloc  = min(alloc, cash * 0.99)
            fill   = close * (1 + SLIPPAGE_PCT)
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
            pos   = {
                "shares": shares, "entry_price": fill,
                "peak_price": fill, "entry_date": date,
                "entry_total": entry_total, "score_at_entry": rs,
            }

    # Force-close open position at OOS end
    if pos is not None:
        close      = float(df_oos.iloc[-1]["close"])
        fill       = close * (1 - SLIPPAGE_PCT)
        proceeds   = pos["shares"] * fill
        exit_cost  = proceeds * COST_PCT
        gross_pnl  = (proceeds - exit_cost) - pos["entry_total"]
        cash      += (proceeds - exit_cost)
        hold       = (dates[-1] - pos["entry_date"]).days
        trades.append({
            "entry_date": pos["entry_date"], "exit_date": dates[-1],
            "entry_price": pos["entry_price"], "exit_price": fill,
            "shares": pos["shares"], "gross_pnl": gross_pnl,
            "hold_days": hold, "reason": "end_of_backtest",
            "entry_total": pos["entry_total"],
        })

    # ── Performance metrics ───────────────────────────────────────────────────
    final_equity = equity_curve[-1][1] if equity_curve else INITIAL_CAPITAL
    n_years      = (end_ts - start_ts).days / 365.25

    gross_total, net_total = compute_tax(trades)
    gross_final  = INITIAL_CAPITAL + gross_total
    net_final    = INITIAL_CAPITAL + net_total

    def cagr(final, yrs=n_years):
        if final <= 0 or yrs <= 0:
            return -100.0
        return ((final / INITIAL_CAPITAL) ** (1.0 / yrs) - 1.0) * 100.0

    eq_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    eq_df["ret"] = eq_df["equity"].pct_change().fillna(0.0)

    roll_max = eq_df["equity"].cummax()
    dd       = (eq_df["equity"] - roll_max) / roll_max
    max_dd   = float(dd.min()) * 100.0

    sharpe = (float(eq_df["ret"].mean() / eq_df["ret"].std() * math.sqrt(252))
              if eq_df["ret"].std() > 1e-9 else 0.0)

    n_trades = len(trades)
    winners  = [t for t in trades if t["gross_pnl"] > 0]
    win_rate = len(winners) / n_trades * 100.0 if n_trades else 0.0
    avg_hold = sum(t["hold_days"] for t in trades) / n_trades if n_trades else 0.0

    # Buy-and-hold comparison
    bh_entry = float(df_oos.iloc[0]["close"])
    bh_exit  = float(df_oos.iloc[-1]["close"])
    bh_ret   = bh_exit / bh_entry - 1.0
    bh_cagr  = cagr(INITIAL_CAPITAL * (1.0 + bh_ret))
    ml_alpha = cagr(net_final) - bh_cagr  # vs buy-and-hold

    # Annual returns
    annual: Dict[int, float] = {}
    for yr in range(2022, 2027):
        y_start = pd.Timestamp(f"{yr}-01-01")
        y_end   = pd.Timestamp(f"{yr}-12-31")
        ye = eq_df[(eq_df.index >= y_start) & (eq_df.index <= y_end)]
        if len(ye) >= 5:
            annual[yr] = round((ye["equity"].iloc[-1] / ye["equity"].iloc[0] - 1.0) * 100.0, 1)

    return {
        "status":            "success",
        "gross_cagr":        round(cagr(gross_final), 2),
        "net_cagr":          round(cagr(net_final),   2),
        "bh_cagr":           round(bh_cagr, 2),
        "vs_bh":             round(cagr(net_final) - bh_cagr, 2),  # positive = beat buy-and-hold
        "max_dd":            round(max_dd, 2),
        "sharpe":            round(sharpe, 2),
        "n_trades":          n_trades,
        "win_rate":          round(win_rate, 1),
        "avg_hold_days":     round(avg_hold, 0),
        "gross_total_L":     round(gross_total / 1e5, 2),
        "net_total_L":       round(net_total / 1e5, 2),
        "pct_above_hurdle":  round(pct_above_hurdle, 1),
        "score_mean":        round(oos_mean, 4),
        "score_std":         round(oos_std, 4),
        "annual":            annual,
        "n_oos_bars":        len(df_oos),
        "trades":            trades,
    }


def _print_score_sample(ticker: str, scores: pd.Series, df_oos: pd.DataFrame):
    print(f"\n  [{ticker}] Score sample (monthly):")
    for date in pd.date_range(scores.index[0], scores.index[-1], freq="3MS"):
        if date in scores.index:
            s = float(scores[date])
            bar = "█" * int(s * 20)
            print(f"    {date.date()}  {s:.3f}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Universal momentum backtest for any NSE ticker"
    )
    parser.add_argument("--ticker",     type=str, default=None,
                        help="Single ticker (e.g. HAL, BEL, RELIANCE)")
    parser.add_argument("--all",        action="store_true",
                        help="Run on all OOS-eligible tickers")
    parser.add_argument("--oos-start",  default=OOS_START)
    parser.add_argument("--oos-end",    default=OOS_END)
    parser.add_argument("--show-scores", action="store_true",
                        help="Print quarterly score evolution for each ticker")
    parser.add_argument("--atr-mult",   type=float, default=ATR_MULTIPLIER,
                        help="ATR trailing stop multiplier (default 4.0)")
    args = parser.parse_args()

    if not args.ticker and not args.all:
        parser.print_help()
        print("\n  Example: python3 scripts/universal_backtest.py --all")
        return

    tickers = ALL_ELIGIBLE if args.all else [args.ticker]
    nifty   = load_nifty50()
    ml_data = load_ml_results()

    if nifty is None:
        print("  ⚠️  NIFTY50 cache not found — relative strength component disabled")

    print(f"\n{'═'*90}")
    print(f"  MARK5 UNIVERSAL MOMENTUM BACKTEST")
    print(f"  Signal: Multi-Factor Momentum (trend + momentum + RS + sharpe + volume)")
    print(f"  OOS   : {args.oos_start} → {args.oos_end}")
    print(f"  Stop  : ATR×{args.atr_mult} trail (10-25% range)  |  Entry: score≥{ENTRY_THRESHOLD}  |  Exit: ≤{EXIT_THRESHOLD}")
    print(f"  Tickers: {len(tickers)}")
    print(f"{'═'*90}\n")

    results: Dict = {}

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:2d}/{len(tickers)}] {ticker:<14s}", end=" ", flush=True)

        df = load_cache(ticker)
        if df is None:
            print("⏭  no cache data")
            results[ticker] = {"status": "no_data"}
            continue

        try:
            r = run_momentum_backtest(
                ticker, df, nifty,
                oos_start=args.oos_start,
                oos_end=args.oos_end,
                show_scores=args.show_scores,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"❌  error: {e}")
            results[ticker] = {"status": "error", "error": str(e)}
            continue

        results[ticker] = r

        if r["status"] == "success":
            nc   = r["net_cagr"]
            dd   = r["max_dd"]
            sr   = r["sharpe"]
            wr   = r["win_rate"]
            nt   = r["n_trades"]
            pah  = r["pct_above_hurdle"]
            bh   = r["bh_cagr"]
            vbh  = r["vs_bh"]

            # Fetch ML result for comparison
            ml_nc = None
            if ticker in ml_data and ml_data[ticker].get("status") == "success":
                ml_nc = ml_data[ticker].get("net_cagr", None)

            flag = "🟢" if nc >= 15 else ("🟡" if nc >= 5 else "🔴")
            vbh_flag = "⬆" if vbh > 2 else ("⬇" if vbh < -2 else "≈")
            ml_str = f"  (ML:{ml_nc:+.1f}%)" if ml_nc is not None else ""
            print(f"{flag}  net={nc:+6.1f}%  B&H={bh:+6.1f}%  vs_BH={vbh:+5.1f}%{vbh_flag}  "
                  f"DD={dd:+5.1f}%  SR={sr:.2f}  WR={wr:.0f}%  {nt}t  pah={pah:.0f}%{ml_str}")
        else:
            print(f"⚠️  {r.get('status')}")

    # ── Summary ───────────────────────────────────────────────────────────────
    success  = {t: r for t, r in results.items() if r.get("status") == "success"}
    by_cagr  = sorted(success.items(), key=lambda x: x[1]["net_cagr"], reverse=True)
    cagrs    = [r["net_cagr"] for _, r in by_cagr]
    bh_cagrs = [r["bh_cagr"]  for _, r in by_cagr]

    print(f"\n{'═'*90}")
    print(f"  FULL UNIVERSE MOMENTUM SUMMARY  ({args.oos_start} → {args.oos_end})")
    print(f"{'═'*90}")

    header = (f"  {'Ticker':<14} {'MomNet':>8} {'B&H':>7} {'vs BH':>7} {'ML Net':>8} "
              f"{'MaxDD':>7} {'Sharpe':>7} {'WR':>5} {'T':>4} {'PAH':>5}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    for ticker, r in by_cagr:
        nc   = r["net_cagr"]
        bh   = r["bh_cagr"]
        vbh  = r["vs_bh"]
        dd   = r["max_dd"]
        sr   = r["sharpe"]
        wr   = r["win_rate"]
        nt   = r["n_trades"]
        pah  = r["pct_above_hurdle"]
        flag = "🟢" if nc >= 15 else ("🟡" if nc >= 5 else "🔴")
        vbh_flag = "⬆" if vbh > 2 else ("⬇" if vbh < -2 else "≈")

        ml_nc = None
        if ticker in ml_data and ml_data[ticker].get("status") == "success":
            ml_nc = ml_data[ticker].get("net_cagr")
        ml_str = f"{ml_nc:>+7.1f}%" if ml_nc is not None else f"{'N/A':>8}"

        print(f"  {flag} {ticker:<12s} {nc:>+7.1f}% {bh:>+6.1f}% {vbh:>+6.1f}%{vbh_flag}  {ml_str} "
              f"{dd:>+6.1f}%  {sr:>6.2f}  {wr:>4.0f}%  {nt:>3d}  {pah:>4.0f}%")

    print(f"\n  Distribution — Momentum Net CAGR:")
    if cagrs:
        sorted_cagrs = sorted(cagrs)
        n = len(sorted_cagrs)
        prof5  = sum(1 for c in cagrs if c >= 5)
        prof15 = sum(1 for c in cagrs if c >= 15)
        prof20 = sum(1 for c in cagrs if c >= 20)
        print(f"    Total tickers         : {len(success)}")
        print(f"    Profitable  (>5%  net): {prof5}/{len(success)} = {prof5/len(success)*100:.0f}%")
        print(f"    Strong      (>15% net): {prof15}/{len(success)} = {prof15/len(success)*100:.0f}%")
        print(f"    Target met  (>20% net): {prof20}/{len(success)} = {prof20/len(success)*100:.0f}%")
        print(f"    Median net CAGR       : {sorted_cagrs[n//2]:+.1f}%")
        print(f"    Mean   net CAGR       : {sum(cagrs)/n:+.1f}%")
        print(f"    Bottom / Top          : {min(cagrs):+.1f}% / {max(cagrs):+.1f}%")

    # Comparison vs ML system
    ml_cagrs = [ml_data[t]["net_cagr"]
                for t in success
                if t in ml_data and ml_data[t].get("status") == "success"]
    if ml_cagrs and len(ml_cagrs) >= 3:
        mom_vs_ml = sorted(success.keys())
        matched = [(t, success[t]["net_cagr"], ml_data[t]["net_cagr"])
                   for t in mom_vs_ml
                   if t in ml_data and ml_data[t].get("status") == "success"]
        improvements = [m - l for _, m, l in matched]
        print(f"\n  Momentum vs ML System (matched {len(matched)} tickers):")
        print(f"    Momentum median: {sorted([m for _,m,_ in matched])[len(matched)//2]:+.1f}%")
        print(f"    ML median      : {sorted([l for _,_,l in matched])[len(matched)//2]:+.1f}%")
        print(f"    Avg improvement: {sum(improvements)/len(improvements):+.1f} pp")
        print(f"    Momentum beats ML in: {sum(1 for d in improvements if d > 0)}/{len(improvements)} tickers")

    # Year-by-year
    print(f"\n  ANNUAL RETURNS:")
    print(f"  {'Ticker':<14} {'2022':>7} {'2023':>7} {'2024':>7} {'2025':>7} {'2026':>7}")
    print("  " + "─" * 50)
    for ticker, r in by_cagr:
        ann = r.get("annual", {})
        row = f"  {ticker:<14}"
        for yr in [2022, 2023, 2024, 2025, 2026]:
            v = ann.get(yr)
            row += f"{v:>+6.1f}%" if v is not None else f"{'—':>7}"
        print(row)

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)

    serializable = {}
    for k, v in results.items():
        sd = {}
        for kk, vv in v.items():
            if kk == "trades":
                sd[kk] = [{
                    kk2: (str(vv2) if isinstance(vv2, (pd.Timestamp, str)) else float(vv2))
                    for kk2, vv2 in t.items()
                } for t in vv]
            elif isinstance(vv, dict):
                sd[kk] = {str(k2): float(v2) if isinstance(v2, (int, float, np.floating)) else v2
                           for k2, v2 in vv.items()}
            elif isinstance(vv, (np.floating, np.integer)):
                sd[kk] = float(vv)
            else:
                sd[kk] = vv
        serializable[k] = sd

    out_json = os.path.join(REPORTS_DIR, "universal_backtest_results.json")
    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\n  Results → {out_json}")
    print(f"{'═'*90}")

    return results


if __name__ == "__main__":
    main()

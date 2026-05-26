"""
MARK5 Mean Reversion Standalone Backtest v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OOS period: 2022-01-01 → 2026-04-30.

Reports per stock and overall:
  - Win rate, CAGR, trades
  - Overlap days: bars where both a MR and a hypothetical momentum
    position could be open on the same ticker (overlap check)

CHANGELOG:
- [2026-05-26] v1.0: Initial creation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.data.nse_data_provider import fetch_equity_ohlcv
from core.models.momentum_signal import MomentumSignal
from core.strategies.mean_reversion import MeanReversionStrategy, TARGET_UNIVERSE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [MR_BACKTEST] | %(levelname)s | %(message)s",
)
logger = logging.getLogger("MARK5.MRBacktest")

# ── Config ────────────────────────────────────────────────────────────────────

DATA_START   = "2019-01-01"   # include warmup for SMA200
OOS_START    = "2022-01-01"
END_DATE     = "2026-04-30"
INITIAL_CAP  = 5_00_00_000.0

SLIPPAGE_PCT = 0.001
BROKERAGE    = 20.0
STT_PCT      = 0.001

UNIVERSE = [f"{t}.NS" for t in TARGET_UNIVERSE]


# ── Simulator ─────────────────────────────────────────────────────────────────

def simulate_mr_stock(
    df_full: pd.DataFrame,
    ticker: str,
    oos_start: str,
    equity_start: float,
) -> Dict:
    """
    Simulate MR strategy on a single stock.
    Returns dict with trades, win_rate, final_equity, daily_pnl.
    """
    from core.strategies.mean_reversion import (
        PROFIT_TARGET_PCT, STOP_LOSS_PCT, TIME_STOP_BARS
    )

    ms  = MomentumSignal()
    mr  = MeanReversionStrategy()

    oos_start_ts = pd.Timestamp(oos_start)
    oos_df       = df_full[df_full.index >= oos_start_ts].copy()

    if len(oos_df) < 5:
        return {"trades": [], "win_rate": None, "final_equity": equity_start, "daily_pnl": []}

    equity      = equity_start
    trades      = []
    daily_pnl   = {}  # index → daily pct return while in position

    pos          = False
    entry_price  = 0.0
    entry_bar_i  = 0   # absolute index into df_full
    entry_date   = None

    close_full   = df_full["close"]

    for oos_pos, (date, _) in enumerate(oos_df.iterrows()):
        abs_i = df_full.index.get_loc(date)
        df_slice = df_full.iloc[: abs_i + 1]

        cur = float(close_full.iloc[abs_i])

        if pos:
            # Evaluate exit
            bars_held = abs_i - entry_bar_i
            prev_close = float(close_full.iloc[abs_i - 1]) if abs_i > 0 else cur
            daily_pnl[date] = (cur - prev_close) / prev_close

            low_i = float(df_full["low"].iloc[abs_i]) if "low" in df_full.columns else cur
            profit_px = entry_price * (1 + PROFIT_TARGET_PCT)
            stop_px   = entry_price * (1 - STOP_LOSS_PCT)

            exit_reason = None
            exit_price  = cur

            if cur >= profit_px:
                exit_reason = "profit_target"
                exit_price  = cur
            elif low_i <= stop_px:
                exit_reason = "stop_loss"
                exit_price  = stop_px
            elif bars_held >= TIME_STOP_BARS:
                exit_reason = "time_stop"
                exit_price  = cur

            if exit_reason:
                cost    = BROKERAGE + exit_price * (equity * 0.07 / entry_price) * (STT_PCT + SLIPPAGE_PCT)
                shares  = max(int((equity * 0.07) / entry_price), 1)
                net_pnl = (exit_price - entry_price) * shares - cost
                equity  = max(equity + net_pnl, 1.0)

                trades.append({
                    "ticker":      ticker,
                    "entry_date":  str(entry_date),
                    "exit_date":   str(date.date()),
                    "entry_price": round(entry_price, 2),
                    "exit_price":  round(exit_price, 2),
                    "return_pct":  round((exit_price / entry_price - 1) * 100, 2),
                    "exit_reason": exit_reason,
                    "bars_held":   bars_held,
                })
                pos = False
        else:
            daily_pnl[date] = 0.0

            # Evaluate entry (need full slice for SMA200)
            score = ms.score(df_slice, nifty=None, fii_5d=0.0)
            if mr.should_enter(df_slice, ticker, score, open_positions=None):
                entry_price = cur * (1 + SLIPPAGE_PCT)
                entry_bar_i  = abs_i
                entry_date   = date.date()
                pos          = True
                logger.debug(f"  [{ticker}] ENTER @ {entry_price:.2f} on {date.date()} | score={score:.3f}")

    # Force close any open position at end
    if pos and len(oos_df) > 0:
        last_close = float(close_full.iloc[-1])
        shares = max(int((equity * 0.07) / entry_price), 1)
        net    = (last_close - entry_price) * shares
        equity = max(equity + net, 1.0)
        trades.append({
            "ticker":      ticker,
            "entry_date":  str(entry_date),
            "exit_date":   str(oos_df.index[-1].date()),
            "entry_price": round(entry_price, 2),
            "exit_price":  round(last_close, 2),
            "return_pct":  round((last_close / entry_price - 1) * 100, 2),
            "exit_reason": "end_of_backtest",
            "bars_held":   len(oos_df) - 1,
        })

    win_rate = (
        sum(1 for t in trades if t["return_pct"] > 0) / len(trades)
        if trades else None
    )

    return {
        "trades":      trades,
        "win_rate":    win_rate,
        "final_equity": equity,
        "daily_pnl":   [(str(k.date()), v) for k, v in daily_pnl.items()],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_backtest(
    output_path: str = "reports/mean_reversion_results.json",
) -> Dict:

    logger.info(f"MR Backtest OOS={OOS_START}→{END_DATE}")

    stock_results: Dict[str, Dict] = {}

    for ticker in sorted(UNIVERSE):
        logger.info(f"Loading [{ticker}]...")
        df = fetch_equity_ohlcv(ticker, DATA_START, END_DATE)
        if df is None or len(df) < 210:
            logger.warning(f"[{ticker}] insufficient data — skipping")
            continue

        result = simulate_mr_stock(df, ticker, OOS_START, INITIAL_CAP)
        stock_results[ticker] = {
            "trades":      result["trades"],
            "win_rate":    round(result["win_rate"], 4) if result["win_rate"] is not None else None,
            "n_trades":    len(result["trades"]),
            "final_equity": round(result["final_equity"], 2),
        }
        n_win = sum(1 for t in result["trades"] if t["return_pct"] > 0)
        wr_str = f"{result['win_rate']:.0%}" if result["win_rate"] is not None else "N/A"
        logger.info(f"  [{ticker}] trades={len(result['trades'])} WR={wr_str} wins={n_win}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    all_trades   = [t for r in stock_results.values() for t in r["trades"]]
    total_wins   = sum(1 for t in all_trades if t["return_pct"] > 0)
    overall_wr   = total_wins / len(all_trades) if all_trades else None

    # ── Overlap check ─────────────────────────────────────────────────────────
    # Load momentum portfolio results if available to find overlap days
    overlap_count = 0
    mom_path = "reports/momentum_portfolio_results.json"
    if os.path.exists(mom_path):
        with open(mom_path) as f:
            mom_results = json.load(f)
        mom_trade_dates: Dict[str, set] = {}
        for t in mom_results.get("trades", []):
            tic = t["ticker"].replace(".NS", "")
            mom_trade_dates.setdefault(tic, set()).add(t["entry_date"])

        for t in all_trades:
            base = t["ticker"].replace(".NS", "")
            if base in mom_trade_dates and t["entry_date"] in mom_trade_dates[base]:
                overlap_count += 1

    results = {
        "summary": {
            "oos_start":    OOS_START,
            "end_date":     END_DATE,
            "total_trades": len(all_trades),
            "overall_win_rate": round(overall_wr, 4) if overall_wr is not None else None,
            "overlap_with_momentum_trades": overlap_count,
        },
        "per_stock": stock_results,
        "all_trades": all_trades,
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "═" * 60)
    logger.info(f"  Total trades:   {len(all_trades)}")
    logger.info(f"  Overall WR:     {overall_wr:.0%}" if overall_wr else "  Overall WR: N/A")
    logger.info(f"  Overlap (mom):  {overlap_count}")
    logger.info("═" * 60)
    logger.info(f"Results → {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARK5 Mean Reversion Backtest")
    parser.add_argument("--output", default="reports/mean_reversion_results.json")
    args = parser.parse_args()
    run_backtest(output_path=args.output)

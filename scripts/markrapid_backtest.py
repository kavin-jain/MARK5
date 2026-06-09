#!/usr/bin/env python3
"""
MARKRAPID Backtest Runner
==========================
Run the full OOS backtest for the MARKRAPID swing trading system.

Usage:
    python3 scripts/markrapid_backtest.py
    python3 scripts/markrapid_backtest.py --start 2022-01-01 --end 2026-05-21
    python3 scripts/markrapid_backtest.py --capital 10000 --compound
    python3 scripts/markrapid_backtest.py --tickers HAL,TRENT,BEL,ZOMATO

Output:
    reports/markrapid_results.json — full trade log + metrics
    Console output with trade table + annual returns
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="MARKRAPID — Aggressive Swing Trade Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--start",    default="2022-01-01", help="Backtest start date")
    p.add_argument("--end",      default="2026-05-21", help="Backtest end date")
    p.add_argument("--capital",  type=float, default=10_000, help="Capital per trade (₹)")
    p.add_argument("--compound", action="store_true", help="Reinvest profits")
    p.add_argument("--tickers",  type=str,   default="",    help="Comma-sep tickers (default: full universe)")
    p.add_argument("--no-news",  action="store_true", help="Disable news proxy (pure technical)")
    p.add_argument("--output",   type=str,   default="",    help="Output JSON path (default: reports/)")
    p.add_argument("--quiet",    action="store_true", help="Suppress progress output")
    return p.parse_args()


def main():
    args = parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger("MARKRAPID")
    logger.info("=" * 60)
    logger.info("MARKRAPID — Aggressive Swing Trade Backtest")
    logger.info("=" * 60)
    logger.info(f"Period:  {args.start} → {args.end}")
    logger.info(f"Capital: ₹{args.capital:,.0f} per trade")
    logger.info(f"Mode:    {'COMPOUND' if args.compound else 'FIXED ₹10k'}")

    # ── Import MARKRAPID ────────────────────────────────────────────────────
    from markrapid.backtest import load_universe_data, run_backtest, save_results, print_results
    from markrapid.config import UNIVERSE

    # ── Resolve tickers ─────────────────────────────────────────────────────
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        logger.info(f"Tickers: {tickers}")
    else:
        tickers = UNIVERSE
        logger.info(f"Universe: {len(tickers)} stocks")

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("\nLoading OHLCV data (yfinance cache)...")
    all_data = load_universe_data(
        start=args.start,
        end=args.end,
        tickers=tickers,
        verbose=not args.quiet,
    )

    if not all_data:
        print("ERROR: No data loaded. Check internet connection.", file=sys.stderr)
        sys.exit(1)

    logger.info(f"Loaded {len(all_data)} tickers.\n")

    # ── Run backtest ─────────────────────────────────────────────────────────
    logger.info("Running backtest...")
    results = run_backtest(
        all_data=all_data,
        start=args.start,
        end=args.end,
        capital=args.capital,
        compound=args.compound,
        use_news_proxy=not args.no_news,
        verbose=not args.quiet,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = args.output or str(_PROJECT / "reports" / "markrapid_results.json")
    save_results(results, out_path)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_results(results)

    n_trades = len(results.get("trades", []))
    summary  = results.get("summary", {})
    print(f"Results saved → {out_path}")
    print(f"\nQuick stats: {n_trades} trades | "
          f"WR={summary.get('win_rate_pct', 0):.0f}% | "
          f"EV={summary.get('ev_per_trade_pct', 0):+.1f}%/trade | "
          f"Payoff={summary.get('payoff_ratio', 0):.1f}:1\n")


if __name__ == "__main__":
    main()

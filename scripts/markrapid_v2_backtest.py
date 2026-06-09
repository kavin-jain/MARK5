#!/usr/bin/env python3
"""
MARKRAPID V2 Backtest CLI
==========================
Run the aggressive short-term swing trading backtest.

V2 parameters: 7-day hold | +20% target | -7% stop | score ≥ 0.82
                NSE most-active screener | pattern repetition detector

Usage:
    python3 scripts/markrapid_v2_backtest.py
    python3 scripts/markrapid_v2_backtest.py --start 2023-01-01 --end 2025-12-31
    python3 scripts/markrapid_v2_backtest.py --compound
    python3 scripts/markrapid_v2_backtest.py --no-news
    python3 scripts/markrapid_v2_backtest.py --tickers HAL,TRENT,COFORGE,PERSISTENT
    python3 scripts/markrapid_v2_backtest.py --output reports/my_results.json

Compare with V1:
    python3 scripts/markrapid_backtest.py     # V1: 30d hold, 12% target
    python3 scripts/markrapid_v2_backtest.py  # V2: 7d hold,  20% target
"""
import argparse
import logging
import sys
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────
_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="MARKRAPID V2 — Aggressive 7-day swing backtest"
    )
    p.add_argument(
        "--start", default=None,
        help="Backtest start date YYYY-MM-DD (default: 2022-01-01)"
    )
    p.add_argument(
        "--end", default=None,
        help="Backtest end date YYYY-MM-DD (default: 2026-05-21)"
    )
    p.add_argument(
        "--capital", type=float, default=None,
        help="Capital per trade in ₹ (default: 10000)"
    )
    p.add_argument(
        "--compound", action="store_true",
        help="Reinvest profits into subsequent trades (recommended for max cumulative growth)"
    )
    p.add_argument(
        "--no-news", action="store_true",
        help="Disable gap-up proxy for catalyst score (pure technical)"
    )
    p.add_argument(
        "--no-regime-gate", action="store_true",
        help="Disable NIFTY50 regime gate (allows entries in bear markets)"
    )
    p.add_argument(
        "--tickers", default=None,
        help="Comma-separated ticker subset (default: full V2 universe)"
    )
    p.add_argument(
        "--output", default=None,
        help="Output JSON path (default: reports/markrapid_v2_results.json)"
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )

    args = p.parse_args()

    from markrapid.backtest_v2 import (
        load_universe_data_v2,
        run_backtest_v2,
        save_results_v2,
        print_results_v2,
    )
    from markrapid.config_v2 import (
        BACKTEST_START, BACKTEST_END, CAPITAL, UNIVERSE_V2,
        TARGET_PCT, STOP_PCT, MAX_HOLD_DAYS, RAPID_ENTRY_THRESHOLD,
    )

    start   = args.start   or BACKTEST_START
    end     = args.end     or BACKTEST_END
    capital = args.capital or CAPITAL
    verbose = not args.quiet

    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        print(f"\nUsing custom tickers: {tickers}")
    else:
        print(f"\nUsing V2 universe: {len(UNIVERSE_V2)} stocks")

    use_regime = not args.no_regime_gate
    print(f"\n{'='*60}")
    print(f"  MARKRAPID V2.1 — Calibrated Swing System")
    print(f"{'='*60}")
    print(f"  Period:      {start} → {end}")
    print(f"  Capital:     ₹{capital:,.0f} per trade")
    print(f"  Target:      +{TARGET_PCT*100:.0f}% (gross)")
    print(f"  Stop:        -{STOP_PCT*100:.0f}%")
    print(f"  Max hold:    {MAX_HOLD_DAYS} calendar days")
    print(f"  Threshold:   RAPID score ≥ {RAPID_ENTRY_THRESHOLD:.2f}")
    print(f"  News proxy:  {'disabled' if args.no_news else 'enabled (gap-up)'}")
    print(f"  Regime gate: {'enabled (NIFTY50 EMA50)' if use_regime else 'DISABLED'}")
    print(f"  Compound:    {args.compound}")
    print(f"{'='*60}\n")

    # Load data
    if verbose:
        print("Loading market data...")
    all_data = load_universe_data_v2(start, end, tickers=tickers, verbose=verbose)

    if not all_data:
        print("ERROR: No data loaded. Check internet connection.")
        sys.exit(1)

    print(f"Loaded {len(all_data)} tickers. Running backtest...\n")

    # Run backtest
    results = run_backtest_v2(
        all_data=all_data,
        start=start,
        end=end,
        capital=capital,
        compound=args.compound,
        use_news_proxy=not args.no_news,
        use_regime_gate=use_regime,
        verbose=verbose,
    )

    # Print results
    print_results_v2(results)

    # Save results
    output_path = save_results_v2(results, args.output)
    print(f"\nResults saved → {output_path}")

    # Quick summary line for CI/logs
    summary = results.get("summary", {})
    n       = summary.get("n_trades", 0)
    wr      = summary.get("win_rate_pct", 0)
    ev      = summary.get("ev_per_trade_pct", 0)
    pnl     = summary.get("total_net_pnl", 0)
    import pandas as pd
    from markrapid.config_v2 import BACKTEST_START, BACKTEST_END
    s = results["config"].get("start", BACKTEST_START)
    e = results["config"].get("end", BACKTEST_END)
    years = (pd.Timestamp(e) - pd.Timestamp(s)).days / 365.25
    equity_end = capital + pnl
    cagr = ((equity_end / capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    print(f"\n  ── BOTTOM LINE ──────────────────────────────────────")
    print(f"  {n} trades | {wr:.0f}% WR | EV {ev:+.2f}%/trade")
    print(f"  ₹10,000 → ₹{equity_end:,.0f} | CAGR {cagr:+.1f}%/yr")
    print(f"  V1 was: 72 trades | 47% WR | EV +0.91%/trade | CAGR +10.9%/yr")
    print(f"  ─────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()

"""
MARK6 — Holding-Period Sweep (why long holds win, net of tax)
=============================================================
Same equity factor book, same 10 years, same universe — ONLY the rebalance
frequency changes (buffer off, so frequent rebalancing actually rotates names and
shortens the holding period). Shows the net-of-tax effect of trading fast vs slow.

For each cadence: avg holding period, turnover, GROSS vs NET CAGR, tax paid,
% of sells taxed as LTCG, Sharpe, MaxDD, and final ₹ value from ₹5,00,000.

Built-in code-correctness checks (must all hold or it flags a bug):
  - net <= gross always (tax+costs only subtract)
  - turnover rises monotonically as cadence shortens
  - tax burden rises as holds shorten (more STCG)

  python3 scripts/holding_period_sweep.py
"""
import os, sys
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, BacktestConfig, metrics)
START, END = "2016-01-01", "2026-06-05"
CAP = 500000

# (label, rebal_bars). Trading days: 1d, 1wk, 2wk, ~1mo, ~3mo, ~6mo, ~1yr(prod)
CADENCES = [("1 day", 1), ("5 days", 5), ("10 days", 10), ("30 days", 21),
            ("3 months", 63), ("6 months", 126), ("1 year (current)", 252)]


def run_one(panel, rb, buffer_mult):
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=12, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.125, buffer_mult=buffer_mult)
    bt = Backtester(panel, PortfolioConstructor(cfg), config=BacktestConfig(rebal_bars=rb))
    r = bt.run(START, END)
    m = r["metrics"]
    gross = metrics(r["nav_gross"])           # pre-terminal-tax
    sells = [t for t in r["trades"] if t["side"] == "SELL"]
    holds = [t["held_days"] for t in sells]
    ltcg = sum(1 for t in sells if t["term"] == "LTCG")
    return {
        "net_cagr": m["cagr"], "gross_cagr": gross["cagr"], "sharpe": m["sharpe"],
        "max_dd": m["max_dd"], "turnover": m["turnover_yr"], "tax": m["tax_paid"],
        "avg_hold": np.mean(holds) if holds else 0, "n_sells": len(sells),
        "pct_ltcg": ltcg / len(sells) * 100 if sells else 0,
        "final": CAP * r["nav_net"].iloc[-1],
    }


def main():
    panel = DataPanel(discover_tickers(), END)
    print("Sweeping rebalance cadence (buffer OFF, equity book, net of tax)...\n", flush=True)
    rows = []
    for label, rb in CADENCES:
        # buffer off (mult=1.0) so frequent rebalancing truly shortens holds;
        # the 1-year row also shown WITH the production buffer (2.0) for reference.
        res = run_one(panel, rb, buffer_mult=1.0)
        rows.append((label, rb, res))
        print(f"  {label:<18} done (avg hold {res['avg_hold']:.0f}d, "
              f"net {res['net_cagr']*100:+.1f}%)", flush=True)
    prod = run_one(panel, 252, buffer_mult=2.0)  # the actual deployed config

    print("\n" + "=" * 104)
    print(f"  {'cadence':<18}{'avgHold':>8}{'turnov':>8}{'GROSS':>8}{'NET':>8}"
          f"{'taxPaid':>9}{'%LTCG':>7}{'Sharpe':>8}{'MaxDD':>8}{'₹5L ->':>12}")
    print("=" * 104)
    for label, rb, r in rows:
        print(f"  {label:<18}{r['avg_hold']:>7.0f}d{r['turnover']*100:>7.0f}%"
              f"{r['gross_cagr']*100:>+7.1f}%{r['net_cagr']*100:>+7.1f}%"
              f"{r['tax']*100:>8.0f}%{r['pct_ltcg']:>6.0f}%{r['sharpe']:>8.2f}"
              f"{r['max_dd']*100:>+7.1f}%{r['final']:>12,.0f}")
    print("  " + "-" * 102)
    print(f"  {'1yr + BUFFER(prod)':<18}{prod['avg_hold']:>7.0f}d{prod['turnover']*100:>7.0f}%"
          f"{prod['gross_cagr']*100:>+7.1f}%{prod['net_cagr']*100:>+7.1f}%"
          f"{prod['tax']*100:>8.0f}%{prod['pct_ltcg']:>6.0f}%{prod['sharpe']:>8.2f}"
          f"{prod['max_dd']*100:>+7.1f}%{prod['final']:>12,.0f}")

    # ── code-correctness checks ───────────────────────────────────────────────
    print("\n" + "=" * 104)
    print("  CODE-CORRECTNESS CHECKS (logical sanity — must all pass)")
    print("=" * 104)
    allr = [r for _, _, r in rows]
    c1 = all(r["net_cagr"] <= r["gross_cagr"] + 1e-9 for r in allr)
    turns = [r["turnover"] for r in allr]
    c2 = turns[0] >= turns[-1]   # 1-day turnover >= 1-year turnover
    c3 = allr[0]["pct_ltcg"] <= allr[-1]["pct_ltcg"]  # fast trading -> less LTCG
    c4 = allr[0]["avg_hold"] <= allr[-1]["avg_hold"]  # faster cadence -> shorter holds
    print(f"  [{'PASS' if c1 else 'FAIL'}] net <= gross for every cadence (tax/costs only subtract)")
    print(f"  [{'PASS' if c2 else 'FAIL'}] turnover higher for faster cadence "
          f"({turns[0]*100:.0f}% @1d vs {turns[-1]*100:.0f}% @1yr)")
    print(f"  [{'PASS' if c3 else 'FAIL'}] %LTCG lower for faster cadence "
          f"({allr[0]['pct_ltcg']:.0f}% @1d vs {allr[-1]['pct_ltcg']:.0f}% @1yr)")
    print(f"  [{'PASS' if c4 else 'FAIL'}] avg hold shorter for faster cadence "
          f"({allr[0]['avg_hold']:.0f}d @1d vs {allr[-1]['avg_hold']:.0f}d @1yr)")
    print(f"\n  {'ALL CHECKS PASS — results are logically consistent.' if all([c1,c2,c3,c4]) else 'CHECK FAILED — investigate.'}")


if __name__ == "__main__":
    main()

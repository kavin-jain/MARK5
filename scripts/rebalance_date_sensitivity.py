"""
MARK6 — F4: rebalance-date (anchor) sensitivity of the deployed system
======================================================================
The deployed engine rebalances every 126 bars anchored to the backtest start
date. If the headline result depended on WHICH day the cycle lands (calendar/
expiry effects, F4), staggering the anchor would scatter the results. Small
dispersion = anchor-robust; large = fragile (and a possible calendar edge).

Runs the deployed equity config from 13 staggered start anchors (0..120 bars,
step 10 ≈ every 2 weeks across a half-cycle) over the same ~10y span.

  python3 scripts/rebalance_date_sensitivity.py
"""
import os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester)
from core.portfolio.backtest import BacktestConfig

END = "2026-06-09"
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}


def main():
    panel = DataPanel(discover_tickers(), END)
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=12, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.125, factor_weights=MOM)
    cal = panel.trading_calendar("2016-01-01", END)

    print(f"\n══ F4 — anchor sensitivity (126d rebalance, net, ~10y each) ══")
    print(f"  {'anchor (start)':<18}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>9}")
    cagrs, sharpes = [], []
    for k in range(0, 121, 10):
        start = str(cal[k].date())
        m = Backtester(panel, PortfolioConstructor(cfg),
                       BacktestConfig(rebal_bars=126)).run(start, END)["metrics"]
        cagrs.append(m["cagr"]); sharpes.append(m["sharpe"])
        print(f"  {start:<18}{m['cagr']*100:>+7.1f}%{m['sharpe']:>8.2f}"
              f"{m['max_dd']*100:>+8.1f}%")
    c = np.array(cagrs)
    print(f"\n  CAGR: mean {c.mean()*100:+.1f}%  std {c.std()*100:.1f}pp  "
          f"min {c.min()*100:+.1f}%  max {c.max()*100:+.1f}%  "
          f"spread {(c.max()-c.min())*100:.1f}pp")
    print(f"  Sharpe: mean {np.mean(sharpes):.2f}  "
          f"min {np.min(sharpes):.2f}  max {np.max(sharpes):.2f}")
    verdict = ("ROBUST — anchor day does not drive the result"
               if c.std() < 0.015 else
               "SENSITIVE — investigate calendar structure before trusting headline")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()

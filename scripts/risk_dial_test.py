"""
MARK6 — Risk-Dial Frontier (how high can return go honestly, at what drawdown?)
===============================================================================
Leverage is dead (financing≈return). The only remaining honest lever to higher
return is MORE RISK via concentration + harder factor tilt. This maps the real
return/drawdown frontier: sweep n_hold (fewer = more concentrated) and
tilt_strength, all OOS net of tax. Flags any config that nets >=18% at a
survivable (> -55%) drawdown. No p-hacking — full grid reported.

  python3 scripts/risk_dial_test.py
"""
import os, sys
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester)

END = "2026-05-21"
MOM_HEAVY = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}


def main():
    panel = DataPanel(discover_tickers(), END)
    windows = [("2016-01-01", END, "full16-26"), ("2022-01-01", END, "recent22-26")]
    n_holds = [5, 8, 12, 20]
    tilts = [0.5, 1.5, 3.0]

    for s, e, lab in windows:
        print("\n" + "=" * 92)
        print(f"  {lab}  — risk-dial sweep (concentration x tilt), net of tax. "
              f"flag = >=18% CAGR & DD>-55%")
        print("=" * 92)
        print(f"  {'n_hold':>7}{'tilt':>6}{'wts':>10}{'netCAGR':>9}{'Sharpe':>8}{'MaxDD':>8}{'flag'}")
        best = None
        for wlabel, wts in [("blend", None), ("mom!", MOM_HEAVY)]:
            for n in n_holds:
                for t in tilts:
                    kw = dict(mode="factor_tilt", n_hold=n, base_weighting="inverse_vol",
                              tilt_strength=t, max_weight=max(0.08, 1.5 / n))
                    if wts:
                        kw["factor_weights"] = wts
                    cfg = ConstructionConfig(**kw)
                    bt = Backtester(panel, PortfolioConstructor(cfg))
                    m = bt.run(s, e)["metrics"]
                    flag = "  <<<" if (m["cagr"] >= 0.18 and m["max_dd"] > -0.55) else ""
                    if best is None or m["cagr"] > best[0]:
                        best = (m["cagr"], m["max_dd"], n, t, wlabel)
                    print(f"  {n:>7}{t:>6.1f}{wlabel:>10}{m['cagr']*100:>+8.1f}%"
                          f"{m['sharpe']:>8.2f}{m['max_dd']*100:>+7.1f}%{flag}")
        print(f"  -> max CAGR: {best[0]*100:+.1f}% at DD {best[1]*100:+.1f}% "
              f"(n_hold={best[2]}, tilt={best[3]}, {best[4]})")


if __name__ == "__main__":
    main()

"""
MARK6 — Generate TODAY's deployable portfolio (the executable deliverable).
==========================================================================
Turns the validated, audited system into a concrete instruction: exactly which
stocks and weights to hold now. Config is the deployed v7.1 book:
  - 20-name factor book (momentum-heavy, inverse-vol weighted)
  - GOLDBEES (gold) + MON100 (US Nasdaq-100) diversifier sleeves

PAPER mode. Equity book refreshed every 6 months, sleeves annually.

  python3 scripts/generate_portfolio.py --capital 500000
"""
import os, sys, argparse, json
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, FactorLibrary, composite_score)

GOLD_WEIGHT = 0.25
US_WEIGHT = 0.25            # MON100 (Nasdaq-100) — uncorrelated sleeve
STALE_WARN_DAYS = 14        # warn if the data cache lags today by more than this


def validate(args):
    if not (0 < args.capital <= 1e12):
        sys.exit(f"ERROR: --capital must be a positive amount (got {args.capital:,.0f}). "
                 f"Example: --capital 500000")
    for name, v in (("--gold", args.gold), ("--us", args.us)):
        if not (0 <= v < 1):
            sys.exit(f"ERROR: {name} must be a fraction in [0, 1) (got {v}).")
    if args.gold + args.us >= 1:
        sys.exit(f"ERROR: gold + US sleeves must leave room for equity "
                 f"(gold {args.gold} + us {args.us} >= 1).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=500000)
    ap.add_argument("--gold", type=float, default=GOLD_WEIGHT)
    ap.add_argument("--us", type=float, default=US_WEIGHT)
    args = ap.parse_args()
    validate(args)

    tickers = discover_tickers()
    if not tickers:
        sys.exit("ERROR: data cache is empty — run scripts/refetch_all.py first "
                 "(it fetches the pinned universe in config/universe_tickers.json).")
    panel = DataPanel(tickers, str(pd.Timestamp.today().date()))
    asof = panel.close.index[-1]                       # latest real data date
    age = (pd.Timestamp.today() - asof).days
    if age > STALE_WARN_DAYS:
        print(f"  ⚠ DATA IS {age} DAYS OLD (last print {asof.date()}). Holdings below "
              f"reflect that date — run scripts/refetch_all.py before acting on them.")

    cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.08,
                             factor_weights={"momentum": 0.45, "low_vol": 0.15,
                                             "trend": 0.25, "stability": 0.15})
    con = PortfolioConstructor(cfg)

    elig = panel.eligible(asof, 252, 0.40)
    if len(elig) < cfg.n_hold:
        sys.exit(f"ERROR: only {len(elig)} eligible names as-of {asof.date()} — "
                 f"cannot build a {cfg.n_hold}-stock book. Data cache is likely "
                 f"stale or incomplete; run scripts/refetch_all.py.")
    facs = FactorLibrary.DEFAULT_FACTORS
    raw = {f: {} for f in facs}
    vol = {}
    for t in elig:
        row = FactorLibrary.compute_all(panel.close[t]).loc[:asof]
        if row.empty:
            continue
        last = row.iloc[-1]
        for f in facs:
            raw[f][t] = last.get(f, float("nan"))
        vol[t] = -last.get("low_vol", float("nan"))
    comp = composite_score({f: pd.Series(raw[f]) for f in facs}, cfg.factor_weights)
    w_eq = con.target_weights(comp, pd.Series(vol), currently_held=[])

    eq_frac = 1 - args.gold - args.us
    eq_cap = args.capital * eq_frac
    print("=" * 64)
    print(f"  MARK6 DEPLOYABLE PORTFOLIO  (PAPER)   as-of {asof.date()}")
    print(f"  Capital ₹{args.capital:,.0f}   |   {100*eq_frac:.0f}% equity / "
          f"{100*args.gold:.0f}% gold / {100*args.us:.0f}% US")
    print("=" * 64)
    print(f"  {'#':<3}{'ticker':<16}{'weight':>9}{'₹ alloc':>14}")
    for i, (t, w) in enumerate(w_eq.sort_values(ascending=False).items(), 1):
        print(f"  {i:<3}{t:<16}{w*100*eq_frac:>8.1f}%{eq_cap*w:>13,.0f}")
    print(f"  {'':3}{'GOLDBEES (gold)':<16}{args.gold*100:>8.1f}%{args.capital*args.gold:>13,.0f}")
    print(f"  {'':3}{'MON100 (US Nq100)':<16}{args.us*100:>8.1f}%{args.capital*args.us:>13,.0f}")
    print("=" * 64)
    print(f"  Holdings: {len(w_eq)} stocks + gold + US ETF")
    print(f"  Rebalance: equity book every 6 MONTHS (P12), sleeves annually.")
    print(f"  Tax: harvest nothing mid-cycle; gains/losses net at FY level (P11).")
    # expected-performance footer is READ from the generated evidence, never hardcoded
    res_path = os.path.join(_ROOT, "reports", "mark6_results.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            res = json.load(f)
        full = next((w for k, w in res.get("windows", {}).items() if "FULL" in k), None)
        if full:
            print(f"  Backtested (see reports/): equity sleeve {full['factor']['cagr']*100:+.1f}% "
                  f"net CAGR full-period, vs Nifty TRI {full['nifty']['cagr']*100:+.1f}%.")
    print(f"  Caveats: survivorship-inflated ~1-2pp; expect -25 to -35% drawdowns")
    print(f"  you MUST hold through. PAPER mode — not investment advice.")


if __name__ == "__main__":
    main()

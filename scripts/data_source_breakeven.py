"""
Is any PAID data source worth it? Break-even arithmetic, not opinion.
====================================================================
A data subscription is an investment like any other: it must return more than it
costs. That is computable, and the answer turns out to depend almost entirely on
CAPITAL rather than on how good the data is.

The chain, using standard active-management theory:

  Grinold's Law            IR = IC x sqrt(breadth)
  Clarke-de Silva-Thorley  realised IR = TC x IC x sqrt(breadth)

`TC` is the TRANSFER COEFFICIENT — how much of a signal actually reaches the
portfolio after real-world constraints. A long-only book with an 8% name cap, a
30% sector cap and only 20 slots cannot express a signal fully; empirically
TC ~ 0.3-0.5 for constrained long-only. Ignoring TC is the single most common way
data-vendor ROI gets overstated, so it is explicit here.

Then:
  extra gross return = extra IR x portfolio volatility
  extra net return   = extra gross x (1 - effective tax) - extra turnover cost
  rupee value        = extra net return x capital
  VERDICT            = rupee value vs annual subscription cost

Breadth is measured from THIS system: 20 names, refreshed twice a year.
Volatility, tax rate and the incumbent signal's strength are taken from the
deployed book, not from assumptions.

  python3 scripts/data_source_breakeven.py
"""
import json
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── measured properties of the deployed system (not assumptions) ─────────────
N_HOLD, REBAL_PER_YEAR = 20, 2
BREADTH = N_HOLD * REBAL_PER_YEAR          # 40 independent bets per year
EQ_VOL = 0.221                             # equity sleeve annualised vol, measured
EQ_WEIGHT = 0.50                           # equity is half the book
TC = 0.40                                  # transfer coefficient, long-only + caps
EFF_TAX = 0.15                             # blended; 88% of gains are LTCG @12.5%
EXTRA_TURNOVER_COST = 0.002                # a new signal adds trading; 0.2%/yr

CAPITALS = [5e5, 25e5, 1e7, 5e7, 2.5e8, 1e9]

# Indicative annual costs in INR. VERIFY BEFORE PURCHASE — vendor pricing changes
# and several of these are quote-only. They are order-of-magnitude, not offers.
SOURCES = [
    {"name": "Kite Connect (data + execution)", "cost": 24_000,
     "category": "intraday/tick",
     "note": "F1's enabler. Intraday effects documented but every public backtest "
             "excludes costs, and SEBI finds 70-93% of retail intraday traders lose "
             "money. Fundamentally incompatible with this book's long-hold tax design."},
    {"name": "Screener.in premium", "cost": 4_000, "category": "fundamentals",
     "note": "Fundamentals as a TILT already falsified on 12 years of real data (K15)."},
    {"name": "Trendlyne premium", "cost": 12_000, "category": "shareholding + estimates",
     "note": "Shareholding path explicitly falsified (K7/I1): the free NSE XBRL run "
             "concluded paid Trendlyne would NOT have helped. Estimate-revision data "
             "is the part that is genuinely untested."},
    {"name": "Analyst estimate revisions (institutional feed)", "cost": 150_000,
     "category": "estimates",
     "note": "The ONE category with strong global literature that this project has "
             "never tested and cannot obtain free. Chan-Jegadeesh-Lakonishok: revision "
             "momentum is among the most robust documented anomalies."},
    {"name": "Refinitiv / Bloomberg terminal", "cost": 2_000_000,
     "category": "everything",
     "note": "Institutional pricing. Delivers estimates, PIT fundamentals, holdings."},
]

# Plausible IC for a genuinely orthogonal, well-documented signal. Anchored to
# what this project has actually measured: the ownership signal died at IC -0.025;
# delivery data reached +0.023 and failed significance; the incumbent factor blend's
# own implied non-factor IC is ~0.03.
IC_SCENARIOS = [("weak", 0.03), ("good", 0.05), ("exceptional", 0.08)]


def extra_net_return(ic):
    """Incremental NET return to the whole book from an orthogonal signal of this IC."""
    ir = TC * ic * np.sqrt(BREADTH)          # realised information ratio
    gross_eq = ir * EQ_VOL                   # on the equity sleeve
    net_eq = gross_eq * (1 - EFF_TAX) - EXTRA_TURNOVER_COST
    return max(0.0, net_eq) * EQ_WEIGHT      # equity is half the book


def main():
    print("\n" + "=" * 92)
    print("  WHAT A NEW SIGNAL IS WORTH, BY IC")
    print("=" * 92)
    print(f"  breadth = {N_HOLD} names x {REBAL_PER_YEAR} refreshes = {BREADTH} bets/yr")
    print(f"  transfer coefficient {TC} (long-only, 8% name cap, 20 slots)")
    print(f"  equity vol {EQ_VOL*100:.1f}%, equity weight {EQ_WEIGHT*100:.0f}%, "
          f"effective tax {EFF_TAX*100:.0f}%\n")
    print(f"  {'IC':>12}{'realised IR':>14}{'extra NET return to the book':>32}")
    print("  " + "-" * 88)
    vals = {}
    for lab, ic in IC_SCENARIOS:
        r = extra_net_return(ic)
        vals[lab] = r
        ir = TC * ic * np.sqrt(BREADTH)
        print(f"  {lab:>6} {ic:<5.3f}{ir:>14.3f}{r*100:>29.2f}pp")

    print("\n" + "=" * 92)
    print("  BREAK-EVEN — annual rupee value of the signal vs subscription cost")
    print("=" * 92)
    header = f"  {'capital':>12}" + "".join(f"{lab:>14}" for lab, _ in IC_SCENARIOS)
    print(header)
    print("  " + "-" * 88)
    for c in CAPITALS:
        row = f"  Rs {c/1e5:>7,.0f}L"
        for lab, _ in IC_SCENARIOS:
            row += f"{vals[lab]*c:>13,.0f}"
        print(row)
    print("\n  (rupees per year of extra NET return, at each capital)")

    print("\n" + "=" * 92)
    print("  VERDICT PER SOURCE — minimum capital for the source to pay for itself")
    print("=" * 92)
    print(f"  {'source':<44}{'cost/yr':>10}   {'break-even capital (good IC 0.05)':>34}")
    print("  " + "-" * 88)
    out = []
    for s in SOURCES:
        need = s["cost"] / vals["good"] if vals["good"] > 0 else float("inf")
        verdict = ("viable at your capital" if need <= 5e5 else
                   f"needs Rs {need/1e5:,.0f}L+")
        print(f"  {s['name']:<44}{s['cost']:>10,}   {verdict:>34}")
        out.append({**s, "breakeven_capital_inr": float(need)})

    print("\n" + "=" * 92)
    print("  READING")
    print("=" * 92)
    v5 = vals["good"] * 5e5
    print(f"  At Rs 5,00,000, a GOOD orthogonal signal (IC 0.05) is worth about")
    print(f"  Rs {v5:,.0f}/year. Every paid source above costs more than that.")
    print(f"  The cheapest (Screener, Rs 4,000) needs roughly "
          f"Rs {SOURCES[1]['cost']/vals['good']/1e5:,.0f} lakh of capital to break even —")
    print(f"  and its category is already falsified here anyway.")
    print()
    print("  The constraint is NOT data quality. It is that a percentage edge on a small")
    print("  book produces few rupees, while data is priced in absolute rupees. Paid data")
    print("  becomes rational somewhere around Rs 1-5 crore, not at retail scale.")

    p = os.path.join(_ROOT, "reports", "data_source_breakeven.json")
    json.dump({"assumptions": {"breadth": BREADTH, "transfer_coefficient": TC,
                               "equity_vol": EQ_VOL, "equity_weight": EQ_WEIGHT,
                               "effective_tax": EFF_TAX,
                               "extra_turnover_cost": EXTRA_TURNOVER_COST},
               "value_by_ic": {k: float(v) for k, v in vals.items()},
               "sources": out}, open(p, "w"), indent=1)
    print(f"\n  saved -> {p}\n")


if __name__ == "__main__":
    main()

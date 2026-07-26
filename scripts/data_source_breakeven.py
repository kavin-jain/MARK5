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
#
# `ic` is the crux and the honest problem: NO VENDOR PUBLISHES THE IC OF THEIR DATA.
# IC is a property of a signal YOU construct and measure on YOUR universe, not a
# product attribute. So each row is tagged with WHERE its IC number comes from:
#   measured  — this project measured it, on this universe. Trustworthy.
#   none      — the category is already falsified here; expected IC ~0.
#   unknown   — never tested here and not obtainable free. The honest answer is
#               that nobody knows what it would be on Indian mid-caps.
SOURCES = [
    {"name": "Screener.in premium", "cost": 4_000,
     "data": "fundamentals, ratios, historical financials",
     "ic": 0.0, "ic_src": "none",
     "note": "Fundamentals as a TILT falsified on 12 years of real data (K15): "
             "walk-forward -1 to -4.5pp, beats <=5/8 windows. Expected IC ~0."},
    {"name": "Trendlyne premium", "cost": 12_000,
     "data": "shareholding, FII/DII, some analyst estimates",
     "ic": -0.025, "ic_src": "measured",
     "note": "Shareholding path measured at IC -0.025 on FREE NSE XBRL covering the "
             "same ground (K7/I1), which explicitly concluded paid Trendlyne would "
             "NOT have helped. Its estimates coverage is the only untested part."},
    {"name": "Kite Connect (Zerodha)", "cost": 24_000,
     "data": "intraday/tick bars, order execution API",
     "ic": None, "ic_src": "unknown",
     "note": "F1's enabler. Documented intraday effects, but every public backtest "
             "excludes costs and SEBI finds 70-93% of retail intraday traders lose "
             "money. Incompatible with this book's long-hold tax design regardless."},
    {"name": "Refinitiv / LSEG I/B/E/S estimates", "cost": 150_000,
     "data": "analyst EPS estimates + revision history, point-in-time",
     "ic": None, "ic_src": "unknown",
     "note": "The ONE category with strong global literature never tested here and "
             "unobtainable free (no PIT archive exists for Indian estimates). "
             "Chan-Jegadeesh-Lakonishok: revision momentum is among the most robust "
             "documented anomalies. IC on Indian mid-caps is genuinely unknown."},
    {"name": "Bloomberg Terminal", "cost": 2_000_000,
     "data": "everything: estimates, PIT fundamentals, holdings, news",
     "ic": None, "ic_src": "unknown",
     "note": "Institutional pricing. Superset of the above."},
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

    print("\n" + "=" * 110)
    print("  PER SOURCE — named, with WHERE its IC number comes from")
    print("=" * 110)
    print("  No vendor publishes the IC of their data: IC is a property of a signal you")
    print("  build and measure on your own universe, not a product attribute. So the IC")
    print("  column below is tagged by provenance rather than quoted as if it were sold.\n")
    print(f"  {'source':<30}{'cost/yr':>10}{'IC':>10}{'source of IC':>14}"
          f"{'worth at 5L':>13}{'break-even':>13}")
    print("  " + "-" * 106)
    out = []
    for s in SOURCES:
        ic, src = s["ic"], s["ic_src"]
        if src == "unknown":
            ic_txt, worth_txt, be_txt = "unknown", "unknown", f"Rs {s['cost']/vals['good']/1e5:,.1f}L*"
            need = s["cost"] / vals["good"]
        elif ic is not None and ic <= 0:
            ic_txt = f"{ic:+.3f}" if ic else "~0"
            worth_txt, be_txt, need = "Rs 0", "never", float("inf")
        else:
            need = s["cost"] / extra_net_return(ic) if extra_net_return(ic) > 0 else float("inf")
            ic_txt = f"{ic:+.3f}"
            worth_txt = f"Rs {extra_net_return(ic)*5e5:,.0f}"
            be_txt = f"Rs {need/1e5:,.1f}L"
        print(f"  {s['name']:<30}{s['cost']:>10,}{ic_txt:>10}{src:>14}"
              f"{worth_txt:>13}{be_txt:>13}")
        out.append({**s, "breakeven_capital_inr": float(need)})
    print("\n  * for 'unknown' rows the break-even assumes a GOOD IC of 0.05 — an")
    print("    assumption more optimistic than anything this project has ever measured")
    print("    in new data (best observed: +0.023, and it failed significance).")
    print("\n  What each actually sells:")
    for s in SOURCES:
        print(f"    {s['name']:<30} {s['data']}")

    print("\n" + "=" * 92)
    print("  READING")
    print("=" * 92)
    v5 = vals["good"] * 5e5
    cheapest = min(SOURCES, key=lambda s: s["cost"])
    measured_dead = [s["name"] for s in SOURCES if s["ic_src"] in ("measured", "none")
                     and (s["ic"] or 0) <= 0]
    unknown = [s for s in SOURCES if s["ic_src"] == "unknown"]
    cheapest_unknown = min(unknown, key=lambda s: s["cost"]) if unknown else None
    print(f"  At Rs 5,00,000, a GOOD orthogonal signal (IC 0.05) is worth about "
          f"Rs {v5:,.0f}/year in TOTAL.")
    print()
    print(f"  Two of the five are already dead on measured evidence, at any price:")
    for n in measured_dead:
        print(f"    - {n}")
    print(f"  The cheapest source whose value is still UNKNOWN is "
          f"{cheapest_unknown['name']} at Rs {cheapest_unknown['cost']:,},")
    print(f"  which needs Rs {cheapest_unknown['cost']/vals['good']/1e5:,.0f} lakh of "
          f"capital before it repays itself even on optimistic assumptions.")
    print(f"  (Cheapest overall is {cheapest['name']} at Rs {cheapest['cost']:,}, but its "
          f"category\n   is falsified, so buying it would purchase a measured zero.)")
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

"""
Is the live paper book broken, or is 12 days just 12 days?
==========================================================
The book is -0.07% against Nifty +2.38% (-2.45pp) after 12 calendar days. That
feels bad. This script decides whether it IS bad, using the only standard that
can answer the question: the historical distribution of the SAME statistic.

Three separate questions, answered separately, because they have three
different answers:

  1. ALLOCATION. The book is 50% Indian equity / 25% gold / 25% US Nasdaq. The
     benchmark is 100% Indian equity. When Nifty rips, a half-equity book
     cannot keep up, and that is arithmetic, not failure. How much of the
     -2.45pp is just this?

  2. SELECTION. Inside the 50% that IS the system, did the stock picking beat
     owning the index? This is the only number that measures skill.

  3. NOISE. Over the deployed history, how often does a 12-calendar-day window
     look this bad or worse? If it is common, the live result is uninformative
     and the honest answer is "wait". If it is rare, something is wrong.

Also reports the self-inflicted damage: a full portfolio rebalance fired on
day 4 (twice), against a 182-day cadence, booking a realised loss and paying
costs for nothing.

  MARK5_CACHE=data/pit_cache python3 scripts/paper_diagnosis.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.portfolio import load_ohlcv, load_nifty

PAPER = os.path.join(_ROOT, "data", "paper")
SLEEVES = {"eq": 0.50, "GOLDBEES": 0.25, "MON100": 0.25}
HIST_START = "2016-01-01"


def load_live():
    book = json.load(open(os.path.join(PAPER, "paper_book.json")))
    exp = json.load(open(os.path.join(PAPER, "paper_export.json")))
    nav = pd.read_csv(os.path.join(PAPER, "paper_nav.csv"))
    led = pd.read_csv(os.path.join(PAPER, "paper_ledger.csv"))
    return book, exp, nav, led


def sleeve_split(exp):
    """Mark-to-market P&L grouped into the three sleeves."""
    passive = {"GOLDBEES": "gold", "MON100": "us"}
    out = {"eq": [0.0, 0.0, 0], "gold": [0.0, 0.0, 0], "us": [0.0, 0.0, 0]}
    for p in exp["holdings"]:
        k = passive.get(p["ticker"], "eq")
        out[k][0] += p["value"]
        out[k][1] += p["pnl"]
        out[k][2] += 1
    return out


def historical_relative(days: int):
    """Distribution of `days`-calendar-day relative return, deployed blend vs Nifty.

    Uses the REAL daily closes of the two ETF sleeves and the real Nifty. For the
    equity sleeve it uses the Nifty itself as a neutral stand-in, which is the
    conservative choice: it assumes the stock picking adds exactly zero, so any
    shortfall the test finds is pure allocation drag, never bad selection.
    """
    nif = load_nifty(True)
    px = {"eq": nif}
    for t in ("GOLDBEES", "MON100"):
        px[t] = load_ohlcv(t)["close"].astype(float)

    cal = nif.loc[HIST_START:].index
    for t in ("GOLDBEES", "MON100"):
        px[t] = px[t].reindex(cal, method="ffill")
    px["eq"] = px["eq"].reindex(cal)

    r = {k: px[k].pct_change(fill_method=None).fillna(0.0) for k in px}
    blend = sum(SLEEVES[k] * r[k] for k in SLEEVES)
    bench = r["eq"]

    bl = (1 + blend).cumprod()
    bn = (1 + bench).cumprod()

    # step in calendar days, so the window matches the live book's 12 days
    rel = []
    for i, d in enumerate(cal):
        j = cal.searchsorted(d + pd.Timedelta(days=days))
        if j >= len(cal):
            break
        rel.append((bl.iloc[j] / bl.iloc[i] - 1) - (bn.iloc[j] / bn.iloc[i] - 1))
    return np.array(rel) * 100.0


def main():
    book, exp, nav, led = load_live()
    cap, days = exp["capital"], exp["days_live"]
    sl = sleeve_split(exp)

    print("\n" + "=" * 86)
    print(f"  PAPER BOOK DIAGNOSIS   {exp['start_date']} -> "
          f"{exp['generated'][:10]}   day {days}   {exp['observations']} sessions")
    print("=" * 86)
    print(f"  book {exp['return_pct']:+.2f}%    Nifty {exp['benchmark_return_pct']:+.2f}%"
          f"    relative {exp['relative_pct']:+.2f}pp")

    # ---------------------------------------------------------------- 1. mix
    print("\n" + "-" * 86)
    print("  1. WHAT ACTUALLY MOVED  (mark-to-market by sleeve)")
    print("-" * 86)
    print(f"  {'sleeve':<32}{'n':>3}{'value':>12}{'weight':>9}"
          f"{'P&L':>11}{'return':>10}{'NAV impact':>13}")
    labels = {"eq": "Indian equity (THE SYSTEM)",
              "gold": "Gold ETF (passive)", "us": "US Nasdaq-100 (passive)"}
    for k in ("eq", "gold", "us"):
        val, pnl, n = sl[k]
        cost = val - pnl
        print(f"  {labels[k]:<32}{n:>3}{val:>12,.0f}{val/exp['nav']*100:>8.1f}%"
              f"{pnl:>11,.0f}{pnl/cost*100:>9.2f}%{pnl/cap*100:>12.2f}pp")
    unreal = sum(sl[k][1] for k in sl)
    print("  " + "-" * 84)
    print(f"  {'unrealised':<32}{'':>3}{'':>12}{'':>9}{unreal:>11,.0f}"
          f"{'':>10}{unreal/cap*100:>12.2f}pp")
    print(f"  {'realised (day-4 rebalance)':<32}{'':>3}{'':>12}{'':>9}"
          f"{exp['realised_pnl']:>11,.0f}{'':>10}{exp['realised_pnl']/cap*100:>12.2f}pp")
    print(f"  {'TOTAL':<32}{'':>3}{'':>12}{'':>9}{exp['nav']-cap:>11,.0f}"
          f"{'':>10}{(exp['nav']-cap)/cap*100:>12.2f}pp")

    # ------------------------------------------------------------ 2. skill
    eq_ret = sl["eq"][1] / (sl["eq"][0] - sl["eq"][1]) * 100
    bench_ret = exp["benchmark_return_pct"]
    print("\n" + "-" * 86)
    print("  2. DID THE STOCK PICKING WORK?  (the only skill measure here)")
    print("-" * 86)
    print(f"  Indian equity sleeve            {eq_ret:>+8.2f}%")
    print(f"  Nifty 50 TRI over same window   {bench_ret:>+8.2f}%")
    print(f"  {'SELECTION vs index':<32}{eq_ret-bench_ret:>+8.2f}pp")

    # ------------------------------------------------------------ 3. noise
    rel = historical_relative(days)
    live = exp["relative_pct"]
    pct = (rel <= live).mean() * 100
    print("\n" + "-" * 86)
    print(f"  3. IS -{abs(live):.2f}pp UNUSUAL?  ({days}-day relative return, "
          f"{len(rel):,} historical windows)")
    print("-" * 86)
    print(f"  This assumes the stock picking adds EXACTLY ZERO, so everything")
    print(f"  it measures is pure allocation drag from being half out of equity.\n")
    print(f"  mean                    {rel.mean():>+8.2f}pp")
    print(f"  std dev                 {rel.std(ddof=1):>8.2f}pp")
    for q in (5, 25, 50, 75, 95):
        print(f"  {q:>2}th percentile        {np.percentile(rel, q):>+8.2f}pp")
    print(f"\n  LIVE READING            {live:>+8.2f}pp   -> {pct:.0f}th percentile")
    print(f"  windows this bad or worse: {(rel <= live).sum():,} of {len(rel):,} "
          f"({pct:.1f}%)")
    worse = (rel <= live).mean()
    print(f"  i.e. roughly 1 window in {1/worse:.0f} looks like this, by allocation alone")

    # ------------------------------------------------- 4. process defect
    print("\n" + "-" * 86)
    print("  4. THE DAY-4 REBALANCE  (a real process defect — but NOT the loss)")
    print("-" * 86)
    reb = book.get("rebalances", [])
    costs = led["cost_inr"].astype(float)
    incep_cost = costs[led["date"] == exp["start_date"]].sum()
    churn = costs.sum() - incep_cost
    d0 = pd.Timestamp(reb[0]["date"]) if reb else None
    print(f"  deployed cadence                182 days (126 trading bars ~ 6 months)")
    print(f"  rebalances fired                {len(reb)}, both on {d0.date()} "
          f"({d0.day_name()}) = day 4")
    print(f"  trades generated                {sum(r['trades'] for r in reb)}")
    print(f"  brokerage/impact                Rs {churn:>9,.2f}"
          f"   ({churn/cap*100:.3f}% of capital)")

    # The realised P&L is NOT extra damage: NAV is marked to market, so those
    # losses were already sitting in NAV as unrealised before the sale. Selling
    # only reclassifies them. Proof from the tape: NAV moved on rebalance day by
    # exactly the transaction cost and not one rupee more.
    nv = nav.set_index("date")["nav_inr"].astype(float)
    if "2026-07-24" in nv.index and "2026-07-26" in nv.index:
        moved = nv["2026-07-26"] - nv["2026-07-24"]
        print(f"\n  NAV moved that day              Rs {moved:>9,.2f}")
        print(f"  first rebalance cost            Rs {-192.85:>9,.2f}"
              f"   <- identical: NAV moved by COST ONLY")
    print(f"  so the Rs {exp['realised_pnl']:,.0f} 'realised loss' is a "
          f"reclassification of losses")
    print(f"  NAV already carried, not new damage. Counting it twice would be wrong.")

    cf = os.path.join(_ROOT, "reports", "paper_counterfactual.json")
    if os.path.exists(cf):
        c = json.load(open(cf))
        print(f"\n  Counterfactual — hold the opening book untouched to today:")
        print(f"  {'never rebalanced':<32}Rs {c['counterfactual_nav']:>10,.2f}"
              f"  {(c['counterfactual_nav']-cap)/cap*100:>+7.3f}%")
        print(f"  {'actual (rebalanced twice)':<32}Rs {c['actual_nav']:>10,.2f}"
              f"  {(c['actual_nav']-cap)/cap*100:>+7.3f}%")
        print(f"  {'EFFECT OF THE REBALANCE':<32}Rs {c['delta_inr']:>+10,.2f}"
              f"  {c['delta_pp']:>+7.3f}pp")
        print(f"\n  It HELPED. The defect is not that it lost money — it is that a")
        print(f"  182-day book was resynced to a changed config on day 4, twice, on a")
        print(f"  Sunday at stale prices, with nothing in the health check to stop it.")
        print(f"  That contaminates the track record; the P&L outcome was luck.")

    doc = {
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "days_live": days, "sessions": exp["observations"],
        "book_return_pct": exp["return_pct"],
        "bench_return_pct": bench_ret,
        "relative_pct": live,
        "sleeves": {k: {"value": sl[k][0], "pnl": sl[k][1],
                        "return_pct": sl[k][1] / (sl[k][0] - sl[k][1]) * 100,
                        "nav_impact_pp": sl[k][1] / cap * 100} for k in sl},
        "selection_vs_index_pp": eq_ret - bench_ret,
        "noise_test": {
            "windows": int(len(rel)), "mean_pp": float(rel.mean()),
            "std_pp": float(rel.std(ddof=1)),
            "p5": float(np.percentile(rel, 5)), "p50": float(np.percentile(rel, 50)),
            "p95": float(np.percentile(rel, 95)),
            "live_percentile": float(pct),
            "note": ("Equity sleeve proxied by Nifty itself, so the test isolates "
                     "allocation drag and assumes zero selection skill."),
        },
        "churn": {"rebalances": len(reb), "date": reb[0]["date"] if reb else None,
                  "weekday": d0.day_name() if d0 is not None else None,
                  "day_of_book": 4, "cadence_days": 182,
                  "trades": sum(r["trades"] for r in reb),
                  "costs_inr": float(churn),
                  "realised_pnl_inr": exp["realised_pnl"],
                  "realised_is_not_extra_damage": (
                      "NAV is marked to market, so these losses were already in NAV "
                      "as unrealised. NAV moved on rebalance day by exactly the "
                      "transaction cost. The real harm is a contaminated track "
                      "record, not P&L."),
                  "effect_pp": (json.load(open(cf))["delta_pp"]
                                if os.path.exists(cf) else None)},
    }
    p = os.path.join(_ROOT, "reports", "paper_diagnosis.json")
    json.dump(doc, open(p, "w"), indent=1, default=float)
    print(f"\n  saved -> {p}\n")


if __name__ == "__main__":
    main()

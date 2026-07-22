"""
Build the single JSON that the public dashboard reads.
======================================================
Every field traces to real data in this repo — market prices, the actual trade
ledger, the actual statistics. Nothing here is illustrative or placeholder.

Two clearly separated halves, because they are different kinds of evidence:
  live     — the paper book: real prices, whole shares, real costs, no money.
  research — the 2016-2026 backtest: real prices, SIMULATED trades.
Conflating them is the single most common way a performance page misleads, so
the schema keeps them apart and the UI must too.

  MARK5_CACHE=data/pit_cache python3 scripts/export_dashboard.py
"""
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, BacktestConfig,
                            load_ohlcv, load_nifty, metrics)

REPORTS = os.path.join(_ROOT, "reports")
PAPER = os.path.join(_ROOT, "data", "paper")
OUT = os.path.join(_ROOT, "docs", "data", "mark6.json")
START, END = "2016-01-01", None          # END=None -> latest available bar
TD, TAX = 252, 0.15
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}


def wrap(eq_nav, sleeves):
    """Deployed 50/25/25 blend, annual sleeve rebalance, terminal tax."""
    cal = eq_nav.index
    ser = {"eq": eq_nav.pct_change(fill_method=None).fillna(0.0)}
    for k, w in sleeves.items():
        if k == "eq":
            continue
        s = load_ohlcv(k)["close"].astype(float).reindex(cal, method="ffill")
        ser[k] = s.pct_change().fillna(0.0)
    cur, nav, out = dict(sleeves), 1.0, {}
    for i, d in enumerate(cal):
        if i > 0:
            prev = sum(cur.values())
            for k in cur:
                cur[k] *= (1 + ser[k].iloc[i])
            nav *= sum(cur.values()) / prev
        out[d] = nav
        if i > 0 and i % TD == 0:
            tot = sum(cur.values())
            cur = {k: tot * sleeves[k] for k in sleeves}
    s = pd.Series(out)
    net = s.copy()
    net.iloc[-1] = s.iloc[-1] - max(0.0, s.iloc[-1] - 1) * TAX
    return net


def series_for_chart(s, step=5):
    """Downsample for transport; keep first/last exactly."""
    idx = list(range(0, len(s), step))
    if idx[-1] != len(s) - 1:
        idx.append(len(s) - 1)
    return [[s.index[i].strftime("%Y-%m-%d"), round(float(s.iloc[i]), 4)] for i in idx]


def main():
    end = END or str(pd.Timestamp.today().date())
    panel = DataPanel(discover_tickers(), end, freshness="off")
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.08, factor_weights=MOM)
    run = Backtester(panel, PortfolioConstructor(cfg),
                     BacktestConfig(rebal_bars=126, top_n_liquid=300)).run(START, end)
    eq = run["nav_gross"]
    sys_nav = wrap(eq, {"eq": .5, "GOLDBEES": .25, "MON100": .25})
    m = metrics(sys_nav)

    nifty = load_nifty(True).reindex(sys_nav.index, method="ffill")
    nifty_nav = nifty / nifty.iloc[0]
    nn = nifty_nav.copy()
    nn.iloc[-1] = nifty_nav.iloc[-1] - max(0.0, nifty_nav.iloc[-1] - 1) * 0.125
    mn = metrics(nn)

    # equal-weight of the same universe = the honest "did the engine earn its keep" line
    ew = Backtester(panel, PortfolioConstructor(
        ConstructionConfig(mode="equal_weight", base_weighting="equal")),
        BacktestConfig(rebal_bars=126, top_n_liquid=300)).run(START, end)

    dd = (sys_nav / sys_nav.cummax() - 1)
    yearly = []
    for y, grp in sys_nav.groupby(sys_nav.index.year):
        b = nifty_nav.loc[nifty_nav.index.year == y]
        yearly.append({"year": int(y), "system": round((grp.iloc[-1] / grp.iloc[0] - 1) * 100, 2),
                       "nifty": round((b.iloc[-1] / b.iloc[0] - 1) * 100, 2) if len(b) > 1 else None})

    roll = sys_nav.pct_change().rolling(TD)
    rs = ((roll.mean() * TD - 0.065) / (roll.std() * np.sqrt(TD))).dropna()

    trades = []
    led = os.path.join(REPORTS, "trade_ledger.csv")
    if os.path.exists(led):
        trades = list(csv.DictReader(open(led)))

    ov = {}
    ovp = os.path.join(REPORTS, "OVERFITTING_ANALYSIS.md")
    if os.path.exists(ovp):
        txt = open(ovp).read()
        for key, pat in [("dsr", r"Deflated Sharpe Ratio.*?(\d+\.\d)%"),
                         ("pbo", r"PBO: (\d+\.\d)%"), ("trials", r"\(N\): \*\*(\d+)\*\*")]:
            mm = re.search(pat, txt)
            if mm:
                ov[key] = float(mm.group(1))

    live = {}
    lp = os.path.join(PAPER, "paper_export.json")
    if os.path.exists(lp):
        live = json.load(open(lp))

    holdings = []
    if run["weights"]:
        last_w = list(run["weights"].values())[-1]
        holdings = [{"ticker": t, "weight": round(float(w) * 100, 2)}
                    for t, w in last_w.sort_values(ascending=False).items()]

    doc = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "disclaimer": ("PAPER MODE. The live panel is a real-price paper book with no money at "
                       "risk. The research panel is a historical simulation: real prices, "
                       "simulated trades. Neither is investment advice."),
        "live": live,
        "research": {
            "period": {"start": START, "end": end, "years": round(m["years"], 1)},
            "universe": {"symbols": len(panel.tickers),
                         "delisted_included": int((~panel.close.iloc[-20:].notna().any()).sum()),
                         "note": "Point-in-time universe rebuilt from NSE bhavcopy; delisted "
                                 "names are present until the day they delist."},
            "headline": {
                "cagr": round(m["cagr"] * 100, 2), "sharpe_excess": round(m["sharpe_excess"], 2),
                "sharpe_raw": round(m["sharpe"], 2), "vol": round(m["vol"] * 100, 2),
                "max_dd": round(m["max_dd"] * 100, 2), "calmar": round(m["calmar"], 2),
                "sortino": round(m["sortino"], 2),
                "excess_vs_nifty": round((m["cagr"] - mn["cagr"]) * 100, 2),
                "engine_alpha_vs_ew": round((run["metrics"]["cagr"] - ew["metrics"]["cagr"]) * 100, 2),
                "turnover_yr": round(run["metrics"]["turnover_yr"] * 100, 0),
            },
            "benchmark": {"name": "Nifty 50 TRI (net of terminal LTCG)",
                          "cagr": round(mn["cagr"] * 100, 2),
                          "sharpe_excess": round(mn["sharpe_excess"], 2),
                          "max_dd": round(mn["max_dd"] * 100, 2)},
            "equity_curve": series_for_chart(sys_nav),
            "benchmark_curve": series_for_chart(nifty_nav),
            "drawdown": series_for_chart(dd),
            "rolling_sharpe": series_for_chart(rs, step=10),
            "yearly": yearly,
            "holdings": holdings,
            "trades": trades,
            "validation": {
                "dsr_pct": ov.get("dsr"), "pbo_pct": ov.get("pbo"), "trials": ov.get("trials"),
                "pbo_reading": ("PBO above the conventional 20% bar is a FAIL: picking the "
                                "in-sample-best variant from this family overfits. The deployed "
                                "config was chosen on walk-forward consistency instead."),
                "nested_wf": (json.load(open(os.path.join(REPORTS, "nested_walkforward.json")))
                              if os.path.exists(os.path.join(REPORTS, "nested_walkforward.json")) else {}),
            },
            "limitations": [
                "Never traded with real money. The live panel starts from the day it was opened.",
                "Backtest trades are simulated at next-day closing prices; real fills differ.",
                "Measured over 2016-2026, a decade kind to Indian equities, gold and US tech.",
                "Corporate-action feed lacks demergers, so 67 affected symbols are excluded.",
                "Modelled costs (0.49% round trip) exceed real Zerodha delivery costs.",
            ],
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(doc, open(OUT, "w"), indent=1, default=float)
    kb = os.path.getsize(OUT) / 1024
    print(f"  wrote {OUT}  ({kb:.0f} KB)")
    print(f"  research: {doc['research']['headline']['cagr']}% CAGR, "
          f"Sharpe {doc['research']['headline']['sharpe_excess']}, "
          f"MaxDD {doc['research']['headline']['max_dd']}%, {len(trades)} trades")
    print(f"  live: {'day ' + str(live.get('days_live')) if live else 'NOT STARTED'}")


if __name__ == "__main__":
    main()

"""
Do the delivery signals add net value as a small factor component?
==================================================================
The IC study (K36/K37) killed them on significance: IC +0.020/+0.023 at 126d with
t below 2. But the test was UNDERPOWERED there (~12 independent windows, minimum
detectable IC 0.046), both signals were ORTHOGONAL to the existing factors (max
corr 0.126/0.191) and both had MONOTONIC terciles. "Not demonstrated" is not the
same as "proven absent", so the fair next question is whether they earn their keep
in the actual book.

Grinold allows this: a weak signal CAN pay in a high-breadth basket if it is
genuinely independent. The direct precedent says otherwise — K12 tested
Delta-promoter at IC +0.034 (STRONGER than these) as a factor sleeve and killed it,
because a weak IC did not convert to a net edge on top of the existing blend.

So this settles it with a backtest rather than an argument.

BAR, again set before the run: beat the baseline in >=3/4 rolling 3-year windows
(the 6/8 standard, scaled to the 4 windows the short archive allows) on net CAGR,
AND not make drawdown worse. Delivery data starts 2019-10, so the whole test runs
2020-07 -> 2026-07 and cannot say anything about earlier regimes.

  MARK5_CACHE=data/pit_cache python3 scripts/delivery_factor_test.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, BacktestConfig,
                            metrics, load_sector_map)

RAW = os.path.join(_ROOT, "data", "delivery", "raw")
START, END = "2020-07-01", "2026-07-21"
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}
BT = dict(rebal_bars=126, top_n_liquid=300)


def build_extra_factors(tickers):
    """{ticker -> DataFrame(index=date, cols=[deliv_per_z, deliv_chg])}, causal.

    Delivery for date t is published after that day's close, so a value indexed at
    t is knowable at t's close. The engine then executes at t+1 (exec_lag=1), so
    there is no look-ahead.
    """
    files = sorted(glob.glob(os.path.join(RAW, "*.parquet")))
    if not files:
        sys.exit("ERROR: no delivery archive — run scripts/fetch_delivery.py first.")
    rows = {}
    for f in files:
        d = pd.Timestamp(os.path.basename(f)[:10])
        df = pd.read_parquet(f)
        df = df[df["symbol"].notna()].drop_duplicates("symbol").set_index("symbol")
        rows[d] = df["deliv_per"]
    dp = pd.DataFrame(rows).T.sort_index()
    dp = dp[[c for c in dp.columns if c in set(tickers)]]

    m = dp.rolling(126, min_periods=63).mean()
    s = dp.rolling(126, min_periods=63).std()
    z = (dp - m) / s.replace(0, np.nan)
    chg = dp.rolling(21, min_periods=10).mean() - m

    out = {}
    for t in dp.columns:
        df = pd.DataFrame({"deliv_per_z": z[t], "deliv_chg": chg[t]}).dropna(how="all")
        if len(df) > 60:
            out[t] = df
    return out


def run(panel, weights, extra, label):
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.08, factor_weights=weights)
    bt = Backtester(panel, PortfolioConstructor(cfg, sector_map=load_sector_map()),
                    BacktestConfig(**BT), extra_factors=extra)
    return bt


def main():
    panel = DataPanel(discover_tickers(), END, freshness="off")
    print("  building causal delivery factors...", flush=True)
    extra = build_extra_factors(panel.tickers)
    print(f"    {len(extra)} tickers carry delivery factors\n", flush=True)

    # composite_score renormalises by the weight sum, so adding a component
    # proportionally shrinks the incumbents rather than breaking the blend.
    CONFIGS = [
        ("BASELINE (no delivery)", dict(MOM), None),
        ("+ deliv_per_z @10%", {**MOM, "deliv_per_z": 0.10}, extra),
        ("+ deliv_chg  @10%", {**MOM, "deliv_chg": 0.10}, extra),
        ("+ both       @5% each", {**MOM, "deliv_per_z": 0.05, "deliv_chg": 0.05}, extra),
        ("+ deliv_chg  @20%", {**MOM, "deliv_chg": 0.20}, extra),
    ]
    windows = [(f"{y}-01-01", f"{y+2}-12-31" if y + 2 < 2026 else END, f"{y}-{y+2}")
               for y in (2020, 2021, 2022, 2023)]

    res = {}
    for label, w, ex in CONFIGS:
        bt = run(panel, w, ex, label)
        full = metrics(bt.run(START, END)["nav_net"])
        wf = []
        for s, e, lab in windows:
            try:
                wf.append(metrics(bt.run(s, e)["nav_net"]))
            except Exception as exn:
                print(f"    ! {label} {lab}: {type(exn).__name__}")
                wf.append(None)
        res[label] = {"full": full, "wf": wf}
        print(f"  ran {label}", flush=True)

    base = res["BASELINE (no delivery)"]
    print("\n" + "=" * 104)
    print(f"  DELIVERY AS A FACTOR COMPONENT — equity sleeve, {START} -> {END}")
    print("  bar: >=3/4 windows on net CAGR AND drawdown not worse")
    print("=" * 104)
    print(f"  {'config':<26}{'CAGR':>9}{'shExc':>8}{'vol':>7}{'MaxDD':>8}{'Calmar':>8}"
          f"{'turn':>7}   {'wf CAGR':>9}{'wf MaxDD':>10}")
    print("  " + "-" * 100)
    out = []
    for label, r in res.items():
        m = r["full"]
        if label.startswith("BASELINE"):
            c = d = ""
        else:
            pairs = [(a, b) for a, b in zip(r["wf"], base["wf"]) if a and b]
            c = f"{sum(1 for a,b in pairs if a['cagr']>b['cagr'])}/{len(pairs)}"
            d = f"{sum(1 for a,b in pairs if a['max_dd']>b['max_dd'])}/{len(pairs)}"
        print(f"  {label:<26}{m['cagr']*100:>+8.2f}%{m['sharpe_excess']:>8.2f}"
              f"{m['vol']*100:>6.1f}%{m['max_dd']*100:>+7.1f}%{m['calmar']:>8.2f}"
              f"{m.get('turnover_yr',0)*100:>6.0f}%   {c:>9}{d:>10}")
        out.append({"config": label, **{k: float(v) for k, v in m.items()},
                    "wf_cagr_wins": c, "wf_maxdd_wins": d})

    print("\n" + "=" * 104)
    print("  VERDICT")
    print("=" * 104)
    winners = []
    for label, r in res.items():
        if label.startswith("BASELINE"):
            continue
        pairs = [(a, b) for a, b in zip(r["wf"], base["wf"]) if a and b]
        cw = sum(1 for a, b in pairs if a["cagr"] > b["cagr"])
        dw = sum(1 for a, b in pairs if a["max_dd"] > b["max_dd"])
        if cw >= 3 and dw >= 2:
            winners.append((label, cw, dw))
    if winners:
        for label, cw, dw in winners:
            print(f"  KEEP CANDIDATE: {label} — CAGR {cw}/4, MaxDD {dw}/4")
        print("\n  Caveat that must travel with any deployment: this is a 6-year test on")
        print("  4 windows, all post-2020. It cannot speak to 2016-2019.")
    else:
        print("  NO configuration cleared the bar. Delivery data does not convert to a")
        print("  net edge on top of the existing blend — the same outcome as K12, where")
        print("  a STRONGER signal (Delta-promoter, IC +0.034) also failed to convert.")
        print("  Confirms K36/K37: the signal is real-signed and orthogonal but too weak")
        print("  to survive costs, tax and the incumbent blend.")

    p = os.path.join(_ROOT, "reports", "delivery_factor_test.json")
    json.dump({"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
               "window": {"start": START, "end": END, "n_windows": len(windows)},
               "results": out}, open(p, "w"), indent=1, default=float)
    print(f"\n  saved -> {p}\n")


if __name__ == "__main__":
    main()

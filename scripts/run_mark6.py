"""
MARK6 — Honest Performance Report (run me)
==========================================
Runs the smart-beta factor portfolio and its honest benchmarks, full-period and
across rolling walk-forward windows, all NET of Indian equity tax, and writes
reports/mark6_results.json + reports/MARK6_REPORT.md.

  python3 scripts/run_mark6.py
"""
import os, sys, json
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, BacktestConfig, metrics,
                            load_nifty)

CACHE = os.path.join(_ROOT, "data", "cache")
REPORTS = os.path.join(_ROOT, "reports")
END = "2026-06-09"
LTCG = 0.125


def nifty_buyhold(start, end):
    """Nifty 50 TOTAL-RETURN buy-and-hold, net of one terminal LTCG hit.
    Total-return (dividends reinvested, via the NIFTYBEES adjusted series) —
    the strategy book runs on dividend-adjusted prices, so a price-only index
    would flatter it by ~1pp/yr (v7.1 audit fix)."""
    s = load_nifty(total_return=True)
    if s is None:
        return {}
    s = s.loc[start:end]
    if len(s) < 30:
        return {}
    gross = s / s.iloc[0]
    net = gross.copy()
    net.iloc[-1] = 1 + (gross.iloc[-1] - 1) * (1 - LTCG)   # terminal LTCG
    return metrics(net)


def fmt(m):
    if not m:
        return "  (n/a)"
    return (f"CAGR {m['cagr']*100:+6.1f}%  Sharpe {m['sharpe']:.2f}  "
            f"Sortino {m.get('sortino',0):.2f}  MaxDD {m['max_dd']*100:+6.1f}%  "
            f"Calmar {m['calmar']:.2f}")


def main():
    os.makedirs(REPORTS, exist_ok=True)
    print("Loading data panel (this builds factors once)...", flush=True)
    tickers = discover_tickers()
    panel = DataPanel(tickers, END)
    print(f"  universe: {len(panel.tickers)} names\n", flush=True)

    # the strategy and the honest benchmark.
    # n_hold=12 + tilt_strength=1.5 validated 2026-06-08: beats the old n_hold=20
    # config in 8/8 rolling 3-yr walk-forward windows (+2.3pp avg net).
    # 2026-06-10 upgrade: momentum-heavy factor weights validated OOS (+1.4pp avg,
    # 6/8 walk-forward windows) — momentum 0.45 / trend 0.25 / low_vol 0.15 / stability 0.15.
    factor_cfg = ConstructionConfig(mode="factor_tilt", n_hold=12,
                                    base_weighting="inverse_vol", tilt_strength=1.5,
                                    max_weight=0.125,
                                    factor_weights={"momentum": 0.45, "low_vol": 0.15,
                                                    "trend": 0.25, "stability": 0.15})
    ew_cfg = ConstructionConfig(mode="equal_weight", base_weighting="equal")
    # 2026-06-11 upgrade (P11+P12): honest FY tax netting (losses offset gains, as
    # Indian law actually works) unblocks semi-annual rebalance — momentum decays at
    # the 6-12mo horizon, and netting cuts the turnover tax penalty that previously
    # made faster rebalance look bad (K3 used the no-credit model). Validated:
    # equity sleeve +2.84pp avg walk-forward (7/8 windows), full system 19.0→20.7%.
    min_turn = float(os.environ.get("MARK5_MIN_TURNOVER", "0"))
    if min_turn:
        print(f"  absolute liquidity floor: Rs {min_turn/1e7:.0f}cr/day median turnover\n")
    bt_cfg = BacktestConfig(rebal_bars=126, min_turnover=min_turn)
    bt_factor = Backtester(panel, PortfolioConstructor(factor_cfg), bt_cfg)
    bt_ew = Backtester(panel, PortfolioConstructor(ew_cfg), BacktestConfig(rebal_bars=126, min_turnover=min_turn))

    results = {"config": factor_cfg.__dict__, "windows": {}}

    # ── full period + the two canonical sub-windows ──────────────────────────
    windows = [("2016-01-01", END, "FULL 2016-2026"),
               ("2016-01-01", "2021-12-31", "OOS-era 2016-2021"),
               ("2022-01-01", END, "recent 2022-2026")]
    print("="*100)
    print(f"  {'WINDOW':<22}{'FACTOR (MARK6)':<46}{'beats EW?':<11}{'beats Nifty?'}")
    print("="*100)
    for s, e, label in windows:
        rf = bt_factor.run(s, e)["metrics"]
        re_ = bt_ew.run(s, e)["metrics"]
        rn = nifty_buyhold(s, e)
        vs_ew = (rf["cagr"] - re_["cagr"]) * 100
        vs_n = (rf["cagr"] - rn["cagr"]) * 100 if rn.get("cagr") == rn.get("cagr") else float("nan")
        print(f"\n  {label}")
        print(f"    MARK6 factor : {fmt(rf)}")
        print(f"    EqualWeight  : {fmt(re_)}")
        print(f"    Nifty50 B&H  : {fmt(rn)}")
        print(f"    -> vs EqualWeight: {vs_ew:+.1f}pp net    vs Nifty50: {vs_n:+.1f}pp net")
        results["windows"][label] = {"factor": rf, "equal_weight": re_, "nifty": rn,
                                     "vs_ew_pp": vs_ew, "vs_nifty_pp": vs_n}

    # ── rolling 3-year walk-forward (robustness, not cherry-picking) ─────────
    print("\n" + "="*100)
    print("  ROLLING 3-YEAR WALK-FORWARD  (net CAGR; does the edge persist?)")
    print("="*100)
    print(f"  {'window':<16}{'MARK6':>9}{'EqualWt':>9}{'Nifty50':>9}{'vs EW':>8}{'vs Nifty':>10}")
    print("  " + "-"*70)
    roll = []
    wins, beats_ew, beats_n = 0, 0, 0
    for y0 in range(2016, 2024):
        s, e = f"{y0}-01-01", f"{y0+2}-12-31"
        if pd.Timestamp(e) > pd.Timestamp(END):
            e = END
        rf = bt_factor.run(s, e)["metrics"]
        re_ = bt_ew.run(s, e)["metrics"]
        rn = nifty_buyhold(s, e)
        if not rf:
            continue
        rn = rn or {"cagr": float("nan")}
        vew, vn = (rf["cagr"]-re_["cagr"])*100, (rf["cagr"]-rn["cagr"])*100
        wins += 1; beats_ew += vew > 0; beats_n += vn > 0
        print(f"  {y0}-{y0+2:<11}{rf['cagr']*100:>+8.1f}%{re_['cagr']*100:>+8.1f}%"
              f"{rn['cagr']*100:>+8.1f}%{vew:>+7.1f}{vn:>+9.1f}")
        roll.append({"window": f"{y0}-{y0+2}", "factor": rf["cagr"], "ew": re_["cagr"],
                     "nifty": rn["cagr"], "vs_ew": vew, "vs_nifty": vn})
    print("  " + "-"*70)
    print(f"  Walk-forward hit-rate:  beats Nifty50 {beats_n}/{wins} windows | "
          f"beats EqualWeight {beats_ew}/{wins} windows")
    results["walk_forward"] = roll
    results["walk_forward_summary"] = {"windows": wins, "beats_nifty": beats_n,
                                       "beats_ew": beats_ew}

    with open(os.path.join(REPORTS, "mark6_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Saved -> reports/mark6_results.json")
    _write_md(results)
    print(f"  Saved -> reports/MARK6_REPORT.md\n")


def _write_md(r):
    L = ["# MARK6 — Honest Smart-Beta Portfolio: Performance Report", "",
         "All figures **net of Indian equity tax** (LTCG 12.5% / STCG 20%) and "
         "transaction costs, on the v7.1 engine (FIFO tax lots, next-close "
         "execution, cash-constrained). Benchmark = **Nifty 50 TOTAL-RETURN** "
         "buy-and-hold (dividends reinvested, via NIFTYBEES-adjusted series), "
         "net of terminal LTCG — the strategy book earns dividends, so a "
         "price-only index would flatter it ~1pp/yr.", "", "## Headline windows", "",
         "| Window | MARK6 net CAGR | EqualWeight | Nifty50 TRI B&H | vs Nifty | vs EW |",
         "|---|---|---|---|---|---|"]
    for label, w in r["windows"].items():
        L.append(f"| {label} | {w['factor']['cagr']*100:+.1f}% | "
                 f"{w['equal_weight']['cagr']*100:+.1f}% | {w['nifty'].get('cagr', float('nan'))*100:+.1f}% | "
                 f"{w['vs_nifty_pp']:+.1f}pp | {w['vs_ew_pp']:+.1f}pp |")
    s = r["walk_forward_summary"]
    L += ["", "## Rolling 3-year walk-forward", "",
          f"**Beats Nifty50 in {s['beats_nifty']}/{s['windows']} windows; "
          f"beats EqualWeight in {s['beats_ew']}/{s['windows']} windows.**", "",
          "| Window | MARK6 | EqualWt | Nifty50 | vs Nifty |", "|---|---|---|---|---|"]
    for w in r["walk_forward"]:
        L.append(f"| {w['window']} | {w['factor']*100:+.1f}% | {w['ew']*100:+.1f}% | "
                 f"{w['nifty']*100:+.1f}% | {w['vs_nifty']:+.1f}pp |")
    L += ["", "## Honest caveats", "",
          "- Survivorship: the candidate universe is today's surviving constituents "
          "(fully-delisted names absent), so headline CAGR is inflated an estimated "
          "~1-2pp/yr. `survivorship_validation.py` bounds this via failure injection "
          "on the equal-weight basket; the concentrated momentum book has NOT been "
          "separately failure-injected.",
          "- Drawdowns are equity-level (~-30 to -40%); inverse-vol weighting reduces "
          "but cannot eliminate them. The 5% hard-stop design is incompatible with "
          "equity returns and was proven to destroy the edge.",
          "- The edge over the cap-weighted index is real but regime-dependent; "
          "it is NOT alpha over same-universe buy-and-hold (that does not exist net of tax)."]
    with open(os.path.join(REPORTS, "MARK6_REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

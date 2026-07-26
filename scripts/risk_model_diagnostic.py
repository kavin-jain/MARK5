"""
Diagnostic: how many INDEPENDENT bets is the equity book actually making?
========================================================================
The book weights by inverse volatility. That is a DIAGONAL risk model — it uses
each name's own volatility and assumes nothing about how they move together.
If the 20 selected names are in fact one big correlated bet (momentum tends to
concentrate in whatever theme is running), then:

  - the portfolio's true risk is far higher than the sum of its parts implies,
  - and Grinold's IR = IC x sqrt(breadth) is being computed against a breadth
    the book does not actually have.

This script measures that, at every real rebalance date, on the honest PIT
universe. It changes nothing and proposes nothing — it only establishes whether
the correlation gap is real and how big it is.

Reported per rebalance:
  avg_corr   mean pairwise correlation of the selected names (252d daily)
  n_eff      effective number of independent bets, from the eigenvalue spread
             of the correlation matrix: exp(entropy of normalised eigenvalues).
             20 names at zero correlation -> 20. One perfectly-correlated blob -> 1.
  div_ratio  diversification ratio = (weighted average name vol) / (portfolio vol).
             1.0 means diversification bought nothing.
  top_sector largest sector share, using config/sector_map.json

  MARK5_CACHE=data/pit_cache python3 scripts/risk_model_diagnostic.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, BacktestConfig)

START, END = "2016-01-01", "2026-07-21"
LOOKBACK = 252
DEPLOYED = dict(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                tilt_strength=1.5, max_weight=0.08,
                factor_weights={"momentum": .45, "low_vol": .15,
                                "trend": .25, "stability": .15})


def effective_bets(corr: np.ndarray) -> float:
    """exp(Shannon entropy of normalised eigenvalues) — the standard measure of
    how many independent directions of risk a correlation matrix really has."""
    ev = np.linalg.eigvalsh(corr)
    ev = ev[ev > 1e-10]
    p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    sm_path = os.path.join(_ROOT, "config", "sector_map.json")
    sectors = json.load(open(sm_path))["sectors"] if os.path.exists(sm_path) else {}

    panel = DataPanel(discover_tickers(), END, freshness="off")
    cfg = ConstructionConfig(**DEPLOYED)
    bt = Backtester(panel, PortfolioConstructor(cfg),
                    BacktestConfig(rebal_bars=126, top_n_liquid=300))
    run = bt.run(START, END)

    rets = panel.close.pct_change(fill_method=None)
    rows = []
    for d, w in run["weights"].items():
        w = w[w > 0]
        names = [t for t in w.index if t in rets.columns]
        if len(names) < 3:
            continue
        # strictly causal: only returns up to and including the signal date
        hist = rets.loc[:d, names].tail(LOOKBACK).dropna(axis=1, thresh=LOOKBACK // 2)
        names = list(hist.columns)
        if len(names) < 3:
            continue
        hist = hist.fillna(0.0)
        ww = w[names].values
        ww = ww / ww.sum()
        C = np.corrcoef(hist.values.T)
        S = np.cov(hist.values.T) * 252
        iu = np.triu_indices_from(C, k=1)
        port_vol = float(np.sqrt(ww @ S @ ww))
        name_vol = np.sqrt(np.diag(S))
        wavg_vol = float(ww @ name_vol)
        sec = {}
        for t, x in zip(names, ww):
            sec[sectors.get(t, f"~{t}")] = sec.get(sectors.get(t, f"~{t}"), 0.0) + x
        rows.append({
            "date": d.date(), "n": len(names),
            "avg_corr": float(C[iu].mean()),
            "n_eff": effective_bets(C),
            "port_vol": port_vol, "wavg_vol": wavg_vol,
            "div_ratio": wavg_vol / port_vol if port_vol else np.nan,
            "top_sector": max(sec.values()),
            "unmapped": sum(1 for t in names if t not in sectors),
        })

    df = pd.DataFrame(rows)
    print("\nEQUITY BOOK — REALISED DIVERSIFICATION AT EACH REBALANCE")
    print("=" * 88)
    print(f"  {'date':<12}{'n':>3}{'avg_corr':>10}{'n_eff':>8}{'port_vol':>10}"
          f"{'wavg_vol':>10}{'div_ratio':>11}{'top_sect':>10}{'unmapped':>10}")
    print("  " + "-" * 84)
    for r in rows:
        print(f"  {str(r['date']):<12}{r['n']:>3}{r['avg_corr']:>10.3f}{r['n_eff']:>8.1f}"
              f"{r['port_vol']*100:>9.1f}%{r['wavg_vol']*100:>9.1f}%{r['div_ratio']:>11.2f}"
              f"{r['top_sector']*100:>9.0f}%{r['unmapped']:>10}")
    print("  " + "-" * 84)
    print(f"  {'MEAN':<12}{df['n'].mean():>3.0f}{df['avg_corr'].mean():>10.3f}"
          f"{df['n_eff'].mean():>8.1f}{df['port_vol'].mean()*100:>9.1f}%"
          f"{df['wavg_vol'].mean()*100:>9.1f}%{df['div_ratio'].mean():>11.2f}"
          f"{df['top_sector'].mean()*100:>9.0f}%{df['unmapped'].mean():>10.1f}")

    print("\n  READING")
    print(f"  - The book holds {df['n'].mean():.0f} names but makes about "
          f"{df['n_eff'].mean():.1f} independent bets.")
    print(f"  - Mean pairwise correlation {df['avg_corr'].mean():.3f}. Diversification ratio "
          f"{df['div_ratio'].mean():.2f}\n    (1.00 would mean diversification bought nothing).")
    print(f"  - Largest sector averages {df['top_sector'].mean()*100:.0f}% of the book against a "
          f"configured 30% cap\n    that is never enforced; {df['unmapped'].mean():.1f} names per "
          f"rebalance are unmapped and would escape it anyway.")
    lost = 1 - df["n_eff"].mean() / df["n"].mean()
    print(f"  - Grinold: IR = IC x sqrt(breadth). Using n_eff instead of n cuts sqrt(breadth) "
          f"by {(1-np.sqrt(df['n_eff'].mean()/df['n'].mean()))*100:.0f}%,\n    i.e. {lost*100:.0f}% "
          f"of the nominal breadth is an illusion.")
    out = os.path.join(_ROOT, "reports", "risk_model_diagnostic.json")
    json.dump({"per_rebalance": rows, "mean": df.mean(numeric_only=True).to_dict()},
              open(out, "w"), indent=1, default=str)
    print(f"\n  saved -> {out}\n")


if __name__ == "__main__":
    main()

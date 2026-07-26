"""
Can this system reach Sharpe 1.1? The mathematical ceiling, not an opinion.
==========================================================================
Before hunting for tweaks, establish what is even attainable. For any set of
assets the maximum Sharpe achievable by ANY fixed combination is a closed form:

    w* proportional to  Sigma^-1 mu          (tangency portfolio)
    S*  =  sqrt( mu' Sigma^-1 mu )           (its Sharpe)

If S* < 1.1, then no reweighting of these assets reaches the target and the only
honest routes are (a) a better equity sleeve, or (b) a genuinely new return
stream. If S* > 1.1, the target is reachable and the question becomes whether a
ROBUST allocation gets there — the tangency weights themselves are in-sample
optimal and must never be deployed directly.

It also answers, exactly, what any candidate asset is worth. Adding an asset with
Sharpe S_new and correlation rho to the current portfolio moves the ceiling to

    S_combined = sqrt( S_p^2 + ( (S_new - rho*S_p) / sqrt(1-rho^2) )^2 )

so the asset improves the portfolio if and only if  S_new > rho * S_p.
That single inequality is the whole "is this diversifier worth it" question, and
it is why an asset with a MEDIOCRE standalone Sharpe can still be valuable if its
correlation is low enough — and why a high-Sharpe asset that moves with the book
is worth nothing.

All returns are excess of the 6.5% Indian risk-free rate, daily, annualised.

  MARK5_CACHE=data/pit_cache python3 scripts/sharpe_ceiling.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, BacktestConfig,
                            load_ohlcv, load_nifty, metrics, load_sector_map)

START, END = "2016-01-01", "2026-07-21"
TD, RF = 252, 0.065
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}
DEPLOYED = {"eq": .50, "GOLDBEES": .25, "MON100": .25}
TARGET = 1.1

CANDIDATES = ["LTGILTBEES", "GILT5YBEES", "SILVERBEES", "MAFANG", "LIQUIDBEES"]


def ann_stats(r: pd.DataFrame):
    """Annualised excess-return vector and covariance matrix."""
    mu = r.mean().values * TD - RF
    S = r.cov().values * TD
    return mu, S


def tangency(mu, S):
    """w* ∝ Σ⁻¹μ (long/short, unconstrained) and its Sharpe √(μ'Σ⁻¹μ)."""
    inv = np.linalg.pinv(S)
    w = inv @ mu
    s = float(np.sqrt(max(0.0, mu @ inv @ mu)))
    return (w / w.sum() if abs(w.sum()) > 1e-12 else w), s


def long_only_max_sharpe(mu, S, n_iter=200000, seed=7):
    """Max Sharpe subject to w>=0, sum(w)=1 — the constraint we actually face
    (no shorting at retail). Projected-gradient ascent from several starts;
    reported alongside the unconstrained figure because the unconstrained
    tangency is usually unreachable in practice."""
    n = len(mu)
    rng = np.random.default_rng(seed)
    best_w, best_s = None, -np.inf
    for _ in range(12):
        w = rng.dirichlet(np.ones(n))
        step = 0.02
        for i in range(n_iter // 12):
            var = w @ S @ w
            if var <= 0:
                break
            sd = np.sqrt(var)
            g = mu / sd - (mu @ w) * (S @ w) / (sd ** 3)     # d(Sharpe)/dw
            w = w + step * g
            w = np.maximum(w, 0.0)
            t = w.sum()
            if t <= 0:
                break
            w = w / t
            step *= 0.99995
        var = w @ S @ w
        s = (mu @ w) / np.sqrt(var) if var > 0 else -np.inf
        if s > best_s:
            best_w, best_s = w.copy(), float(s)
    return best_w, best_s


def series_for(key, cal):
    if key == "eq":
        return None
    s = load_ohlcv(key)
    if s is None:
        return None
    return (s["close"].astype(float).reindex(cal, method="ffill")
            .pct_change(fill_method=None))


def main():
    panel = DataPanel(discover_tickers(), END, freshness="off")
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.08, factor_weights=MOM)
    run = Backtester(panel, PortfolioConstructor(cfg, sector_map=load_sector_map()),
                     BacktestConfig(rebal_bars=126, top_n_liquid=300)).run(START, END)
    eq = run["nav_gross"]
    cal = eq.index

    cols = {"eq": eq.pct_change(fill_method=None)}
    for k in ("GOLDBEES", "MON100"):
        cols[k] = series_for(k, cal)
    R = pd.DataFrame(cols).dropna()

    mu, S = ann_stats(R)
    names = list(R.columns)
    sharpes = {n: (R[n].mean() * TD - RF) / (R[n].std() * np.sqrt(TD)) for n in names}
    C = R.corr()

    print("\n" + "=" * 88)
    print("  SLEEVE STATISTICS   " + f"{START} -> {END}   (excess of {RF*100:.1f}% rf)")
    print("=" * 88)
    print(f"  {'sleeve':<14}{'ann.return':>12}{'ann.vol':>10}{'Sharpe':>9}")
    for n in names:
        print(f"  {n:<14}{(R[n].mean()*TD)*100:>11.2f}%{(R[n].std()*np.sqrt(TD))*100:>9.1f}%"
              f"{sharpes[n]:>9.2f}")
    print("\n  correlation matrix")
    print("   " + "".join(f"{n:>12}" for n in names))
    for n in names:
        print(f"  {n:<12}" + "".join(f"{C.loc[n,m]:>12.3f}" for m in names))

    # deployed portfolio
    wd = np.array([DEPLOYED[n] for n in names])
    s_dep = (mu @ wd) / np.sqrt(wd @ S @ wd)
    _, s_tan = tangency(mu, S)
    w_lo, s_lo = long_only_max_sharpe(mu, S)

    print("\n" + "=" * 88)
    print("  THE CEILING")
    print("=" * 88)
    print(f"  Deployed 50/25/25                       Sharpe {s_dep:.3f}")
    print(f"  Best LONG-ONLY combination of these 3   Sharpe {s_lo:.3f}"
          f"   weights " + " ".join(f"{n} {w*100:.0f}%" for n, w in zip(names, w_lo)))
    print(f"  Unconstrained tangency (allows shorts)  Sharpe {s_tan:.3f}")
    print(f"  TARGET                                  Sharpe {TARGET:.3f}")
    verdict = ("REACHABLE by reweighting alone" if s_lo >= TARGET else
               "NOT reachable by reweighting these three assets")
    print(f"\n  -> {verdict}")
    if s_lo < TARGET:
        print(f"     The best possible long-only mix of the equity book, gold and Nasdaq")
        print(f"     tops out at {s_lo:.3f}, and that is an IN-SAMPLE optimum computed with")
        print(f"     perfect hindsight — a deployable allocation would score below it.")

    # ── what would close the gap on the CURRENT portfolio ────────────────────
    vol_dep = float(np.sqrt(wd @ S @ wd))
    ret_dep = float(mu @ wd) + RF
    print("\n" + "=" * 88)
    print("  GAP DECOMPOSITION — what exactly has to change")
    print("=" * 88)
    need_ret = RF + TARGET * vol_dep
    need_vol = (ret_dep - RF) / TARGET
    print(f"  Now: return {ret_dep*100:.2f}%  vol {vol_dep*100:.2f}%  Sharpe {s_dep:.3f}")
    print(f"  (a) hold vol, raise return to {need_ret*100:.2f}%   -> need "
          f"{(need_ret-ret_dep)*100:+.2f}pp of extra return")
    print(f"  (b) hold return, cut vol to  {need_vol*100:.2f}%   -> need "
          f"{(need_vol-vol_dep)*100:+.2f}pp less volatility")
    print(f"  (c) any combination on the line between them.")

    # ── marginal value of each candidate diversifier ─────────────────────────
    print("\n" + "=" * 88)
    print("  MARGINAL VALUE OF A NEW ASSET     rule: it helps iff  S_new > rho x S_port")
    print("=" * 88)
    port = R @ wd
    s_p = (port.mean() * TD - RF) / (port.std() * np.sqrt(TD))
    print(f"  current portfolio Sharpe S_p = {s_p:.3f}\n")
    print(f"  {'asset':<14}{'history':>10}{'S_new':>8}{'rho':>8}{'hurdle':>9}"
          f"{'verdict':>10}{'new ceiling':>13}")
    print("  " + "-" * 84)
    rows = []
    for k in CANDIDATES:
        s = series_for(k, cal)
        if s is None:
            continue
        df = pd.DataFrame({"p": port, "a": s}).dropna()
        if len(df) < 500:
            continue
        yrs = (df.index[-1] - df.index[0]).days / 365.25
        s_new = (df["a"].mean() * TD - RF) / (df["a"].std() * np.sqrt(TD))
        rho = float(df["p"].corr(df["a"]))
        # recompute S_p on the OVERLAPPING window, else the comparison is unfair
        s_p_ov = (df["p"].mean() * TD - RF) / (df["p"].std() * np.sqrt(TD))
        hurdle = rho * s_p_ov
        helps = s_new > hurdle
        ceil = np.sqrt(s_p_ov ** 2 + ((s_new - rho * s_p_ov) / np.sqrt(max(1e-9, 1 - rho ** 2))) ** 2)
        print(f"  {k:<14}{yrs:>9.1f}y{s_new:>8.2f}{rho:>8.3f}{hurdle:>9.2f}"
              f"{'HELPS' if helps else 'no':>10}{ceil:>13.3f}")
        rows.append({"asset": k, "years": yrs, "sharpe": s_new, "rho": rho,
                     "hurdle": hurdle, "helps": bool(helps), "new_ceiling": ceil,
                     "s_p_overlap": s_p_ov})
    print("\n  'new ceiling' = the best Sharpe attainable if that asset is added, at its")
    print("  optimal weight, measured only over the window where it has data. Short")
    print("  histories flatter themselves — LTGILTBEES starts 2018, SILVERBEES 2022.")

    out = {"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
           "period": {"start": START, "end": END},
           "sleeves": {n: {"ann_return_pct": float(R[n].mean() * TD * 100),
                           "ann_vol_pct": float(R[n].std() * np.sqrt(TD) * 100),
                           "sharpe_excess": float(sharpes[n])} for n in names},
           "correlations": C.to_dict(),
           "deployed_sharpe": float(s_dep),
           "long_only_max_sharpe": float(s_lo),
           "long_only_max_weights": dict(zip(names, map(float, w_lo))),
           "unconstrained_tangency_sharpe": float(s_tan),
           "target": TARGET,
           "reachable_by_reweighting": bool(s_lo >= TARGET),
           "gap": {"current_return_pct": ret_dep * 100, "current_vol_pct": vol_dep * 100,
                   "need_return_pct": need_ret * 100, "need_vol_pct": need_vol * 100},
           "candidates": rows}
    p = os.path.join(_ROOT, "reports", "sharpe_ceiling.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n  saved -> {p}\n")


if __name__ == "__main__":
    main()

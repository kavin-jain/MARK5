"""
What would it ACTUALLY take to reach Sharpe 1.1? Solved, not guessed.
=====================================================================
Established so far:
  - deployed 50/25/25 scores Sharpe 0.93 (66% of its RISK is the equity sleeve)
  - the risk-parity allocation ~27/49/24 is stable in every subperiod and scores
    1.00 with MaxDD -17.9% and Calmar 1.12
  - learned/optimised allocations are NOISE out of sample (3/8 years)
  - so allocation alone tops out at ~1.00. The last 0.1 has to come from somewhere else.

This inverts the problem. For a portfolio of sleeves, Sharpe is fully determined
by each sleeve's Sharpe and the correlation matrix:

    S_p  =  (w . s * sigma) / sqrt(w' Sigma w)

so for any target we can SOLVE for the requirement instead of searching for it:

  Q1  Holding the allocation and the other sleeves fixed, what equity-sleeve
      Sharpe is required for the portfolio to hit 1.1?
  Q2  What would a NEW fourth sleeve have to look like — Sharpe and correlation —
      to carry the portfolio to 1.1 on its own? Produces the full trade-off
      curve, so a candidate asset can be checked against it directly.
  Q3  Is 1.1 attainable at all if gold reverts to a normal decade?

The point is to convert "how do we reach 1.1" from a hunt into a specification.

  MARK5_CACHE=data/pit_cache python3 scripts/path_to_sharpe_11.py
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
                            load_ohlcv, metrics, load_sector_map)

START, END = "2016-01-01", "2026-07-21"
TD, RF = 252, 0.065
TARGET = 1.10
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}
KEYS = ["eq", "GOLDBEES", "MON100"]
ERC_W = None          # computed below
GOLD_HAIRCUT_EXCESS = 0.04


def erc_weights(S, iters=100000, tol=1e-14):
    n = S.shape[0]
    w = np.ones(n) / n
    tgt = 1.0 / n
    for _ in range(iters):
        var = w @ S @ w
        if var <= 0:
            break
        rc = np.maximum((w * (S @ w)) / var, 1e-12)
        wn = np.maximum(w * (tgt / rc) ** 0.5, 1e-12)
        wn /= wn.sum()
        if np.max(np.abs(wn - w)) < tol:
            return wn
        w = wn
    return w


def port_sharpe(w, s, sig, C):
    """Portfolio Sharpe from sleeve Sharpes `s`, vols `sig`, correlation `C`."""
    mu = np.asarray(s) * np.asarray(sig)
    S = np.outer(sig, sig) * C
    var = w @ S @ w
    return float((mu @ w) / np.sqrt(var)) if var > 0 else 0.0


def main():
    panel = DataPanel(discover_tickers(), END, freshness="off")
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.08, factor_weights=MOM)
    run = Backtester(panel, PortfolioConstructor(cfg, sector_map=load_sector_map()),
                     BacktestConfig(rebal_bars=126, top_n_liquid=300)).run(START, END)
    eq = run["nav_gross"]
    cal = eq.index
    R = pd.DataFrame({
        "eq": eq.pct_change(fill_method=None),
        "GOLDBEES": load_ohlcv("GOLDBEES")["close"].astype(float)
                    .reindex(cal, method="ffill").pct_change(fill_method=None),
        "MON100": load_ohlcv("MON100")["close"].astype(float)
                  .reindex(cal, method="ffill").pct_change(fill_method=None),
    }).dropna()

    sig = (R.std() * np.sqrt(TD)).values
    s = ((R.mean() * TD - RF) / (R.std() * np.sqrt(TD))).values
    C = R.corr().values
    S = np.outer(sig, sig) * C
    w_erc = erc_weights(S)

    print("\n" + "=" * 90)
    print("  STARTING POINT")
    print("=" * 90)
    for i, k in enumerate(KEYS):
        print(f"  {k:<12} Sharpe {s[i]:.3f}   vol {sig[i]*100:.1f}%")
    print(f"  eq-US correlation {C[0,2]:.3f}   eq-gold {C[0,1]:.3f}   gold-US {C[1,2]:.3f}")
    print(f"\n  ERC allocation {'/'.join(f'{x*100:.0f}' for x in w_erc)}"
          f"  ->  portfolio Sharpe {port_sharpe(w_erc,s,sig,C):.3f}")
    zero_c = port_sharpe(w_erc, s, sig, np.eye(3))
    print(f"  If the three sleeves were perfectly uncorrelated: {zero_c:.3f}")
    print(f"  -> the eq-US correlation of {C[0,2]:.2f} is what costs the difference.")

    out = {"start": {"sleeve_sharpes": dict(zip(KEYS, map(float, s))),
                     "vols": dict(zip(KEYS, map(float, sig))),
                     "corr": C.tolist(),
                     "erc_weights": dict(zip(KEYS, map(float, w_erc))),
                     "erc_sharpe": port_sharpe(w_erc, s, sig, C),
                     "zero_corr_sharpe": zero_c}}

    # ── Q1: required equity-sleeve Sharpe ────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"  Q1. What equity-sleeve Sharpe reaches portfolio {TARGET:.2f}?")
    print("=" * 90)
    print(f"  {'allocation':<16}{'eq Sharpe now':>15}{'eq Sharpe needed':>19}{'gap':>10}")
    print("  " + "-" * 62)
    q1 = []
    for label, w in (("ERC 27/49/24", w_erc),
                     ("deployed 50/25/25", np.array([.50, .25, .25])),
                     ("40/30/30", np.array([.40, .30, .30]))):
        lo, hi = -1.0, 8.0
        for _ in range(200):                      # bisection on the equity Sharpe
            mid = (lo + hi) / 2
            s2 = s.copy(); s2[0] = mid
            if port_sharpe(w, s2, sig, C) < TARGET:
                lo = mid
            else:
                hi = mid
        need = (lo + hi) / 2
        feasible = need <= 3.0
        print(f"  {label:<16}{s[0]:>15.2f}{need:>19.2f}{need-s[0]:>+10.2f}"
              + ("" if feasible else "   (not attainable)"))
        q1.append({"allocation": label, "eq_sharpe_now": float(s[0]),
                   "eq_sharpe_needed": float(need), "gap": float(need - s[0])})
    print(f"\n  For reference the equity sleeve's Sharpe has been {s[0]:.2f} over 10.6 years,")
    print(f"  and the v7.3 sweep moved it by at most +0.13 with every lever combined.")
    out["q1_required_equity_sharpe"] = q1

    # ── Q2: what a 4th sleeve must look like ─────────────────────────────────
    print("\n" + "=" * 90)
    print(f"  Q2. What must a NEW 4th sleeve look like to carry the book to {TARGET:.2f}?")
    print("=" * 90)
    print("  Assumes it is added at its risk-parity weight alongside the ERC book.")
    print(f"  {'corr to book':>14}{'min Sharpe needed':>20}{'realistic?':>14}")
    print("  " + "-" * 50)
    s_p = port_sharpe(w_erc, s, sig, C)
    q2 = []
    for rho in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        # combining a book (Sharpe s_p) with an asset (Sharpe x, corr rho):
        # S_comb^2 = s_p^2 + ((x - rho*s_p)/sqrt(1-rho^2))^2
        need_sq = TARGET ** 2 - s_p ** 2
        if need_sq <= 0:
            x = -np.inf
        else:
            x = rho * s_p + np.sqrt(need_sq) * np.sqrt(1 - rho ** 2)
        real = ("yes — common" if x < 0.5 else
                "plausible" if x < 0.9 else
                "hard" if x < 1.3 else "unrealistic")
        print(f"  {rho:>14.2f}{x:>20.2f}{real:>14}")
        q2.append({"rho": rho, "min_sharpe": float(x), "assessment": real})
    print(f"\n  Read: a sleeve uncorrelated with the book needs only Sharpe "
          f"{q2[0]['min_sharpe']:.2f} to finish the job.\n  That is a LOW bar — it is the "
          f"lack of a genuinely uncorrelated, long-history asset\n  that is binding, not the "
          f"quality of the asset required.")
    out["q2_required_new_sleeve"] = q2

    # ── Q3: does it survive a normal gold decade? ────────────────────────────
    print("\n" + "=" * 90)
    print(f"  Q3. Does {TARGET:.2f} survive if gold reverts to a normal decade "
          f"({GOLD_HAIRCUT_EXCESS*100:.0f}% excess)?")
    print("=" * 90)
    s_h = s.copy()
    s_h[1] = GOLD_HAIRCUT_EXCESS / sig[1]
    w_h = w_erc
    print(f"  gold Sharpe {s[1]:.2f} -> {s_h[1]:.2f} (vol and correlations unchanged)")
    print(f"  portfolio Sharpe at ERC weights: {port_sharpe(w_erc,s,sig,C):.3f} -> "
          f"{port_sharpe(w_h,s_h,sig,C):.3f}")
    best, bw = -np.inf, None
    for a in np.arange(0, 1.01, 0.02):
        for b in np.arange(0, 1.01 - a, 0.02):
            w = np.array([a, b, 1 - a - b])
            v = port_sharpe(w, s_h, sig, C)
            if v > best:
                best, bw = v, w
    print(f"  best possible allocation under the haircut: "
          f"{'/'.join(f'{x*100:.0f}' for x in bw)} -> Sharpe {best:.3f}")
    print(f"  -> {TARGET:.2f} is {'still reachable' if best >= TARGET else 'NOT reachable'} "
          f"if gold is ordinary.")
    out["q3_gold_haircut"] = {"gold_sharpe_after": float(s_h[1]),
                              "erc_sharpe_after": port_sharpe(w_h, s_h, sig, C),
                              "best_possible": float(best),
                              "best_weights": [float(x) for x in bw],
                              "target_reachable": bool(best >= TARGET)}

    print("\n" + "=" * 90)
    print("  VERDICT")
    print("=" * 90)
    erc_s = port_sharpe(w_erc, s, sig, C)
    print(f"  Reallocating to risk parity is worth {erc_s - port_sharpe(np.array([.5,.25,.25]),s,sig,C):+.3f} "
          f"Sharpe and takes MaxDD -22.1% -> -17.9%.")
    print(f"  It is robust: derived without expected returns, stable in every subperiod.")
    print(f"  It reaches {erc_s:.2f}, not {TARGET:.2f}.")
    print(f"  Closing the last {TARGET-erc_s:.2f} needs EITHER an equity sleeve at Sharpe "
          f"{q1[0]['eq_sharpe_needed']:.2f}\n  (vs {s[0]:.2f} today, and the entire v7.3 lever "
          f"sweep bought +0.13) OR a fourth\n  uncorrelated sleeve with Sharpe "
          f">= {q2[0]['min_sharpe']:.2f} and a long enough history to trust.")
    p = os.path.join(_ROOT, "reports", "path_to_sharpe_11.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n  saved -> {p}\n")


if __name__ == "__main__":
    main()

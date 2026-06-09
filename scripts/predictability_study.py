"""
MARK6 — Predictability Study: CAN a system pick the HAL-type winner ex-ante?
============================================================================
Tests the user's exact thesis: "find the right small/midcaps before they run,
beat HAL's +600%, or it's not alpha."

We measure, in the actual data:
  1. INFORMATION COEFFICIENT (Spearman rank corr) of ex-ante signals vs FORWARD
     1y/2y/3y returns. High IC => signals predict who wins. Low IC => they don't.
  2. DECILE ANALYSIS: forward return by ex-ante signal decile (mean/median + the
     P(>2x), P(>5x) hit rates). Shows whether the multibaggers cluster in the
     top signal decile or are essentially unpredictable.
  3. THE WINNER'S EX-ANTE RANK: for each formation year, where did that year's
     eventual best performer rank on the signal BEFORE the run? (HAL/BEL/TRENT too)

Verdict logic: if the eventual 5-10x winners were NOT reliably top-ranked ex-ante,
then "predict the 600% stock" is survivorship bias, and a basket (MARK6) is the
only honest way to capture the premium.
"""
import os, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio.universe import DataPanel, discover_tickers
from core.portfolio.factors import FactorLibrary, composite_score

END = "2026-05-21"


def main():
    tickers = discover_tickers()
    panel = DataPanel(tickers, END)
    close = panel.close
    print(f"Universe: {len(panel.tickers)} names\n")

    # precompute factor series
    facs = {t: FactorLibrary.compute_all(close[t]) for t in panel.tickers}

    formation = [f"{y}-01-15" for y in range(2016, 2024)]
    horizons = {"1y": 252, "2y": 504, "3y": 756}

    ic_rows = []          # (formation, horizon, IC_momentum, IC_composite, n)
    pooled = {h: {"sig": [], "fwd": []} for h in horizons}
    winner_ranks = []     # (formation, horizon, winner, fwd_ret, signal_pct)

    for f in formation:
        fd = pd.Timestamp(f)
        elig = panel.eligible(fd)
        if len(elig) < 20:
            continue
        # ex-ante signals as-of formation
        mom = {}
        comp_panel = {fn: {} for fn in FactorLibrary.DEFAULT_FACTORS}
        for t in elig:
            row = facs[t].loc[:fd]
            if row.empty:
                continue
            last = row.iloc[-1]
            mom[t] = last.get("momentum", np.nan)
            for fn in FactorLibrary.DEFAULT_FACTORS:
                comp_panel[fn][t] = last.get(fn, np.nan)
        mom = pd.Series(mom)
        comp = composite_score({fn: pd.Series(comp_panel[fn]) for fn in comp_panel})

        for h, bars in horizons.items():
            td = fd + pd.Timedelta(days=int(bars * 1.4))  # calendar approx
            fwd = {}
            for t in elig:
                s = close[t].loc[fd:td].dropna()
                if len(s) > bars * 0.7:
                    fwd[t] = s.iloc[-1] / s.iloc[0] - 1
            fwd = pd.Series(fwd)
            common = comp.index.intersection(fwd.index)
            if len(common) < 20:
                continue
            ic_m = mom.reindex(common).corr(fwd.reindex(common), method="spearman")
            ic_c = comp.reindex(common).corr(fwd.reindex(common), method="spearman")
            ic_rows.append((f[:4], h, ic_m, ic_c, len(common)))
            pooled[h]["sig"].extend(comp.reindex(common).tolist())
            pooled[h]["fwd"].extend(fwd.reindex(common).tolist())
            # winner location
            win = fwd.idxmax()
            sig_pct = (comp.reindex(common) < comp.get(win, np.nan)).mean() * 100
            winner_ranks.append((f[:4], h, win, fwd[win], sig_pct))

    # ── IC table ─────────────────────────────────────────────────────────────
    print("="*78)
    print("  INFORMATION COEFFICIENT  (Spearman rank corr: ex-ante signal vs forward return)")
    print("  IC ~0 = no predictive power | IC>0.10 = useful | IC>0.30 = strong")
    print("="*78)
    icdf = pd.DataFrame(ic_rows, columns=["year", "h", "IC_mom", "IC_comp", "n"])
    for h in horizons:
        sub = icdf[icdf.h == h]
        print(f"  {h}: mean IC_momentum={sub.IC_mom.mean():+.3f}  "
              f"mean IC_composite={sub.IC_comp.mean():+.3f}  "
              f"(range {sub.IC_comp.min():+.2f}..{sub.IC_comp.max():+.2f}, {len(sub)} years)")

    # ── decile analysis (pooled) ─────────────────────────────────────────────
    print("\n" + "="*78)
    print("  FORWARD RETURN BY EX-ANTE SIGNAL DECILE (pooled, 2y horizon)")
    print("="*78)
    s = pd.DataFrame({"sig": pooled["2y"]["sig"], "fwd": pooled["2y"]["fwd"]}).dropna()
    s["decile"] = pd.qcut(s.sig, 10, labels=False, duplicates="drop")
    print(f"  {'decile':<8}{'mean fwd':>10}{'median':>9}{'P(>2x)':>9}{'P(>5x)':>9}{'max':>9}")
    for d in sorted(s.decile.unique()):
        g = s[s.decile == d].fwd
        print(f"  {'top' if d==s.decile.max() else ('bot' if d==0 else int(d)):<8}"
              f"{g.mean()*100:>+9.0f}%{g.median()*100:>+8.0f}%"
              f"{(g>1).mean()*100:>8.0f}%{(g>4).mean()*100:>8.0f}%{g.max()*100:>+8.0f}%")

    # ── where did the winners rank ex-ante? ──────────────────────────────────
    print("\n" + "="*78)
    print("  THE EVENTUAL WINNER'S EX-ANTE SIGNAL PERCENTILE (could we have known?)")
    print("  100% = winner was top-ranked beforehand | ~50% = signal was blind to it")
    print("="*78)
    wdf = pd.DataFrame(winner_ranks, columns=["year", "h", "winner", "fwd", "sig_pct"])
    w2 = wdf[wdf.h == "2y"]
    for _, r in w2.iterrows():
        flag = "TOP-decile ✓" if r.sig_pct >= 90 else ("top-half" if r.sig_pct >= 50 else "BELOW median ✗")
        print(f"  {r.year}: best 2y performer = {r.winner:<12} ({r.fwd*100:+.0f}%)  "
              f"ex-ante signal pct = {r.sig_pct:>3.0f}%  [{flag}]")
    print(f"\n  Winners that were top-decile ex-ante: "
          f"{(w2.sig_pct>=90).sum()}/{len(w2)}  |  below-median ex-ante: {(w2.sig_pct<50).sum()}/{len(w2)}")

    # ── HAL/BEL/TRENT specific ───────────────────────────────────────────────
    print("\n" + "="*78)
    print("  CASE STUDY: were HAL/BEL/TRENT flagged BEFORE their runs?")
    print("="*78)
    for t in ["HAL", "BEL", "TRENT"]:
        if t not in close.columns:
            print(f"  {t}: no data"); continue
        s = close[t].dropna()
        start = s.index[0]
        fd = max(pd.Timestamp("2018-06-01"), start + pd.Timedelta(days=300))
        if fd not in facs[t].index:
            fd = facs[t].loc[:fd].index[-1] if len(facs[t].loc[:fd]) else None
        if fd is None:
            continue
        elig = panel.eligible(fd)
        comp_panel = {fn: {} for fn in FactorLibrary.DEFAULT_FACTORS}
        for tk in elig:
            row = facs[tk].loc[:fd]
            if not row.empty:
                for fn in FactorLibrary.DEFAULT_FACTORS:
                    comp_panel[fn][tk] = row.iloc[-1].get(fn, np.nan)
        comp = composite_score({fn: pd.Series(comp_panel[fn]) for fn in comp_panel})
        fwd = s.loc[fd:].iloc[-1] / s.loc[:fd].iloc[-1] - 1
        pct = (comp < comp.get(t, np.nan)).mean() * 100 if t in comp.index else np.nan
        print(f"  {t}: as-of {str(fd.date())} signal pct={pct:>3.0f}%  -> subsequent return {fwd*100:+.0f}%")


if __name__ == "__main__":
    main()

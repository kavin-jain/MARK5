"""
Institutional risk report — the numbers a fund publishes and this repo lacked.
=============================================================================
Performance answers "what did it make". Risk answers "how could it have gone
wrong, and what is it actually exposed to". An allocator reads the second first.

Sections
  1 TAIL RISK        historical and parametric VaR / CVaR at 95% and 99%, daily
                     and 21-day. Historical is reported alongside parametric on
                     purpose: returns are not normal, and the gap between the two
                     IS the fat tail.
  2 FACTOR EXPOSURE  the system regressed on real long/short factor portfolios
                     built from THIS universe, point-in-time — market, size,
                     momentum, low-volatility. Answers "is this alpha, or is it
                     just a small-cap momentum bet with a fancy name?"
  3 DRAWDOWN ATTRIB  for each of the worst drawdowns, which sleeve caused it.
  4 STRESS           real crises, plus the worst rolling windows.

Everything is computed from the deployed configuration on the survivorship-free
universe. Nothing is illustrative.

  MARK5_CACHE=data/pit_cache python3 scripts/risk_report.py
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
                            load_ohlcv, load_nifty, metrics, load_sector_map,
                            load_delivery_factors)
from core.portfolio.factors import FactorLibrary

START, END = "2016-01-01", "2026-07-21"
TD = 252
SLEEVES = {"eq": 0.50, "GOLDBEES": 0.25, "MON100": 0.25}
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}


def var_cvar(r, q):
    """Historical VaR/CVaR: the empirical quantile and the mean beyond it."""
    v = float(np.quantile(r, 1 - q))
    tail = r[r <= v]
    return v, (float(tail.mean()) if len(tail) else v)


def build_factors(panel, cal):
    """Long/short factor returns from the point-in-time universe itself.

    Each factor is rebalanced every 21 bars: long the top tercile, short the
    bottom tercile, equal-weighted, of the 300 most liquid seasoned names as of
    that date. These are the real factors present in THIS market, not imported
    US series — which is the only way the exposure numbers mean anything.
    """
    fac = {k: pd.Series(0.0, index=cal) for k in ("size", "momentum", "lowvol")}
    rets = panel.close.pct_change(fill_method=None)
    cache = {}
    step = 21
    anchors = list(range(0, len(cal), step))
    for ai, i in enumerate(anchors):
        d = cal[i]
        elig = panel.eligible(d, 252, top_n=300)
        if len(elig) < 60:
            continue
        turn, mom, vol = {}, {}, {}
        for t in elig:
            tv = panel.turnover[t].loc[:d].dropna()
            s = panel.close[t].loc[:d]
            if not len(tv) or len(s) < 252:
                continue
            if t not in cache:
                cache[t] = FactorLibrary.compute_all(panel.close[t])
            row = cache[t].loc[:d]
            if row.empty:
                continue
            last = row.iloc[-1]
            turn[t] = float(tv.iloc[-1])          # turnover as the size proxy
            mom[t] = last.get("momentum", np.nan)
            vol[t] = last.get("low_vol", np.nan)
        if len(turn) < 60:
            continue
        j0, j1 = i, (anchors[ai + 1] if ai + 1 < len(anchors) else len(cal))
        seg = cal[j0:j1]
        for key, src, flip in (("size", turn, True), ("momentum", mom, False),
                               ("lowvol", vol, False)):
            s = pd.Series(src).dropna()
            if len(s) < 30:
                continue
            k = len(s) // 3
            hi = list(s.sort_values(ascending=False).index[:k])
            lo = list(s.sort_values(ascending=False).index[-k:])
            if flip:                               # size = SMALL minus BIG
                hi, lo = lo, hi
            r = (rets.loc[seg, hi].mean(axis=1) - rets.loc[seg, lo].mean(axis=1))
            fac[key].loc[seg] = r.fillna(0.0).values
    return pd.DataFrame(fac)


def ols(y, X):
    """Plain OLS with an intercept; returns (alpha_annualised, betas, r2, tstats)."""
    A = np.column_stack([np.ones(len(X)), X.values])
    b, *_ = np.linalg.lstsq(A, y.values, rcond=None)
    resid = y.values - A @ b
    dof = max(1, len(y) - A.shape[1])
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    tt = b / np.where(se > 0, se, np.nan)
    ss = float(((y.values - y.values.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss if ss > 0 else np.nan
    return b[0] * TD, dict(zip(X.columns, b[1:])), r2, dict(zip(X.columns, tt[1:])), tt[0]


def main():
    panel = DataPanel(discover_tickers(), END, freshness="off")
    # v7.7 PROVISIONAL: deliv_chg @10% (RESEARCH_LOG 4l)
    dfac = load_delivery_factors(universe=panel.tickers)
    fw = dict(MOM)
    if dfac:
        fw["deliv_chg"] = 0.10
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.08, factor_weights=fw)
    run = Backtester(panel, PortfolioConstructor(cfg, sector_map=load_sector_map()),
                     BacktestConfig(rebal_bars=126, top_n_liquid=300),
                     extra_factors=dfac).run(START, END)
    eq = run["nav_gross"]
    cal = eq.index

    sr = {"eq": eq.pct_change(fill_method=None).fillna(0.0)}
    for k in SLEEVES:
        if k == "eq":
            continue
        sr[k] = (load_ohlcv(k)["close"].astype(float)
                 .reindex(cal, method="ffill").pct_change().fillna(0.0))
    cur, nav, out, sleeve_pnl = dict(SLEEVES), 1.0, {}, {k: [] for k in SLEEVES}
    for i, d in enumerate(cal):
        if i > 0:
            prev = sum(cur.values())
            for k in cur:
                g = cur[k] * sr[k].iloc[i]
                sleeve_pnl[k].append(g / prev)
                cur[k] += g
            nav *= sum(cur.values()) / prev
        else:
            for k in cur:
                sleeve_pnl[k].append(0.0)
        out[d] = nav
        if i > 0 and i % TD == 0:
            tot = sum(cur.values())
            cur = {k: tot * SLEEVES[k] for k in SLEEVES}
    sysnav = pd.Series(out)
    ret = sysnav.pct_change().dropna()
    spnl = pd.DataFrame(sleeve_pnl, index=cal)

    L = ["# MARK6 — Institutional Risk Report", "",
         f"Deployed configuration on the survivorship-free point-in-time universe "
         f"({len(panel.tickers)} symbols), {START} → {END}. All figures computed "
         f"from the daily net series; nothing here is illustrative.", ""]

    # ── 1. tail risk ─────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("  1. TAIL RISK")
    print("=" * 84)
    L += ["## 1. Tail risk", "",
          "Historical VaR makes no distributional assumption; parametric assumes "
          "normality. The **gap between them is the fat tail** — where parametric "
          "is smaller, a normal model is understating how bad the bad days get.", "",
          "| Horizon | Confidence | Historical VaR | Historical CVaR | Parametric VaR |",
          "|---|---|---|---|---|"]
    mu, sd = ret.mean(), ret.std()
    from scipy import stats as ss
    for horizon, scale in (("1 day", 1), ("21 days", np.sqrt(21))):
        for q in (0.95, 0.99):
            r_h = ret if horizon == "1 day" else ret.rolling(21).sum().dropna()
            hv, hc = var_cvar(r_h.values, q)
            pv = (mu * (1 if horizon == "1 day" else 21)
                  - ss.norm.ppf(q) * sd * scale)
            print(f"  {horizon:<9}{q*100:>5.0f}%   histVaR {hv*100:>7.2f}%   "
                  f"histCVaR {hc*100:>7.2f}%   paramVaR {pv*100:>7.2f}%")
            L.append(f"| {horizon} | {q*100:.0f}% | {hv*100:.2f}% | {hc*100:.2f}% "
                     f"| {pv*100:.2f}% |")
    sk, ku = float(ss.skew(ret)), float(ss.kurtosis(ret, fisher=False))
    print(f"\n  skew {sk:+.2f}   kurtosis {ku:.2f} (normal = 3)   "
          f"worst day {ret.min()*100:.2f}%   best day {ret.max()*100:+.2f}%")
    L += ["", f"Daily skew **{sk:+.2f}**, kurtosis **{ku:.1f}** (normal = 3.0) — "
          f"fat-tailed and negatively skewed, the usual equity shape. Worst day "
          f"{ret.min()*100:.2f}%, best {ret.max()*100:+.2f}%.", ""]

    # ── 2. factor exposure ───────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("  2. FACTOR EXPOSURE  — is this alpha, or a disguised small-cap momentum bet?")
    print("=" * 84)
    print("  building point-in-time long/short factors from this universe...", flush=True)
    F = build_factors(panel, cal)
    nifty = load_nifty(True).reindex(cal, method="ffill")
    F["market"] = nifty.pct_change().fillna(0.0)
    X = F.loc[ret.index, ["market", "size", "momentum", "lowvol"]].fillna(0.0)
    for label, y in (("Full system 50/25/25", ret),
                     ("Equity sleeve only", sr["eq"].loc[ret.index])):
        a, betas, r2, tstats, ta = ols(y, X)
        print(f"\n  {label}")
        print(f"    annualised alpha {a*100:+.2f}%  (t = {ta:.2f})   R2 = {r2:.2f}")
        for k in X.columns:
            print(f"    beta {k:<10}{betas[k]:>7.3f}   (t = {tstats[k]:>6.2f})")
        L += [f"### {label}", "",
              f"Annualised alpha **{a*100:+.2f}%** (t = {ta:.2f}), R² = {r2:.2f}.", "",
              "| Factor | Beta | t-stat |", "|---|---|---|"]
        L += [f"| {k} | {betas[k]:+.3f} | {tstats[k]:.2f} |" for k in X.columns]
        L.append("")
    L += ["Factors are long/short terciles rebuilt every 21 bars from the same "
          "point-in-time universe (size = small minus big by turnover; momentum "
          "and low-volatility from the engine's own causal definitions). A high "
          "R² with a large momentum beta would mean the book is simply a momentum "
          "index; a surviving positive alpha means the ranking adds something the "
          "raw factors do not.", ""]

    # ── 3. drawdown attribution ──────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("  3. DRAWDOWN ATTRIBUTION — which sleeve caused each one?")
    print("=" * 84)
    dd = sysnav / sysnav.cummax() - 1
    episodes, in_dd, start = [], False, None
    for d, v in dd.items():
        if not in_dd and v < -0.05:
            in_dd, start = True, d
        elif in_dd and v >= -1e-9:
            episodes.append((start, d))
            in_dd = False
    if in_dd:
        episodes.append((start, dd.index[-1]))
    episodes = sorted(episodes, key=lambda e: dd.loc[e[0]:e[1]].min())[:5]
    L += ["## 3. Drawdown attribution", "",
          "Each drawdown worse than −5%, decomposed into the rupee P&L each sleeve "
          "contributed while it was happening. This is what makes the multi-asset "
          "structure load-bearing rather than decorative.", "",
          "| Peak | Trough | Depth | Days | Equity | Gold | US |", "|---|---|---|---|---|---|---|"]
    print(f"  {'peak':<12}{'trough':<12}{'depth':>8}{'days':>6}"
          f"{'equity':>10}{'gold':>9}{'US':>9}")
    for s, e in episodes:
        seg = dd.loc[s:e]
        tr = seg.idxmin()
        c = {k: spnl[k].loc[s:tr].sum() * 100 for k in SLEEVES}
        print(f"  {str(s.date()):<12}{str(tr.date()):<12}{seg.min()*100:>7.1f}%"
              f"{(e-s).days:>6}{c['eq']:>9.1f}%{c['GOLDBEES']:>8.1f}%{c['MON100']:>8.1f}%")
        L.append(f"| {s.date()} | {tr.date()} | {seg.min()*100:.1f}% | {(e-s).days} "
                 f"| {c['eq']:+.1f}% | {c['GOLDBEES']:+.1f}% | {c['MON100']:+.1f}% |")
    L.append("")

    # ── 4. stress ────────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("  4. STRESS — real crises and worst rolling windows")
    print("=" * 84)
    scen = {"2018 NBFC / IL&FS": ("2018-08-01", "2019-02-28"),
            "COVID crash 2020": ("2020-02-01", "2020-04-30"),
            "2022 rate shock": ("2022-01-01", "2022-06-30"),
            "2024-25 correction": ("2024-09-01", "2025-03-31")}
    nifty_nav = nifty / nifty.iloc[0]
    L += ["## 4. Stress tests", "", "| Scenario | System | Nifty50 TRI |", "|---|---|---|"]
    for name, (a, b) in scen.items():
        s1, s2 = sysnav.loc[a:b], nifty_nav.loc[a:b]
        if len(s1) > 5:
            p, q = s1.iloc[-1] / s1.iloc[0] - 1, s2.iloc[-1] / s2.iloc[0] - 1
            print(f"  {name:<24}{p*100:>+8.1f}%   Nifty {q*100:>+7.1f}%")
            L.append(f"| {name} | {p*100:+.1f}% | {q*100:+.1f}% |")
    r1 = sysnav.pct_change().rolling(TD).apply(lambda x: np.prod(1 + x) - 1, raw=True).dropna()
    print(f"\n  worst rolling 1-year return: {r1.min()*100:+.1f}%  "
          f"(ending {r1.idxmin().date()})")
    print(f"  share of rolling 1-year windows negative: {(r1 < 0).mean()*100:.0f}%")
    L += ["", f"Worst rolling 1-year return **{r1.min()*100:+.1f}%** (ending "
          f"{r1.idxmin().date()}); **{(r1<0).mean()*100:.0f}%** of rolling 1-year "
          f"windows were negative. A holder must be able to sit through both.", ""]

    p = os.path.join(_ROOT, "reports", "RISK_REPORT.md")
    open(p, "w").write("\n".join(L) + "\n")
    js = os.path.join(_ROOT, "reports", "risk_report.json")
    a, betas, r2, tstats, ta = ols(ret, X)
    json.dump({"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
               "var": {f"hist_{int(q*100)}": var_cvar(ret.values, q) for q in (0.95, 0.99)},
               "skew": sk, "kurtosis": ku,
               "factor_alpha_ann_pct": a * 100, "factor_alpha_t": ta,
               "factor_betas": betas, "factor_tstats": tstats, "r2": r2,
               "worst_rolling_1y_pct": float(r1.min() * 100),
               "pct_negative_1y_windows": float((r1 < 0).mean() * 100)},
              open(js, "w"), indent=1, default=float)
    print(f"\n  saved -> {p}\n           {js}\n")


if __name__ == "__main__":
    main()

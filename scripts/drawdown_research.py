"""
Drawdown research — every remaining weight/selection lever against MaxDD.
========================================================================
The equity sleeve's -47% drawdown is the system's worst feature and the reason
the multi-asset wrapper is load-bearing rather than cosmetic. Market timing,
stops and circuit breakers are already KILLED (K2/K4/K5) and are not revisited;
everything here is weight-space or selection-space and always fully invested.

The diagnostic (risk_model_diagnostic.py) already established WHY the drawdown
is large: it is not correlation (the book makes 14.9 independent bets of a
nominal 20 and sits at its diversification limit) but the raw volatility of the
names selected -- 47.6% annualised on average. So every lever here attacks
selection or concentration, not the covariance structure.

Levers
  R    rank-transform            score on ORDER, not raw value (validated 7/8 on
                                 MaxDD in the previous sweep; carried as the
                                 reference to beat)
  D    diversification-aware     greedy select: maximise score minus lambda x
       selection                 average correlation to what is already picked.
                                 Attacks the 25% of nominal breadth the
                                 diagnostic showed is illusory.
  S    sector cap ENFORCED       max_sector_weight is dead code in production --
                                 sector_map is never passed. This turns it on.
  W6   tighter name cap          max_weight 0.08 -> 0.06
  TR   name-level trend screen   drop names below their own 200d MA before
                                 ranking. Per-name, not market timing.
  DD   downside-deviation screen drop the worst quartile by trailing downside
                                 deviation (not total vol -- K28 showed cutting
                                 total vol costs return; this cuts only the
                                 asymmetric part)
  N25  n_hold 25                 flagged by capital_flexibility.py on a single
                                 full-period path; PBO=76.7% says validate it
                                 walk-forward before believing it

  MARK5_CACHE=data/pit_cache python3 scripts/drawdown_research.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, BacktestConfig, metrics)
import core.portfolio.factors as FF

sys.argv = ["x"]
import importlib.util as _u
_s = _u.spec_from_file_location("er", os.path.join(_ROOT, "scripts",
                                                   "edge_research_2026_07.py"))
er = _u.module_from_spec(_s)
_s.loader.exec_module(er)

START, END = er.START, er.END
SECTORS = json.load(open(os.path.join(_ROOT, "config", "sector_map.json")))["sectors"]


# ── D: diversification-aware selection ───────────────────────────────────────
class DivConstructor(PortfolioConstructor):
    """Greedy selection trading raw score against correlation to the book so far.

    The diagnostic showed the book makes 14.9 independent bets of a nominal 20.
    Ranking purely by score is blind to that: two names can both score well and
    be the same bet. This picks the best name by (score - lam * mean corr to
    already-selected), which is the standard greedy diversification heuristic.
    """

    def __init__(self, cfg, lam=0.5, sector_map=None):
        super().__init__(cfg, sector_map)
        self.lam = lam
        self.corr = None            # set per rebalance by DivBacktester

    def select(self, composite, currently_held):
        cfg = self.cfg
        if self.corr is None or cfg.mode == "equal_weight":
            return super().select(composite, currently_held)
        ranked = composite.sort_values(ascending=False)
        rank_of = {t: i for i, t in enumerate(ranked.index)}
        exit_rank = int(cfg.n_hold * cfg.buffer_mult)
        keep = [t for t in currently_held if rank_of.get(t, 10**9) < exit_rank][:cfg.n_hold]
        picked = list(keep)
        pool = [t for t in ranked.index if t not in picked]
        C = self.corr
        z = (composite - composite.mean()) / (composite.std(ddof=0) or 1.0)
        while len(picked) < cfg.n_hold and pool:
            if picked:
                inb = [t for t in picked if t in C.index]
                pen = (C.loc[[t for t in pool if t in C.index], inb].mean(axis=1)
                       if inb else pd.Series(0.0, index=pool))
            else:
                pen = pd.Series(0.0, index=pool)
            adj = z.reindex(pool).fillna(0.0) - self.lam * pen.reindex(pool).fillna(0.0)
            picked.append(str(adj.idxmax()))
            pool.remove(picked[-1])
        return picked[:cfg.n_hold]


class DivBacktester(Backtester):
    """Supplies the constructor a strictly causal correlation matrix each rebalance."""

    def _factor_panel(self, asof, names):
        comp, vol = super()._factor_panel(asof, names)
        if isinstance(self.con, DivConstructor):
            top = list(comp.sort_values(ascending=False).index[:120])
            h = (self.panel.close.loc[:asof, top].pct_change(fill_method=None)
                 .tail(252).dropna(axis=1, thresh=126).fillna(0.0))
            self.con.corr = h.corr() if h.shape[1] >= 3 else None
        return comp, vol


# ── screens ──────────────────────────────────────────────────────────────────
def trend_screen(panel):
    def screen(asof, elig):
        keep = []
        for t in elig:
            s = panel.close[t].loc[:asof]
            if len(s) < 200:
                continue
            if float(s.iloc[-1]) >= float(s.tail(200).mean()):
                keep.append(t)
        return keep if len(keep) >= 40 else elig
    return screen


def downside_screen(panel, pct=0.25):
    def screen(asof, elig):
        dd = {}
        for t in elig:
            r = panel.close[t].loc[:asof].pct_change(fill_method=None).tail(252).dropna()
            if len(r) < 126:
                continue
            neg = np.minimum(r.values, 0.0)
            dd[t] = float(np.sqrt(np.mean(neg ** 2)) * np.sqrt(252))
        if len(dd) < 40:
            return elig
        cut = np.nanquantile(list(dd.values()), 1.0 - pct)
        keep = [t for t in elig if dd.get(t, 0.0) <= cut]
        return keep if len(keep) >= 40 else elig
    return screen


def run_spec(spec, s, e, panel):
    ctx = er.use_rank_z() if spec.get("rank") else None
    if ctx:
        ctx.__enter__()
    try:
        cfg = ConstructionConfig(**{**er.BASE_CON, **(spec.get("con") or {})})
        smap = SECTORS if spec.get("sector") else None
        if spec.get("div"):
            con = DivConstructor(cfg, lam=spec["div"], sector_map=smap)
            klass = DivBacktester
        else:
            con = PortfolioConstructor(cfg, sector_map=smap)
            klass = Backtester
        sc = None
        if spec.get("screen") == "trend":
            sc = trend_screen(panel)
        elif spec.get("screen") == "downside":
            sc = downside_screen(panel)
        bt = klass(panel, con, BacktestConfig(**{**er.BASE_BT, **(spec.get("bt") or {})}),
                   screen=sc)
        r = bt.run(s, e)
        return metrics(r["nav_net"]), metrics(er.wrap(r["nav_gross"]))
    except Exception as ex:
        print(f"    ! {spec['name']} {s}: {type(ex).__name__}: {ex}")
        return None, None


SPECS = [
    {"name": "BASELINE (deployed)"},
    {"name": "R   rank-transform", "rank": True},
    {"name": "D   diversify-select .5", "div": 0.5},
    {"name": "D   diversify-select 1.0", "div": 1.0},
    {"name": "S   sector cap ENFORCED", "sector": True},
    {"name": "W6  name cap 6%", "con": {"max_weight": 0.06}},
    {"name": "TR  above-200MA screen", "screen": "trend"},
    {"name": "DD  downside-dev screen", "screen": "downside"},
    {"name": "N25 n_hold 25", "con": {"n_hold": 25}},
    {"name": "R + S + N25", "rank": True, "sector": True, "con": {"n_hold": 25}},
    {"name": "R + TR", "rank": True, "screen": "trend"},
]


def main():
    panel = er.panel()
    print(f"\n  panel {len(panel.tickers)} symbols | bar: >=6/8 rolling 3-yr windows, per metric\n")
    windows = []
    for y0 in range(2016, 2024):
        e = f"{y0+2}-12-31"
        if pd.Timestamp(e) > pd.Timestamp(END):
            e = END
        windows.append((f"{y0}-01-01", e))

    res = {}
    for sp in SPECS:
        fe, fs = run_spec(sp, START, END, panel)
        wf = [run_spec(sp, s, e, panel) for s, e in windows]
        res[sp["name"]] = {"eq": fe, "sy": fs, "wf": wf}
        print(f"  ran {sp['name']}")

    b = res["BASELINE (deployed)"]
    for lvl, idx, label in ((0, "eq", "EQUITY SLEEVE"),
                            (1, "sy", "FULL SYSTEM 50/25/25")):
        print("\n" + "=" * 114)
        print(f"  {label}")
        print("=" * 114)
        print(f"  {'config':<24}{'CAGR':>8}{'shExc':>7}{'vol':>7}{'MaxDD':>8}{'Calmar':>7}"
              f"   {'wf wins':>9}{'CAGR':>7}{'Sharpe':>8}{'MaxDD':>8}{'Calmar':>8}")
        print("  " + "-" * 110)
        for n, r in res.items():
            m = r[idx]
            if not m:
                continue
            cw = []
            for k in ("cagr", "sharpe_excess", "max_dd", "calmar"):
                w = sum(1 for x, y in zip(r["wf"], b["wf"])
                        if x[lvl] and y[lvl] and x[lvl][k] > y[lvl][k])
                cw.append(f"{w}/8")
            tag = ["", "", "", ""] if n.startswith("BASELINE") else cw
            print(f"  {n:<24}{m['cagr']*100:>+7.2f}%{m['sharpe_excess']:>7.2f}"
                  f"{m['vol']*100:>6.1f}%{m['max_dd']*100:>+7.1f}%{m['calmar']:>7.2f}"
                  f"   {'':>9}{tag[0]:>7}{tag[1]:>8}{tag[2]:>8}{tag[3]:>8}")

    p = os.path.join(_ROOT, "reports", "drawdown_research.json")
    json.dump({k: {"eq": v["eq"], "sy": v["sy"]} for k, v in res.items()},
              open(p, "w"), indent=1, default=float)
    print(f"\n  saved -> {p}\n")


if __name__ == "__main__":
    main()

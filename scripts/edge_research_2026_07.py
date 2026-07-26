"""
Edge research — structural levers not yet tested on the honest PIT universe.
===========================================================================
Every idea here is WEIGHT-SPACE or SELECTION-SPACE and always fully invested.
Market timing, stops, circuit breakers and leverage are already KILLED (K2/K4/
K5/K13) and are not revisited.

Judged the way this project judges everything: rolling 3-year walk-forward, and
an idea only counts if it beats the deployed baseline in >=6/8 windows. A better
full-period mean with a losing walk-forward record is how this repo has been
fooled before (K21/K22), so the win-count is the headline, not the mean.

Levers under test
  T  tranching            P14 is a validated KEEP that was never wired into
                          production. 3 sleeves, rebalance cycles staggered.
  N  n_hold               breadth. Grinold: IR = IC x sqrt(breadth). n_hold was
                          last tuned (P5) on the SURVIVOR cache; never re-tested
                          on the honest PIT universe, where the factor's edge
                          over equal-weight is larger.
  B  buffer_mult          turnover. The book runs 255%/yr on a semi-annual
                          cadence, i.e. it replaces itself every rebalance.
  V  max-vol screen       the diagnostic showed selected names average 47.6%
                          annualised vol. Drop the most extreme before ranking.
  R  rank-transform       z-score the RANKS, not the raw factor values.
                          Momentum is heavily right-skewed; a handful of extreme
                          values dominate a raw z-score even after clipping.

Reported at BOTH levels, because they are different decisions:
  equity sleeve  — the part this repo actually builds
  full system    — 50/25/25 with gold + Nasdaq, the published headline. A sleeve
                   gain is roughly halved here, which is the honest way to see it.

  MARK5_CACHE=data/pit_cache python3 scripts/edge_research_2026_07.py
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
                            metrics, tranched_run, load_ohlcv)
from core.portfolio import factors as F

START, END = "2016-01-01", "2026-07-21"
TD, TAX = 252, 0.15
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}
BASE_CON = dict(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                tilt_strength=1.5, max_weight=0.08, factor_weights=MOM)
BASE_BT = dict(rebal_bars=126, top_n_liquid=300)

_PANEL = None
_SLEEVE = {}


def panel():
    global _PANEL
    if _PANEL is None:
        _PANEL = DataPanel(discover_tickers(), END, freshness="off")
    return _PANEL


def wrap(eq_nav):
    """Deployed 50/25/25 blend, annual sleeve rebalance, terminal tax.
    Identical to scripts/export_dashboard.py so results stay comparable."""
    cal = eq_nav.index
    ser = {"eq": eq_nav.pct_change(fill_method=None).fillna(0.0)}
    sl = {"eq": .5, "GOLDBEES": .25, "MON100": .25}
    for k in sl:
        if k == "eq":
            continue
        if k not in _SLEEVE:
            _SLEEVE[k] = load_ohlcv(k)["close"].astype(float)
        ser[k] = _SLEEVE[k].reindex(cal, method="ffill").pct_change().fillna(0.0)
    cur, nav, out = dict(sl), 1.0, {}
    for i, d in enumerate(cal):
        if i > 0:
            prev = sum(cur.values())
            for k in cur:
                cur[k] *= (1 + ser[k].iloc[i])
            nav *= sum(cur.values()) / prev
        out[d] = nav
        if i > 0 and i % TD == 0:
            tot = sum(cur.values())
            cur = {k: tot * sl[k] for k in sl}
    s = pd.Series(out)
    net = s.copy()
    net.iloc[-1] = s.iloc[-1] - max(0.0, s.iloc[-1] - 1) * TAX
    return net


# ── R: rank-transform composite ──────────────────────────────────────────────
_orig_z = F.cross_sectional_z


def rank_z(values: pd.Series, clip: float = 3.0) -> pd.Series:
    """Z-score the cross-sectional RANKS rather than the raw values.

    Raw momentum is right-skewed: one name up 400% sets the scale and squashes
    every real distinction below it. Ranking makes the score depend on ORDER,
    which is all the composite actually uses, and is immune to the tail.
    """
    v = values.astype(float)
    r = v.rank(method="average")
    mu, sd = r.mean(), r.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=v.index)
    return ((r - mu) / sd).clip(-clip, clip).reindex(v.index)


class use_rank_z:
    def __enter__(self):
        F.cross_sectional_z = rank_z
        import core.portfolio.factors as ff
        ff.cross_sectional_z = rank_z

    def __exit__(self, *a):
        F.cross_sectional_z = _orig_z
        import core.portfolio.factors as ff
        ff.cross_sectional_z = _orig_z


# ── V: max-volatility screen ─────────────────────────────────────────────────
def vol_screen(pct: float):
    """Drop the most volatile `pct` fraction of eligible names BEFORE ranking.

    An exclusion screen, not a tilt — K10 killed tilting the composite harder
    toward low-vol (it costs CAGR). Removing only the extreme tail is a
    different intervention: it never reweights the survivors.
    """
    p = panel()

    def screen(asof, elig):
        vols = {}
        for t in elig:
            s = p.close[t].loc[:asof].tail(252)
            if len(s) < 126:
                continue
            vols[t] = float(s.pct_change(fill_method=None).std() * np.sqrt(252))
        if len(vols) < 30:
            return elig
        cut = np.nanquantile(list(vols.values()), 1.0 - pct)
        keep = [t for t in elig if vols.get(t, 0.0) <= cut]
        return keep if len(keep) >= 40 else elig
    return screen


# ── runner ───────────────────────────────────────────────────────────────────
def build(con_kw=None, bt_kw=None, screen=None):
    cfg = ConstructionConfig(**{**BASE_CON, **(con_kw or {})})
    bt = BacktestConfig(**{**BASE_BT, **(bt_kw or {})})
    return Backtester(panel(), PortfolioConstructor(cfg), bt, screen=screen)


def run_one(spec, s, e):
    """-> (equity metrics, full-system metrics). None if the window is unusable."""
    ctx = use_rank_z() if spec.get("rank") else None
    if ctx:
        ctx.__enter__()
    try:
        bt = build(spec.get("con"), spec.get("bt"),
                   vol_screen(spec["volcut"]) if spec.get("volcut") else None)
        if spec.get("tranches"):
            r = tranched_run(bt, s, e, n_tranches=spec["tranches"],
                             stagger_bars=spec.get("stagger"))
        else:
            r = bt.run(s, e)
        return metrics(r["nav_net"]), metrics(wrap(r["nav_gross"])), r["metrics"]
    except Exception as ex:
        print(f"    ! {spec['name']} {s}: {type(ex).__name__}: {ex}")
        return None, None, None
    finally:
        if ctx:
            ctx.__exit__()


SPECS = [
    {"name": "BASELINE (deployed)"},
    {"name": "T   tranched x3 (P14)", "tranches": 3, "stagger": 42},
    {"name": "R   rank-transform", "rank": True},
    {"name": "V25 drop top-25% vol", "volcut": 0.25},
    {"name": "B4  buffer 4.0", "con": {"buffer_mult": 4.0}},
    {"name": "T+R", "tranches": 3, "stagger": 42, "rank": True},
    {"name": "T+V25", "tranches": 3, "stagger": 42, "volcut": 0.25},
    {"name": "T+R+V25", "tranches": 3, "stagger": 42, "rank": True, "volcut": 0.25},
    {"name": "T+R+B4", "tranches": 3, "stagger": 42, "rank": True,
     "con": {"buffer_mult": 4.0}},
]
# A risk lever judged on a RETURN win-count is being judged on the wrong axis.
# Each metric gets its own walk-forward win-count; "sign" says which direction
# counts as a win.
JUDGE = [("cagr", +1), ("sharpe_excess", +1), ("max_dd", +1), ("calmar", +1)]


def main():
    print(f"\n  panel: {len(panel().tickers)} symbols  |  {START} -> {END}")
    print(f"  bar: beat the deployed baseline in >=6/8 rolling 3-year windows,")
    print(f"       counted separately for each metric the lever actually targets.\n")

    windows = []
    for y0 in range(2016, 2024):
        s, e = f"{y0}-01-01", f"{y0+2}-12-31"
        if pd.Timestamp(e) > pd.Timestamp(END):
            e = END
        windows.append((s, e, f"{y0}-{y0+2}"))

    res = {}
    for spec in SPECS:
        eqf, syf, extra = run_one(spec, START, END)
        wf = []
        for s, e, lab in windows:
            eqw, syw, _ = run_one(spec, s, e)
            wf.append({"window": lab, "eq": eqw, "sy": syw})
        res[spec["name"]] = {"full_eq": eqf, "full_sy": syf,
                             "turnover": extra.get("turnover_yr") if extra else None,
                             "wf": wf}
        print(f"  ran {spec['name']}")

    base = res["BASELINE (deployed)"]

    def table(level, key, lvl):
        print("\n" + "=" * 118)
        print(f"  {level}")
        print("=" * 118)
        print(f"  {'config':<22}{'CAGR':>8}{'shExc':>7}{'vol':>7}{'MaxDD':>8}{'Calmar':>7}"
              f"{'turn':>6}   {'wf wins  CAGR':>14}{'Sharpe':>9}{'MaxDD':>8}{'Calmar':>8}")
        print("  " + "-" * 114)
        for name, r in res.items():
            m = r[key]
            if not m:
                continue
            cells = []
            for metric, _ in JUDGE:
                d = [w[lvl][metric] if w[lvl] else None for w in r["wf"]]
                b = [w[lvl][metric] if w[lvl] else None for w in base["wf"]]
                pairs = [(x, y) for x, y in zip(d, b) if x is not None and y is not None]
                wins = sum(1 for x, y in pairs if x > y)
                cells.append("" if name.startswith("BASELINE") else f"{wins}/{len(pairs)}")
            print(f"  {name:<22}{m['cagr']*100:>+7.2f}%{m['sharpe_excess']:>7.2f}"
                  f"{m['vol']*100:>6.1f}%{m['max_dd']*100:>+7.1f}%{m['calmar']:>7.2f}"
                  f"{(r['turnover'] or 0)*100:>5.0f}%   {cells[0]:>14}{cells[1]:>9}"
                  f"{cells[2]:>8}{cells[3]:>8}")

    table("EQUITY SLEEVE  (the book this repo builds)", "full_eq", "eq")
    table("FULL SYSTEM 50/25/25  (the published headline)", "full_sy", "sy")

    out = os.path.join(_ROOT, "reports", "edge_research_2026_07.json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(f"\n  saved -> {out}\n")


if __name__ == "__main__":
    main()

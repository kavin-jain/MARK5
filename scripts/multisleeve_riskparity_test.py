"""
MARK6 — Phase D: multi-sleeve risk-parity expansion (2026-06-10)
================================================================
The targets (CAGR>=20, Sharpe>=1.1, MaxDD<=25, Calmar>=0.8) require nearly DOUBLING
return-per-risk. That is a DIVERSIFICATION problem (vol must fall while return holds),
not a stock-signal problem. The only robustly-validated lever in this project is
"add uncorrelated positive-return sleeves, risk-weighted" (P7->P8). This extends it.

HONEST DATA RULE (no synthetic backfill):
  - Full period 2016-2026: only equity / gold / US-Nasdaq have real history.
  - Recent  2022-2026:      equity / gold / US / SILVER / long-gilt ALL have real
                            Indian-listed data -> the honest multi-sleeve test.

Sleeve weighting methods (all causal, annual reset):
  - fixed         : hand-set weights (current deployed = eq60/g20/us20)
  - inverse_vol   : weight inversely to trailing-1y sleeve vol (risk parity, naive)
  - inverse_vol_capped : risk parity but cap equity<=60% so it can't dominate
Equity book = momentum-heavy (shipped this session).

KEEP bar: an allocation only ships if it beats the deployed eq60/g20/us20 on the
SAME window, net of tax, AND does not rely on a sleeve's single-regime lucky run
(flagged explicitly). Recent-window wins are labelled regime-contingent.

  python3 scripts/multisleeve_riskparity_test.py
"""
import os, sys
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, load_ohlcv, metrics)

END = "2026-06-09"
TAX = 0.15
TD = 252
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}


def sleeve_ret(name, cal):
    df = load_ohlcv(name)
    if df is None:
        return None
    s = df["close"].astype(float).reindex(cal).ffill().bfill()
    return s.pct_change(fill_method=None).fillna(0.0)


def equity_ret(panel, start, end, cal):
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=12, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.125, factor_weights=MOM)
    nav = Backtester(panel, PortfolioConstructor(cfg)).run(start, end)["nav_net"]
    return nav.pct_change(fill_method=None).fillna(0.0).reindex(cal).fillna(0.0)


def blend(rets: dict, base_w: dict, method="fixed", eq_cap=None):
    """Annual-rebalanced multi-sleeve blend. Risk-parity resets weights yearly to
    inverse trailing-1y vol (causal: uses only data before the reset date)."""
    sleeves = list(base_w)
    cal = rets[sleeves[0]].index
    cur = {s: base_w[s] for s in sleeves}
    nav, out = 1.0, {}
    for i, d in enumerate(cal):
        if i > 0:
            prev = sum(cur.values())
            for s in sleeves:
                cur[s] *= (1 + rets[s].iloc[i])
            nav *= sum(cur.values()) / prev
        out[d] = nav
        if i > 0 and i % TD == 0:
            tot = sum(cur.values())
            if method == "fixed":
                w = dict(base_w)
            else:  # inverse_vol risk parity, causal trailing 1y
                iv = {}
                for s in sleeves:
                    v = rets[s].iloc[max(0, i - TD):i].std() * np.sqrt(TD)
                    iv[s] = 1.0 / v if v and v > 1e-6 else 0.0
                ssum = sum(iv.values()) or 1.0
                w = {s: iv[s] / ssum for s in sleeves}
                if eq_cap is not None and "eq" in w and w["eq"] > eq_cap:
                    # cap equity, redistribute excess pro-rata to the rest
                    excess = w["eq"] - eq_cap
                    w["eq"] = eq_cap
                    others = [s for s in sleeves if s != "eq"]
                    osum = sum(w[s] for s in others) or 1.0
                    for s in others:
                        w[s] += excess * w[s] / osum
            for s in sleeves:
                cur[s] = tot * w[s]
    nav_s = pd.Series(out)
    n = nav_s.copy()
    g = nav_s.iloc[-1] - 1
    n.iloc[-1] = nav_s.iloc[-1] - max(0, g) * TAX
    return metrics(n)


def show(title, rets, configs):
    print(f"\n{title}")
    print(f"  {'config':<30}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'Calmar':>8}")
    for name, (bw, method, cap) in configs.items():
        use = {s: rets[s] for s in bw}
        m = blend(use, bw, method, cap)
        hit = "  <-- hits target" if (m["cagr"] >= 0.20 and m["sharpe"] >= 1.1
                                      and m["max_dd"] >= -0.25 and m["calmar"] >= 0.8) else ""
        print(f"  {name:<30}{m['cagr']*100:>+7.1f}%{m['sharpe']:>8.2f}"
              f"{m['max_dd']*100:>+7.1f}%{m['calmar']:>8.2f}{hit}")


def main():
    panel = DataPanel(discover_tickers(), END)

    # ── FULL PERIOD 2016-2026: only 3 long-history sleeves are real ──────────
    start = "2016-01-01"
    cal = panel.trading_calendar(start, END)
    rets = {"eq": equity_ret(panel, start, END, cal),
            "gold": sleeve_ret("GOLDBEES", cal), "us": sleeve_ret("MON100", cal)}
    show("FULL 2016-2026 (real data: equity/gold/US only)", rets, {
        "deployed eq60/g20/us20":      ({"eq": .6, "gold": .2, "us": .2}, "fixed", None),
        "eq50/g25/us25":               ({"eq": .5, "gold": .25, "us": .25}, "fixed", None),
        "risk-parity (eq/gold/us)":    ({"eq": .5, "gold": .25, "us": .25}, "inverse_vol", None),
        "risk-parity eq-cap50":        ({"eq": .5, "gold": .25, "us": .25}, "inverse_vol", 0.50),
    })

    # ── RECENT 2022-2026: all 5 sleeves have real Indian data ────────────────
    start = "2022-01-01"
    cal = panel.trading_calendar(start, END)
    rets = {"eq": equity_ret(panel, start, END, cal),
            "gold": sleeve_ret("GOLDBEES", cal), "us": sleeve_ret("MON100", cal),
            "silver": sleeve_ret("SILVERBEES", cal), "gilt": sleeve_ret("LTGILTBEES", cal)}
    show("RECENT 2022-2026 (real data: 5 sleeves) — regime-contingent", rets, {
        "deployed eq60/g20/us20":          ({"eq": .6, "gold": .2, "us": .2}, "fixed", None),
        "5-sleeve fixed eq50":             ({"eq": .5, "gold": .15, "us": .15, "silver": .1, "gilt": .1}, "fixed", None),
        "5-sleeve risk-parity":            ({"eq": .4, "gold": .15, "us": .15, "silver": .15, "gilt": .15}, "inverse_vol", None),
        "5-sleeve risk-parity eq-cap50":   ({"eq": .4, "gold": .15, "us": .15, "silver": .15, "gilt": .15}, "inverse_vol", 0.50),
    })
    print("\n  NOTE: silver (data from 2022) + gilt (2018) cannot extend to the full")
    print("  period without synthetic backfill — recent-window only, by design.")


if __name__ == "__main__":
    main()

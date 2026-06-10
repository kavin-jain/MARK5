"""
MARK6 — Multi-Asset v2: raise Sharpe past 1.0 via global diversification + risk parity
======================================================================================
The institutional report's gap = Sharpe 0.88 (<1.0). Literature + our knowledge base
(Markowitz 'only free lunch'; Bridgewater risk parity) say the robust fix is MORE
uncorrelated positive-return sleeves, risk-weighted. Equity book stays INTACT; we only
improve the allocation around it.

Sleeves (all free, India-listed, full 2015-26 history):
  equity = MARK6 12-name factor book   gold = GOLDBEES   us = MON100 (Nasdaq-100)
Configs: current 80/20 vs adding US, fixed vs inverse-vol risk-parity. Net of tax.
Reports Sharpe/CAGR/MaxDD/Calmar, full + walk-forward (robustness).

  python3 scripts/multiasset_v2_test.py
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


def aret(name, cal):
    df = load_ohlcv(name)
    if df is None:
        return None
    return df["close"].astype(float).reindex(cal).ffill().bfill().pct_change(fill_method=None).fillna(0.0)


def net(nav):
    n = nav.copy(); g = nav.iloc[-1] - 1; n.iloc[-1] = nav.iloc[-1] - max(0, g) * TAX
    return metrics(n)


def blend(rets: dict, weights: dict, rebal=TD, risk_parity=False, lookback=TD):
    """Annually-rebalanced multi-sleeve blend. If risk_parity, weights are reset each
    year to inverse-trailing-vol (causal)."""
    cal = next(iter(rets.values())).index
    sleeves = list(weights)
    w = {s: weights[s] for s in sleeves}
    nav, out = 1.0, {}
    cur = dict(w)
    for i, d in enumerate(cal):
        if i > 0:
            prev = sum(cur.values())
            for s in sleeves:
                cur[s] *= (1 + rets[s].iloc[i])
            nav *= sum(cur.values()) / prev
        out[d] = nav
        if i > 0 and i % rebal == 0:
            tot = sum(cur.values())
            if risk_parity:
                iv = {}
                for s in sleeves:
                    v = rets[s].iloc[max(0, i - lookback):i].std() * np.sqrt(TD)
                    iv[s] = 1.0 / v if v > 1e-6 else 0.0
                ssum = sum(iv.values()) or 1.0
                w = {s: iv[s] / ssum for s in sleeves}
            cur = {s: tot * w[s] for s in sleeves}
    return pd.Series(out)


def main():
    panel = DataPanel(discover_tickers(), END)
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=12, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.125)
    eq_nav = Backtester(panel, PortfolioConstructor(cfg)).run("2016-01-01", END)["nav_gross"]
    cal = eq_nav.index
    r = {"eq": eq_nav.pct_change(fill_method=None).fillna(0.0),
         "gold": aret("GOLDBEES", cal), "us": aret("MON100", cal)}

    # correlations (the diversification evidence)
    print("Sleeve correlations (daily):")
    print(f"  eq-gold {r['eq'].corr(r['gold']):+.2f}  eq-us {r['eq'].corr(r['us']):+.2f}  "
          f"gold-us {r['gold'].corr(r['us']):+.2f}\n")

    configs = {
        "equity_only":      {"eq": 1.0},
        "eq80_gold20 (now)": {"eq": .80, "gold": .20},
        "eq70_g15_us15":    {"eq": .70, "gold": .15, "us": .15},
        "eq60_g20_us20":    {"eq": .60, "gold": .20, "us": .20},
        "eq50_g25_us25":    {"eq": .50, "gold": .25, "us": .25},
    }
    rp = {"eq": .34, "gold": .33, "us": .33}   # seed; risk-parity resets annually

    def navof(name, w, rp_flag=False):
        rr = {s: r[s] for s in w}
        return blend(rr, w, risk_parity=rp_flag)

    print("=" * 86)
    print("  FULL 2016-2026 (net of tax)")
    print("=" * 86)
    print(f"  {'config':<20}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>8}{'MaxDD':>8}{'Calmar':>8}")
    rows = list(configs.items()) + [("risk_parity_3", rp)]
    full = {}
    for name, w in rows:
        m = net(navof(name, w, rp_flag=(name == "risk_parity_3")))
        full[name] = m
        flag = "  <<< Sharpe>=1" if m["sharpe"] >= 1.0 else ""
        print(f"  {name:<20}{m['cagr']*100:>+7.1f}%{m['sharpe']:>8.2f}{m['sortino']:>8.2f}"
              f"{m['max_dd']*100:>+7.1f}%{m['calmar']:>8.2f}{flag}")

    print("\n" + "=" * 86)
    print("  ROLLING 3-YR WALK-FORWARD — avg Sharpe / avg CAGR / worst DD (robustness)")
    print("=" * 86)
    print(f"  {'config':<20}{'avgSharpe':>11}{'avgCAGR':>9}{'worstDD':>9}")
    for name, w in rows:
        sh, cg, dd = [], [], []
        for y in range(2016, 2024):
            s, e = f"{y}-01-01", min(f"{y+2}-12-31", END)
            sub = {k: v.loc[s:e] for k, v in r.items()}
            ww = {k: w[k] for k in w}
            nav = blend({k: sub[k] for k in ww}, ww, risk_parity=(name == "risk_parity_3"))
            m = net(nav)
            if m:
                sh.append(m["sharpe"]); cg.append(m["cagr"]); dd.append(m["max_dd"])
        print(f"  {name:<20}{np.mean(sh):>11.2f}{np.mean(cg)*100:>+8.1f}%{np.min(dd)*100:>+8.1f}%")

    print("\n  US (MON100) ~26%/yr 2015-26 is a strong but regime-dependent return; the")
    print("  diversification (low corr) Sharpe benefit is the robust part. Caveat noted.")


if __name__ == "__main__":
    main()

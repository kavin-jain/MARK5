"""
MARK6 — Multi-Asset + Volatility-Targeting Test
===============================================
Attacks the system's real weakness: the ~-35% drawdown. Two grounded, untested
levers (no HFT, no new paid data):
  (1) MULTI-ASSET: blend the MARK6 equity book with GOLD (GOLDBEES) +/- CASH
      (LIQUIDBEES). Gold rallies in equity crashes -> lower drawdown (All-Weather).
  (2) VOL-TARGETING: scale equity exposure to hold ~constant volatility; de-risk
      into cash when realized vol spikes (managed-futures risk control, NOT timing).

All net of a fair terminal tax haircut. Reported on the FULL window AND every
rolling 3-yr walk-forward window (robustness — "run twice"). KEEP a variant only if
it improves risk-adjusted return (Sharpe/Calmar) or cuts drawdown without gutting CAGR.

  python3 scripts/multiasset_voltarget_test.py
"""
import os, sys
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, load_ohlcv, metrics)
END = "2026-06-05"
TAX = 0.15
TRADING = 252


def asset_rets(name, cal):
    df = load_ohlcv(name)
    if df is None:
        return None
    return df["close"].astype(float).reindex(cal).ffill().pct_change(fill_method=None).fillna(0.0)


def net_metrics(nav):
    net = nav.copy()
    g = nav.iloc[-1] - 1.0
    net.iloc[-1] = nav.iloc[-1] - max(0.0, g) * TAX
    return metrics(net)


def combine(r_eq, r_gold, r_cash, w_eq, w_gold, rebal=TRADING):
    """Annually-rebalanced fixed-weight blend (weights drift between rebalances)."""
    w_cash = 1.0 - w_eq - w_gold
    nav = 1.0
    we, wg, wc = w_eq, w_gold, w_cash
    out = {}
    for i, d in enumerate(r_eq.index):
        if i > 0:
            re_, rg, rc = r_eq.iloc[i], r_gold.iloc[i], r_cash.iloc[i]
            we *= (1 + re_); wg *= (1 + rg); wc *= (1 + rc)
            tot = we + wg + wc
            nav *= tot / (we_prev + wg_prev + wc_prev) if (we_prev+wg_prev+wc_prev) else 1
        we_prev, wg_prev, wc_prev = we, wg, wc
        out[d] = nav
        if i % rebal == 0 and i > 0:           # annual rebalance back to targets
            tot = we + wg + wc
            we, wg, wc = tot*w_eq, tot*w_gold, tot*w_cash
            we_prev, wg_prev, wc_prev = we, wg, wc
    return pd.Series(out)


def vol_target(r_eq, r_cash, target=0.15, lookback=63):
    """Scale equity exposure to target vol (cap 1.0 = NO leverage); rest in cash."""
    rv = r_eq.rolling(lookback, min_periods=20).std() * np.sqrt(TRADING)
    w = (target / rv).clip(upper=1.0).shift(1).fillna(1.0)   # shift -> causal
    blended = w * r_eq + (1 - w) * r_cash
    return (1 + blended).cumprod()


def run_window(panel, cfg, s, e):
    bt = Backtester(panel, PortfolioConstructor(cfg))
    run = bt.run(s, e)
    nav_eq = run["nav_gross"]
    cal = nav_eq.index
    r_eq = nav_eq.pct_change(fill_method=None).fillna(0.0)
    r_gold = asset_rets("GOLDBEES", cal)
    r_cash = asset_rets("LIQUIDBEES", cal)
    if r_gold is None:
        r_gold = pd.Series(0.0, index=cal)
    if r_cash is None:
        r_cash = pd.Series(0.06/252, index=cal)
    variants = {
        "equity_only":      net_metrics(nav_eq),
        "eq80_gold20":      net_metrics(combine(r_eq, r_gold, r_cash, 0.80, 0.20)),
        "eq70_gold30":      net_metrics(combine(r_eq, r_gold, r_cash, 0.70, 0.30)),
        "eq60_gold40":      net_metrics(combine(r_eq, r_gold, r_cash, 0.60, 0.40)),
        "voltarget15":      net_metrics(vol_target(r_eq, r_cash, 0.15)),
        "eq70_gold30+vt":   net_metrics(combine(vol_target(r_eq, r_cash, 0.15).pct_change().fillna(0.0),
                                                r_gold, r_cash, 0.70, 0.30)),
    }
    return variants


def main():
    panel = DataPanel(discover_tickers(), END)
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=12, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.125)
    names = ["equity_only", "eq80_gold20", "eq70_gold30", "eq60_gold40", "voltarget15", "eq70_gold30+vt"]

    print("=" * 96)
    print("  FULL 2016-2026 — net of tax")
    print("=" * 96)
    print(f"  {'variant':<18}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>8}{'MaxDD':>8}{'Calmar':>8}")
    full = run_window(panel, cfg, "2016-01-01", END)
    for n in names:
        m = full[n]
        print(f"  {n:<18}{m['cagr']*100:>+7.1f}%{m['sharpe']:>8.2f}{m['sortino']:>8.2f}"
              f"{m['max_dd']*100:>+7.1f}%{m['calmar']:>8.2f}")

    print("\n" + "=" * 96)
    print("  ROLLING 3-YR WALK-FORWARD — avg net CAGR / avg Calmar / worst DD (robustness)")
    print("=" * 96)
    agg = {n: {"cagr": [], "calmar": [], "dd": []} for n in names}
    for y in range(2016, 2024):
        w = run_window(panel, cfg, f"{y}-01-01", min(f"{y+2}-12-31", END))
        for n in names:
            agg[n]["cagr"].append(w[n]["cagr"]); agg[n]["calmar"].append(w[n]["calmar"])
            agg[n]["dd"].append(w[n]["max_dd"])
    print(f"  {'variant':<18}{'avgCAGR':>9}{'avgCalmar':>11}{'worstDD':>9}")
    for n in names:
        print(f"  {n:<18}{np.mean(agg[n]['cagr'])*100:>+8.1f}%{np.mean(agg[n]['calmar']):>11.2f}"
              f"{np.min(agg[n]['dd'])*100:>+8.1f}%")

    print("\n  Read: if a blend lifts Calmar/Sharpe & cuts worstDD vs equity_only at similar CAGR,")
    print("  it's a real risk-adjusted improvement (the honest 'better portfolio' win).")


if __name__ == "__main__":
    main()

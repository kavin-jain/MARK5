"""
MARK6 — Efficiency Research: honest-tax netting, tax-loss harvesting, faster knobs
==================================================================================
Researches the OPEN levers for making the deployed system more efficient,
per RESEARCH_LOG (K3 already proved raw shorter holding = monotonically worse):

  A. baseline        : current engine (per-trade tax, losses give NO credit)
  B. FY netting      : same trades, REAL Indian tax — losses offset gains within
                       the fiscal year (STCL vs any; LTCL vs LTCG; 8-yr carryforward)
  C. TLH             : B + tax-loss harvesting — monthly check, sell any holding
                       below a loss threshold and rebuy immediately (India has no
                       wash-sale rule), banking the loss credit
  D. semi-annual     : B with 126-bar rebalance (re-test K3's plateau under
                       honest netting — netting reduces the turnover tax penalty)
  E. FIP momentum    : information-discreteness (frog-in-the-pan) as a small
                       momentum-quality component (Da-Gurun-Warachka 2014)
  F. sleeve freq     : wrapper-level rebalance frequency for the 50/25/25 blend
                       (rebalancing premium between ~uncorrelated sleeves)

All net of tax & costs. Full period + rolling 3-yr walk-forward for finalists.

  python3 scripts/efficiency_research.py
"""
import os, sys
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, load_ohlcv, metrics)
from core.portfolio.backtest import BacktestConfig, TRADING_DAYS
from core.portfolio.factors import FactorLibrary

END = "2026-06-09"
START = "2016-01-01"
TD = 252
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}
MOM_FIP = {"momentum": .35, "fip": .10, "low_vol": .15, "trend": .25, "stability": .15}


# ──────────────────────────────────────────────────────────────────────────────
# Netting backtester: identical trade logic, honest Indian FY tax netting,
# optional monthly tax-loss harvesting.
# ──────────────────────────────────────────────────────────────────────────────
class NettingBacktester(Backtester):
    def __init__(self, *a, tlh_threshold: float | None = None,
                 tlh_check_bars: int = 21, **kw):
        super().__init__(*a, **kw)
        self.tlh_threshold = tlh_threshold      # e.g. -0.07 → harvest at -7%
        self.tlh_check_bars = tlh_check_bars

    def run(self, start: str, end: str) -> dict:
        cfg = self.cfg
        cal = self.panel.trading_calendar(start, end)
        prices = self.panel.close.loc[start:end].reindex(cal).ffill(limit=5)
        rets = prices.ffill().pct_change(fill_method=None)
        tickers = list(prices.columns)

        pos = {t: 0.0 for t in tickers}
        basis = {t: 0.0 for t in tickers}
        entry = {t: None for t in tickers}
        cash, traded = 1.0, 0.0
        nav_hist, weights_hist = {}, {}
        last_rebal, n_rebal = -10**9, 0
        # FY tax ledger (Indian fiscal year Apr 1 – Mar 31)
        fy = {"stcg": 0.0, "ltcg": 0.0}
        cf_stcl, cf_ltcl = 0.0, 0.0            # carried-forward losses (stored >0)
        tax_paid, harvests = 0.0, 0

        def fy_of(d):  # fiscal year key: FY ending Mar of year X -> X
            return d.year + (1 if d.month >= 4 else 0)

        cur_fy = fy_of(cal[0])

        def settle_fy(today=None):
            """Net the FY ledger per Indian rules, deduct tax from cash.
            If cash can't cover the tax, sell positions pro-rata to fund it
            (no implicit leverage — keeps the comparison vs engine-A honest)."""
            nonlocal cash, tax_paid, cf_stcl, cf_ltcl, fy
            st, lt = fy["stcg"], fy["ltcg"]
            # apply carryforward losses (stored positive)
            stl = cf_stcl + max(0.0, -st)       # this-yr STCL joins carryforward pool
            st = max(0.0, st)
            ltl = cf_ltcl + max(0.0, -lt)
            lt = max(0.0, lt)
            use = min(stl, st); st -= use; stl -= use       # STCL vs STCG first
            use = min(stl, lt); lt -= use; stl -= use       # then STCL vs LTCG
            use = min(ltl, lt); lt -= use; ltl -= use       # LTCL vs LTCG only
            tax = st * cfg.stcg + lt * cfg.ltcg
            cash -= tax
            tax_paid += tax
            cf_stcl, cf_ltcl = stl, ltl
            fy = {"stcg": 0.0, "ltcg": 0.0}
            if cash < -1e-12 and today is not None:
                tot = sum(pos.values())
                if tot > 0:
                    need = -cash
                    for t in list(pos.keys()):
                        if pos[t] > 0:
                            cash += realize(t, pos[t] * min(1.0, need / tot), today)

        def realize(t, sell_val, today):
            """Sell; book gain/loss to FY ledger (no immediate tax)."""
            nonlocal traded
            if pos[t] <= 0:
                return 0.0
            frac = min(1.0, sell_val / pos[t])
            gain = sell_val - basis[t] * frac
            held = (today - entry[t]).days if entry[t] is not None else 0
            fy["ltcg" if held > cfg.ltcg_days else "stcg"] += gain
            traded += sell_val
            pos[t] -= sell_val
            basis[t] -= basis[t] * frac
            if pos[t] < 1e-12:
                pos[t], basis[t], entry[t] = 0.0, 0.0, None
            return sell_val - sell_val * (cfg.cost_pct / 2 + cfg.slippage_pct)

        def buy(t, buy_val, today):
            nonlocal cash, traded
            cash -= buy_val + buy_val * (cfg.cost_pct / 2 + cfg.slippage_pct)
            traded += buy_val
            if entry[t] is None or pos[t] <= 0:
                entry[t] = today
            else:
                wold = pos[t] / (pos[t] + buy_val)
                entry[t] = pd.Timestamp(int(wold * entry[t].value + (1 - wold) * today.value))
            pos[t] += buy_val
            basis[t] += buy_val

        ret_rows = rets.values
        col = {t: i for i, t in enumerate(tickers)}
        for i, d in enumerate(cal):
            if i > 0:
                r = ret_rows[i]
                for t in tickers:
                    v = pos[t]
                    if v > 0:
                        rt = r[col[t]]
                        if rt == rt:
                            pos[t] = v * (1 + rt)
            # FY rollover → settle netted tax
            if fy_of(d) != cur_fy:
                settle_fy(d)
                cur_fy = fy_of(d)
            # tax-loss harvest check (monthly): sell loser + rebuy instantly.
            # Skip right before a rebalance (within 21 bars) — pointless churn.
            if (self.tlh_threshold is not None and i % self.tlh_check_bars == 0
                    and (i - last_rebal) < cfg.rebal_bars - 21 and i > 0):
                nonlocal_names = [t for t in tickers if pos[t] > 0 and basis[t] > 0]
                for t in nonlocal_names:
                    unreal = (pos[t] - basis[t]) / basis[t]
                    if unreal < self.tlh_threshold:
                        val = pos[t]
                        proceeds = realize(t, val, d)
                        cash += proceeds
                        rebuy = min(proceeds, cash)
                        if rebuy > 0:
                            buy(t, rebuy, d)
                        harvests += 1
            if (i - last_rebal) >= cfg.rebal_bars:
                n_rebal += 1
                last_rebal = i
                if n_rebal > cfg.warmup_skip:
                    elig = self.panel.eligible(d, cfg.min_history, cfg.liquidity_pct)
                    elig = [t for t in elig if t in col]
                    if self.screen is not None and elig:
                        elig = self.screen(d, elig) or elig
                    if elig:
                        comp, vol = self._factor_panel(d, elig)
                        held = [t for t in tickers if pos[t] > 0]
                        tw = self.con.target_weights(comp, vol, held)
                        nav = cash + sum(pos.values())
                        target = {t: nav * w for t, w in tw.items()}
                        for t in tickers:
                            tgt = target.get(t, 0.0)
                            if pos[t] > tgt + 1e-9:
                                cash += realize(t, pos[t] - tgt, d)
                        for t, tgt in target.items():
                            if tgt > pos[t] + 1e-9:
                                buy(t, tgt - pos[t], d)
                        weights_hist[d] = tw
            nav_hist[d] = cash + sum(pos.values())

        # terminal liquidation into the FY ledger, then settle
        last = cal[-1]
        for t in tickers:
            if pos[t] > 0:
                cash += realize(t, pos[t], last)
        settle_fy()
        nav_hist[last] = cash
        nav = pd.Series(nav_hist)
        yrs = (cal[-1] - cal[0]).days / 365.25
        m = metrics(nav)
        m.update({"turnover_yr": traded / yrs / nav.mean(), "tax_paid": tax_paid,
                  "n_rebalances": n_rebal, "harvests": harvests})
        return {"nav_net": nav, "nav_gross": nav, "metrics": m, "weights": weights_hist}


# ── FIP (information discreteness) extra factor ──────────────────────────────
def fip_factors(panel: DataPanel) -> dict:
    """ID = sign(formation ret) * (%neg - %pos days) over the 12-1 formation
    window. CONTINUOUS info (low ID) = better momentum (Da-Gurun-Warachka).
    Factor returned as -ID so higher = better."""
    out = {}
    for t in panel.tickers:
        c = panel.close[t].dropna()
        if len(c) < 300:
            continue
        ret = c.pct_change(fill_method=None)
        # formation window: t-252 .. t-21
        form_ret = c.shift(21) / c.shift(252) - 1.0
        sgn = np.sign(form_ret)
        pos_d = (ret > 0).rolling(231).mean().shift(21)
        neg_d = (ret < 0).rolling(231).mean().shift(21)
        fip = -(sgn * (neg_d - pos_d))          # higher = more continuous
        out[t] = pd.DataFrame({"fip": fip})
    return out


# ── wrapper (sleeve blend) with configurable rebalance freq + tax ────────────
def sleeve_ret(name, cal):
    df = load_ohlcv(name)
    s = df["close"].astype(float).reindex(cal).ffill().bfill()
    return s.pct_change(fill_method=None).fillna(0.0)


def wrap(eq_ret, gold, us, rebal_bars=252, w=(0.50, 0.25, 0.25), tax=0.15):
    """3-sleeve wrapper. Rebalance every `rebal_bars`; realized rebalance gains
    taxed at 20% STCG if rebal_bars < 252 else 12.5% LTCG (approx); terminal 12.5%."""
    rate = 0.20 if rebal_bars < 252 else 0.125
    rets = [eq_ret.values, gold.values, us.values]
    weights = list(w)
    cal = eq_ret.index
    cur = list(weights)
    cb = list(weights)                          # cost basis per sleeve
    nav, out, tax_acc = 1.0, {}, 0.0
    for i, d in enumerate(cal):
        if i > 0:
            prev = sum(cur)
            for s in range(3):
                cur[s] *= (1 + rets[s][i])
            nav *= sum(cur) / prev
        out[d] = nav
        if i > 0 and i % rebal_bars == 0:
            tot = sum(cur)
            for s in range(3):
                tgt = tot * weights[s]
                if cur[s] > tgt:                # selling: realize gain pro-rata
                    frac = (cur[s] - tgt) / cur[s]
                    gain = (cur[s] - cb[s]) * frac
                    tax_acc += max(0.0, gain) * rate
                    cb[s] *= (1 - frac)
                cb[s] += max(0.0, tgt - cur[s])
                cur[s] = tgt
            # costs on turnover
            turn = sum(abs(tot * weights[s] - c) for s, c in enumerate(cur))
            nav -= nav * 0.001 * (turn / tot if tot > 0 else 0)
    nav_s = pd.Series(out)
    g = nav_s.iloc[-1] - 1
    term = max(0.0, g) * 0.125 + tax_acc        # terminal LTCG + accrued rebal tax
    n = nav_s.copy()
    n.iloc[-1] = nav_s.iloc[-1] - term
    return n


def report(label, m, extra=""):
    print(f"  {label:<34}{m['cagr']*100:>+7.1f}%{m['sharpe']:>7.2f}"
          f"{m['max_dd']*100:>+8.1f}%{m['calmar']:>7.2f}  {extra}")


def main():
    panel = DataPanel(discover_tickers(), END)
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=12, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.125, factor_weights=MOM)

    print(f"\n══ EQUITY SLEEVE (full {START}→{END}, net of tax+costs) ══")
    print(f"  {'config':<34}{'CAGR':>8}{'Sharpe':>7}{'MaxDD':>9}{'Calmar':>7}")

    # A. baseline (current engine: per-trade tax, no loss credit)
    a = Backtester(panel, PortfolioConstructor(cfg)).run(START, END)
    report("A baseline (no loss credit)", a["metrics"],
           f"tax={a['metrics']['tax_paid']:.3f}")

    # B. FY netting, same trades
    b = NettingBacktester(panel, PortfolioConstructor(cfg)).run(START, END)
    report("B FY netting (honest tax)", b["metrics"],
           f"tax={b['metrics']['tax_paid']:.3f}")

    # C. netting + TLH at -7% and -12%, monthly
    cs = {}
    for thr in (-0.07, -0.12):
        c = NettingBacktester(panel, PortfolioConstructor(cfg),
                              tlh_threshold=thr).run(START, END)
        cs[thr] = c
        report(f"C TLH {int(thr*100)}% (netting+harvest)", c["metrics"],
               f"harvests={c['metrics']['harvests']} tax={c['metrics']['tax_paid']:.3f}")

    # D. semi-annual rebalance under netting
    d = NettingBacktester(panel, PortfolioConstructor(cfg),
                          config=BacktestConfig(rebal_bars=126)).run(START, END)
    report("D semi-annual rebal (netting)", d["metrics"],
           f"turn={d['metrics']['turnover_yr']:.0%} tax={d['metrics']['tax_paid']:.3f}")

    # E. FIP momentum-quality component (per-trade engine for comparability w/ A)
    fcfg = ConstructionConfig(mode="factor_tilt", n_hold=12, base_weighting="inverse_vol",
                              tilt_strength=1.5, max_weight=0.125, factor_weights=MOM_FIP)
    e = Backtester(panel, PortfolioConstructor(fcfg),
                   extra_factors=fip_factors(panel)).run(START, END)
    report("E FIP 10% (vs A)", e["metrics"])

    # ── F. wrapper sleeve-rebalance frequency (uses B's gross equity curve) ──
    print(f"\n══ WRAPPER 50/25/25 sleeve-rebalance frequency (gross eq = B) ══")
    print(f"  {'config':<34}{'CAGR':>8}{'Sharpe':>7}{'MaxDD':>9}{'Calmar':>7}")
    cal = b["nav_net"].index
    eqr = b["nav_gross"].pct_change(fill_method=None).fillna(0.0)
    gold, us = sleeve_ret("GOLDBEES", cal), sleeve_ret("MON100", cal)
    for rb, lab in [(63, "F quarterly sleeves (STCG)"),
                    (126, "F semi-annual sleeves (STCG)"),
                    (252, "F annual sleeves (LTCG, current)")]:
        nav = wrap(eqr, gold, us, rebal_bars=rb)
        report(lab, metrics(nav))

    # ── walk-forward: B(annual-net) vs D126 vs D63 vs E(FIP) ────────────────
    print(f"\n══ ROLLING 3-YR WALK-FORWARD (equity sleeve, net) ══")
    print(f"  window     B-net12m  D-net6m   D-net3m   E-FIP     D6-B    E-B")
    fipf = fip_factors(panel)
    d_b, d3_b, e_b = [], [], []
    for y in range(2016, 2024):
        s, e2 = f"{y}-01-01", min(f"{y+2}-12-31", END)
        mb = NettingBacktester(panel, PortfolioConstructor(cfg)).run(s, e2)["metrics"]
        md = NettingBacktester(panel, PortfolioConstructor(cfg),
                               config=BacktestConfig(rebal_bars=126)).run(s, e2)["metrics"]
        md3 = NettingBacktester(panel, PortfolioConstructor(cfg),
                                config=BacktestConfig(rebal_bars=63)).run(s, e2)["metrics"]
        me = NettingBacktester(panel, PortfolioConstructor(fcfg),
                               extra_factors=fipf).run(s, e2)["metrics"]
        d_b.append(md["cagr"] - mb["cagr"])
        d3_b.append(md3["cagr"] - mb["cagr"])
        e_b.append(me["cagr"] - mb["cagr"])
        print(f"  {y}-{y+2}: {mb['cagr']*100:+7.1f}% {md['cagr']*100:+8.1f}% "
              f"{md3['cagr']*100:+8.1f}% {me['cagr']*100:+8.1f}% "
              f"{100*d_b[-1]:+6.1f}pp {100*e_b[-1]:+5.1f}pp")
    for lab, dd in [("semi-annual (D126)", d_b), ("quarterly (D63)", d3_b), ("FIP (E)", e_b)]:
        print(f"  => {lab}: {np.mean(dd)*100:+.2f}pp avg vs B, "
              f"beats {sum(1 for x in dd if x > 0.001)}/8 windows, "
              f"worst {min(dd)*100:+.1f}pp")

    # ── G. FULL SYSTEM: 50/25/25 wrapper, annual vs semi-annual equity sleeve ─
    print(f"\n══ G. FULL SYSTEM (50/25/25 wrapper, annual sleeves) ══")
    print(f"  {'config':<34}{'CAGR':>8}{'Sharpe':>7}{'MaxDD':>9}{'Calmar':>7}")
    eqr_d = d["nav_gross"].pct_change(fill_method=None).fillna(0.0)
    nav_b = wrap(eqr, gold, us)                      # B: annual equity (current)
    nav_d = wrap(eqr_d, gold, us)                    # D: semi-annual equity
    report("G wrapper + annual eq (current)", metrics(nav_b))
    report("G wrapper + semi-annual eq", metrics(nav_d))

    print(f"\n  wrapper-level walk-forward (3-yr windows):")
    wf = []
    for y in range(2016, 2024):
        s, e2 = f"{y}-01-01", min(f"{y+2}-12-31", END)
        rb = NettingBacktester(panel, PortfolioConstructor(cfg)).run(s, e2)
        rd = NettingBacktester(panel, PortfolioConstructor(cfg),
                               config=BacktestConfig(rebal_bars=126)).run(s, e2)
        calw = rb["nav_net"].index
        gw, uw = sleeve_ret("GOLDBEES", calw), sleeve_ret("MON100", calw)
        mb_ = metrics(wrap(rb["nav_gross"].pct_change(fill_method=None).fillna(0.0), gw, uw))
        md_ = metrics(wrap(rd["nav_gross"].pct_change(fill_method=None).fillna(0.0), gw, uw))
        wf.append((md_["cagr"] - mb_["cagr"], md_["sharpe"] - mb_["sharpe"],
                   md_["max_dd"] - mb_["max_dd"]))
        print(f"  {y}-{y+2}: annual {mb_['cagr']*100:+6.1f}% (Sh {mb_['sharpe']:.2f}) | "
              f"semi {md_['cagr']*100:+6.1f}% (Sh {md_['sharpe']:.2f}) | "
              f"Δ {100*wf[-1][0]:+5.1f}pp")
    dc = [x[0] for x in wf]
    print(f"  => FULL SYSTEM semi-annual: {np.mean(dc)*100:+.2f}pp avg, "
          f"beats {sum(1 for x in dc if x > 0.001)}/8, worst {min(dc)*100:+.1f}pp, "
          f"avg ΔSharpe {np.mean([x[1] for x in wf]):+.2f}")


if __name__ == "__main__":
    main()

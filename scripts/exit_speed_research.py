"""
MARK6 — Exit-Speed Research (post-P11/P12, honest FY-netting tax engine)
========================================================================
Re-does the holding-period question PRECISELY now that the tax model is correct
(K3's 2026-06-08 sweep used the no-loss-credit engine and is invalid):

  SWEEP A — symmetric full-rebalance frequency: 21/42/63/126/189/252 bars.
  SWEEP B — ASYMMETRIC exits: full rebalance stays at 126 bars, but every
            `check_bars` we re-rank and SELL any holding that fell below the
            exit rank (deranked = fading momentum), replacing it with the
            top-ranked name not held. Entries stay slow, exits get fast.
            Variants: check every 21/42/63 bars × exit_rank 24 (=buffer) / 18.

Every config: full period 2016-2026 + rolling 3-yr walk-forward (8 windows)
for the finalists. Net of Indian tax (FY netting) + costs + slippage.
Reports avg hold days per config so the "how fast do we cut" question gets a
precise, evidence-graded answer.

  python3 scripts/exit_speed_research.py
"""
import os, sys
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, metrics)
from core.portfolio.backtest import BacktestConfig

END = "2026-06-09"
START = "2016-01-01"
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}


def make_cfg():
    return ConstructionConfig(mode="factor_tilt", n_hold=12, base_weighting="inverse_vol",
                              tilt_strength=1.5, max_weight=0.125, factor_weights=MOM)


class AsymmetricExitBacktester(Backtester):
    """P12 engine + fast derank-exit checks between full rebalances.

    Every `check_bars`, holdings are re-ranked against the point-in-time
    eligible universe; any holding whose composite rank fell to/below
    `exit_rank` is sold in full and the proceeds buy the best-ranked name(s)
    not currently held (equal split). Full reweight only at `rebal_bars`.
    """

    def __init__(self, *a, check_bars: int = 21, exit_rank: int = 24, **kw):
        super().__init__(*a, **kw)
        self.check_bars = check_bars
        self.exit_rank = exit_rank

    def run(self, start: str, end: str) -> dict:
        cfg = self.cfg
        cal = self.panel.trading_calendar(start, end)
        prices = self.panel.close.loc[start:end].reindex(cal).ffill(limit=5)
        rets = prices.ffill().pct_change(fill_method=None)
        tickers = list(prices.columns)

        pos = {t: 0.0 for t in tickers}
        basis = {t: 0.0 for t in tickers}
        entry = {t: None for t in tickers}
        cash, tax_paid, traded = 1.0, 0.0, 0.0
        nav_hist, weights_hist = {}, {}
        trades = []
        last_rebal, n_rebal = -10**9, 0
        fy = {"stcg": 0.0, "ltcg": 0.0}
        cf_stcl, cf_ltcl = 0.0, 0.0

        def fy_of(d):
            return d.year + (1 if d.month >= 4 else 0)

        cur_fy = fy_of(cal[0])

        def net_tax(st, lt, stl, ltl):
            stl += max(0.0, -st); st = max(0.0, st)
            ltl += max(0.0, -lt); lt = max(0.0, lt)
            use = min(stl, st); st -= use; stl -= use
            use = min(stl, lt); lt -= use; stl -= use
            use = min(ltl, lt); lt -= use; ltl -= use
            return st * cfg.stcg + lt * cfg.ltcg, stl, ltl

        def settle_fy(today):
            nonlocal cash, tax_paid, cf_stcl, cf_ltcl, fy
            tax, cf_stcl, cf_ltcl = net_tax(fy["stcg"], fy["ltcg"], cf_stcl, cf_ltcl)
            fy = {"stcg": 0.0, "ltcg": 0.0}
            if tax <= 0:
                return
            cash -= tax
            tax_paid += tax
            if cash < -1e-12:
                tot = sum(pos.values())
                if tot > 0:
                    short = -cash
                    for t in tickers:
                        if pos[t] > 0:
                            cash += realize(t, pos[t] * min(1.0, short / tot), today)

        def realize(t, sell_val, today):
            nonlocal tax_paid, traded
            if pos[t] <= 0:
                return 0.0
            frac = min(1.0, sell_val / pos[t])
            gain = sell_val - basis[t] * frac
            held = (today - entry[t]).days if entry[t] is not None else 0
            fy["ltcg" if held > cfg.ltcg_days else "stcg"] += gain
            traded += sell_val
            trades.append({"date": today, "ticker": t, "side": "SELL",
                           "gain": float(gain), "held_days": int(held),
                           "term": "LTCG" if held > cfg.ltcg_days else "STCG"})
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
            if fy_of(d) != cur_fy:
                settle_fy(d)
                cur_fy = fy_of(d)

            at_rebal = (i - last_rebal) >= cfg.rebal_bars
            at_check = (not at_rebal and i % self.check_bars == 0
                        and any(v > 0 for v in pos.values()))

            if at_check:
                # fast derank-exit: sell faders, replace with leaders
                elig = self.panel.eligible(d, cfg.min_history, cfg.liquidity_pct)
                elig = [t for t in elig if t in col]
                if elig:
                    comp, _ = self._factor_panel(d, elig)
                    ranked = comp.sort_values(ascending=False)
                    rank_of = {t: j for j, t in enumerate(ranked.index)}
                    held = [t for t in tickers if pos[t] > 0]
                    exits = [t for t in held
                             if rank_of.get(t, 10**9) >= self.exit_rank]
                    if exits:
                        proceeds = 0.0
                        for t in exits:
                            proceeds += realize(t, pos[t], d)
                        cash += proceeds
                        adds = [t for t in ranked.index
                                if pos[t] <= 0 and t not in exits][:len(exits)]
                        if adds:
                            spend = min(cash, proceeds)
                            for t in adds:
                                buy(t, spend / len(adds), d)

            if at_rebal:
                n_rebal += 1
                last_rebal = i
                if n_rebal > cfg.warmup_skip:
                    elig = self.panel.eligible(d, cfg.min_history, cfg.liquidity_pct)
                    elig = [t for t in elig if t in col]
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

        last = cal[-1]
        st, lt = fy["stcg"], fy["ltcg"]
        for t in tickers:
            if pos[t] > 0:
                g = pos[t] - basis[t]
                if entry[t] is not None and (last - entry[t]).days > cfg.ltcg_days:
                    lt += g
                else:
                    st += g
        term_tax, _, _ = net_tax(st, lt, cf_stcl, cf_ltcl)
        nav = pd.Series(nav_hist)
        net = nav.copy()
        net.iloc[-1] = nav.iloc[-1] - term_tax
        yrs = (cal[-1] - cal[0]).days / 365.25
        m = metrics(net)
        sells = [t for t in trades if t["side"] == "SELL"]
        m.update({"turnover_yr": traded / yrs / nav.mean(),
                  "tax_paid": tax_paid + term_tax,
                  "avg_hold": np.mean([t["held_days"] for t in sells]) if sells else 0})
        return {"nav_net": net, "nav_gross": nav, "metrics": m, "trades": trades}


def hold_days(run):
    sells = [t for t in run.get("trades", []) if t.get("side") == "SELL"]
    return np.mean([t["held_days"] for t in sells]) if sells else float("nan")


def row(label, m, hold=None):
    h = f"{hold:>5.0f}d" if hold == hold and hold is not None else "    —"
    print(f"  {label:<36}{m['cagr']*100:>+7.1f}%{m['sharpe']:>7.2f}"
          f"{m['max_dd']*100:>+8.1f}%{m['calmar']:>7.2f}"
          f"{m['turnover_yr']*100:>7.0f}%  {h}")


def main():
    panel = DataPanel(discover_tickers(), END)
    cfg = make_cfg()

    # ── SWEEP A: symmetric rebalance frequency ───────────────────────────────
    print(f"\n══ SWEEP A — full-rebalance frequency (FY-netting tax, {START}→{END}) ══")
    print(f"  {'config':<36}{'CAGR':>8}{'Sharpe':>7}{'MaxDD':>9}{'Calmar':>7}{'turn':>7}{'hold':>7}")
    for rb in (21, 42, 63, 126, 189, 252):
        r = Backtester(panel, PortfolioConstructor(cfg),
                       BacktestConfig(rebal_bars=rb)).run(START, END)
        row(f"A rebal {rb}d (~{rb//21}mo)", r["metrics"], hold_days(r))

    # ── SWEEP B: asymmetric — slow entries (126d), fast derank exits ─────────
    print(f"\n══ SWEEP B — asymmetric: 126d rebalance + fast derank-exit checks ══")
    print(f"  {'config':<36}{'CAGR':>8}{'Sharpe':>7}{'MaxDD':>9}{'Calmar':>7}{'turn':>7}{'hold':>7}")
    b_runs = {}
    for chk in (21, 42, 63):
        for xr in (24, 18):
            r = AsymmetricExitBacktester(panel, PortfolioConstructor(cfg),
                                         BacktestConfig(rebal_bars=126),
                                         check_bars=chk, exit_rank=xr).run(START, END)
            b_runs[(chk, xr)] = r
            row(f"B check {chk}d, exit_rank {xr}", r["metrics"], r["metrics"]["avg_hold"])

    # ── walk-forward: P12 baseline vs the two best B variants ────────────────
    best = sorted(b_runs, key=lambda k: -b_runs[k]["metrics"]["cagr"])[:2]
    print(f"\n══ ROLLING 3-YR WALK-FORWARD — P12(126d) vs best asymmetric variants ══")
    hdr = "  window     P12-126d "
    for k in best:
        hdr += f"  B{k[0]}d/x{k[1]}  "
    print(hdr)
    diffs = {k: [] for k in best}
    for y in range(2016, 2024):
        s, e = f"{y}-01-01", min(f"{y+2}-12-31", END)
        mb = Backtester(panel, PortfolioConstructor(cfg),
                        BacktestConfig(rebal_bars=126)).run(s, e)["metrics"]
        line = f"  {y}-{y+2}: {mb['cagr']*100:+8.1f}%"
        for k in best:
            mm = AsymmetricExitBacktester(panel, PortfolioConstructor(cfg),
                                          BacktestConfig(rebal_bars=126),
                                          check_bars=k[0], exit_rank=k[1]
                                          ).run(s, e)["metrics"]
            diffs[k].append(mm["cagr"] - mb["cagr"])
            line += f" {mm['cagr']*100:+9.1f}%"
        print(line)
    for k in best:
        dd = diffs[k]
        print(f"  => B check{k[0]}d/exit{k[1]}: {np.mean(dd)*100:+.2f}pp avg, "
              f"beats {sum(1 for x in dd if x > 0.001)}/8, worst {min(dd)*100:+.1f}pp")


if __name__ == "__main__":
    main()

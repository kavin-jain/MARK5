"""
MARK6 — Walk-Forward Backtester (tax-aware, survivorship-aware)
===============================================================
Daily NAV simulation with per-name average-cost lot tracking, Indian equity tax,
real transaction costs, and point-in-time universe selection.

Tax model (fy_netting=True, the default — matches actual Indian law):
  Realised gains/losses accrue to a fiscal-year (Apr–Mar) ledger and are NETTED:
  STCL offsets STCG then LTCG; LTCL offsets LTCG only; unabsorbed losses carry
  forward (8-yr limit not binding at our horizons). Net STCG taxed 20%, net LTCG
  12.5%. Tax is paid each April from cash (positions sold pro-rata if needed —
  no implicit leverage). The legacy per-trade model (losses earn NO credit)
  over-taxed turnover and is kept only via fy_netting=False for reproduction.

Design choices that make the numbers trustworthy:
  - Point-in-time eligibility every rebalance (no survivorship/look-ahead).
  - Factors precomputed once per name as causal series; only as-of values used.
  - Tax applied to BOTH the strategy and the equal-weight benchmark so any
    reported alpha is net-to-net and honest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .construction import PortfolioConstructor
from .factors import FactorLibrary, composite_score
from .universe import DataPanel

TRADING_DAYS = 252


@dataclass
class BacktestConfig:
    cost_pct: float = 0.0029          # round-trip commission
    slippage_pct: float = 0.001       # one-way slippage (each side)
    ltcg: float = 0.125
    stcg: float = 0.20
    ltcg_days: int = 365
    rebal_bars: int = 252             # annual
    min_history: int = 252
    liquidity_pct: float = 0.40
    no_trade_band: float = 0.0        # skip reweight trades smaller than band*NAV on names
                                      # we're KEEPING (full entries/exits always execute).
                                      # Cuts needless STCG-realizing weight-churn. 0 = off.
    fy_netting: bool = True           # True = honest Indian FY tax netting (losses offset
                                      # gains, settled each April). False = legacy per-trade
                                      # model (no loss credit) for reproducing old results.
    warmup_skip: int = 0              # 0 = enter immediately (factors are valid at start —
                                      # built from pre-window history, no look-ahead). Was 1,
                                      # which left the book in CASH for the first ~year of
                                      # every window (drag of up to ~3.6pp in up-regime
                                      # walk-forward windows; unfair vs day-1-invested Nifty).


def metrics(nav: pd.Series) -> dict:
    nav = nav.dropna()
    if len(nav) < 3:
        return {}
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    ret = nav.pct_change(fill_method=None).dropna()
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ret.mean() / ret.std() * np.sqrt(TRADING_DAYS)) if ret.std() > 0 else 0.0
    downside = ret[ret < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (ret.mean() * TRADING_DAYS / downside) if downside > 0 else 0.0
    dd = (nav / nav.cummax() - 1).min()
    calmar = cagr / abs(dd) if dd != 0 else 0.0
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "sortino": sortino,
            "max_dd": dd, "calmar": calmar, "years": yrs}


class Backtester:
    def __init__(self, panel: DataPanel, constructor: PortfolioConstructor,
                 config: BacktestConfig | None = None,
                 extra_factors: dict | None = None,
                 screen=None):
        self.panel = panel
        self.con = constructor
        self.cfg = config or BacktestConfig()
        # optional point-in-time universe screen: callable(asof, eligible) -> eligible.
        # Used for exclusion-style filters (e.g. quality screen) that act BEFORE
        # factor ranking — distinct from tilt weights, which were falsified (K11/K15).
        self.screen = screen
        # precompute causal factor series per name (once)
        self._factors: dict[str, pd.DataFrame] = {}
        for t in panel.tickers:
            self._factors[t] = FactorLibrary.compute_all(panel.close[t])
        # optional external (shareholding-derived) causal factors, off by default.
        # {ticker -> DataFrame(index=disclosure date, cols=extra factor names)}
        self.extra_factors = extra_factors or {}
        self.extra_names = tuple(self.con.cfg.factor_weights.keys()) if extra_factors else ()
        self.extra_names = tuple(n for n in self.extra_names
                                 if n not in FactorLibrary.DEFAULT_FACTORS)

    def _factor_panel(self, asof: pd.Timestamp, names: list[str]) -> tuple[pd.Series, pd.Series]:
        """Composite score + recent annualised vol for `names` as-of `asof`."""
        fnames = FactorLibrary.DEFAULT_FACTORS
        raw = {f: {} for f in fnames}
        raw.update({f: {} for f in self.extra_names})
        vol = {}
        for t in names:
            fdf = self._factors[t]
            row = fdf.loc[:asof]
            if row.empty:
                continue
            last = row.iloc[-1]
            for f in fnames:
                raw[f][t] = last.get(f, np.nan)
            vol[t] = -last.get("low_vol", np.nan)  # low_vol = -annualised vol
            # external factors: latest disclosed value strictly as-of `asof`
            if self.extra_names and t in self.extra_factors:
                erow = self.extra_factors[t].loc[:asof]
                if not erow.empty:
                    elast = erow.iloc[-1]
                    for f in self.extra_names:
                        raw[f][t] = elast.get(f, np.nan)
        panel = {f: pd.Series(raw[f]) for f in raw}
        comp = composite_score(panel, self.con.cfg.factor_weights)
        return comp, pd.Series(vol)

    def run(self, start: str, end: str) -> dict:
        cfg = self.cfg
        cal = self.panel.trading_calendar(start, end)
        prices = self.panel.close.loc[start:end].reindex(cal).ffill(limit=5)
        rets = prices.ffill().pct_change(fill_method=None)
        tickers = list(prices.columns)

        pos = {t: 0.0 for t in tickers}      # market value
        basis = {t: 0.0 for t in tickers}    # cost basis
        entry = {t: None for t in tickers}   # value-weighted entry date
        cash, tax_paid, traded = 1.0, 0.0, 0.0
        nav_hist, weights_hist = {}, {}
        trades = []                          # institutional trade ledger
        last_rebal, n_rebal = -10**9, 0
        # fiscal-year (Apr–Mar) netting ledger; losses stored as negative accruals
        fy = {"stcg": 0.0, "ltcg": 0.0}
        cf_stcl, cf_ltcl = 0.0, 0.0          # carried-forward losses (stored positive)

        def fy_of(d):
            return d.year + (1 if d.month >= 4 else 0)

        cur_fy = fy_of(cal[0])

        def net_tax(st, lt, stl, ltl):
            """Indian netting: STCL vs STCG then LTCG; LTCL vs LTCG only.
            Returns (tax, leftover_stcl, leftover_ltcl)."""
            stl += max(0.0, -st); st = max(0.0, st)
            ltl += max(0.0, -lt); lt = max(0.0, lt)
            use = min(stl, st); st -= use; stl -= use
            use = min(stl, lt); lt -= use; stl -= use
            use = min(ltl, lt); lt -= use; ltl -= use
            return st * cfg.stcg + lt * cfg.ltcg, stl, ltl

        def settle_fy(today):
            """Pay the netted FY tax from cash; sell pro-rata if cash short
            (sale gains accrue to the NEW fiscal year — no implicit leverage)."""
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
            rate = cfg.ltcg if held > cfg.ltcg_days else cfg.stcg
            if cfg.fy_netting:
                fy["ltcg" if held > cfg.ltcg_days else "stcg"] += gain
                tax = 0.0                     # settled at FY level
            else:
                tax = max(0.0, gain) * rate   # legacy: per-trade, no loss credit
                tax_paid += tax
            traded += sell_val
            px = prices.loc[today, t] if today in prices.index else np.nan
            trades.append({"date": today, "ticker": t, "side": "SELL", "price": float(px),
                           "value": float(sell_val), "gain": float(gain),
                           "held_days": int(held), "tax_rate": rate,
                           "tax": float(tax),
                           "term": "LTCG" if held > cfg.ltcg_days else "STCG"})
            pos[t] -= sell_val
            basis[t] -= basis[t] * frac
            if pos[t] < 1e-12:
                pos[t], basis[t], entry[t] = 0.0, 0.0, None
            return sell_val - tax - sell_val * (cfg.cost_pct / 2 + cfg.slippage_pct)

        def buy(t, buy_val, today):
            nonlocal cash, traded
            cash -= buy_val + buy_val * (cfg.cost_pct / 2 + cfg.slippage_pct)
            traded += buy_val
            px = prices.loc[today, t] if today in prices.index else np.nan
            trades.append({"date": today, "ticker": t, "side": "BUY", "price": float(px),
                           "value": float(buy_val), "gain": 0.0, "held_days": 0,
                           "tax_rate": 0.0, "tax": 0.0, "term": ""})
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
                        if rt == rt:  # not nan
                            pos[t] = v * (1 + rt)
            if cfg.fy_netting and fy_of(d) != cur_fy:
                settle_fy(d)
                cur_fy = fy_of(d)
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
                        band = cfg.no_trade_band * nav
                        for t in tickers:
                            tgt = target.get(t, 0.0)
                            if pos[t] > tgt + 1e-9:
                                # always honour full exits; band only mutes small reweights
                                if tgt <= 1e-9 or (pos[t] - tgt) > band:
                                    cash += realize(t, pos[t] - tgt, d)
                        for t, tgt in target.items():
                            if tgt > pos[t] + 1e-9:
                                # always honour new entries; band mutes small top-ups
                                if pos[t] <= 1e-9 or (tgt - pos[t]) > band:
                                    buy(t, tgt - pos[t], d)
                        weights_hist[d] = tw
            nav_hist[d] = cash + sum(pos.values())

        # terminal liquidation tax (fair vs buy & hold which also embeds it)
        last = cal[-1]
        if cfg.fy_netting:
            # hypothetical liquidation joins the pending FY ledger, netted properly
            st, lt = fy["stcg"], fy["ltcg"]
            for t in tickers:
                if pos[t] > 0:
                    g = pos[t] - basis[t]
                    long_term = entry[t] is not None and (last - entry[t]).days > cfg.ltcg_days
                    if long_term:
                        lt += g
                    else:
                        st += g
            term_tax, _, _ = net_tax(st, lt, cf_stcl, cf_ltcl)
        else:
            term_tax = sum(max(0.0, pos[t] - basis[t]) *
                           (cfg.ltcg if (entry[t] and (last - entry[t]).days > cfg.ltcg_days) else cfg.stcg)
                           for t in tickers if pos[t] > 0)
        nav = pd.Series(nav_hist)
        net = nav.copy()
        net.iloc[-1] = nav.iloc[-1] - term_tax
        yrs = (cal[-1] - cal[0]).days / 365.25
        m = metrics(net)
        m.update({"turnover_yr": traded / yrs / nav.mean(), "tax_paid": tax_paid + term_tax,
                  "n_rebalances": n_rebal, "avg_holdings": np.mean([len(w) for w in weights_hist.values()]) if weights_hist else 0})
        return {"nav_net": net, "nav_gross": nav, "metrics": m, "weights": weights_hist,
                "trades": trades}

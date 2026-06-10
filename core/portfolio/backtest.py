"""
MARK6 — Walk-Forward Backtester (tax-aware, survivorship-aware)
===============================================================
Daily NAV simulation with per-name average-cost lot tracking, Indian equity tax
(LTCG 12.5% >365d / STCG 20% <=365d on realised gains + terminal liquidation),
real transaction costs, and point-in-time universe selection.

Design choices that make the numbers trustworthy:
  - Point-in-time eligibility every rebalance (no survivorship/look-ahead).
  - Factors precomputed once per name as causal series; only as-of values used.
  - Tax applied to BOTH the strategy and the equal-weight benchmark so any
    reported alpha is net-to-net and honest.
  - Annual rebalance default -> low turnover -> gains qualify for LTCG (the single
    biggest net-return lever discovered this session).
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

        def realize(t, sell_val, today):
            nonlocal tax_paid, traded
            if pos[t] <= 0:
                return 0.0
            frac = min(1.0, sell_val / pos[t])
            gain = sell_val - basis[t] * frac
            held = (today - entry[t]).days if entry[t] is not None else 0
            rate = cfg.ltcg if held > cfg.ltcg_days else cfg.stcg
            tax = max(0.0, gain) * rate
            tax_paid += tax
            traded += sell_val
            px = prices.loc[today, t] if today in prices.index else np.nan
            trades.append({"date": today, "ticker": t, "side": "SELL", "price": float(px),
                           "value": float(sell_val), "gain": float(gain),
                           "held_days": int(held), "tax_rate": rate,
                           "tax": float(max(0.0, gain) * rate),
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

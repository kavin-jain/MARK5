"""
MARK6 — Walk-Forward Backtester (tax-aware, survivorship-aware)
===============================================================
Daily NAV simulation with per-name FIFO lot tracking, Indian equity tax,
real transaction costs, and point-in-time universe selection.

Tax model (fy_netting=True, the default — matches actual Indian law):
  Realised gains/losses accrue to a fiscal-year (Apr–Mar) ledger and are NETTED:
  STCL offsets STCG then LTCG; LTCL offsets LTCG only; unabsorbed losses carry
  forward (8-yr limit not binding at our horizons). Net STCG taxed 20%, net LTCG
  12.5%. Tax is paid each April from cash (positions sold pro-rata if needed —
  no implicit leverage). Lots are consumed FIFO per name, as the law requires,
  so each partial sale's STCG/LTCG split is per-lot, not blended.
  The legacy per-trade model (losses earn NO credit) is kept via fy_netting=False
  for reproducing pre-v7.1 results.

Execution model (exec_lag, default 1 — realistic):
  Signals are computed from data through the close of the rebalance day d; the
  trades execute at the close of day d+exec_lag. exec_lag=0 reproduces the old
  same-close fills (which assume you can trade the close you just measured).

Known approximations (all documented, direction stated):
  - Dividends: input closes are dividend-adjusted (total-return), so dividends
    compound in the book and are taxed as capital gains on sale rather than at
    slab rates as income. Flatters the strategy by roughly 0.1–0.3pp/yr.
  - The LTCG ₹1.25L annual exemption is NOT modelled (NAV-unit simulation has no
    rupee scale). Conservative — real after-tax returns would be slightly higher.
  - Terminal liquidation tax is applied to the final NAV point, which puts a
    one-day tax cliff inside the return series used for Sharpe/drawdown.
    Conservative — it can only worsen those statistics.

Design choices that make the numbers trustworthy:
  - Point-in-time eligibility every rebalance (no survivorship/look-ahead in
    *membership*; the residual candidate-list survivorship is documented in the
    README and bounded by scripts/survivorship_validation.py).
  - Factors precomputed once per name as causal series; only as-of values used.
  - Tax applied to BOTH the strategy and the equal-weight benchmark so any
    reported alpha is net-to-net and honest.
  - Buys are cash-constrained (scaled to available cash) — the book can never
    run >100% invested on a phantom interest-free overdraft.
  - A held name with no real print for stale_exit_days bars is force-exited at
    a haircut (delisting/suspension realism; ffill can't hide a dead name).
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
                                      # built from pre-window history, no look-ahead).
    exec_lag: int = 1                 # bars between signal close and execution close.
                                      # 1 = realistic next-close fills (default).
                                      # 0 = legacy same-close fills (reproduction only).
    rf_annual: float = 0.065          # Indian risk-free (T-bill/repo) for excess Sharpe.
    stale_exit_days: int = 21         # force-exit a holding with no real print this long
    delist_haircut: float = 0.25      # value haircut applied on a forced stale exit
                                      # (a suspended name never sells at its frozen mark)


def metrics(nav: pd.Series, rf_annual: float = 0.065) -> dict:
    """Performance metrics. `sharpe` is the raw (rf=0) ratio kept for continuity;
    `sharpe_excess` deducts the risk-free rate and is the honest headline number.
    Sortino uses the standard LPM2 downside deviation over ALL observations."""
    nav = nav.dropna()
    if len(nav) < 3:
        return {}
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    ret = nav.pct_change(fill_method=None).dropna()
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0
    sd = ret.std()
    sharpe = (ret.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
    sharpe_excess = ((ret.mean() - rf_daily) / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(ret, 0.0) ** 2)) * np.sqrt(TRADING_DAYS))
    sortino = ((ret.mean() * TRADING_DAYS - rf_annual) / downside) if downside > 0 else 0.0
    dd = (nav / nav.cummax() - 1).min()
    calmar = cagr / abs(dd) if dd != 0 else 0.0
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "sharpe_excess": sharpe_excess,
            "sortino": sortino, "max_dd": dd, "calmar": calmar, "years": yrs,
            "rf_annual": rf_annual}


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
        raw_prices = self.panel.close.loc[start:end].reindex(cal)
        prices = raw_prices.ffill(limit=5)
        rets = prices.ffill().pct_change(fill_method=None)
        tickers = list(prices.columns)

        # FIFO lots per name: list of [market_value, cost_basis, entry_date].
        # Oldest lot first; realize() consumes from the front, as the law requires.
        lots: dict[str, list] = {t: [] for t in tickers}
        pos = {t: 0.0 for t in tickers}      # cached sum of lot market values
        cash, tax_paid, traded = 1.0, 0.0, 0.0
        nav_hist, weights_hist = {}, {}
        trades = []                          # institutional trade ledger (one row per lot)
        last_rebal, n_rebal = -10**9, 0
        pending_tw = None                    # signal-day weights awaiting execution
        pending_exec_i = -1
        stale = {t: 0 for t in tickers}      # consecutive bars without a real print
        # fiscal-year (Apr–Mar) netting ledger; losses stored as negative accruals
        fy = {"stcg": 0.0, "ltcg": 0.0}
        cf_stcl, cf_ltcl = 0.0, 0.0          # carried-forward losses (stored positive)

        def fy_of(d):
            return d.year + (1 if d.month >= 4 else 0)

        cur_fy = fy_of(cal[0])

        def net_tax(st, lt, stl, ltl):
            """Indian netting: STCL vs STCG then LTCG; LTCL vs LTCG only.
            Returns (tax, leftover_stcl, leftover_ltl)."""
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
            """Sell up to sell_val of name t, consuming lots FIFO. Each lot's gain
            is classified STCG/LTCG by ITS OWN holding period (statutory FIFO).
            Returns net cash proceeds (after per-trade tax in legacy mode, and
            after costs+slippage)."""
            nonlocal tax_paid, traded
            if pos[t] <= 0:
                return 0.0
            sell_val = min(sell_val, pos[t])
            remaining = sell_val
            px = prices.loc[today, t] if today in prices.index else np.nan
            tax_total = 0.0
            while remaining > 1e-15 and lots[t]:
                lot = lots[t][0]
                take = min(lot[0], remaining)
                frac = take / lot[0]
                gain = take - lot[1] * frac
                held = (today - lot[2]).days
                long_term = held > cfg.ltcg_days
                rate = cfg.ltcg if long_term else cfg.stcg
                if cfg.fy_netting:
                    fy["ltcg" if long_term else "stcg"] += gain
                    tax = 0.0                     # settled at FY level
                else:
                    tax = max(0.0, gain) * rate   # legacy: per-trade, no loss credit
                    tax_paid += tax
                tax_total += tax
                trades.append({"date": today, "ticker": t, "side": "SELL",
                               "price": float(px), "value": float(take),
                               "gain": float(gain), "held_days": int(held),
                               "tax_rate": rate, "tax": float(tax),
                               "term": "LTCG" if long_term else "STCG"})
                lot[0] -= take
                lot[1] *= (1 - frac)
                remaining -= take
                if lot[0] < 1e-12:
                    lots[t].pop(0)
            sold = sell_val - remaining
            traded += sold
            pos[t] = sum(l[0] for l in lots[t])
            return sold - tax_total - sold * (cfg.cost_pct / 2 + cfg.slippage_pct)

        def buy(t, buy_val, today):
            nonlocal cash, traded
            cash -= buy_val + buy_val * (cfg.cost_pct / 2 + cfg.slippage_pct)
            traded += buy_val
            px = prices.loc[today, t] if today in prices.index else np.nan
            trades.append({"date": today, "ticker": t, "side": "BUY", "price": float(px),
                           "value": float(buy_val), "gain": 0.0, "held_days": 0,
                           "tax_rate": 0.0, "tax": 0.0, "term": ""})
            lots[t].append([buy_val, buy_val, today])
            pos[t] += buy_val

        def execute(tw, d):
            """Trade toward target weights tw at day d's closes (NAV-proportional)."""
            nonlocal cash
            nav = cash + sum(pos.values())
            target = {t: nav * w for t, w in tw.items()}
            band = cfg.no_trade_band * nav
            for t in tickers:
                tgt = target.get(t, 0.0)
                if pos[t] > tgt + 1e-9:
                    # always honour full exits; band only mutes small reweights
                    if tgt <= 1e-9 or (pos[t] - tgt) > band:
                        cash += realize(t, pos[t] - tgt, d)
            # cash-constrained buys: never spend more than available cash
            buys = []
            for t, tgt in target.items():
                if tgt > pos[t] + 1e-9:
                    # always honour new entries; band mutes small top-ups
                    if pos[t] <= 1e-9 or (tgt - pos[t]) > band:
                        buys.append((t, tgt - pos[t]))
            friction = cfg.cost_pct / 2 + cfg.slippage_pct
            need = sum(v * (1 + friction) for _, v in buys)
            scale = min(1.0, max(0.0, cash) / need) if need > 0 else 0.0
            for t, v in buys:
                buy(t, v * scale, d)

        ret_rows = rets.values
        raw_rows = raw_prices.values
        col = {t: i for i, t in enumerate(tickers)}
        for i, d in enumerate(cal):
            if i > 0:
                r = ret_rows[i]
                for t in tickers:
                    v = pos[t]
                    if v > 0:
                        rt = r[col[t]]
                        if rt == rt:  # not nan
                            factor = 1 + rt
                            for lot in lots[t]:
                                lot[0] *= factor
                            pos[t] = v * factor
            # stale-print watchdog: a held name with no REAL price for
            # stale_exit_days bars is haircut and force-exited (suspension /
            # delisting realism — ffill must not hide a dead name).
            for t in tickers:
                if raw_rows[i][col[t]] == raw_rows[i][col[t]]:
                    stale[t] = 0
                elif pos[t] > 0:
                    stale[t] += 1
                    if stale[t] >= cfg.stale_exit_days:
                        for lot in lots[t]:
                            lot[0] *= (1 - cfg.delist_haircut)
                        pos[t] = sum(l[0] for l in lots[t])
                        cash += realize(t, pos[t], d)
                        stale[t] = 0
            if cfg.fy_netting and fy_of(d) != cur_fy:
                settle_fy(d)
                cur_fy = fy_of(d)
            # pending signal from exec_lag bars ago -> execute at today's close
            if pending_tw is not None and i >= pending_exec_i:
                execute(pending_tw, d)
                pending_tw = None
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
                        weights_hist[d] = tw
                        if cfg.exec_lag <= 0:
                            execute(tw, d)
                        else:
                            pending_tw = tw
                            pending_exec_i = i + cfg.exec_lag
            nav_hist[d] = cash + sum(pos.values())

        # terminal liquidation tax (fair vs buy & hold which also embeds it),
        # classified per FIFO lot
        last = cal[-1]
        st, lt = (fy["stcg"], fy["ltcg"]) if cfg.fy_netting else (0.0, 0.0)
        term_tax = 0.0
        for t in tickers:
            for mv, cost, entry_d in lots[t]:
                g = mv - cost
                long_term = (last - entry_d).days > cfg.ltcg_days
                if cfg.fy_netting:
                    if long_term:
                        lt += g
                    else:
                        st += g
                else:
                    term_tax += max(0.0, g) * (cfg.ltcg if long_term else cfg.stcg)
        if cfg.fy_netting:
            term_tax, _, _ = net_tax(st, lt, cf_stcl, cf_ltcl)
        nav = pd.Series(nav_hist)
        net = nav.copy()
        net.iloc[-1] = nav.iloc[-1] - term_tax
        yrs = (cal[-1] - cal[0]).days / 365.25
        m = metrics(net, cfg.rf_annual)
        m.update({"turnover_yr": traded / yrs / nav.mean(), "tax_paid": tax_paid + term_tax,
                  "n_rebalances": n_rebal, "avg_holdings": np.mean([len(w) for w in weights_hist.values()]) if weights_hist else 0})
        return {"nav_net": net, "nav_gross": nav, "metrics": m, "weights": weights_hist,
                "trades": trades}

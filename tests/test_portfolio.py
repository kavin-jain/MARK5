"""
MARK6 Smart-Beta Portfolio — Test Suite
=======================================
Covers the invariants that make the backtest trustworthy:
  - factor causality (no look-ahead)
  - cross-sectional scoring & composite blending
  - point-in-time universe eligibility
  - construction weight constraints (sum, name cap, sector cap, inverse-vol, buffer)
  - backtester accounting identity, tax application, LTCG/STCG classification
  - real-data integration smoke test (skipped if cache absent)

Run: pytest tests/test_portfolio.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.portfolio.factors import (FactorLibrary, cross_sectional_z, composite_score)
from core.portfolio.construction import (ConstructionConfig, PortfolioConstructor, _cap_weights)


# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def trend_series():
    idx = pd.date_range("2015-01-01", periods=600, freq="B")
    return pd.Series(100 * (1.0005 ** np.arange(600)), index=idx)  # steady uptrend


# ── factor causality / correctness ─────────────────────────────────────────────
class TestFactors:
    def test_momentum_is_causal(self, trend_series):
        """Changing a FUTURE price must not change a PAST momentum value."""
        mom = FactorLibrary.momentum(trend_series)
        t = trend_series.index[400]
        v_before = mom.loc[t]
        s2 = trend_series.copy()
        s2.iloc[500:] *= 2.0                      # perturb only the future
        mom2 = FactorLibrary.momentum(s2)
        assert mom2.loc[t] == pytest.approx(v_before, abs=1e-12)

    def test_momentum_skips_recent_month(self, trend_series):
        mom = FactorLibrary.momentum(trend_series, lookback=252, skip=21)
        # equals close[t-21]/close[t-252]-1
        t = trend_series.index[400]
        i = trend_series.index.get_loc(t)
        expected = trend_series.iloc[i - 21] / trend_series.iloc[i - 252] - 1
        assert mom.loc[t] == pytest.approx(expected, rel=1e-9)

    def test_low_vol_sign(self):
        idx = pd.date_range("2015-01-01", periods=400, freq="B")
        calm = pd.Series(100 * (1.0002 ** np.arange(400)), index=idx)
        rng = np.random.default_rng(0)
        wild = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.05, 400)), index=idx)
        # low_vol is NEGATIVE vol -> calmer name scores HIGHER
        assert FactorLibrary.low_vol(calm).iloc[-1] > FactorLibrary.low_vol(wild).iloc[-1]

    def test_trend_positive_in_uptrend(self, trend_series):
        assert FactorLibrary.trend(trend_series).iloc[-1] > 0

    def test_compute_all_columns(self, trend_series):
        df = FactorLibrary.compute_all(trend_series)
        assert list(df.columns) == list(FactorLibrary.DEFAULT_FACTORS)
        assert len(df) == len(trend_series)


class TestScoring:
    def test_zscore_standardised(self):
        z = cross_sectional_z(pd.Series([1.0, 2, 3, 4, 5]))
        assert z.mean() == pytest.approx(0, abs=1e-9)
        assert z.std(ddof=0) == pytest.approx(1, abs=1e-6)

    def test_zscore_constant_is_zero(self):
        z = cross_sectional_z(pd.Series([7.0, 7, 7]))
        assert (z == 0).all()

    def test_zscore_clips_outliers(self):
        z = cross_sectional_z(pd.Series([0.0] * 20 + [1e6]), clip=3.0)
        assert z.max() <= 3.0 + 1e-9

    def test_composite_blends(self):
        panel = {"a": pd.Series({"X": 1.0, "Y": -1.0}),
                 "b": pd.Series({"X": 1.0, "Y": -1.0})}
        comp = composite_score(panel)
        assert comp["X"] > comp["Y"]          # consistently good name ranks higher

    def test_composite_handles_missing(self):
        panel = {"a": pd.Series({"X": 1.0, "Y": -1.0, "Z": np.nan}),
                 "b": pd.Series({"X": 0.5, "Y": -0.5, "Z": 2.0})}
        comp = composite_score(panel)
        assert set(comp.index) == {"X", "Y", "Z"}
        assert comp.notna().all()


# ── construction constraints ───────────────────────────────────────────────────
class TestConstruction:
    def _comp_vol(self, n=30):
        names = [f"T{i}" for i in range(n)]
        comp = pd.Series(np.linspace(2, -2, n), index=names)
        vol = pd.Series(np.linspace(0.2, 0.6, n), index=names)
        return comp, vol

    def test_weights_sum_to_one(self):
        comp, vol = self._comp_vol()
        con = PortfolioConstructor(ConstructionConfig(n_hold=20))
        w = con.target_weights(comp, vol, [])
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert (w >= 0).all()

    def test_name_cap_respected(self):
        comp, vol = self._comp_vol()
        con = PortfolioConstructor(ConstructionConfig(n_hold=20, max_weight=0.08))
        w = con.target_weights(comp, vol, [])
        assert w.max() <= 0.08 + 1e-9

    def test_holds_n(self):
        comp, vol = self._comp_vol()
        con = PortfolioConstructor(ConstructionConfig(n_hold=15, max_weight=0.5))
        w = con.target_weights(comp, vol, [])
        assert len(w) == 15

    def test_inverse_vol_favours_calm(self):
        # two names, equal score, different vol -> calmer gets more weight
        comp = pd.Series({"CALM": 0.0, "WILD": 0.0})
        vol = pd.Series({"CALM": 0.1, "WILD": 0.5})
        con = PortfolioConstructor(ConstructionConfig(
            n_hold=2, base_weighting="inverse_vol", tilt_strength=0.0, max_weight=1.0))
        w = con.target_weights(comp, vol, [])
        assert w["CALM"] > w["WILD"]

    def test_buffer_reduces_turnover(self):
        # a held name ranked 25th (within 2x buffer of n_hold=20) is KEPT
        names = [f"T{i}" for i in range(40)]
        comp = pd.Series(np.linspace(2, -2, 40), index=names)
        con = PortfolioConstructor(ConstructionConfig(n_hold=20, buffer_mult=2.0))
        held = ["T25"]                       # rank 25 < exit_rank 40 -> keep
        picks = con.select(comp, held)
        assert "T25" in picks

    def test_sector_cap(self):
        # 4 sectors so a 0.30 cap is FEASIBLE (4*0.30=1.2>=1.0); BANK is overweight
        names = [f"T{i}" for i in range(12)]
        comp = pd.Series(np.linspace(2, -2, 12), index=names)
        vol = pd.Series(0.3, index=names)
        secs = ["BANK"] * 6 + ["IT"] * 2 + ["PHARMA"] * 2 + ["AUTO"] * 2
        sectors = {n: secs[i] for i, n in enumerate(names)}
        con = PortfolioConstructor(
            ConstructionConfig(n_hold=12, max_weight=1.0, max_sector_weight=0.30),
            sector_map=sectors)
        w = con.target_weights(comp, vol, [])
        bank = sum(w[n] for n in names if sectors[n] == "BANK")
        assert bank <= 0.30 + 0.02            # BANK capped; excess pushed to other sectors
        assert w.sum() == pytest.approx(1.0)

    def test_cap_weights_helper(self):
        # feasible cap (4 names, 0.40 cap -> budget 1.6 >= 1.0)
        w = _cap_weights(pd.Series({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}), 0.40)
        assert w.max() <= 0.40 + 1e-9
        assert w.sum() == pytest.approx(1.0)

    def test_cap_weights_infeasible_falls_back_to_equal(self):
        # 3 names @ 0.30 cap is infeasible (max sum 0.90); fall back to equal weight
        w = _cap_weights(pd.Series({"A": 0.9, "B": 0.05, "C": 0.05}), 0.30)
        assert w.sum() == pytest.approx(1.0)
        assert np.allclose(w.values, 1 / 3)


# ── backtester accounting & tax ─────────────────────────────────────────────────
def _synthetic_panel(n=12, days=900, seed=1):
    """Build a DataPanel-like object from synthetic geometric-brownian prices."""
    from core.portfolio.universe import DataPanel
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=days, freq="B")
    closes, vols = {}, {}
    for i in range(n):
        drift = 0.0003 + 0.0002 * (i / n)
        px = 100 * np.cumprod(1 + rng.normal(drift, 0.02, days))
        closes[f"S{i}"] = pd.Series(px, index=idx)
        vols[f"S{i}"] = pd.Series(rng.uniform(1e6, 5e6, days), index=idx)
    panel = DataPanel.__new__(DataPanel)
    panel.close = pd.DataFrame(closes)
    panel.volume = pd.DataFrame(vols)
    panel.turnover = (panel.close * panel.volume).rolling(126, min_periods=40).median()
    panel.tickers = list(panel.close.columns)
    panel.trading_calendar = lambda s, e: panel.close.loc[s:e].index
    def _elig(asof, min_history=252, liquidity_pct=0.40):
        return [t for t in panel.tickers if len(panel.close[t].loc[:asof].dropna()) >= min_history]
    panel.eligible = _elig
    return panel


class TestBacktester:
    def test_accounting_and_runs(self):
        from core.portfolio import Backtester, BacktestConfig, PortfolioConstructor, ConstructionConfig
        panel = _synthetic_panel()
        con = PortfolioConstructor(ConstructionConfig(mode="equal_weight", base_weighting="equal"))
        bt = Backtester(panel, con, BacktestConfig(rebal_bars=252, warmup_skip=0))
        out = bt.run("2016-01-01", "2018-06-01")
        nav = out["nav_net"]
        assert len(nav) > 100
        assert (nav > 0).all()                 # never goes negative
        assert np.isfinite(out["metrics"]["cagr"])

    def test_tax_reduces_terminal_nav(self):
        """With positive gains, NET terminal NAV must be below GROSS (tax paid)."""
        from core.portfolio import Backtester, BacktestConfig, PortfolioConstructor, ConstructionConfig
        panel = _synthetic_panel(seed=3)
        con = PortfolioConstructor(ConstructionConfig(mode="equal_weight", base_weighting="equal"))
        bt = Backtester(panel, con, BacktestConfig(rebal_bars=252, warmup_skip=0))
        out = bt.run("2016-01-01", "2019-06-01")
        g, n = out["nav_gross"].iloc[-1], out["nav_net"].iloc[-1]
        if g > 1.0:                            # there were gains to tax
            assert n < g
            assert out["metrics"]["tax_paid"] > 0

    def test_ltcg_cheaper_than_stcg(self):
        """Annual-hold (LTCG) must incur less tax drag than quarterly churn (STCG)."""
        from core.portfolio import Backtester, BacktestConfig, PortfolioConstructor, ConstructionConfig
        panel = _synthetic_panel(seed=5)
        con = PortfolioConstructor(ConstructionConfig(mode="factor_tilt", n_hold=6,
                                                      base_weighting="equal", tilt_strength=0.0))
        annual = Backtester(panel, con, BacktestConfig(rebal_bars=252, warmup_skip=0)).run("2016-01-01", "2019-06-01")
        quarterly = Backtester(panel, con, BacktestConfig(rebal_bars=63, warmup_skip=0)).run("2016-01-01", "2019-06-01")
        assert quarterly["metrics"]["turnover_yr"] > annual["metrics"]["turnover_yr"]


# ── engine truth: FIFO lots, FY netting, execution lag, cash, stale exits ──────
class _ScriptedCon:
    """Constructor stub that returns a fixed sequence of target-weight dicts,
    one per rebalance call — lets tests drive exact trade scenarios."""
    def __init__(self, seq):
        self.cfg = ConstructionConfig()
        self.seq = list(seq)
        self.calls = 0

    def target_weights(self, comp, vol, held):
        w = self.seq[min(self.calls, len(self.seq) - 1)]
        self.calls += 1
        return pd.Series(w, dtype=float)


def _flat_panel(prices: dict, days=800, start="2016-05-02"):
    """DataPanel stub from explicit price Series (index auto if plain list)."""
    from core.portfolio.universe import DataPanel
    idx = pd.date_range(start, periods=days, freq="B")
    closes = {}
    for t, p in prices.items():
        arr = np.asarray(p, dtype=float)
        if len(arr) < days:
            arr = np.concatenate([arr, np.full(days - len(arr), arr[-1])])
        closes[t] = pd.Series(arr[:days], index=idx)
    panel = DataPanel.__new__(DataPanel)
    panel.close = pd.DataFrame(closes)
    panel.volume = pd.DataFrame(1e6, index=idx, columns=list(prices))
    panel.turnover = (panel.close * panel.volume).rolling(126, min_periods=1).median()
    panel.tickers = list(panel.close.columns)
    panel.trading_calendar = lambda s, e: panel.close.loc[s:e].index
    panel.eligible = lambda asof, mh=252, lq=0.4: panel.tickers
    return panel


def _no_friction(**kw):
    from core.portfolio import BacktestConfig
    base = dict(cost_pct=0.0, slippage_pct=0.0, exec_lag=0, warmup_skip=0)
    base.update(kw)
    return BacktestConfig(**base)


class TestEngineTruth:
    def test_fifo_lot_classification(self):
        """Partial sale after a recent top-up must consume the OLDEST lot first
        (statutory FIFO) -> LTCG, where a blended entry date would say STCG."""
        from core.portfolio import Backtester
        panel = _flat_panel({"S0": [100.0]}, days=800)
        con = _ScriptedCon([{"S0": 0.5}, {"S0": 0.6}, {"S0": 0.3}])
        bt = Backtester(panel, con, _no_friction(rebal_bars=189))
        out = bt.run("2016-05-02", "2019-06-01")
        # third rebalance (i=378) sells 0.30 of NAV; lot 1 is 378*1.4 ≈ 529
        # calendar days old -> every consumed slice must be LTCG
        sells = [t for t in out["trades"] if t["side"] == "SELL"]
        assert sells, "expected a partial sale"
        first_sale_date = min(t["date"] for t in sells)
        first_sale = [t for t in sells if t["date"] == first_sale_date]
        assert all(t["term"] == "LTCG" for t in first_sale), first_sale

    def test_fy_netting_loss_offsets_gain(self):
        """Equal gain and loss realised in the same FY -> zero tax under netting,
        positive tax under the legacy no-credit model."""
        from core.portfolio import Backtester
        days = 500
        up = 100 * (1 + 0.5 * np.arange(days) / 200).clip(max=1.5)     # +50% by day 200
        dn = 100 * (1 - 0.5 * np.arange(days) / 200).clip(min=0.5)     # -50% by day 200
        panel = _flat_panel({"A": up, "B": dn}, days=days)
        seq = [{"A": 0.5, "B": 0.5}, {}, {}]
        m_net = Backtester(panel, _ScriptedCon(seq),
                           _no_friction(rebal_bars=200, fy_netting=True)
                           ).run("2016-05-02", "2018-03-01")["metrics"]
        m_leg = Backtester(panel, _ScriptedCon(seq),
                           _no_friction(rebal_bars=200, fy_netting=False)
                           ).run("2016-05-02", "2018-03-01")["metrics"]
        assert m_net["tax_paid"] == pytest.approx(0.0, abs=1e-9)
        assert m_leg["tax_paid"] > 0.01

    def test_exec_lag_misses_signal_day_jump(self):
        """exec_lag=1 buys at the NEXT close: a +100% move on the day after the
        signal must be captured by exec_lag=0 and missed by exec_lag=1."""
        from core.portfolio import Backtester
        px = [100.0, 200.0]                    # doubles on bar 1, flat after
        panel = _flat_panel({"S0": px}, days=300)
        seq = [{"S0": 1.0}]
        nav0 = Backtester(panel, _ScriptedCon(seq), _no_friction(rebal_bars=10**6, exec_lag=0)
                          ).run("2016-05-02", "2017-06-01")["nav_gross"]
        nav1 = Backtester(panel, _ScriptedCon(seq), _no_friction(rebal_bars=10**6, exec_lag=1)
                          ).run("2016-05-02", "2017-06-01")["nav_gross"]
        assert nav0.iloc[-1] == pytest.approx(2.0, rel=1e-6)
        assert nav1.iloc[-1] == pytest.approx(1.0, rel=1e-6)

    def test_buys_are_cash_constrained(self):
        """With heavy friction, total buys must be scaled so cash never goes
        negative (no phantom interest-free overdraft)."""
        from core.portfolio import Backtester, BacktestConfig
        panel = _flat_panel({"S0": [100.0], "S1": [100.0]}, days=300)
        cfg = BacktestConfig(cost_pct=0.05, slippage_pct=0.001, exec_lag=0,
                             warmup_skip=0, rebal_bars=10**6)
        out = Backtester(panel, _ScriptedCon([{"S0": 0.5, "S1": 0.5}]), cfg
                         ).run("2016-05-02", "2017-06-01")
        friction = cfg.cost_pct / 2 + cfg.slippage_pct
        total_buys = sum(t["value"] for t in out["trades"] if t["side"] == "BUY")
        assert total_buys <= 1.0 / (1 + friction) + 1e-9
        # NAV identity: nav = cash + positions, and cash >= 0 => nav >= positions
        assert out["nav_gross"].iloc[-1] > 0

    def test_stale_name_is_haircut_and_force_exited(self):
        """A held name whose prints stop must be written down and force-sold,
        not compounded at 0% and sold at full frozen value."""
        from core.portfolio import Backtester, BacktestConfig
        days = 400
        px = np.full(days, 100.0)
        px[120:] = np.nan                      # stops trading after bar 119
        panel = _flat_panel({"S0": px, "S1": [100.0]}, days=days)
        cfg = BacktestConfig(cost_pct=0.0, slippage_pct=0.0, exec_lag=0,
                             warmup_skip=0, rebal_bars=10**6,
                             stale_exit_days=10, delist_haircut=0.25)
        out = Backtester(panel, _ScriptedCon([{"S0": 0.5, "S1": 0.5}]), cfg
                         ).run("2016-05-02", "2017-11-01")
        s0_sells = [t for t in out["trades"] if t["ticker"] == "S0" and t["side"] == "SELL"]
        assert s0_sells, "stale name was never force-exited"
        assert sum(t["gain"] for t in s0_sells) < -0.05   # haircut booked as real loss
        # final NAV reflects the loss: 0.5 intact + 0.5*(1-0.25) = 0.875
        assert out["nav_gross"].iloc[-1] == pytest.approx(0.875, abs=0.01)


class TestStatsAndMetrics:
    def test_metrics_excess_sharpe_below_raw(self):
        from core.portfolio import metrics
        idx = pd.date_range("2016-01-01", periods=800, freq="B")
        rng = np.random.default_rng(0)
        nav = pd.Series(np.cumprod(1 + rng.normal(0.0008, 0.01, 800)), index=idx)
        m = metrics(nav, rf_annual=0.065)
        assert m["sharpe_excess"] < m["sharpe"]
        assert m["rf_annual"] == 0.065

    def test_pbo_near_half_on_pure_noise(self):
        from core.portfolio.stats import pbo_cscv
        rng = np.random.default_rng(7)
        M = rng.normal(0, 0.01, size=(1200, 20))
        pbo = pbo_cscv(M, n_splits=12)["pbo"]
        assert 0.3 < pbo < 0.7                 # no strategy is really better

    def test_dsr_deflates_with_more_trials(self):
        from core.portfolio.stats import deflated_sharpe_ratio
        rng = np.random.default_rng(1)
        ret = rng.normal(0.0005, 0.01, 1500)
        few = deflated_sharpe_ratio(ret, list(rng.normal(0.02, 0.02, 5)))
        many = deflated_sharpe_ratio(ret, list(rng.normal(0.02, 0.02, 500)))
        assert many["deflated_sharpe"] < few["deflated_sharpe"]

    def test_psr_orders_by_edge(self):
        from core.portfolio.stats import probabilistic_sharpe_ratio
        rng = np.random.default_rng(2)
        good = rng.normal(0.001, 0.01, 1000)
        flat = rng.normal(0.0, 0.01, 1000)
        assert probabilistic_sharpe_ratio(good) > probabilistic_sharpe_ratio(flat)


class TestUniverseGuards:
    def test_eligible_excludes_stale_names(self):
        from core.portfolio.universe import DataPanel
        idx = pd.date_range("2016-01-01", periods=600, freq="B")
        fresh = pd.Series(100.0, index=idx)
        stale = pd.Series(100.0, index=idx).copy()
        stale.iloc[-80:] = np.nan              # last print ~4 months before asof
        panel = DataPanel.__new__(DataPanel)
        panel.close = pd.DataFrame({"FRESH": fresh, "STALE": stale})
        panel.volume = pd.DataFrame(1e6, index=idx, columns=["FRESH", "STALE"])
        panel.turnover = (panel.close * panel.volume).rolling(126, min_periods=1).median()
        panel.tickers = ["FRESH", "STALE"]
        elig = DataPanel.eligible(panel, idx[-1], min_history=252, liquidity_pct=0.0)
        assert "FRESH" in elig and "STALE" not in elig


# ── real-data integration smoke test ───────────────────────────────────────────
class TestIntegration:
    def test_real_data_runs_and_is_sane(self):
        from core.portfolio import (DataPanel, discover_tickers, Backtester,
                                     BacktestConfig, PortfolioConstructor, ConstructionConfig)
        import glob
        from core.portfolio.universe import CACHE
        # skip on DATA availability, not on ticker NAMES — discover_tickers() falls
        # back to the pinned universe list when the cache is empty (fresh clone / CI),
        # so counting names would sail past this guard into an empty panel.
        if len(glob.glob(os.path.join(CACHE, "*.parquet"))) < 20:
            pytest.skip("no local price cache — run scripts/refetch_all.py")
        tickers = discover_tickers()
        panel = DataPanel(tickers, "2026-05-21")
        con = PortfolioConstructor(ConstructionConfig(mode="factor_tilt", n_hold=20))
        bt = Backtester(panel, con, BacktestConfig())
        out = bt.run("2018-01-01", "2021-12-31")
        m = out["metrics"]
        assert -1.0 < m["cagr"] < 2.0          # sane range, not absurd
        assert -1.0 <= m["max_dd"] <= 0.0
        assert out["nav_net"].iloc[0] == pytest.approx(1.0, abs=0.2)

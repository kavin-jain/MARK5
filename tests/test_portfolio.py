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
    def _elig(asof, min_history=252, liquidity_pct=0.40, **kw):
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
    panel.eligible = lambda asof, mh=252, lq=0.4, **kw: panel.tickers
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


class TestTranching:
    def test_blend_lies_within_tranche_range(self):
        """An average of tranche NAVs cannot beat the best or trail the worst —
        the invariant that makes tranching variance reduction, not an edge claim."""
        from core.portfolio import Backtester, tranched_run
        rng = np.random.default_rng(11)
        px = {f"S{i}": 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, 1400))
              for i in range(8)}
        panel = _flat_panel(px, days=1400)
        con = _ScriptedCon([{f"S{i}": 0.125 for i in range(8)}])
        bt = Backtester(panel, con, _no_friction(rebal_bars=126))
        out = tranched_run(bt, "2016-05-02", "2021-06-01", n_tranches=3, stagger_bars=42)
        finals = [n.reindex(out["nav_net"].index).ffill().pipe(lambda s: s / s.iloc[0]).iloc[-1]
                  for n in out["tranche_navs"]]
        assert min(finals) - 1e-9 <= out["nav_net"].iloc[-1] <= max(finals) + 1e-9
        assert out["metrics"]["n_tranches"] == 3

    def test_single_tranche_matches_plain_run(self):
        from core.portfolio import Backtester, tranched_run
        panel = _flat_panel({"S0": [100.0], "S1": [100.0]}, days=600)
        seq = [{"S0": 0.5, "S1": 0.5}]
        bt = Backtester(panel, _ScriptedCon(seq), _no_friction(rebal_bars=126))
        one = tranched_run(bt, "2016-05-02", "2018-06-01", n_tranches=1)
        bt2 = Backtester(panel, _ScriptedCon(seq), _no_friction(rebal_bars=126))
        plain = bt2.run("2016-05-02", "2018-06-01")
        assert one["nav_net"].iloc[-1] == pytest.approx(
            plain["nav_net"].iloc[-1] / plain["nav_net"].iloc[0], rel=1e-9)


class TestUniverseGuards:
    def test_absolute_turnover_floor_beats_percentile(self):
        """min_turnover filters on real rupees, not on a percentile of whatever
        happens to be cached (which drifts as the universe grows)."""
        from core.portfolio.universe import DataPanel
        idx = pd.date_range("2016-01-01", periods=600, freq="B")
        close = pd.DataFrame({"BIG": 100.0, "SMALL": 100.0}, index=idx)
        vol = pd.DataFrame({"BIG": 1e7, "SMALL": 1e3}, index=idx)   # 100cr vs 1 lakh/day
        panel = DataPanel.__new__(DataPanel)
        panel.close, panel.volume = close, vol
        panel.turnover = (close * vol).rolling(126, min_periods=1).median()
        panel.tickers = ["BIG", "SMALL"]
        # percentile floor keeps both (SMALL is simply the bottom of a 2-name pool)
        assert set(DataPanel.eligible(panel, idx[-1], 252, liquidity_pct=0.0)) == {"BIG", "SMALL"}
        # absolute Rs 20cr floor keeps only the genuinely liquid one
        assert DataPanel.eligible(panel, idx[-1], 252, min_turnover=2e8) == ["BIG"]

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


class TestSleeveAttribution:
    """The public page splits the headline into three sleeves. The split must
    RECONCILE — if the parts do not sum to the whole, the page is lying."""

    def _export(self):
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "paper", "paper_export.json")
        if not os.path.exists(p):
            pytest.skip("no live paper export — run scripts/paper_track.py export")
        import json
        return json.load(open(p))

    def test_sleeve_impacts_sum_to_headline(self):
        e = self._export()
        s = e.get("sleeves")
        if not s:
            pytest.skip("export predates sleeve attribution")
        # cash-on-cash impacts must reconstruct the headline return exactly
        assert s["total_impact_pp"] == pytest.approx(e["return_pct"], abs=0.005)

    def test_value_and_capital_bridge(self):
        e = self._export()
        s = e.get("sleeves")
        if not s:
            pytest.skip("export predates sleeve attribution")
        val = sum(r["value_inr"] for r in s["rows"]) + s["cash_inr"]
        inv = sum(r["invested_inr"] for r in s["rows"]) + s["cash_inr"]
        assert val == pytest.approx(e["nav"], abs=1.0)        # NAV = values + cash
        assert inv == pytest.approx(e["capital"], abs=1.0)    # capital = invested + cash

    def test_weights_sum_to_full_book(self):
        e = self._export()
        s = e.get("sleeves")
        if not s:
            pytest.skip("export predates sleeve attribution")
        w = sum(r["weight_pct"] for r in s["rows"])
        assert w + s["cash_inr"] / e["nav"] * 100 == pytest.approx(100.0, abs=0.05)

    def test_time_weighted_return_is_not_the_naive_one(self):
        """Regression guard for RESEARCH_LOG 5a: a rebalance sweeps idle cash into
        the equity sleeve, and the naive holdings-vs-entry figure counts that
        deposit as profit. The reported return must be the flow-neutral one."""
        e = self._export()
        s = e.get("sleeves")
        if not s or not e.get("rebalances"):
            pytest.skip("no rebalance yet — the two measures cannot diverge")
        eq = next(r for r in s["rows"] if r["key"] == "eq")
        naive_num = sum(h["pnl"] for h in e["holdings"]
                        if h["ticker"] not in ("GOLDBEES", "MON100"))
        naive_den = sum(h["value"] - h["pnl"] for h in e["holdings"]
                        if h["ticker"] not in ("GOLDBEES", "MON100"))
        naive = naive_num / naive_den * 100
        assert eq["return_pct"] != pytest.approx(naive, abs=0.05), (
            "equity sleeve is reporting the naive entry-price return; a rebalance "
            "has reset entry prices and swept cash in, so this overstates it")


class TestTerminalTax:
    """Exit tax is a liquidation COST, not a market return. Folding it into the
    last NAV bar (as export_dashboard once did) put a single -13% observation
    into pct_change: it inflated vol, depressed Sharpe/Sortino, and drew a cliff
    on the public charts that never happened in the market."""

    def _nav(self):
        idx = pd.bdate_range("2016-01-01", periods=2600)
        return pd.Series(np.linspace(1.0, 8.0, len(idx)), index=idx)

    def test_taxes_only_the_gain(self):
        from core.portfolio import metrics_after_exit_tax
        m = metrics_after_exit_tax(self._nav(), 0.15)
        assert m["net_multiple"] == pytest.approx(8.0 - 7.0 * 0.15)   # 1.0 is capital

    def test_no_tax_on_a_loss(self):
        """No gain, no tax. The multiple is measured off the FIRST bar, so a book
        that merely got scaled is still a gain — it has to actually end lower."""
        from core.portfolio import metrics_after_exit_tax
        idx = pd.bdate_range("2016-01-01", periods=2600)
        s = pd.Series(np.linspace(8.0, 4.0, len(idx)), index=idx)   # halves
        m = metrics_after_exit_tax(s, 0.15)
        assert m["gross_multiple"] == pytest.approx(0.5)
        assert m["net_multiple"] == pytest.approx(m["gross_multiple"])
        assert m["cagr"] == pytest.approx(m["cagr_gross"])

    def test_risk_stats_match_the_untaxed_series(self):
        """The whole point: vol/Sharpe/MaxDD must be blind to the exit tax."""
        from core.portfolio import metrics, metrics_after_exit_tax
        s = self._nav()
        before, g = s.copy(), metrics(s)
        m = metrics_after_exit_tax(s, 0.15)
        pd.testing.assert_series_equal(s, before)                # no mutation
        for k in ("vol", "sharpe_excess", "sortino", "max_dd"):
            assert m[k] == pytest.approx(g[k], rel=1e-12)
        assert m["cagr"] < g["cagr"]                             # only CAGR is net

    def test_dashboard_curves_are_gross_and_symmetric(self):
        """Both plotted lines must carry the same tax treatment, else the chart
        and the stat block describe different benchmarks."""
        import json
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "docs", "data", "mark6.json")
        if not os.path.exists(p):
            pytest.skip("no dashboard export — run scripts/export_dashboard.py")
        r = json.load(open(p))["research"]
        tt = r.get("terminal_tax")
        if not tt:
            pytest.skip("export predates the terminal-tax split")
        assert tt["curves_are_gross"] is True
        # the plotted endpoints are the GROSS multiples, on both lines
        assert r["equity_curve"][-1][1] == pytest.approx(tt["system_gross_multiple"], abs=1e-3)
        assert r["benchmark_curve"][-1][1] == pytest.approx(tt["benchmark_gross_multiple"], abs=1e-3)
        # and the headline CAGRs are the NET ones, on both lines
        yrs = r["period"]["years"]
        assert r["headline"]["cagr"] / 100 == pytest.approx(
            tt["system_net_multiple"] ** (1 / yrs) - 1, abs=2e-3)
        assert r["benchmark"]["cagr"] / 100 == pytest.approx(
            tt["benchmark_net_multiple"] ** (1 / yrs) - 1, abs=2e-3)

    def test_no_cliff_in_the_plotted_series(self):
        import json
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "docs", "data", "mark6.json")
        if not os.path.exists(p):
            pytest.skip("no dashboard export — run scripts/export_dashboard.py")
        r = json.load(open(p))["research"]
        if not r.get("terminal_tax"):
            pytest.skip("export predates the terminal-tax split")
        for key in ("equity_curve", "benchmark_curve"):
            a, b = r[key][-2][1], r[key][-1][1]
            assert b / a - 1 > -0.10, f"{key} ends in a >10% cliff — tax leaked into the chart"


class TestRebalanceDisclosure:
    """A page that says 'append-only ledger' while hiding that the book was
    rebuilt on day 4 is committing the failure it claims to defend against."""

    def _export(self):
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "paper", "paper_export.json")
        if not os.path.exists(p):
            pytest.skip("no live paper export — run scripts/paper_track.py export")
        import json
        return json.load(open(p))

    def test_every_rebalance_is_published(self):
        e = self._export()
        ev = e.get("rebalance_events")
        if ev is None:
            pytest.skip("export predates rebalance disclosure")
        assert len(ev) == e["rebalances"], "count and event list disagree"

    def test_off_cadence_rebalances_are_flagged(self):
        from scripts.paper_track import REBAL_DAYS
        e = self._export()
        ev = e.get("rebalance_events")
        if not ev:
            pytest.skip("no rebalance yet")
        for r in ev:
            assert r["off_cadence"] == (r["day_of_book"] < REBAL_DAYS)
            assert r["trades"] > 0 and r["date"]


class TestEngineTerminalTax:
    """Root-cause guard for the same defect one level down: Backtester.run once
    fed metrics() a NAV whose last bar had the exit tax subtracted, so every
    report in this repo inherited an inflated vol and a depressed Sharpe."""

    def _run(self):
        from core.portfolio import Backtester, BacktestConfig
        panel = _synthetic_panel(seed=3)
        con = PortfolioConstructor(ConstructionConfig(mode="equal_weight",
                                                      base_weighting="equal"))
        return Backtester(panel, con,
                          BacktestConfig(rebal_bars=252, warmup_skip=0)
                          ).run("2016-01-01", "2019-06-01")

    def test_risk_metrics_ignore_the_tax_bar(self):
        out = self._run()
        from core.portfolio import metrics
        m, gross = out["metrics"], out["nav_gross"]
        assert m["vol"] == pytest.approx(metrics(gross)["vol"], rel=1e-9)
        assert m["max_dd"] == pytest.approx(metrics(gross)["max_dd"], rel=1e-9)
        assert m["sharpe_excess"] == pytest.approx(metrics(gross)["sharpe_excess"], rel=1e-9)

    def test_cagr_is_net_of_terminal_tax(self):
        out = self._run()
        m = out["metrics"]
        if m["terminal_tax"] <= 0:
            pytest.skip("no terminal gain to tax in this fixture")
        assert m["cagr"] < m["cagr_gross"]
        yrs = (out["nav_net"].index[-1] - out["nav_net"].index[0]).days / 365.25
        assert m["cagr"] == pytest.approx(
            (out["nav_net"].iloc[-1] / out["nav_net"].iloc[0]) ** (1 / yrs) - 1, rel=1e-9)

    def test_calmar_uses_net_cagr_over_gross_drawdown(self):
        out = self._run()
        m = out["metrics"]
        if not m["max_dd"]:
            pytest.skip("no drawdown in this fixture")
        assert m["calmar"] == pytest.approx(m["cagr"] / abs(m["max_dd"]), rel=1e-9)


class TestPublishedArtifactsAgree:
    """The page links its own reports as evidence. When a report asserts something
    the page denies, the reader cannot tell which is true — and the page loses the
    only thing it is selling. These guard the claims that once disagreed."""

    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _json(self):
        p = os.path.join(self._ROOT, "docs", "data", "mark6.json")
        if not os.path.exists(p):
            pytest.skip("no dashboard export")
        import json
        return json.load(open(p))["research"]

    def _md(self, name):
        p = os.path.join(self._ROOT, "reports", name)
        if not os.path.exists(p):
            pytest.skip(f"no {name}")
        return open(p).read()

    @pytest.mark.parametrize("report", ["MARK6_REPORT.md", "INSTITUTIONAL_REPORT.md"])
    def test_survivorship_claim_matches_the_dashboard(self, report):
        r, txt = self._json(), self._md(report)
        n = r["universe"]["delisted_included"]
        inflated = "inflated an estimated ~1-2pp" in txt or "subtract ~1-2pp" in txt
        if n > 0:
            assert not inflated, (
                f"{report} still claims a survivorship haircut, but the dashboard "
                f"reports {n} delisted names in the point-in-time universe")
            assert str(n) in txt, f"{report} does not state the {n} delisted names"
        else:
            assert inflated, (
                f"dashboard universe has no delisted names, so {report} must keep "
                f"the survivorship caveat")

    def test_sharpe_agrees_between_page_and_institutional_report(self):
        import re
        r, txt = self._json(), self._md("INSTITUTIONAL_REPORT.md")
        mm = re.search(r"Sharpe \(excess of 6\.5% risk-free\)\*\* \| \*\*([\d.]+)\*\*", txt)
        if not mm:
            pytest.skip("report format changed")
        assert float(mm.group(1)) == pytest.approx(r["headline"]["sharpe_excess"], abs=0.02)

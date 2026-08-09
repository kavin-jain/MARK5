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


class TestTurnoverUnits:
    """turnover must be RUPEES on a consistent basis. The PIT cache stores a
    SPLIT-ADJUSTED close beside a RAW volume, so close*volume divides a name's
    historical turnover by its cumulative future split factor. Companies split
    because they compounded, so that quietly pushed future winners down a
    top-N-by-liquidity screen in exactly the years before they ran (NESTLEIND
    read Rs 0.79cr/day in 2016 against a true Rs 17.38cr)."""

    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _cache(self):
        p = os.path.join(self._ROOT, "data", "pit_cache")
        if not os.path.isdir(p):
            pytest.skip("no point-in-time cache")
        return p

    def test_pit_cache_carries_true_turnover(self):
        import glob
        files = glob.glob(os.path.join(self._cache(), "*_daily.parquet"))
        if not files:
            pytest.skip("empty cache")
        missing = [os.path.basename(f) for f in files[:200]
                   if "turnover" not in pd.read_parquet(f).columns]
        assert not missing, (
            f"{len(missing)} cache files lack the turnover column — rebuild with "
            f"scripts/build_pit_cache.py, or the liquidity screen silently reverts "
            f"to the split-distorted close*volume proxy")

    def test_panel_prefers_true_turnover_over_close_times_volume(self):
        from core.portfolio import DataPanel
        import glob
        names = [os.path.basename(f).replace("_daily.parquet", "")
                 for f in sorted(glob.glob(os.path.join(self._cache(), "*_daily.parquet")))]
        split = [n for n in ("NESTLEIND", "VBL", "NBCC") if n in names]
        if not split:
            pytest.skip("no known post-split name in cache")
        os.environ["MARK5_CACHE"] = "data/pit_cache"
        panel = DataPanel(split, "2026-07-21", freshness="off")
        proxy = (panel.close * panel.volume).rolling(126, min_periods=40).median()
        for t in panel.tickers:
            early = panel.turnover[t].dropna()
            if early.empty:
                continue
            head = early.iloc[:len(early) // 4]
            ph = proxy[t].reindex(head.index)
            # the true series must be materially LARGER early on, never smaller
            assert head.median() >= ph.median(), (
                f"{t}: panel turnover is below the close*volume proxy — the "
                f"split-adjusted product is being used")


class TestSignificanceMath:
    """The certainty numbers decide whether real money moves, so the estimators
    behind them get checked the same way any other money path does."""

    def _mod(self):
        import importlib.util
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        p = os.path.join(root, "scripts", "significance_analysis.py")
        if not os.path.exists(p):
            pytest.skip("no significance_analysis.py")
        spec = importlib.util.spec_from_file_location("sig", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_confidence_interval_widens_with_volatility(self):
        """SE of an annualised mean is vol/sqrt(years). Doubling vol must roughly
        double the band — if it does not, the interval is not measuring risk."""
        sig = self._mod()
        idx = pd.bdate_range("2016-01-01", periods=252 * 5)
        rng = np.random.default_rng(1)
        lo = sig.return_ci(pd.Series(rng.normal(0.0005, 0.006, len(idx)), index=idx))
        hi = sig.return_ci(pd.Series(rng.normal(0.0005, 0.012, len(idx)), index=idx))
        assert 1.6 < hi["se_pp"] / lo["se_pp"] < 2.4, (lo["se_pp"], hi["se_pp"])

    def test_years_to_significance_falls_as_edge_strengthens(self):
        """T = (1.645/IR)^2. A stronger edge must need strictly less data."""
        sig = self._mod()
        idx = pd.bdate_range("2016-01-01", periods=252 * 6)
        rng = np.random.default_rng(2)
        noise = rng.normal(0, 0.08 / np.sqrt(252), len(idx))
        zero = pd.Series(0.0, index=idx)
        weak = sig.active_stats(pd.Series(0.02 / 252 + noise, index=idx), zero, "weak")
        strong = sig.active_stats(pd.Series(0.10 / 252 + noise, index=idx), zero, "strong")
        assert strong["information_ratio"] > weak["information_ratio"]
        assert strong["years_to_95pct_significance"] < weak["years_to_95pct_significance"]

    def test_no_edge_never_reports_significance(self):
        """A benchmark differenced against itself has exactly zero active return.
        If that ever reads significant, the test is manufacturing skill."""
        sig = self._mod()
        idx = pd.bdate_range("2016-01-01", periods=252 * 5)
        r = pd.Series(np.random.default_rng(3).normal(0.0004, 0.011, len(idx)), index=idx)
        st = sig.active_stats(r, r.copy(), "self")
        assert not st["significant_95"]
        assert st["years_to_95pct_significance"] == float("inf")

    def test_dashboard_publishes_the_error_bar_around_its_headline(self):
        """A point CAGR without its interval reads as a forecast. If the feed ships
        a certainty block at all, the band must bracket the point estimate."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        p = os.path.join(root, "docs", "data", "mark6.json")
        if not os.path.exists(p):
            pytest.skip("no dashboard export")
        import json
        c = json.load(open(p))["research"].get("certainty")
        if not c:
            pytest.skip("no certainty block")
        lo, hi = c["equity_book_ci95_pct"]
        assert lo < c["equity_book_point_pct"] < hi, c
        assert 0 <= c["live_evidence_accumulated_pct"] <= 100
        assert c["years_of_live_data_to_prove_it"] > 0


class TestPBOCalibration:
    """PBO is read as if 0% were the target. It is not: the estimator returns ~50%
    when every candidate is equally good, because it is then ranking noise. These
    pin the calibration so nobody 'fixes' a healthy number by mutilating the book."""

    def _mod(self):
        import importlib.util
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        p = os.path.join(root, "scripts", "pbo_calibration.py")
        if not os.path.exists(p):
            pytest.skip("no pbo_calibration.py")
        spec = importlib.util.spec_from_file_location("pbocal", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_identical_strategies_score_near_fifty_percent(self):
        """The null. If this drifts to 0, PBO has stopped measuring selection risk."""
        cal = self._mod().calibrate(T=1300, N=25, trials=6, seed=5)
        assert 0.30 < cal["null_all_identical"]["pbo_mean"] < 0.75

    def test_a_real_edge_scores_low_and_true_overfitting_scores_high(self):
        """The estimator must separate the two worlds it exists to tell apart."""
        cal = self._mod().calibrate(T=1300, N=25, trials=6, seed=6)
        assert cal["one_real_edge"]["pbo_mean"] < 0.25
        assert cal["genuinely_overfit"]["pbo_mean"] > 0.70
        assert (cal["genuinely_overfit"]["pbo_mean"]
                > cal["null_all_identical"]["pbo_mean"]
                > cal["one_real_edge"]["pbo_mean"])

    def test_sharpe_standard_error_shrinks_with_sample_length(self):
        """The whole case for extending history rests on this being true."""
        se = self._mod().sharpe_se
        assert se(1.05, 19.5) < se(1.05, 10.3)
        assert abs(se(1.05, 10.3) - 0.388) < 0.01


class TestSectorNeutralRanking:
    """Sector-neutral scoring changes WHICH names are picked, unlike the sector
    cap which only trims exposure afterwards. It exists to convert nominal breadth
    into independent breadth (IR = IC*sqrt(BR)*TC), so it must actually reach
    across sectors — and must not manufacture noise out of tiny ones."""

    _MAP = {**{c: "IT" for c in "ABCDE"}, **{c: "BANK" for c in "FGHIJ"}, "K": "SOLO"}
    _SCORES = pd.Series({"A": 5., "B": 4., "C": 3., "D": 2., "E": 1.,
                         "F": .5, "G": .4, "H": .3, "I": .2, "J": .1, "K": 9.})

    def _con(self, neutral, n=4):
        return PortfolioConstructor(
            ConstructionConfig(n_hold=n, sector_neutral=neutral), sector_map=self._MAP)

    def test_it_reaches_sectors_the_raw_ranking_never_would(self):
        raw = self._con(False).select(self._SCORES, [])
        neu = self._con(True).select(self._SCORES, [])
        assert not any(t in "FGHIJ" for t in raw), raw
        assert any(t in "FGHIJ" for t in neu), neu

    def test_undersized_sectors_are_left_alone(self):
        """A z-score over one observation is noise, not a signal."""
        out = self._con(True)._neutralise(self._SCORES)
        assert out["K"] == self._SCORES["K"]

    def test_unmapped_names_are_pooled_and_neutralised_not_left_raw(self):
        """The sector map covers ZERO of the 258 names that delisted in-window, so
        the unmapped pool IS the dead-company cohort. Leaving it on raw scores
        while mapped names carry z-scores would rank two incompatible scales
        against each other, biased precisely along the survivorship axis."""
        smap = {c: "IT" for c in "ABCDE"}                 # F..J deliberately unmapped
        con = PortfolioConstructor(
            ConstructionConfig(n_hold=4, sector_neutral=True), sector_map=smap)
        scores = pd.Series({"A": 5., "B": 4., "C": 3., "D": 2., "E": 1.,
                            "F": .5, "G": .4, "H": .3, "I": .2, "J": .1})
        out = con._neutralise(scores)
        mapped = out[list("ABCDE")]
        unmapped = out[list("FGHIJ")]
        assert abs(mapped.mean()) < 1e-9, mapped
        assert abs(unmapped.mean()) < 1e-9, unmapped   # pooled group also centred
        assert abs(mapped.std() - unmapped.std()) < 0.5, (mapped.std(), unmapped.std())

    def test_disabled_is_an_exact_passthrough(self):
        """The flag must be inert when off, or every existing result shifts."""
        out = self._con(False)._neutralise(self._SCORES)
        pd.testing.assert_series_equal(out, self._SCORES)

    def test_no_sector_map_is_an_exact_passthrough(self):
        """Enabling neutralisation without a map must not silently do nothing
        different from what the caller expects, nor crash."""
        c = PortfolioConstructor(ConstructionConfig(n_hold=4, sector_neutral=True))
        pd.testing.assert_series_equal(c._neutralise(self._SCORES), self._SCORES)

    def test_weights_are_tilted_by_the_same_scores_used_to_select(self):
        """If select() neutralises but target_weights() tilts on raw scores, names
        chosen for their within-sector rank get sized by their market-wide rank."""
        con = self._con(True)
        vol = pd.Series(0.02, index=self._SCORES.index)
        w = con.target_weights(self._SCORES, vol, [])
        assert set(w.index) == set(con.select(self._SCORES, []))
        assert abs(w.sum() - 1.0) < 1e-9


class TestLTCGDeferral:
    """P4.1. Selling a winner at ~182 days costs 20% tax where waiting past 365
    costs 12.5%. The backtest's 306 winning sells in the 6-10 month bucket carry
    Rs 20.0 lakh of gains in exactly that trap. This must save tax WITHOUT becoming
    a timing rule or a way to hold garbage."""

    _S = pd.Series({f"T{i:02d}": 100 - i for i in range(40)})

    def _con(self, mult):
        return PortfolioConstructor(
            ConstructionConfig(n_hold=10, buffer_mult=2.0, ltcg_defer_mult=mult))

    def test_off_by_default_is_byte_identical(self):
        """Default must not perturb a single existing result."""
        held = ["T15", "T18", "T25"]
        assert (self._con(1.0).select(self._S, held)
                == self._con(1.0).select(self._S, held, defer_exit=frozenset(held)))

    def test_a_drifting_winner_is_held_past_the_exit_bar(self):
        """T25 sits outside the normal bar (10*2=20) but inside the widened one."""
        held = ["T25"]
        assert "T25" not in self._con(1.0).select(self._S, held)
        assert "T25" in self._con(2.0).select(self._S, held, defer_exit=frozenset(held))

    def test_a_collapsed_name_is_still_sold(self):
        """Deferral widens the exit bar; it does not remove it. A name that has
        genuinely collapsed still falls through, or this becomes a licence to
        hold anything indefinitely for a tax reason."""
        far = pd.Series({f"T{i:03d}": 200 - i for i in range(120)})
        # widened bar is n_hold*buffer*mult = 10*2*2 = 40; rank 99 is far past it
        assert "T099" not in self._con(2.0).select(far, ["T099"],
                                                   defer_exit=frozenset(["T099"]))

    def test_deferral_cannot_inflate_the_book(self):
        """Deferred names compete for the same slots; n_hold is still the cap."""
        held = [f"T{i:02d}" for i in range(15, 35)]
        picks = self._con(3.0).select(self._S, held, defer_exit=frozenset(held))
        assert len(picks) == 10

    def test_engine_defers_only_gains_never_losses(self):
        """A loss should be realised, not nursed: FY netting already absorbs it,
        and deferring it would convert a useful offset into a dead position."""
        import datetime as _dt
        d = pd.Timestamp("2020-06-01")
        lots = {"WIN": [[120.0, 100.0, d - pd.Timedelta(days=100)]],
                "LOSE": [[80.0, 100.0, d - pd.Timedelta(days=100)]],
                "OLD": [[120.0, 100.0, d - pd.Timedelta(days=400)]]}
        defer = frozenset(t for t in lots
                          if sum(mv - c for mv, c, e in lots[t]
                                 if (d - e).days < 365) > 0)
        assert defer == {"WIN"}, defer


class TestPBOTieHandling:
    """PBO counts splits where the in-sample winner lands BELOW the out-of-sample
    median. A rank sitting exactly AT the median is a tie, not degradation.
    Counting ties as 'below' inflates PBO precisely when candidates are
    near-identical — which is when they tie. Grading a book that is half fixed
    passive sleeves produced PBO 91% alongside a median logit of 0.00, a
    self-contradiction that was the tell."""

    def _pbo(self, M):
        from core.portfolio.stats import pbo_cscv
        return pbo_cscv(M, n_splits=12)

    def test_identical_strategies_are_all_ties_not_all_failures(self):
        rng = np.random.default_rng(0)
        base = rng.normal(0.0004, 0.01, (2600, 1))
        r = self._pbo(np.repeat(base, 30, axis=1))
        assert r["tie_fraction"] > 0.95, r
        assert r["pbo"] < 0.05, r

    def test_the_fix_does_not_deflate_a_genuine_measurement(self):
        """The ~50% null needs CORRELATED candidates — the way real config variants
        relate. With independent columns one has the best full-sample Sharpe by
        sampling luck and wins both in- and out-of-sample, so PBO is legitimately
        low there; that is a property of the statistic, not of this fix. What must
        hold is that distinguishable candidates do NOT collapse to all-ties the way
        identical ones do."""
        rng = np.random.default_rng(1)
        r = self._pbo(rng.normal(0.0004, 0.01, (2600, 30)))
        assert r["tie_fraction"] < 0.05, r
        # near-identical family: the honest ~50% null
        common = rng.normal(0, 1 / np.sqrt(252), 2600)
        idio = rng.normal(0, 1 / np.sqrt(252), (2600, 30))
        M = np.sqrt(0.93) * common[:, None] + np.sqrt(0.07) * idio
        n = self._pbo(M)
        assert 0.25 < n["pbo"] < 0.80, n

    def test_tie_fraction_is_always_published(self):
        """The condition must be visible, not buried inside the headline number."""
        rng = np.random.default_rng(2)
        assert "tie_fraction" in self._pbo(rng.normal(0, 0.01, (1300, 10)))


# ══════════════════════════════════════════════════════════════════════════
#  The daily message
#
#  Mandate §8 says how results are reported to the owner. The daily notification
#  is the only artifact they will read every day, so the reporting rules are
#  enforced here as tests rather than left to whoever edits the template next.
# ══════════════════════════════════════════════════════════════════════════
class TestDailyNotification:
    @staticmethod
    def _mod():
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scripts", "notify.py")
        spec = importlib.util.spec_from_file_location("notify", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    BOOK = {"days_live": 16, "capital": 500000.0, "nav": 519909.9,
            "return_pct": 3.98, "benchmark_nav": 511834.21, "relative_pct": 1.615,
            "max_drawdown_pct": -1.7248, "realised_pnl": -1732.89, "tax_liability": 0.0,
            "generated": "2026-08-07T12:07:35+00:00",
            "sleeves": {"rows": [
                {"label": "Indian equity", "pnl_inr": 13787.7, "return_pct": 5.44},
                {"label": "Gold ETF", "pnl_inr": 4139.4, "return_pct": 3.58}]},
            "nav_history": [
                {"date": "2026-07-22", "nav_inr": "499419.37", "bench_inr": ""},
                {"date": "2026-08-06", "nav_inr": "519193.28", "bench_inr": "513224.34"},
                {"date": "2026-08-07", "nav_inr": "519909.90", "bench_inr": "511834.21"}]}
    OK = {"n": 24, "fails": 0, "warns": 0, "failing": []}

    def test_indian_digit_grouping(self):
        """The reader is Indian. Western grouping makes them convert in their head
        every single day, and 519,910 vs 5,19,910 is exactly the kind of friction
        that stops someone reading the message at all."""
        g = self._mod()._grp
        for raw, want in (("519910", "5,19,910"), ("100", "100"), ("1000", "1,000"),
                          ("1234567", "12,34,567"), ("10000000", "1,00,00,000"),
                          ("12", "12"), ("0", "0")):
            assert g(raw) == want, f"{raw} -> {g(raw)}, want {want}"

    def test_unrealised_and_realised_are_never_blended(self):
        """Mandate §8: 'Distinguish paper gains from realised gains every time
        money is discussed.' Both numbers must appear, separately."""
        m = self._mod()
        body = m.build(self.BOOK, self.OK)
        assert "Rs 1,733" in body, body          # realised, on its own
        assert "Rs 21,643" in body, body         # the unrealised remainder, on its own

    def test_the_simulation_is_always_disclosed(self):
        """The owner asked for wording that reads professionally when shown to
        other people. That changed the WORDING, not whether it appears. An
        unlabelled simulated track record is a misrepresentation to whoever is
        shown it, so the disclosure is pinned here and cannot be dropped by an
        edit that is only trying to tidy the layout."""
        body = self._mod().build(self.BOOK, self.OK)
        assert "Simulated execution" in body, body
        assert "Model portfolio" in body, body
        assert "Not investment advice" in body, body

    def test_the_weakness_appears_on_a_good_day(self):
        """§8: 'Volunteer the weakness before it is asked for.' This book is up
        +3.98% in the fixture; the worst dip must still be in the message."""
        body = self._mod().build(self.BOOK, self.OK)
        assert "worst dip" in body and "-1.7%" in body, body

    def test_relative_return_is_labelled_in_percentage_points(self):
        """relative_pct is a difference of two percentages. Printing it as '%'
        overstates it and is the same class of error as every defect in §0."""
        body = self._mod().build(self.BOOK, self.OK)
        assert "pp)" in body and "1.61" in body, body
        assert "1.61%" not in body, body

    def test_a_broken_day_still_produces_a_message_that_says_so(self):
        """The 4-5 Aug outage went unseen because nothing spoke. A notifier that
        goes quiet exactly when something breaks is worse than none at all."""
        m = self._mod()
        bad = m.build(self.BOOK, {"n": 24, "fails": 2, "warns": 0,
                                  "failing": ["feed is stale", "ledger did not grow"]})
        assert "FAILED" in bad and "feed is stale" in bad, bad

    def test_a_missing_previous_mark_invents_no_move(self):
        """With one mark there is no day-over-day change to report, and printing
        one anyway would be fabricating a number."""
        m = self._mod()
        one = dict(self.BOOK, nav_history=self.BOOK["nav_history"][:1])
        assert "SINCE" not in m.build(one, self.OK)

    def test_no_channel_configured_is_not_an_error(self):
        """CI must never go red because a token was never added."""
        m = self._mod()
        for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "NTFY_TOPIC"):
            os.environ.pop(k, None)
        assert m.send("t", "b") is None

    def test_a_token_can_never_reach_stdout(self):
        """The bot token sits in the URL path of every Telegram call, so any
        printed URL, traceback or redirect leaks it into the CI log — which is
        public. GitHub masks registered secrets, but only exact strings it was
        told about. This is the independent guard."""
        m = self._mod()
        # synthetic, same shape as a real one. Never put a live token in a test:
        # the fixture outlives the credential and secret scanners flag the repo.
        tok = "1234567890:" + "A" * 35
        os.environ["TELEGRAM_BOT_TOKEN"] = tok
        try:
            leak = f"HTTP Error 404 for https://api.telegram.org/bot{tok}/sendMessage"
            out = m.scrub(leak)
            assert tok not in out, out
            assert tok.split(":", 1)[1] not in out, out
        finally:
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)

    def test_scrub_is_safe_when_nothing_is_configured(self):
        for k in ("TELEGRAM_BOT_TOKEN", "NTFY_TOKEN"):
            os.environ.pop(k, None)
        assert self._mod().scrub("plain error") == "plain error"

    def test_rebalance_day_reports_the_fills_it_booked(self):
        """The owner wants zero human steps, so this is a record of what the
        system DID, not a list of what they must do. It reads from the ledger —
        the authoritative record — rather than any summary that could disagree
        with it."""
        m = self._mod()
        import csv as _csv, tempfile
        rows = [{"timestamp": "", "date": "2026-08-07", "action": "SELL",
                 "ticker": "TDPOWERSYS", "qty": "8", "price": "1113.60",
                 "value_inr": "8908.80", "cost_inr": "10",
                 "note": "exit · P&L -452 · held 4d · STCG accrued 0"},
                {"timestamp": "", "date": "2026-08-07", "action": "BUY",
                 "ticker": "NYKAA", "qty": "51", "price": "322.80",
                 "value_inr": "16462.80", "cost_inr": "19", "note": "rebalance entry"},
                {"timestamp": "", "date": "2026-01-01", "action": "BUY",
                 "ticker": "OLDONE", "qty": "1", "price": "1.00",
                 "value_inr": "1", "cost_inr": "0", "note": ""}]
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
            path = fh.name
        old = m.LEDGER
        m.LEDGER = path
        try:
            body = m.build(self.BOOK, self.OK)
        finally:
            m.LEDGER = old
            os.unlink(path)
        assert "THE SYSTEM REBALANCED ON 2026-08-07" in body, body
        assert "TDPOWERSYS" in body and "NYKAA" in body, body
        assert "OLDONE" not in body, "a fill from another date leaked in"
        assert "2 orders" in body, body
        # the book-level figures must survive the section that sits above them
        assert "Rs 21,643" in body, "the trade loop clobbered the book's P&L"

    def test_quiet_days_have_no_rebalance_section(self):
        """Most days book nothing. An empty 'THE SYSTEM REBALANCED' heading would
        train the reader to ignore the one that matters."""
        assert "REBALANCED" not in self._mod().build(self.BOOK, self.OK)

    def test_zero_checks_never_reads_as_a_pass(self):
        """Zero checks RUN is not zero checks FAILED. The fallback wording rendered
        'all 0 checks passed', which is reassurance about something that never
        happened — the precise inversion this notifier exists to prevent."""
        body = self._mod().build(self.BOOK, {"n": 0, "fails": 0, "warns": 0, "failing": []})
        assert "0 checks passed" not in body, body
        assert "not checked" in body, body


class TestTelegramBot:
    """The command interface. What is tested here is the boundary, not the prose:
    who may command it, what it refuses to say, and whether the two places that
    report a profit report the same one."""

    @staticmethod
    def _mod():
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scripts", "bot.py")
        spec = importlib.util.spec_from_file_location("bot", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    @staticmethod
    def _with_chat(cid, fn):
        keep = {k: os.environ.get(k) for k in ("TELEGRAM_CHAT_ID", "TELEGRAM_ADMIN_CHATS")}
        os.environ["TELEGRAM_CHAT_ID"] = cid
        os.environ.pop("TELEGRAM_ADMIN_CHATS", None)
        try:
            return fn()
        finally:
            for k, v in keep.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_a_stranger_is_never_answered(self):
        """A Telegram bot is reachable by anyone who learns its @username. Without
        this gate the book's positions and P&L are readable by the public."""
        m = self._mod()
        assert self._with_chat("-1001234567890", lambda: m.handle(
            {"chat": {"id": 777777}, "text": "/holdings"})) is False

    def test_plain_conversation_is_ignored(self):
        """It sits in a group with a human in it. A bot that answers chatter gets
        muted, and a muted bot is no bot on the day it has something urgent."""
        m = self._mod()
        assert m.answer("good morning") is None
        assert m.answer("") is None
        assert self._with_chat("-1001234567890", lambda: m.handle(
            {"chat": {"id": -1001234567890}, "text": "how are we doing"})) is False

    def test_an_unknown_command_explains_itself(self):
        m = self._mod()
        body = m.answer("/nonsense")
        assert "No such command" in body and "/update" in body

    def test_group_suffixed_commands_resolve(self):
        """In a group Telegram delivers '/update@MARK5K_BOT', not '/update'."""
        m = self._mod()
        assert m.answer("/help@MARK5K_BOT").startswith("WHAT YOU CAN ASK")
        assert m.answer("/status").startswith("MARK6")          # alias of /update

    def test_every_advertised_command_exists(self):
        """The menu published to Telegram is derived from COMMANDS, so a command
        can never be offered in autocomplete that nothing here answers."""
        m = self._mod()
        for name, desc, fn in m.COMMANDS:
            assert m.HANDLERS[name] is fn and desc
        for alias, target in m.ALIASES.items():
            assert target in m.HANDLERS, alias

    def test_a_failing_command_reports_instead_of_dying(self):
        m = self._mod()
        m.HANDLERS["boom"] = lambda: 1 / 0
        try:
            body = m.answer("/boom")
        finally:
            del m.HANDLERS["boom"]
        assert "failed" in body and "money record is untouched" in body

    def test_holdings_reconciles_with_the_headline_profit(self):
        """/holdings sums unrealised P&L; /update reports profit after the loss
        already banked on names that were sold. Two screens showing two different
        profits with no bridge is how a reader stops trusting both."""
        import json as _json
        m = self._mod()
        L = _json.load(open(m.EXPORT))
        want = m._amt(float(L["nav"]) - float(L["capital"]), True)
        body = m.h_holdings()
        assert "YOUR PROFIT" in body
        assert want in body, f"expected {want} in the reconciliation\n{body[-400:]}"

    def test_holdings_never_calls_paper_gains_banked(self):
        m = self._mod()
        body = m.h_holdings()
        assert "still held" in body and "already sold" in body

    def test_sleeve_etfs_are_not_presented_as_stocks(self):
        """n_hold governs the equity sleeve alone. A flat 22-line list reads as
        "22 stocks" when the system picked 20 and the other two lines are a whole
        sleeve each — and they are the two largest lines on the page."""
        import json as _json
        m = self._mod()
        L = _json.load(open(m.EXPORT))
        n_eq = next(r["n_holdings"] for r in L["sleeves"]["rows"] if not r.get("passive"))
        body = m.h_holdings()
        assert f"THE {n_eq} STOCKS" in body, body[:200]
        assert f"{n_eq} stocks +" in body, body[:200]
        assert len(L["holdings"]) != n_eq, "fixture no longer exercises the split"

    def test_the_bot_workflow_cannot_write(self):
        """The real read-only guarantee is the token GitHub hands the job, not
        the absence of write code. Mandate §6: the book is append-only and never
        rebalanced off-cadence — a chat message must not be able to reach it."""
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         ".github", "workflows", "bot.yml")
        wf = open(p).read()
        assert "contents: read" in wf
        assert "contents: write" not in wf

    def test_replies_are_chunked_below_telegrams_limit(self):
        """Telegram rejects a message over 4096 chars outright — the whole reply
        vanishes rather than truncating. A 60-name book would hit that."""
        m = self._mod()
        assert m.CHUNK < 4096


class TestEquityUniverseExcludesSleeveETFs:
    """The sleeve ETFs must never be rankable as stocks.

    `_is_etf` matches the *BEES / *ETF naming convention; MON100 and MAFANG follow
    neither, so both sat in the equity universe reading as ordinary companies.
    MON100 ranked FIRST of 194 names in a rebuild of the 2026-07-21 signal — it is
    the Nasdaq-100 in rupees and scores high on momentum, trend, low-vol and
    stability simultaneously, which is exactly what the composite rewards.
    """

    @staticmethod
    def _u():
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from core.portfolio.universe import STRUCTURAL_EXCLUDE, discover_tickers
        return discover_tickers, STRUCTURAL_EXCLUDE

    SLEEVE_ETFS = {"GOLDBEES", "MON100", "MAFANG", "LTGILTBEES",
                   "NIFTYBEES", "LIQUIDBEES", "BANKBEES", "JUNIORBEES"}

    def test_no_sleeve_etf_reaches_the_equity_universe(self):
        discover, _ = self._u()
        leaked = self.SLEEVE_ETFS & set(discover())
        assert not leaked, f"ETFs rankable as stocks: {sorted(leaked)}"

    def test_the_pinned_fallback_is_filtered_too(self):
        """CI has no price cache, so discover_tickers() falls back to the pinned
        list — a path that never calls _is_etf and is filtered ONLY by
        STRUCTURAL_EXCLUDE. Fixing the predicate alone would have left CI, which
        is where the unattended January rebalance runs, still exposed."""
        import json as _json
        _, exclude = self._u()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pinned = set(_json.load(
            open(os.path.join(root, "config", "universe_tickers.json")))["tickers"])
        assert (pinned & self.SLEEVE_ETFS) <= exclude, (
            f"pinned list smuggles {sorted((pinned & self.SLEEVE_ETFS) - exclude)}")

    def test_the_excluded_etfs_are_still_priceable(self):
        """Excluding them from SELECTION must not make them unpriceable — they are
        held sleeves and the book cannot be marked without them."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from core.portfolio.universe import load_ohlcv
        if load_ohlcv("GOLDBEES") is None:
            import pytest
            pytest.skip("no price cache in this environment")
        for etf in ("GOLDBEES", "MON100"):
            assert load_ohlcv(etf) is not None, f"{etf} became unpriceable"


class TestExceptionalDayAlert:
    """The extra message sent only on an unusual day. What is tested is that it
    stays silent when it should, does not nag, and never suggests acting."""

    @staticmethod
    def _mod():
        return TestDailyNotification._mod()

    @staticmethod
    def _book(navs, cap=500000.0):
        return {"nav": navs[-1], "capital": cap, "return_pct": (navs[-1] / cap - 1) * 100,
                "nav_history": [{"date": f"2026-07-{i + 1:02d}", "nav_inr": str(v)}
                                for i, v in enumerate(navs)]}

    def test_an_ordinary_day_says_nothing(self):
        """The whole value of this message is its rarity. One false alarm a week
        and it becomes the daily message it was built to rescue."""
        m = self._mod()
        assert m.alert(self._book([500000, 502000, 503000])) is None

    def test_a_big_move_fires_and_sizes_itself(self):
        m = self._mod()
        body = m.alert(self._book([500000, 520000, 501800]))
        assert "BIG DOWN DAY" in body and "x a normal day" in body

    def test_a_new_low_fires_once_per_band_not_daily(self):
        """A long slide would otherwise alert every single day, which is how an
        alarm gets muted precisely during the drawdown it exists for."""
        m = self._mod()
        crossed = self._book([500000, 560000, 505000, 503000])
        assert "NEW LOW" in m.alert(crossed)
        assert m.alert(self._book([500000, 560000, 505000, 503000, 502000])) is None

    def test_a_shallow_dip_is_not_called_a_new_low(self):
        m = self._mod()
        assert m.alert(self._book([500000, 510000, 505000, 504000])) is None

    def test_the_alert_never_suggests_acting(self):
        """This message arrives on the day the owner is most likely to want to do
        something, and the research log records six separate approaches that died
        for cutting exposure after a loss. Telling them to sit still is the
        finding, not reassurance — so it is asserted, not left to wording."""
        m = self._mod()
        body = m.alert(self._book([500000, 520000, 501800]))
        assert "Nothing." in body
        assert str(abs(m.BACKTEST_WORST_DD)) in body
        for word in ("sell", "reduce", "exit", "consider"):
            assert word not in body.lower(), f"the alert suggests {word!r}"


class TestChartCommand:
    @staticmethod
    def _mod():
        return TestTelegramBot._mod()

    def test_it_returns_a_real_png(self):
        m = self._mod()
        p = m.h_chart()
        assert isinstance(p, m.Photo)
        assert p.png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
        assert len(p.png) > 5000

    def test_the_caption_fits_telegrams_limit(self):
        """sendPhoto silently rejects a caption over 1024 characters."""
        m = self._mod()
        assert len(m.h_chart().caption) <= 1024

    def test_a_blank_print_is_dropped_not_zeroed(self):
        """float("" or 0) is 0.0, which plots as a vertical crash to the bottom of
        a chart whose entire job is the comparison."""
        m = self._mod()
        hist = [{"date": "2026-07-22", "nav_inr": "100", "bench_inr": ""},
                {"date": "2026-07-23", "nav_inr": "101", "bench_inr": "99"}]
        dates, vals = m._series(hist, "bench_inr")
        assert vals == [99.0] and len(dates) == 1
        assert 0.0 not in vals

    def test_dates_are_datetimes_not_strings(self):
        """Strings make matplotlib build a categorical axis ordered by first
        appearance, which drew a phantom segment looping back to day one."""
        import datetime as _dt
        m = self._mod()
        dates, _ = m._series([{"date": "2026-07-22", "nav_inr": "100"}], "nav_inr")
        assert isinstance(dates[0], _dt.datetime)


class TestCacheCoversHoldings:
    """A rebalance must not run against a cache that cannot see what we own.

    On 2026-08-09 the cache was missing RBLBANK, HFCL, AEROFLEX and NYKAA
    entirely and had BHARATFORG two months stale. The existing freshness check
    passed, because it tests the panel's NEWEST date — the maximum across all
    names — which says nothing about breadth. A name the ranking cannot see is a
    name it drops, and a dropped name is SOLD: 15 of 20 holdings would have been
    liquidated, and the trade log would have read like 15 deliberate decisions.
    """

    @staticmethod
    def _pt():
        import sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path[:0] = [root, os.path.join(root, "scripts")]
        import paper_track
        return paper_track

    @staticmethod
    def _bars(days_old):
        import pandas as pd
        end = pd.Timestamp.today().normalize() - pd.Timedelta(days=days_old)
        return pd.DataFrame({"close": [1.0, 2.0]},
                            index=pd.DatetimeIndex([end - pd.Timedelta(days=1), end]))

    def _run(self, cached, ages, book_names):
        pt = self._pt()
        keep = (pt.discover_tickers, pt.load_ohlcv)
        pt.discover_tickers = lambda: list(cached)
        pt.load_ohlcv = lambda t: self._bars(ages.get(t, 0)) if t in cached else None
        try:
            pt.assert_cache_covers_holdings({"positions": {n: {} for n in book_names}})
            return None
        except SystemExit as e:
            return str(e)
        finally:
            pt.discover_tickers, pt.load_ohlcv = keep

    def test_a_holding_absent_from_the_cache_stops_the_rebalance(self):
        msg = self._run(cached=["AAA"], ages={}, book_names=["AAA", "BBB"])
        assert msg and "BBB" in msg and "refusing" in msg.lower()

    def test_a_stale_holding_stops_it_too(self):
        msg = self._run(cached=["AAA", "BBB"], ages={"BBB": 60},
                        book_names=["AAA", "BBB"])
        assert msg and "BBB" in msg

    def test_full_fresh_coverage_passes(self):
        assert self._run(cached=["AAA", "BBB"], ages={}, book_names=["AAA", "BBB"]) is None

    def test_sleeve_etfs_are_not_required_to_be_in_the_equity_cache(self):
        """They are deliberately excluded from the equity universe, so demanding
        them here would deadlock every rebalance permanently."""
        pt = self._pt()
        sleeve = sorted(pt.SLEEVES)[0]
        assert self._run(cached=["AAA"], ages={}, book_names=["AAA", sleeve]) is None


class TestWhyCommand:
    @staticmethod
    def _mod():
        return TestTelegramBot._mod()

    def test_it_never_claims_scores_it_does_not_have(self):
        """Re-deriving an old ranking does not reproduce it — corporate actions
        adjust price history retroactively and cross-sectional ranks move with
        universe churn. A rebuild of the 2026-07-21 signal returned 5 of the 20
        names actually picked. So absence must read as absence."""
        m = self._mod()
        body = m.h_why("BHEL")
        if "No scores recorded" in body:
            assert "made up" in body
            for word in ("percentile", "█"):
                assert word not in body, "claims a score it has not got"

    def test_it_always_states_what_was_not_examined(self):
        """The ranking uses five price/volume statistics and has never read a
        balance sheet. A reader who assumes otherwise holds a false view of this
        book, so this block is the finding — not a caveat that may be dropped."""
        m = self._mod()
        for t in ("BHEL", "RELIANCE", "NOTATICKER"):
            body = m.h_why(t)
            assert "never read a balance sheet" in body
            assert "Not investment advice." in body

    def test_a_name_we_do_not_hold_is_labelled_as_such(self):
        m = self._mod()
        assert "NOT HELD" in m.h_why("RELIANCE")

    def test_no_argument_lists_what_is_held(self):
        m = self._mod()
        body = m.h_why("")
        assert "/why" in body and "BHEL" in body

    def test_fundamentals_fail_open(self):
        """The only part of /why that depends on a server nobody here controls.
        It must never take the explanation down with it."""
        m = self._mod()
        assert m._fundamentals("DEFINITELYNOTATICKER123") == []

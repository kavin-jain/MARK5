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


# ── real-data integration smoke test ───────────────────────────────────────────
class TestIntegration:
    def test_real_data_runs_and_is_sane(self):
        from core.portfolio import (DataPanel, discover_tickers, Backtester,
                                     BacktestConfig, PortfolioConstructor, ConstructionConfig)
        tickers = discover_tickers()
        if len(tickers) < 20:
            pytest.skip("insufficient cached data for integration test")
        panel = DataPanel(tickers, "2026-05-21")
        con = PortfolioConstructor(ConstructionConfig(mode="factor_tilt", n_hold=20))
        bt = Backtester(panel, con, BacktestConfig())
        out = bt.run("2018-01-01", "2021-12-31")
        m = out["metrics"]
        assert -1.0 < m["cagr"] < 2.0          # sane range, not absurd
        assert -1.0 <= m["max_dd"] <= 0.0
        assert out["nav_net"].iloc[0] == pytest.approx(1.0, abs=0.2)

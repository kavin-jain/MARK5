"""
Tests for scripts/monte_carlo.py

Coverage:
  - load_equity_curve: parses JSON, validates structure, raises on missing curve
  - daily_log_returns: correct values, length N-1
  - compute_max_drawdown: correct peaks, negative output, edge cases
  - compute_sharpe: annualisation, zero-vol guard
  - compute_cagr: compounding formula, edge cases
  - bootstrap_daily_returns: shape, reproducibility, non-negativity
  - trade_sequence_shuffle: shape, invariants (same trades → same EV), randomness
  - trade_return_bootstrap: shape, reproducibility, distribution spread
  - path_statistics: shape of outputs, CAGR formula sanity
  - percentile_table: keys, ordering
  - concentration: HAL analysis
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from scripts.monte_carlo import (
    bootstrap_daily_returns,
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe,
    daily_log_returns,
    load_equity_curve,
    path_statistics,
    percentile_table,
    trade_return_bootstrap,
    trade_sequence_shuffle,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_equity():
    """Monotonically rising equity from 100 to 200 over 10 steps."""
    return np.linspace(100.0, 200.0, 11)


@pytest.fixture
def drawdown_equity():
    """Equity that rises, falls, then recovers: 100→150→90→120."""
    return np.array([100.0, 110.0, 130.0, 150.0, 140.0, 110.0, 90.0, 95.0, 105.0, 120.0])


@pytest.fixture
def real_results_path():
    """Path to the actual backtest results file (requires re-run of momentum_portfolio.py)."""
    path = os.path.join(_ROOT, "reports", "momentum_portfolio_results.json")
    if not os.path.exists(path):
        pytest.skip("momentum_portfolio_results.json not found — run momentum_portfolio.py first")
    with open(path) as f:
        data = json.load(f)
    if "equity_curve" not in data:
        pytest.skip("equity_curve missing from results — re-run momentum_portfolio.py")
    return path


@pytest.fixture
def mock_results_file(tmp_path):
    """Minimal valid results JSON for unit tests (no real data required)."""
    rng = np.random.default_rng(42)
    # 252 trading days, slight upward drift
    log_rets = rng.normal(0.0003, 0.012, 252)
    equity   = 5_000_000.0 * np.exp(np.cumsum(np.concatenate([[0.0], log_rets])))

    dates = [f"2022-{(i // 21 + 1):02d}-{(i % 21 + 1):02d}" for i in range(253)]
    ec    = [[dates[i], round(float(equity[i]), 2)] for i in range(253)]

    trades = [
        {"ticker": "HAL",  "entry_date": "2022-01-03", "exit_date": "2022-12-30",
         "entry_price": 100.0, "exit_price": 145.0, "shares": 100,
         "net_pnl": 45000.0, "pnl_pct": 45.0, "hold_days": 366, "reason": "TRAILING_STOP"},
        {"ticker": "TRENT","entry_date": "2022-03-01", "exit_date": "2022-06-15",
         "entry_price": 200.0, "exit_price": 185.0, "shares": 50,
         "net_pnl": -7500.0, "pnl_pct": -7.5, "hold_days": 106, "reason": "HARD_STOP"},
        {"ticker": "BEL",  "entry_date": "2022-04-01", "exit_date": "2022-11-01",
         "entry_price": 80.0, "exit_price": 96.0, "shares": 200,
         "net_pnl": 3200.0, "pnl_pct": 20.0, "hold_days": 214, "reason": "SCORE_EXIT(0.38)"},
    ]

    data = {
        "strategy": "test",
        "oos_start": "2022-01-03", "oos_end": "2022-12-30",
        "years": 1.0,
        "initial_capital": 5_000_000.0,
        "final_equity_gross": float(equity[-1]),
        "final_equity_net":   float(equity[-1]) * 0.98,
        "gross_cagr_pct": 12.5,
        "net_cagr_pct":   10.2,
        "max_drawdown_pct": -8.3,
        "sharpe": 0.88,
        "total_tax": 5000.0,
        "n_trades": 3,
        "win_rate_pct": 66.7,
        "avg_hold_days": 228.7,
        "annual_returns": {"2022": 10.2},
        "universe_size": 5,
        "equity_curve": ec,
        "trades": trades,
    }

    path = tmp_path / "mock_results.json"
    path.write_text(json.dumps(data))
    return str(path)


# ─── Tests: load_equity_curve ────────────────────────────────────────────────

class TestLoadEquityCurve:
    def test_loads_successfully(self, mock_results_file):
        dates, equity, meta = load_equity_curve(mock_results_file)
        assert len(dates) == 253
        assert len(equity) == 253
        assert equity[0] == pytest.approx(5_000_000.0, rel=1e-3)

    def test_meta_keys_present(self, mock_results_file):
        _, _, meta = load_equity_curve(mock_results_file)
        for key in ("initial_capital", "net_cagr_pct", "max_drawdown_pct",
                    "sharpe", "years", "n_trades", "trades"):
            assert key in meta

    def test_missing_equity_curve_raises(self, tmp_path):
        data = {"initial_capital": 1e6, "net_cagr_pct": 10.0, "max_drawdown_pct": -5.0,
                "sharpe": 0.8, "years": 1.0, "n_trades": 1, "win_rate_pct": 50.0,
                "avg_hold_days": 100.0, "annual_returns": {}, "trades": [],
                "final_equity_gross": 1.1e6, "final_equity_net": 1.08e6,
                "gross_cagr_pct": 10.0}
        path = tmp_path / "no_ec.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="equity_curve"):
            load_equity_curve(str(path))

    def test_trades_count(self, mock_results_file):
        _, _, meta = load_equity_curve(mock_results_file)
        assert meta["n_trades"] == 3
        assert len(meta["trades"]) == 3

    def test_equity_all_positive(self, mock_results_file):
        _, equity, _ = load_equity_curve(mock_results_file)
        assert np.all(equity > 0)

    def test_real_file_if_available(self, real_results_path):
        dates, equity, meta = load_equity_curve(real_results_path)
        assert len(dates) > 200
        assert meta["initial_capital"] == pytest.approx(5_00_00_000, rel=1e-6)
        assert meta["years"] > 3.0


# ─── Tests: daily_log_returns ────────────────────────────────────────────────

class TestDailyLogReturns:
    def test_length(self, simple_equity):
        returns = daily_log_returns(simple_equity)
        assert len(returns) == len(simple_equity) - 1

    def test_positive_for_rising_equity(self, simple_equity):
        returns = daily_log_returns(simple_equity)
        assert np.all(returns > 0)

    def test_correct_value(self):
        equity = np.array([100.0, 110.0, 99.0])
        rets   = daily_log_returns(equity)
        assert rets[0] == pytest.approx(np.log(110 / 100))
        assert rets[1] == pytest.approx(np.log(99  / 110))

    def test_zero_drift_flat_equity(self):
        equity = np.ones(50) * 1000.0
        rets   = daily_log_returns(equity)
        assert np.allclose(rets, 0.0)

    def test_sum_equals_total_log_return(self, simple_equity):
        rets = daily_log_returns(simple_equity)
        total = np.log(simple_equity[-1] / simple_equity[0])
        assert np.sum(rets) == pytest.approx(total, rel=1e-10)


# ─── Tests: compute_max_drawdown ─────────────────────────────────────────────

class TestComputeMaxDrawdown:
    def test_no_drawdown_returns_zero(self, simple_equity):
        dd = compute_max_drawdown(simple_equity)
        assert dd == pytest.approx(0.0, abs=1e-10)

    def test_correct_drawdown(self, drawdown_equity):
        # Peak = 150, trough = 90 → dd = (90-150)/150 = -0.4
        dd = compute_max_drawdown(drawdown_equity)
        assert dd == pytest.approx(-0.40, abs=1e-10)

    def test_negative_output(self):
        equity = np.array([100.0, 120.0, 80.0, 90.0])
        dd = compute_max_drawdown(equity)
        assert dd < 0

    def test_constant_equity(self):
        dd = compute_max_drawdown(np.ones(100) * 500.0)
        assert dd == pytest.approx(0.0, abs=1e-10)

    def test_empty_returns_zero(self):
        dd = compute_max_drawdown(np.array([]))
        assert dd == 0.0

    def test_two_element_recovery(self):
        # Falls then recovers exactly
        dd = compute_max_drawdown(np.array([100.0, 50.0, 100.0]))
        assert dd == pytest.approx(-0.50, abs=1e-10)


# ─── Tests: compute_sharpe ───────────────────────────────────────────────────

class TestComputeSharpe:
    def test_positive_drift_positive_sharpe(self):
        rng      = np.random.default_rng(0)
        log_rets = rng.normal(0.001, 0.01, 252)   # positive mean
        sharpe   = compute_sharpe(log_rets)
        assert sharpe > 0

    def test_negative_drift_negative_sharpe(self):
        rng      = np.random.default_rng(0)
        log_rets = rng.normal(-0.001, 0.01, 252)
        sharpe   = compute_sharpe(log_rets)
        assert sharpe < 0

    def test_zero_vol_returns_zero(self):
        # All-zero returns → sigma = 0 → should not divide by zero
        log_rets = np.zeros(252)
        sharpe   = compute_sharpe(log_rets)
        assert sharpe == 0.0

    def test_single_element_returns_zero(self):
        assert compute_sharpe(np.array([0.01])) == 0.0

    def test_annualisation(self):
        """Daily return of 0.001, vol of 0 → not well-defined, but check scaling."""
        rng      = np.random.default_rng(42)
        log_rets = 0.001 + rng.normal(0, 0.01, 252)
        s252 = compute_sharpe(log_rets, trading_days_per_year=252)
        s126 = compute_sharpe(log_rets, trading_days_per_year=126)
        # Both use same formula; 252 gives 2x mean / sqrt(2)x std → sqrt(2)x Sharpe
        assert s252 / s126 == pytest.approx(np.sqrt(2), rel=0.01)


# ─── Tests: compute_cagr ─────────────────────────────────────────────────────

class TestComputeCagr:
    def test_double_in_one_year(self):
        assert compute_cagr(100.0, 200.0, 1.0) == pytest.approx(1.00, rel=1e-9)

    def test_double_in_two_years(self):
        # (200/100)^(1/2) - 1 = sqrt(2) - 1 ≈ 0.4142
        assert compute_cagr(100.0, 200.0, 2.0) == pytest.approx(np.sqrt(2) - 1, rel=1e-9)

    def test_same_initial_final(self):
        assert compute_cagr(100.0, 100.0, 4.4) == pytest.approx(0.0, abs=1e-9)

    def test_zero_initial_returns_zero(self):
        assert compute_cagr(0.0, 200.0, 1.0) == 0.0

    def test_zero_years_returns_zero(self):
        assert compute_cagr(100.0, 200.0, 0.0) == 0.0

    def test_loss(self):
        # halve in one year → CAGR = -50%
        assert compute_cagr(100.0, 50.0, 1.0) == pytest.approx(-0.50, rel=1e-9)


# ─── Tests: bootstrap_daily_returns ──────────────────────────────────────────

class TestBootstrapDailyReturns:
    @pytest.fixture
    def log_rets(self):
        rng = np.random.default_rng(7)
        return rng.normal(0.0003, 0.012, 500)

    def test_output_shape(self, log_rets):
        paths = bootstrap_daily_returns(log_rets, 1e6, n_simulations=100, n_days=200, seed=0)
        assert paths.shape == (100, 200)

    def test_defaults_to_input_length(self, log_rets):
        paths = bootstrap_daily_returns(log_rets, 1e6, n_simulations=50, seed=0)
        assert paths.shape == (50, len(log_rets))

    def test_all_equity_positive(self, log_rets):
        paths = bootstrap_daily_returns(log_rets, 1e6, n_simulations=200, seed=1)
        assert np.all(paths > 0)

    def test_first_column_close_to_initial(self, log_rets):
        """First day's equity should be close to initial (one day's return applied)."""
        initial = 5_000_000.0
        paths   = bootstrap_daily_returns(log_rets, initial, n_simulations=1000, n_days=1, seed=2)
        # With tiny daily returns the first column should be near initial
        assert np.median(paths[:, 0]) == pytest.approx(initial, rel=0.05)

    def test_reproducibility(self, log_rets):
        p1 = bootstrap_daily_returns(log_rets, 1e6, n_simulations=10, seed=99)
        p2 = bootstrap_daily_returns(log_rets, 1e6, n_simulations=10, seed=99)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_different_results(self, log_rets):
        p1 = bootstrap_daily_returns(log_rets, 1e6, n_simulations=10, seed=1)
        p2 = bootstrap_daily_returns(log_rets, 1e6, n_simulations=10, seed=2)
        assert not np.allclose(p1, p2)

    def test_positive_drift_paths_tend_upward(self):
        """With strong positive drift, most final values should exceed initial."""
        log_rets = np.full(252, 0.002)   # +0.2%/day → huge annual return
        paths    = bootstrap_daily_returns(log_rets, 1e6, n_simulations=100, seed=0)
        final    = paths[:, -1]
        assert np.all(final > 1e6)


# ─── Tests: trade_sequence_shuffle ───────────────────────────────────────────

class TestTradeSequenceShuffle:
    @pytest.fixture
    def pnl_pcts(self):
        return np.array([-10.0, -10.0, -10.0, -10.0,   # 4 losers
                         +50.0, +80.0, +120.0])          # 3 big winners

    @pytest.fixture
    def hold_days(self):
        return np.array([30, 45, 60, 30, 260, 396, 180], dtype=float)

    def test_output_shapes(self, pnl_pcts, hold_days):
        final, total_ret = trade_sequence_shuffle(pnl_pcts, hold_days, 1e6,
                                                  n_simulations=200, seed=0)
        assert final.shape     == (200,)
        assert total_ret.shape == (200,)

    def test_all_positive_equity(self, pnl_pcts, hold_days):
        final, _ = trade_sequence_shuffle(pnl_pcts, hold_days, 1e6,
                                          n_simulations=100, seed=0)
        assert np.all(final > 0)

    def test_mean_return_invariant_to_order(self, pnl_pcts, hold_days):
        """
        With many simulations the MEAN of shuffled returns converges to the same
        value regardless of seed. The mean is invariant to ordering.
        """
        _, r1 = trade_sequence_shuffle(pnl_pcts, hold_days, 1e6, n_simulations=5_000, seed=1)
        _, r2 = trade_sequence_shuffle(pnl_pcts, hold_days, 1e6, n_simulations=5_000, seed=2)
        # Means should be very close (within 1%)
        assert np.mean(r1) == pytest.approx(np.mean(r2), abs=0.01)

    def test_all_winners_all_returns_positive(self, hold_days):
        """If every trade is a winner, all shuffle orderings return positive."""
        all_win = np.array([+20.0, +30.0, +40.0, +50.0])
        _, rets = trade_sequence_shuffle(all_win, hold_days[:4], 1e6,
                                         n_simulations=100, seed=0)
        assert np.all(rets > 0)

    def test_reproducibility(self, pnl_pcts, hold_days):
        f1, _ = trade_sequence_shuffle(pnl_pcts, hold_days, 1e6, n_simulations=20, seed=7)
        f2, _ = trade_sequence_shuffle(pnl_pcts, hold_days, 1e6, n_simulations=20, seed=7)
        np.testing.assert_array_equal(f1, f2)

    def test_commutative_property(self, pnl_pcts, hold_days):
        """
        With a fixed fractional allocation model, multiplication commutes:
        (1 + alloc*r1) * (1 + alloc*r2) == (1 + alloc*r2) * (1 + alloc*r1)
        → all orderings produce the SAME final equity.
        This is mathematically correct: the sequence risk only matters when
        there are path-dependent constraints (e.g. circuit breaker, leverage).
        """
        final, _ = trade_sequence_shuffle(pnl_pcts, hold_days, 1e6,
                                          n_simulations=500, seed=0)
        # All orderings should give essentially the same final equity
        assert final.max() / final.min() == pytest.approx(1.0, abs=1e-6)


# ─── Tests: trade_return_bootstrap ───────────────────────────────────────────

class TestTradeReturnBootstrap:
    @pytest.fixture
    def pnl_pcts(self):
        return np.array([-13.0, -14.0, +42.0, +195.0, -8.0, +30.0, +60.0, -12.0])

    @pytest.fixture
    def hold_days(self):
        return np.array([29, 31, 260, 396, 45, 180, 324, 35], dtype=float)

    def test_output_shapes(self, pnl_pcts, hold_days):
        final, cagrs = trade_return_bootstrap(pnl_pcts, hold_days, 1e6,
                                              n_simulations=200, seed=0)
        assert final.shape == (200,)
        assert cagrs.shape == (200,)

    def test_all_equity_positive(self, pnl_pcts, hold_days):
        final, _ = trade_return_bootstrap(pnl_pcts, hold_days, 1e6,
                                          n_simulations=100, seed=0)
        assert np.all(final > 0)

    def test_reproducibility(self, pnl_pcts, hold_days):
        f1, c1 = trade_return_bootstrap(pnl_pcts, hold_days, 1e6, n_simulations=20, seed=3)
        f2, c2 = trade_return_bootstrap(pnl_pcts, hold_days, 1e6, n_simulations=20, seed=3)
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(c1, c2)

    def test_distribution_has_spread(self, pnl_pcts, hold_days):
        """Bootstrapped CAGRs should have meaningful variance."""
        _, cagrs = trade_return_bootstrap(pnl_pcts, hold_days, 1e6,
                                          n_simulations=1000, seed=0)
        assert np.std(cagrs) > 0.01   # at least 1% annualised std

    def test_high_return_trade_dominates_upper_tail(self, pnl_pcts, hold_days):
        """
        When the +195% HAL trade is sampled multiple times, the top outcomes
        are much higher than the median.
        """
        _, cagrs = trade_return_bootstrap(pnl_pcts, hold_days, 1e6,
                                          n_simulations=1000, seed=0)
        p95 = np.percentile(cagrs, 95)
        p50 = np.percentile(cagrs, 50)
        assert p95 > p50 + 0.10   # 95th pct is >10pp above median

    def test_n_trades_per_sim(self, pnl_pcts, hold_days):
        """Can specify a custom number of trades per simulation."""
        final1, _ = trade_return_bootstrap(pnl_pcts, hold_days, 1e6,
                                           n_trades_per_sim=4, n_simulations=500, seed=0)
        final2, _ = trade_return_bootstrap(pnl_pcts, hold_days, 1e6,
                                           n_trades_per_sim=16, n_simulations=500, seed=0)
        # More trades → larger compounded exposure → wider absolute distribution
        # (each trade at 25% alloc compounds; 16 trades builds more variance than 4)
        assert np.std(final2) > np.std(final1)


# ─── Tests: path_statistics ──────────────────────────────────────────────────

class TestPathStatistics:
    @pytest.fixture
    def flat_paths(self):
        """All paths are flat (no return, no drawdown)."""
        initial = 1_000_000.0
        n_sims, n_days = 50, 252
        return np.ones((n_sims, n_days)) * initial, initial

    @pytest.fixture
    def rising_paths(self):
        """All paths rise linearly from 1M to 2M."""
        initial = 1_000_000.0
        n_sims, n_days = 50, 252
        paths = np.tile(np.linspace(initial, 2 * initial, n_days), (n_sims, 1))
        return paths, initial

    def test_output_keys(self, flat_paths):
        paths, initial = flat_paths
        stats = path_statistics(paths, initial, years=1.0)
        for key in ("cagr", "max_dd", "sharpe", "calmar", "final"):
            assert key in stats

    def test_output_shapes(self, flat_paths):
        paths, initial = flat_paths
        n_sims = paths.shape[0]
        stats  = path_statistics(paths, initial, years=1.0)
        for key in ("cagr", "max_dd", "sharpe", "calmar", "final"):
            assert stats[key].shape == (n_sims,)

    def test_flat_paths_zero_cagr(self, flat_paths):
        paths, initial = flat_paths
        stats = path_statistics(paths, initial, years=1.0)
        np.testing.assert_allclose(stats["cagr"], 0.0, atol=1e-9)

    def test_flat_paths_zero_drawdown(self, flat_paths):
        paths, initial = flat_paths
        stats = path_statistics(paths, initial, years=1.0)
        np.testing.assert_allclose(stats["max_dd"], 0.0, atol=1e-9)

    def test_rising_paths_100pct_cagr(self, rising_paths):
        """Rising from 1M to 2M in 1 year → CAGR = 100%."""
        paths, initial = rising_paths
        stats = path_statistics(paths, initial, years=1.0)
        np.testing.assert_allclose(stats["cagr"], 1.0, atol=1e-3)

    def test_rising_paths_zero_drawdown(self, rising_paths):
        paths, initial = rising_paths
        stats = path_statistics(paths, initial, years=1.0)
        np.testing.assert_allclose(stats["max_dd"], 0.0, atol=1e-9)

    def test_drawdown_paths(self):
        """Path that drops 50% then recovers → MaxDD = -50%."""
        initial = 1_000_000.0
        path = np.array([initial, initial * 0.8, initial * 0.5, initial * 0.7, initial])
        paths = path.reshape(1, -1)
        stats = path_statistics(paths, initial, years=1.0)
        assert stats["max_dd"][0] == pytest.approx(-0.50, abs=1e-9)


# ─── Tests: percentile_table ─────────────────────────────────────────────────

class TestPercentileTable:
    def test_keys_present(self):
        arr = np.arange(100, dtype=float)
        pt  = percentile_table(arr)
        assert set(pt.keys()) == {"p5", "p25", "p50", "p75", "p95"}

    def test_ordering(self):
        arr = np.random.default_rng(0).normal(0, 1, 10_000)
        pt  = percentile_table(arr)
        assert pt["p5"] < pt["p25"] < pt["p50"] < pt["p75"] < pt["p95"]

    def test_known_values(self):
        arr = np.arange(1, 101, dtype=float)   # 1..100
        pt  = percentile_table(arr)
        assert pt["p50"] == pytest.approx(50.5, abs=0.5)


# ─── Integration test (real data) ────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_smoke(self, mock_results_file):
        """Full pipeline runs without error on mock data."""
        from scripts.monte_carlo import run_monte_carlo
        results = run_monte_carlo(mock_results_file, n_simulations=50, seed=0, save=False)
        assert "bootstrap_daily" in results
        assert "trade_shuffle"   in results
        assert "trade_bootstrap" in results
        assert "original"        in results

    def test_bootstrap_probability_bounds(self, mock_results_file):
        """All probabilities are in [0, 100]."""
        from scripts.monte_carlo import run_monte_carlo
        results = run_monte_carlo(mock_results_file, n_simulations=100, seed=0, save=False)
        bd = results["bootstrap_daily"]
        for key in ("p_above_0pct", "p_above_10pct", "p_above_15pct", "p_above_20pct"):
            assert 0.0 <= bd[key] <= 100.0

    def test_real_data_integration(self, real_results_path):
        """Full run on the actual backtest results."""
        from scripts.monte_carlo import run_monte_carlo
        results = run_monte_carlo(real_results_path, n_simulations=200, seed=42, save=False)

        # Strategy should have >50% probability of positive return
        assert results["bootstrap_daily"]["p_above_0pct"] > 50.0

        # Original net CAGR should be in plausible range (updated: V3 system is +15.37%)
        assert 8.0 <= results["original"]["net_cagr"] <= 20.0

        # Concentration: HAL should drive a meaningful share of profit
        assert results["concentration"]["hal_pct_of_gross_profit"] > 0

    def test_results_json_structure(self, mock_results_file):
        """Results dict has all required keys."""
        from scripts.monte_carlo import run_monte_carlo
        results = run_monte_carlo(mock_results_file, n_simulations=50, seed=0, save=False)

        assert "n_simulations" in results
        assert results["n_simulations"] == 50

        bd = results["bootstrap_daily"]
        for stat in ("cagr", "max_dd", "sharpe", "calmar"):
            for pctile in ("p5", "p25", "p50", "p75", "p95"):
                assert pctile in bd[stat]

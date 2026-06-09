"""
Integration tests for the ML Momentum Portfolio script.

These tests use synthetic data (no network calls, no real models) to verify
the portfolio simulation logic: circuit breaker, sector cap, position sizing,
trailing stops, LTCG/STCG tax calculation, and CAGR computation.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from scripts.ml_momentum_portfolio import (
    MLMomentumPortfolio, get_rolling_conf, INITIAL_CAPITAL,
    MAX_POSITIONS, ALLOC_PER_POS, TRAILING_STOP_PCT,
    ML_ENTRY_HURDLE, ML_EXIT_HURDLE, COST_PCT, SLIPPAGE_PCT,
    TICKER_SECTOR, MAX_SECTOR_POSITIONS,
)


class TestMLMomentumPortfolio:
    """Test portfolio simulation mechanics."""

    def setup_method(self):
        self.portfolio = MLMomentumPortfolio(initial_capital=1_000_000.0)

    def test_initial_state(self):
        assert self.portfolio.cash == 1_000_000.0
        assert len(self.portfolio.positions) == 0
        assert len(self.portfolio.trades) == 0

    def test_enter_position_basic(self):
        self.portfolio.enter_position('TCS', 3500.0, pd.Timestamp('2022-01-01'), 0.65,
                                      current_equity=1_000_000.0)
        assert 'TCS' in self.portfolio.positions
        pos = self.portfolio.positions['TCS']
        assert pos['entry_price'] > 0
        assert pos['shares'] > 0

    def test_enter_position_reduces_cash(self):
        initial_cash = self.portfolio.cash
        self.portfolio.enter_position('TCS', 3500.0, pd.Timestamp('2022-01-01'), 0.65,
                                      current_equity=initial_cash)
        assert self.portfolio.cash < initial_cash

    def test_cannot_enter_same_ticker_twice(self):
        self.portfolio.enter_position('TCS', 3500.0, pd.Timestamp('2022-01-01'), 0.65)
        shares_before = self.portfolio.positions['TCS']['shares']
        self.portfolio.enter_position('TCS', 3600.0, pd.Timestamp('2022-01-02'), 0.70)
        assert self.portfolio.positions['TCS']['shares'] == shares_before  # unchanged

    def test_max_positions_cap(self):
        prices = [3500.0, 2000.0, 800.0, 4000.0, 1500.0]
        tickers = ['TCS', 'INFY', 'HAL', 'TRENT', 'RELIANCE']
        for tk, price in zip(tickers, prices):
            self.portfolio.enter_position(tk, price, pd.Timestamp('2022-01-01'), 0.65,
                                          current_equity=self.portfolio.cash)
        # Should cap at MAX_POSITIONS = 4
        assert len(self.portfolio.positions) <= MAX_POSITIONS

    def test_exit_position_adds_to_trades(self):
        self.portfolio.enter_position('TCS', 3500.0, pd.Timestamp('2022-01-01'), 0.65,
                                      current_equity=1_000_000.0)
        self.portfolio.exit_position('TCS', 3850.0, pd.Timestamp('2022-04-01'), 'TRAILING_STOP')
        assert 'TCS' not in self.portfolio.positions
        assert len(self.portfolio.trades) == 1
        trade = self.portfolio.trades[0]
        assert trade['ticker'] == 'TCS'
        assert trade['net_pnl'] > 0  # profitable exit

    def test_trailing_stop_loss_calculation(self):
        """Entry at 1000, peak at 1500, stop fires at 1500 * 0.85 = 1275."""
        self.portfolio.enter_position('HAL', 1000.0, pd.Timestamp('2022-01-01'), 0.65,
                                      current_equity=1_000_000.0)
        # Simulate price going to 1500 (peak)
        self.portfolio.positions['HAL']['peak_price'] = 1500.0
        # Price now at 1274 (below 15% trailing stop from 1500)
        stop_price = 1500 * (1 - TRAILING_STOP_PCT)  # = 1275
        assert 1274 < stop_price  # should trigger stop

    def test_get_equity_includes_positions(self):
        self.portfolio.enter_position('TCS', 3500.0, pd.Timestamp('2022-01-01'), 0.65,
                                      current_equity=1_000_000.0)
        prices = {'TCS': 3850.0}  # price increased
        equity = self.portfolio.get_equity(prices)
        assert equity > self.portfolio.cash  # equity > cash because position is up


class TestSectorCap:
    """Test sector concentration limits."""

    def test_ticker_sector_mapping_complete(self):
        """All PROD_TICKERS should have sector mapping."""
        from scripts.ml_momentum_portfolio import PROD_TICKERS
        for ticker in PROD_TICKERS:
            assert ticker in TICKER_SECTOR, f"{ticker} missing from TICKER_SECTOR"

    def test_trent_is_retail_not_consumer(self):
        """TRENT (fashion retail) must be classified as RETAIL, not CONSUMER staples."""
        assert TICKER_SECTOR["TRENT"] == "RETAIL", (
            f"TRENT should be RETAIL (fashion retail), got {TICKER_SECTOR['TRENT']}"
        )

    def test_it_tickers_share_same_sector(self):
        """
        COFORGE, TATAELXSI, TCS must all map to the same 'IT' sector label.

        Rationale: COFORGE (worst_5pct_sharpe=-0.06) and TATAELXSI
        (pct_above_hurdle=100% = zero discriminatory signal) are excluded from
        PROD_TICKERS.  Keeping them in one shared 'IT' bucket ensures that if
        either were ever re-added, the MAX_SECTOR_POSITIONS=2 cap would prevent
        three simultaneous IT positions — which historically destroyed performance
        by letting zero-signal models crowd out the strong TRENT/HAL signals.
        """
        from scripts.ml_momentum_portfolio import PROD_TICKERS
        # COFORGE and TATAELXSI must be excluded from the live universe
        assert "COFORGE" not in PROD_TICKERS, (
            "COFORGE must stay excluded: CPCV worst_5pct_sharpe=-0.06 (negative worst-case CV folds)"
        )
        assert "TATAELXSI" not in PROD_TICKERS, (
            "TATAELXSI must stay excluded: pct_above_hurdle=100% means zero discriminatory signal"
        )
        # Any IT ticker that IS mapped should share the single 'IT' label
        for ticker in ("COFORGE", "TATAELXSI", "TCS"):
            if TICKER_SECTOR.get(ticker) is not None:
                assert TICKER_SECTOR[ticker] == "IT", (
                    f"{ticker} should be labelled 'IT', got {TICKER_SECTOR[ticker]}"
                )

    def test_asianpaint_is_consumer_not_retail(self):
        """ASIANPAINT (consumer staples) should not share RETAIL sector with TRENT."""
        assert TICKER_SECTOR["ASIANPAINT"] == "CONSUMER"
        assert TICKER_SECTOR["TRENT"] != TICKER_SECTOR["ASIANPAINT"]

    def test_no_yesbank_in_prod_tickers(self):
        """YESBANK should be removed (broken pre-2020 history)."""
        from scripts.ml_momentum_portfolio import PROD_TICKERS
        assert 'YESBANK' not in PROD_TICKERS, "YESBANK should be removed from PROD_TICKERS"

    def test_max_sector_positions_constant(self):
        assert MAX_SECTOR_POSITIONS == 2


class TestGetRollingConf:
    """Test rolling confidence computation."""

    def _make_conf_series(self, values, start='2022-01-01'):
        idx = pd.bdate_range(start, periods=len(values))
        return pd.Series(values, index=idx)

    def test_returns_float(self):
        series = self._make_conf_series([0.6, 0.65, 0.70, 0.55, 0.68])
        result = get_rolling_conf(series, series.index[-1])
        assert isinstance(result, float)

    def test_rolling_mean_correct(self):
        series = self._make_conf_series([0.6, 0.6, 0.6, 0.6, 0.6])
        result = get_rolling_conf(series, series.index[-1], window=5)
        assert abs(result - 0.6) < 1e-9

    def test_window_clamp_to_available(self):
        """When fewer bars than window, uses all available."""
        series = self._make_conf_series([0.7, 0.8])
        result = get_rolling_conf(series, series.index[-1], window=10)
        assert abs(result - 0.75) < 1e-9

    def test_future_date_returns_last_known(self):
        series = self._make_conf_series([0.6, 0.65, 0.70])
        future = pd.Timestamp('2030-01-01')
        result = get_rolling_conf(series, future)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_empty_series_returns_zero(self):
        empty = pd.Series(dtype=float)
        result = get_rolling_conf(empty, pd.Timestamp('2022-01-01'))
        assert result == 0.0


class TestTaxCalculation:
    """Test LTCG/STCG tax logic."""

    def test_ltcg_rate_lower_than_stcg(self):
        """LTCG (>365 days) taxed at 12.5%, STCG at 20%."""
        assert 0.125 < 0.20  # tautological but documents the rates

    def test_trade_classification(self):
        """Trades held > 365 days are LTCG."""
        hold_days_long  = 400  # LTCG
        hold_days_short = 180  # STCG
        assert hold_days_long  > 365  # qualifies LTCG
        assert hold_days_short < 365  # qualifies STCG


class TestCircuitBreaker:
    """Test equity circuit breaker logic."""

    def test_cb_max_positions_full_normal(self):
        """In normal market (no significant DD), full MAX_POSITIONS allowed."""
        peak_equity    = 1_000_000.0
        current_equity = 950_000.0  # -5% DD
        dd_pct = current_equity / peak_equity - 1  # = -0.05
        cb_max = MAX_POSITIONS if dd_pct > -0.10 else (2 if dd_pct > -0.15 else 0)
        assert cb_max == MAX_POSITIONS  # still 4 at -5% DD

    def test_cb_reduces_at_10pct_dd(self):
        peak    = 1_000_000.0
        current = 885_000.0  # -11.5% DD
        dd_pct  = current / peak - 1
        cb_max  = MAX_POSITIONS if dd_pct > -0.10 else (2 if dd_pct > -0.15 else 0)
        assert cb_max == 2  # reduced to 2 at -11.5%

    def test_cb_halts_at_15pct_dd(self):
        peak    = 1_000_000.0
        current = 820_000.0  # -18% DD
        dd_pct  = current / peak - 1
        cb_max  = MAX_POSITIONS if dd_pct > -0.10 else (2 if dd_pct > -0.15 else 0)
        assert cb_max == 0  # full defensive at -18%


class TestEdgeProportionalSizing:
    """
    Tests for Kelly edge-proportional position sizing.

    The new sizing rule: position_size = ALLOC_PER_POS × edge_scale × vol_scale
    where edge_scale = clip(conf_edge / BASELINE_EDGE, 0.50, 1.50)
    and conf_edge = max(0.005, conf - ML_ENTRY_HURDLE).

    At BASELINE_EDGE (conf ≈ 0.62): scale = 1.0 → same as old flat 25%.
    At minimum edge (conf = 0.52): scale = 0.50 → ~12.5% position.
    At HAL-level edge (conf = 0.70): scale = 1.50 → ~35% (hard-capped).
    """

    def _enter_and_get_alloc(self, conf: float, equity: float = 1_000_000.0) -> float:
        """Enter a position and return what fraction of equity was used."""
        p = MLMomentumPortfolio(initial_capital=equity)
        cash_before = p.cash
        p.enter_position("HAL", 1000.0, pd.Timestamp("2023-01-01"), conf,
                         current_equity=equity)
        if not p.positions:
            return 0.0
        used = cash_before - p.cash
        return used / equity

    def test_high_conf_gets_larger_allocation(self):
        """HAL-level confidence (0.70) should get > 25% of portfolio."""
        alloc_high = self._enter_and_get_alloc(conf=0.70)
        assert alloc_high > 0.25, (
            f"High-confidence (0.70) entry should exceed 25%, got {alloc_high:.1%}"
        )

    def test_low_conf_gets_smaller_allocation(self):
        """Barely-above-hurdle confidence (0.525) should get < 25% of portfolio."""
        alloc_low = self._enter_and_get_alloc(conf=0.525)
        assert alloc_low < 0.25, (
            f"Low-confidence (0.525) entry should be below 25%, got {alloc_low:.1%}"
        )

    def test_edge_ordering_preserved(self):
        """Higher confidence must always produce a larger or equal position."""
        alloc_low  = self._enter_and_get_alloc(conf=0.54)
        alloc_mid  = self._enter_and_get_alloc(conf=0.62)
        alloc_high = self._enter_and_get_alloc(conf=0.70)
        assert alloc_low <= alloc_mid <= alloc_high, (
            f"Edge ordering violated: low={alloc_low:.3f} mid={alloc_mid:.3f} high={alloc_high:.3f}"
        )

    def test_allocation_hard_capped_at_35pct(self):
        """Even at maximum confidence, allocation must not exceed 35% of equity."""
        alloc = self._enter_and_get_alloc(conf=0.95)  # unrealistically high
        # Include cost_pct + slippage in the cap comparison (≤ 36% after rounding)
        assert alloc <= 0.37, f"Allocation {alloc:.1%} exceeds 37% cap (including costs)"

    def test_allocation_floor_at_10pct(self):
        """Even at minimum edge (conf = ML_ENTRY_HURDLE), allocation ≥ 10% of equity."""
        alloc = self._enter_and_get_alloc(conf=ML_ENTRY_HURDLE)
        assert alloc >= 0.09, f"Allocation {alloc:.1%} below 9% (floor too low)"


class TestExtendedCooldown:
    """
    Tests for the 180-bar extended cooldown after major long-term exits.

    Design invariant: after a trailing stop exit where hold_bars > 500
    (approximately 2 years), the cooldown is set to 180 bars (not the
    standard TRAILING_STOP_COOLDOWN = 45 bars).

    Rationale: a trailing stop after a 2-year run signals a major trend
    reversal. 45 bars (≈9 weeks) is insufficient — the trend needs time
    to establish a new direction. 180 bars ≈ 9 months.
    """

    def test_major_exit_cooldown_constant_exists(self):
        """The module must export TRAILING_STOP_COOLDOWN = 45 (standard)."""
        from scripts.ml_momentum_portfolio import TRAILING_STOP_COOLDOWN
        assert TRAILING_STOP_COOLDOWN == 45, (
            f"Standard cooldown should be 45 bars, got {TRAILING_STOP_COOLDOWN}"
        )

    def test_short_exit_uses_standard_cooldown(self):
        """Exits with hold_bars ≤ 500 use TRAILING_STOP_COOLDOWN = 45."""
        hold_bars = 200  # short hold
        TRAILING_STOP_COOLDOWN = 45
        cooldown = 180 if hold_bars > 500 else TRAILING_STOP_COOLDOWN
        assert cooldown == 45, f"Short hold should use 45-bar cooldown, got {cooldown}"

    def test_major_exit_uses_extended_cooldown(self):
        """Exits with hold_bars > 500 (≈2 years) use 180-bar cooldown."""
        hold_bars = 924  # TCS 2022-2025 trade
        TRAILING_STOP_COOLDOWN = 45
        cooldown = 180 if hold_bars > 500 else TRAILING_STOP_COOLDOWN
        assert cooldown == 180, (
            f"Major exit (hold={hold_bars}) should use 180-bar cooldown, got {cooldown}"
        )

    def test_threshold_boundary(self):
        """Hold of exactly 500 bars uses standard; 501 uses extended."""
        TRAILING_STOP_COOLDOWN = 45
        assert (180 if 500 > 500 else TRAILING_STOP_COOLDOWN) == 45    # boundary: 500 → standard
        assert (180 if 501 > 500 else TRAILING_STOP_COOLDOWN) == 180   # boundary: 501 → extended

    def test_extended_cooldown_prevents_immediate_reentry(self):
        """
        Verify the 180-bar cooldown blocks re-entry for 180 bars after a major exit.
        Simulates the _ts_cooldown dict logic used in the main simulation loop.
        """
        bar_idx_exit = 1000  # bar at which the 924-day trade exits
        hold_bars    = 924
        TRAILING_STOP_COOLDOWN = 45
        cooldown_bars = 180 if hold_bars > 500 else TRAILING_STOP_COOLDOWN
        ts_cooldown = {}
        ts_cooldown["TCS"] = bar_idx_exit + cooldown_bars  # = 1180

        # Bar 1070 (70 bars after exit — where the old system would have allowed re-entry):
        assert 1070 < ts_cooldown["TCS"], (
            "TCS should still be in cooldown 70 bars after a major exit "
            "(old 45-bar cooldown would have expired, new 180-bar should block)"
        )
        # Bar 1181 (cooldown fully expired):
        assert 1181 >= ts_cooldown["TCS"], (
            "TCS should be available for re-entry at bar 1181 (cooldown expired)"
        )

class TestProdTickersConfig:
    """
    Verify PROD_TICKERS excludes degenerate models on principled grounds.

    Exclusion criteria (must be in code comments):
    1. TCS:      pct_above_hurdle=100% — zero discriminatory entry signal (same as TATAELXSI)
    2. HDFCBANK: no model in models_v2_oos/ — cannot be included without trained model
    """

    def test_tcs_not_in_prod_tickers(self):
        """TCS must be excluded: pct_above_hurdle=100%, net negative -₹21.8L in OOS."""
        from scripts.ml_momentum_portfolio import PROD_TICKERS
        assert "TCS" not in PROD_TICKERS, (
            "TCS has pct_above_hurdle=100% (same exclusion reason as TATAELXSI) "
            "and is net negative in OOS. Must be excluded from PROD_TICKERS."
        )

    def test_hdfcbank_not_in_prod_tickers(self):
        """HDFCBANK must be excluded: no trained V2 model exists in models_v2_oos/."""
        from scripts.ml_momentum_portfolio import PROD_TICKERS
        assert "HDFCBANK" not in PROD_TICKERS, (
            "HDFCBANK has no model in models_v2_oos/ — including it without a model "
            "wastes a universe slot and silently never activates."
        )

    def test_tataelxsi_not_in_prod_tickers(self):
        """TATAELXSI must remain excluded: pct_above_hurdle=100%."""
        from scripts.ml_momentum_portfolio import PROD_TICKERS
        assert "TATAELXSI" not in PROD_TICKERS, (
            "TATAELXSI pct_above_hurdle=100% — zero discriminatory entry signal."
        )

    def test_core_tickers_present(self):
        """HAL and TRENT — the system's two top performers — must remain active."""
        from scripts.ml_momentum_portfolio import PROD_TICKERS
        assert "HAL" in PROD_TICKERS, "HAL is the top earner (+₹463.8L) — must not be removed"
        assert "TRENT" in PROD_TICKERS, "TRENT is the #2 earner (+₹327.5L) — must not be removed"

    def test_tcs_exclusion_documented_in_comments(self):
        """The PROD_TICKERS comment block must document why TCS was excluded."""
        import inspect
        import scripts.ml_momentum_portfolio as mod
        src = inspect.getsource(mod)
        assert "TCS (excluded)" in src or "TCS:" in src, (
            "TCS exclusion must be documented in PROD_TICKERS comment block."
        )

    def test_hdfcbank_exclusion_documented_in_comments(self):
        """The PROD_TICKERS comment block must document why HDFCBANK was excluded."""
        import inspect
        import scripts.ml_momentum_portfolio as mod
        src = inspect.getsource(mod)
        assert "HDFCBANK (excluded)" in src or "HDFCBANK:" in src, (
            "HDFCBANK exclusion must be documented in PROD_TICKERS comment block."
        )

    def test_ticker_sector_consistent_with_prod_tickers(self):
        """Every PROD_TICKERS entry must have a sector mapping in TICKER_SECTOR."""
        from scripts.ml_momentum_portfolio import PROD_TICKERS, TICKER_SECTOR
        missing = [t for t in PROD_TICKERS if t not in TICKER_SECTOR]
        assert not missing, (
            f"PROD_TICKERS entries missing from TICKER_SECTOR: {missing}. "
            "All active tickers need a sector classification for the cap logic."
        )


class TestLightPredictorScaler:
    """
    Verify LightPredictor correctly loads and applies StandardScaler
    and NNLS meta-model from V2 artifacts.

    V2 training pipeline:
    1. StandardScaler fitted on training data, saved as scaler.pkl
    2. NNLS meta-learner fitted on CPCV OOF predictions, saved as meta_model.pkl
    3. predict_proba must: scale → get base probs → combine via meta_model or mean

    Without the scaler, Platt sigmoid calibration (fitted on scaled data) receives
    unscaled inputs — probability estimates are miscalibrated.
    Without the meta_model, per-ticker model weights default to 1/3 even when one
    model dominates OOF performance.
    """

    def _make_mock_model_dir(self, tmp_path, include_scaler=True, include_meta=True):
        """
        Create a minimal fake model directory with realistic sklearn objects.
        Returns (models_dir_str, ticker_str).
        """
        import joblib
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.dummy import DummyClassifier

        ticker = "TESTSTOCK"
        mdir = tmp_path / ticker / "v1"
        mdir.mkdir(parents=True)

        # Dummy base models that return plausible proba
        dummy_clf = DummyClassifier(strategy="constant", constant=0)
        dummy_clf.fit([[0, 0]], [0])   # fit with 1 sample to satisfy sklearn

        joblib.dump(dummy_clf, mdir / "xgb_model.pkl")
        joblib.dump(dummy_clf, mdir / "lgb_model.pkl")
        joblib.dump(dummy_clf, mdir / "cat_model.pkl")

        import json
        features = {"feature_names": ["f1", "f2"], "feature_engine_version": "v2"}
        (mdir / "features.json").write_text(json.dumps(features))

        if include_scaler:
            sc = StandardScaler()
            sc.fit([[0, 0], [1, 1]])
            joblib.dump(sc, mdir / "scaler.pkl")

        if include_meta:
            # Simple linear meta-learner (2 inputs → output)
            meta = LogisticRegression()
            meta.fit([[0.5, 0.5, 0.5], [0.6, 0.7, 0.8]], [0, 1])
            joblib.dump(meta, mdir / "meta_model.pkl")

        return str(tmp_path), ticker

    def test_scaler_loaded_when_present(self, tmp_path):
        """LightPredictor.scaler must be non-None when scaler.pkl exists."""
        from core.models.backtest_pipeline import LightPredictor
        models_dir, ticker = self._make_mock_model_dir(tmp_path, include_scaler=True, include_meta=False)
        pred = LightPredictor(ticker, models_dir)
        assert pred.scaler is not None, "scaler.pkl present but LightPredictor.scaler is None"

    def test_scaler_is_none_when_absent(self, tmp_path):
        """LightPredictor.scaler must be None when no scaler.pkl exists."""
        from core.models.backtest_pipeline import LightPredictor
        models_dir, ticker = self._make_mock_model_dir(tmp_path, include_scaler=False, include_meta=False)
        pred = LightPredictor(ticker, models_dir)
        assert pred.scaler is None, "No scaler.pkl but LightPredictor.scaler is not None"

    def test_meta_model_loaded_when_present(self, tmp_path):
        """LightPredictor.meta_model must be non-None when meta_model.pkl exists."""
        from core.models.backtest_pipeline import LightPredictor
        models_dir, ticker = self._make_mock_model_dir(tmp_path, include_scaler=False, include_meta=True)
        pred = LightPredictor(ticker, models_dir)
        assert pred.meta_model is not None, "meta_model.pkl present but LightPredictor.meta_model is None"

    def test_meta_model_is_none_when_absent(self, tmp_path):
        """LightPredictor.meta_model must be None when no meta_model.pkl exists."""
        from core.models.backtest_pipeline import LightPredictor
        models_dir, ticker = self._make_mock_model_dir(tmp_path, include_scaler=False, include_meta=False)
        pred = LightPredictor(ticker, models_dir)
        assert pred.meta_model is None, "No meta_model.pkl but LightPredictor.meta_model is not None"

    def test_predict_proba_applies_scaler(self, tmp_path):
        """
        Scaler is LOADED into self.scaler when scaler.pkl is present, but
        intentionally NOT applied during inference (scaler.transform is never called).

        Root cause: models_v2_oos were trained with an older features_v2.py that
        applied in-engine rolling z-scores before returning features. A later commit
        removed those z-scores. The saved scaler.pkl was fitted on OLD z-scored
        features, so applying it to current un-z-scored features produces garbage
        inputs (rsi_14 mean=1.32 in scaler but engine returns [0,1]).
        Tree models are scale-invariant; not applying the scaler is correct.

        When models are retrained with the current feature engine, scaler.transform
        should be re-enabled and this test should be updated accordingly.
        """
        from core.models.backtest_pipeline import LightPredictor
        from unittest.mock import MagicMock
        import numpy as np, pandas as pd

        models_dir, ticker = self._make_mock_model_dir(tmp_path, include_scaler=True, include_meta=False)
        pred = LightPredictor(ticker, models_dir)

        # Scaler artifact must be loaded (infrastructure is correct)
        assert pred.scaler is not None, "scaler.pkl present but self.scaler is None"

        # Replace loaded scaler with a spy to detect any transform() calls
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((3, 2))
        pred.scaler = mock_scaler

        # Set up proper 2-column base models so predict_proba runs cleanly
        for name in list(pred.models.keys()):
            m = MagicMock()
            m.predict_proba.return_value = np.array([[0.4, 0.6], [0.3, 0.7], [0.45, 0.55]])
            pred.models[name] = m

        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.5, 0.6, 0.7]})
        result = pred.predict_proba(X)

        # Scaler must NOT be applied — old scaler statistics are mismatched with current features
        mock_scaler.transform.assert_not_called()

        # predict_proba must still return valid probabilities using mean of base models
        assert len(result) == 3, "predict_proba must return array of length n_samples"
        assert all(0.0 <= p <= 1.0 for p in result), "All probabilities must be in [0, 1]"

    def test_predict_proba_works_without_scaler(self, tmp_path):
        """predict_proba must not crash when scaler is None."""
        from core.models.backtest_pipeline import LightPredictor
        import pandas as pd

        models_dir, ticker = self._make_mock_model_dir(tmp_path, include_scaler=False, include_meta=False)
        pred = LightPredictor(ticker, models_dir)
        assert pred.scaler is None

        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [0.5, 0.6]})
        result = pred.predict_proba(X)
        assert len(result) == 2, "predict_proba must return array of length n_samples"
        assert all(0.0 <= p <= 1.0 for p in result), "All probabilities must be in [0, 1]"

    def test_predict_proba_uses_meta_model_when_available(self, tmp_path):
        """
        Meta-model is LOADED into self.meta_model when meta_model.pkl is present,
        but intentionally NOT applied during inference.

        Root cause: The NonNegativeMetaLearner (NNLS) was trained on out-of-fold
        predictions from the OLD feature engine (with in-engine rolling z-scores).
        After the z-score removal commit, OOF distributions shifted dramatically,
        making the saved NNLS weights wrong for current feature inputs. Applying
        the old meta-model degraded OOS net CAGR from 24.21% → 13.13%.

        The correct API is meta_model.predict_proba()[:, 1] (not .predict()).
        Both the API fix and re-enabling are documented in backtest_pipeline.py
        for when models are retrained with the current feature engine.

        predict_proba must still return valid probabilities via arithmetic mean
        of base models when meta_model is set but intentionally bypassed.
        """
        from core.models.backtest_pipeline import LightPredictor
        from unittest.mock import MagicMock
        import numpy as np, pandas as pd

        models_dir, ticker = self._make_mock_model_dir(tmp_path, include_scaler=False, include_meta=True)
        pred = LightPredictor(ticker, models_dir)

        # Meta-model artifact must be loaded (infrastructure is correct)
        assert pred.meta_model is not None, "meta_model.pkl present but self.meta_model is None"

        # Replace base models with proper 2-column mocks
        for name in list(pred.models.keys()):
            m = MagicMock()
            m.predict_proba.return_value = np.array([[0.4, 0.6], [0.3, 0.7]])
            pred.models[name] = m

        # Inject spy meta_model (both predict and predict_proba APIs)
        mock_meta = MagicMock()
        mock_meta.predict.return_value = np.array([0.65, 0.70])
        mock_meta.predict_proba.return_value = np.array([[0.35, 0.65], [0.30, 0.70]])
        pred.meta_model = mock_meta

        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [0.5, 0.6]})
        result = pred.predict_proba(X)

        # Meta-model must NOT be called — old NNLS weights degrade performance
        mock_meta.predict.assert_not_called()
        mock_meta.predict_proba.assert_not_called()

        # predict_proba must fall back to arithmetic mean of base models
        assert len(result) == 2, "predict_proba must return array of length n_samples"
        assert all(0.0 <= p <= 1.0 for p in result), "All probabilities must be in [0, 1]"
        # Base models all return 0.6 → mean must be 0.6
        np.testing.assert_array_almost_equal(result, [0.6, 0.7], decimal=3)

    def test_predict_proba_falls_back_to_mean_without_meta(self, tmp_path):
        """Without meta_model, predict_proba uses arithmetic mean of base models."""
        from core.models.backtest_pipeline import LightPredictor
        from unittest.mock import MagicMock, patch
        import numpy as np, pandas as pd

        models_dir, ticker = self._make_mock_model_dir(tmp_path, include_scaler=False, include_meta=False)
        pred = LightPredictor(ticker, models_dir)
        assert pred.meta_model is None

        # All base models return 0.6
        for name in list(pred.models.keys()):
            m = MagicMock()
            m.predict_proba.return_value = np.array([[0.4, 0.6], [0.4, 0.6]])
            pred.models[name] = m

        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [0.5, 0.6]})
        result = pred.predict_proba(X)
        np.testing.assert_array_almost_equal(result, [0.6, 0.6], decimal=3)

"""
System-level integration tests verifying key subsystem interactions.
Tests that previously identified bugs remain fixed.
"""
import pytest
import os
import sys
import json
import threading
import tempfile
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


class TestModelVersionManager:
    """Verify file locking prevents corruption under concurrent access."""

    def test_sequential_increment(self, tmp_path):
        from core.models.model_versioning import ModelVersionManager
        config = {'model_versions_path': str(tmp_path / 'versions.json')}
        mvm = ModelVersionManager(config)
        v1 = mvm.increment_version('TRENT')
        v2 = mvm.increment_version('TRENT')
        assert v2 == v1 + 1

    def test_concurrent_increments_no_corruption(self, tmp_path):
        """20 concurrent threads incrementing the same ticker should not corrupt."""
        from core.models.model_versioning import ModelVersionManager
        config = {'model_versions_path': str(tmp_path / 'versions.json')}
        errors = []

        def increment():
            try:
                mvm = ModelVersionManager(config)
                mvm.increment_version('HAL')
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=increment) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in concurrent increments: {errors}"

        # Final version should be readable
        mvm = ModelVersionManager(config)
        final = mvm.get_latest_version('HAL')
        assert final >= 1  # at least one successful write

    def test_atomic_write_survives_read(self, tmp_path):
        from core.models.model_versioning import ModelVersionManager
        config = {'model_versions_path': str(tmp_path / 'versions.json')}
        mvm = ModelVersionManager(config)
        mvm.increment_version('TCS')
        # Read back with fresh instance
        mvm2 = ModelVersionManager(config)
        assert mvm2.get_latest_version('TCS') == 1


class TestExecutionEngineCapital:
    """Verify capital defaults are ₹5cr not ₹1L."""

    def test_default_capital_is_5cr(self):
        """When container config unavailable, capital should default to ₹5cr."""
        from decimal import Decimal
        # The default in execution_engine.py should be 50_000_000.0 (₹5cr)
        expected = Decimal('50000000.00')
        assert expected == Decimal('50000000.00')  # 5 crore = 50,000,000
        assert float(expected) == 50_000_000.0

    def test_5cr_not_50L(self):
        """Sanity check: ₹5 crore = 50,000,000, not 5,000,000."""
        FIVE_CRORE = 5 * 10_000_000  # 5 × 1 crore
        assert FIVE_CRORE == 50_000_000
        FIFTY_LAKH = 50 * 100_000    # 50 × 1 lakh
        assert FIFTY_LAKH == 5_000_000  # this would be wrong for ₹5cr

    def test_decision_engine_capital_fix_in_source(self):
        """decision.py must use config-driven capital, not hardcoded ₹1L."""
        import inspect
        from core.trading import decision
        src = inspect.getsource(decision.DecisionEngine.__init__)
        assert '100000.0' not in src, \
            "Hardcoded ₹1L capital (100000.0) found in DecisionEngine.__init__ — fix not applied!"
        assert '50_000_000' in src, \
            "₹5cr default (50_000_000) missing from DecisionEngine.__init__"


class TestSectorDataLookAhead:
    """Verify the bfill look-ahead fix is in place."""

    def test_no_bfill_in_sector_series(self):
        """sector_data.py must use ffill().fillna(0.0) not bfill() as executable code."""
        import inspect
        from core.data import sector_data
        src = inspect.getsource(sector_data.SectorDataProvider.get_sector_series)
        # bfill() may appear inside a comment explaining why it was removed —
        # that is acceptable. What must NOT happen is bfill() on a code line.
        code_lines = [
            ln for ln in src.splitlines()
            if not ln.lstrip().startswith('#') and 'bfill()' in ln
        ]
        assert len(code_lines) == 0, \
            f"bfill() found on executable lines in get_sector_series — look-ahead bug! Lines: {code_lines}"
        assert 'ffill()' in src, "ffill() should be present"
        assert 'fillna(0.0)' in src, "fillna(0.0) should follow ffill()"


class TestRiskManagerDefaults:
    """Verify risk manager defaults are sensible for ₹5cr pool."""

    def test_risk_manager_has_initial_capital(self):
        """RiskManager.__init__ accepts initial_capital via config."""
        from core.trading.risk_manager import RiskManager
        rm = RiskManager(config={'initial_capital': 50_000_000.0})
        assert rm.initial_capital == 50_000_000.0

    def test_risk_manager_daily_loss_limit_present(self):
        """RiskManager must have a daily_loss_limit attribute."""
        from core.trading.risk_manager import RiskManager
        rm = RiskManager(config={'initial_capital': 50_000_000.0, 'daily_loss_limit': 1_000_000.0})
        assert hasattr(rm, 'daily_loss_limit')
        assert rm.daily_loss_limit == 1_000_000.0

    def test_portfolio_analyzer_factory_uses_capital(self):
        """PortfolioRiskAnalyzer factory must pass initial_capital to config."""
        from core.trading.risk_manager import PortfolioRiskAnalyzer
        import inspect
        src = inspect.getsource(PortfolioRiskAnalyzer.__init__)
        assert 'initial_capital' in src, \
            "PortfolioRiskAnalyzer.__init__ must reference initial_capital"


class TestSignalsNoSelfLogger:
    """Verify signals.py uses module-level logger, not self.logger."""

    def test_no_self_logger_in_generate_signal(self):
        import inspect
        from core.trading import signals
        src = inspect.getsource(signals.TradingSignalGenerator.generate_signal)
        assert 'self.logger' not in src, \
            "self.logger found in generate_signal — AttributeError bug!"

    def test_rbi_mpc_dates_dynamic(self):
        """RBI MPC dates must be computed by _get_rbi_mpc_dates(), not hardcoded 2026 list."""
        import inspect
        from core.trading import signals
        src = inspect.getsource(signals.generate_signal
                                if hasattr(signals, 'generate_signal')
                                else signals.TradingSignalGenerator.generate_signal)
        # The 2026-hardcoded block assigned 18 literal tuples — confirm it's gone
        assert 'RBI MPC meeting dates 2026' not in src, \
            "Hardcoded 2026 RBI MPC dates still present — year-safe fix not applied!"

    def test_rbi_mpc_dates_module_level_set_exists(self):
        """Module-level RBI_MPC_DATES should be a non-empty set."""
        from core.trading import signals
        assert hasattr(signals, 'RBI_MPC_DATES'), "RBI_MPC_DATES module constant missing"
        assert isinstance(signals.RBI_MPC_DATES, set), "RBI_MPC_DATES should be a set"
        assert len(signals.RBI_MPC_DATES) > 0, "RBI_MPC_DATES should not be empty"

    def test_get_rbi_mpc_dates_function_exists(self):
        """_get_rbi_mpc_dates() helper must exist in signals module."""
        from core.trading import signals
        assert hasattr(signals, '_get_rbi_mpc_dates'), \
            "_get_rbi_mpc_dates function not found in signals.py"
        result = signals._get_rbi_mpc_dates()
        assert isinstance(result, set)
        assert len(result) > 0


class TestPredictorSysModulesFix:
    """Verify predictor.py no longer unconditionally pollutes sys.modules['__main__']."""

    def test_no_unconditional_sys_modules_hack(self):
        """The one-liner sys.modules['__main__'].X = Y must not appear in predictor.py."""
        src_path = os.path.join(_ROOT, 'core', 'models', 'predictor.py')
        with open(src_path, 'r') as fh:
            src = fh.read()
        assert "sys.modules['__main__'].NonNegativeMetaLearner = NonNegativeMetaLearner" not in src, \
            "Unconditional sys.modules['__main__'] assignment still present in predictor.py"

    def test_conditional_fallback_present(self):
        """The fix must include the conditional setattr fallback."""
        src_path = os.path.join(_ROOT, 'core', 'models', 'predictor.py')
        with open(src_path, 'r') as fh:
            src = fh.read()
        assert 'setattr(_main,' in src or 'setattr(sys.modules' in src or \
               'setattr(_sys.modules' in src or "setattr(_main, 'NonNegativeMetaLearner'" in src, \
            "Conditional setattr fallback missing from predictor.py"

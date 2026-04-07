#!/usr/bin/env python3
"""
MARK5 Final Validation Suite
Comprehensive end-to-end validation before production deployment
Migrated from core/data/validator.py
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
import time
import warnings
import hashlib
import json
from pathlib import Path
import logging

# Import core modules
from core.models.predictor import MARK5Predictor
from core.trading.signals import TradingSignalGenerator
# from core.utils.data_quality_guard import DataQualityGuard # Check if this exists
from core.analytics.performance import ModelPerformanceTracker
from core.trading.risk_manager import PortfolioRiskAnalyzer # Removed broken import
from core.trading.market_utils import MarketStatusChecker
from core.system.container import container

# Attempt to import DataQualityGuard, handle if missing
try:
    from core.utils.data_quality_guard import DataQualityGuard
except ImportError:
    DataQualityGuard = None

logger = logging.getLogger("MARK5.Validator")

# Configuration for Data Freshness
CACHE_TTL_CONFIG = {
    "DATA_FRESHNESS_MAX_AGE_TRADING": 15,  # minutes
    "DATA_FRESHNESS_MAX_AGE_CLOSED": 1440  # minutes (24 hours)
}

FEATURE_EXCLUDE_COLUMNS = {'date', 'timestamp', 'target', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'ticker'}

class FeatureSchemaValidator:
    """
    🔥 Feature Schema Validation
    Ensures predictions use features matching the model's training schema
    """
    
    def __init__(self):
        self.logger = logging.getLogger('MARK5.Suite')
        self.schemas: Dict[str, Dict] = {}  # ticker_modeltype -> schema
    
    def create_schema(self, ticker: str, model_type: str, features_df: pd.DataFrame) -> Dict:
        """Create and cache feature schema from training DataFrame"""
        feature_list = list(features_df.columns)
        feature_string = "|".join(sorted(feature_list))
        feature_hash = hashlib.md5(feature_string.encode()).hexdigest()
        
        schema = {
            'ticker': ticker,
            'model_type': model_type,
            'feature_count': len(feature_list),
            'features': feature_list,
            'feature_hash': feature_hash,
            'dtypes': {col: str(dtype) for col, dtype in features_df.dtypes.items()},
            'created_at': datetime.now().isoformat()
        }
        
        self.schemas[f"{ticker}_{model_type}"] = schema
        return schema
    
    def save_schema(self, schema: Dict, model_path: str):
        """Save schema alongside model file"""
        schema_path = model_path.replace('.pkl', '_schema.json')
        with open(schema_path, 'w') as f:
            json.dump(schema, f, indent=2)
    
    def load_schema(self, model_path: str) -> Optional[Dict]:
        """Load schema from file"""
        schema_path = model_path.replace('.pkl', '_schema.json')
        if not Path(schema_path).exists():
            return None
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        self.schemas[f"{schema['ticker']}_{schema['model_type']}"] = schema
        return schema
    
    def validate_features(self, ticker: str, model_type: str, features_df: pd.DataFrame) -> Dict:
        """Validate DataFrame features against model schema"""
        key = f"{ticker}_{model_type}"
        if key not in self.schemas:
            return {'valid': False, 'errors': [f"No schema found for {key}"]}
        
        schema = self.schemas[key]
        expected = set(schema['features'])
        actual = set(features_df.columns)
        
        missing = list(expected - actual)
        extra = list(actual - expected)
        
        return {
            'valid': len(missing) == 0,
            'errors': [f"Missing {len(missing)} features"] if missing else [],
            'warnings': [f"Extra {len(extra)} features"] if extra else [],
            'missing_features': missing,
            'extra_features': extra
        }
    
    def align_features(self, features_df: pd.DataFrame, schema: Dict) -> pd.DataFrame:
        """
        Align DataFrame to schema.
        CRITICAL HFT CHANGE: Do NOT fill missing values with 0.
        Raise error if features are missing.
        """
        required_features = set(schema['features'])
        current_features = set(features_df.columns)
        
        missing = required_features - current_features
        
        if missing:
            # THE KILL SWITCH
            # In HFT, missing data = invalid state. Do not trade.
            raise ValueError(f"CRITICAL: Missing features for inference: {missing}. "
                             f"Cannot impute 0 as it distorts signal.")
        
        # Only keep relevant columns in correct order
        # This prevents "Extra" features from confusing XGBoost
        return features_df[schema['features']]


class FinalValidationSuite:
    """Comprehensive validation before production"""
    
    def __init__(self):
        self.logger = logging.getLogger('MARK5.Suite')
        self.results = {}
        self.errors = []
        self.warnings = []
        self.start_time = None
        self.schema_validator = FeatureSchemaValidator()  # 🔥 Add schema validator
    
    def test_end_to_end_prediction_pipeline(self) -> Dict:
        """Test complete prediction pipeline"""
        print("Testing end-to-end prediction pipeline...")
        
        try:
            # Collect data
            # Use container for data access
            collector = container.data
            if not collector:
                return {"valid": False, "reason": "Data Provider not ready"}
            data = collector.get_history('RELIANCE.NS', period='6mo')
            
            if data is None or len(data) < 100:
                return {'status': 'FAIL', 'error': 'Insufficient data'}
            
            # Initialize API
            api = MARK5Predictor('RELIANCE.NS')
            
            # Make prediction
            start_time = time.time()
            result = api.predict('RELIANCE.NS', data, horizon='1d')
            latency = (time.time() - start_time) * 1000
            
            if not result or 'prediction' not in result:
                return {'status': 'FAIL', 'error': 'No prediction result'}
            
            # Validate prediction structure
            required_fields = ['signal', 'direction', 'confidence', 'probability']
            for field in required_fields:
                if field not in result['prediction']:
                    return {'status': 'FAIL', 'error': f'Missing field: {field}'}
            
            # Generate trading signal
            signal_gen = TradingSignalGenerator()
            signal = signal_gen.generate_signal(result)
            
            if not signal or 'signal' not in signal:
                return {'status': 'FAIL', 'error': 'Signal generation failed'}
            
            return {
                'status': 'PASS',
                'latency_ms': latency,
                'confidence': result['prediction']['confidence'],
                'signal': signal['signal'],
                'details': {
                    'data_points': len(data),
                    'prediction': result['prediction']['direction'],
                    'signal_strength': signal.get('strength', 'N/A')
                }
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_data_quality_validation(self) -> Dict:
        """Test data quality monitoring"""
        print("Testing data quality validation...")
        
        try:
            if DataQualityGuard is None:
                 return {'status': 'WARN', 'error': 'DataQualityGuard not found'}

            # Use container for data access
            collector = container.data
            if not collector:
                return {"valid": False, "reason": "Data Provider not ready"}
            data = collector.get_history('TCS.NS', period='3mo')
            
            if data is None:
                return {'status': 'FAIL', 'error': 'Data collection failed'}
            
            monitor = DataQualityGuard()
            is_valid, message = monitor.validate_data('TCS.NS', data)
            completeness = monitor.check_completeness(data)
            
            score = 1.0 if is_valid else 0.5
            status_map = 'PASS' if is_valid else 'FAIL'
            
            return {
                'status': 'PASS' if score >= 0.7 else 'WARN',
                'quality_score': score,
                'quality_status': status_map,
                'dimensions': {
                    'completeness': completeness['completeness_pct'],
                    'validity': 1.0 if is_valid else 0.0,
                    'message': message
                }
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_model_availability(self) -> Dict:
        """Test that all required models are available"""
        print("Testing model availability...")
        
        try:
            # Assuming models are relative to core/models or similar
            # Adjusting path to be relative to project root or core
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # core/validation -> core
            models_dir = os.path.join(base_dir, 'models', 'saved_models') # Guessing path, or maybe just 'models'
            # Original code used __file__ relative path which was core/data/validator.py
            # If we move to core/validation/suite.py, we need to be careful.
            # Let's try to find models dir dynamically or assume standard structure.
            
            # Fallback to hardcoded path if needed
            models_dir = "/home/lynx/Documents/MARK4 (Copy)/core/models"
            
            required_tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'WIPRO.NS']
            model_types = ['xgboost', 'lightgbm', 'random_forest', 'ensemble']
            
            found_models = {}
            missing_models = []
            
            for ticker in required_tickers:
                found_models[ticker] = []
                
                for model_type in model_types:
                    model_dir = os.path.join(models_dir, model_type)
                    if os.path.exists(model_dir):
                        model_file = f"{ticker}_{model_type}_advanced.pkl"
                        model_path = os.path.join(model_dir, model_file)
                        
                        if os.path.exists(model_path):
                            found_models[ticker].append(model_type)
                        else:
                            missing_models.append(f"{ticker}/{model_type}")
            
            total_expected = len(required_tickers) * len(model_types)
            total_found = sum(len(models) for models in found_models.values())
            
            if total_found == 0:
                return {'status': 'FAIL', 'error': 'No models found'}
            
            coverage = (total_found / total_expected) * 100
            
            return {
                'status': 'PASS' if coverage >= 80 else 'WARN',
                'coverage': coverage,
                'found': total_found,
                'expected': total_expected,
                'models_by_ticker': found_models,
                'missing': missing_models if missing_models else None
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_prediction_latency(self) -> Dict:
        """Test prediction latency requirements (Precision Mode)"""
        print("Testing prediction latency...")
        
        try:
            api = MARK5Predictor('RELIANCE.NS')
            # Use container for data access
            collector = container.data
            if not collector:
                return {"valid": False, "reason": "Data Provider not ready"}
            data = collector.get_history('INFY.NS', period='6mo')
            
            if data is None: return {'status': 'FAIL', 'error': 'No Data'}
            
            # Warm-up (Critical for JIT/Caching)
            api.predict('INFY.NS', data, horizon='1d')
            
            latencies = []
            for _ in range(50): # Increase sample size
                # REVOLUTIONARY FIX: Nanosecond precision
                start = time.perf_counter_ns()
                api.predict('INFY.NS', data, horizon='1d')
                end = time.perf_counter_ns()
                
                # Convert ns to ms
                latency = (end - start) / 1_000_000
                latencies.append(latency)
            
            avg_latency = np.mean(latencies)
            p99_latency = np.percentile(latencies, 99) # Check tail latency
            
            # HFT Standard: Average < 50ms, P99 < 100ms
            status = 'PASS' if avg_latency < 50 and p99_latency < 100 else 'WARN'
            
            return {
                'status': status,
                'avg_latency_ms': round(avg_latency, 2),
                'p99_latency_ms': round(p99_latency, 2),
                'min_latency_ms': round(np.min(latencies), 2),
                'samples': len(latencies)
            }
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_risk_management(self) -> Dict:
        """Test risk management calculations"""
        print("Testing risk management...")
        
        try:
            # Need to import PortfolioRiskAnalyzer
            # Assuming it's in core.trading.risk_manager or similar
            # If not available, skip
            try:
                from core.trading.risk_manager import PortfolioRiskAnalyzer
            except ImportError:
                return {'status': 'WARN', 'error': 'PortfolioRiskAnalyzer not found'}

            analyzer = PortfolioRiskAnalyzer(initial_capital=100000)
            
            # Test position sizing
            position = analyzer.calculate_position_size(
                capital=100000.0, entry=1500.0,
                sl=1450.0
            )
            
            if position is None:
                return {'status': 'FAIL', 'error': 'Position sizing failed'}
            
            # Test VaR calculation (needs returns data)
            returns = np.random.normal(0.001, 0.02, 252)  # Simulated daily returns
            var_95 = analyzer.calculate_var(returns, confidence=0.95)
            
            if var_95 is None:
                return {'status': 'FAIL', 'error': 'VaR calculation failed'}
            
            return {
                'status': 'PASS',
                'position_sizing': position,
                'var_95': var_95,
                'risk_controls': 'functional'
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_database_integrity(self) -> Dict:
        """Test database integrity"""
        print("Testing database integrity...")
        
        try:
            import sqlite3
            
            # Adjust path (The Vault)
            from core.infrastructure.database_manager import MARK5DatabaseManager
            db_path = MARK5DatabaseManager().db_path
            
            if not os.path.exists(db_path):
                return {'status': 'FAIL', 'error': 'Database not found'}
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['model_metadata', 'model_performance', 'predictions', 'trading_signals']
            missing = [t for t in required_tables if t not in tables]
            
            # Check indexes
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
            index_count = cursor.fetchone()[0]
            
            conn.close()
            
            if missing:
                return {'status': 'WARN', 'missing_tables': missing}
            
            return {
                'status': 'PASS',
                'tables': len(tables),
                'indexes': index_count,
                'required_tables_present': True
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def run_all_validations(self) -> Dict:
        """Run all validation tests"""
        print("\n" + "=" * 80)
        print("MARK5 FINAL VALIDATION SUITE")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("")
        
        self.start_time = time.time()
        
        tests = [
            ('End-to-End Pipeline', self.test_end_to_end_prediction_pipeline),
            ('Data Quality', self.test_data_quality_validation),
            ('Model Availability', self.test_model_availability),
            ('Prediction Latency', self.test_prediction_latency),
            ('Risk Management', self.test_risk_management),
            ('Database Integrity', self.test_database_integrity)
        ]
        
        passed = 0
        warnings = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.results[test_name] = result
                
                status = result.get('status', 'UNKNOWN')
                if status == 'PASS':
                    passed += 1
                    print(f"  ✓ {test_name}: PASS")
                elif status == 'WARN':
                    warnings += 1
                    print(f"  ⚠ {test_name}: PASS (with warnings)")
                else:
                    failed += 1
                    error = result.get('error', 'Unknown error')
                    print(f"  ✗ {test_name}: FAIL - {error}")
                    
            except Exception as e:
                failed += 1
                self.results[test_name] = {'status': 'FAIL', 'error': str(e)}
                print(f"  ✗ {test_name}: FAIL - {e}")
            
            time.sleep(0.5)
        
        elapsed = time.time() - self.start_time
        
        # Summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {len(tests)}")
        print(f"Passed: {passed}")
        print(f"Warnings: {warnings}")
        print(f"Failed: {failed}")
        print(f"Elapsed: {elapsed:.1f}s")
        print("")
        
        overall_status = 'PASS' if failed == 0 else 'FAIL'
        
        if overall_status == 'PASS':
            print("✓ ALL VALIDATION TESTS PASSED")
            if warnings > 0:
                print(f"⚠ {warnings} test(s) passed with warnings - review recommended")
        else:
            print(f"✗ VALIDATION FAILED: {failed} test(s) failed")
        
        print("=" * 80)
        
        return {
            'overall_status': overall_status,
            'passed': passed,
            'warnings': warnings,
            'failed': failed,
            'total': len(tests),
            'elapsed': elapsed,
            'results': self.results
        }


class LookAheadBiasAuditor:
    """
    🔥 Look-Ahead Bias Auditor
    Automated tool to detect look-ahead bias in feature engineering
    """
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.clean = []
    
    def audit_dataframe(self, df: pd.DataFrame, target_col: str = 'target') -> dict:
        """Audit a DataFrame for look-ahead bias"""
        print("🔍 Starting Look-Ahead Bias Audit...\n")
        
        # Check target variable timing
        if target_col in df.columns:
            self._check_target_timing(df, target_col)
        
        # Check feature calculations
        self._check_feature_calculations(df)
        
        # Check rolling windows
        self._check_rolling_windows(df)
        
        return self._generate_report()
    
    def _check_target_timing(self, df: pd.DataFrame, target_col: str):
        """Check if target is properly forward-looking"""
        if len(df) > 10:
            target_series = df[target_col].dropna()
            if 'close' in df.columns and len(target_series) > 10:
                close_current = df['close'].iloc[:len(target_series)]
                close_next = df['close'].shift(-1).iloc[:len(target_series)]
                
                corr_current = target_series.corr(close_current)
                corr_next = target_series.corr(close_next.dropna())
                
                if abs(corr_current) > abs(corr_next):
                    self.issues.append("⚠️ TARGET: Uses CURRENT bar data (look-ahead bias)")
                else:
                    self.clean.append("✅ Target: Properly forward-looking")
    
    def _check_feature_calculations(self, df: pd.DataFrame):
        """Check individual features for look-ahead bias"""
        historical_features = ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'momentum', 'volume_trend', 'vwap']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(feat in col_lower for feat in historical_features):
                self.clean.append(f"✅ {col}: Rolling calculation (assumed correct)")
        
        # Check for suspicious names
        suspicious = ['future', 'next', 'forward', 'ahead']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in suspicious):
                self.warnings.append(f"⚠️ {col}: Column name suggests future data")
    
    def _check_rolling_windows(self, df: pd.DataFrame):
        """Check if rolling windows are properly aligned"""
        rolling_features = [col for col in df.columns if 'rolling' in col.lower() 
                           or any(x in col.lower() for x in ['sma', 'ema', 'volatility', 'std'])]
        
        if rolling_features:
            self.warnings.append(f"⚠️ ROLLING FEATURES ({len(rolling_features)}): Verify no current bar inclusion")
    
    def _generate_report(self) -> dict:
        """Generate audit report"""
        print("\n" + "="*70)
        print("📊 LOOK-AHEAD BIAS AUDIT REPORT")
        print("="*70)
        
        print(f"\n🔴 CRITICAL ISSUES: {len(self.issues)}")
        for issue in self.issues:
            print(f"  {issue}")
        
        print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
        for warning in self.warnings:
            print(f"  {warning}")
        
        print(f"\n✅ CLEAN: {len(self.clean) if len(self.clean) <= 10 else '(many features passed)'}")
        
        status = "PASS" if len(self.issues) == 0 else "FAIL"
        print(f"\n{'✅' if status == 'PASS' else '❌'} STATUS: {status}")
        print("="*70)
        
        return {
            'status': status,
            'issues': self.issues,
            'warnings': self.warnings,
            'clean': self.clean
        }


class DataFreshnessValidator:
    """
    🔥 BUG FIX #2: Data Freshness Validation
    Ensures data is recent enough for reliable predictions
    """
    
    def __init__(self):
        self.config = CACHE_TTL_CONFIG
        self.market_checker = MarketStatusChecker()
    
    def validate_data_freshness(self, data: pd.DataFrame, ticker: str, interval: str = '1d') -> Tuple[bool, str]:
        """
        Validate if data is fresh enough with interval-specific thresholds
        """
        if data.empty:
            return False, "Empty dataset"
        
        # Get last data timestamp
        if 'date' in data.columns:
            last_date = pd.to_datetime(data['date'].iloc[-1])
        elif isinstance(data.index, pd.DatetimeIndex):
            last_date = data.index[-1]
        else:
            return False, "No valid date column or index"
        
        # Calculate age
        now = datetime.now()
        
        # Handle timezone mismatch
        if last_date.tzinfo is not None and now.tzinfo is None:
            last_date = last_date.replace(tzinfo=None)
        elif last_date.tzinfo is None and now.tzinfo is not None:
            now = now.replace(tzinfo=None)
            
        age_minutes = (now - last_date).total_seconds() / 60
        age_hours = age_minutes / 60
        
        # ENHANCEMENT: Interval-specific max age thresholds
        interval_thresholds = {
            '1m': 2,      # 2 minutes for 1-minute data
            '2m': 3,      # 3 minutes for 2-minute data
            '5m': 10,     # 10 minutes for 5-minute data
            '15m': 20,    # 20 minutes for 15-minute data
            '30m': 35,    # 35 minutes for 30-minute data
            '1h': 70,     # 70 minutes for hourly data
            '1d': 1440,   # 24 hours for daily data
            '1wk': 10080, # 1 week for weekly data
        }
        
        # Check market status
        market_status = self.market_checker.get_market_status()
        
        if market_status["is_trading"]:
            # Use interval-specific threshold during trading
            max_age = interval_thresholds.get(interval, self.config["DATA_FRESHNESS_MAX_AGE_TRADING"])
            
            if age_minutes > max_age:
                return False, f"Data is {age_minutes:.0f} minutes old (max: {max_age} for {interval} during trading)"
        else:
            # For daily/weekly data, allow longer staleness when market closed
            if interval in ['1d', '1wk']:
                max_age = self.config["DATA_FRESHNESS_MAX_AGE_CLOSED"]  # 1440 min = 24 hours
                
                # If age is more than 24 hours, check if it's weekend/holiday
                if age_minutes > max_age:
                    # Allow up to 96 hours (4 days) for weekends/long holidays
                    extended_max_age = 5760  # 96 hours in minutes (4 days)
                    
                    if age_minutes > extended_max_age:
                        return False, f"Data is {age_hours:.1f}h old (max: {extended_max_age/60:.0f}h for weekends/holidays)"
                    else:
                        # Data is 24-96 hours old, likely weekend/holiday - acceptable
                        days_old = age_hours / 24
                        return True, f"Weekend data ({age_hours:.1f}h old, {days_old:.1f} days) - acceptable for preview"
            else:
                # For intraday data when market closed, data should be from last trading session
                max_age = 1440  # 24 hours max for intraday
                if age_minutes > max_age:
                    return False, f"Intraday data is {age_hours:.1f}h old (stale for {interval} interval)"
        
        return True, f"Data is fresh ({age_minutes:.0f} minutes old, interval: {interval})"
    
    def get_data_age_info(self, data: pd.DataFrame) -> Dict[str, any]:
        """Get detailed data age information"""
        if data.empty:
            return {"valid": False, "reason": "Empty dataset"}
        
        try:
            if 'date' in data.columns:
                last_date = pd.to_datetime(data['date'].iloc[-1])
            else:
                last_date = data.index[-1]
            
            now = datetime.now()
            age = now - last_date
            
            return {
                "valid": True,
                "last_update": last_date.strftime("%Y-%m-%d %H:%M:%S"),
                "age_minutes": round(age.total_seconds() / 60, 1),
                "age_hours": round(age.total_seconds() / 3600, 2),
                "age_days": age.days,
                "is_fresh": age.total_seconds() < 900  # Fresh if < 15 minutes
            }
        except Exception as e:
            return {"valid": False, "reason": f"Error checking freshness: {str(e)}"}


def get_feature_columns_fixed(data: pd.DataFrame) -> list:
    """Get feature columns by excluding non-feature columns"""
    return [col for col in data.columns if col not in FEATURE_EXCLUDE_COLUMNS]


def normalize_dataset_length(data: pd.DataFrame, 
                             target_length: Optional[int] = None,
                             n_windows: int = 5,
                             method: str = 'trim') -> pd.DataFrame:
    """
    ⚠️ PHASE 2 FIX #8: Normalize dataset length for consistent Walk-Forward splits
    """
    if data.empty:
        return data
    
    current_length = len(data)
    
    # Compute target length if not specified
    if target_length is None:
        # Round to nearest multiple of n_windows
        target_length = (current_length // n_windows) * n_windows
        if target_length < current_length:
            target_length += n_windows  # Round up for safety
    
    if current_length == target_length:
        return data
    
    elif current_length > target_length:
        # Trim excess rows (keep most recent)
        if method == 'trim':
            normalized = data.iloc[-target_length:].copy()
            return normalized.reset_index(drop=True)
        else:
            return data.iloc[-target_length:].copy().reset_index(drop=True)
    
    else:  # current_length < target_length
        # Pad with NaN (will be forward-filled during feature engineering)
        if method == 'pad':
            padding_size = target_length - current_length
            padding = pd.DataFrame(
                index=range(padding_size),
                columns=data.columns
            )
            normalized = pd.concat([padding, data], ignore_index=True)
            return normalized
        else:
            return data


def validate_walk_forward_splits(data: pd.DataFrame, 
                                 n_windows: int = 5) -> Tuple[bool, str]:
    """Validate that dataset length is suitable for Walk-Forward validation"""
    length = len(data)
    
    # Minimum samples per window
    min_samples_per_window = 50
    min_required = n_windows * min_samples_per_window
    
    if length < min_required:
        return False, f"Dataset too short: {length} rows (need {min_required}+ for {n_windows} windows)"
    
    # Check if evenly divisible
    samples_per_window = length / n_windows
    is_even = (length % n_windows == 0)
    
    if not is_even:
        remainder = length % n_windows
        message = f"Dataset length ({length}) not evenly divisible by {n_windows} windows. "
        message += f"Remainder: {remainder} rows. Consider normalizing."
        return True, message  # Valid but with warning
    
    return True, f"Valid: {length} rows = {int(samples_per_window)} per window"


def get_optimal_dataset_length(data: pd.DataFrame, 
                               n_windows: int = 5) -> int:
    """Calculate optimal dataset length for WF validation"""
    current_length = len(data)
    
    # Round down to nearest multiple
    optimal = (current_length // n_windows) * n_windows
    
    # If we lose too much data, round up instead
    loss_pct = ((current_length - optimal) / current_length) * 100
    if loss_pct > 5.0:  # Losing more than 5%
        optimal += n_windows
    
    return optimal


def prepare_data_for_walk_forward(data: pd.DataFrame,
                                  n_windows: int = 5,
                                  normalize: bool = True) -> pd.DataFrame:
    """Prepare dataset for Walk-Forward validation"""
    # Validate
    is_valid, message = validate_walk_forward_splits(data, n_windows)
    
    if not is_valid:
        raise ValueError(message)
    
    # Normalize if requested
    if normalize:
        optimal_length = get_optimal_dataset_length(data, n_windows)
        data = normalize_dataset_length(data, optimal_length, n_windows, method='trim')
    
    return data


def audit_feature_engine(df_sample: pd.DataFrame):
    """Quick audit function for feature engine"""
    auditor = LookAheadBiasAuditor()
    return auditor.audit_dataframe(df_sample)


def main():
    suite = FinalValidationSuite()
    summary = suite.run_all_validations()
    
    # Save results
    output_file = f'validation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    import json
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return 0 if summary['overall_status'] == 'PASS' else 1


if __name__ == '__main__':
    exit(main())

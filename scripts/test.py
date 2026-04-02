#!/usr/bin/env python3
"""
MARK5 PRODUCTION VALIDATION PIPELINE v5.0 — META-LABELING EDITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v5.0: Updated for Meta-Labeling pipeline.
      - Models predict probability of hitting Profit Target (Class 1).
      - Strategy simulation uses Triple-Barrier exits (PT, SL, Time).
      - Key metrics are Accuracy, Brier Score, and realized Strategy Returns.
      - Supports --symbol for single-stock runs.
"""

import sys
import os
import argparse
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')
from datetime import time as datetime_time  # For session boundary checks

# Mark5 Environment Setup
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Load .env for API credentials (Kite API key, secret, access token)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(project_root, '.env'))
except ImportError:
    pass  # dotenv not installed — rely on shell-exported env vars

# Third-party imports
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.preprocessing import StandardScaler
    import joblib
except ImportError as e:
    print(f"❌ Missing critical library: {e}")
    print("Run: pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost")
    sys.exit(1)

# Kite Connect (primary data source)
KITE_DATA_AVAILABLE = False
try:
    from core.data.adapters.kite_adapter import KiteFeedAdapter
    KITE_DATA_AVAILABLE = True
except ImportError:
    pass

# yfinance (deprecated fallback only — will be removed)
YFINANCE_AVAILABLE = False
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    pass

if not KITE_DATA_AVAILABLE and not YFINANCE_AVAILABLE:
    print("❌ No data source available. Install kiteconnect or yfinance.")
    sys.exit(1)

# MARK5 Core Imports (Production Infrastructure)
MARK5_AVAILABLE = True
REGIME_DETECTOR_AVAILABLE = True    

try:
    from core.models.features import AdvancedFeatureEngine
    from core.models.registry import RobustModelRegistry
    from core.models.training.trainer import MARK5MLTrainer
    from core.models.predictor import MARK5Predictor
    from core.infrastructure.database_manager import MARK5DatabaseManager
    from core.trading.risk_manager import PortfolioRiskAnalyzer
    from core.analytics.journal import TradeJournal
    from core.utils.common import setup_logger
    from core.utils.config_manager import ConfigManager
    from core.trading.position_sizer import VolatilityAwarePositionSizer, MarketRegime
    from core.trading.market_utils import TransactionCostCalculator
    import joblib
    import json
    import core.models.predictor as predictor_module

    # NO MONKEY PATCHES — use real production thresholds for honest validation
    
    # Try to import MarketRegimeDetector separately - don't fail if missing
    try:
        from core.analytics.regime_detector import MarketRegimeDetector
        REGIME_DETECTOR_AVAILABLE = True
        print("✅ MarketRegimeDetector loaded")
    except (ImportError, AttributeError):
        print("⚠️ MarketRegimeDetector not available - using fallback")
        REGIME_DETECTOR_AVAILABLE = False
        
except ImportError as e:
    print(f"⚠️ MARK5 Core Import Error: {e}")
    import traceback
    traceback.print_exc()
    print("Falling back to standalone mode...")
    MARK5_AVAILABLE = False
    REGIME_DETECTOR_AVAILABLE = False
    from core.models.features import AdvancedFeatureEngine
    from core.utils.common import setup_logger
    
    # Mock components for standalone mode
    class PortfolioRiskAnalyzer:
        def __init__(self, *args, **kwargs): pass
        def calculate_max_drawdown(self, equity_curve): return 0.0
        def check_portfolio_risk(self, equity_curve): return {'is_safe': True, 'alerts': []}
        
    class TradeJournal:
        def __init__(self, *args, **kwargs): pass
        def log_trade(self, *args, **kwargs): pass
        
    class MarketRegimeDetector:
        def __init__(self, *args, **kwargs): pass
        def detect_regime(self, *args, **kwargs): return "NORMAL"

# Configure Logging
os.makedirs('logs', exist_ok=True)
logger = setup_logger("MARK5.ValidationPipeline", level=logging.INFO, log_file="logs/validation_pipeline.log")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Import Nifty 50 Universe
try:
    from scripts.nifty50_universe import NIFTY_50, NIFTY_50_TICKERS, NIFTY_TOP10, NIFTY_TEST3, MARK5_ALPHA, SECTORS, MIXED_150
except ImportError:
    try:
        from nifty50_universe import NIFTY_50, NIFTY_50_TICKERS, NIFTY_TOP10, NIFTY_TEST3, MARK5_ALPHA, SECTORS, MIXED_150
    except ImportError:
        NIFTY_50 = {}
        NIFTY_50_TICKERS = ['RELIANCE.NS']
        NIFTY_TOP10 = ['RELIANCE.NS']
        NIFTY_TEST3 = ['RELIANCE.NS']
        MARK5_ALPHA = ['SBIN.NS', 'RELIANCE.NS', 'HINDUNILVR.NS']
        MIXED_150 = ['RELIANCE.NS']
        SECTORS = {}

WATCHLIST = NIFTY_TOP10 if 'NIFTY_TOP10' in dir() else ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'ITC.NS']

# INITIAL PLACEHOLDERS (Will be updated dynamically by CLI arguments)
end_date_obj = datetime.now()
start_date_obj = end_date_obj - timedelta(days=1100)  # 3 years of daily bars
START_DATE = start_date_obj.strftime('%Y-%m-%d')
END_DATE = end_date_obj.strftime('%Y-%m-%d')
INTERVAL = '1d'

# Absolute path resolutions
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'reports', 'validation_2026')
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
DATA_CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', 'cache')

# Create directories
for directory in [OUTPUT_DIR, CHARTS_DIR, DATA_CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# STEP 1: DATA COLLECTION WITH QUALITY CHECKS
# ═══════════════════════════════════════════════════════════════════

def download_data(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Download historical data via Kite Connect API, with yfinance fallback.
    
    FIX A-04: Primary source is now Kite Connect historical API.
    yfinance is used ONLY as a deprecation fallback during migration.
    """
    data = {}
    quality_records = []
    logger.info(f"📥 Downloading Data ({START_DATE} to {END_DATE}) for {len(symbols)} stocks")
    
    # Determine which data source to use
    use_kite = KITE_DATA_AVAILABLE and os.getenv('KITE_ACCESS_TOKEN', '') not in ('', 'YOUR_ACCESS_TOKEN_HERE')
    
    if use_kite:
        logger.info("📡 Data source: Kite Connect API")
    elif YFINANCE_AVAILABLE:
        logger.warning("⚠️ DEPRECATED: Using yfinance fallback. Migrate to Kite Connect.")
    else:
        logger.error("❌ No data source configured")
        return data
    
    # Pre-initialize Kite adapter once if using Kite
    kite_adapter = None
    if use_kite:
        try:
            config = {
                'api_key': os.getenv('KITE_API_KEY', ''),
                'api_secret': os.getenv('KITE_API_SECRET', ''),
                'access_token': os.getenv('KITE_ACCESS_TOKEN', ''),
            }
            kite_adapter = KiteFeedAdapter(config)
            from kiteconnect import KiteConnect
            kite_adapter.kite = KiteConnect(api_key=config['api_key'])
            kite_adapter.kite.set_access_token(config['access_token'])
            
            # Build instrument map
            instruments = kite_adapter.kite.instruments('NSE')
            kite_adapter._build_instrument_map(instruments)
            logger.info(f"Kite instrument map: {len(kite_adapter.symbol_to_token)} symbols")
        except Exception as e:
            logger.error(f"Kite initialization failed: {e}. Falling back to yfinance.")
            kite_adapter = None
            use_kite = False
    
    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {symbol}...", end=" ", flush=True)
        
        try:
            # Check Parquet cache first
            cache_path = os.path.join(DATA_CACHE_DIR, f"{symbol.replace('.', '_')}_{INTERVAL}.parquet")
            use_cache = False
            
            if os.path.exists(cache_path):
                cache_age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
                if cache_age_hours < 168:  # 7-day cache for historical data
                    df = pd.read_parquet(cache_path)
                    use_cache = True
                    print(f"cached ({len(df)} rows)", flush=True)
            
            if not use_cache:
                df = None
                
                if use_kite and kite_adapter:
                    # === KITE CONNECT (Primary) ===
                    from datetime import datetime as dt
                    import pytz
                    IST = pytz.timezone('Asia/Kolkata')
                    from_dt = IST.localize(dt.strptime(START_DATE, '%Y-%m-%d'))
                    to_dt = IST.localize(dt.strptime(END_DATE, '%Y-%m-%d'))
                    
                    df = kite_adapter.fetch_ohlcv(
                        symbol=symbol,
                        from_date=from_dt,
                        to_date=to_dt,
                        interval=INTERVAL
                    )
                    
                    # Session boundary enforcement: keep only market hours (9:15-15:30 IST) if intraday
                    if df is not None and not df.empty and INTERVAL != '1d':
                        if df.index.tz is None:
                            import pytz as _pytz
                            df.index = df.index.tz_localize(_pytz.timezone('Asia/Kolkata'))
                        market_open = datetime_time(9, 15)
                        market_close = datetime_time(15, 30)
                        bar_times = df.index.time
                        session_mask = [(t >= market_open and t <= market_close) for t in bar_times]
                        pre_filter = len(df)
                        df = df.loc[session_mask]
                        logger.info(f"   Session filter: {pre_filter} → {len(df)} bars (removed {pre_filter - len(df)} off-hours bars)")
                    
                    # Strip timezone for compatibility with existing pipeline
                    if df is not None and not df.empty and df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                elif YFINANCE_AVAILABLE:
                    # === YFINANCE (Deprecated fallback) ===
                    for attempt in range(3):
                        try:
                            df = yf.download(
                                symbol, start=START_DATE, end=END_DATE,
                                interval=INTERVAL, progress=False, auto_adjust=True
                            )
                            if df is not None and not df.empty:
                                break
                        except Exception:
                            if attempt < 2:
                                time.sleep(3)
                    
                    if df is not None and not df.empty:
                        # Handle yfinance MultiIndex columns
                        if isinstance(df.columns, pd.MultiIndex):
                            try:
                                df = df.xs(symbol, level=0, axis=1)
                            except KeyError:
                                pass
                        
                        # Standardize columns
                        new_cols = []
                        for c in df.columns:
                            if isinstance(c, tuple):
                                new_cols.append(str(c[0]).lower() if len(c) > 0 else str(c).lower())
                            else:
                                new_cols.append(str(c).lower())
                        df.columns = new_cols
                
                if df is None or df.empty:
                    print("❌ no data")
                    quality_records.append({'symbol': symbol, 'status': 'no_data', 'rows': 0})
                    continue
                
                # FIX S-03: Max 2-period forward-fill per RULE 49
                # Was: df = df.ffill().bfill() — blindly filled ALL gaps
                df = df.ffill(limit=2)
                # Only backfill the very first rows (if data starts with NaN)
                first_valid = df.first_valid_index()
                if first_valid is not None:
                    loc = df.index.get_loc(first_valid)
                    if loc > 0:
                        df.iloc[:loc] = df.iloc[:loc].bfill()
                
                # Save to Parquet cache
                df.to_parquet(cache_path)
                source = "kite" if (use_kite and kite_adapter) else "yfinance"
                print(f"{source} ({len(df)} rows)", flush=True)
            
            # Data Quality Checks
            quality_score, quality_detail = run_quality_checks(df, symbol)
            quality_records.append(quality_detail)
            
            if quality_score < 0.5:
                print(f"    ⚠️ Low quality ({quality_score:.0%}), skipping")
                continue
            
            data[symbol] = df
            
        except Exception as e:
            print(f"❌ {e}")
            quality_records.append({'symbol': symbol, 'status': 'error', 'rows': 0, 'error': str(e)})
    
    # Export quality report
    if quality_records:
        quality_df = pd.DataFrame(quality_records)
        quality_path = os.path.join(OUTPUT_DIR, 'data_quality_report.csv')
        quality_df.to_csv(quality_path, index=False)
        logger.info(f"📄 Data quality report: {quality_path}")
    
    return data

def run_quality_checks(df: pd.DataFrame, symbol: str) -> Tuple[float, Dict]:
    """Run data quality checks. Returns (score 0-1, detail dict)."""
    detail = {
        'symbol': symbol,
        'status': 'ok',
        'rows': len(df),
        'start_date': str(df.index[0].date()) if len(df) > 0 else '',
        'end_date': str(df.index[-1].date()) if len(df) > 0 else '',
    }
    
    checks_passed = 0
    total_checks = 6
    
    # 1. Sufficient data (>15000 bars for 5-min, ~200 trading days)
    if len(df) > 15000:
        checks_passed += 1
    detail['sufficient_data'] = len(df) > 15000
    
    # 2. Low missing data
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if missing_pct < 0.05:
        checks_passed += 1
    detail['missing_pct'] = round(missing_pct * 100, 2)
    
    # 3. Price stability (no crazy vol)
    if 'close' in df.columns:
        price_std = df['close'].std()
        price_mean = df['close'].mean()
        if price_mean > 0 and (price_std / price_mean) < 2.0:
            checks_passed += 1
        detail['cv'] = round(price_std / price_mean, 3) if price_mean > 0 else 0
    
    # 4. No extreme moves (>5% in one 5-min bar)
    if 'close' in df.columns:
        returns = df['close'].pct_change()
        extreme_moves = (returns.abs() > 0.05).sum()
        if extreme_moves < len(df) * 0.001:
            checks_passed += 1
        detail['extreme_moves'] = int(extreme_moves)
    
    # 5. Has volume data
    if 'volume' in df.columns and df['volume'].sum() > 0:
        checks_passed += 1
        avg_vol = df['volume'].mean()
        detail['avg_volume'] = int(avg_vol)
        detail['zero_vol_days'] = int((df['volume'] == 0).sum())
    
    # 6. OHLC integrity (Low <= Open,Close <= High)
    if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        ohlc_valid = ((df['low'] <= df['open']) & (df['low'] <= df['close']) & 
                      (df['high'] >= df['open']) & (df['high'] >= df['close']))
        integrity_pct = ohlc_valid.mean()
        if integrity_pct > 0.99:
            checks_passed += 1
        detail['ohlc_integrity'] = round(integrity_pct * 100, 2)
    
    score = checks_passed / total_checks
    detail['quality_score'] = round(score, 2)
    return score, detail

# ═══════════════════════════════════════════════════════════════════
# STEP 2: PREDICTION GENERATION USING MARK5 ENGINE
# ═══════════════════════════════════════════════════════════════════

class MARK5ValidationEngine:
    """Uses production MARK5 components for validation"""
    
    def __init__(self, optimize: bool = False, skip_training: bool = False):
        self.optimize = optimize
        self.skip_training = skip_training
        self.feature_engine = AdvancedFeatureEngine()
        
        if MARK5_AVAILABLE:
            try:
                self.config_manager = ConfigManager()
                self.model_registry = RobustModelRegistry(
                    registry_path=os.path.join(project_root, "models", "registry.json"),
                    base_model_dir=os.path.join(project_root, "models")
                )
                self.db_manager = MARK5DatabaseManager()
                self.learning_engine = LearningEngine(self.db_manager)
                self.trainer = MARK5MLTrainer()
                self.trade_journal = TradeJournal(self.db_manager)
                
                try:
                    if REGIME_DETECTOR_AVAILABLE:
                        # FIX: Pass config and db_manager
                        self.regime_detector = MarketRegimeDetector(
                            config=self.config_manager.get_config(),
                            db_manager=self.db_manager
                        )
                        logger.info("✅ Using MARK5 MarketRegimeDetector")
                    else:
                        self.regime_detector = None
                        logger.info("ℹ️  Using built-in regime detection")
                except Exception as e:
                    logger.warning(f"⚠️ Could not initialize MarketRegimeDetector: {e}")
                    self.regime_detector = None
                    
                logger.info("✅ MARK5 Production components initialized")
            except Exception as e:
                logger.warning(f"⚠️ MARK5 components unavailable: {e}")
                import traceback
                traceback.print_exc()
                self.trainer = None
                self.trade_journal = None
        else:
            self.trainer = None
            self.trade_journal = None
    
    def generate_predictions(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using Real MARK5 Production Pipeline.
        1. Trains model on first 80% of data (persisting artifacts).
        2. "Paper Walks" through the final 20% using MARK5Predictor.
        """
        logger.info(f"🧠 Running MARK5 Production System for {symbol}...")
        
        try:
            # STEP 0: Prepare Raw Data for Training
            # IMPORTANT: Do NOT pre-engineer features here. The trainer calls
            # AdvancedFeatureEngine.engineer_all_features() internally.
            # Passing pre-engineered data causes "double engineering" → garbage features.
            logger.info("⚙️  Preparing raw OHLCV data for Training...")
            
            # CRITICAL FIX for Forex/Indices with Zero Volume
            if 'volume' not in df.columns or df['volume'].sum() == 0:
                logger.warning(f"⚠️ {symbol}: Zero/Missing volume detected. Injecting synthetic volume.")
                df['volume'] = 1000.0
            
            # STEP 1: Strict 80/20 Split on RAW data
            # Bound df to the requested window BEFORE splitting
            if START_DATE:
                df = df[df.index >= pd.Timestamp(START_DATE)].copy()
            if END_DATE:
                df = df[df.index <= pd.Timestamp(END_DATE)].copy()

            train_size = int(len(df) * 0.8)
            train_df_raw = df.iloc[:train_size].copy()
            
            # Identify Split Point by Timestamp
            split_time = train_df_raw.index[-1]
            logger.info(f"   Data Split Time: {split_time}")
            logger.info(f"   Train Size: {len(train_df_raw)} bars")
            
            # STEP 2: Train Production Model
            if not self.skip_training:
                logger.info("⚙️  Simulating Production Training Cycle...")
                if not self.trainer:
                    raise RuntimeError("MARK5MLTrainer not initialized")
                    
                # TRIGGER OPTUNA IF REQUESTED
                # Train on feature-engineered training split (REGRESSION mode)
                # Pass sector and pre-built context to trainer
                from scripts.nifty50_universe import NIFTY_50
                stock_sector = NIFTY_50.get(symbol, {}).get('sector', '') \
                               if hasattr(NIFTY_50, 'get') else ''
                
                # Reuse context already built for simulation
                self.trainer.config.sector = stock_sector
                self.trainer.config.feature_context = feature_context \
                    if 'feature_context' in dir() else None
                
                train_result = self.trainer.train_advanced_ensemble(symbol, train_df_raw)
                
                if train_result.get('status') != 'success':
                    logger.error(f"❌ Training failed: {train_result.get('reason')}")
                    return pd.DataFrame()
                
                # Classification metrics: Log Loss, AUC, Brier Score
                loss = train_result.get('log_loss', 0)
                auc = train_result.get('auc', 0)
                brier = train_result.get('brier', 0)
                logger.info(f"   ✅ Training Complete. AUC={auc:.4f} | Loss={loss:.4f} | Brier={brier:.4f}")
                logger.info(f"   💾 Artifacts persisted to models/{symbol}/v{train_result.get('version')}")
                
                # CRITICAL: Register Model in Registry so Predictor can find it
                logger.info("   📝 Registering model version with RobustModelRegistry...")
                try:
                    # Construct absolute path using project_root
                    model_version = train_result.get('version')
                    model_dir = os.path.join(project_root, "models", symbol, f"v{model_version}")
                    
                    # FIX: Registry expects a file path, not directory. pointing to features.json
                    # which is a guaranteed artifact.
                    registry_artifact_path = os.path.join(model_dir, "features.json")
                    
                    self.model_registry.register_model(
                        ticker=symbol,
                        model_type='ensemble',
                        path=os.path.abspath(registry_artifact_path),
                        metadata={
                            'metrics': {
                                'log_loss': train_result.get('log_loss', 0.0),
                                'roc_auc': train_result.get('roc_auc', 0.0),
                                'brier_score': train_result.get('brier_score', 0.0),
                                'timestamp': datetime.now().isoformat()
                            },
                            'version': model_version
                        }
                    )
                    logger.info(f"   ✅ Model v{model_version} registered successfully")
                except Exception as e:
                     logger.error(f"❌ Failed to register model: {e}")
            else:
                logger.info("⏭️  Skipping Training Cycle (using cached models)...")

            # STEP 3: Paper Walk (Simulate Live Trading)
            logger.info("🚶 Starting 'Paper Walk' Validation (Simulating Live Inference)...")
            
            # Initialize Predictor (it will load the model we just trained)
            predictor = MARK5Predictor(symbol)
            predictor.reload_artifacts()
            
            # ── FEATURE REGIME GATE ──────────────────────────────────────
            model_version = getattr(predictor, 'version', 0)
            model_dir = os.path.join(project_root, "models", symbol, f"v{model_version}") if model_version else ""
            features_path = os.path.join(model_dir, "features.json") if model_dir else ""
            try:
                with open(features_path) as f:
                    feature_data = json.load(f)
                
                # Check if it's a dict with shap_importance (newer schemas) or a flat list (older schemas)
                if isinstance(feature_data, dict) and 'shap_importance' in feature_data:
                    shap_importance = feature_data.get('shap_importance', {})
                    top_features = list(shap_importance.keys())[:10]
                elif isinstance(feature_data, list):
                    top_features = feature_data[:10]
                else:
                    top_features = []
                
                # Updated after Daily-Bar pivot — intraday features replaced
                TREND_FEATURES = {'relative_strength_nifty', 'sector_rel_strength', 'dist_52w_high', 'fii_flow_3d'}
                trend_count = len(TREND_FEATURES & set(top_features))
                
                logger.info(f"   🔍 Feature Regime Gate | {symbol} | "
                            f"Trend features in top 10: {trend_count}/{len(TREND_FEATURES)} | "
                            f"Top features: {top_features[:5]}")
                
                if trend_count < 1:
                    mean_rev_features = {'rsi_14', 'efficiency_ratio', 'volume_confirmation', 'atr_regime'}
                    found_mean_rev = mean_rev_features & set(top_features[:5])
                    logger.warning(f"   ⚠️ LOW TREND SIGNAL {symbol} (trend_count={trend_count})")
                    logger.warning(f"   ⚠️ Mean-reversion features driving model: {found_mean_rev}")
                    # WARNING ONLY — don't block predictions, let the model's probability
                    # hurdle and confidence gates filter bad signals instead
                    
            except FileNotFoundError:
                logger.warning(f"   ⚠️ features.json not found for {symbol}, proceeding without gate")
            except Exception as e:
                logger.warning(f"   ⚠️ Error checking Feature Regime Gate for {symbol}: {e}")
            # ─────────────────────────────────────────────────────────────
            
            # Define Test Set (Raw Data) based on Split Time
            # We use the original raw 'df' for the simulation loop because Predictor expects raw data.
            raw_test_df = df.iloc[train_size:].copy()
            
            logger.info(f"   Test Set Size: {len(raw_test_df)} bars")
            
            results = []
            log_interval = max(1, len(raw_test_df) // 10)
            
            # Pre-calculate True Returns on FULL raw df for validation
            full_df = df.copy()
            PROFIT_TARGET_PCT = Decimal('0.02')   # placeholder, overwritten per bar
            STOP_LOSS_PCT = Decimal('0.01')        # placeholder, overwritten per bar
            # 1. Simulates Triple-Barrier exits (Target 4%, Stop 2%, Time 15 bars)
            # 2. Tracks trade outcomes explicitly.
            
            initial_capital = Decimal('50000000.00') # 5 Crore (Institutional scale)
            current_capital = initial_capital
            
            # Phase 6: Adaptive simulation barriers (must match training)
            sim_close = raw_test_df['close'] if 'close' in raw_test_df.columns else raw_test_df['Close']
            sim_vol = sim_close.pct_change().rolling(60).std().iloc[-1] if len(sim_close) > 60 else 0.012
            VOL_BASELINE = 0.012
            sim_vol_ratio = max(0.5, min(2.0, sim_vol / VOL_BASELINE)) if sim_vol > 0 else 1.0
            
            MAX_HOLD_BARS = max(3, min(10, int(7 * sim_vol_ratio)))  # 3-10 trading days (swing)
            import math
            sim_hold_scale = math.sqrt(MAX_HOLD_BARS / 7.0)  # Normalized to 7-day baseline
            
            # Barriers: 2×ATR PT, 1×ATR SL (asymmetric, swing-appropriate)
            raw_pt = 0.025 * sim_vol_ratio * sim_hold_scale
            PROFIT_TARGET_PCT = Decimal(str(round(max(0.02, raw_pt), 4)))
            
            STOP_LOSS_PCT = Decimal(str(round(0.012 * sim_vol_ratio * sim_hold_scale, 4)))
            COOLDOWN_BARS = 1
            logger.info(f"   Adaptive Sim Barriers: PT={float(PROFIT_TARGET_PCT):.2%}, SL={float(STOP_LOSS_PCT):.2%}, Hold={MAX_HOLD_BARS}d (vol_ratio={sim_vol_ratio:.2f}, hold_scale={sim_hold_scale:.2f})")
            
            txn_calculator = TransactionCostCalculator()
            

            # Initialize V2 Alpha Position Sizer
            position_sizer = VolatilityAwarePositionSizer(
                initial_capital=float(initial_capital),
                default_risk_per_trade=0.015,       # RULE 8: 1.5% max risk per trade
                max_position_size_pct=0.075,         # RULE 11: 7.5% max per position
                atr_stop_multiplier=2.0,
                max_concurrent_positions=1
            )
            
            # Cash yield: idle capital earns overnight fund returns (realistic)
            DAILY_CASH_YIELD = Decimal('0.065') / Decimal('252')  # 6.5% annualized
            
            current_position_qty = 0
            current_position_avg_price = Decimal('0.00')
            entry_bar_idx = -999
            
            # AT ENTRY values
            entry_pt_pct = Decimal('0.00')
            entry_sl_pct = Decimal('0.00')
            entry_action = None
            entry_confidence = 0.0
            
            rejection_stats = {
                'HOLD Signal': 0, 'Cooldown': 0,
                'Insufficient capital': 0, 'Emergency stop': 0
            }
            cooldown_remaining = 0
            trade_log = []
            
            test_indices = raw_test_df.index
            
            # ====================================================================
            # PRE-COMPUTE features on the FULL dataset ONCE (fixes train/inference 
            # distribution mismatch — training sees 80k bars, inference was seeing
            # 300-bar windows with wildly different rolling statistics)
            # ====================================================================
            from core.models.features import AdvancedFeatureEngine
            sim_feature_engine = AdvancedFeatureEngine()
            
            # Build feature context for daily bars (NIFTY, FII, sector)
            try:
                from core.data.market_data import MarketDataProvider
                from core.data.fii_data import FIIDataProvider
                mdp = MarketDataProvider()
                fii_provider = FIIDataProvider()
                
                feature_context = mdp.build_feature_context(
                    full_df, 
                    sector=NIFTY_50.get(symbol, {}).get('sector', '') if hasattr(NIFTY_50, 'get') else '',
                    start_date=START_DATE, 
                    end_date=END_DATE
                )
                
                # FII data: use synthetic for backtesting (real data used in live)
                fii_series = fii_provider.get_fii_flow(START_DATE, END_DATE)
                if fii_series.empty:
                    fii_series = fii_provider.generate_synthetic_fii_data(full_df.index)
                feature_context['fii_net'] = fii_series
                
                logger.info(f"   📊 Feature context: NIFTY={'yes' if 'nifty_close' in feature_context else 'no'}, "
                           f"FII={'real' if not fii_series.empty else 'synthetic'}, "
                           f"Sector={'yes' if 'sector_etf_close' in feature_context else 'no'}")
            except Exception as e:
                logger.warning(f"   ⚠️ Feature context unavailable ({e}). Using fallback features.")
                feature_context = {}
            
            precomputed_features = sim_feature_engine.engineer_all_features(full_df, context=feature_context)
            
            # Scale features using the trained scaler
            precomputed_schema = precomputed_features.reindex(
                columns=predictor._container.schema
            ).fillna(0)
            precomputed_scaled = pd.DataFrame(
                predictor._container.scaler.transform(precomputed_schema),
                index=precomputed_schema.index,
                columns=precomputed_schema.columns
            )
            logger.info(f"   ✅ Pre-computed {len(precomputed_scaled)} feature rows for simulation")
            
            # ====================================================================
            # PROBABILITY DISTRIBUTION DIAGNOSTIC
            # Quick scan of ALL test-set probabilities before the slow walk
            # ====================================================================
            model_is_overconfident = False
            container = predictor._container
            test_probs = []
            for t_idx in test_indices[::10]:  # Sample every 10th bar for speed
                if t_idx in precomputed_scaled.index:
                    try:
                        X_diag = precomputed_scaled.loc[t_idx].values.reshape(1, -1)
                        m_probs = []
                        for name, model in container.models.items():
                            w = container.weights.get(name, 1.0)
                            if w <= 0: continue
                            p = model.predict_proba(X_diag)[0]
                            m_probs.append(float(p[1]) if len(p) > 1 else float(p[0]))
                        if m_probs:
                            test_probs.append(float(np.mean(m_probs)))
                    except:
                        pass
            
            if test_probs:
                tp = np.array(test_probs)
                pct_above_52 = (tp >= 0.52).sum() / len(tp) * 100
                pct_above_50 = (tp >= 0.50).sum() / len(tp) * 100
                logger.info(f"   📊 PROB DIAGNOSTIC (sampled {len(tp)} bars):")
                logger.info(f"      Mean={tp.mean():.4f} | Std={tp.std():.4f} | Min={tp.min():.4f} | Max={tp.max():.4f}")
                logger.info(f"      P25={np.percentile(tp,25):.4f} | P50={np.percentile(tp,50):.4f} | P75={np.percentile(tp,75):.4f} | P95={np.percentile(tp,95):.4f}")
                logger.info(f"      Bars >= 0.52 hurdle: {pct_above_52:.1f}% | Bars >= 0.50: {pct_above_50:.1f}%")
                
                if pct_above_52 > 80.0:
                    logger.warning(f"   🚨 OVERCONFIDENT MODEL ({pct_above_52:.1f}% bars > hurdle). Restricting signals.")
                    model_is_overconfident = True
                    pending_signal = None
                    pending_confidence = 0.0
            
            pending_signal = None       # Signal from previous bar waiting to execute
            pending_confidence = 0.0
            pending_pred_result = None
            
            for i, current_time in enumerate(test_indices):
                window_data = full_df.loc[:current_time].tail(300)
                current_bar = window_data.iloc[-1]
                current_close = Decimal(str(current_bar['close']))
                
                # FIX: Reset variables to prevent leak from previous iteration
                should_exit = False
                exit_reason = None
                
                # === DAILY BAR: Morning gap check (RULE 9 / RULE 26) ===
                # On daily bars, check if today's open gaps against our position
                prev_close = Decimal(str(window_data['close'].iloc[-2])) if len(window_data) > 1 else current_close
                current_open = Decimal(str(current_bar['open']))
                gap_pct = float((current_open - prev_close) / prev_close) if prev_close > 0 else 0.0
                
                # No intraday time gates on daily bars
                opening_block = False  # Not applicable for daily bars
                late_session = False   # Not applicable for daily bars
                
                # ATR for emergency stop
                atr_val = Decimal('0')
                try:
                    highs = window_data['high'].values
                    lows = window_data['low'].values
                    closes = window_data['close'].values
                    if len(closes) > 14:
                        tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]))
                        tr = np.maximum(tr, np.abs(lows[1:] - closes[:-1]))
                        atr_val = Decimal(str(float(np.mean(tr[-14:]))))
                    else:
                        atr_val = current_close * Decimal('0.01')
                except Exception:
                    atr_val = current_close * Decimal('0.01')
                    
                # ====================================================================
                # PREDICTION & RISK RULES
                # ====================================================================
                is_skip_time = False
                skip_reason = ""
                
                # RULE 23: Bear Market Gate (Filter Out Long Entries If Nifty is Crashing)
                nifty_below_200_ema = False
                try:
                    if feature_context is not None and 'nifty_close' in feature_context:
                        nifty_series = feature_context['nifty_close']
                        if nifty_series is not None:
                            nifty_aligned = nifty_series.loc[:current_time]
                            if len(nifty_aligned) > 200:
                                n_close = nifty_aligned.iloc[-1]
                                n_close_20d = nifty_aligned.iloc[-21]
                                nifty_20d_return = float((n_close / n_close_20d) - 1.0)
                                nifty_200ema = nifty_aligned.ewm(span=200, adjust=False).mean().iloc[-1]
                                
                                nifty_below_200_ema = (n_close < nifty_200ema)
                                
                                if nifty_below_200_ema and nifty_20d_return < -0.05:
                                    is_skip_time = True
                                    skip_reason = "BEAR REGIME GATE"
                except Exception as e:
                    pass

                if model_is_overconfident:
                    pred_result = {
                        'status': 'success',
                        'signal': 'HOLD',
                        'confidence': 0.50,
                        'probs': [0.50, 0.50]
                    }
                    predicted_probability = 0.50
                    signal_txt = "HOLD (Model Overconfidence Reject)"
                    pending_signal = None
                    pending_confidence = 0.0
                elif not is_skip_time:
                    try:
                        # For Paper Walk testing ONLY, pass precomputed features to prevent huge slowdown
                        if precomputed_scaled is not None and current_time in precomputed_scaled.index:
                            X_current = precomputed_scaled.loc[current_time].values.reshape(1,-1)
                            
                            m_probs = []
                            for name, model in container.models.items():
                                w = container.weights.get(name, 1.0)
                                if w <= 0: continue
                                probs = model.predict_proba(X_current)[0]
                                m_probs.append(float(probs[1]) if len(probs) > 1 else float(probs[0]))
                                if float(current_close) > 0:
                                    current_atr_pct = float(atr_val) / float(current_close)
                                    pt_pct = Decimal(str(round(current_atr_pct * 2.0, 4)))
                                    sl_pct = Decimal(str(round(current_atr_pct * 1.0, 4)))
                                    PROFIT_TARGET_PCT = pt_pct
                                    STOP_LOSS_PCT = sl_pct
                            
                            if m_probs:
                                confidence = float(np.mean(m_probs))
                                if confidence >= 0.52:
                                    pred_result = {'signal': 'BUY', 'confidence': confidence}
                                else:
                                    # Fallback message matching MARK5Predictor.predict standard
                                    pred_result = {'signal': f'HOLD (Conf {confidence:.0%} < 52% hurdle)', 'confidence': confidence}
                            else:
                                pred_result = {'signal': 'HOLD', 'confidence': 0.0}
                        else:
                            # Generate prediction dynamically
                            pred_result = predictor.predict(window_data)
                    except Exception as e:
                        logger.error(f"Prediction failed at {current_time}: {e}")
                        pred_result = {'signal': 'HOLD (Error)'}
                else:
                    pred_result = {'signal': f'HOLD ({skip_reason})', 'confidence': 0.0}
                signal_txt = pred_result.get('signal', 'HOLD')
                predicted_probability = float(pred_result.get('confidence', 0.0))
                if float(current_close) > 0:
                    current_atr_pct = float(atr_val) / float(current_close)
            
                    # Base PT on 2.5x ATR, SL on 1.5x ATR to satisfy minimum R:R >= 1.5 rule
                    pt_pct = Decimal(str(round(current_atr_pct * 2.5, 4)))
                    sl_pct = Decimal(str(round(current_atr_pct * 1.5, 4)))
            
                    PROFIT_TARGET_PCT = pt_pct
                    STOP_LOSS_PCT = sl_pct
        
                # Continuous Learning & Drift Detection
                if i > 0 and i % 20 == 0 and hasattr(self, 'learning_engine') and self.learning_engine:
                    try:
                        override_df = pd.DataFrame(trade_log) if trade_log else None
                        cycle = self.learning_engine.run_learning_cycle(symbol, override_df=override_df)
                        if cycle and cycle.get('action') == 'HALT_TRADING':
                            logger.warning(f"🚨 LearningEngine Drift Detected | HALTING {symbol} | Reason: {cycle.get('reason')}")
                            break  # Terminal drift: abort walk for this symbol
                    except Exception as e:
                        pass
        
                # RE-EVALUATE PRECOMPUTED FEATURES (only if not skipping or overconfident)
                if not is_skip_time and not model_is_overconfident:
                    if current_time in precomputed_scaled.index:
                        try:
                            X_current = precomputed_scaled.loc[current_time].values.reshape(1, -1)
                            container = predictor._container
                    
                            model_probs = []
                            for name, model in container.models.items():
                                w = container.weights.get(name, 1.0)
                                if w <= 0: continue
                                try:
                                    probs = model.predict_proba(X_current)[0]
                                    prob_buy = float(probs[1]) if len(probs) > 1 else float(probs[0])
                                    model_probs.append(prob_buy)
                                except Exception:
                                    continue
                    
                            if model_probs:
                                arith_mean = float(np.mean(model_probs))
                                PROBABILITY_HURDLE = 0.52
                                signal = "BUY" if arith_mean >= PROBABILITY_HURDLE else "HOLD"
                                if signal == "HOLD" and arith_mean >= 0.50:
                                    signal = f"HOLD (Conf {arith_mean:.0%} < {PROBABILITY_HURDLE:.0%} hurdle)"
                                pred_result = {
                                    'status': 'success',
                                    'signal': signal,
                                    'confidence': arith_mean,
                                    'probs': [1.0 - arith_mean, arith_mean]
                                }
                        except Exception as e:
                            pass  # Keep default HOLD
                
                signal_txt = str(pred_result.get('signal', 'HOLD'))
                confidence = float(pred_result.get('confidence', 0.0))
                predicted_probability = float(pred_result.get('confidence', 0.0))
                
                action = "HOLD"
                trade_pnl = Decimal('0.00')
                txn_cost = Decimal('0.00')
                txn_qty = 0
                txn_price = current_close
                slippage = current_close * Decimal('0.0005')  # 0.05% per RULE 8
                
                should_exit = False
                exit_reason = None
                unrealized_pnl_pct = 0.0
                
                # --- EXIT LOGIC (check FIRST) ---
                if current_position_qty > 0:  # LONG EXIT
                    bars_held = i - entry_bar_idx
                    
                    # RULE 9: Adverse gap check — if gap > 3% against LONG position, exit at open
                    if gap_pct < -0.03:
                        should_exit = True
                        exit_reason = "GAP_ADVERSE_EXIT"
                        sell_price = current_open  # Exit at today's open price
                    
                    # Triple-Barrier Checks (only if not already forced closed)
                    if not should_exit:
                        current_high = Decimal(str(current_bar['high']))
                        current_low = Decimal(str(current_bar['low']))
                        
                        # Use entry PT and SL frozen at the time of trade entry
                        profit_target_price = current_position_avg_price * (Decimal('1') + entry_pt_pct)
                        stop_loss_price = current_position_avg_price * (Decimal('1') - entry_sl_pct)
                        
                        # 1. Stop Loss Barrier
                        if current_low <= stop_loss_price:
                            should_exit = True
                            exit_reason = "STOP_LOSS_HIT"
                            max_loss_price = current_position_avg_price * (Decimal('1') - entry_sl_pct * Decimal('2'))
                            raw_sell = min(Decimal(str(current_bar['open'])), stop_loss_price)
                            sell_price = max(raw_sell, max_loss_price) - slippage
                        
                        # 2. Profit Target Barrier
                        elif current_high >= profit_target_price:
                            should_exit = True
                            exit_reason = "PROFIT_TARGET_HIT"
                            sell_price = max(Decimal(str(current_bar['open'])), profit_target_price) - slippage
                            
                        # 3. Time Barrier
                        elif bars_held >= MAX_HOLD_BARS:
                            should_exit = True
                            exit_reason = "TIME_BARRIER_HIT"
                            sell_price = current_close - slippage
                        
                        # 4. Emergency Volatility Stop
                        elif atr_val > 0 and current_position_avg_price > 0:
                            emergency_pct = (atr_val * Decimal('3.0')) / current_position_avg_price
                            loss_pct = (current_position_avg_price - current_close) / current_position_avg_price
                            if loss_pct > emergency_pct:
                                should_exit = True
                                exit_reason = "EMERGENCY_STOP"
                                sell_price = current_close - slippage
                                rejection_stats['Emergency stop'] += 1
                    
                    if should_exit:
                        cost_dict = txn_calculator.calculate_sell_costs(
                            sell_price * Decimal(current_position_qty), current_position_qty
                        )
                        total_cost = Decimal(str(cost_dict['total_cost']))
                        
                        gross_pnl = (sell_price - current_position_avg_price) * Decimal(current_position_qty)
                        net_pnl = gross_pnl - total_cost
                        
                        current_capital += (sell_price * Decimal(current_position_qty)) - total_cost
                        
                        unrealized_pnl_pct = float((sell_price - current_position_avg_price) / current_position_avg_price) * 100
                        
                        trade_pnl = net_pnl
                        txn_cost = total_cost
                        action = exit_reason
                        txn_qty = current_position_qty
                        txn_price = sell_price
                        
                        # LONG exit trade_log.append (around line 1001):
                        trade_log.append({
                            'symbol': symbol,
                            'action': entry_action,
                            'confidence': float(entry_confidence),
                            'entry_time': test_indices[entry_bar_idx] if entry_bar_idx >= 0 else current_time,
                            'exit_time': current_time,          # ADD THIS
                            'exit_reason': exit_reason,
                            'bars_held': bars_held,
                            'pnl': float(net_pnl),
                            'pnl_pct': unrealized_pnl_pct,
                            'entry_price': float(current_position_avg_price),
                            'exit_price': float(sell_price),
                            'qty': current_position_qty
                        })
                        
                        position_sizer.record_trade_result(float(net_pnl))
                        position_sizer.close_position(symbol)
                        
                        current_position_qty = 0
                        current_position_avg_price = Decimal('0.00')
                        cooldown_remaining = COOLDOWN_BARS
                
                # BUG-05 FIX: SHORT position exit logic (inverted barriers)
                elif current_position_qty < 0:
                    bars_held = i - entry_bar_idx
                    abs_qty = abs(current_position_qty)
                    
                    should_exit = False
                    exit_reason = ""
                    
                    # RULE 9: Adverse gap check — if gap > 3% against SHORT position, exit at open
                    if gap_pct > 0.03:
                        should_exit = True
                        exit_reason = "GAP_ADVERSE_EXIT"
                        cover_price = current_open  # Exit at today's open price
                    
                    # Triple-Barrier Checks (only if not already forced closed)
                    if not should_exit:
                        current_high = Decimal(str(current_bar['high']))
                        current_low = Decimal(str(current_bar['low']))
                        
                        # SHORT barriers are INVERTED:
                        # SL hit when price RISES (high >= entry * (1 + sl_pct))
                        # PT hit when price DROPS (low <= entry * (1 - pt_pct))
                        stop_loss_price = current_position_avg_price * (Decimal('1') + entry_sl_pct)
                        profit_target_price = current_position_avg_price * (Decimal('1') - entry_pt_pct)
                        
                        # 1. Stop Loss (price rose against short)
                        if current_high >= stop_loss_price:
                            should_exit = True
                            exit_reason = "SHORT_STOP_LOSS_HIT"
                            max_loss_price = current_position_avg_price * (Decimal('1') + entry_sl_pct * Decimal('2'))
                            raw_cover = max(Decimal(str(current_bar['open'])), stop_loss_price)
                            cover_price = min(raw_cover, max_loss_price) + slippage
                        
                        # 2. Profit Target (price dropped in our favor)
                        elif current_low <= profit_target_price:
                            should_exit = True
                            exit_reason = "SHORT_PROFIT_TARGET_HIT"
                            cover_price = min(Decimal(str(current_bar['open'])), profit_target_price) + slippage
                        
                        # 3. Time Barrier
                        elif bars_held >= MAX_HOLD_BARS:
                            should_exit = True
                            exit_reason = "SHORT_TIME_BARRIER_HIT"
                            cover_price = current_close + slippage
                        
                        # 4. Emergency Stop
                        elif atr_val > 0 and current_position_avg_price > 0:
                            emergency_pct = (atr_val * Decimal('3.0')) / current_position_avg_price
                            loss_pct = (current_close - current_position_avg_price) / current_position_avg_price
                            if loss_pct > emergency_pct:
                                should_exit = True
                                exit_reason = "SHORT_EMERGENCY_STOP"
                                cover_price = current_close + slippage
                                rejection_stats['Emergency stop'] += 1
                    
                    if should_exit:
                        # Cover cost: buying back shares
                        cost_dict = txn_calculator.calculate_buy_costs(
                            cover_price * Decimal(abs_qty), abs_qty
                        )
                        total_cost = Decimal(str(cost_dict['total_cost']))
                        
                        # SHORT PnL: (entry - cover) × qty
                        gross_pnl = (current_position_avg_price - cover_price) * Decimal(abs_qty)
                        net_pnl = gross_pnl - total_cost
                        
                        # Return capital: entry proceeds + PnL - costs
                        current_capital += (current_position_avg_price * Decimal(abs_qty)) + net_pnl - total_cost
                        
                        unrealized_pnl_pct = float((current_position_avg_price - cover_price) / current_position_avg_price) * 100
                        
                        trade_pnl = net_pnl
                        txn_cost = total_cost
                        action = exit_reason
                        txn_qty = abs_qty
                        txn_price = cover_price
                        
                        trade_log.append({
                            'symbol': symbol,
                            'action': entry_action,
                            'confidence': float(entry_confidence),
                            'entry_time': test_indices[entry_bar_idx] if entry_bar_idx >= 0 else current_time,
                            'exit_reason': exit_reason,
                            'bars_held': bars_held,
                            'pnl': float(net_pnl),
                            'pnl_pct': unrealized_pnl_pct,
                            'entry_price': float(current_position_avg_price),
                            'exit_price': float(cover_price),
                            'qty': abs_qty
                        })
                        
                        position_sizer.record_trade_result(float(net_pnl))
                        position_sizer.close_position(symbol)
                        
                        current_position_qty = 0
                        current_position_avg_price = Decimal('0.00')
                        cooldown_remaining = COOLDOWN_BARS
                        
                        # --- RULE 30: Evaluate immediately after a trade closes ---
                        completed = [t for t in trade_log 
                                     if t.get('exit_reason') in 
                                     ('PROFIT_TARGET_HIT','STOP_LOSS_HIT',
                                      'TIME_BARRIER_HIT','GAP_ADVERSE_EXIT')]
                        recent_completed = completed[-10:] if len(completed) >= 10 else []
                        if len(recent_completed) >= 10:
                            live_wr = sum(1 for t in recent_completed if t.get('pnl_pct', 0) > 0) / len(recent_completed)
                            live_pf = (sum(t['pnl_pct'] for t in recent_completed if t.get('pnl_pct',0) > 0) /
                                       max(abs(sum(t['pnl_pct'] for t in recent_completed if t.get('pnl_pct',0) < 0)), 0.001))
                            
                            if live_wr < 0.35 and live_pf < 0.80:
                                logger.warning(f"🚨 Rule 30 Halt Triggered for {symbol} | WR: {live_wr:.2f}, PF: {live_pf:.2f}. Pausing for 20 bars.")
                                cooldown_remaining = 20
                
                # --- VALIDATION GATE SYNCHRONIZATION (Mirrors decision.py) ---
                # FIX: Predict result actually contains `features` key from AdvancedFeatureEngine. 
                # Weekly trend and RSI are NOT in raw DataFrames!
                # Compute weekly trend from daily data (resample)
                weekly_trend = None
                weekly_rsi = None
                try:
                    weekly_close = full_df.loc[:current_time]['close'].resample('W').last().dropna()
                    if len(weekly_close) >= 14:
                        # Weekly trend: price above 10-week EMA = bullish
                        w_ema10 = weekly_close.ewm(span=10).mean()
                        weekly_trend = 1 if weekly_close.iloc[-1] > w_ema10.iloc[-1] else 0
                        # Weekly RSI
                        w_delta = weekly_close.diff()
                        w_gain = w_delta.clip(lower=0).rolling(14).mean()
                        w_loss = (-w_delta.clip(upper=0)).rolling(14).mean()
                        w_rs = w_gain / (w_loss + 1e-9)
                        weekly_rsi = float(100 - (100 / (1 + w_rs.iloc[-1])))
                except Exception:
                    pass
                # Check Weekly Gates for validation simulation
                if "BUY" not in signal_txt and current_position_qty == 0:
                    if (weekly_trend is not None and int(weekly_trend) == 0 and
                        confidence < 0.48 and
                        nifty_below_200_ema):
                        
                        # Only enter SHORT if we've had low confidence for 2+ consecutive bars
                        prev_conf = float(results[-1]['confidence']) if results else confidence
                        if prev_conf < 0.48:
                            signal_txt = "SHORT"
                
                if "BUY" in signal_txt:
                    veto = False

                    # Only veto on weekly trend if CLEARLY bearish: both below EMA AND RSI < 45
                    if weekly_trend is not None and int(weekly_trend) == 0:
                        if weekly_rsi is not None and float(weekly_rsi) < 45:
                            logger.info(f"🚫 WEEKLY TREND VETO | {symbol} | weekly_rsi={weekly_rsi:.1f} < 45 AND below EMA")
                            veto = True

                    # RSI overbought veto: only at extreme overbought (>80) AND regime is explicitly bearish
                    # Don't fire when regime column is missing (defaults to RANGING)
                    if weekly_rsi is not None and float(weekly_rsi) > 80:
                        regime = str(df['regime'].iloc[i]).upper() if 'regime' in df.columns else ''
                        if regime in ('BEAR', 'BEAR_MARKET'):
                            logger.info(f"🚫 WEEKLY RSI VETO | {symbol} | weekly_rsi={weekly_rsi:.1f} > 80 in BEAR regime")
                            veto = True

                    if float(PROFIT_TARGET_PCT) / float(STOP_LOSS_PCT) < 1.5:
                        veto = True

                    if veto:
                        signal_txt = "HOLD"
                        action = "HOLD"
                        pred_result['reason'] = 'Weekly Veto / RR Rejected'

                elif "SHORT" in signal_txt:
                    veto = False
                    if weekly_trend is not None and int(weekly_trend) == 1:
                        logger.info(f"🚫 WEEKLY TREND VETO | {symbol} | SHORT blocked | weekly_trend=BULLISH | weekly_rsi={weekly_rsi:.1f} | daily_conf={confidence*100:.1f}%")
                        veto = True
                    if weekly_rsi is not None and float(weekly_rsi) < 30:
                        trend_str = 'BULL' if (weekly_trend is not None and int(weekly_trend)==1) else 'BEAR'
                        logger.info(f"🚫 WEEKLY RSI VETO | {symbol} | SHORT blocked | weekly_rsi={weekly_rsi:.1f} < 30 | weekly_trend={trend_str}")
                        veto = True
                    if float(PROFIT_TARGET_PCT) / float(STOP_LOSS_PCT) < 1.5:
                        logger.info(f"❌ RR REJECTED: {symbol} R:R={(float(PROFIT_TARGET_PCT)/float(STOP_LOSS_PCT)):.2f} < 1.5")
                        veto = True
                        
                    if veto:
                        signal_txt = "HOLD"
                        action = "HOLD"
                        pred_result['reason'] = 'Weekly Veto / RR Rejected'
                        
                # Rule 30 gate moved to post-trade evaluation (sets cooldown_remaining)
                
                
                # --- ENTRY LOGIC (RULE 9, 11, 21 ENFORCEMENT) ---
                # Execute PREVIOUS bar's signal at TODAY's open (RULE 40)
                if pending_signal == "BUY" and current_position_qty == 0 \
                        and cooldown_remaining <= 0:
                    equity = current_capital
                    buy_price = current_open + slippage  # Next-day open
                    confidence = pending_confidence
                    pred_result = pending_pred_result
                    
                    # Daily bars: no late-session discount
                    effective_confidence = confidence
                    
                    # Update Sizer's view of equity
                    position_sizer.update_capital(float(equity))
                    
                    # Compute size via V2 Sizer
                    atr_float = float(atr_val) if float(atr_val) > 0 else float(buy_price * Decimal('0.02'))
                    
                    qty_int, size_details = position_sizer.calculate_size(
                        symbol=symbol,
                        price=float(buy_price),
                        atr=atr_float,
                        conviction=effective_confidence
                    )
                    
                    qty = qty_int
                    
                    if qty > 0:
                        cost_dict = txn_calculator.calculate_buy_costs(buy_price * Decimal(qty), qty)
                        total_cost = Decimal(str(cost_dict['total_cost']))
                        required_cash = (buy_price * Decimal(qty)) + total_cost
                        
                        if required_cash <= current_capital:
                            current_capital -= required_cash
                            current_position_qty = qty
                            current_position_avg_price = buy_price
                            entry_bar_idx = i
                            txn_cost = total_cost
                            action = "BUY"
                            txn_qty = qty
                            txn_price = buy_price
                            
                            # Register with Sizer
                            position_sizer.register_position(symbol, float(required_cash))
                            
                            # Freeze PT/SL dynamically computed at this moment
                            entry_pt_pct = PROFIT_TARGET_PCT
                            entry_sl_pct = STOP_LOSS_PCT
                            entry_action = action
                            entry_confidence = confidence
                            
                        else:
                            rejection_stats['Insufficient capital'] += 1
                    else:
                        rejection_stats['Insufficient capital'] += 1
                
                # BUG-05 FIX: SHORT trade entry logic
                elif "SHORT" in signal_txt and current_position_qty == 0 and cooldown_remaining <= 0 and not opening_block:
                    equity = current_capital
                    short_price = current_close - slippage  # Sell at lower price (adverse slippage)
                    
                    # Daily bars: no late-session discount
                    effective_confidence = confidence
                    
                    position_sizer.update_capital(float(equity))
                    
                    atr_float = float(atr_val) if float(atr_val) > 0 else float(short_price * Decimal('0.02'))
                    
                    qty_int, size_details = position_sizer.calculate_size(
                        symbol=symbol,
                        price=float(short_price),
                        atr=atr_float,
                        conviction=effective_confidence
                    )
                    
                    qty = qty_int
                    
                    if qty > 0:
                        # SHORT: We receive cash from selling, hold margin
                        cost_dict = txn_calculator.calculate_sell_costs(short_price * Decimal(qty), qty)
                        total_cost = Decimal(str(cost_dict['total_cost']))
                        # Margin requirement: hold full value as collateral
                        margin_required = short_price * Decimal(qty)
                        
                        if margin_required + total_cost <= current_capital:
                            current_capital -= total_cost  # Pay entry costs only
                            current_position_qty = -qty  # NEGATIVE = SHORT
                            current_position_avg_price = short_price
                            entry_bar_idx = i
                            txn_cost = total_cost
                            action = "SHORT"
                            txn_qty = qty
                            txn_price = short_price
                            
                            position_sizer.register_position(symbol, float(margin_required))
                            
                            # SHORT barriers are inverted: PT when price drops, SL when price rises
                            entry_pt_pct = PROFIT_TARGET_PCT
                            entry_sl_pct = STOP_LOSS_PCT
                            entry_action = action
                            entry_confidence = confidence
                        else:
                            rejection_stats['Insufficient capital'] += 1
                    else:
                        rejection_stats['Insufficient capital'] += 1
                        
                else:
                    if cooldown_remaining > 0:
                        cooldown_remaining -= 1
                        if "BUY" in signal_txt or "SHORT" in signal_txt:
                            rejection_stats['Cooldown'] += 1
                        else:
                            reason = pred_result.get('reason', 'HOLD Signal')
                            rejection_stats[reason] = rejection_stats.get(reason, 0) + 1
                    elif "BUY" not in signal_txt and "SHORT" not in signal_txt:
                        reason = pred_result.get('reason', 'HOLD Signal')
                        if reason == 'HOLD Signal' and i % 50 == 0:
                            logger.info(f"🔍 DEBUG NULL REASON | signal_txt: {signal_txt} | payload: {pred_result}")
                        rejection_stats[reason] = rejection_stats.get(reason, 0) + 1
                
                # Buffer signal for next-bar execution
                if "BUY" in signal_txt and current_position_qty == 0:
                    pending_signal = "BUY"
                    pending_confidence = confidence
                    pending_pred_result = pred_result
                else:
                    pending_signal = None
                    pending_confidence = 0.0
                    pending_pred_result = None
                
                # Cash yield on idle capital (overnight fund returns)
                # FIX: Removed passive cash compounding to prevent ghost returns in metrics
                # if current_position_qty == 0:
                #     cash_yield = current_capital * DAILY_CASH_YIELD
                #     current_capital += cash_yield


                # Log Record (includes predicted_return for IC calculation)
                record = {
                    'timestamp': current_time,
                    'close': float(current_close),
                    'predicted_signal': 1 if "BUY" in signal_txt else (-1 if "SELL" in signal_txt else 0),
                    'predicted_probability': float(predicted_probability),
                    'confidence': float(confidence),
                    'signals_txt': signal_txt,
                    'action': action,
                    'position_qty': current_position_qty,
                    'capital': float(current_capital) + (float(current_close) * current_position_qty if current_position_qty > 0 else 0),
                    'trade_pnl': float(trade_pnl),
                    'txn_cost': float(txn_cost),
                    'dataset_type': 'Test'
                }
                
                # Attach exit reason directly if this log was an exit (simplifies metric analysis)
                if 'should_exit' in locals() and should_exit:
                    record['exit_reason'] = exit_reason
                    record['bars_held'] = i - entry_bar_idx  # ADD THIS
                    record['trade_pnl_pct'] = float(unrealized_pnl_pct) / 100.0 if 'unrealized_pnl_pct' in locals() else 0.0
                else:
                    record['exit_reason'] = None
                    record['trade_pnl_pct'] = 0.0
                
                results.append(record)
                
                if (i + 1) % log_interval == 0:
                    prog = (i + 1) / len(raw_test_df)
                    # Show EQUITY (cash + unrealized position value), not just cash
                    equity_now = current_capital + (current_close * Decimal(current_position_qty) if current_position_qty > 0 else Decimal('0'))
                    logger.info(f"   Walk Progress: {prog:.0%} ({i+1}/{len(raw_test_df)}) | Equity: ₹{equity_now:,.2f}")
            
            # Create Results DataFrame
            res_df = pd.DataFrame(results)
            if not res_df.empty:
                res_df.set_index('timestamp', inplace=True)
                # Strategy Returns from Equity Curve (correct method)
                res_df['equity'] = res_df['capital']
                
                res_df['strategy_return_net'] = res_df['equity'].pct_change().fillna(0)
                
                # Direction Correct: Only count when model made a trade and we exited it.
                # A trade is "correct" (True Positive) if it hits the profit target.
                res_df['direction_correct'] = False
                exit_mask = res_df['exit_reason'].notna()
                res_df.loc[exit_mask, 'direction_correct'] = (res_df.loc[exit_mask, 'exit_reason'] == 'PROFIT_TARGET_HIT')
            
            # Final equity calculation
            final_equity = current_capital
            if current_position_qty > 0:
                final_close = Decimal(str(full_df['close'].iloc[-1]))
                final_equity += final_close * Decimal(current_position_qty)
            
            logger.info(f"✅ Paper Walk Complete. Final Equity: ₹{final_equity:,.2f} (Cash: ₹{current_capital:,.2f})")
            logger.info(f"📊 Rejection Stats: {json.dumps(rejection_stats, indent=2)}")
            
            # Trade-level diagnostics
            if trade_log:
                wins = [t for t in trade_log if t['pnl'] > 0]
                losses = [t for t in trade_log if t['pnl'] <= 0]
                avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
                avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
                logger.info(f"📊 Trade Summary: {len(trade_log)} trades | {len(wins)}W/{len(losses)}L | AvgWin: ₹{avg_win:,.0f} | AvgLoss: ₹{avg_loss:,.0f}")
                for t in trade_log[:5]:
                    logger.info(f"   Trade: {t['exit_reason']} | {t['bars_held']}bars | PnL: ₹{t['pnl']:,.0f} ({t['pnl_pct']:.2f}%)")
            
            res_df = self._ensure_required_columns(res_df)
            return res_df, trade_log
            
        except Exception as e:
            logger.error(f"❌ MARK5 pipeline failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIX: Ensure all required columns exist for visualization
        This prevents KeyError crashes in the plotting functions
        """
        df = df.copy()
        
        # Ensure true_return_5d exists
        if 'true_return_5d' not in df.columns:
            if 'close' in df.columns:
                df['true_return_5d'] = df['close'].pct_change(5).shift(-5)
            else:
                df['true_return_5d'] = 0.0
        
        # Ensure true_return_1d exists
        if 'true_return_1d' not in df.columns:
            if 'close' in df.columns:
                df['true_return_1d'] = df['close'].pct_change(1).shift(-1)
            else:
                df['true_return_1d'] = 0.0
        
        # CRITICAL: Ensure direction_correct exists
        if 'direction_correct' not in df.columns:
            if 'predicted_probability' in df.columns and 'exit_reason' in df.columns:
                df['direction_correct'] = df['exit_reason'] == 'PROFIT_TARGET_HIT'
            else:
                # Fallback: create dummy column to prevent crash
                logger.warning(f"⚠️  Could not calculate direction_correct - creating dummy column")
                df['direction_correct'] = False
        
        # Ensure basic prediction columns exist
        if 'predicted_signal' not in df.columns:
            df['predicted_signal'] = 0
        
        if 'confidence' not in df.columns:
            df['confidence'] = 0.5
        
        if 'predicted_probability' not in df.columns:
            df['predicted_probability'] = 0.5
        
        # Ensure action exists
        if 'action' not in df.columns:
            df['action'] = "HOLD"

        # Ensure market_return exists for performance calc
        if 'market_return' not in df.columns:
            if 'close' in df.columns:
                df['market_return'] = df['close'].pct_change().fillna(0)
                # If market return is massive outlier, clip it (e.g. split)
                df['market_return'] = df['market_return'].clip(-0.5, 0.5)
            else:
                df['market_return'] = 0.0
        
        return df

    def _use_mark5_prediction_engine(self, symbol: str, df: pd.DataFrame, 
                                     models: Dict) -> pd.DataFrame:
        """Use MARK5's trained models to generate predictions"""
        
        # Prepare feature matrix
        exclude_cols = ['timestamp', 'regime', 'predicted_signal', 
                       'confidence', 'predicted_return', 'direction_correct',
                       'true_signal', 'true_return_1d', 'true_return_5d',
                       'position', 'position_change', 'trade_occurred', 
                       'transaction_cost', 'strategy_return_gross', 'strategy_return_net',
                       'market_return']
        
        feature_cols = [c for c in df.columns 
                       if c not in exclude_cols and 
                       pd.api.types.is_numeric_dtype(df[c])]
        
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Get predictions from MARK5 ensemble
        logger.info(f"   Using MARK5 models for ensemble prediction...")
        predictions = self._batch_predict_ensemble(symbol, X, models)
        
        # Create result DataFrame
        result_df = df.copy()
        result_df['predicted_signal'] = predictions['signal']
        result_df['confidence'] = predictions['confidence']
        result_df['predicted_return'] = predictions['predicted_return']
        
        # Calculate TRUE labels for validation
        result_df['true_return_1d'] = result_df['close'].pct_change(1).shift(-1)
        result_df['true_return_5d'] = result_df['close'].pct_change(5).shift(-5)
        
        # Calculate if prediction direction was correct
        result_df['direction_correct'] = (
            np.sign(result_df['true_return_5d']) == np.sign(result_df['predicted_return'])
        )
        
        # Filter for valid evaluation rows
        result_df = result_df.dropna(subset=['true_return_5d'])
        
        logger.info(f"   Generated {len(result_df)} predictions with validation labels")
        
        return result_df

    def _batch_predict_ensemble(self, symbol: str, features: pd.DataFrame, models_metadata: Dict) -> Dict[str, np.ndarray]:
        """Batch prediction using loaded regression models.
        
        v4.0: Uses .predict() (returns float) instead of .predict_proba().
        Models are regressors that predict continuous forward returns.
        """
        if not models_metadata:
            return {
                'signal': np.zeros(len(features)), 
                'confidence': np.zeros(len(features)), 
                'predicted_return': np.zeros(len(features))
            }
            
        all_preds = []
        valid_models = 0
        COST_HURDLE = 0.0015  # 0.15% minimum predicted return to trade
        
        for mid, meta in models_metadata.items():
            try:
                if hasattr(meta, 'predict'):
                     model = meta
                elif isinstance(meta, dict) and 'path' in meta:
                    path = meta['path']
                    if not os.path.exists(path):
                        continue
                    model = joblib.load(path)
                else:
                    model = meta 

                if hasattr(model, "predict"):
                    preds = model.predict(features)
                    all_preds.append(preds)
                    valid_models += 1
            except Exception:
                continue
                
        if valid_models == 0:
            logger.warning("No valid models for batch prediction")
            return {
                'signal': np.zeros(len(features)), 
                'confidence': np.zeros(len(features)), 
                'predicted_return': np.zeros(len(features))
            }
            
        # IC-weighted averaging (equal weights as fallback)
        predicted_return = np.mean(all_preds, axis=0)
        
        # Derive signals from predicted return vs cost hurdle
        signals = np.where(predicted_return > COST_HURDLE, 1,
                          np.where(predicted_return < -COST_HURDLE, -1, 0))
        
        # Confidence from prediction magnitude and model agreement
        pred_std = np.std(all_preds, axis=0) if valid_models > 1 else np.zeros(len(features))
        agreement = 1.0 / (1.0 + pred_std * 100)  # Higher agreement = higher confidence
        magnitude = np.clip(np.abs(predicted_return) / 0.02, 0, 1)  # 2% = max confidence
        confidence = 0.5 * agreement + 0.5 * magnitude
        
        return {
            'signal': signals,
            'confidence': confidence,
            'predicted_return': predicted_return
        }

    def _standalone_prediction_fallback(self, symbol: str, df_features: pd.DataFrame) -> pd.DataFrame:
        """Simple fallback if MARK5 core is missing"""
        # Split data
        split_idx = int(len(df_features) * 0.8)
        df_train = df_features.iloc[:split_idx].copy()
        df_test = df_features.iloc[split_idx:].copy()
        
        # Create labels manually
        df_train['true_return_5d'] = df_train['close'].pct_change(5).shift(-5)
        conditions = [
            (df_train['true_return_5d'] > 0.03),
            (df_train['true_return_5d'] < -0.03)
        ]
        choices = [1, -1]
        df_train['true_signal'] = np.select(conditions, choices, default=0)
        df_train = df_train.dropna(subset=['true_return_5d'])
        
        # Train simple model
        from sklearn.ensemble import RandomForestClassifier
        exclude = ['true_return_5d', 'true_signal', 'timestamp', 'regime']
        cols = [c for c in df_train.columns if c not in exclude and pd.api.types.is_numeric_dtype(df_train[c])]
        
        X_train = df_train[cols].fillna(0)
        y_train = df_train['true_signal'].astype(int)
        
        if len(X_train) > 0:
            model = RandomForestClassifier(n_estimators=50, max_depth=5)
            model.fit(X_train, y_train)
            
            X_test = df_test[cols].fillna(0)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test).max(axis=1)
            
            df_test['predicted_signal'] = preds
            df_test['confidence'] = proba
            df_test['predicted_return'] = preds * 0.01 * proba
        
        return df_test

    def _detect_regime_fallback(self, df: pd.DataFrame) -> np.ndarray:
        """Detect market regime"""
        volatility = df['close'].pct_change().rolling(20).std()
        limit = volatility.quantile(0.75)
        vol_high = volatility > limit
        
        if 'close' in df.columns:
            sma20 = df['close'].rolling(20).mean()
            trend = (sma20 - sma20.shift(20)) / sma20.shift(20)
            trending = trend.abs() > 0.05
        else:
            trending = pd.Series(False, index=df.index)
        
        conditions = [
            trending & ~vol_high,
            vol_high,
        ]
        choices = ['Trending', 'Volatile']
        regime = np.select(conditions, choices, default='Ranging')
        
        return regime

    def log_validation_results(self, symbol: str, df: pd.DataFrame, metrics: Dict):
        """Log validation results to MARK5's Trade Journal"""
        if not self.trade_journal:
            return
        
        trades = df[df['predicted_signal'] != 0].copy()
        logger.info(f"📝 Would log {len(trades)} trades to journal (Simulated)")

# ═══════════════════════════════════════════════════════════════════
# STEP 3: ADVANCED ANALYTICS & METRICS
# ═══════════════════════════════════════════════════════════════════

class PerformanceAnalyzer:
    """Calculate advanced trading metrics"""
    
    @staticmethod
    def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        risk_analyzer = PortfolioRiskAnalyzer(
            initial_capital=100000.0,
            max_position_size=1.0,
            max_daily_loss=100000.0,
            max_drawdown=1.0
        )
        
        df = df.copy()
        
        if 'predicted_signal' in df.columns:
            df['position'] = df['predicted_signal'].clip(-1, 1)
        else:
            df['position'] = 0
            
        df['position_change'] = df['position'].diff().fillna(df['position'])
        df['trade_occurred'] = df['position_change'] != 0
        
        def calculate_transaction_cost(row):
            if not row['trade_occurred']:
                return 0.0
            base_cost_pct = 0.0015
            cost_multiplier = abs(row['position_change'])
            return base_cost_pct * cost_multiplier
        
        if 'strategy_return_net' not in df.columns:
            # Fallback if log doesn't have PnL tracked (standalone mode)
            if 'predicted_return' in df.columns:
                 df['strategy_return_net'] = df['predicted_return']
            else:
                 df['strategy_return_net'] = 0.0
        
        # Calculate Equity Curve
        if 'equity' in df.columns:
            # Use tracked equity
            equity_curve = df['equity'].values
            # Calculate returns from equity
            df['strategy_return_net'] = df['equity'].pct_change().fillna(0)
            cum_strategy = (df['equity'] / df['equity'].iloc[0]) - 1
        else:
            # Simulation from returns
            cum_strategy = (1 + df['strategy_return_net']).cumprod() - 1
            initial_cap = 100000.0
            equity_series = initial_cap * (1 + cum_strategy)
            equity_curve = equity_series.values

        if 'market_return' not in df.columns:
            df['market_return'] = df['close'].pct_change().fillna(0)
        
        cum_market = (1 + df['market_return']).cumprod() - 1
        
        metrics['total_return_strategy'] = cum_strategy.iloc[-1] if len(cum_strategy) > 0 else 0
        metrics['total_return_market'] = cum_market.iloc[-1] if len(cum_market) > 0 else 0
        metrics['excess_return'] = metrics['total_return_strategy'] - metrics['total_return_market']
        
        # Direction accuracy (regression: does predicted sign match actual sign?)
        # CRITICAL FIX: Only measure accuracy on active predictions (BUY/SELL),
        # CRITICAL FIX: Measure PT hit rate explicitly based on exiting action strings
        if 'action' in df.columns:
            # FIX: Only measure accuracy against actual resolved trades
            pt_exits = df['action'] == 'PROFIT_TARGET_HIT'
            sl_exits = df['action'] == 'STOP_LOSS_HIT'
            all_exits = pt_exits | sl_exits
            
            if all_exits.sum() > 0:
                metrics['accuracy'] = float(pt_exits.sum() / all_exits.sum())
                # For backward compatibility / log expectations, we leave weighted accuracy zeroed if irrelevant
                metrics['weighted_accuracy'] = metrics['accuracy']
            else:
                metrics['accuracy'] = 0.0
                metrics['weighted_accuracy'] = 0.0
        elif 'direction_correct' in df.columns:
            metrics['accuracy'] = df['direction_correct'].mean()
            if 'confidence' in df.columns:
                total_conf = df['confidence'].sum()
                if total_conf > 0:
                    metrics['weighted_accuracy'] = (
                        (df['direction_correct'] * df['confidence']).sum() / total_conf
                    )
                else:
                    metrics['weighted_accuracy'] = 0.0
        
        # Evaluate Brier Score for probabilities (closer to 0 is better)
        if 'predicted_probability' in df.columns and 'direction_correct' in df.columns:
            exit_events = df.dropna(subset=['exit_reason'])
            if len(exit_events) > 0:
                y_true = exit_events['direction_correct'].astype(int)
                y_pred = exit_events['predicted_probability']
                # Standard Brier Score computation
                brier_score = ((y_pred - y_true)**2).mean()
                metrics['brier_score'] = brier_score
            else:
                metrics['brier_score'] = None # FIX: Return None, do not penalize with 1.0
        else:
            metrics['brier_score'] = None
        
        initial_cap = 100000.0
        equity_series = initial_cap * (1 + cum_strategy)
        equity_curve = equity_series.values
        metrics['max_drawdown'] = risk_analyzer.calculate_max_drawdown(equity_curve)
        # Portfolio Sharpe: adjust risk-free deduction proportionally to capital utilization
        # A strategy that chooses to be in cash is earning the risk-free rate on idle capital
        position_held_mask_sharpe = df['position_qty'] > 0
        capital_utilization = position_held_mask_sharpe.sum() / max(len(df), 1)
        metrics['capital_utilization'] = float(capital_utilization)
        
        # ── TRADE-LEVEL SHARPE (correct for low-frequency swing systems) ──
        # Daily-bar Sharpe is meaningless with ~95% cash days diluting the mean.
        # Compute Sharpe from individual completed trade returns instead.
        trade_returns = df.loc[df['exit_reason'].notna(), 'trade_pnl_pct'].tolist() if 'trade_pnl_pct' in df.columns else []
        
        if len(trade_returns) >= 3:
            trade_ret_array = np.array(trade_returns)  # pct returns per trade
            n_trades = len(trade_ret_array)
            
            # Annualize by actual trading frequency
            test_period_years = len(df) / 252.0
            trades_per_year = n_trades / test_period_years if test_period_years > 0 else n_trades
            
            mean_trade = trade_ret_array.mean()
            std_trade  = trade_ret_array.std(ddof=1)
            
            # The correct risk-free hurdle scales with the actual holding period of the trade,
            # not divided by trades per year. Approx 5-10 bars held avg across portfolio.
            # 7 days / 252 = 0.027 years * 6.5% = 0.18% hurdle per trade.
            avg_bars_held = df.loc[df['exit_reason'].notna(), 'bars_held'].mean() if 'bars_held' in df.columns else 7
            # Scale rf by capital utilization: idle cash does not incur rf cost
            capital_util = float(position_held_mask_sharpe.sum()) / max(len(df), 1)
            rf_per_trade = 0.065 * (avg_bars_held / 252.0) * max(capital_util, 0.1)
            
            if std_trade > 1e-8:
                metrics['sharpe_ratio'] = float(
                    ((mean_trade - rf_per_trade) / std_trade) * np.sqrt(trades_per_year)
                )
            else:
                metrics['sharpe_ratio'] = 0.0
                
            metrics['sharpe_trades_only'] = metrics['sharpe_ratio']
        else:
            # Fallback to daily for insufficient trades
            adjusted_returns = df['strategy_return_net'].copy()
            adjusted_returns[position_held_mask_sharpe] -= (0.065 / 252)
            metrics['sharpe_ratio'] = PerformanceAnalyzer.calculate_sharpe(
                adjusted_returns, risk_free_rate=0.0, interval='1d'
            )
            metrics['sharpe_trades_only'] = metrics['sharpe_ratio']
        if len(trade_returns) >= 3:
            trade_ret_array = np.array(trade_returns)
            test_period_years = len(df) / 252.0
            trades_per_year = len(trade_ret_array) / test_period_years if test_period_years > 0 else 0
            
            avg_bars_held = df.loc[df['exit_reason'].notna(), 'bars_held'].mean() if 'bars_held' in df.columns else 7
            rf_per_trade = 0.065 * (avg_bars_held / 252.0)
            
            shortfalls = np.minimum(trade_ret_array - rf_per_trade, 0)
            downside_var = np.mean(shortfalls ** 2)
            
            if downside_var > 1e-10:
                mean_excess = trade_ret_array.mean() - rf_per_trade
                metrics['sortino_ratio'] = float(
                    (mean_excess / np.sqrt(downside_var)) * np.sqrt(trades_per_year)
                )
            else:
                metrics['sortino_ratio'] = 0.0
        else:
            metrics['sortino_ratio'] = 0.0
        
        metrics['calmar_ratio'] = (
            metrics['total_return_strategy'] / abs(metrics['max_drawdown']) 
            if metrics['max_drawdown'] != 0 else 0
        )
        
        # Win rate: count actual completed trades (exits) with positive PnL
        if 'action' in df.columns or 'exit_reason' in df.columns:
            # FIX: Count all valid structural exits against the total win rate denominator
            exit_col = 'exit_reason' if 'exit_reason' in df.columns else 'action'
            exit_mask = df[exit_col].isin(['STOP_LOSS_HIT', 'PROFIT_TARGET_HIT', 'TIME_BARRIER_HIT', 'EMERGENCY_STOP'])
            exit_trades = df.loc[exit_mask]
            if len(exit_trades) > 0:
                # trade_pnl column has the actual PnL of each completed trade
                pnl_col = 'trade_pnl' if 'trade_pnl' in df.columns else 'trade_pnl_pct' if 'trade_pnl_pct' in df.columns else 'strategy_return_net'
                winning_exits = (exit_trades[pnl_col] > 0).sum()
                metrics['win_rate'] = float(winning_exits / len(exit_trades))
            else:
                metrics['win_rate'] = 0.0
        else:
            trade_returns = df.loc[df['trade_occurred'], 'strategy_return_net']
            winning_trades = (trade_returns > 0).sum()
            metrics['win_rate'] = winning_trades / len(trade_returns) if len(trade_returns) > 0 else 0
        
        risk_check = risk_analyzer.check_portfolio_risk(equity_curve)
        metrics['risk_safe'] = 1.0 if risk_check.get('is_safe', True) else 0.0
        metrics['risk_alerts_count'] = len(risk_check.get('alerts', []))

        return metrics
    
    @staticmethod
    def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.05, interval: str = '1d') -> float:
        """Calculate Sharpe Ratio"""
        if len(returns) < 2:
            return 0.0
            
        std_ret = returns.std()
        if std_ret < 1e-8:
            return 0.0
        
        periods_map = {
            '1m': 252 * 375,
            '5m': 252 * 75,
            '15m': 252 * 25,
            '1h': 252 * 6.25,
            '1d': 252
        }
        periods_per_year = periods_map.get(interval, 252)
        
        excess_return = returns.mean() - (risk_free_rate / periods_per_year)
        return float((excess_return / std_ret) * np.sqrt(periods_per_year))
    
    @staticmethod
    def calculate_sortino(returns: pd.Series, risk_free_rate: float = 0.05, interval: str = '1d') -> float:
        """Calculate Sortino Ratio"""
        if len(returns) < 2:
            return 0.0
        
        periods_map = {
            '1m': 252 * 375,
            '5m': 252 * 75,
            '15m': 252 * 25,
            '1h': 252 * 6.25,
            '1d': 252
        }
        periods_per_year = periods_map.get(interval, 252)
        
        mar = risk_free_rate / periods_per_year
        excess_return = returns.mean() - mar
        
        shortfalls = np.minimum(returns - mar, 0)
        downside_variance = np.mean(shortfalls ** 2)
        
        if downside_variance <= 0:
            return 0.0
            
        return float((excess_return / np.sqrt(downside_variance)) * np.sqrt(periods_per_year))


class AdvancedVisualizer:
    """Create comprehensive validation charts"""
    
    @staticmethod
    def create_comprehensive_report(symbol: str, df: pd.DataFrame, metrics: Dict[str, float], trade_log: list = None):
        """Generate multi-panel analysis report"""
        
        fig = plt.figure(figsize=(20, 18))
        gs = GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.3, height_ratios=[3, 3, 3, 4])
        
        colors = {
            'primary': '#2E86AB',
            'secondary': '#F77F00',
            'success': '#06A77D',
            'danger': '#D62828',
            'neutral': '#6C757D'
        }
        
        ax1 = fig.add_subplot(gs[0, :])
        AdvancedVisualizer._plot_price_signals(ax1, df, symbol, colors, trade_log=trade_log)
        
        ax2 = fig.add_subplot(gs[1, :])
        AdvancedVisualizer._plot_cumulative_returns(ax2, df, colors)
        
        ax3 = fig.add_subplot(gs[2, 0])
        AdvancedVisualizer._plot_rolling_accuracy(ax3, df, colors)
        
        ax4 = fig.add_subplot(gs[2, 1])
        AdvancedVisualizer._plot_confidence_distribution(ax4, df, colors)
        
        ax5 = fig.add_subplot(gs[2, 2])
        AdvancedVisualizer._plot_signal_distribution(ax5, df, colors)
        
        ax6 = fig.add_subplot(gs[3, :])
        AdvancedVisualizer._plot_metrics_table(ax6, metrics, colors)
        
        plt.suptitle(f'{symbol} - MARK5 Model Validation Report', 
                     fontsize=20, fontweight='bold', y=0.95)
        
        if not os.path.exists(CHARTS_DIR):
            os.makedirs(CHARTS_DIR)
            
        output_path = os.path.join(CHARTS_DIR, f'{symbol}_comprehensive.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"📊 Saved comprehensive report: {symbol}")
    
    @staticmethod
    def _plot_price_signals(ax, df, symbol, colors, trade_log=None):
        """Plot price with actual trade entries and exits from trade_log."""
        ax.plot(df.index, df['close'], label='Close Price', color=colors['primary'], linewidth=1.5, alpha=0.7)

        exit_color_map = {
            'STOP_LOSS_HIT':    colors['danger'],
            'PROFIT_TARGET_HIT':'#FFD700',
            'TIME_BARRIER_HIT': colors['secondary'],
            'GAP_ADVERSE_EXIT': colors['neutral'],
            'EMERGENCY_STOP':   colors['neutral'],
        }

        if trade_log:
            entries = [(t['entry_time'], t['entry_price']) for t in trade_log]
            exits   = [(t['exit_time'],  t['exit_price'],  t['exit_reason']) for t in trade_log]

            if entries:
                e_times, e_prices = zip(*entries)
                ax.scatter(e_times, e_prices, marker='^', s=120,
                        color=colors['success'], label=f'Entry ({len(entries)})',
                        alpha=0.9, edgecolors='black', linewidth=0.5, zorder=5)

            for reason, label in [
                ('STOP_LOSS_HIT',    'Stop Loss'),
                ('PROFIT_TARGET_HIT','Profit Target'),
                ('TIME_BARRIER_HIT', 'Time Exit'),
            ]:
                pts = [(t, p) for t, p, r in exits if r == reason]
                if pts:
                    x_times, x_prices = zip(*pts)
                    ax.scatter(x_times, x_prices, marker='v', s=120,
                            color=exit_color_map[reason],
                            label=f'{label} ({len(pts)})',
                            alpha=0.9, edgecolors='black', linewidth=0.5, zorder=5)

            # Draw entry→exit lines for each completed trade
            for t in trade_log:
                ax.plot([t['entry_time'], t['exit_time']],
                        [t['entry_price'], t['exit_price']],
                        color='gray', linewidth=0.6, alpha=0.3, zorder=3)
        else:
            # Fallback to action column if no trade_log
            if 'action' in df.columns:
                for action_val, marker, color, label_prefix in [
                    ('BUY',              '^', colors['success'],   'Entry'),
                    ('STOP_LOSS_HIT',    'v', colors['danger'],    'Stop Loss'),
                    ('PROFIT_TARGET_HIT','v', '#FFD700',           'Profit Target'),
                    ('TIME_BARRIER_HIT', 'v', colors['secondary'], 'Time Exit'),
                ]:
                    mask = df['action'] == action_val
                    if mask.sum() > 0:
                        ax.scatter(df.index[mask], df['close'][mask], marker=marker, s=120,
                                color=color, label=f'{label_prefix} ({mask.sum()})',
                                alpha=0.9, edgecolors='black', linewidth=0.5, zorder=5)

        ax.set_title('Price Action & Trade Entries/Exits', fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Price (₹)', fontsize=10)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('')
    
    @staticmethod
    def _plot_cumulative_returns(ax, df, colors):
        """Plot cumulative returns comparison"""
        df_copy = df.copy()
        if 'strategy_return_net' in df_copy.columns:
             df_copy['cum_strategy'] = (1 + df_copy['strategy_return_net']).cumprod() - 1
        else:
             df_copy['cum_strategy'] = 0
             
        if 'market_return' in df_copy.columns:
            df_copy['cum_market'] = (1 + df_copy['market_return']).cumprod() - 1
        else:
            df_copy['cum_market'] = 0
        
        ax.plot(df_copy.index, df_copy['cum_market'] * 100, 
               label='Buy & Hold', color=colors['neutral'], linestyle='--', linewidth=2, alpha=0.7)
        ax.plot(df_copy.index, df_copy['cum_strategy'] * 100, 
               label='MARK5 Strategy (Net)', color=colors['secondary'], linewidth=2.5)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.8)
        ax.fill_between(df_copy.index, df_copy['cum_strategy'] * 100, 0, 
                       where=(df_copy['cum_strategy'] > 0), alpha=0.2, color=colors['success'], label='Profit Zone')
        ax.fill_between(df_copy.index, df_copy['cum_strategy'] * 100, 0, 
                       where=(df_copy['cum_strategy'] < 0), alpha=0.2, color=colors['danger'], label='Loss Zone')
        
        ax.set_title('Cumulative Returns - Strategy vs Market', fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Return (%)', fontsize=10)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    @staticmethod
    def _plot_rolling_accuracy(ax, df, colors):
        """Plot expanding accuracy on CLOSED TRADES ONLY (not all bars)."""
        if 'direction_correct' in df.columns and 'exit_reason' in df.columns:
            exit_mask = df['exit_reason'].notna()
            exit_acc = df.loc[exit_mask, 'direction_correct'].astype(float)
            if exit_acc.sum() >= 1:
                expanding_acc = exit_acc.expanding().mean() * 100
                rolling_acc = expanding_acc.reindex(df.index, method='ffill').fillna(50.0)
                n_trades = int(exit_mask.sum())
                title = f'Expanding Accuracy ({n_trades} closed trades)'
            else:
                rolling_acc = pd.Series(50.0, index=df.index)
                title = 'Accuracy (no closed trades)'
        else:
            rolling_acc = pd.Series(50.0, index=df.index)
            title = 'Accuracy (data unavailable)'

        ax.plot(df.index, rolling_acc, color=colors['primary'], linewidth=2)
        ax.axhline(y=50, color=colors['danger'], linestyle='--', alpha=0.5,
                   linewidth=1.5, label='Random (50%)')
        ax.fill_between(df.index, rolling_acc, 50,
                        where=(rolling_acc > 50), alpha=0.3, color=colors['success'])
        ax.fill_between(df.index, rolling_acc, 50,
                        where=(rolling_acc < 50), alpha=0.3, color=colors['danger'])

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=9)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def _plot_confidence_distribution(ax, df, colors):
        """Plot confidence score distribution - FIXED VERSION"""
        if 'confidence' not in df.columns or 'direction_correct' not in df.columns:
            # If required columns missing, show message
            ax.text(0.5, 0.5, 'Data Not Available', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Accuracy by Confidence Level', fontsize=11, fontweight='bold')
            return
             
        # Only evaluate on bars where a trade was actually made and closed
        trade_df = df[df['exit_reason'].notna()].copy()
        if len(trade_df) < 3:
            ax.text(0.5, 0.5, 'Insufficient Trades', ha='center', va='center', fontsize=12)
            return

        confidence_bins = pd.cut(
            trade_df['confidence'],
            bins=[0.52, 0.60, 0.70, 0.80, 1.0],
            labels=['52-60%', '60-70%', '70-80%', '80%+']
        )
        accuracy_by_conf = trade_df.groupby(confidence_bins, observed=True)['direction_correct'].mean() * 100
        counts = trade_df.groupby(confidence_bins, observed=True).size()
        
        if len(accuracy_by_conf) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
            return

        x_range = range(len(accuracy_by_conf))
        bars = ax.bar(x_range, accuracy_by_conf.values, 
                     color=[colors['danger'], colors['secondary'], colors['success'], colors['primary']], 
                     alpha=0.7, edgecolor='black', linewidth=1)
        
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'n={count}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xticks(list(x_range))
        ax.set_xticklabels(accuracy_by_conf.index, fontsize=9)
        ax.set_title('Accuracy by Confidence Level', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
    
    @staticmethod
    def _plot_signal_distribution(ax, df, colors):
        """Plot signal type distribution"""
        signal_map = {
            'Strong Buy': (df['predicted_signal'] == 2).sum() if (df['predicted_signal'] == 2).any() else 0,
            'Buy': (df['predicted_signal'] == 1).sum() if (df['predicted_signal'] == 1).any() else 0,
            'Hold': (df['predicted_signal'] == 0).sum() if (df['predicted_signal'] == 0).any() else 0,
            'Sell': (df['predicted_signal'] == -1).sum() if (df['predicted_signal'] == -1).any() else 0,
            'Strong Sell': (df['predicted_signal'] == -2).sum() if (df['predicted_signal'] == -2).any() else 0
        }
        
        signal_map = {k: v for k, v in signal_map.items() if v > 0}
        
        labels = list(signal_map.keys())
        sizes = list(signal_map.values())
        
        color_map = {
            'Strong Buy': colors['success'],
            'Buy': '#90E0C1',
            'Hold': colors['neutral'],
            'Sell': '#FFA07A',
            'Strong Sell': colors['danger']
        }
        colors_pie = [color_map.get(l, 'gray') for l in labels]
        
        if sizes:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                              colors=colors_pie, startangle=90,
                                              textprops={'fontsize': 9})
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        ax.set_title('Signal Distribution', fontsize=11, fontweight='bold')
    
    @staticmethod
    def _plot_metrics_table(ax, metrics, colors):
        """Display key metrics in table format"""
        ax.axis('off')
        
        metric_data = [
            ['ACCURACY METRICS', ''],
            ['Overall Accuracy', f"{metrics.get('accuracy', 0):.1%}"],
            ['Weighted Accuracy', f"{metrics.get('weighted_accuracy', 0):.1%}"],
            ['', ''],
            ['RETURN METRICS', ''],
            ['Strategy Return', f"{metrics.get('total_return_strategy', 0):.2%}"],
            ['Market Return', f"{metrics.get('total_return_market', 0):.2%}"],
            ['Excess Return', f"{metrics.get('excess_return', 0):.2%}"],
            ['Win Rate', f"{metrics.get('win_rate', 0):.1%}"],
            ['', ''],
            ['RISK METRICS', ''],
            ['Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"],
            ['Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"],
            ['Max Drawdown', f"{metrics.get('max_drawdown', 0):.2%}"],
            ['Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.2f}"],
        ]
        
        table = ax.table(cellText=metric_data, cellLoc='left',
                        loc='center', colWidths=[0.5, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)
        
        for i in [0, 4, 10]:
            table[(i, 0)].set_facecolor(colors['primary'])
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_facecolor(colors['primary'])
        
        for i in range(len(metric_data)):
            if i not in [0, 3, 4, 9, 10]:
                table[(i, 0)].set_facecolor('#F8F9FA')
                table[(i, 1)].set_facecolor('#FFFFFF')
                table[(i, 1)].set_text_props(weight='bold')
    
    @staticmethod
    def create_interactive_chart(symbol: str, df: pd.DataFrame, metrics: Dict[str, float]):
        """Create interactive Plotly chart"""
        
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(
                f'{symbol} - Price & Signals',
                'Cumulative Returns',
                'Rolling Accuracy'
            ),
            vertical_spacing=0.08
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['close'], name='Close Price',
                      line=dict(color='#2E86AB', width=2)),
            row=1, col=1
        )
        
        buy_df = df[df['predicted_signal'] > 0]
        if not buy_df.empty:
            fig.add_trace(
                go.Scatter(x=buy_df.index, y=buy_df['close'], name='Buy Signal',
                          mode='markers', marker=dict(symbol='triangle-up', size=10, color='#06A77D')),
                row=1, col=1
            )
        
        sell_df = df[df['predicted_signal'] < 0]
        if not sell_df.empty:
            fig.add_trace(
                go.Scatter(x=sell_df.index, y=sell_df['close'], name='Sell Signal',
                          mode='markers', marker=dict(symbol='triangle-down', size=10, color='#D62828')),
                row=1, col=1
            )
        
        df_copy = df.copy()
        if 'strategy_return_net' in df_copy.columns:
            df_copy['cum_strategy'] = (1 + df_copy['strategy_return_net']).cumprod() - 1
        else:
            df_copy['cum_strategy'] = 0
            
        if 'market_return' in df_copy.columns:
            df_copy['cum_market'] = (1 + df_copy['market_return']).cumprod() - 1
        else:
            df_copy['cum_market'] = 0
        
        fig.add_trace(
            go.Scatter(x=df_copy.index, y=df_copy['cum_market'] * 100, name='Buy & Hold',
                      line=dict(color='gray', width=2, dash='dash')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df_copy.index, y=df_copy['cum_strategy'] * 100, name='MARK5 Strategy',
                      line=dict(color='#F77F00', width=3)),
            row=2, col=1
        )
        
        window = min(50, len(df) // 10)
        if window < 10: window = 10
        if 'direction_correct' in df.columns:
            rolling_acc = df['direction_correct'].rolling(window=window).mean() * 100
            
            fig.add_trace(
                go.Scatter(x=df.index, y=rolling_acc, name='Rolling Accuracy',
                          line=dict(color='#2E86AB', width=2)),
                row=3, col=1
            )
        
            fig.add_hline(y=50, line_dash="dash", line_color="red", 
                        annotation_text="Random (50%)", row=3, col=1)
        
        fig.update_layout(
            height=1000,
            showlegend=True,
            title_text=f"{symbol} - Interactive Validation Dashboard",
            title_font_size=18,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=3, col=1)
        
        if not os.path.exists(INTERACTIVE_DIR):
            os.makedirs(INTERACTIVE_DIR)
            
        output_path = os.path.join(INTERACTIVE_DIR, f'{symbol}_interactive.html')
        fig.write_html(output_path)
        
        logger.info(f"📊 Saved interactive chart: {symbol}")

# ═══════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════

def create_summary_report(all_metrics: Dict[str, Dict[str, float]]):
    """Create cross-stock summary report with heatmap."""
    
    df_numeric = pd.DataFrame(all_metrics).T
    df_numeric.index.name = 'Symbol'
    
    # Save raw numeric CSV
    summary_path = os.path.join(OUTPUT_DIR, 'validation_summary.csv')
    df_numeric.to_csv(summary_path, float_format='%.4f')
    logger.info(f"📄 Summary saved: {summary_path}")
    
    if df_numeric.empty:
        return
    
    try:
        # --- Cross-Stock Heatmap ---
        plt.figure(figsize=(16, max(6, len(df_numeric) * 0.4)))
        
        key_metrics = ['accuracy', 'sharpe_ratio', 'win_rate', 'total_return_strategy', 
                       'total_return_market', 'max_drawdown', 'sortino_ratio']
        key_metrics = [k for k in key_metrics if k in df_numeric.columns]
        
        df_heatmap = df_numeric[key_metrics].copy()
        df_heatmap.index = [s.replace('.NS', '') for s in df_heatmap.index]
        
        # Sort by Sharpe ratio
        if 'sharpe_ratio' in df_heatmap.columns:
            df_heatmap = df_heatmap.sort_values('sharpe_ratio', ascending=False)
        
        sns.heatmap(df_heatmap.astype(float), annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                    cbar_kws={'label': 'Performance Score'}, linewidths=0.5)
        
        plt.title('MARK5 Nifty 50 — Cross-Stock Performance Heatmap', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'cross_stock_heatmap.png'), dpi=200, bbox_inches='tight')
        plt.close()
        logger.info("📊 Cross-stock heatmap created")
    except Exception as e:
        logger.error(f"Failed to create heatmap: {e}")


def create_portfolio_report(all_metrics: Dict[str, Dict[str, float]], all_equity_curves: Dict[str, pd.Series]):
    """Create portfolio-level aggregate reports."""
    
    if not all_metrics:
        logger.warning("No metrics available for portfolio report")
        return
    
    df_metrics = pd.DataFrame(all_metrics).T
    df_metrics.index.name = 'Symbol'
    
    try:
        # ═══════════════════════════════════════════════════════════
        # 1. STOCK RANKING TABLE (sorted by Sharpe)
        # ═══════════════════════════════════════════════════════════
        rank_cols = ['sharpe_ratio', 'total_return_strategy', 'accuracy', 'win_rate', 'max_drawdown']
        rank_cols = [c for c in rank_cols if c in df_metrics.columns]
        df_rank = df_metrics[rank_cols].copy()
        df_rank.index = [s.replace('.NS', '') for s in df_rank.index]
        
        if 'sharpe_ratio' in df_rank.columns:
            df_rank = df_rank.sort_values('sharpe_ratio', ascending=False)
        
        fig, ax = plt.subplots(figsize=(14, max(4, len(df_rank) * 0.35)))
        ax.axis('off')
        ax.set_title('MARK5 — Stock Rankings by Sharpe Ratio', fontsize=14, fontweight='bold', pad=15)
        
        cell_text = []
        for idx, row in df_rank.iterrows():
            cell_text.append([f"{v:.3f}" for v in row.values])
        
        colors = []
        for _, row in df_rank.iterrows():
            row_colors = []
            for col in rank_cols:
                val = row[col]
                if col == 'max_drawdown':
                    row_colors.append('#ff6b6b' if val < -0.10 else '#51cf66' if val > -0.05 else '#ffd43b')
                elif col == 'sharpe_ratio':
                    row_colors.append('#51cf66' if val > 0.5 else '#ff6b6b' if val < 0 else '#ffd43b')
                elif col in ['accuracy', 'win_rate']:
                    row_colors.append('#51cf66' if val > 0.55 else '#ff6b6b' if val < 0.45 else '#ffd43b')
                else:
                    row_colors.append('#51cf66' if val > 0 else '#ff6b6b')
            colors.append(row_colors)
        
        table = ax.table(cellText=cell_text, rowLabels=df_rank.index.tolist(),
                        colLabels=[c.replace('_', ' ').title() for c in rank_cols],
                        cellColours=colors, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'stock_rankings.png'), dpi=200, bbox_inches='tight')
        plt.close()
        logger.info("📊 Stock rankings created")
        
        # ═══════════════════════════════════════════════════════════
        # 2. SECTOR PERFORMANCE BREAKDOWN
        # ═══════════════════════════════════════════════════════════
        if NIFTY_50:
            sector_data = {}
            for ticker, metrics in all_metrics.items():
                sector = NIFTY_50.get(ticker, {}).get('sector', 'Unknown')
                if sector not in sector_data:
                    sector_data[sector] = {'returns': [], 'sharpes': [], 'accuracies': []}
                sector_data[sector]['returns'].append(metrics.get('total_return_strategy', 0))
                sector_data[sector]['sharpes'].append(metrics.get('sharpe_ratio', 0))
                sector_data[sector]['accuracies'].append(metrics.get('accuracy', 0))
            
            if sector_data:
                sectors = sorted(sector_data.keys())
                avg_returns = [np.mean(sector_data[s]['returns']) * 100 for s in sectors]
                avg_sharpes = [np.mean(sector_data[s]['sharpes']) for s in sectors]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                bar_colors = ['#51cf66' if r > 0 else '#ff6b6b' for r in avg_returns]
                ax1.barh(sectors, avg_returns, color=bar_colors, edgecolor='white')
                ax1.set_xlabel('Average Return (%)')
                ax1.set_title('Avg Return by Sector', fontweight='bold')
                ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                
                bar_colors2 = ['#51cf66' if s > 0.5 else '#ff6b6b' if s < 0 else '#ffd43b' for s in avg_sharpes]
                ax2.barh(sectors, avg_sharpes, color=bar_colors2, edgecolor='white')
                ax2.set_xlabel('Average Sharpe Ratio')
                ax2.set_title('Avg Sharpe by Sector', fontweight='bold')
                ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                
                plt.suptitle('MARK5 — Sector Performance Breakdown', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, 'sector_performance.png'), dpi=200, bbox_inches='tight')
                plt.close()
                logger.info("📊 Sector performance chart created")
        
        # ═══════════════════════════════════════════════════════════
        # 3. AGGREGATE PORTFOLIO EQUITY CURVE
        # ═══════════════════════════════════════════════════════════
        if all_equity_curves:
            fig, ax = plt.subplots(figsize=(16, 8))
            
            # Align all curves to common index and compute equal-weight portfolio
            aligned = pd.DataFrame(all_equity_curves)
            aligned = aligned.ffill().bfill()
            
            # Equal-weight portfolio
            portfolio = aligned.mean(axis=1) * 100
            
            # Market benchmark (average of market returns if available)
            ax.plot(portfolio.index, portfolio.values, color='#339af0', linewidth=2.5, label='Equal-Weight Portfolio (%)')
            
            # Individual stocks (faded)
            for ticker, curve in all_equity_curves.items():
                short_name = ticker.replace('.NS', '')
                ax.plot(curve.index, curve.values * 100, alpha=0.15, linewidth=0.8, label=None)
            
            # Highlight top 3 and bottom 3
            final_returns = {t: c.iloc[-1] if len(c) > 0 else 0 for t, c in all_equity_curves.items()}
            sorted_tickers = sorted(final_returns.items(), key=lambda x: x[1], reverse=True)
            
            for i, (ticker, _) in enumerate(sorted_tickers[:3]):
                c = all_equity_curves[ticker]
                ax.plot(c.index, c.values, linewidth=1.5, alpha=0.8, 
                       label=f"🟢 {ticker.replace('.NS', '')}")
            
            for i, (ticker, _) in enumerate(sorted_tickers[-3:]):
                c = all_equity_curves[ticker]
                ax.plot(c.index, c.values, linewidth=1.5, alpha=0.8, linestyle='--',
                       label=f"🔴 {ticker.replace('.NS', '')}")
            
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_title('MARK5 — Portfolio Equity Curves (Cumulative Returns)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Cumulative Return')
            ax.set_xlabel('Date')
            ax.legend(loc='upper left', fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'portfolio_equity_curve.png'), dpi=200, bbox_inches='tight')
            plt.close()
            logger.info("📊 Portfolio equity curve created")
        
        # ═══════════════════════════════════════════════════════════
        # 4. PORTFOLIO DASHBOARD
        # ═══════════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Top-left: Return distribution
        ax = axes[0, 0]
        returns = [m.get('total_return_strategy', 0) * 100 for m in all_metrics.values()]
        colors_ret = ['#51cf66' if r > 0 else '#ff6b6b' for r in returns]
        ax.bar(range(len(returns)), sorted(returns, reverse=True), color=colors_ret, edgecolor='white')
        ax.set_title('Strategy Returns Distribution (%)', fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Stock (ranked)')
        
        # Top-right: Accuracy distribution
        ax = axes[0, 1]
        accuracies = [m.get('accuracy', 0) * 100 for m in all_metrics.values()]
        ax.hist(accuracies, bins=15, color='#339af0', edgecolor='white', alpha=0.8)
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
        ax.axvline(x=np.mean(accuracies), color='green', linestyle='-', alpha=0.7, 
                  label=f'Avg ({np.mean(accuracies):.1f}%)')
        ax.set_title('Direction Accuracy Distribution', fontweight='bold')
        ax.set_xlabel('Accuracy (%)')
        ax.legend()
        
        # Bottom-left: Sharpe distribution
        ax = axes[1, 0]
        sharpes = [m.get('sharpe_ratio', 0) for m in all_metrics.values()]
        colors_sh = ['#51cf66' if s > 0 else '#ff6b6b' for s in sharpes]
        ax.bar(range(len(sharpes)), sorted(sharpes, reverse=True), color=colors_sh, edgecolor='white')
        ax.set_title('Sharpe Ratio Distribution', fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.5, label='Good (1.0)')
        ax.set_xlabel('Stock (ranked)')
        ax.legend()
        
        # Bottom-right: Win rate vs Sharpe scatter
        ax = axes[1, 1]
        win_rates = [m.get('win_rate', 0) * 100 for m in all_metrics.values()]
        ax.scatter(win_rates, sharpes, c=returns, cmap='RdYlGn', s=80, edgecolors='white', zorder=3)
        ax.set_xlabel('Win Rate (%)')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Win Rate vs Sharpe (color = return)', fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.2)
        
        plt.suptitle(f'MARK5 — Portfolio Dashboard ({len(all_metrics)} Stocks)', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'portfolio_dashboard.png'), dpi=200, bbox_inches='tight')
        plt.close()
        logger.info("📊 Portfolio dashboard created")
        
        # ═══════════════════════════════════════════════════════════
        # 5. PORTFOLIO SUMMARY STATS
        # ═══════════════════════════════════════════════════════════
        print(f"\n{'═' * 70}")
        print("📊 PORTFOLIO SUMMARY")
        print(f"{'═' * 70}")
        
        total_stocks = len(all_metrics)
        profitable = sum(1 for m in all_metrics.values() if m.get('total_return_strategy', 0) > 0)
        
        print(f"  Stocks Analyzed:     {total_stocks}")
        print(f"  Profitable Stocks:   {profitable}/{total_stocks} ({profitable/total_stocks:.0%})")
        print(f"  Avg Accuracy:        {np.mean(accuracies):.1f}%")
        print(f"  Avg Sharpe:          {np.mean(sharpes):.2f}")
        print(f"  Avg Return:          {np.mean(returns):.2f}%")
        print(f"  Median Return:       {np.median(returns):.2f}%")
        print(f"  Best Stock:          {sorted_tickers[0][0].replace('.NS', '')} ({sorted_tickers[0][1]:.2%})" if all_equity_curves else "")
        print(f"  Worst Stock:         {sorted_tickers[-1][0].replace('.NS', '')} ({sorted_tickers[-1][1]:.2%})" if all_equity_curves else "")
        print(f"  Avg Win Rate:        {np.mean(win_rates):.1f}%")
        print(f"  Avg Max Drawdown:    {np.mean([m.get('max_drawdown', 0) for m in all_metrics.values()]):.2%}")
        print(f"{'═' * 70}")
        
    except Exception as e:
        logger.error(f"Portfolio report failed: {e}")
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

def main():
    """Main execution pipeline"""
    global START_DATE, END_DATE, OUTPUT_DIR, CHARTS_DIR, WATCHLIST
    
    print("=" * 70)
    print("🚀 MARK5 PRODUCTION VALIDATION PIPELINE v4.0 — REGRESSION EDITION")
    print("=" * 70)
    print(f"📅 Period: {START_DATE} to {END_DATE}")
    print(f"📈 Stocks: {len(WATCHLIST)}")
    print(f"⏱️  Interval: {INTERVAL}")
    print(f"📂 Output: {OUTPUT_DIR}")
    print(f"💾 Cache: {DATA_CACHE_DIR}")
    print("=" * 70)
    
    pipeline_start = time.time()
    
    # Step 1: Download data
    print("\n[STEP 1/5] Downloading Historical Data...")
    data = download_data(WATCHLIST)
    
    if not data:
        logger.error("❌ No data downloaded. Exiting.")
        return
    
    print(f"✅ Downloaded data for {len(data)} stocks")
    
    # ── [NEW CROSS-SECTIONAL PORTFOLIO RUN PATH] ─────────────────────
    """
MARK5 test.py — CROSS-SECTIONAL SIMULATION BLOCK v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DROP-IN REPLACEMENT for the block between:
  # ── [NEW CROSS-SECTIONAL PORTFOLIO RUN PATH] ─────────────────────
  ...
  # ── [/END CROSS-SECTIONAL] ───────────────────────────────────────

WHAT CHANGED FROM v1.0:
  v1.0: Iterated Fridays only. Held top-3 all week. No ML. No exits.
  v2.0: Iterates ALL business days.
        Fridays:  re-rank universe → refresh top-10 eligible stocks.
        Daily:    ML gate on top-10 → enter positions (conf >= 0.55).
        Daily:    Triple barrier exits (PT=2.5×ATR, SL=1.5×ATR, Time=10).
        Result:   Better entries, intra-week exits, ML-confirmed momentum.

PASTE INSTRUCTIONS:
  In test.py, find the line:
    if not getattr(args, 'per_stock', False):
  Replace everything from that line up to (but NOT including):
    # ── [/END CROSS-SECTIONAL] ───────────────────────────────────────
  with the contents of this file.
"""

    # ── [NEW CROSS-SECTIONAL PORTFOLIO RUN PATH v2.0] ──────────────────────
    if not getattr(args, 'per_stock', False):
        from core.models.ranker import CrossSectionalRanker
        from core.models.predictor import MARK5Predictor
        print("\n[STEP 2/5] Running CROSS-SECTIONAL RANKING + ML Portfolio Simulation v2.0")

        # ── NIFTY data ─────────────────────────────────────────────────────
        from core.data.market_data import MarketDataProvider
        mp = MarketDataProvider()
        
        # FIX: Use Kite/Cache instead of yfinance for NIFTY50
        nifty_df = mp.get_nifty50_data(START_DATE, END_DATE)
        if nifty_df is not None and not nifty_df.empty:
            nifty_close = nifty_df['close']
        else:
            print("⚠️ NIFTY50 data unavailable from Kite/Cache — attempting yfinance fallback")
            import yfinance as yf
            nifty_v = yf.download('^NSEI', start=START_DATE, end=END_DATE,
                                progress=False, auto_adjust=True)
            if not nifty_v.empty:
                nifty_close = nifty_v['Close']
                if hasattr(nifty_close, 'columns'): # handle multi-index
                    nifty_close = nifty_close.iloc[:, 0]
            else:
                print("❌ FATAL: NIFTY50 data unavailable. Benchmark will be zero.")
                nifty_close = pd.Series(0, index=pd.date_range(START_DATE, END_DATE))

        nifty_close.name = 'close'
        if nifty_close.index.tz is not None:
            nifty_close.index = nifty_close.index.tz_localize(None)

        try:
            nifty_base_mask = nifty_close.index >= pd.Timestamp(START_DATE)
            if nifty_base_mask.any():
                nifty_base = float(nifty_close.loc[nifty_base_mask].iloc[0])
            else:
                nifty_base = float(nifty_close.iloc[0])
        except Exception:
            nifty_base = 1.0 # prevent div zero

        ranker = CrossSectionalRanker(top_n=3)

        start_ts = pd.Timestamp(START_DATE)
        end_ts   = pd.Timestamp(END_DATE)

        # ── FII data ───────────────────────────────────────────────────────
        from core.data.fii_data import FIIDataProvider
        fii = FIIDataProvider()
        fii_start = (start_ts - pd.Timedelta(days=100)).strftime('%Y-%m-%d')
        fii_net = fii.get_fii_flow(start_date=fii_start, end_date=END_DATE)

        # ── Pre-load all predictors (reused every day — no repeated disk I/O) ──
        print("  Loading ML models for all stocks...")
        all_predictors = {}
        for sym in list(data.keys()):
            try:
                p = MARK5Predictor(sym)
                p.reload_artifacts()
                if p._container is not None:
                    all_predictors[sym] = p
            except Exception:
                pass
        print(f"  Loaded {len(all_predictors)}/{len(data)} ML models")

        # ── Simulation state ───────────────────────────────────────────────
        # active_positions: {sym: {entry_price, atr_pct, bars_held, entry_date}}
        active_positions = {}
        ranked_top_10    = []       # refreshed every Friday
        portfolio_value  = 1.0
        portfolio_curve  = [1.0]
        nifty_curve      = [1.0]
        trade_log        = []
        weekly_returns   = []
        ml_gate_log      = []       # tracks pass rate for cash drag diagnostic

        # All business days in test window
        all_dates = pd.bdate_range(start_ts, end_ts, freq='B')

        ML_THRESHOLD     = 0.55
        ML_THRESHOLD_BEAR = 0.70
        MAX_BARS_HOLD    = 10
        MAX_POSITIONS    = 3

        prev_portfolio_value = 1.0
        prev_date = None

        for curr_ts in all_dates:
            curr_ts = pd.Timestamp(curr_ts)

            # ── Bear market gate (RULE 23) ─────────────────────────────────
            nifty_hist = nifty_close.loc[nifty_close.index <= curr_ts]
            nifty_bear = False
            if len(nifty_hist) >= 200:
                nifty_200ema = float(nifty_hist.ewm(span=200, adjust=False).mean().iloc[-1])
                curr_nifty   = float(nifty_hist.iloc[-1])
                prev_nifty   = float(nifty_hist.iloc[-20])
                nifty_20d_ret = (curr_nifty / prev_nifty) - 1.0
                if curr_nifty < nifty_200ema and nifty_20d_ret < -0.05:
                    nifty_bear = True

            min_conf  = ML_THRESHOLD_BEAR if nifty_bear else ML_THRESHOLD
            eff_n     = 1 if nifty_bear else MAX_POSITIONS

            # ── LAYER 3: Triple barrier exits (runs every day) ─────────────
            for sym in list(active_positions.keys()):
                pos = active_positions[sym]
                if sym not in data:
                    continue
                df_sym = data[sym]
                if curr_ts not in df_sym.index:
                    continue

                bar = df_sym.loc[curr_ts]
                pos['bars_held'] += 1

                entry  = pos['entry_price']
                atr_p  = pos['atr_pct']
                pt     = entry * (1.0 + 2.5 * atr_p)
                sl     = entry * (1.0 - 1.5 * atr_p)

                exit_reason = None
                exit_price  = None

                if float(bar['low']) <= sl:
                    exit_reason = 'STOP_LOSS'
                    exit_price  = sl
                elif float(bar['high']) >= pt:
                    exit_reason = 'PROFIT_TARGET'
                    exit_price  = pt
                elif pos['bars_held'] >= MAX_BARS_HOLD:
                    exit_reason = 'TIME_BARRIER'
                    exit_price  = float(bar['close'])

                if exit_reason:
                    pnl_pct = (exit_price - entry) / entry
                    # Deduct 0.15% transaction cost on exit leg
                    pnl_pct -= 0.0015
                    trade_log.append({
                        'symbol':      sym,
                        'entry_date':  pos['entry_date'],
                        'exit_date':   curr_ts,
                        'exit_reason': exit_reason,
                        'pnl_pct':     pnl_pct,
                        'bars_held':   pos['bars_held'],
                        'confidence':  pos.get('ml_confidence', 0.0),
                    })
                    del active_positions[sym]
                    logger.info(
                        f"EXIT {sym} | {exit_reason} | "
                        f"pnl={pnl_pct:+.2%} | bars={pos['bars_held']}"
                    )

            # ── LAYER 1: Re-rank every Friday ──────────────────────────────
            if curr_ts.weekday() == 4:
                try:
                    ranked_full  = ranker.rank_universe(
                        data, nifty_close, fii_net, curr_ts
                    )
                    ranked_top_10 = ranked_full[:10]
                except Exception as e:
                    logger.warning(f"Ranking failed on {curr_ts.date()}: {e}")

            # ── LAYER 2: ML entry filter (runs every day) ──────────────────
            open_slots = eff_n - len(active_positions)

            if open_slots > 0 and ranked_top_10:
                candidates = []

                for sym, rank_score in ranked_top_10:
                    if sym in active_positions:
                        continue
                    if sym not in data:
                        continue

                    predictor = all_predictors.get(sym)
                    if predictor is None:
                            # No model for this stock — allow entry on ranking signal alone
                            # Ranking has already confirmed strong momentum; ML would be additive
                        candidates.append((sym, rank_score, ML_THRESHOLD))  # neutral confidence
                        continue

                    try:
                        df_sym = data[sym]
                        if curr_ts not in df_sym.index:
                            continue

                        hist = df_sym.loc[df_sym.index <= curr_ts].tail(300)
                        if len(hist) < 30:
                            continue

                        result     = predictor.predict(hist)
                        confidence = float(result.get('confidence', 0.0))

                        if confidence >= min_conf:
                            candidates.append((sym, rank_score, confidence))
                    except Exception:
                        continue

                # Sort by ML confidence, fill open slots
                candidates.sort(key=lambda x: x[2], reverse=True)
                ml_gate_log.append(len(candidates))

                for sym, rank_score, confidence in candidates[:open_slots]:
                    df_sym = data[sym]
                    if curr_ts not in df_sym.index:
                        continue

                    bar = df_sym.loc[curr_ts]

                    # Compute ATR for this stock
                    hist_for_atr = df_sym.loc[df_sym.index <= curr_ts].tail(20)
                    if len(hist_for_atr) >= 15:
                        highs  = hist_for_atr['high'].values
                        lows   = hist_for_atr['low'].values
                        closes = hist_for_atr['close'].values
                        tr = np.maximum(
                            highs[1:] - lows[1:],
                            np.abs(highs[1:] - closes[:-1])
                        )
                        tr = np.maximum(tr, np.abs(lows[1:] - closes[:-1]))
                        atr_pct = float(np.mean(tr[-14:])) / float(bar['close'])
                    else:
                        atr_pct = 0.02

                    entry_price = float(bar['close']) * 1.0005  # 0.05% slippage
                    # Deduct 0.15% transaction cost on entry leg
                    entry_price *= 1.0015

                    active_positions[sym] = {
                        'entry_price':  entry_price,
                        'atr_pct':      atr_pct,
                        'bars_held':    0,
                        'entry_date':   curr_ts,
                        'ml_confidence': confidence,
                    }
                    logger.info(
                        f"ENTER {sym} | conf={confidence:.2%} | "
                        f"rank={rank_score:+.3f} | atr={atr_pct:.2%}"
                    )

            # ── Daily portfolio mark-to-market ─────────────────────────────
            # Compute daily return as average return across active positions
            if prev_date is not None and active_positions:
                day_rets = []
                for sym, pos in active_positions.items():
                    if sym not in data:
                        continue
                    df_sym = data[sym]
                    curr_rows = df_sym.loc[df_sym.index == curr_ts]
                    prev_rows = df_sym.loc[df_sym.index == prev_date]
                    if curr_rows.empty or prev_rows.empty:
                        continue
                    r = float(curr_rows['close'].iloc[0]) / float(prev_rows['close'].iloc[0]) - 1.0
                    day_rets.append(r)

                if day_rets:
                    # Scale by capital deployment
                    deployment = len(active_positions) / MAX_POSITIONS
                    avg_r = np.mean(day_rets) * deployment
                    portfolio_value *= (1.0 + avg_r)

            portfolio_curve.append(portfolio_value)

            try:
                nifty_now = float(nifty_close.loc[nifty_close.index <= curr_ts].iloc[-1])
                nifty_curve.append(nifty_now / nifty_base)
            except Exception:
                nifty_curve.append(nifty_curve[-1])

            prev_date = curr_ts

        # ── Results ────────────────────────────────────────────────────────
        years = (all_dates[-1] - all_dates[0]).days / 365.25
        port_total  = (portfolio_value - 1) * 100
        nifty_total = (
            float(nifty_close.loc[nifty_close.index <= end_ts].iloc[-1])
            / nifty_base - 1
        ) * 100
        port_cagr  = ((portfolio_value) ** (1 / max(years, 0.1)) - 1) * 100
        nifty_cagr = ((1 + nifty_total / 100) ** (1 / max(years, 0.1)) - 1) * 100

        p_curve  = pd.Series(portfolio_curve)
        p_max_dd = (p_curve / p_curve.cummax() - 1.0).min() * 100
        n_curve  = pd.Series(nifty_curve)
        n_max_dd = (n_curve / n_curve.cummax() - 1.0).min() * 100

        # Trade-level stats
        n_trades  = len(trade_log)
        wins      = [t for t in trade_log if t['pnl_pct'] > 0]
        losses    = [t for t in trade_log if t['pnl_pct'] <= 0]
        win_rate  = len(wins) / n_trades if n_trades > 0 else 0.0
        avg_win   = np.mean([t['pnl_pct'] for t in wins])   * 100 if wins   else 0.0
        avg_loss  = np.mean([t['pnl_pct'] for t in losses]) * 100 if losses else 0.0

        pt_exits  = sum(1 for t in trade_log if t['exit_reason'] == 'PROFIT_TARGET')
        sl_exits  = sum(1 for t in trade_log if t['exit_reason'] == 'STOP_LOSS')
        tb_exits  = sum(1 for t in trade_log if t['exit_reason'] == 'TIME_BARRIER')

        # ML gate pass rate (cash drag diagnostic)
        avg_pass = np.mean(ml_gate_log) if ml_gate_log else 0.0

        print(f"\n{'='*60}")
        print(f"Period: {START_DATE} → {END_DATE} ({years:.1f} years)")
        print(f"{'='*60}")
        print(f"Portfolio total:  {port_total:+.1f}%   CAGR: {port_cagr:+.1f}%   Max DD: {p_max_dd:.1f}%")
        print(f"NIFTY total:      {nifty_total:+.1f}%   CAGR: {nifty_cagr:+.1f}%   Max DD: {n_max_dd:.1f}%")
        print(f"Excess return:    {port_total - nifty_total:+.1f}%")
        print(f"{'='*60}")
        print(f"\n📊 TRADE STATISTICS")
        print(f"{'─'*60}")
        print(f"  Total trades:         {n_trades}")
        print(f"  Win rate:             {win_rate:.1%}  ({len(wins)}W / {len(losses)}L)")
        print(f"  Avg win:              {avg_win:+.2f}%")
        print(f"  Avg loss:             {avg_loss:+.2f}%")
        print(f"  PT hits:              {pt_exits}  ({pt_exits/n_trades:.0%})" if n_trades else "  PT hits: 0")
        print(f"  SL hits:              {sl_exits}  ({sl_exits/n_trades:.0%})" if n_trades else "  SL hits: 0")
        print(f"  Time exits:           {tb_exits}  ({tb_exits/n_trades:.0%})" if n_trades else "  Time exits: 0")
        print(f"\n📊 ML GATE DIAGNOSTICS")
        print(f"{'─'*60}")
        print(f"  Avg stocks passing ML gate per day: {avg_pass:.1f}/10")
        if avg_pass < 1.5:
            print(f"  ⚠️  LOW PASS RATE — cash drag likely. Consider lowering threshold to 0.52.")
        elif avg_pass > 4.0:
            print(f"  ⚠️  HIGH PASS RATE — ML gate may not be filtering enough. Consider raising to 0.58.")
        else:
            print(f"  ✅ Pass rate healthy.")
        print(f"{'='*60}")

        if port_cagr > nifty_cagr:
            print("✅ COMBINED SYSTEM OUTPERFORMS MARKET")
        else:
            print("❌ UNDERPERFORMED MARKET")
            print("   → Run pure ranking (remove ML gate) to compare:")
            print(f"   → If pure ranking outperforms, ML gate is hurting. Lower threshold or remove.")

        # Save equity curve chart
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
            p_series = pd.Series(portfolio_curve, index=list(all_dates) + [end_ts])
            n_series = pd.Series(nifty_curve,     index=list(all_dates) + [end_ts])

            ax1.plot(p_series.index, (p_series - 1) * 100,
                     color='#339af0', linewidth=2.5, label='MARK5 Ranking+ML')
            ax1.plot(n_series.index, (n_series - 1) * 100,
                     color='gray', linewidth=2, linestyle='--', label='NIFTY50')
            ax1.fill_between(p_series.index,
                             (p_series - 1) * 100, (n_series - 1) * 100,
                             where=(p_series > n_series), alpha=0.2, color='green')
            ax1.fill_between(p_series.index,
                             (p_series - 1) * 100, (n_series - 1) * 100,
                             where=(p_series <= n_series), alpha=0.2, color='red')
            ax1.set_title(f'MARK5 Ranking+ML vs NIFTY50 | {START_DATE} → {END_DATE}',
                          fontsize=13, fontweight='bold')
            ax1.set_ylabel('Cumulative Return (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', alpha=0.2)

            if trade_log:
                trade_df = pd.DataFrame(trade_log)
                colors_bar = ['#51cf66' if r > 0 else '#ff6b6b'
                              for r in trade_df['pnl_pct']]
                ax2.bar(range(len(trade_df)), trade_df['pnl_pct'] * 100,
                        color=colors_bar, alpha=0.7)
                ax2.axhline(y=0, color='black', alpha=0.5)
                ax2.set_title('Individual Trade P&L (%)', fontweight='bold')
                ax2.set_ylabel('P&L (%)')
                ax2.set_xlabel('Trade #')
                ax2.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            chart_path = os.path.join(OUTPUT_DIR, 'portfolio_equity_curve.png')
            plt.savefig(chart_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"\n📊 Chart saved: {chart_path}")
        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")

        return
    # ── [/END CROSS-SECTIONAL] ───────────────────────────────────────
    
    # Step 2: Initialize MARK5 Engine
    print("[STEP 2/5] Initializing MARK5 Prediction Engine...")
    engine = MARK5ValidationEngine(
        optimize=getattr(args, 'optimize', False),
        skip_training=getattr(args, 'skip_training', False)
    )
    print("✅ Engine initialized\n")
    
    # Step 3: Generate predictions and analyze
    print("[STEP 3/5] Training Models & Generating Predictions...")
    all_metrics = {}
    all_equity_curves = {}
    stock_times = []
    
    for i, (symbol, df) in enumerate(data.items(), 1):
        try:
            stock_start = time.time()
            
            # Progress with ETA
            if stock_times:
                avg_time = np.mean(stock_times)
                remaining = (len(data) - i + 1) * avg_time
                eta_str = f" (ETA: {remaining/60:.0f}m)"
            else:
                eta_str = ""
            
            print(f"\n{'─' * 60}")
            print(f"[{i}/{len(data)}] Processing: {symbol}{eta_str}")
            print(f"{'─' * 60}")
            
            # Generate predictions
            df_predictions, trade_log_sym = engine.generate_predictions(symbol, df)

            if df_predictions.empty:
                logger.warning(f"⚠️ No predictions for {symbol}")
                stock_times.append(time.time() - stock_start)
                continue
            
            # Log results
            engine.log_validation_results(symbol, df_predictions, {})
            
            # Calculate metrics (STRICTLY ON TEST SET)
            test_results = df_predictions[df_predictions['dataset_type'] == 'Test']
            if test_results.empty:
                 logger.warning(f"⚠️ Test set empty for {symbol}")
                 stock_times.append(time.time() - stock_start)
                 continue
                 
            analyzer = PerformanceAnalyzer()
            metrics = analyzer.calculate_metrics(test_results)
            all_metrics[symbol] = metrics
            
            # Save equity curve for portfolio report
            if 'strategy_return_net' in test_results.columns:
                cum_ret = (1 + test_results['strategy_return_net']).cumprod() - 1
                all_equity_curves[symbol] = cum_ret
            
            # Print key metrics
            elapsed = time.time() - stock_start
            stock_times.append(elapsed)
            
            print(f"\n📊 {symbol.replace('.NS', '')} ({elapsed:.0f}s):")
            if metrics.get('brier_score') is not None: print(f"   Brier Score: {metrics['brier_score']:.4f}")
            if 'accuracy' in metrics: print(f"   PT Hit Rate: {metrics['accuracy']:.1%}")
            if 'sharpe_ratio' in metrics: print(f"   Sharpe: {metrics['sharpe_ratio']:.2f} (trades-only: {metrics.get('sharpe_trades_only', 0):.2f})")
            if 'total_return_strategy' in metrics: print(f"   Return: {metrics['total_return_strategy']:.2%}")
            if 'win_rate' in metrics: print(f"   Win Rate: {metrics['win_rate']:.1%}")
            if 'max_drawdown' in metrics: print(f"   Max DD: {metrics['max_drawdown']:.2%}")
            
            # Create per-stock visualizations
            visualizer = AdvancedVisualizer()
            visualizer.create_comprehensive_report(symbol, df_predictions, metrics, trade_log_sym)
            
        except Exception as e:
            logger.error(f"❌ Failed to process {symbol}: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            stock_times.append(time.time() - stock_start if 'stock_start' in dir() else 0)
    
    print(f"\n{'═' * 70}")
    
    # Step 4: Create summary
    print("\n[STEP 4/5] Creating Summary Report...")
    if all_metrics:
        create_summary_report(all_metrics)
        print("✅ Summary report created")
    
    # Step 5: Create portfolio reports
    print("\n[STEP 5/5] Creating Portfolio Reports...")
    if all_metrics:
        create_portfolio_report(all_metrics, all_equity_curves)
        print("✅ Portfolio reports created")
    
    # Final summary
    total_time = time.time() - pipeline_start
    print(f"\n{'═' * 70}")
    print("✅ VALIDATION PIPELINE COMPLETE")
    print(f"{'═' * 70}")
    print(f"\n⏱️  Total Time: {total_time/60:.1f} minutes")
    print(f"📂 Reports Location: {OUTPUT_DIR}")
    print(f"\n📁 Validation Artifacts Saved to: {OUTPUT_DIR}/")
    print(f"   • Per-Stock Charts:       {CHARTS_DIR}/")
    print(f"   • Data Quality Report:     {OUTPUT_DIR}/data_quality_report.csv")
    print(f"   • Performance Summary:     {OUTPUT_DIR}/validation_summary.csv")
    print(f"   • Cross-Stock Heatmap:     {OUTPUT_DIR}/cross_stock_heatmap.png")
    print(f"   • Portfolio Dashboard:     {OUTPUT_DIR}/portfolio_dashboard.png")
    print(f"   • Portfolio Equity Curve:  {OUTPUT_DIR}/portfolio_equity_curve.png")
    print(f"   • Sector Performance:      {OUTPUT_DIR}/sector_performance.png")
    print(f"   • Stock Rankings:          {OUTPUT_DIR}/stock_rankings.png")
    print(f"\n📈 Stocks Analyzed: {len(all_metrics)}/{len(WATCHLIST)}")
    
    if all_metrics:
        briers = [m['brier_score'] for m in all_metrics.values() if m.get('brier_score') is not None]
        accuracies = [m.get('accuracy', 0) for m in all_metrics.values()]
        sharpes = [m.get('sharpe_ratio', 0) for m in all_metrics.values()]
        returns = [m.get('total_return_strategy', 0) for m in all_metrics.values()]
        
        if briers:
            print(f"\n🧠 Avg Brier:     {np.mean(briers):.4f}")
        else:
            print(f"\n🧠 Avg Brier:     N/A (No Trades)")
        print(f"🎯 Avg PT Hit:    {np.mean(accuracies):.1%}")
        sharpe_trades = [m.get('sharpe_trades_only', 0) for m in all_metrics.values()]
        print(f"📊 Avg Sharpe:    {np.mean(sharpes):.2f} (trades-only: {np.mean(sharpe_trades):.2f})")
        print(f"💰 Avg Return:    {np.mean(returns):.2%}")
    
    print(f"\n{'═' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MARK5 Validation Pipeline v3.0')
    parser.add_argument('--symbol', type=str, help='Stock symbol(s) (comma separated)', default=None)
    parser.add_argument('--preset', type=str, 
                        choices=['default', 'nifty50', 'top10', 'test3', 'alpha', 'mixed150'],
                        help='Use predefined watchlist', default=None)
    parser.add_argument('--start', type=str, help='Start date YYYY-MM-DD', default=None)
    parser.add_argument('--end', type=str, help='End date YYYY-MM-DD', default=None)
    parser.add_argument('--horizon', type=str, choices=['1_month', '3_months', '6_months', '1_year', 'max'], default='max', help='Training data horizon')
    parser.add_argument('--initial_capital', type=float, help='Initial capital', default=10000.0)
    parser.add_argument('--no-cache', action='store_true', help='Force model retraining (keeps data cache)')
    parser.add_argument('--force-download', action='store_true', help='Force re-download data (clears data cache)')
    parser.add_argument('--optimize', action='store_true', help='Run Optuna hyperparameter optimization before training')
    parser.add_argument('--skip-training', action='store_true', help='Skip training and use existing models')
    parser.add_argument('--per-stock', action='store_true', help='Run legacy per-stock ML simulation instead of cross-sectional ranker')
    
    args = parser.parse_args()
    
    # Update global configuration based on horizon
    if args.horizon:
        horizon_days = {
            '1_month': 30, '3_months': 90, 
            '6_months': 180, '1_year': 365, 'max': 2000
        }
        start_date_obj = end_date_obj - timedelta(days=horizon_days[args.horizon])
        START_DATE = start_date_obj.strftime('%Y-%m-%d')
        OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'reports', args.horizon, f'validation_{end_date_obj.year}')
        CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
        for directory in [OUTPUT_DIR, CHARTS_DIR, DATA_CACHE_DIR]:
            os.makedirs(directory, exist_ok=True)
        print(f"🔧 Horizon: {args.horizon} ({horizon_days[args.horizon]} days)")
    
    if args.symbol:
        WATCHLIST = [s.strip() for s in args.symbol.split(',')]
        print(f"🔧 Custom Watchlist: {WATCHLIST}")
    elif args.preset == 'alpha':
        WATCHLIST = MARK5_ALPHA
        print(f"🔧 MARK5 Alpha Portfolio: {WATCHLIST}")
    elif args.preset == 'nifty50':
        WATCHLIST = NIFTY_50_TICKERS
        print(f"🔧 Nifty 50 ({len(WATCHLIST)} stocks)")
    elif args.preset == 'top10':
        WATCHLIST = NIFTY_TOP10
        print(f"🔧 Nifty Top 10: {WATCHLIST}")
    elif args.preset == 'test3':
        WATCHLIST = NIFTY_TEST3
        print(f"🔧 Test 3: {WATCHLIST}")
    elif args.preset == 'mixed150':
        WATCHLIST = MIXED_150
        print(f"🔧 Mixed 150 Universe ({len(WATCHLIST)} stocks)")
    elif args.preset == 'default':
        try:
            from core.utils.constants import DEFAULT_WATCHLIST
            WATCHLIST = DEFAULT_WATCHLIST
        except ImportError:
            WATCHLIST = ['RELIANCE.NS']
        print(f"🔧 Default Watchlist ({len(WATCHLIST)} stocks)")
    
    if args.force_download:
        # Clear DATA cache only
        import shutil
        if os.path.exists(DATA_CACHE_DIR):
            shutil.rmtree(DATA_CACHE_DIR)
            os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        print("🔧 Data cache cleared — will re-download all data")

    if args.no_cache:
        # Clear MODEL artifacts only (keep downloaded data)
        import shutil
        model_dirs = ['models/', 'core/models/saved/']
        for md in model_dirs:
            if os.path.exists(md):
                shutil.rmtree(md)
                os.makedirs(md, exist_ok=True)
        print("🔧 Model cache cleared — will retrain all models (data cache preserved)")
        
    if args.start:
        START_DATE = args.start
        print(f"🔧 Start Date: {START_DATE}")
        
    if args.end:
        END_DATE = args.end
        print(f"🔧 End Date: {END_DATE}")

    main()
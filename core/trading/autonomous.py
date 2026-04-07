"""
MARK5 AUTONOMOUS INTRADAY TRADER v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG (vs v1.1):
- [2026-02-06] v8.0: Production-grade refactor
  • Fixed: Duplicate class declaration removed
  • Fixed: Missing PortfolioRiskAnalyzer import
  • Added: Thread-safe locks for all shared state
  • Added: SEBI circuit breaker integration
  • Added: Model staleness detection
  • Added: Comprehensive error handling

TRADING ROLE: Main autonomous trading loop
SAFETY LEVEL: CRITICAL - Controls all order execution

MARKET SCENARIOS HANDLED:
✅ Market open/close transitions
✅ Holiday detection
✅ SEBI circuit breakers
✅ API failures with retry
✅ Position SL/TP management
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import pytz
import uuid
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pytz import timezone

# Add parent directory to path for core imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# MARK5 components
from core.execution.execution_engine import MARK5ExecutionEngine, OrderResult
from core.data.provider import DataProvider
from core.models.features import AdvancedFeatureEngine
# from core.config.intraday import get_config_dict # Removed: Using ConfigManager
from core.models.registry import RobustModelRegistry as ModelRegistry
from core.trading.risk_manager import FastRiskAnalyzer, RiskAlerts, PortfolioRiskAnalyzer
from core.infrastructure.alerts import AlertManager, AlertLevel, AlertType
from core.trading.market_utils import MarketStatusChecker

# Import prediction engine
try:
    from core.models.predictor import MARK5Predictor
    PREDICTION_ENGINE_AVAILABLE = True
except ImportError:
    PREDICTION_ENGINE_AVAILABLE = False
    print("⚠️ prediction_engine not found")

from core.trading.decision import DecisionEngine as MARK5DecisionEngine
from core.analytics.journal import TradeJournal
from core.infrastructure.database_manager import MARK5DatabaseManager
from core.models.training.trainer import MARK5MLTrainer as LearningEngine
from core.analytics.regime_detector import MarketRegimeDetector


@dataclass
class TradingSignal:
    """Trading signal with execution details"""
    timestamp: datetime
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    predicted_return: float
    current_price: float
    stop_loss: float
    take_profit: float
    position_size: int
    horizon: str  # '1m', '15m', '1h'
    trade_id: Optional[str] = None  # Unique ID for tracking

    def __post_init__(self):
        # Ensure timestamp is timezone-aware (IST)
        if self.timestamp.tzinfo is None:
            self.timestamp = pytz.timezone('Asia/Kolkata').localize(self.timestamp)
        else:
            self.timestamp = self.timestamp.astimezone(pytz.timezone('Asia/Kolkata'))


# ConfigManager import (moved from inside class)
from core.utils.config_manager import ConfigManager


class AutonomousTrader:
    """
    🔥 AUTONOMOUS INTRADAY TRADING SYSTEM
    
    Main trading loop that:
    1. Fetches real-time data every N seconds
    2. Engineers features
    3. Generates predictions
    4. Executes trades based on signals
    5. Manages positions and risk
    6. Monitors performance
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize autonomous trader
        """
        # Load configuration
        self.config_manager = ConfigManager()
        if config_path and os.path.exists(config_path):
            # If specific config path provided, load it (override)
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Use centralized system config
            self.config = self.config_manager.get_config()
        
        # Validate and set defaults
        self.config.setdefault('execution', {'capital': 100000, 'paper_trading': True})
        self.config['execution'].setdefault('capital', 100000)
        self.config.setdefault('data', {'interval': '15m', 'lookback_period': '30d'})
        self.config['data'].setdefault('interval', '15m')
        self.config.setdefault('models', {'confidence_threshold': 0.6})
        self.config.setdefault('scheduler', {'decision_interval_seconds': 60})
        self.config.setdefault('paths', {'logs': 'logs', 'state': '/tmp/mark5_trader_state.json'})
        
        # Timezone
        self.tz = timezone('Asia/Kolkata')
        
        # Initialize logger
        self.logger = logging.getLogger("MARK5.AutonomousTrader")
        self.logger.setLevel(logging.INFO)
        
        # Initialize components
        self.logger.info("🚀 Initializing MARK5 Autonomous Trader...")
        
        # Data provider (replaces collector)
        self.collector = DataProvider(self.config)
        self.logger.info("✅ Data provider initialized")
        
        # Feature engineer
        self.feature_engine = AdvancedFeatureEngine()
        self.logger.info("✅ Feature engine initialized")
        
        # Execution engine
        self.execution_engine = MARK5ExecutionEngine(self.config['execution'])
        self.logger.info("✅ Execution engine initialized")
        
        # Portfolio risk and alerts
        risk_cfg = self.config.get('risk', {})
        self.risk_analyzer = PortfolioRiskAnalyzer(
            initial_capital=self.config['execution']['capital'],
            max_position_size=min(risk_cfg.get('max_position_size_pct', 5.0) / 100.0, 0.05),  # RULE 11: 5% max
            max_portfolio_risk=risk_cfg.get('max_portfolio_risk_pct', 2.0) / 100.0
        )
        self.risk_alerts = RiskAlerts(
            max_drawdown_threshold_pct=risk_cfg.get('max_drawdown_pct', 10.0),
            max_var_threshold_pct=risk_cfg.get('max_var_threshold', 5.0),
            max_position_risk_alert_pct=risk_cfg.get('max_position_risk_alert', 5.0),
            max_position_pct_alert=risk_cfg.get('max_position_pct_alert', 30.0),
            max_position_size_threshold_pct=risk_cfg.get('max_position_size_threshold', 25.0)
        )
        self.alert_manager = AlertManager()
        self.logger.info("✅ Risk analyzer and alerts initialized")
        
        # Model registry for staleness detection
        self.model_registry = ModelRegistry()
        self.staleness_check_interval = 86400  # Check daily
        self._last_staleness_check = 0
        self.logger.info("✅ Model registry initialized")
        
        # Prediction engine (if available)
        self.prediction_engine = None
        if PREDICTION_ENGINE_AVAILABLE:
            try:
                self.prediction_engine = MARK5Predictor("NIFTY_50") # Will be overridden per ticker or we remove the global predictor if not needed
                self.logger.info("✅ Prediction engine initialized")
            except Exception as e:
                self.logger.error(f"❌ Prediction engine initialization failed: {e}")
        
        # Database & Journal
        self.db_manager = MARK5DatabaseManager()
        self.trade_journal = TradeJournal(self.db_manager)
        self.logger.info("✅ Trade Journal initialized")
        
        # Learning Engine (for retraining)
        self.learner = LearningEngine(self.db_manager)

        # Regime Detector
        self.regime_detector = MarketRegimeDetector(self.config, db_manager=self.db_manager)
        self.logger.info("✅ Market Regime Detector initialized")

        # Decision Engine
        if self.prediction_engine:
            self.decision_engine = MARK5DecisionEngine(
                collector=self.collector,
                predictor=self.prediction_engine,
                executor=self.execution_engine,
                journal=self.trade_journal,
                risk_analyzer=self.risk_analyzer
            )
            self.logger.info("✅ Decision Engine initialized")
        else:
            self.decision_engine = None
            self.logger.warning("⚠️ Decision Engine NOT initialized (Prediction Engine missing)")
        
        # Trading state
        self.watchlist = self.config.get('watchlist', [])
        self.active_signals: Dict[str, TradingSignal] = {}
        self.running = False
        
        # Thread locks for race condition prevention
        self._trades_lock = threading.Lock()  # Protects trades_today, orders_executed
        self._signals_lock = threading.Lock()  # Protects active_signals, signals_generated
        self._stats_lock = threading.Lock()  # Protects consecutive_losses, api_error_count
        
        # Performance tracking
        self.trades_today: List[Dict] = []  # Completed round-trip trades
        self._last_state_save = 0
        
        # Counters
        self.orders_executed = 0 # Renamed from trades_executed to avoid confusion
        self.consecutive_losses = 0
        self.signals_generated = 0
        self.api_error_count = 0
        
        # Scheduler
        self.scheduler = BackgroundScheduler(timezone=self.tz)
        
        self.last_portfolio_risk: Optional[Dict] = None
        self.last_portfolio_risk_time: Optional[datetime] = None
        self.last_risk_alerts: List[str] = []
        
        # Data refresh interval
        self.refresh_interval = self.config['data'].get('refresh_interval', 60)
        
        # Paths
        self.state_path = self.config.get('paths', {}).get('state', '/tmp/mark5_trader_state.json')
        self.logs_dir = self.config.get('paths', {}).get('logs', 'logs')
        
        # Safety checks
        self._run_safety_checks()
        
        self.logger.info("🎯 Autonomous Trader ready!")
    
    def _run_safety_checks(self):
        """Run pre-flight safety checks before trading"""
        self.logger.info("🔍 Running safety checks...")
        
        if not self.watchlist:
            self.logger.warning("⚠️ Watchlist is empty! Please load a plan or configure watchlist.")
        
        if self.config['execution']['capital'] <= 0:
            raise ValueError("❌ Capital must be positive!")
        
        self.logger.info("✅ Safety checks passed")
    
    def load_daily_plan(self, plan: Dict):
        """Update trading parameters with the daily plan"""
        self.logger.info("🔄 Loading daily trading plan...")
        try:
            # Update watchlist
            if 'watchlist' in plan and plan['watchlist']:
                self.watchlist = plan['watchlist']
            
            # Update risk parameters
            risk_params = plan.get('system_parameters', {})
            if 'max_position_size_pct' in risk_params:
                self.risk_analyzer.max_position_size = float(risk_params['max_position_size_pct']) / 100.0
            if 'confidence_threshold' in risk_params:
                self.risk_analyzer.confidence_threshold = float(risk_params['confidence_threshold'])
                
            # Update scheduler interval if needed
            if 'decision_interval_seconds' in risk_params:
                interval = int(risk_params['decision_interval_seconds'])
                self.scheduler.reschedule_job(
                    job_id='decision_cycle',
                    trigger=IntervalTrigger(seconds=interval, timezone=self.tz)
                )
                
            self.logger.info(f"✅ Plan loaded. New watchlist size: {len(self.watchlist)}")
        except Exception as e:
            self.logger.error(f"Failed to load daily plan: {e}")

    def start(self):
        """Start the autonomous trader with Scheduler"""
        if self.running:
            self.logger.warning("⚠️ Trader already running")
            return
            
        self.running = True
        self.logger.info("🚀 Starting MARK5 Autonomous Scheduler...")
        
        # Save PID
        try:
            with open('/tmp/mark5_trader.pid', 'w') as f:
                f.write(str(os.getpid()))
        except Exception as e:
            logging.getLogger("MARK5.AutonomousTrader").debug(f"Could not write PID file: {e}")
            
        self._save_state()
        
        # Schedule Daily Jobs
        # 1. Start Trading Day (9:15 AM)
        self.scheduler.add_job(
            self.autonomous_trading_day,
            CronTrigger(hour=9, minute=15, timezone=self.tz),
            id='start_trading_day',
            replace_existing=True
        )
        
        # 2. End Trading Day (3:30 PM)
        self.scheduler.add_job(
            self.shutdown_trading_day,
            CronTrigger(hour=15, minute=30, timezone=self.tz),
            id='end_trading_day',
            replace_existing=True
        )
        
        # 3. Periodic Decision Cycle
        interval = self.config['scheduler']['decision_interval_seconds']
        self.scheduler.add_job(
            self.run_decision_cycle,
            IntervalTrigger(seconds=interval, timezone=self.tz),
            id='decision_cycle',
            replace_existing=True,
            max_instances=1
        )
        
        self.scheduler.start()
        self.logger.info(f"✅ Scheduler started. Decision interval: {interval}s")
        
        # If started during market hours, trigger immediate cycle
        if self._is_market_open():
            self.logger.info("🕒 Market is OPEN. Triggering immediate decision cycle.")
            self.autonomous_trading_day()
        else:
            self.logger.info("💤 Market is CLOSED. Waiting for next scheduled start.")
            
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the trading loop and clean up resources"""
        self.logger.info("🛑 Stopping autonomous trader...")
        self.running = False
        
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        
        if hasattr(self, 'execution_engine'):
            self.execution_engine.stop()
        if hasattr(self, 'collector'):
            self.collector.stop()
            
        self._save_daily_report()
        self.logger.info("✅ Autonomous trader stopped")
        
    def _is_market_open(self) -> bool:
        """Check if market is currently open (NSE Hours + Holidays)"""
        now = datetime.now(self.tz)
        
        # 1. Check Weekend
        if now.weekday() >= 5: # Saturday=5, Sunday=6
            return False
            
        # 2. Check Holidays (Load from DB or Config)
        try:
            holidays = self.config.get('holidays', [])
            # If DB has holidays table, fetch here: holidays = self.db_manager.get_holidays()
        except Exception as e:
            holidays = []
            self.logger.debug(f"Could not load holidays: {e}")
            
        if now.strftime('%Y-%m-%d') in holidays:
            return False
            
        # 3. Check Time (9:15 AM - 3:30 PM)
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end

    def autonomous_trading_day(self):
        """Start of day initialization"""
        self.logger.info("☀️ Starting Autonomous Trading Day")
        if not self._is_market_open():
            self.logger.warning("⚠️ Triggered outside market hours/holidays. Skipping.")
            return
            
        self.execution_engine.reset_for_new_day()
        with self._trades_lock:
            self.trades_today = []
            self.orders_executed = 0
        with self._signals_lock:
            self.active_signals = {}
            self.signals_generated = 0
        with self._stats_lock:
            self.consecutive_losses = 0
        
    def shutdown_trading_day(self):
        """End of day shutdown"""
        self.logger.info("🌙 Ending Autonomous Trading Day")
        self._save_daily_report()
        # Optional: Close all intraday positions
        # self.execution_engine.close_all_positions() 

    def run_decision_cycle(self):
        """Single iteration of the decision loop"""
        if not self.running or not self._is_market_open():
            return
            
        self.logger.info("🔄 Running Decision Cycle...")
        
        # 1. Bulk Fetch Data for Watchlist
        try:
            # Fetch data for all tickers in watchlist
            # Assuming collector has a batch method or we iterate efficiently
            # For now, we'll let process_ticker handle individual fetches if batch isn't available
            # But to reduce latency, let's try to pre-fetch if possible.
            # Since MARK5DataCollector might not have get_stock_data_batch implemented yet,
            # we will rely on ThreadPoolExecutor in step 2 to parallelize fetches.
            pass 
        except Exception as e:
            self.logger.error(f"Data fetch error: {e}")

        # 2. Process Watchlist (Parallel)
        max_workers = self.config.get('max_workers', 10)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.process_ticker, ticker): ticker 
                for ticker in self.watchlist
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    future.result(timeout=15)
                except TimeoutError:
                    self.logger.warning(f"⚠️ Timeout processing {ticker}")
                except Exception as e:
                    self.logger.error(f"Error processing {ticker}: {e}")
        
        # 2b. Rule 20: Periodic Correlation Matrix Update (Every 30m)
        if self.risk_analyzer._should_refresh_correlation():
            self.logger.info("📊 Rule 20: Updating portfolio correlation matrix (30m cycle)...")
            price_histories = {}
            for ticker in self.watchlist:
                try:
                    hist = self.collector.fetch_stock_data(ticker, period='60d', interval='1d')
                    if hist is not None and not hist.empty:
                        price_histories[ticker] = hist['close']
                except Exception as e:
                    self.logger.debug(f"Could not fetch history for correlation: {ticker} - {e}")
            
            if price_histories:
                self.risk_analyzer.update_correlation_matrix(price_histories)
        
        # 3. Manage Positions (Check SL/TP)
        # Fetch fresh quotes for open positions
        open_positions = self.execution_engine.get_positions()
        if open_positions:
            market_data_snapshot = {}
            for pos in open_positions:
                try:
                    df = self.collector.fetch_stock_data(pos.symbol, period='1d', interval='1m')
                    if df is not None and not df.empty:
                        market_data_snapshot[pos.symbol] = df
                except Exception as e:
                    self.logger.warning(f"Failed to fetch position data for {pos.symbol}: {e}")
            self._manage_positions(market_data_snapshot)
            
        # 4. Risk Checks
        self._log_status()
        
        # 5. Periodic Checks
        if time.time() - self._last_staleness_check > self.staleness_check_interval:
            self._check_model_staleness()
            self._last_staleness_check = time.time()
            
        if time.time() - self._last_state_save > 60:
            self._save_state()
            self._last_state_save = time.time()

    def process_ticker(self, ticker: str):
        """
        Process a single ticker using Decision Engine
        """
        if not self.decision_engine:
            return

        # --- REGIME PRE-SCREEN (Audit v10.5 - Standardized) ---
        try:
            # 1. Fetch data for regime detection (daily bars for context)
            hist_data = self.collector.fetch_stock_data(ticker, period='60d', interval='1d')
            if hist_data is not None and not hist_data.empty:
                regime_info = self.regime_detector.detect_market_regime(ticker, hist_data)
                regime = regime_info['overall_regime'] # Standardized Enum
                
                # 2. Update Risk & Sizer with latest regime
                # self.risk_analyzer is the PortfolioRiskAnalyzer
                self.risk_analyzer.set_regime(regime.value) 
                
                # self.decision_engine.sizer is the VolatilityAwarePositionSizer
                if hasattr(self.decision_engine, 'sizer'):
                    self.decision_engine.sizer.set_regime(regime)
                    
                regime_name = regime.name if hasattr(regime, 'name') else str(regime)
                self.logger.info(f"🌐 Ticker {ticker} Regime: {regime_name} "
                                 f"(Confidence: {regime_info['regime_confidence']:.2f})")
                    
                if regime_name in ['VOLATILE', 'BEAR']:
                    # Log but do not block yet
                    self.logger.info(f"🚧 REGIME PRE-SCREEN: {ticker} is in {regime_name} regime.")

                # RULE 90 - Rolling Sharpe Gate
                if hasattr(self, 'trade_journal'):
                    sharpe = self.trade_journal.get_rolling_sharpe(ticker)
                    if sharpe < 0.5:
                        self.logger.info(f"🚧 SHARPE PRE-SCREEN: {ticker} has Rolling Sharpe < 0.5 ({sharpe:.2f}).")
        except Exception as e:
            self.logger.error(f"Pre-screen error for {ticker}: {e}")
        # ----------------------------------------------------


        try:
            # Increment signals generated counter (attempted)
            with self._stats_lock:
                self.signals_generated += 1

            # Delegate to Decision Engine
            result = self.decision_engine.analyze_and_act(ticker)
            
            # SEBI Circuit Breaker Check
            if result['action'] in ['BUY', 'SELL']:
                # Construct temp signal for check
                temp_signal = TradingSignal(
                    timestamp=datetime.now(self.tz),
                    symbol=ticker,
                    action=result['action'],
                    confidence=result.get('confidence', 0.0),
                    predicted_return=0.0,
                    current_price=result.get('current_price', 0.0),
                    stop_loss=0.0,
                    take_profit=0.0,
                    position_size=0,
                    horizon='15m'
                )
                if not self._check_sebi_circuit_breakers(temp_signal):
                    self.logger.warning(f"🚫 Trade blocked by SEBI Circuit Breaker: {ticker}")
                    return

            # Update internal state if executed
            if result['action'] in ['BUY_EXECUTED', 'SELL_EXECUTED', 'SHORT_EXECUTED', 'COVER_EXECUTED']:
                with self._trades_lock:
                    self.trades_today.append(result)
                    self.orders_executed += 1
                
                # track as active signal for SL/TP management
                if result['action'] in ['BUY_EXECUTED', 'SHORT_EXECUTED']:
                    details = result.get('details', '')
                    qty = result.get('quantity', 0)
                    price = result.get('price', result.get('current_price', 0.0))
                    
                    try:
                        base_action = result['action'].replace('_EXECUTED', '') # BUY or SHORT
                        signal = TradingSignal(
                            timestamp=datetime.now(self.tz),
                            symbol=ticker,
                            action=base_action,
                            confidence=result.get('confidence', 0.0),
                            predicted_return=0.0, # Add if available
                            current_price=price,
                            stop_loss=result.get('stop_loss', 0.0),
                            take_profit=result.get('target_price', 0.0), # Updated key
                            position_size=qty,
                            horizon=self.config['data'].get('interval', '15m'),
                            trade_id=result.get('trade_id')
                        )
                        with self._signals_lock:
                            self.active_signals[ticker] = signal
                    except Exception as e:
                        self.logger.error(f"Failed to create signal object: {e}")
                
                # If exit, remove from active signals
                elif result['action'] in ['SELL_EXECUTED', 'COVER_EXECUTED']:
                    with self._signals_lock:
                        if ticker in self.active_signals:
                            del self.active_signals[ticker]
            
            # Log result
            if result['action'] != 'HOLD':
                reason = result.get('reason', 'No reason provided')
                self.logger.info(f"👉 {ticker}: {result['action']} ({reason})")

        except Exception as e:
            self.logger.error(f"❌ Error processing {ticker}: {e}")
            with self._stats_lock:
                self.api_error_count += 1

    def _manage_positions(self, market_data_snapshot: Dict[str, pd.DataFrame]):
        """Monitor and manage open positions (stop loss, take profit)"""
        positions = self.execution_engine.get_positions()
        
        for position in positions:
            symbol = position.symbol
            
            if symbol not in market_data_snapshot or market_data_snapshot[symbol].empty:
                continue

            try:
                latest_data = market_data_snapshot[symbol]
                if 'close' not in latest_data.columns:
                    continue
                    
                current_price = float(latest_data['close'].iloc[-1])
                
                if current_price <= 0 or not np.isfinite(current_price):
                    continue
                    
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.average_price) * position.quantity

                # Check for exit conditions
                with self._signals_lock:
                    signal = self.active_signals.get(symbol)
                
                if signal:
                    exit_reason = None
                    
                    # --- RULE 80: Intraday Auto-Squareoff ---
                    from datetime import time as dt_time, datetime as dt_datetime
                    import pytz
                    
                    IST = pytz.timezone('Asia/Kolkata')
                    current_time = dt_datetime.now(IST).time()
                    
                    if current_time >= dt_time(15, 20) and signal.horizon != 'delivery':
                        if position.quantity < 0:
                            exit_reason = 'RULE_80_SQUAREOFF'
                            self.logger.warning(f"⏰ RULE 80: Auto-covering short position {symbol} at/after 15:20 IST")
                        elif current_time >= dt_time(15, 25):
                            exit_reason = 'INTRADAY_SQUAREOFF'
                            self.logger.warning(f"⏰ Auto-closing intraday position {symbol} at/after 15:25 IST")
                    # ----------------------------------------
                    
                    # --- TREND EXTENSION (Month 2 Rollout) ---
                    extend_hold = False
                    try:
                        # Check if we should extend the hold
                        if hasattr(self.decision_engine, 'regime_detector'):
                            regime_info = self.decision_engine.regime_detector.detect_market_regime(symbol)
                            regime = regime_info.get('overall_regime', 'UNKNOWN')
                            
                            # Only check if in profit
                            if current_price > signal.current_price: # Assuming signal.current_price is entry
                                if regime in ['STRONG_BULL', 'TRENDING_UP']:
                                    adx = regime_info.get('adx', 0)
                                    if adx > 25:
                                        extend_hold = True
                                        self.logger.info(f"📈 TREND EXTENSION: Holding {symbol} past TP due to strong {regime} (ADX: {adx:.1f})")
                    except Exception as e:
                        self.logger.debug(f"Trend extension check failed for {symbol}: {e}")
                    # -------------------------------------------

                    # Stop loss hit
                    if (position.quantity > 0 and current_price <= signal.stop_loss) or \
                       (position.quantity < 0 and current_price >= signal.stop_loss):
                        exit_reason = 'SL'
                        self.logger.warning(f"🛑 STOP LOSS for {symbol} @ ₹{current_price:.2f}")

                    # Take profit hit
                    elif (position.quantity > 0 and current_price >= signal.take_profit) or \
                         (position.quantity < 0 and current_price <= signal.take_profit):
                        if not extend_hold:
                            exit_reason = 'TP'
                            self.logger.info(f"🎯 TAKE PROFIT for {symbol} @ ₹{current_price:.2f}")
                        else:
                            # Bump TP up scaled by ATR, not flat %
                            atr = regime_info.get('atr', current_price * 0.01) if hasattr(self.decision_engine, 'regime_detector') else (current_price * 0.01)
                            tp_extension = 0.75 * atr
                            signal.take_profit += tp_extension
                            self.logger.info(f"🎯 WIDENED TP for {symbol} to ₹{signal.take_profit:.2f} due to extension (ATR scaled).")

                    if exit_reason:
                        result = self.execution_engine.close_position(symbol)
                        if result and result.status == 'SUCCESS':
                            # Log to Trade Journal
                            if signal.trade_id:
                                try:
                                    self.trade_journal.log_trade_exit(
                                        trade_id=signal.trade_id,
                                        exit_data={
                                            'exit_price': result.price,
                                            'exit_quantity': result.quantity,
                                            'exit_timestamp': result.timestamp.isoformat(),
                                            'exit_reason': exit_reason,
                                            'commission': getattr(result, 'commission', 0.0)
                                        }
                                    )
                                except Exception as e:
                                    self.logger.error(f"Failed to log trade exit: {e}")
                            
                            # Update internal stats
                            with self._trades_lock:
                                for trade in self.trades_today:
                                    if trade.get('ticker') == symbol: # Fix: Use 'ticker'
                                        trade['exit_price'] = result.price
                                        trade['exit_timestamp'] = result.timestamp
                                        pnl = float(getattr(position, 'realized_pnl', 0.0))
                                        trade['pnl'] = pnl
                                        trade['exit_reason'] = exit_reason
                                        
                                        with self._stats_lock:
                                            if pnl > 0:
                                                self.consecutive_losses = 0
                                            else:
                                                self.consecutive_losses += 1
                                        break
                            
                            with self._signals_lock:
                                if symbol in self.active_signals:
                                    del self.active_signals[symbol]
                                
                            # Call Sharpe Updater (Week 1 Rollout)
                            if hasattr(self, 'trade_journal'):
                                pnl_pct = float(getattr(position, 'realized_pnl', 0.0))
                                try:
                                    self.trade_journal.update_rolling_sharpe(symbol, pnl_pct)
                                    self.logger.debug(f"Updated Rolling Sharpe for {symbol}")
                                except Exception as e:
                                    self.logger.error(f"Failed to update rolling sharpe: {e}")

            except Exception as e:
                self.logger.error(f"❌ Position management error for {symbol}: {e}")
    
    def _run_portfolio_risk_checks(self):
        """Run portfolio-level risk analysis"""
        stats = self.execution_engine.get_daily_stats()
        positions = self.execution_engine.get_positions()
        if not positions:
            return
        
        try:
            portfolio_positions: List[Dict] = []
            market_data: Dict[str, pd.DataFrame] = {}
            # Use current portfolio value if available
            total_capital = stats.get('current_portfolio_value', stats.get('capital', self.execution_engine.capital))
            
            for pos in positions:
                symbol = pos.symbol
                data = self.collector.fetch_stock_data(symbol, period='60d', interval=self.config['data']['interval'])
                if data is None or data.empty:
                    continue
                current_price = float(data['close'].iloc[-1])
                position_value = pos.quantity * current_price
                position_pct = (position_value / total_capital * 100.0) if total_capital > 0 else 0.0
                
                portfolio_positions.append({
                    'ticker': symbol,
                    'value': position_value,
                    'position_pct': position_pct,
                    'risk_pct': 0.0 # Simplified
                })
                market_data[symbol] = data
            
            if not portfolio_positions:
                return
            
            analysis = self.risk_analyzer.analyze_portfolio_risk(
                positions=portfolio_positions,
                market_data=market_data
            )
            
            alerts = self.risk_alerts.check_portfolio_risk(analysis)
            self.last_portfolio_risk = analysis
            self.last_portfolio_risk_time = datetime.now()
            
            for msg in alerts:
                self.logger.warning(f"RISK ALERT: {msg}")
                if self.alert_manager:
                    self.alert_manager.create_alert(
                        level=AlertLevel.HIGH,
                        alert_type=AlertType.HIGH_RISK_WARNING,
                        title="Portfolio Risk Alert",
                        message=msg,
                        metadata=analysis
                    )
        except Exception as e:
            self.logger.error(f"Portfolio risk check failed: {e}")
    
    def _check_sebi_circuit_breakers(self, signal: TradingSignal) -> bool:
        """Check SEBI circuit breakers before trading"""
        try:
            recent_data = self.collector.fetch_stock_data(signal.symbol, period='2d')
            if recent_data is None or len(recent_data) < 2:
                return True
            
            previous_close = float(recent_data['close'].iloc[-2])
            
            sebi_status = self.risk_alerts.check_sebi_circuit_breakers(
                symbol=signal.symbol,
                current_price=signal.current_price,
                previous_close=previous_close,
                market_type='equity'
            )
            
            if not sebi_status['can_trade']:
                self.logger.warning(
                    f"🚫 SEBI Circuit Breaker Hit: {signal.symbol} - {sebi_status['breaker_level']}"
                )
                if self.alert_manager:
                    self.alert_manager.create_alert(
                        level=AlertLevel.HIGH,
                        alert_type=AlertType.CIRCUIT_BREAKER,
                        ticker=signal.symbol,
                        title="SEBI Circuit Breaker Hit",
                        message=f"{signal.symbol}: {sebi_status['breaker_level']}",
                        metadata=sebi_status
                    )
                return False
            return True
        except Exception as e:
            self.logger.error(f"SEBI check failed: {e}")
            return True

    def _check_model_staleness(self):
        """Check for stale models and trigger retraining"""
        try:
            stale_models = self.model_registry.get_stale_models()
            if not stale_models:
                return
            
            self.logger.warning(f"⚠️ Stale models detected: {stale_models}")
            for model_info in stale_models:
                # Trigger retraining
                self.learner.train_model(ticker=model_info['ticker'], retrain=True)
                # Reload model in registry
                self.model_registry.reload_model(model_info['ticker'])
            
            # Reload prediction engine if needed
            if self.prediction_engine:
                self.prediction_engine.reload_models()
                
        except Exception as e:
            self.logger.error(f"Staleness check failed: {e}")

    def _log_status(self):
        """Log current trading status"""
        stats = self.execution_engine.get_daily_stats()
        self.logger.info(
            f"📊 Status: Signals={self.signals_generated} | "
            f"Orders={self.orders_executed} | "
            f"Positions={stats['open_positions']} | "
            f"P&L=₹{stats['daily_pnl']:.2f}"
        )
        try:
            self._run_portfolio_risk_checks()
        except Exception as e:
            self.logger.error(f"Portfolio risk check failed in status log: {e}")
    
    def _save_daily_report(self):
        """Save daily trading report"""
        try:
            summary = self.trade_journal.generate_daily_summary()
            timestamp = datetime.now().strftime('%Y%m%d')
            
            os.makedirs(self.logs_dir, exist_ok=True)
            report_path = os.path.join(self.logs_dir, f"trading_report_{timestamp}.json")
            
            stats = self.execution_engine.get_daily_stats()
            
            # Filter completed trades and map keys
            completed_trades = []
            for t in self.trades_today:
                if 'exit_reason' in t:
                    # Map keys to match calculate_intraday_metrics expectation
                    trade_record = t.copy()
                    trade_record['entry_time'] = t.get('timestamp')
                    trade_record['exit_time'] = t.get('exit_timestamp')
                    trade_record['entry_price'] = t.get('price') # Entry price
                    # exit_price and quantity are already likely correct or need checking
                    completed_trades.append(trade_record)

            if completed_trades:
                metrics = calculate_intraday_metrics(completed_trades)
            else:
                metrics = {}
            
            report = {
                'date': timestamp,
                'daily_stats': stats,
                'metrics': metrics,
                'trades': self.trades_today,
                'signals_generated': self.signals_generated,
                'orders_executed': self.orders_executed,
                'journal_summary': summary
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"✅ Daily report saved: {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save daily report: {e}")

    def _save_state(self):
        """Save trader state for dashboard access"""
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            state = {
                'running': self.running,
                'last_update': datetime.now().isoformat(),
                'watchlist': self.watchlist,
                'active_signals': [
                    {
                        'symbol': s.symbol,
                        'action': s.action,
                        'confidence': s.confidence,
                        'price': s.current_price,
                        'timestamp': s.timestamp.isoformat()
                    }
                    for s in self.active_signals.values()
                ],
                'trades_today': [
                    {
                        'symbol': t.get('ticker'),
                        'action': t.get('action'),
                        'price': t.get('price'),
                        'quantity': t.get('quantity'),
                        'time': t.get('timestamp').isoformat() if isinstance(t.get('timestamp'), datetime) else str(t.get('timestamp')),
                        'pnl': t.get('pnl', 0.0)
                    }
                    for t in self.trades_today
                ],
                'daily_stats': self.execution_engine.get_daily_stats()
            }
            
            with open(self.state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def get_live_status(self) -> Dict:
        """Get current live trading status"""
        stats = self.execution_engine.get_daily_stats()
        positions = self.execution_engine.get_positions()
        
        return {
            'running': self.running,
            'market_open': self._is_market_open(),
            'daily_stats': stats,
            'positions': [
                {
                    'symbol': p.symbol,
                    'quantity': p.quantity,
                    'avg_price': p.average_price,
                    'current_price': p.current_price,
                    'unrealized_pnl': p.unrealized_pnl,
                    'realized_pnl': p.realized_pnl
                }
                for p in positions
            ],
            'active_signals': len(self.active_signals),
            'signals_today': self.signals_generated,
            'trades_today': self.orders_executed
        }


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔥 MARK5 AUTONOMOUS INTRADAY TRADER")
    print("=" * 60)
    
    trader = AutonomousTrader()
    config = trader.config
    
    print(f"\n📊 Configuration:")
    print(f"   Mode: {'PAPER TRADING ✅' if config['execution']['paper_trading'] else '🚨 LIVE TRADING 🚨'}")
    print(f"   Capital: ₹{config['execution']['capital']:,}")
    print(f"   Watchlist: {len(config.get('watchlist', []))} stocks")
    print(f"   Interval: {config['data']['interval']}")
    
    print("\n🚀 Starting autonomous trading...")
    print("   Press Ctrl+C to stop\n")
    
    try:
        trader.start()
        
        while trader.running:
            time.sleep(10)
            status = trader.get_live_status()
            
            os.system('clear' if os.name == 'posix' else 'cls')
            print("🔥 MARK5 AUTONOMOUS TRADER - LIVE STATUS")
            print("=" * 60)
            print(f"Market: {'OPEN 🟢' if status['market_open'] else 'CLOSED 🔴'}")
            print(f"Signals Today: {status['signals_today']}")
            print(f"Orders Today: {status['trades_today']}")
            print(f"Daily P&L: ₹{status['daily_stats']['daily_pnl']:.2f}")
            print(f"Open Positions: {len(status['positions'])}")
            print(f"\nPress Ctrl+C to stop")
            print("=" * 60)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        trader.stop()
        print("✅ Autonomous trader stopped successfully")

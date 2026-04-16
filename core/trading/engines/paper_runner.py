#!/usr/bin/env python3
"""
MARK5 HEADLESS PAPER TRADING RUNNER v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Production hardening & standardized header

TRADING ROLE: Paper trading system launcher
SAFETY LEVEL: LOW - Testing environment only

FEATURES:
✅ Forces PAPER_TRADING=True
✅ Initializes database and data collector
✅ Starts Autonomous Trader
✅ Graceful signal handling (SIGINT/SIGTERM)
✅ Test mode with forced market open
"""

import os
import sys
import time
import logging
import signal
import json
from datetime import datetime
import pandas as pd

# DEBUG: Check trainer module
try:
    # Remove stale debug imports
    # import core.models.trainer
    # import core.data.collector
    from core.models.training.trainer import MARK5MLTrainer
    print(f"DEBUG: MARK5MLTrainer source: {MARK5MLTrainer.train_advanced_ensemble}")
except Exception as e:
    print(f"DEBUG: Failed to inspect trainer: {e}")

# Ensure core is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# from core.autonomous_trader import AutonomousTrader # Moved to main()
from core.utils.config_manager import ConfigManager

# Configure Logging
log_dir = os.path.join(current_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"paper_trading_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("MARK5.Runner")

def signal_handler(sig, frame):
    logger.info("🛑 Interrupt received, shutting down...")
    if 'trader' in globals() and trader:
        trader.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    global trader
    
    # 0. Parse Arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run MARK5 Paper Trading')
    parser.add_argument('--test-mode', action='store_true', help='Force market open for testing')
    args = parser.parse_args()
    
    logger.info("🚀 Starting MARK5 Paper Trading System...")
    
    # 1. Load & Override Config
    config_manager = ConfigManager()
    config = config_manager.get_config().model_dump() # Convert Pydantic to dict
    
    # FORCE PAPER TRADING SETTINGS
    if 'execution' not in config:
        config['execution'] = {}
    config['execution']['paper_trading'] = True
    config['execution']['capital'] = 100000  # Reset capital for testing
    
    # Force Daily Settings for Paper Trading (Matching recalibrated models)
    if 'data' not in config:
        config['data'] = {}
    config['data']['interval'] = '1d'
    config['data']['lookback_period'] = '600d'
    
    # Update watchlist for Yahoo Finance
    config['watchlist'] = [
        'COFORGE.NS', 'PERSISTENT.NS', 'KPITTECH.NS', 'HAL.NS', 'POLYCAB.NS'
    ]
    
    logger.info(f"📋 Watchlist: {config['watchlist']}")
    logger.info(f"💰 Initial Capital: ₹{config['execution']['capital']:,.2f}")
    
    # 2. Initialize Trader
    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            from datetime import time as dt_time
            if isinstance(obj, (datetime, dt_time)):
                return obj.isoformat()
            # Handle Pydantic SecretStr
            if hasattr(obj, 'get_secret_value'):
                return obj.get_secret_value()
            return super().default(obj)

    try:
        # Save temp config for this run
        temp_config_path = os.path.join(current_dir, "temp_paper_config.json")
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=4, cls=CustomJSONEncoder)
            
        # 🔥 TEST MODE: Monkey-patch MarketStatusChecker to force market open
        if args.test_mode:
            logger.warning("⚠️ TEST MODE ENABLED: Forcing Market Open Status")
            # Patch the utility class used by AutonomousTrader
            import core.trading.market_utils
            core.trading.market_utils.MarketStatusChecker.is_market_open = lambda self: True
            
        # 🔥 CRITICAL: Initialize System Container (BUG-C-09 FIX)
        from core.system.container import container
        from types import SimpleNamespace
        
        # Convert dict config to SimpleNamespace for dot-notation access
        def dict_to_sns(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_sns(v) for k, v in d.items()})
            return d
            
        sns_config = dict_to_sns(config)
        container.register('config', sns_config)
        container.register('oms', SimpleNamespace(executor=None))
        
        # Import here to ensure patches are applied BEFORE the class is loaded
        from core.trading.autonomous import AutonomousTrader
        
        # 🔥 TEST MODE: Force AutonomousTrader internal status
        if args.test_mode:
            AutonomousTrader._is_market_open = lambda self: True
            
        trader = AutonomousTrader(config_path=temp_config_path)
        logger.info(f"🔍 Trader Watchlist: {trader.watchlist}")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Force immediate decision cycle in test mode
        if args.test_mode:
            logger.info("🧪 Triggering immediate decision cycle for verification...")
            trader.running.set() # ensure flag is set
            trader.run_decision_cycle()
            logger.info("✅ Decision cycle complete")
        else:
            trader.start()
            
            # Use threading event to keep alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("🛑 Interrupt received, shutting down...")
                trader.stop()
            
    except Exception as e:
        logger.critical(f"❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'trader' in locals() and trader.running:
            trader.stop()
        logger.info("👋 Paper trading session ended")

if __name__ == "__main__":
    main()

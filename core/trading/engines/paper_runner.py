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
    import core.models.trainer
    import core.data.collector
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
    config = config_manager.get_config()
    
    # FORCE PAPER TRADING SETTINGS
    config['execution']['paper_trading'] = True
    config['execution']['capital'] = 100000  # Reset capital for testing
    
    # Force Intraday Settings for Paper Trading
    config['data']['interval'] = '15m'
    config['data']['lookback_period'] = '60d'  # Increased for training data requirements (>200 rows after feature engineering)
    
    # Update watchlist for Yahoo Finance (needs .NS suffix if not using Kite)
    # But Kite uses symbols without suffix. DataCollector handles fallback?
    # Let's use standard symbols and rely on DataCollector to handle suffix if needed, 
    # OR explicitly add .NS if we know we are using Yahoo fallback.
    # For paper trading without Kite, .NS is safer.
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
            return super().default(obj)

    try:
        # Save temp config for this run
        temp_config_path = os.path.join(current_dir, "temp_paper_config.json")
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=4, cls=CustomJSONEncoder)
            
        # 🔥 TEST MODE: Monkey-patch MarketStatusChecker to force market open
        if args.test_mode:
            logger.warning("⚠️ TEST MODE ENABLED: Forcing Market Open Status")
            # Patch the utility function used by AutonomousTrader
            import core.intraday_utils
            core.intraday_utils.is_market_open = lambda: True
            
        # Import here to ensure patches are applied BEFORE the class is loaded
        from core.autonomous_trader import AutonomousTrader
        trader = AutonomousTrader(config_path=temp_config_path)
        logger.info(f"🔍 Trader Watchlist: {trader.watchlist}")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        trader.start()
        
        # Keep main thread alive
        while trader.running:
            time.sleep(1)
            
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

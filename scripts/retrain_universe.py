#!/usr/bin/env python3
"""
MARK5 Universe Retraining Script
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Automates the retraining of all models in the `DEFAULT_WATCHLIST`
(NIFTY Midcap 150) using the MARK5 advanced ensemble trainer.

Usage:
    python3 scripts/retrain_universe.py
    python3 scripts/retrain_universe.py --years 3
"""

import argparse
import os
import sys

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.utils.constants import DEFAULT_WATCHLIST
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | MARK5.MidcapRetrain | %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Retrain all universe models")
    parser.add_argument('--years', type=float, default=3.0, help='Years of history')
    args = parser.parse_args()

    symbols = [s.replace('.NS', '') for s in DEFAULT_WATCHLIST]
    
    logger.info(f"={ '=' * 60 }=")
    logger.info(f"  INITIATING UNIVERSE RETRAINING ({len(symbols)} symbols)")
    logger.info(f"={ '=' * 60 }=")

    trainer_script = os.path.join(project_root, 'core', 'models', 'training', 'trainer.py')

    for symbol in symbols:
        logger.info(f"\n[{symbol}] Initiating CPCV Training Cycle...")
        cmd = [sys.executable, trainer_script, '--symbols', symbol, '--years', str(args.years)]
        
        try:
             # Using subprocess.run to stream output directly to terminal
             subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
             logger.error(f"[{symbol}] Training failed with return code {e.returncode}")
        except KeyboardInterrupt:
             logger.error(f"Training aborted by user.")
             break

    logger.info(f"\n={ '=' * 60 }=")
    logger.info(f"  UNIVERSE RETRAINING COMPLETE")
    logger.info(f"={ '=' * 60 }=")

if __name__ == '__main__':
    main()

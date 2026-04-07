#!/usr/bin/env python3
"""
MARK5 TCN UNIVERSE TRAINER
━━━━━━━━━━━━━━━━━━━━━━━━━━
Trains the VAJRA TCN model for each ticker in the universe to 
generate the 'Deep Learning Features' required by the v10.4 ensemble.
"""

import os, sys, logging, time
from pathlib import Path
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.models.tcn.pipeline import TCNPipelineOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("MARK5.TCN_Universe")

def main():
    CUTOFF_DATE = '2024-09-30'
    cache_dir = Path(PROJECT_ROOT) / "data" / "cache"
    # Use 60m data for TCN (as architected)
    parquets = sorted(cache_dir.glob("*_60m.parquet"))
    
    if not parquets:
        logger.error("No 60m parquet files found in data/cache/")
        return

    orchestrator = TCNPipelineOrchestrator(sequence_length=64)

    results = {"success": [], "failed": []}

    for i, ppath in enumerate(parquets, 1):
        ticker = ppath.stem.replace("_60m", "")
        # Standardize to .NS
        ticker_ns = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
        
        print(f"\n[{i}/{len(parquets)}] {ticker_ns}")
        
        save_path = Path(PROJECT_ROOT) / "models" / ticker_ns / "tcn_model"
        os.makedirs(save_path.parent, exist_ok=True)

        try:
            df = pd.read_parquet(ppath)
            df.columns = [c.lower() for c in df.columns]
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            
            # Blind training (OOS Integrity)
            df = df[df.index <= pd.Timestamp(CUTOFF_DATE)]
            
            if len(df) < 500: # TCN needs more data for stationarity
                logger.warning(f"  {ticker_ns}: insufficient data ({len(df)}) — skipping")
                results["failed"].append(ticker_ns)
                continue

            t0 = time.time()
            # Train the TCN model for this security
            orchestrator.train_production(df, str(save_path))
            
            elapsed = time.time() - t0
            logger.info(f"  ✅ {ticker_ns} TCN Trained | {elapsed:.0f}s")
            results["success"].append(ticker_ns)
            
        except Exception as e:
            logger.error(f"  ❌ {ticker_ns}: TCN training failed — {e}")
            results["failed"].append(ticker_ns)

    print(f"\n{'═'*62}")
    print(f"  TCN UNIVERSE TRAINING COMPLETE")
    print(f"{'═'*62}")
    print(f"  ✅  Success : {len(results['success'])}")
    print(f"  ❌  Failed  : {len(results['failed'])}")
    print(f"{'═'*62}\n")

if __name__ == "__main__":
    main()

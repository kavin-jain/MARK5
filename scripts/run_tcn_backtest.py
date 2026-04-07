#!/usr/bin/env python3
import os, sys, logging, json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.models.tcn.pipeline import TCNPipelineOrchestrator
from core.data.adapters.kite_adapter import KiteFeedAdapter
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("MARK5.TCN_Backtest")

def main():
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)
    
    symbols = ["COFORGE", "PERSISTENT", "MPHASIS", "KPITTECH", "LTTS", "HAL", "BEL", "POLYCAB", "DIXON", "ABB"]
    
    # ── Connect Kite ──────────────────────────────────────────────────────────
    api_key      = os.getenv("KITE_API_KEY", "")
    access_token = os.getenv("KITE_ACCESS_TOKEN", "")
    adapter = KiteFeedAdapter({"api_key": api_key, "access_token": access_token})
    if not adapter.connect():
        logger.error("Kite connection failed")
        return

    orchestrator = TCNPipelineOrchestrator(sequence_length=64)
    results = []

    for ticker in symbols:
        ticker_ns = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
        print(f"\n{'━'*40}\n  Backtesting TCN: {ticker_ns}\n{'━'*40}")
        
        model_path = Path(PROJECT_ROOT) / "models" / ticker_ns / "tcn_model.keras"
        if not model_path.exists():
            logger.warning(f"  {ticker_ns}: Model not found at {model_path}")
            continue

        try:
            # 1. Fetch 60m data (TCN resolution)
            df = adapter.fetch_ohlcv(ticker, period="730d", interval="60m")
            if df is None or len(df) < 500:
                logger.warning(f"  {ticker_ns}: Insufficient data ({len(df) if df is not None else 0})")
                continue
                
            df.columns = [c.lower() for c in df.columns]
            if df.index.tz is not None: df.index = df.index.tz_localize(None)

            # 2. Load Model (Pass path WITHOUT .keras as load() appends it)
            load_path = str(model_path).replace(".keras", "")
            orchestrator.load_production_model(load_path)
            
            # 3. Simulate Signals
            # TCN needs features.generate_features(df)
            features = orchestrator.engineer.generate_features(df)
            X, _, _ = orchestrator._prepare_sequences(features)
            
            if len(X) == 0:
                logger.warning(f"  {ticker_ns}: Sequence generation failed")
                continue
                
            # Predict batch
            # predict() returns (pred_dir, pred_vol)
            prob_batch, vol_batch = orchestrator.model.predict(X)
            prob = prob_batch.flatten()
            
            # 4. Filter Signals (Hurdle 0.55)
            HURDLE = 0.55
            signals = (prob >= HURDLE).astype(int)
            total_signals = int(np.sum(signals))
            avg_prob = float(np.mean(prob))
            max_prob = float(np.max(prob))
            
            logger.info(f"  ✅ {ticker_ns}: AvgProb={avg_prob:.4f} | MaxProb={max_prob:.4f} | BUY_SIGNALS={total_signals}")
            
            results.append({
                "symbol": ticker_ns,
                "signals": total_signals,
                "avg_prob": avg_prob,
                "max_prob": max_prob
            })
            
        except Exception as e:
            logger.error(f"  ❌ {ticker_ns}: TCN backtest failed — {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  TCN VALIDATION SUMMARY")
    print(f"{'═'*60}")
    for r in results:
        print(f"  - {r['symbol']:<15}: {r['signals']:>3} signals (MaxProb={r['max_prob']:.2%})")
    print(f"{'═'*60}\n")

if __name__ == "__main__":
    main()

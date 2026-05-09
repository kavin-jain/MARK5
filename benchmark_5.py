import os
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from core.models.predictor import MARK5Predictor
from core.models.tcn.backtester import RobustBacktester
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("MARK5.EliteSnapshot")

TICKERS = ["RELIANCE.NS", "M&M.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS"]
TRAINING_CUTOFF = "2025-03-31"
OOS_START = "2025-04-01"

def process_one(ticker):
    try:
        # 1. Retrain
        cmd = [
            sys.executable, "core/models/training/trainer.py", 
            "--symbols", ticker.replace('.NS', ''), 
            "--years", "3", 
            "--cutoff", TRAINING_CUTOFF,
            "--is-subprocess"
        ]
        subprocess.run(cmd, capture_output=True)
        
        # 2. Backtest
        predictor = MARK5Predictor(ticker, allow_shadow=True)
        base = ticker.replace('.NS', '')
        df = pd.read_parquet(f"data/cache/{base}_NS_1d.parquet")
        # Need 400 days to ensure 200+ trading bars for MA200/MA50 features
        test_df = df[df.index >= pd.Timestamp(OOS_START) - pd.Timedelta(days=400)].copy()
        
        full_signals = [0] * len(test_df)
        start_idx = test_df.index.get_indexer([pd.Timestamp(OOS_START)], method='bfill')[0]
        
        for i in range(start_idx, len(test_df)):
            subset = test_df.iloc[:i+1]
            res = predictor.predict(subset)
            sig = res.get('signal', 'HOLD')
            full_signals[i] = 1 if 'BUY' in sig else (-1 if 'SELL' in sig else 0)

        signals_series = pd.Series(full_signals, index=test_df.index)
        bt = RobustBacktester(segment='FUTURES')
        _, metrics = bt.run_simulation(test_df[test_df.index >= pd.Timestamp(OOS_START)], 
                                       signals_series[signals_series.index >= pd.Timestamp(OOS_START)], 
                                       symbol=ticker)
        metrics['ticker'] = ticker
        return metrics
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

def run_snapshot():
    print("\n" + "🚀" * 30)
    print("💎 MARK5 ELITE SNAPSHOT (Top 5 Stocks)")
    print("🚀" * 30 + "\n")
    
    results = []
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_one, t): t for t in TICKERS}
        for f in as_completed(futures):
            res = f.result()
            results.append(res)
        if "error" in res:
            print(f"❌ {res['ticker']}: {res['error']}")
        elif res.get('Total Trades', 0) == 0:
            print(f"⚠️ {res['ticker']}: No trades generated.")
        else:
            print(f"✅ {res['ticker']}: Ret={res['Total Return %']:+.2f}% | WR={res['Win Rate %']:.1f}% | Trades={res['Total Trades']}")

    df = pd.DataFrame([r for r in results if "error" not in r])
    if not df.empty:
        print("\n" + "="*50)
        print(f"📊 AVERAGE ELITE ALPHA: {df['Total Return %'].mean():+.2f}%")
        print(f"📈 AVERAGE WIN RATE:   {df['Win Rate %'].mean():.1f}%")
        print("="*50 + "\n")

if __name__ == "__main__":
    run_snapshot()

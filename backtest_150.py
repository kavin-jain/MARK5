import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd

# Set up logging for the script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MARK5.Backtest150")

from core.models.predictor import MARK5Predictor
from core.models.tcn.backtester import RobustBacktester

# OOS Configuration
OOS_START_DATE = "2025-04-01"
TRAINING_CUTOFF = "2025-03-31"

def train_model(ticker: str):
    logger.info(f"Retraining {ticker} with cutoff {TRAINING_CUTOFF} for OOS validation...")
    try:
        # Delete existing corrupt or stale artifacts to ensure fresh start
        mroot = Path("models") / ticker
        if mroot.exists():
            shutil.rmtree(mroot)
            
        cmd = [
            sys.executable, "core/models/training/trainer.py", 
            "--symbols", ticker.replace('.NS', ''), 
            "--years", "3", 
            "--cutoff", TRAINING_CUTOFF
        ]
        # run with timeout to avoid hanging indefinitely on one stock
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600) 
        if result.returncode != 0:
            logger.error(f"Training failed for {ticker}. Error: {result.stderr[:200]}")
            return False
        else:
            logger.info(f"Training successful for {ticker}.")
            return True
    except subprocess.TimeoutExpired:
        logger.error(f"Training timed out for {ticker}.")
        return False
    except Exception as e:
        logger.error(f"Failed to run subprocess for {ticker}: {e}")
        return False

def backtest_ticker(ticker: str) -> dict:
    try:
        # 1. Force Retrain for OOS (Toyota Standard: No compromises on validity)
        success = train_model(ticker)
        if not success:
            return {"ticker": ticker, "error": "OOS Training failed"}
            
        # FIX-6: never run backtests on models that failed the production gate.
        # allow_shadow=True was silently inflating results with certified-untrustworthy models.
        _base = Path('models') / ticker
        _versions = sorted([v for v in _base.iterdir() if v.is_dir() and v.name.startswith('v')],
                           key=lambda p: int(p.name[1:]), reverse=True) if _base.exists() else []
        if _versions:
            _meta_file = _versions[0] / 'metadata.json'
            if _meta_file.exists():
                _meta = json.loads(_meta_file.read_text())
                if not _meta.get('passes_gate', False):
                    logger.warning(f'{ticker} GATE FAIL — skipping (model did not pass production gate)')
                    return {'ticker': ticker, 'error': 'Gate failure — model not production-ready'}
        try:
            predictor = MARK5Predictor(ticker, allow_shadow=False)  # FIX-6: was allow_shadow=True
        except Exception as e:
            return {"ticker": ticker, "error": f"Could not load predictor after training: {e}"}

        # 2. Get Data from Cache
        base_sym = ticker.replace('.NS', '')
        spot_path = f"data/cache/{base_sym}_NS_1d.parquet"
        if not os.path.exists(spot_path):
            spot_path = f"data/cache/{base_sym.replace('.', '_')}_NS_1d.parquet"
            
        if not os.path.exists(spot_path):
            return {"ticker": ticker, "error": "No cached data found"}
            
        df = pd.read_parquet(spot_path)
        if df.empty:
            return {"ticker": ticker, "error": "Empty data"}

        # 3. Filter for OOS Period
        oos_start = pd.Timestamp(OOS_START_DATE)
        if df.index.max() < oos_start:
             return {"ticker": ticker, "error": f"Cache end date {df.index.max()} < OOS Start {oos_start}"}
             
        # Need 250 bars buffer for features like 200EMA, Volatility, etc.
        test_df = df[df.index >= oos_start - pd.Timedelta(days=365)].copy()
        if len(test_df[test_df.index >= oos_start]) < 20:
             return {"ticker": ticker, "error": "Insufficient OOS bars"}

        # 4. Generate Signals (Iterative to simulate live environment)
        full_signals = [0] * len(test_df)
        start_idx = test_df.index.get_indexer([oos_start], method='bfill')[0]
        
        logger.info(f"Generating OOS signals for {ticker} from {test_df.index[start_idx]} to {test_df.index[-1]}...")
        
        for i in range(start_idx, len(test_df)):
            subset = test_df.iloc[:i+1]
            res = predictor.predict(subset)
            sig = res.get('signal', 'HOLD')
            full_signals[i] = 1 if 'BUY' in sig else (-1 if 'SELL' in sig else 0)

        signals_series = pd.Series(full_signals, index=test_df.index)
        
        # 5. Run Backtester
        backtester = RobustBacktester(segment='EQUITY_INTRADAY')
        _, metrics = backtester.run_simulation(test_df, signals_series, symbol=ticker)
        metrics["ticker"] = ticker
        metrics["start_date"] = str(test_df.index[start_idx].date())
        metrics["end_date"] = str(test_df.index[-1].date())
        return metrics

    except Exception as e:
        logger.exception(f"Backtest failed for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}

def run_150_backtest():
    # Discover candidates from cache to ensure we only process stocks we have data for
    import glob
    cached_files = glob.glob("data/cache/*_1d.parquet")
    candidates = []
    for f in cached_files:
        filename = os.path.basename(f)
        if filename.startswith("NIFTY"): continue
        # Parse ticker like RELIANCE_NS_1d.parquet
        parts = filename.split("_")
        if len(parts) >= 3:
            ticker = f"{parts[0]}.NS"
            candidates.append(ticker)
    
    # Sort and take 150
    candidates = sorted(list(set(candidates)))[:150]
    logger.info(f"Loaded {len(candidates)} candidates from cache for backtesting.")

    results = []
    
    # We can use ProcessPoolExecutor for parallel backtesting
    # But since predictor uses PyTorch/XGBoost, it might be heavy. Let's use 4 workers.
    max_workers = min(8, os.cpu_count() or 4)
    logger.info(f"Running backtests with {max_workers} workers...")

    # For safety and progress tracking, we will do it sequentially if parallel crashes, 
    # but let's try parallel first to save time. Actually, sequential is safer for SQLite DB locks in MARK5.
    # Since we want it to be 100% reliable, we will do sequential. The user said "time is not an excuse".
    
    for idx, ticker in enumerate(candidates):
        logger.info(f"[{idx+1}/{len(candidates)}] Processing {ticker}...")
        res = backtest_ticker(ticker)
        results.append(res)

    return results

def generate_report(results: list, output_path: str):
    # Filter valid results
    valid_results = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if not valid_results:
        logger.error("No valid backtest results generated!")
        return

    # Calculate averages
    df = pd.DataFrame(valid_results)
    
    # Extract key metrics
    # RobustBacktester metrics usually include: Total Return %, Sharpe Ratio, Max Drawdown %, Win Rate, Total Trades
    avg_return = df.get("Total Return %", pd.Series([0])).mean()
    avg_sharpe = df.get("Sharpe Ratio", pd.Series([0])).mean()
    avg_win_rate = df.get("Win Rate", pd.Series([0])).mean()
    avg_drawdown = df.get("Max Drawdown %", pd.Series([0])).mean()
    avg_trades = df.get("Total Trades", pd.Series([0])).mean()

    # Needs assessment
    needs_met = []
    if avg_return >= 15.0:
        needs_met.append("✅ Average Return >= 15% target")
    else:
        needs_met.append(f"❌ Average Return < 15% target (Actual: {avg_return:.2f}%)")
        
    if avg_sharpe > 0.5:
        needs_met.append("✅ Average Sharpe Ratio > 0.5 target")
    else:
        needs_met.append(f"❌ Average Sharpe Ratio <= 0.5 target (Actual: {avg_sharpe:.2f})")
        
    if avg_win_rate > 0.44:
        needs_met.append("✅ Average Win Rate > 44% target")
    else:
        needs_met.append(f"❌ Average Win Rate <= 44% target (Actual: {avg_win_rate:.2f})")

    # Generate Markdown
    md = [
        "# MARK5 - 150 Stock Systematic Backtest Report (STRICT OOS)",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Validation Type:** Strict Out-of-Sample (OOS)",
        f"**Training Cutoff:** {TRAINING_CUTOFF}",
        f"**Backtest Horizon:** {OOS_START_DATE} to Present",
        f"**Total Stocks Tested:** {len(results)}",
        f"**Successful Backtests:** {len(valid_results)}",
        f"**Failed Backtests:** {len(errors)}",
        "",
        "## System Averages vs Needs",
        f"- **Average Total Return:** {avg_return:.2f}%",
        f"- **Average Sharpe Ratio:** {avg_sharpe:.2f}",
        f"- **Average Win Rate:** {avg_win_rate:.2f}%",
        f"- **Average Max Drawdown:** {avg_drawdown:.2f}%",
        f"- **Average Trades per Stock:** {avg_trades:.1f}",
        "",
        "### Requirements Check",
    ] + [f"- {n}" for n in needs_met] + [
        "",
        "## Individual Stock Results (Top 20 by Return)",
        "| Ticker | Return % | Sharpe | Win Rate % | Max Drawdown % | Trades |",
        "|--------|----------|--------|------------|----------------|--------|"
    ]

    # Sort by return and add to table
    if "Total Return %" in df.columns:
        df_sorted = df.sort_values(by="Total Return %", ascending=False)
        for _, row in df_sorted.head(20).iterrows():
            t = row.get("ticker", "N/A")
            ret = row.get("Total Return %", 0)
            sh = row.get("Sharpe Ratio", 0)
            wr = row.get("Win Rate", 0)
            dd = row.get("Max Drawdown %", 0)
            tr = row.get("Total Trades", 0)
            md.append(f"| {t} | {ret:.2f}% | {sh:.2f} | {wr:.2f}% | {dd:.2f}% | {tr} |")

    if errors:
        md.append("\n## Errors")
        for e in errors[:10]:
            md.append(f"- **{e['ticker']}**: {e['error']}")
        if len(errors) > 10:
            md.append(f"- ... and {len(errors) - 10} more.")

    report_content = "\n".join(md)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_content)
        
    logger.info(f"Report saved to {output_path}")
    print("\n" + report_content)

if __name__ == "__main__":
    logger.info("Starting Systematic 150 Stock Backtest...")
    results = run_150_backtest()
    report_path = "reports/backtest_150_report.md"
    generate_report(results, report_path)
    logger.info("Done.")

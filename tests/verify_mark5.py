"""
MARK5 VERIFICATION ORCHESTRATOR v1.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-04-17] v1.0: Initial release. Orchestrates tests and model regeneration.

TRADING ROLE: Master verification script for ML pipeline integrity.
SAFETY LEVEL: HIGH
"""

import os
import sys
import argparse
import subprocess
import logging
import json
import time
import gc
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Tuple

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set CPU-only execution for reproducibility and to avoid GPU resource contention in parallel
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [VERIFY] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MARK5.Verify")

ELITE_STOCKS = ["HDFCBANK.NS", "RELIANCE.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS"]
REPORT_PATH = os.path.join(project_root, "reports/verify_mark5_report.md")

def run_tests() -> Tuple[bool, str]:
    """Step 1: Execute the Pytest suite."""
    logger.info("Step 1: Running Pytest suite...")
    test_files = ["tests/test_cpcv_horizons.py", "tests/test_feature_leakage.py"]
    cmd = ["pytest", "-v"] + test_files
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ All tests passed.")
            return True, result.stdout
        else:
            logger.error("❌ Tests failed.")
            return False, result.stdout + "\n" + result.stderr
    except Exception as e:
        logger.error(f"❌ Error running tests: {e}")
        return False, str(e)

def train_symbol(symbol: str) -> Dict:
    """Worker function for parallel model training."""
    # Force CPU-only in the worker process
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    try:
        # Import inside worker to avoid issues with ProcessPoolExecutor
        from core.models.training.trainer import MARK5MLTrainer
        import pandas as pd
        
        # Monkeypatch to force CPU in trainer's internal detection
        MARK5MLTrainer._detect_gpu = lambda self: False
        
        logger.info(f"[{symbol}] Starting training...")
        trainer = MARK5MLTrainer()
        
        # Fetch 3 years of data (standard for MARK5)
        data = trainer.fetch_data_for_training(symbol, years=3.0)
        if data is None or data.empty:
            return {"symbol": symbol, "status": "failed", "reason": "Data fetch failed"}
            
        res = trainer.train_advanced_ensemble(symbol, data)
        res["symbol"] = symbol
        
        # Cleanup
        del data
        gc.collect()
        
        return res
    except Exception as e:
        logger.error(f"[{symbol}] Training crashed: {e}")
        return {"symbol": symbol, "status": "failed", "reason": str(e)}

def get_universe() -> List[str]:
    """Read full universe from config/universe.json."""
    try:
        universe_path = os.path.join(project_root, "config/universe.json")
        with open(universe_path, "r") as f:
            config = json.load(f)
            return config.get("active_universe", [])
    except Exception as e:
        logger.warning(f"Could not read universe.json: {e}. Using ELITE_STOCKS as fallback.")
        return ELITE_STOCKS

def generate_report(test_passed: bool, test_output: str, train_results: List[Dict], duration: float):
    """Step 3: Synthesis & Reporting."""
    logger.info(f"Step 3: Generating report at {REPORT_PATH}...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    test_status = "✅ PASSED" if test_passed else "❌ FAILED"
    
    report = f"""# MARK5 Verification Report
Generated: {timestamp}
Duration: {duration:.2f} seconds

## 1. Test Suite Results
Status: {test_status}

<details>
<summary>Pytest Output</summary>

```
{test_output}
```

</details>

## 2. Model Regeneration Summary
| Symbol | Status | Version | CPCV P(Sharpe>1.5) | Mean Sharpe | Passes Gate | Reason/Error |
|--------|--------|---------|-------------------|-------------|-------------|--------------|
"""
    
    for res in train_results:
        symbol = res.get("symbol", "Unknown")
        status = res.get("status", "failed")
        version = res.get("version", "N/A")
        p_sharpe = res.get("cpcv_p_sharpe", 0.0)
        mean_sharpe = res.get("mean_sharpe", 0.0)
        passes_gate = "✅" if res.get("passes_prod_gate") else "❌"
        reason = res.get("reason", "")
        
        if status == "success":
            report += f"| {symbol} | {status} | v{version} | {p_sharpe:.1%} | {mean_sharpe:.2f} | {passes_gate} | - |\n"
        else:
            report += f"| {symbol} | {status} | - | - | - | - | {reason} |\n"
            
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    logger.info(f"Report written to {REPORT_PATH}")

def main():
    parser = argparse.ArgumentParser(description="MARK5 Master Verification Script")
    parser.add_argument("--fast", action="store_true", help="Run only on top 5 elite stocks")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Step 1: Run Tests
    test_passed, test_output = run_tests()
    if not test_passed:
        logger.error("Hard Gate: Tests failed. Aborting model regeneration.")
        sys.exit(1)
        
    # Step 2: Regenerate Models
    symbols = ELITE_STOCKS if args.fast else get_universe()
    logger.info(f"Step 2: Regenerating models for {len(symbols)} symbols...")
    
    train_results = []
    # Use max_workers to avoid overwhelming the system, but enough for parallelism
    # Limit to 4 to be safe on typical hardware if cpu_count is high
    max_workers = min(os.cpu_count() or 4, len(symbols), 4)
    logger.info(f"Using {max_workers} parallel workers.")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        train_results = list(executor.map(train_symbol, symbols))
        
    duration = time.time() - start_time
    
    # Step 3: Reporting
    generate_report(test_passed, test_output, train_results, duration)
    
    logger.info("Verification process complete.")

if __name__ == "__main__":
    main()

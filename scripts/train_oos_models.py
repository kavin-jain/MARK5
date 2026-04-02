#!/usr/bin/env python3
"""
MARK5 OUT-OF-SAMPLE (OOS) BLIND RETRAIN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reads existing data/cache/*_NS_1d.parquet files, strictly slices the data at
`2024-09-30`, and retrains the ML models on this truncated history.

This ensures models have mathematically ZERO knowledge of the 18-month test 
window (2024-10-01 to 2026-04-01), eliminating in-sample bias for the
final backtest.
"""

import os, sys, argparse, logging, warnings, time
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("MARK5.OOS_Retrain")

def main():
    CUTOFF_DATE = '2024-09-30'

    from core.models.features import FEATURE_COLS
    from core.models.training.trainer import MARK5MLTrainer
    from core.data.market_data import MarketDataProvider
    from core.data.fii_data import FIIDataProvider

    cache_dir = Path(PROJECT_ROOT) / "data" / "cache"
    parquets = sorted(cache_dir.glob("*_NS_1d.parquet"))

    print(f"\n{'═'*62}")
    print(f"  MARK5 OOS BLIND RETRAIN — {len(parquets)} symbols")
    print(f"  HARD CUTOFF: {CUTOFF_DATE} (Forgetting the future)")
    print(f"{'═'*62}\n")

    logger.info("Loading shared market context...")
    nifty_close, fii_series = None, None
    try:
        mp = MarketDataProvider()
        nifty_df = mp.get_nifty50_data("2020-01-01", CUTOFF_DATE) # Slice NIFTY too!
        if nifty_df is not None:
            nifty_close = nifty_df['close']
            logger.info(f"  NIFTY loaded up to cutoff: {len(nifty_close)} bars")
    except Exception as e:
        logger.warning(f"  NIFTY load failed: {e}")

    # No FIIDataProvider locally historically, but will pass None
    context = {'nifty_close': nifty_close, 'fii_net': None}

    results = {"success": [], "failed": []}
    trainer = MARK5MLTrainer()

    for i, ppath in enumerate(parquets, 1):
        symbol = ppath.stem.replace("_NS_1d", "")
        ticker_ns = f"{symbol}.NS"
        print(f"\n[{i}/{len(parquets)}] {symbol}")
        mdir = Path(PROJECT_ROOT) / "models" / ticker_ns
        if mdir.exists() and any(x.name.startswith('v') for x in mdir.iterdir() if x.is_dir()):
            logger.info(f"  {symbol}: Already trained — skipping")
            continue

        try:
            df = pd.read_parquet(ppath)
            df.columns = [c.lower() for c in df.columns]
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df = df.dropna(subset=["open", "high", "low", "close", "volume"])
            
            # --- THE MAGICAL BLINDING LINE ---
            df = df[df.index <= pd.Timestamp(CUTOFF_DATE)]
            
        except Exception as e:
            logger.error(f"  {symbol}: parquet load failed — {e}")
            results["failed"].append(symbol)
            continue

        if len(df) < 250:
            logger.warning(f"  {symbol}: only {len(df)} rows after cutoff (need 250) — skipping")
            results["failed"].append(symbol)
            continue

        logger.info(f"  {symbol}: Blind training on {len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})")

        # Inject context properly without passing natively
        trainer.config.feature_context = context

        t0 = time.time()
        try:
            # Recreate models dir blindly
            import shutil
            mdir = Path(PROJECT_ROOT) / "models" / ticker_ns
            if mdir.exists(): shutil.rmtree(mdir)
            
            result = trainer.train_advanced_ensemble(ticker_ns, df)
            elapsed = time.time() - t0

            if result.get("status") == "success":
                v = result.get("version", "?")
                p_sharpe = result.get("cpcv_p_sharpe", 0)
                logger.info(f"  ✅ {symbol} OOS Model v{v} | P(Sharpe>1.5)={p_sharpe:.0%} | {elapsed:.0f}s")
                results["success"].append(symbol)
            else:
                logger.error(f"  ❌ {symbol}: OOS training failed — {result.get('reason')}")
                results["failed"].append(symbol)
        except Exception as e:
            logger.error(f"  ❌ {symbol}: exception — {e}")
            results["failed"].append(symbol)

    print(f"\n{'═'*62}")
    print(f"  OOS RETRAIN COMPLETE")
    print(f"{'═'*62}")
    print(f"  ✅  Success : {len(results['success'])}")
    print(f"  ❌  Failed  : {len(results['failed'])}")
    print(f"{'═'*62}\n")

if __name__ == "__main__":
    main()

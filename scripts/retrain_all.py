#!/usr/bin/env python3
"""
MARK5 FULL RETRAIN SCRIPT v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fetches 3 years of daily OHLCV from Kite for each symbol
and retrains using the correct MARK5MLTrainer + AdvancedFeatureEngine pipeline.

Old models used corrupted features (atr_regime=ns timestamps, fii_flow_3d=zeros).
This script fixes that by retraining from scratch with the current codebase.

Usage:
    python3 scripts/retrain_all.py
    python3 scripts/retrain_all.py --symbols RELIANCE TCS INFY
    python3 scripts/retrain_all.py --symbols RELIANCE --years 3
"""

import os, sys, argparse, logging, warnings, time
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MARK5.Retrain")

from scripts.nifty50_universe import NIFTY_50, NIFTY_MIDCAP_TICKERS

# ── Universe ──────────────────────────────────────────────────────────────────
# Combined 105-stock universe (NIFTY 50 + NIFTY Midcap 100)
_N50 = [t.replace('.NS', '') for t in NIFTY_50.keys()]
_MID = [t.replace('.NS', '') for t in NIFTY_MIDCAP_TICKERS]
DEFAULT_SYMBOLS = sorted(list(set(_N50 + _MID)))

def parse_args():
    p = argparse.ArgumentParser(description="MARK5 Full Model Retrain")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--years", type=float, default=3.0,
                   help="Years of history to train on (default: 3)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip symbols that already have a valid (non-corrupt) model")
    return p.parse_args()


def is_model_corrupt(ticker_ns: str) -> bool:
    """Forced True: we need a full universe refresh with corrected feature logic."""
    return True


def fetch_ohlcv(symbol: str, years: float, adapter) -> pd.DataFrame:
    """Fetch 60m bars from Kite, cache as parquet."""
    cache_dir = Path(PROJECT_ROOT) / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol}_60m.parquet"

    # Use recent cache if < 24h old
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < 86400:
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"{symbol}: loaded {len(df)} bars from cache")
            return df
        except Exception:
            pass

    try:
        # Fetch 60m data (adapter handle chunking)
        df = adapter.fetch_ohlcv(symbol, period=f"{int(years * 365)}d", interval="60m")
        if df is not None and len(df) >= 500: # Need more bars for 60m
            df.to_parquet(cache_path)
            logger.info(f"{symbol}: fetched {len(df)} 60m bars from Kite ✅")
            return df
    except Exception as e:
        logger.warning(f"{symbol}: Kite fetch failed — {e}")

    # yfinance REMOVED per RULE 4
    logger.error(f"{symbol}: Data source failed — {symbol} skipped.")
    return None


def main():
    args = parse_args()
    symbols = args.symbols or DEFAULT_SYMBOLS

    print(f"\n{'═'*60}")
    print(f"  MARK5 RETRAIN — {len(symbols)} symbols, {args.years:.0f} years")
    print(f"{'═'*60}\n")

    # ── Connect Kite ──────────────────────────────────────────────────────────
    api_key      = os.getenv("KITE_API_KEY", "")
    access_token = os.getenv("KITE_ACCESS_TOKEN", "")
    adapter = None
    if api_key and access_token:
        try:
            from core.data.adapters.kite_adapter import KiteFeedAdapter
            adapter = KiteFeedAdapter({
                "api_key": api_key,
                "access_token": access_token,
                "api_secret": os.getenv("KITE_API_SECRET", ""),
            })
            if adapter.connect():
                logger.info("✅ Kite connected")
            else:
                adapter = None
        except Exception as e:
            logger.warning(f"Kite init failed: {e}")

    # ── Import trainer (uses correct AdvancedFeatureEngine) ───────────────────
    from core.models.training.trainer import MARK5MLTrainer

    results = {"success": [], "failed": [], "skipped": []}

    for i, symbol in enumerate(symbols, 1):
        ticker_ns = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        print(f"\n[{i}/{len(symbols)}] {symbol}")

        # Skip if model is clean and --skip-existing is set
        if args.skip_existing and not is_model_corrupt(ticker_ns):
            logger.info(f"{symbol}: existing model is valid — skipping")
            results["skipped"].append(symbol)
            continue

        # Fetch data
        df = fetch_ohlcv(symbol, args.years, adapter)
        if df is None or len(df) < 300:
            logger.error(f"{symbol}: insufficient data — skipping")
            results["failed"].append(symbol)
            continue

        # Ensure lowercase columns
        df.columns = [c.lower() for c in df.columns]
        for required in ["open", "high", "low", "close", "volume"]:
            if required not in df.columns:
                logger.error(f"{symbol}: missing '{required}' — skipping")
                results["failed"].append(symbol)
                break
        else:
            pass  # all columns present

        t0 = time.time()
        try:
            # Strip timezone from index — trainer expects tz-naive timestamps
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            trainer = MARK5MLTrainer(kite_adapter=adapter)
            result  = trainer.train_advanced_ensemble(ticker_ns, df)

            elapsed = time.time() - t0
            if result.get("status") == "success":
                v = result.get("version", "?")
                auc = result.get("auc", result.get("roc_auc", 0))
                logger.info(
                    f"✅ {symbol}: v{v} | AUC={auc:.4f} | "
                    f"Fbeta={result.get('avg_fbeta', 0):.4f} | "
                    f"{elapsed:.0f}s"
                )
                results["success"].append(symbol)
            else:
                reason = result.get("reason", "unknown")
                logger.error(f"❌ {symbol}: training failed — {reason}")
                results["failed"].append(symbol)

        except Exception as e:
            import traceback
            logger.error(f"❌ {symbol}: exception — {e}")
            traceback.print_exc()
            results["failed"].append(symbol)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  RETRAIN COMPLETE")
    print(f"{'═'*60}")
    print(f"  ✅ Success  : {len(results['success'])} — {results['success']}")
    print(f"  ⏭️  Skipped  : {len(results['skipped'])} — {results['skipped']}")
    print(f"  ❌ Failed   : {len(results['failed'])} — {results['failed']}")
    print(f"{'═'*60}\n")

    if adapter:
        try:
            adapter.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()

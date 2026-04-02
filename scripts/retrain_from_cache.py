#!/usr/bin/env python3
"""
MARK5 FAST RETRAIN FROM CACHE v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reads existing data/cache/*_NS_1d.parquet files and retrains
the CPCV stacking ensemble with the current v12.4 feature schema.

NO NETWORK REQUIRED — reads from local parquet cache only.

Usage:
    python3 scripts/retrain_from_cache.py
    python3 scripts/retrain_from_cache.py --symbols RELIANCE TCS INFY
    python3 scripts/retrain_from_cache.py --max 10   # first 10 stocks only
"""

import os, sys, argparse, logging, warnings, time, json
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MARK5.CacheRetrain")


def parse_args():
    p = argparse.ArgumentParser(description="MARK5 Fast Retrain from Cache")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="Specific symbols (without .NS) to retrain")
    p.add_argument("--max", type=int, default=999,
                   help="Maximum number of symbols to retrain")
    p.add_argument("--min-rows", type=int, default=400,
                   help="Minimum rows of OHLCV data required (default 400 ≈ 1.5yr)")
    p.add_argument("--force", action="store_true",
                   help="Force retrain even if model already has correct feature schema")
    return p.parse_args()


def model_has_correct_schema(ticker_ns: str, expected_features: list) -> bool:
    """Return True if the most recent model uses the current feature schema."""
    import joblib
    models_root = Path(PROJECT_ROOT) / "models" / ticker_ns
    if not models_root.exists():
        return False
    versions = sorted(
        [v for v in os.listdir(models_root) if v.startswith("v") and v[1:].isdigit()],
        key=lambda x: int(x[1:]), reverse=True
    )
    if not versions:
        return False
    vdir = models_root / versions[0]
    feat_file = vdir / "features.json"
    if not feat_file.exists():
        return False
    try:
        with open(feat_file) as f:
            saved_feats = json.load(f)
        return saved_feats == expected_features
    except Exception:
        return False


def main():
    args = parse_args()

    from core.models.features import FEATURE_COLS
    from core.models.training.trainer import MARK5MLTrainer
    from core.data.market_data import MarketDataProvider
    from core.data.fii_data import FIIDataProvider

    cache_dir = Path(PROJECT_ROOT) / "data" / "cache"

    # Discover all cached stocks
    all_parquets = sorted(cache_dir.glob("*_NS_1d.parquet"))

    if args.symbols:
        # Filter to requested symbols
        requested = {s.upper().replace(".NS", "") for s in args.symbols}
        parquets = [p for p in all_parquets
                    if p.stem.replace("_NS_1d", "").upper() in requested]
    else:
        parquets = all_parquets[:args.max]

    print(f"\n{'═'*62}")
    print(f"  MARK5 CACHE RETRAIN — {len(parquets)} symbols")
    print(f"  Feature schema v12.4: {FEATURE_COLS}")
    print(f"{'═'*62}\n")

    # Load shared context once
    logger.info("Loading shared market context...")
    nifty_close = None
    fii_series = None
    try:
        mp = MarketDataProvider()
        nifty_df = mp.get_nifty50_data("2020-01-01", "2026-01-01")
        if nifty_df is not None:
            nifty_close = nifty_df['close']
            logger.info(f"  NIFTY loaded: {len(nifty_close)} bars")
    except Exception as e:
        logger.warning(f"  NIFTY load failed: {e}")

    try:
        fp = FIIDataProvider()
        fii_series = fp.get_fii_flow("2020-01-01", "2026-01-01")
        if fii_series is not None and fii_series.std() > 1e-6:
            logger.info(f"  FII loaded: {len(fii_series)} bars")
        else:
            fii_series = None
            logger.warning("  FII data unavailable — will use None (zero-fill in features)")
    except Exception as e:
        logger.warning(f"  FII load failed: {e}")

    context = {
        'nifty_close': nifty_close,
        'fii_net': fii_series,
    }

    results = {"success": [], "failed": [], "skipped": []}
    trainer = MARK5MLTrainer()

    for i, ppath in enumerate(parquets, 1):
        symbol = ppath.stem.replace("_NS_1d", "")
        ticker_ns = f"{symbol}.NS"

        print(f"\n[{i}/{len(parquets)}] {symbol}")

        # Skip if already on correct schema (unless --force)
        if not args.force and model_has_correct_schema(ticker_ns, FEATURE_COLS):
            logger.info(f"  {symbol}: already on v12.4 schema — skipping (use --force to override)")
            results["skipped"].append(symbol)
            continue

        # Load OHLCV from parquet
        try:
            df = pd.read_parquet(ppath)
            df.columns = [c.lower() for c in df.columns]
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    raise ValueError(f"Missing column: {col}")
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        except Exception as e:
            logger.error(f"  {symbol}: parquet load failed — {e}")
            results["failed"].append(symbol)
            continue

        if len(df) < args.min_rows:
            logger.warning(f"  {symbol}: only {len(df)} rows (need {args.min_rows}) — skipping")
            results["failed"].append(symbol)
            continue

        logger.info(f"  {symbol}: {len(df)} bars | {df.index[0].date()} → {df.index[-1].date()}")

        # Inject context into trainer config (feature engine picks it up)
        trainer.config.feature_context = context

        t0 = time.time()
        try:
            result = trainer.train_advanced_ensemble(ticker_ns, df)
            elapsed = time.time() - t0

            if result.get("status") == "success":
                v = result.get("version", "?")
                p_sharpe = result.get("cpcv_p_sharpe", 0)
                fbeta = result.get("avg_fbeta", 0)
                gate = "✅ PASS" if result.get("passes_prod_gate") else "⚠️ GATE FAIL"
                logger.info(
                    f"  ✅ {symbol} v{v} | P(Sharpe>1.5)={p_sharpe:.0%} | "
                    f"F0.5={fbeta:.4f} | {gate} | {elapsed:.0f}s"
                )
                results["success"].append(symbol)
            else:
                reason = result.get("reason", "unknown")
                logger.error(f"  ❌ {symbol}: training failed — {reason}")
                results["failed"].append(symbol)

        except Exception as e:
            import traceback
            logger.error(f"  ❌ {symbol}: exception — {e}")
            traceback.print_exc()
            results["failed"].append(symbol)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  RETRAIN COMPLETE")
    print(f"{'═'*62}")
    print(f"  ✅  Success : {len(results['success'])} — {results['success']}")
    print(f"  ⏭️   Skipped : {len(results['skipped'])}")
    print(f"  ❌  Failed  : {len(results['failed'])} — {results['failed']}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()

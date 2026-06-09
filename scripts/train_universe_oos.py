"""
MARK5 — Full Universe OOS Training Script
==========================================
Trains V2 models for ALL 32 tickers with sufficient historical data
using OOS cutoff 2021-12-31. This expands the models_v2_oos directory
beyond the original cherry-picked 13 to expose the REAL system performance.

Usage:
    python3 scripts/train_universe_oos.py
    python3 scripts/train_universe_oos.py --trials 10       # faster
    python3 scripts/train_universe_oos.py --skip-existing   # only new tickers
    python3 scripts/train_universe_oos.py --tickers BEL LT  # subset
"""
import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [TRAIN-UNIV] | %(message)s",
)
logger = logging.getLogger("MARK5.TrainUniverse")

# ── All tickers eligible for OOS (≥900 rows before 2021-12-31, data into 2022+)
# Verified 2026-05-25 from local parquet cache.
ALL_ELIGIBLE = [
    # Long history (2015-2026, ~1728 pre-cutoff rows)
    "ASIANPAINT", "BAJFINANCE", "BEL", "BHARTIARTL", "COFORGE",
    "HDFCBANK", "HINDUNILVR", "ICICIBANK", "IDEA", "INFY",
    "ITC", "KOTAKBANK", "LT", "LUPIN", "MARUTI", "MOTHERSON",
    "PERSISTENT", "PNB", "RELIANCE", "SBIN", "SUNPHARMA",
    "TATAELXSI", "TATASTEEL", "TCS", "TITAN", "TRENT", "VOLTAS", "YESBANK",
    # Shorter but sufficient (≥900 pre-cutoff rows)
    "AUBANK",      # 2017-07, 1107 rows
    "IDFCFIRSTB",  # 2015-11, 1514 rows
    "HAL",         # 2018-04, 928 rows
    "BANDHANBNK",  # 2018-03, 930 rows
]

OOS_CUTOFF    = "2021-12-31"
MODELS_DIR    = os.path.join(_ROOT, "models_v2_oos")
CACHE_DIR     = os.path.join(_ROOT, "data", "cache")
REPORTS_DIR   = os.path.join(_ROOT, "reports")


def load_cache(ticker: str) -> pd.DataFrame | None:
    for suffix in ["_daily.parquet", "_NS_1d.parquet"]:
        path = os.path.join(CACHE_DIR, f"{ticker}{suffix}")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                df.columns = [str(c).lower() for c in df.columns]
                if hasattr(df.index, "tz") and df.index.tz is not None:
                    from zoneinfo import ZoneInfo
                    df.index = df.index.tz_convert(ZoneInfo("Asia/Kolkata")).tz_localize(None)
                df = df.sort_index()
                df = df[~df.index.duplicated(keep="last")]
                df = df[df["volume"] > 0]
                required = {"open", "high", "low", "close", "volume"}
                if required.issubset(set(df.columns)):
                    return df
            except Exception as e:
                logger.warning(f"[{ticker}] Cache load error: {e}")
    return None


def already_trained(ticker: str, models_dir: str) -> bool:
    """Returns True if this ticker has a model directory with at least one pkl."""
    for candidate in [ticker, f"{ticker}.NS"]:
        path = os.path.join(models_dir, candidate)
        if not os.path.exists(path):
            continue
        versions = sorted(
            [d for d in os.listdir(path) if d.startswith("v")],
            key=lambda x: int(x[1:]) if x[1:].isdigit() else 0,
        )
        if not versions:
            continue
        latest = os.path.join(path, versions[-1])
        if any(f.endswith(".pkl") for f in os.listdir(latest)):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers to train (default: ALL_ELIGIBLE)")
    parser.add_argument("--trials", type=int, default=15,
                        help="Optuna trials per ticker (default 15; 20 for flagship quality)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip tickers already trained (default: True)")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Retrain even if model exists (overrides --skip-existing)")
    parser.add_argument("--no-optuna", action="store_true", default=False,
                        help="Disable HPO (faster for testing)")
    parser.add_argument("--no-sector", action="store_true", default=False)
    parser.add_argument("--no-fno", action="store_true", default=False)
    args = parser.parse_args()

    tickers = args.tickers or ALL_ELIGIBLE
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Build trainer
    from types import SimpleNamespace
    trainer_config = SimpleNamespace(
        models_dir=MODELS_DIR,
        model_versions_path=os.path.join(MODELS_DIR, "versions.json"),
        prediction_horizon=20,
        transaction_cost=0.0012,
    )
    from core.models.training.trainer_v2 import MARK5MLTrainerV2
    trainer = MARK5MLTrainerV2(
        config=trainer_config,
        use_optuna=not args.no_optuna,
        optuna_trials=args.trials,
        include_sector=not args.no_sector,
        include_fno=not args.no_fno,
    )

    # Partition tickers
    skip_these = []
    train_these = []
    for t in tickers:
        if (not args.force) and already_trained(t, MODELS_DIR):
            skip_these.append(t)
        else:
            train_these.append(t)

    print(f"\n{'═'*72}")
    print(f"  MARK5 FULL UNIVERSE OOS TRAINING")
    print(f"  Cutoff: {OOS_CUTOFF}  |  Models dir: {MODELS_DIR}")
    print(f"  Optuna trials: {args.trials}  |  V2 features (33)")
    print(f"  Total eligible: {len(tickers)}")
    print(f"  Already trained (skipping): {len(skip_these)}  {skip_these}")
    print(f"  To train: {len(train_these)}  {train_these}")
    est_mins = len(train_these) * 5
    print(f"  Est. time: ~{est_mins}–{est_mins*2} minutes")
    print(f"{'═'*72}\n")

    results: dict = {}
    passed, failed, skipped_train = [], [], list(skip_these)

    overall_start = time.time()
    for i, ticker in enumerate(train_these, 1):
        print(f"\n[{i}/{len(train_these)}] ──── {ticker} ────")
        t0 = time.time()

        df = load_cache(ticker)
        if df is None:
            print(f"  ⏭  SKIPPED: no cache data")
            results[ticker] = {"status": "skipped", "reason": "no cache"}
            skipped_train.append(ticker)
            continue

        # Apply training cutoff
        cutoff_ts = pd.Timestamp(OOS_CUTOFF)
        df_train = df[df.index <= cutoff_ts].copy()
        rows_pre = len(df_train)
        rows_oos  = len(df[df.index > cutoff_ts])

        if rows_pre < 500:
            print(f"  ⏭  SKIPPED: only {rows_pre} rows before {OOS_CUTOFF}")
            results[ticker] = {"status": "skipped", "reason": f"only {rows_pre} pre-cutoff rows"}
            skipped_train.append(ticker)
            continue

        print(f"  Training on {rows_pre} rows ({df_train.index[0].date()} → {df_train.index[-1].date()})")
        print(f"  OOS test window: {rows_oos} rows after {OOS_CUTOFF}")

        try:
            r = trainer.train_advanced_ensemble(ticker, df_train, n_trials=args.trials)
            results[ticker] = r
            elapsed = time.time() - t0

            status = r.get("status", "failed")
            if status == "success":
                sharpe  = r.get("mean_sharpe") or r.get("mean_fold_sharpe") or 0
                auc     = r.get("auc", 0.0) or 0.0
                n_feats = r.get("n_features", "?")
                prod    = "✅ GATE-PASS" if r.get("passes_prod_gate", False) else "⚠️ GATE-FAIL"
                print(f"  ✅ DONE ({elapsed:.0f}s) | sharpe={sharpe:+.3f} | AUC={auc:.4f} | features={n_feats} | {prod}")
                passed.append(ticker)
            else:
                print(f"  ❌ FAILED: {r.get('reason','?')[:80]}")
                failed.append(ticker)
        except Exception as e:
            logger.exception(f"[{ticker}] Crashed: {e}")
            results[ticker] = {"status": "failed", "reason": str(e)}
            failed.append(ticker)
        finally:
            gc.collect()

    total_elapsed = time.time() - overall_start
    print(f"\n{'═'*72}")
    print(f"  TRAINING COMPLETE  ({total_elapsed/60:.1f} min)")
    print(f"  Passed    : {len(passed)}")
    print(f"  Failed    : {len(failed)}   {failed}")
    print(f"  Pre-existing (skipped): {len(skip_these)}")
    print(f"  Skipped (no data)     : {len([t for t in skipped_train if t not in skip_these])}")
    print(f"  Models dir: {MODELS_DIR}")
    print(f"{'═'*72}\n")

    # Save results
    out_path = os.path.join(REPORTS_DIR, "train_universe_oos_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {k: {kk: str(vv) for kk, vv in v.items()} for k, v in results.items()},
            f, indent=2,
        )
    print(f"  Results → {out_path}")

    if failed:
        print(f"\n  ⚠️  Failed tickers: {failed}")
        print(f"  Re-run: python3 scripts/train_universe_oos.py --tickers {' '.join(failed)}")

    return results


if __name__ == "__main__":
    main()

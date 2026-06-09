"""
MARK5 — Full Model Retrain Script (Iteration 4)
Retrains all tickers using:
  - Wilder's EWM ATR (alpha=1/14)
  - Asymmetric PT/SL = [3.5, 1.5]  (Iteration 3+)
  - 20-bar CPCV embargo  (matches triple-barrier time horizon)
  - 3-way calibration split (no leakage)
  - Probability hurdle 0.55
  - ITC excluded (BB breakout systematically losing on ITC, WR=31.5% over 11yr)

Usage:
    python3 scripts/retrain_all.py [--tickers HDFCBANK RELIANCE ...]
"""
import os, sys, gc, json, logging, argparse
from datetime import datetime
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | [RETRAIN] | %(message)s")
logger = logging.getLogger("MARK5.Retrain")

# ── Core tickers (no .NS duplicates) ─────────────────────────────────────────
# ITC excluded: BB breakout WR=31.5%, avg_return=-0.547% over 11yr (model AUC=0.331 — predicting wrong direction)
DEFAULT_TICKERS = [
    "ASIANPAINT", "AUBANK", "BAJFINANCE", "BANDHANBNK", "BEL",
    "BHARTIARTL", "COFORGE", "HAL", "HDFCBANK", "HINDUNILVR",
    "ICICIBANK", "IDEA", "IDFCFIRSTB", "INFY",
    "KOTAKBANK", "LT", "LUPIN", "MARUTI", "MOTHERSON",
    "PERSISTENT", "PNB", "RELIANCE", "SBIN", "SUNPHARMA",
    "TATAELXSI", "TATASTEEL", "TCS", "TITAN", "TRENT",
    "VOLTAS", "YESBANK",
]

CACHE_DIR = os.path.join(_ROOT, "data", "cache")
MODELS_DIR = os.path.join(_ROOT, "models")

def load_from_cache(ticker: str) -> pd.DataFrame | None:
    """Load from local parquet cache — no network calls needed."""
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
                df = df[df["volume"] > 0]          # drop zero-volume rows
                df = df[~df.index.duplicated(keep="last")]
                required = {"open", "high", "low", "close", "volume"}
                if required.issubset(set(df.columns)) and len(df) >= 300:
                    logger.info(f"[{ticker}] Loaded {len(df)} rows from cache ({df.index[0].date()} → {df.index[-1].date()})")
                    return df
            except Exception as e:
                logger.warning(f"[{ticker}] Cache load error: {e}")
    return None

def retrain_ticker(ticker: str, trainer, cutoff: str | None = None, n_trials: int = 30) -> dict:
    """Retrain one ticker using cached data. If cutoff given, only use data ≤ cutoff."""
    df = load_from_cache(ticker)
    if df is None:
        logger.error(f"[{ticker}] No cache data found — skipping")
        return {"status": "skipped", "reason": "No cached data"}

    if cutoff:
        cutoff_ts = pd.Timestamp(cutoff)
        df = df[df.index <= cutoff_ts]
        if len(df) < 300:
            logger.error(f"[{ticker}] Only {len(df)} rows before cutoff {cutoff} — skipping")
            return {"status": "skipped", "reason": f"Only {len(df)} rows before cutoff"}
        logger.info(f"[{ticker}] Cutoff {cutoff}: using {len(df)} rows "
                    f"({df.index[0].date()} → {df.index[-1].date()})")

    logger.info(f"[{ticker}] Starting retrain with {len(df)} rows…")
    try:
        result = trainer.train_advanced_ensemble(ticker, df, n_trials=n_trials)
        return result
    except Exception as e:
        logger.exception(f"[{ticker}] Training crashed: {e}")
        return {"status": "failed", "reason": str(e)}
    finally:
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Retrain MARK5 models with fixed code")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                        help="Tickers to train (default: all 32)")
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna trials per ticker (default 30)")
    parser.add_argument("--cutoff", type=str, default=None,
                        help="Training data cutoff date YYYY-MM-DD (data after this date excluded)")
    parser.add_argument("--v1", action="store_true", default=False,
                        help="Use V1 trainer (legacy, 10 features). Default: V2 (33 features + Optuna)")
    parser.add_argument("--no-optuna", action="store_true", default=False,
                        help="Disable Optuna HPO (faster but less accurate)")
    parser.add_argument("--no-sector", action="store_true", default=False,
                        help="Skip sector RS features (faster)")
    parser.add_argument("--no-fno", action="store_true", default=False,
                        help="Skip F&O features (faster, use if no bhav data)")
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Custom models directory (default: models/). "
                             "Use to save OOS comparison models separately.")
    args = parser.parse_args()

    # Determine models directory
    models_dir = args.models_dir if args.models_dir else MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)

    # Build config pointing to the right models directory
    from types import SimpleNamespace
    versions_json = os.path.join(models_dir, "versions.json")
    trainer_config = SimpleNamespace(
        models_dir=models_dir,
        model_versions_path=versions_json,
        prediction_horizon=20,
        transaction_cost=0.0012,
    )

    if args.v1:
        from core.models.training.trainer import MARK5MLTrainer
        trainer = MARK5MLTrainer(config=trainer_config)
        trainer_label = "V1 (10 features)"
    else:
        from core.models.training.trainer_v2 import MARK5MLTrainerV2
        trainer = MARK5MLTrainerV2(
            config=trainer_config,
            use_optuna=not args.no_optuna,
            optuna_trials=args.trials,
            include_sector=not args.no_sector,
            include_fno=not args.no_fno,
        )
        trainer_label = f"V2 (33 features, Optuna={'ON' if not args.no_optuna else 'OFF'})"

    results = {}
    passed, failed, skipped = [], [], []

    print(f"\n{'═'*70}")
    print(f"  MARK5 FULL MODEL RETRAIN — {len(args.tickers)} tickers")
    print(f"  Trainer: {trainer_label}")
    print(f"  Optuna trials: {args.trials} per ticker")
    print(f"  Cutoff: {args.cutoff or 'None (use all data)'}")
    print(f"  Models dir: {models_dir}")
    feat_desc = "33-feature V2" if not args.v1 else "10-feature V1"
    print(f"  Features: {feat_desc}")
    print(f"  Labels: Triple barrier PT/SL=[3.5,1.5] | CPCV N=5 k=2 | embargo=20")
    print(f"{'═'*70}\n")

    start = datetime.now()
    for i, ticker in enumerate(args.tickers, 1):
        print(f"\n[{i}/{len(args.tickers)}] ──── {ticker} ────")
        r = retrain_ticker(ticker, trainer, cutoff=args.cutoff, n_trials=args.trials)
        results[ticker] = r
        status = r.get("status", "failed")
        if status == "success":
            fold_sharpe = r.get("mean_sharpe", r.get("mean_fold_sharpe", r.get("avg_sharpe", 0))) or 0
            auc         = r.get("auc", 0.0) or 0.0
            n_feats     = r.get("n_features", "?")
            feat_ver    = r.get("feature_engine", "v1")
            optuna_used = r.get("optuna_used", False)
            prod = "✅ PROD" if r.get("passes_prod_gate", False) else "⚠️ GATE-FAIL"
            print(f"  ✅ DONE | sharpe={fold_sharpe:+.3f} | AUC={auc:.4f} | "
                  f"features={n_feats}({feat_ver}) | optuna={optuna_used} | {prod}")
            passed.append(ticker)
        elif status == "skipped":
            print(f"  ⏭  SKIPPED: {r.get('reason','?')}")
            skipped.append(ticker)
        else:
            print(f"  ❌ FAILED: {r.get('reason','?')[:80]}")
            failed.append(ticker)

    elapsed = (datetime.now() - start).seconds
    print(f"\n{'═'*70}")
    print(f"  RETRAIN COMPLETE  ({elapsed}s)")
    print(f"  Passed : {len(passed)}/{len(args.tickers)}  {passed}")
    print(f"  Failed : {len(failed)}  {failed}")
    print(f"  Skipped: {len(skipped)}  {skipped}")
    print(f"{'═'*70}\n")

    # Save summary
    suffix = f"_cutoff{args.cutoff.replace('-','')}" if args.cutoff else ""
    if args.models_dir:
        dir_tag = "_" + os.path.basename(args.models_dir.rstrip("/"))
        suffix += dir_tag
    out = os.path.join(_ROOT, "reports", f"retrain_results{suffix}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({k: {kk: str(vv) for kk, vv in v.items()} for k, v in results.items()},
                  f, indent=2)
    print(f"  Retrain summary → {out}")
    return results

if __name__ == "__main__":
    main()

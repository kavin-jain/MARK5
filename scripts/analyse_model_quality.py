#!/usr/bin/env python3
"""
MARK5 Model Signal Quality Analyser
Computes OOS AUC per ticker to identify marginal vs strong models.

Methodology
-----------
For each ticker in the OOS period (2022-2026):
  - Load the V2 model from models_v2_oos/
  - Compute ML confidence scores via LightPredictor.predict_proba()
  - Compute 21-day forward return for each date
  - Label: 1 if forward return > 0.5% (profitable trade threshold), 0 otherwise
  - AUC: roc_auc_score(labels, confidence_scores)

Interpretation
--------------
  AUC > 0.56 : STRONG  — model has genuine directional edge
  AUC 0.52–0.56 : MARGINAL — some edge, monitor carefully
  AUC <= 0.52 : WEAK   — near-random, consider removing from universe

Usage
-----
  python3 scripts/analyse_model_quality.py --models-dir models_v2_oos
  python3 scripts/analyse_model_quality.py --models-dir models_v2_oos --start 2022-01-01 --end 2026-01-01
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.data.nse_data_provider import fetch_equity_ohlcv
from core.models.backtest_pipeline import LightPredictor
from core.models.features import engineer_features_df

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [QUALITY] | %(levelname)s | %(message)s",
)
logger = logging.getLogger("MARK5.ModelQuality")

# Default tickers — V2 PROD universe
V2_PROD_TICKERS = [
    "ASIANPAINT", "AUBANK", "BAJFINANCE", "COFORGE", "HAL",
    "ICICIBANK", "MARUTI", "RELIANCE", "TATAELXSI", "TATASTEEL",
    "TCS", "TRENT", "YESBANK",
]

FORWARD_BARS       = 21    # ~1 trading month
MIN_RETURN_PCT     = 0.5   # label = 1 if 21-day return > 0.5%
AUC_THRESHOLD_WEAK = 0.52  # below this → WEAK / consider removing
AUC_THRESHOLD_GOOD = 0.56  # above this → STRONG


def _fetch_features(ticker: str, start: str, end: str) -> tuple:
    """Fetch OHLCV and engineer features for a ticker over OOS period."""
    # Fetch extra history so rolling indicators are warm at start
    extended_start = pd.Timestamp(start) - pd.DateOffset(years=1)
    df = fetch_equity_ohlcv(ticker, extended_start.strftime("%Y-%m-%d"), end)
    if df is None or df.empty:
        return None, None

    try:
        feat_df = engineer_features_df(df, context={}, is_daily=True)
    except Exception as e:
        logger.warning(f"[{ticker}] Feature engineering failed: {e}")
        return None, None

    # Trim to OOS window
    oos_mask = feat_df.index >= pd.Timestamp(start)
    feat_oos = feat_df.loc[oos_mask].copy()
    df_oos   = df.loc[df.index >= pd.Timestamp(start)].copy()
    return df_oos, feat_oos


def _compute_labels(df_oos: pd.DataFrame, forward_bars: int = FORWARD_BARS,
                    min_return_pct: float = MIN_RETURN_PCT) -> pd.Series:
    """Compute binary labels: 1 if forward return > threshold, else 0."""
    fwd_return = df_oos["close"].shift(-forward_bars) / df_oos["close"] - 1.0
    labels = (fwd_return * 100 >= min_return_pct).astype(int)
    return labels


def _signal_quality_label(auc: float) -> str:
    if auc > AUC_THRESHOLD_GOOD:
        return "STRONG — clear directional edge"
    elif auc > AUC_THRESHOLD_WEAK:
        return "MARGINAL — some edge, monitor"
    else:
        return "WEAK — near-random, consider removing"


def analyse_ticker(ticker: str, models_dir: str, start: str, end: str) -> dict:
    """Run full AUC analysis for a single ticker."""
    predictor = LightPredictor(ticker, models_dir)
    if not predictor.has_models():
        logger.warning(f"[{ticker}] No models found in {models_dir}/")
        return {
            "ticker": ticker,
            "auc": None,
            "n_samples": 0,
            "pct_above_hurdle": None,
            "signal_quality": "NO MODEL",
        }

    df_oos, feat_oos = _fetch_features(ticker, start, end)
    if df_oos is None or feat_oos is None or feat_oos.empty:
        logger.warning(f"[{ticker}] No OOS data available")
        return {
            "ticker": ticker,
            "auc": None,
            "n_samples": 0,
            "pct_above_hurdle": None,
            "signal_quality": "NO DATA",
        }

    labels = _compute_labels(df_oos)

    # Compute AUC via validate_signal_quality on OOS features + forward labels
    auc = predictor.validate_signal_quality(feat_oos, labels)

    # Also compute confidence distribution stats
    try:
        preds = predictor.predict_proba(feat_oos)
        pct_above = float((preds >= 0.52).mean())
        conf_std  = float(np.std(preds))
        conf_max  = float(np.max(preds))
    except Exception:
        pct_above = None
        conf_std  = None
        conf_max  = None

    n_valid = int(feat_oos.index.intersection(labels.dropna().index).shape[0])

    quality_label = _signal_quality_label(auc)
    return {
        "ticker":          ticker,
        "auc":             round(auc, 4),
        "n_samples":       n_valid,
        "pct_above_hurdle": round(pct_above, 3) if pct_above is not None else None,
        "conf_std":        round(conf_std, 4) if conf_std is not None else None,
        "conf_max":        round(conf_max, 4) if conf_max is not None else None,
        "signal_quality":  quality_label,
    }


def print_table(results: list) -> None:
    """Pretty-print the quality table."""
    header = f"{'Ticker':<15} | {'AUC':>6} | {'N':>5} | {'%>0.52':>7} | {'Signal Quality'}"
    divider = "-" * len(header)
    print("\n" + "=" * len(header))
    print("  MARK5 Model Signal Quality Report (OOS AUC Analysis)")
    print("=" * len(header))
    print(header)
    print(divider)
    for r in results:
        auc_str  = f"{r['auc']:.4f}" if r['auc'] is not None else "  N/A "
        n_str    = str(r["n_samples"]) if r["n_samples"] else " N/A"
        pct_str  = f"{r['pct_above_hurdle']:.1%}" if r["pct_above_hurdle"] is not None else "  N/A "
        print(f"  {r['ticker']:<13} | {auc_str:>6} | {n_str:>5} | {pct_str:>7} | {r['signal_quality']}")
    print("=" * len(header))

    # Summary
    valid = [r for r in results if r["auc"] is not None]
    if valid:
        weak     = [r for r in valid if r["auc"] <= AUC_THRESHOLD_WEAK]
        marginal = [r for r in valid if AUC_THRESHOLD_WEAK < r["auc"] <= AUC_THRESHOLD_GOOD]
        strong   = [r for r in valid if r["auc"] > AUC_THRESHOLD_GOOD]
        print(f"\n  Strong (AUC > {AUC_THRESHOLD_GOOD}): {[r['ticker'] for r in strong]}")
        print(f"  Marginal ({AUC_THRESHOLD_WEAK} < AUC <= {AUC_THRESHOLD_GOOD}): {[r['ticker'] for r in marginal]}")
        print(f"  Weak (AUC <= {AUC_THRESHOLD_WEAK}): {[r['ticker'] for r in weak]}")
        if weak:
            print(f"\n  RECOMMENDATION: Consider removing weak tickers from the portfolio universe:")
            for r in weak:
                print(f"    - {r['ticker']} (AUC={r['auc']:.4f})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="MARK5 Model Signal Quality Analyser — OOS AUC per ticker"
    )
    parser.add_argument("--models-dir", default="models_v2_oos",
                        help="Directory containing trained models")
    parser.add_argument("--tickers", nargs="+", default=V2_PROD_TICKERS,
                        help="Tickers to analyse")
    parser.add_argument("--start", default="2022-01-01",
                        help="OOS start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-01-01",
                        help="OOS end date (YYYY-MM-DD)")
    parser.add_argument("--output", default="reports/model_signal_quality.json",
                        help="Output JSON path")
    args = parser.parse_args()

    models_dir = os.path.join(_ROOT, args.models_dir)
    if not os.path.isdir(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        sys.exit(1)

    output_path = os.path.join(_ROOT, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"Analysing {len(args.tickers)} tickers | OOS: {args.start} → {args.end}")
    logger.info(f"Models dir: {models_dir}")

    results = []
    for ticker in args.tickers:
        logger.info(f"--- {ticker} ---")
        result = analyse_ticker(ticker, models_dir, args.start, args.end)
        results.append(result)

    print_table(results)

    output = {
        "meta": {
            "models_dir":   args.models_dir,
            "oos_start":    args.start,
            "oos_end":      args.end,
            "forward_bars": FORWARD_BARS,
            "label_threshold_pct": MIN_RETURN_PCT,
            "auc_threshold_weak":  AUC_THRESHOLD_WEAK,
            "auc_threshold_good":  AUC_THRESHOLD_GOOD,
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

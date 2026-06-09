"""
MARK5 — ML Model Directional Accuracy Diagnostic
=================================================
For each production ticker, directly measures:
  1. ML model directional accuracy on 2022-2026 OOS data
  2. Average return WHEN ML predicts BUY (confidence > 0.52) 20 days forward
  3. Transaction-cost breakeven analysis

This bypasses the walk-forward simulation to check raw signal quality.
"""
import os, sys, json, logging, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

PROD_TICKERS = [
    "ASIANPAINT", "AUBANK", "BAJFINANCE", "BHARTIARTL", "COFORGE",
    "HAL", "PNB", "RELIANCE", "TATAELXSI", "TATASTEEL",
    "TCS", "TRENT", "YESBANK"
]

ML_HURDLE  = 0.52
COST_PCT   = 0.29    # Round-trip delivery costs
OOS_START  = "2022-01-01"
OOS_END    = "2026-05-21"
HORIZONS   = [5, 10, 20]   # bars forward to measure return

CACHE_DIR  = os.path.join(_ROOT, "data", "cache")

def load_cache(ticker):
    for suffix in ["_daily.parquet", "_NS_1d.parquet"]:
        path = os.path.join(CACHE_DIR, f"{ticker}{suffix}")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            if hasattr(df.index, "tz") and df.index.tz:
                from zoneinfo import ZoneInfo
                df.index = df.index.tz_convert(ZoneInfo("Asia/Kolkata")).tz_localize(None)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            return df
    return None


def run_diagnostic():
    from core.models.backtest_pipeline import LightPredictor
    from core.models.features import engineer_features_df, FEATURE_COLS

    print("="*90)
    print("MARK5 — ML DIRECTIONAL ACCURACY DIAGNOSTIC (2022-2026 OOS)")
    print("Measures: When ML predicts BUY (conf≥0.52), does the stock go up?")
    print("="*90)
    print(f"{'TICKER':<14} {'ML_ACC%':>8} {'AVG_5D':>8} {'AVG_10D':>9} {'AVG_20D':>9} "
          f"{'N_SIGNALS':>10} {'ML_ACTIVE_FOLDS':>16}")
    print("-"*90)

    ticker_results = []
    for ticker in PROD_TICKERS:
        df = load_cache(ticker)
        if df is None:
            print(f"{ticker:<14} NO CACHE")
            continue

        # OOS slice
        df_oos = df.loc[OOS_START:OOS_END].copy()
        if len(df_oos) < 100:
            print(f"{ticker:<14} TOO FEW ROWS ({len(df_oos)})")
            continue

        # Engineer features on full history (no leakage: use full df for warmup, slice for OOS)
        try:
            feat = engineer_features_df(df, is_daily=True)
            feat_oos = feat.loc[OOS_START:OOS_END]
        except Exception as e:
            print(f"{ticker:<14} FEAT_ERROR: {e}")
            continue

        # Load ML model
        pred = LightPredictor(ticker, "models")
        if not pred.has_models():
            print(f"{ticker:<14} NO MODELS")
            continue

        # Get ML predictions on OOS
        try:
            proba = pred.predict_proba(feat_oos)
            ml_conf = pd.Series(proba, index=feat_oos.index)
        except Exception as e:
            print(f"{ticker:<14} PRED_ERROR: {e}")
            continue

        # ML diagnostics
        prob_std = float(ml_conf.std())
        prob_max = float(ml_conf.max())
        is_active = (prob_std > 0.005) and (prob_max > ML_HURDLE)

        if not is_active:
            print(f"{ticker:<14} {'ML FLAT':>8} std={prob_std:.4f} max={prob_max:.3f}")
            ticker_results.append({"ticker": ticker, "ml_active": False})
            continue

        # Filter signal bars (ML BUY signals above hurdle)
        buy_mask = ml_conf >= ML_HURDLE
        signal_dates = ml_conf[buy_mask].index
        n_signals = len(signal_dates)

        if n_signals < 5:
            print(f"{ticker:<14} {'FEW SIGS':>8} n={n_signals}")
            ticker_results.append({"ticker": ticker, "ml_active": True, "n_signals": n_signals})
            continue

        # For each signal bar, measure forward return (after costs)
        close = df_oos["close"]
        returns_by_horizon = {h: [] for h in HORIZONS}
        directional_correct = 0

        for sig_date in signal_dates:
            try:
                loc = df_oos.index.get_loc(sig_date)
                entry_price = df_oos["open"].iloc[loc + 1] if loc + 1 < len(df_oos) else close.iloc[loc]
                for h in HORIZONS:
                    exit_loc = min(loc + h + 1, len(df_oos) - 1)
                    exit_price = close.iloc[exit_loc]
                    gross_return = (exit_price / entry_price - 1) * 100
                    net_return = gross_return - COST_PCT  # subtract round-trip costs
                    returns_by_horizon[h].append(net_return)
                # Directional: 10-day forward
                fwd10 = min(loc + 11, len(df_oos) - 1)
                if close.iloc[fwd10] > entry_price:
                    directional_correct += 1
            except (IndexError, KeyError):
                pass

        # Compute stats
        acc = directional_correct / n_signals * 100
        avg_returns = {h: np.mean(returns_by_horizon[h]) if returns_by_horizon[h] else 0
                       for h in HORIZONS}

        flag5  = "✅" if avg_returns[5]  > 0 else "❌"
        flag10 = "✅" if avg_returns[10] > 0 else "❌"
        flag20 = "✅" if avg_returns[20] > 0 else "❌"
        acc_flag = "✅" if acc >= 50 else "❌"

        print(f"{ticker:<14} {acc:>7.1f}%{acc_flag} "
              f"{avg_returns[5]:>+7.2f}%{flag5} "
              f"{avg_returns[10]:>+7.2f}%{flag10} "
              f"{avg_returns[20]:>+7.2f}%{flag20} "
              f"{n_signals:>10} "
              f"  std={prob_std:.4f} ACTIVE")

        ticker_results.append({
            "ticker": ticker, "ml_active": True, "accuracy_10d_pct": acc,
            "avg_5d_net": avg_returns[5], "avg_10d_net": avg_returns[10],
            "avg_20d_net": avg_returns[20], "n_signals": n_signals,
            "prob_std": prob_std, "prob_max": prob_max,
        })

    # Portfolio summary
    active = [r for r in ticker_results if r.get("ml_active") and r.get("n_signals", 0) >= 5]
    print("\n" + "="*90)
    print("SUMMARY:")
    if active:
        avg_acc = np.mean([r["accuracy_10d_pct"] for r in active])
        avg_20d = np.mean([r["avg_20d_net"] for r in active])
        profitable = sum(1 for r in active if r["avg_20d_net"] > 0)
        print(f"  Tickers with active ML : {len(active)}/{len(PROD_TICKERS)}")
        print(f"  Mean directional acc   : {avg_acc:.1f}% (need ≥50% for positive EV)")
        print(f"  Mean 20-day net return : {avg_20d:+.2f}% per signal (after {COST_PCT}% costs)")
        print(f"  Tickers with +ve 20d   : {profitable}/{len(active)}")

        # Annualized projection
        if avg_20d > 0:
            signals_per_ticker_per_year = np.mean([r["n_signals"] / 4 for r in active])
            # If 10% allocation per signal
            ann_per_ticker = signals_per_ticker_per_year * avg_20d * 0.10
            print(f"\n  ANNUAL RETURN PROJECTION (10% alloc per signal):")
            print(f"    Signals/ticker/year    : {signals_per_ticker_per_year:.1f}")
            print(f"    Expected return/signal : {avg_20d:+.2f}%")
            print(f"    Annual gross (×10%pos) : {ann_per_ticker:+.2f}% per ticker")
            print(f"    Net after 20% STCG     : {ann_per_ticker * 0.8:+.2f}%")
    print("="*90)


if __name__ == "__main__":
    run_diagnostic()

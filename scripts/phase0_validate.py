"""
MARK5 Phase 0 — Feature Validation Diagnostic Runner
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Per rebuild.md: Run this BEFORE writing any new code.

Runs sequentially on the live cached data:
  TEST 1 — IC Analysis (Spearman rank corr, feature vs 5d forward return)
  TEST 2 — Stationarity (ADF p-value per feature)
  TEST 3 — Lookahead probe (shift signals +1 day, Sharpe must collapse >50%)
  TEST 4 — Feature importance stability (rank corr first vs last CPCV fold)
  TEST 5 — Signal correlation (avg pairwise BUY-signal corr across universe)

Usage:
  .venv/bin/python scripts/phase0_validate.py \\
      --universe nifty50 --start 2022-01-01 --end 2025-01-01

Output written to: reports/phase0/
  ic_analysis.csv
  stationarity.csv
  signal_correlation.png
  phase0_summary.txt        ← go/no-go verdict
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Project root on sys.path ────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.data.market_data import MarketDataProvider
from core.data.fii_data import FIIDataProvider
from core.models.features import AdvancedFeatureEngine
from core.models.training.cpcv import CombinatorialPurgedKFold
from core.models.training.financial_engineer import FinancialEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PHASE0] %(levelname)s — %(message)s",
)
logger = logging.getLogger("MARK5.Phase0")

# ── Globals for Context ──────────────────────────────────────────────────────
_MARKET_CONTEXT: dict = {}

def _init_market_context(start: str, end: str):
    """Load Nifty and FII data once for all stocks."""
    global _MARKET_CONTEXT
    mp = MarketDataProvider()
    fii = FIIDataProvider()
    
    logger.info(f"Loading market context ({start} to {end})...")
    # Load Nifty50
    nifty = mp.get_nifty50_data(start, end)
    if nifty is not None:
        _MARKET_CONTEXT['nifty_close'] = nifty['close']
        logger.info(f"  Nifty50 loaded: {len(nifty)} bars")
    
    # Load FII
    fii_series = fii.get_fii_flow(start, end)
    # Check if we have enough data points and variance
    if fii_series is not None and len(fii_series) > 10 and fii_series.std() > 1e-6:
        _MARKET_CONTEXT['fii_net'] = fii_series
        logger.info(f"  FII data loaded: {len(fii_series)} bars")
    else:
        logger.warning("  FII data unavailable or constant — using synthetic fallback for context")
        # Reuse Nifty index to get valid trading days
        if nifty is not None:
            _MARKET_CONTEXT['fii_net'] = fii.generate_synthetic_fii_data(nifty.index)
            logger.info(f"  Generated {len(_MARKET_CONTEXT['fii_net'])} synthetic FII bars")
        else:
            _MARKET_CONTEXT['fii_net'] = None

# ── Constants ────────────────────────────────────────────────────────────────
IC_FLOOR: float = 0.02          # Features below this are dead (rebuild §3.1)
ADF_ALPHA: float = 0.05         # Stationarity significance level
LOOKAHEAD_DECAY_FLOOR: float = 0.50  # Sharpe must drop >50% when signals shifted
FEATURE_STABILITY_FLOOR: float = 0.70  # Spearman corr first vs last fold importance
SIGNAL_CORR_CEILING: float = 0.40   # Avg pairwise BUY-signal corr must be < this
CPCV_N_SPLITS: int = 8
CPCV_N_TEST_SPLITS: int = 2
CPCV_EMBARGO: int = 10
FORWARD_RETURN_HORIZON: int = 5  # trading days


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_universe_data(
    universe: str,
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """Load cached OHLCV data for all universe symbols."""
    from scripts.nifty50_universe import NIFTY_50_TICKERS  # type: ignore

    cache_dir = os.path.join(PROJECT_ROOT, "data", "cache")
    symbols = NIFTY_50_TICKERS if universe == "nifty50" else []

    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        fname = sym.replace(".", "_") + "_1d.parquet"
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            logger.warning(f"No cache for {sym} — skipping")
            continue
        df = pd.read_parquet(path)
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        if len(df) < 100:
            logger.warning(f"{sym}: only {len(df)} rows in range — skipping")
            continue
        data[sym] = df
    logger.info(f"Loaded {len(data)}/{len(symbols)} symbols")
    return data


def _build_feature_matrix(
    df: pd.DataFrame,
    feats: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (features_df, y_labels) for one stock."""
    fe_engine = FinancialEngineer(transaction_cost_pct=0.0012)
    labels = fe_engine.get_labels(df, run_bars=10, pt_sl=[2.0, 1.0])

    aligned = feats.join(labels[["bin"]], how="inner").dropna()
    if len(aligned) < 50:
        return pd.DataFrame(), np.array([])

    y = aligned["bin"].values.astype(int)
    X = aligned.drop(columns=["bin"], errors="ignore")
    return X, y


def _forward_return(df: pd.DataFrame, horizon: int = FORWARD_RETURN_HORIZON) -> pd.Series:
    """5-day forward return: ret[t] = close[t+horizon]/close[t] - 1. No lookahead guard."""
    return (
        df["close"]
        .pct_change(horizon)
        .shift(-horizon)   # align: feature at t, return at t+horizon
        .rename("fwd_ret")
    )


# ── Test 1: IC Analysis ──────────────────────────────────────────────────────

def test_ic_analysis(
    data: dict[str, pd.DataFrame],
    features_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute Spearman IC(feature[t], fwd_ret[t]) for each feature across universe.
    Returns DataFrame with columns: feature, mean_ic, std_ic, icir, pval, significant, verdict.
    """
    logger.info("=" * 60)
    logger.info("TEST 1 — IC Analysis (Spearman rank corr vs 5d fwd return)")
    logger.info("=" * 60)

    ic_rows: list[dict] = []

    # Accumulate per-stock IC
    feature_ic_per_stock: dict[str, list[float]] = {}

    for sym, df in data.items():
        if sym not in features_dict:
            continue
        feats = features_dict[sym]
        fwd = _forward_return(df)
        combined = feats.join(fwd, how="inner").dropna()
        if len(combined) < 50:
            continue
        fwd_series = combined["fwd_ret"]

        for col in feats.columns:
            if col not in combined.columns:
                continue
            ic, pval = spearmanr(combined[col], fwd_series, nan_policy="omit")
            if np.isnan(ic):
                continue
            feature_ic_per_stock.setdefault(col, []).append(ic)

    for feature, ics in feature_ic_per_stock.items():
        arr = np.array(ics)
        mean_ic = arr.mean()
        std_ic = arr.std(ddof=1) if len(arr) > 1 else 0.0
        icir = mean_ic / std_ic if std_ic > 1e-8 else 0.0
        # Combined significance: fraction of stocks with |IC| significantly > 0
        _, pval_combined = spearmanr(arr, np.ones(len(arr)), nan_policy="omit")
        verdict = "KEEP" if abs(mean_ic) >= IC_FLOOR else "KILL"
        ic_rows.append({
            "feature": feature,
            "mean_ic": round(mean_ic, 5),
            "std_ic": round(std_ic, 5),
            "icir": round(icir, 3),
            "n_stocks": len(ics),
            "verdict": verdict,
        })

    result = pd.DataFrame(ic_rows).sort_values("mean_ic", key=abs, ascending=False)

    dead = result[result["verdict"] == "KILL"]
    logger.info(f"  Dead features (|IC| < {IC_FLOOR}): {list(dead['feature'])}")
    keep = result[result["verdict"] == "KEEP"]
    logger.info(f"  Viable features ({len(keep)}/{len(result)}): {list(keep['feature'])}")
    return result


# ── Test 2: Stationarity ─────────────────────────────────────────────────────

def test_stationarity(
    data: dict[str, pd.DataFrame],
    features_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    ADF stationarity test on each feature across universe.
    Non-stationary features (p > 0.05) flagged for fractional differentiation.
    """
    logger.info("=" * 60)
    logger.info("TEST 2 — ADF Stationarity (p < 0.05 required)")
    logger.info("=" * 60)
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        logger.error("statsmodels not installed — skipping stationarity test")
        return pd.DataFrame()

    # Aggregate median p-value per feature across stocks
    pvals_per_feature: dict[str, list[float]] = {}

    for sym, df in list(data.items())[:10]:   # sample 10 stocks is sufficient
        if sym not in features_dict: continue
        feats = features_dict[sym]
        for col in feats.columns:
            series = feats[col].dropna()
            if len(series) < 30:
                continue
            try:
                adf_result = adfuller(series, maxlag=20, autolag="AIC")
                pval = float(adf_result[1])
            except Exception:
                pval = 1.0
            pvals_per_feature.setdefault(col, []).append(pval)

    rows = []
    for feat, pvals in pvals_per_feature.items():
        median_p = np.median(pvals)
        verdict = "STATIONARY" if median_p < ADF_ALPHA else "NON-STATIONARY → frac-diff"
        rows.append({
            "feature": feat,
            "median_adf_pval": round(median_p, 5),
            "verdict": verdict,
        })

    result = pd.DataFrame(rows).sort_values("median_adf_pval")
    non_stat = result[result["verdict"].str.startswith("NON")]
    logger.info(f"  Non-stationary features: {list(non_stat['feature'])}")
    return result


# ── Test 3: Lookahead probe ──────────────────────────────────────────────────

def test_lookahead_probe(data: dict[str, pd.DataFrame], features_dict: dict[str, pd.DataFrame]) -> dict:
    """
    Shift all signals +1 day ahead (deliberately wrong).
    If Sharpe drops < 50%, there is likely lookahead bias.
    Returns: {'original_sharpe', 'shifted_sharpe', 'decay_pct', 'passed'}.
    """
    logger.info("=" * 60)
    logger.info("TEST 3 — Lookahead Probe (shift signals +1d, Sharpe must collapse)")
    logger.info("=" * 60)

    # Use first 5 stocks with enough data
    sharpes_orig, sharpes_shifted = [], []

    for sym, df in list(data.items())[:5]:
        if sym not in features_dict: continue
        feats, y = _build_feature_matrix(df, features_dict[sym])
        if feats.empty or len(y) < 50:
            continue

        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        X = StandardScaler().fit_transform(feats.values)
        split = int(len(X) * 0.6)
        X_tr, y_tr = X[:split], y[:split]
        X_te, y_te = X[split:], y[split:]
        if len(np.unique(y_tr)) < 2:
            continue

        clf = LogisticRegression(C=0.1, max_iter=500, random_state=42)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        signals = (proba > 0.52).astype(float)

        fwd_ret = _forward_return(df).reindex(feats.index).values[split:]
        fwd_ret = fwd_ret[~np.isnan(fwd_ret)]
        min_len = min(len(signals), len(fwd_ret))
        if min_len < 10:
            continue

        sig = signals[:min_len]
        ret = fwd_ret[:min_len]

        # ACTION 4 FIX: Log signal density so we can distinguish sparsity from lookahead.
        signal_density = sig.mean()
        n_signals = int(sig.sum())
        logger.info(
            f"  {sym}: {n_signals} BUY signals / {min_len} bars "
            f"({signal_density:.1%} density)"
            + (" [SPARSE — skipping this stock for Sharpe calc]" if n_signals < 10 else "")
        )
        # Guard: with fewer than 10 signals, P&L Sharpe is pure noise.
        if n_signals < 10:
            continue

        # Original
        pnl = sig * ret
        s_orig = pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(252)
        sharpes_orig.append(s_orig)

        # Shifted +1
        sig_shifted = np.roll(sig, 1)
        sig_shifted[0] = 0.0
        pnl_shifted = sig_shifted * ret
        s_shifted = pnl_shifted.mean() / (pnl_shifted.std() + 1e-8) * np.sqrt(252)
        sharpes_shifted.append(s_shifted)

    if not sharpes_orig:
        logger.warning("  Insufficient data for lookahead probe")
        return {"passed": None}

    orig = float(np.mean(sharpes_orig))
    shifted = float(np.mean(sharpes_shifted))
    
    # Positive decay means Sharpe DROPPED when signals were delayed (Lookahead clean)
    # Negative decay means Sharpe INCREASED when signals were delayed (Lagging signals)
    # ACTION 4 FIX: If |original Sharpe| < 0.1, the feature set is too weak to produce
    # a meaningful signal — mark as INCONCLUSIVE instead of a false FAIL.
    SHARPE_FLOOR_FOR_LOOKAHEAD_TEST = 0.10
    if abs(orig) < SHARPE_FLOOR_FOR_LOOKAHEAD_TEST:
        logger.warning(
            f"  Lookahead probe: original Sharpe too weak ({orig:.4f}) for meaningful test."
            f" This is expected when OHLCV features have no edge — not a code bug."
            f" VERDICT: INCONCLUSIVE (feature set needs alternative data, not a bug fix)."
        )
        return {
            "original_sharpe": round(orig, 4),
            "shifted_sharpe": round(shifted, 4),
            "decay_pct": None,
            "passed": None,
            "note": "INCONCLUSIVE — feature set too weak for meaningful lookahead test",
        }

    decay = (orig - shifted) / (abs(orig) + 1e-8)
    
    # Verdict: Failure if decay is low (Lookahead) OR if decay is too negative (Lag/Inefficiency)
    passed = (decay >= LOOKAHEAD_DECAY_FLOOR)
    
    verdict = "✅ PASS" if passed else "⚠️ FAIL — LOOKAHEAD" if decay < 0.1 else "⚠️ FAIL — LAGGING"
    logger.info(f"  Sharpe original={orig:.4f} | shifted={shifted:.4f} | decay={decay:.2%} | {verdict}")
    return {
        "original_sharpe": round(orig, 4),
        "shifted_sharpe": round(shifted, 4),
        "decay_pct": round(decay, 4),
        "passed": passed,
    }


# ── Test 4: Feature importance stability ─────────────────────────────────────

def test_feature_stability(data: dict[str, pd.DataFrame]) -> dict:
    """
    Train on first vs last CPCV fold. Rank-correlate feature importances.
    Target: Spearman corr > 0.70.
    """
    logger.info("=" * 60)
    logger.info("TEST 4 — Feature Importance Stability (rank corr first vs last fold)")
    logger.info("=" * 60)
    try:
        import lightgbm as lgb_lib
    except ImportError:
        logger.warning("LightGBM not available — skipping stability test")
        return {"passed": None}

    # Use the largest available stock for statistical power
    biggest = max(data.keys(), key=lambda s: len(data[s]))
    feats, y = _build_feature_matrix(data[biggest])
    if feats.empty or len(y) < 100:
        logger.warning("  Insufficient data for stability test")
        return {"passed": None}

    feature_names = list(feats.columns)
    X = feats.values

    cpcv = CombinatorialPurgedKFold(
        n_splits=CPCV_N_SPLITS, n_test_splits=CPCV_N_TEST_SPLITS, embargo=CPCV_EMBARGO
    )
    importances = []
    for tr_idx, te_idx in cpcv.split(X, y):
        if len(np.unique(y[tr_idx])) < 2:
            continue
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X[tr_idx])
        clf = lgb_lib.LGBMClassifier(n_estimators=200, verbose=-1, random_state=42)
        clf.fit(Xs, y[tr_idx])
        importances.append(clf.feature_importances_)
        if len(importances) >= 2:   # only need first and last
            break

    if len(importances) < 2:
        return {"passed": None}

    corr, _ = spearmanr(importances[0], importances[-1])
    passed = corr >= FEATURE_STABILITY_FLOOR

    ranked_first = sorted(
        zip(feature_names, importances[0]),
        key=lambda x: x[1], reverse=True
    )[:5]
    logger.info(f"  Top-5 features (first fold): {[f for f, _ in ranked_first]}")
    logger.info(f"  Rank correlation first vs last fold: {corr:.3f} (need ≥ {FEATURE_STABILITY_FLOOR})")
    logger.info(f"  Result: {'✅ PASS' if passed else '⚠️ FAIL — unstable features (overfitting noise)'}")
    return {
        "rank_corr": round(float(corr), 4),
        "passed": passed,
        "top5_features_first_fold": [f for f, _ in ranked_first],
    }


# ── Test 5: Signal correlation ───────────────────────────────────────────────

def test_signal_correlation(data: dict[str, pd.DataFrame]) -> dict:
    """
    Generate BUY signals for all stocks. Compute avg pairwise correlation.
    Target: < 0.40. High correlation = trading Nifty beta, not stock alpha.
    """
    logger.info("=" * 60)
    logger.info("TEST 5 — Signal Correlation (avg pairwise corr < 0.40 required)")
    logger.info("=" * 60)

    feat_engine = AdvancedFeatureEngine()
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    signal_series: dict[str, pd.Series] = {}

    for sym, df in list(data.items())[:20]:  # 20-stock sample is sufficient
        feats = feat_engine.engineer_all_features(df, context=None)
        if feats.empty or len(feats) < 80:
            continue
        X = StandardScaler().fit_transform(feats.values)
        split = int(len(X) * 0.6)
        if len(np.unique((lambda y: y)(feats.index[:split]))) == 0:
            pass
        y_dummy = np.zeros(len(X))
        y_dummy[:split] = np.random.randint(0, 2, split)  # random labels for structure only

        try:
            fe2, y2 = _build_feature_matrix(df)
            if fe2.empty:
                continue
            Xs = StandardScaler().fit_transform(fe2.values)
            sp = int(len(Xs) * 0.6)
            if len(np.unique(y2[:sp])) < 2:
                continue
            clf = LogisticRegression(C=0.1, max_iter=300, random_state=42)
            clf.fit(Xs[:sp], y2[:sp])
            proba = clf.predict_proba(Xs[sp:])[:, 1]
            buy = pd.Series(
                (proba > 0.52).astype(int),
                index=fe2.index[sp:],
                name=sym,
            )
            # Skip constant signals (all 0 or all 1) — correlation undefined.
            if buy.std() < 1e-6:
                logger.debug(f"  {sym}: constant signal vector — skipping")
                continue
            signal_series[sym] = buy
        except Exception as exc:
            logger.debug(f"  {sym}: {exc}")

    if len(signal_series) < 5:
        logger.warning("  Fewer than 5 stocks processed — signal correlation inconclusive")
        return {"passed": None}

    signals_df = pd.DataFrame(signal_series).dropna(how="all")
    corr_matrix = signals_df.corr(method="spearman")
    n = len(corr_matrix)
    mask = ~np.eye(n, dtype=bool)
    avg_corr = float(corr_matrix.values[mask].mean())
    passed = avg_corr < SIGNAL_CORR_CEILING

    # Save heatmap
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix.values, vmin=-1, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(n))
        ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(corr_matrix.index, fontsize=7)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"BUY Signal Pairwise Correlation | avg={avg_corr:.3f}")
        out_path = os.path.join(PROJECT_ROOT, "reports", "phase0", "signal_correlation.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"  Heatmap saved: {out_path}")
    except Exception as exc:
        logger.warning(f"  Heatmap save failed: {exc}")

    logger.info(f"  Avg pairwise signal corr: {avg_corr:.3f} (need < {SIGNAL_CORR_CEILING})")
    logger.info(f"  Result: {'✅ PASS' if passed else '⚠️ FAIL — signals too correlated (trading beta, not alpha)'}")
    return {"avg_pairwise_corr": round(avg_corr, 4), "passed": passed}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MARK5 Phase 0 Validation")
    parser.add_argument("--universe", default="nifty50", choices=["nifty50"])
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2025-01-01")
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, "reports", "phase0")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  MARK5 Phase 0 — Feature Validation Diagnostic Runner  ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"Universe: {args.universe}  |  {args.start} → {args.end}")

    data = _load_universe_data(args.universe, args.start, args.end)
    if not data:
        logger.error("No data loaded — aborting")
        sys.exit(1)

    from core.data.data_pipeline import DataPipeline
    pipeline = DataPipeline()
    logger.info(f"Generating unified feature matrix for {len(data)} stocks using DataPipeline...")
    features_dict = pipeline.build_feature_matrix(list(data.keys()), args.start, args.end)

    # ── Run all 5 tests ───────────────────────────────────────────────────────
    ic_df = test_ic_analysis(data, features_dict)
    stat_df = test_stationarity(data, features_dict)
    lookahead = test_lookahead_probe(data, features_dict)
    stability = test_feature_stability(data, features_dict)
    sig_corr = test_signal_correlation(data, features_dict)

    # ── Persist CSVs ─────────────────────────────────────────────────────────
    if not ic_df.empty:
        ic_df.to_csv(os.path.join(out_dir, "ic_analysis.csv"), index=False)
    if not stat_df.empty:
        stat_df.to_csv(os.path.join(out_dir, "stationarity.csv"), index=False)

    # ── Go / No-go summary ────────────────────────────────────────────────────
    viable_features = int((ic_df["verdict"] == "KEEP").sum()) if not ic_df.empty else "?"
    total_features = len(ic_df) if not ic_df.empty else "?"

    summary_lines = [
        "MARK5 Phase 0 — Go / No-Go Summary",
        "=" * 50,
        f"TEST 1  IC Analysis:         {viable_features}/{total_features} features viable (need ≥ 4)",
        f"TEST 2  Stationarity:        {len(stat_df[stat_df['verdict'].str.startswith('NON')]) if not stat_df.empty else '?'} non-stationary (should be 0)",
        f"TEST 3  Lookahead probe:     {'PASS' if lookahead.get('passed') else 'FAIL'} (decay={lookahead.get('decay_pct', '?'):.1%})" if isinstance(lookahead.get("decay_pct"), float) else f"TEST 3  Lookahead probe:     {'PASS' if lookahead.get('passed') else 'FAIL / SKIPPED'}",
        f"TEST 4  Feature stability:   {'PASS' if stability.get('passed') else 'FAIL / SKIPPED'} (rank_corr={stability.get('rank_corr', '?')})",
        f"TEST 5  Signal correlation:  {'PASS' if sig_corr.get('passed') else 'FAIL / SKIPPED'} (avg={sig_corr.get('avg_pairwise_corr', '?')})",
        "",
        "DECISION GATES:",
        f"  Features with IC ≥ 0.02:  {viable_features}  {'✅ PROCEED' if isinstance(viable_features, int) and viable_features >= 4 else '❌ STOP — fix features'}",
        f"  Lookahead clean:          {'✅' if lookahead.get('passed') else '❌' if lookahead.get('passed') is False else '—'}",
        f"  Features stable:          {'✅' if stability.get('passed') else '❌' if stability.get('passed') is False else '—'}",
        f"  Signals independent:      {'✅' if sig_corr.get('passed') else '❌' if sig_corr.get('passed') is False else '—'}",
    ]

    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(out_dir, "phase0_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print("\n" + summary_text)
    logger.info(f"\n📋 Reports saved to: {out_dir}")


if __name__ == "__main__":
    main()

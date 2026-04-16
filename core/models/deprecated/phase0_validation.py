#!/usr/bin/env python3
"""
MARK5 PHASE 0 VALIDATION v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE:
    Validate the current MARK5 system BEFORE building anything new.
    Run this script and read its output before touching a single line
    of features.py, trainer.py, or ensemble.py.

OUTPUTS:
    1. Feature IC / ICIR table  — which features have information content
    2. Stationarity tests (ADF) — which features need frac diff
    3. Three bias tests          — lookahead, importance stability, signal correlation
    4. CPCV Sharpe distribution  — actual out-of-sample performance
    5. Known bugs list           — deterministic audit findings
    6. Written report:           phase0_report.md

USAGE:
    python phase0_validation.py
    python phase0_validation.py --tickers RELIANCE.NS TCS.NS HDFCBANK.NS --period 5y
    python phase0_validation.py --skip-cpcv   # fast mode, skips CPCV
    python phase0_validation.py --output my_report.md

REQUIREMENTS:
    pip install yfinance lightgbm statsmodels scipy scikit-learn
    (all already in requirements.txt)

TRADING ROLE: Validation / diagnostic — no live trading path
SAFETY LEVEL: LOW — read-only analysis, no model saving
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — every magic number documented
# ─────────────────────────────────────────────────────────────────────────────

# Features with IC below this contribute pure noise; kill them
IC_KILL_THRESHOLD: float = 0.02

# ADF p-value: reject unit-root hypothesis → stationary
STATIONARITY_PVALUE: float = 0.05

# CPCV minimum bar: P(Sharpe > this threshold) must exceed PASS_RATE
CPCV_SHARPE_TARGET: float = 1.5
CPCV_SHARPE_PASS_RATE: float = 0.70

# Bias test 3: average pairwise signal correlation target
MAX_SIGNAL_CORRELATION: float = 0.40

# Bias test 2: feature importance rank correlation across folds
FEATURE_STABILITY_MIN: float = 0.70

# IC computed against this forward return horizon (trading days)
IC_HORIZON_DAYS: int = 5

# Minimum samples needed to compute meaningful statistics
MIN_SAMPLES_IC: int = 30
MIN_SAMPLES_CPCV: int = 100

# CPCV parameters
CPCV_N_SPLITS: int = 8        # C(8,2) = 28 test combinations
CPCV_N_TEST_SPLITS: int = 2
CPCV_EMBARGO_BARS: int = 10   # bars purged between train/test

# Default universe: representative Nifty 50 stocks
DEFAULT_TICKERS: List[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS", "HINDUNILVR.NS",
    "WIPRO.NS", "SUNPHARMA.NS", "AXISBANK.NS", "MARUTI.NS", "TATAMOTORS.NS",
    "BAJFINANCE.NS", "LT.NS", "HCLTECH.NS", "TITAN.NS", "TECHM.NS",
]

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MARK5.Phase0")

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS — graceful degradation
# ─────────────────────────────────────────────────────────────────────────────

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False
    logger.warning("yfinance not installed: pip install yfinance")

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False
    logger.warning("lightgbm not installed: pip install lightgbm")

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# MARK5 module imports — try the project layout first
_MARK5_FEATURES_OK = False
_MARK5_FE_OK = False
_MARK5_CPCV_OK = False

try:
    from core.models.features import AdvancedFeatureEngine
    _MARK5_FEATURES_OK = True
except ImportError:
    try:
        from features import AdvancedFeatureEngine  # root-level fallback
        _MARK5_FEATURES_OK = True
    except ImportError:
        logger.error("Cannot import AdvancedFeatureEngine — check PYTHONPATH")

try:
    from core.models.training.financial_engineer import FinancialEngineer
    _MARK5_FE_OK = True
except ImportError:
    try:
        from financial_engineer import FinancialEngineer
        _MARK5_FE_OK = True
    except ImportError:
        logger.warning("Cannot import FinancialEngineer — triple-barrier labels disabled")

try:
    from core.models.cpcv import CombinatorialPurgedKFold
    _MARK5_CPCV_OK = True
except ImportError:
    try:
        from cpcv import CombinatorialPurgedKFold
        _MARK5_CPCV_OK = True
    except ImportError:
        logger.warning("Cannot import CombinatorialPurgedKFold — CPCV disabled")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_data(tickers: List[str], period: str = "3y") -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data from yfinance and run basic quality checks.

    Args:
        tickers: NSE ticker symbols, e.g. ['RELIANCE.NS', 'TCS.NS']
        period:  Lookback period string, e.g. '3y', '5y'

    Returns:
        Dict mapping ticker → clean OHLCV DataFrame (lowercase columns).
        Tickers that fail quality checks are excluded with a logged warning.

    Raises:
        RuntimeError: If yfinance is not installed.
    """
    if not YFINANCE_OK:
        raise RuntimeError("Install yfinance: pip install yfinance")

    logger.info(f"\n{'='*60}")
    logger.info(f"DATA DOWNLOAD  ({len(tickers)} tickers, period={period})")
    logger.info(f"{'='*60}")

    data: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)

            if raw.empty:
                logger.warning(f"  {ticker}: empty download, skipping")
                continue

            # Flatten MultiIndex columns (yfinance >= 0.2)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw.columns = [c.lower() for c in raw.columns]

            # Require OHLCV
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(raw.columns):
                logger.warning(f"  {ticker}: missing columns {required - set(raw.columns)}")
                continue

            # Require minimum history
            if len(raw) < 150:
                logger.warning(f"  {ticker}: only {len(raw)} rows — need ≥150, skipping")
                continue

            # Reject if >5% NaN in close
            nan_pct = raw["close"].isna().mean()
            if nan_pct > 0.05:
                logger.warning(f"  {ticker}: {nan_pct:.1%} NaN in close, skipping")
                continue

            # Drop remaining NaN rows
            raw = raw.dropna(subset=list(required))

            data[ticker] = raw
            logger.info(
                f"  {ticker}: {len(raw)} rows  "
                f"{raw.index[0].date()} → {raw.index[-1].date()}"
            )

        except Exception as exc:
            logger.error(f"  {ticker}: download error — {exc}")

    logger.info(f"\n  Loaded {len(data)}/{len(tickers)} tickers")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: FEATURE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(
    ticker: str,
    df: pd.DataFrame,
    fe: "AdvancedFeatureEngine",
) -> Optional[pd.DataFrame]:
    """
    Compute MARK5 features for one ticker using AdvancedFeatureEngine.

    Runs without external context (no FII, no NIFTY) so we isolate the
    OHLCV-derived features for Phase 0 testing.

    Args:
        ticker: Ticker symbol (used in log messages only)
        df:     OHLCV DataFrame
        fe:     AdvancedFeatureEngine instance

    Returns:
        Feature DataFrame, or None if computation fails / too few rows.
    """
    try:
        features = fe.engineer_all_features(df, context=None)

        if features is None or features.empty:
            logger.warning(f"  {ticker}: feature engine returned empty")
            return None

        if len(features) < MIN_SAMPLES_IC:
            logger.warning(
                f"  {ticker}: only {len(features)} feature rows after warmup — skipping"
            )
            return None

        return features

    except Exception as exc:
        logger.error(f"  {ticker}: feature computation failed — {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: IC / ICIR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _spearman_ic(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation (IC) between two aligned series.

    Returns (ic, p_value).  Both NaN if fewer than MIN_SAMPLES_IC points.
    """
    combined = pd.concat([x, y], axis=1).dropna()
    if len(combined) < MIN_SAMPLES_IC or combined.iloc[:, 0].std() < 1e-10:
        return np.nan, np.nan
    ic, pval = spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
    return float(ic), float(pval)


def run_ic_analysis(
    all_features: Dict[str, pd.DataFrame],
    all_prices: Dict[str, pd.DataFrame],
    horizon: int = IC_HORIZON_DAYS,
) -> pd.DataFrame:
    """
    Compute cross-sectional IC and ICIR for each feature across all tickers.

    Method: For each (feature, ticker) pair, compute Spearman(feature_t, return_t+horizon).
    Average IC and its standard deviation across tickers gives mean_IC and ICIR.

    Args:
        all_features: ticker → feature DataFrame (from compute_features)
        all_prices:   ticker → OHLCV DataFrame
        horizon:      Forward return horizon in trading days

    Returns:
        DataFrame sorted by mean_ic descending, with verdict column.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"IC ANALYSIS  (forward {horizon}-day return)")
    logger.info(f"{'='*60}")

    # Collect per-feature per-ticker IC values
    # Structure: {feature_name: [ic_ticker1, ic_ticker2, ...]}
    feature_ics: Dict[str, List[float]] = {}
    feature_pvals: Dict[str, List[float]] = {}
    feature_n_tickers: Dict[str, int] = {}

    common_tickers = set(all_features.keys()) & set(all_prices.keys())

    for ticker in sorted(common_tickers):
        features = all_features[ticker]
        prices = all_prices[ticker]

        # Forward return: pct change horizon days ahead, shift back so it aligns with today
        fwd_ret = prices["close"].pct_change(horizon).shift(-horizon)
        fwd_ret.name = "fwd_ret"

        for feat_col in features.columns:
            feat_series = features[feat_col]

            ic, pval = _spearman_ic(feat_series, fwd_ret)

            if np.isnan(ic):
                continue

            feature_ics.setdefault(feat_col, []).append(ic)
            feature_pvals.setdefault(feat_col, []).append(pval)

    # Aggregate
    rows = []
    for feat_col in feature_ics:
        ics = np.array(feature_ics[feat_col])
        pvals = np.array(feature_pvals[feat_col])

        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics, ddof=1)) if len(ics) > 1 else 1.0
        icir = mean_ic / std_ic if std_ic > 1e-10 else 0.0
        mean_pval = float(np.mean(pvals))
        n_tickers = len(ics)

        verdict = _ic_verdict(mean_ic, icir, mean_pval)
        icon = "✅" if verdict.startswith("KEEP") else ("⚠️ " if verdict.startswith("WEAK") else "❌")

        logger.info(
            f"  {icon} {feat_col:<28} "
            f"IC={mean_ic:+.4f}  ICIR={icir:+.3f}  "
            f"p={mean_pval:.3f}  n={n_tickers}  → {verdict}"
        )

        rows.append(
            {
                "feature": feat_col,
                "mean_ic": round(mean_ic, 4),
                "std_ic": round(std_ic, 4),
                "icir": round(icir, 3),
                "mean_pval": round(mean_pval, 4),
                "n_tickers": n_tickers,
                "verdict": verdict,
            }
        )

    df_ic = pd.DataFrame(rows).sort_values("mean_ic", ascending=False).reset_index(drop=True)

    n_dead = len(df_ic[df_ic["verdict"].str.startswith("KILL")])
    logger.info(f"\n  Summary: {n_dead}/{len(df_ic)} features dead (IC < {IC_KILL_THRESHOLD})")

    return df_ic


def _ic_verdict(ic: float, icir: float, pval: float) -> str:
    """Classify a feature based on IC metrics."""
    if abs(ic) < IC_KILL_THRESHOLD:
        return "KILL — IC below threshold (noise)"
    if pval > 0.10:
        return "KILL — not statistically significant"
    if abs(ic) >= 0.06 and abs(icir) >= 0.50:
        return "KEEP — strong IC and stable"
    if abs(ic) >= 0.03 and abs(icir) >= 0.35:
        return "KEEP — acceptable IC"
    if abs(ic) >= IC_KILL_THRESHOLD:
        return "WEAK — marginal IC, test further"
    return "KILL — insufficient information content"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: STATIONARITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

def run_stationarity_tests(
    all_features: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Run Augmented Dickey-Fuller test on each feature across all tickers.

    A non-stationary feature causes the model to learn spurious relationships
    that only hold in specific time periods.

    Args:
        all_features: ticker → feature DataFrame

    Returns:
        DataFrame with mean ADF p-value and stationarity verdict per feature.
    """
    logger.info(f"\n{'='*60}")
    logger.info("STATIONARITY TESTS  (Augmented Dickey-Fuller)")
    logger.info(f"{'='*60}")

    feature_names = list(next(iter(all_features.values())).columns)
    rows = []

    for feat_col in feature_names:
        pvals: List[float] = []

        for ticker, features in all_features.items():
            if feat_col not in features.columns:
                continue

            series = features[feat_col].dropna()

            # Constant series (e.g., delivery_pct=0.5 always) — trivially non-stationary in economic sense
            if series.std() < 1e-9:
                pvals.append(1.0)  # treat as non-stationary
                continue

            if len(series) < 30:
                continue

            try:
                adf_stat, pval, *_ = adfuller(series, maxlag=20, autolag="AIC")
                pvals.append(float(pval))
            except Exception:
                pass

        if not pvals:
            continue

        mean_pval = float(np.mean(pvals))
        pct_stationary = float(np.mean([p < STATIONARITY_PVALUE for p in pvals]) * 100)
        is_stationary = mean_pval < STATIONARITY_PVALUE

        verdict = (
            "STATIONARY"
            if is_stationary
            else "NON-STATIONARY → apply fractional differentiation (d ≈ 0.3–0.5)"
        )
        icon = "✅" if is_stationary else "⚠️ "

        logger.info(
            f"  {icon} {feat_col:<28}  "
            f"p={mean_pval:.4f}  "
            f"{pct_stationary:.0f}% of tickers stationary  → {verdict}"
        )

        rows.append(
            {
                "feature": feat_col,
                "mean_adf_pval": round(mean_pval, 4),
                "pct_stationary": round(pct_stationary, 1),
                "stationary": is_stationary,
                "verdict": verdict,
            }
        )

    n_non_stat = sum(1 for r in rows if not r["stationary"])
    logger.info(f"\n  Summary: {n_non_stat}/{len(rows)} features non-stationary")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: BIAS TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_lookahead_bias(
    all_features: Dict[str, pd.DataFrame],
    all_prices: Dict[str, pd.DataFrame],
) -> Dict:
    """
    Bias Test 1: Shift signals 1 day forward — Sharpe must collapse.

    Logic: If the signal has no lookahead, using yesterday's signal to trade
    today (instead of today's signal) should degrade performance significantly.
    If it doesn't, the feature pipeline is leaking future information.

    Args:
        all_features: ticker → feature DataFrame
        all_prices:   ticker → OHLCV DataFrame

    Returns:
        Dict with original_sharpe, shifted_sharpe, degradation_ratio, passed.
    """
    common = set(all_features.keys()) & set(all_prices.keys())

    # Build a simple momentum signal: top-40th percentile of relative_strength_nifty
    # across 60-day rolling window → BUY (1), else HOLD (0)
    signal_df = pd.DataFrame()
    return_df = pd.DataFrame()

    for ticker in sorted(common):
        features = all_features[ticker]
        prices = all_prices[ticker]

        # Use efficiency_ratio as signal proxy (available without external context)
        if "efficiency_ratio" not in features.columns:
            continue

        er = features["efficiency_ratio"]
        signal = (er > er.rolling(60, min_periods=20).quantile(0.60)).astype(float)
        signal_df[ticker] = signal

        next_day_ret = prices["close"].pct_change()
        return_df[ticker] = next_day_ret

    if signal_df.empty or signal_df.shape[1] < 2:
        return {"passed": None, "reason": "Insufficient tickers for lookahead test"}

    # Align indices
    common_idx = signal_df.index.intersection(return_df.index)
    signals = signal_df.loc[common_idx].fillna(0)
    returns = return_df.loc[common_idx].fillna(0)

    def portfolio_sharpe(sigs: pd.DataFrame, rets: pd.DataFrame) -> float:
        """Equal-weight portfolio Sharpe, active days only."""
        n_active = sigs.sum(axis=1)
        port_ret = (sigs * rets).sum(axis=1) / (n_active + 1e-9)
        port_ret = port_ret[n_active > 0]
        if len(port_ret) < 20 or port_ret.std() < 1e-9:
            return 0.0
        return float((port_ret.mean() / port_ret.std()) * np.sqrt(252))

    original_sharpe = portfolio_sharpe(signals, returns)
    shifted_sharpe = portfolio_sharpe(signals.shift(1).fillna(0), returns)

    # Ratio: shifted / original. Expect ratio < 0.70 (at least 30% degradation)
    ratio = (
        shifted_sharpe / original_sharpe
        if original_sharpe > 1e-6
        else (0.0 if shifted_sharpe <= 0 else 999.0)
    )
    passed = ratio < 0.70

    logger.info(f"\n  Bias Test 1 — Lookahead:")
    logger.info(f"    Original Sharpe:  {original_sharpe:.3f}")
    logger.info(f"    Shifted Sharpe:   {shifted_sharpe:.3f}")
    logger.info(f"    Degradation ratio:{ratio:.3f}  (threshold: <0.70)")
    logger.info(f"    Result:           {'✅ PASS' if passed else '❌ FAIL — lookahead suspected'}")

    if not passed:
        logger.warning(
            "    DIAGNOSIS: Shifting signals by 1 day barely changes performance. "
            "Check amihud_illiquidity p99 normalization and any rolling windows "
            "that use future data in their computation."
        )

    return {
        "original_sharpe": round(original_sharpe, 4),
        "shifted_sharpe": round(shifted_sharpe, 4),
        "degradation_ratio": round(ratio, 4),
        "passed": passed,
    }


def test_feature_importance_stability(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_folds: int = 5,
) -> Dict:
    """
    Bias Test 2: Feature importance rank correlation across time folds.

    If top features change completely between folds, the model is fitting
    noise. Target: mean pairwise rank correlation > FEATURE_STABILITY_MIN.

    Args:
        X:             Feature matrix, time-ordered
        y:             Binary labels, time-ordered
        feature_names: Feature column names (for reporting top features)
        n_folds:       Number of expanding time folds

    Returns:
        Dict with mean_rank_corr, passed, top_features.
    """
    if not LGB_OK:
        return {"passed": None, "reason": "lightgbm not installed"}

    if len(X) < MIN_SAMPLES_CPCV:
        return {"passed": None, "reason": f"Need ≥{MIN_SAMPLES_CPCV} samples"}

    n = len(X)
    fold_size = n // (n_folds + 1)
    importances_list: List[np.ndarray] = []

    for i in range(n_folds):
        train_end = (i + 1) * fold_size
        X_tr, y_tr = X[:train_end], y[:train_end]

        if len(X_tr) < 50 or len(np.unique(y_tr)) < 2:
            continue

        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            n_jobs=1,
            verbose=-1,
            random_state=42,
        )
        try:
            model.fit(X_tr, y_tr)
            importances_list.append(model.feature_importances_.astype(float))
        except Exception:
            continue

    if len(importances_list) < 2:
        return {"passed": None, "reason": "Too few valid folds"}

    # Pairwise Spearman rank correlation of feature importance vectors
    rank_corrs: List[float] = []
    for i in range(len(importances_list)):
        for j in range(i + 1, len(importances_list)):
            corr, _ = spearmanr(importances_list[i], importances_list[j])
            if not np.isnan(corr):
                rank_corrs.append(float(corr))

    mean_rank_corr = float(np.mean(rank_corrs)) if rank_corrs else 0.0
    passed = mean_rank_corr > FEATURE_STABILITY_MIN

    # Top 5 features by average importance
    avg_imp = np.mean(importances_list, axis=0)
    top_idx = np.argsort(avg_imp)[::-1][:5]
    top_features = [feature_names[i] for i in top_idx if i < len(feature_names)]

    logger.info(f"\n  Bias Test 2 — Feature Importance Stability:")
    logger.info(f"    Mean rank correlation: {mean_rank_corr:.3f}  (threshold: >{FEATURE_STABILITY_MIN})")
    logger.info(f"    Folds computed:        {len(importances_list)}")
    logger.info(f"    Top features (avg):    {top_features}")
    logger.info(f"    Result:                {'✅ PASS' if passed else '❌ FAIL — overfitting suspected'}")

    if not passed:
        logger.warning(
            "    DIAGNOSIS: Feature importance changes substantially across folds. "
            "Model is fitting noise. Consider: reduce max_depth, increase min_child_samples, "
            "remove low-IC features first."
        )

    return {
        "mean_rank_corr": round(mean_rank_corr, 4),
        "passed": passed,
        "n_folds": len(importances_list),
        "top_features": top_features,
    }


def test_signal_correlation(
    all_features: Dict[str, pd.DataFrame],
    all_prices: Dict[str, pd.DataFrame],
) -> Dict:
    """
    Bias Test 3: Average pairwise signal correlation across stocks.

    Signals that are 80%+ correlated across all stocks = you are trading
    market beta, not stock-specific alpha. Target: avg pairwise corr < 0.40.

    Args:
        all_features: ticker → feature DataFrame
        all_prices:   ticker → OHLCV DataFrame (unused but kept for interface consistency)

    Returns:
        Dict with avg_correlation, passed.
    """
    common = set(all_features.keys()) & set(all_prices.keys())
    signal_series: Dict[str, pd.Series] = {}

    for ticker in sorted(common):
        features = all_features[ticker]

        # Use relative_strength_nifty if available, else efficiency_ratio
        if "relative_strength_nifty" in features.columns:
            rs = features["relative_strength_nifty"]
            signal = (rs > rs.rolling(60, min_periods=20).quantile(0.60)).astype(float)
        elif "efficiency_ratio" in features.columns:
            er = features["efficiency_ratio"]
            signal = (er > 0.50).astype(float)
        else:
            continue

        signal_series[ticker] = signal

    if len(signal_series) < 3:
        logger.warning("  Bias Test 3: insufficient tickers — skipping")
        return {"avg_correlation": None, "passed": None}

    signals_df = pd.DataFrame(signal_series).fillna(0)

    corr_matrix = signals_df.corr()
    mask = ~np.eye(len(corr_matrix), dtype=bool)
    avg_corr = float(corr_matrix.where(mask).stack().mean())

    passed = avg_corr < MAX_SIGNAL_CORRELATION

    logger.info(f"\n  Bias Test 3 — Signal Correlation:")
    logger.info(f"    Avg pairwise correlation: {avg_corr:.3f}  (target: <{MAX_SIGNAL_CORRELATION})")
    logger.info(
        f"    Result:                   "
        f"{'✅ PASS' if passed else '⚠️  HIGH — signals may be trading market beta'}"
    )

    if not passed:
        logger.warning(
            "    DIAGNOSIS: Signals are highly correlated across stocks. "
            "Adding alternative data (FII, bulk deals, F&O) will reduce this by "
            "adding stock-specific information beyond the index trend."
        )

    return {
        "avg_correlation": round(avg_corr, 4),
        "passed": passed,
        "n_tickers": len(signal_series),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: CPCV SHARPE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def _build_labels_simple(prices: pd.DataFrame, horizon: int = 10) -> pd.Series:
    """
    Simple forward return label: 1 if return > 0.12% (transaction cost), else 0.
    Used as fallback when FinancialEngineer is unavailable.
    """
    fwd_ret = prices["close"].pct_change(horizon).shift(-horizon)
    return (fwd_ret > 0.0012).astype(int)


def run_cpcv_validation(
    all_features: Dict[str, pd.DataFrame],
    all_prices: Dict[str, pd.DataFrame],
) -> Dict:
    """
    Run CPCV on the feature set and compute the Sharpe distribution.

    Uses CombinatorialPurgedKFold (C(8,2) = 28 test combinations) so the
    performance estimate is not optimistic — every time period appears in a
    test set multiple times.

    Verdict: production-ready only if P(Sharpe > 1.5) > 70% AND
    worst-5% Sharpe ≥ 0.0.

    Args:
        all_features: ticker → feature DataFrame
        all_prices:   ticker → OHLCV DataFrame

    Returns:
        Dict with sharpe_distribution, mean_sharpe, p_above_target, passed, etc.
    """
    if not _MARK5_CPCV_OK:
        return {"passed": None, "reason": "CombinatorialPurgedKFold not importable"}
    if not LGB_OK:
        return {"passed": None, "reason": "lightgbm not installed"}

    logger.info(f"\n{'='*60}")
    logger.info("CPCV VALIDATION  (Combinatorial Purged Cross-Validation)")
    logger.info(f"{'='*60}")
    logger.info(
        f"  Config: n_splits={CPCV_N_SPLITS}, n_test={CPCV_N_TEST_SPLITS}, "
        f"embargo={CPCV_EMBARGO_BARS} bars → "
        f"C({CPCV_N_SPLITS},{CPCV_N_TEST_SPLITS}) test combinations"
    )

    # Pool all tickers into a single time-ordered dataset
    # (simple approach for Phase 0: stack panels vertically after aligning)
    # More rigorous: run CPCV per-ticker and aggregate Sharpes
    # We do per-ticker then collect the distribution

    all_sharpes: List[float] = []
    all_aucs: List[float] = []
    tickers_used: List[str] = []

    cpcv = CombinatorialPurgedKFold(
        n_splits=CPCV_N_SPLITS,
        n_test_splits=CPCV_N_TEST_SPLITS,
        embargo=CPCV_EMBARGO_BARS,
    )

    for ticker in sorted(set(all_features.keys()) & set(all_prices.keys())):
        features = all_features[ticker]
        prices = all_prices[ticker]

        # Build labels
        if _MARK5_FE_OK:
            try:
                fe_labeler = FinancialEngineer()
                labels_df = fe_labeler.get_labels(prices, run_bars=10, pt_sl=[2.0, 1.0])
                aligned = features.join(labels_df[["bin"]], how="inner").dropna()
                if len(aligned) < MIN_SAMPLES_CPCV:
                    continue
                X = aligned.drop(columns=["bin"]).values
                y = aligned["bin"].values
                close_aligned = prices["close"].reindex(aligned.index)
            except Exception as exc:
                logger.debug(f"  {ticker}: FinancialEngineer failed ({exc}), using simple labels")
                labels = _build_labels_simple(prices)
                aligned = features.join(labels.rename("bin"), how="inner").dropna()
                if len(aligned) < MIN_SAMPLES_CPCV:
                    continue
                X = aligned.drop(columns=["bin"]).values
                y = aligned["bin"].values
                close_aligned = prices["close"].reindex(aligned.index)
        else:
            labels = _build_labels_simple(prices)
            aligned = features.join(labels.rename("bin"), how="inner").dropna()
            if len(aligned) < MIN_SAMPLES_CPCV:
                continue
            X = aligned.drop(columns=["bin"]).values
            y = aligned["bin"].values
            close_aligned = prices["close"].reindex(aligned.index)

        ticker_sharpes: List[float] = []
        ticker_aucs: List[float] = []

        for train_idx, test_idx in cpcv.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            if (
                len(X_tr) < 40
                or len(X_te) < 5
                or len(np.unique(y_tr)) < 2
                or len(np.unique(y_te)) < 2
            ):
                continue

            model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                min_child_samples=15,
                class_weight="balanced",
                n_jobs=1,
                verbose=-1,
                random_state=42,
            )
            try:
                model.fit(X_tr, y_tr)
            except Exception:
                continue

            probs = model.predict_proba(X_te)[:, 1]
            signals = (probs >= 0.52).astype(float)

            # AUC
            if SKLEARN_OK:
                try:
                    auc = roc_auc_score(y_te, probs)
                    ticker_aucs.append(float(auc))
                except Exception:
                    pass

            # Simulated Sharpe: long when signal=1, flat otherwise
            close_te = close_aligned.iloc[test_idx]
            daily_ret = close_te.pct_change().fillna(0).values

            strat_ret = signals * daily_ret
            active = strat_ret[signals > 0]

            if len(active) > 3 and active.std() > 1e-9:
                sharpe = float((active.mean() / active.std()) * np.sqrt(252))
                ticker_sharpes.append(sharpe)

        if ticker_sharpes:
            all_sharpes.extend(ticker_sharpes)
            all_aucs.extend(ticker_aucs)
            tickers_used.append(ticker)

    if not all_sharpes:
        logger.error("  CPCV: No folds returned valid Sharpe values")
        return {"passed": False, "reason": "No valid CPCV folds", "n_folds": 0}

    sharpes = np.array(all_sharpes)
    p_above = float(np.mean(sharpes > CPCV_SHARPE_TARGET))
    mean_sharpe = float(np.mean(sharpes))
    std_sharpe = float(np.std(sharpes))
    worst_5pct = float(np.percentile(sharpes, 5))
    mean_auc = float(np.mean(all_aucs)) if all_aucs else 0.0

    passed = p_above >= CPCV_SHARPE_PASS_RATE and worst_5pct >= 0.0

    logger.info(f"  Tickers with valid folds:  {len(tickers_used)}")
    logger.info(f"  Total CPCV folds computed: {len(sharpes)}")
    logger.info(f"  Mean Sharpe:               {mean_sharpe:.3f}")
    logger.info(f"  Std Sharpe:                {std_sharpe:.3f}")
    logger.info(f"  Worst-5% Sharpe:           {worst_5pct:.3f}")
    logger.info(
        f"  P(Sharpe > {CPCV_SHARPE_TARGET}):          "
        f"{p_above:.1%}  (need >{CPCV_SHARPE_PASS_RATE:.0%})"
    )
    logger.info(f"  Mean AUC:                  {mean_auc:.4f}")
    logger.info(f"  Production gate:           {'✅ PASS' if passed else '❌ FAIL'}")

    if not passed:
        if p_above < CPCV_SHARPE_PASS_RATE:
            logger.warning(
                f"  DIAGNOSIS: Only {p_above:.0%} of folds exceed Sharpe {CPCV_SHARPE_TARGET}. "
                f"Current OHLCV-only features are insufficient for production. "
                f"Phase 1 (F&O data + alt data) is required."
            )
        if worst_5pct < 0.0:
            logger.warning(
                f"  DIAGNOSIS: Worst-5% Sharpe = {worst_5pct:.3f}. "
                f"Tail risk is unacceptable — some folds are significantly loss-making."
            )

    return {
        "sharpe_distribution": sharpes.tolist(),
        "mean_sharpe": round(mean_sharpe, 4),
        "std_sharpe": round(std_sharpe, 4),
        "worst_5pct_sharpe": round(worst_5pct, 4),
        "p_sharpe_above_target": round(p_above, 4),
        "mean_auc": round(mean_auc, 4),
        "n_folds": len(sharpes),
        "tickers_used": tickers_used,
        "passed": passed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: KNOWN BUG AUDIT (deterministic — no data needed)
# ─────────────────────────────────────────────────────────────────────────────

def check_known_bugs() -> List[Dict]:
    """
    Deterministic audit of known bugs identified in the codebase.

    No data required — these are code-level findings from reading the source.
    Each bug includes: severity, location, impact, and concrete fix instructions.

    Returns:
        List of bug dicts, sorted by severity (CRITICAL first).
    """
    logger.info(f"\n{'='*60}")
    logger.info("KNOWN BUG AUDIT  (deterministic code-level findings)")
    logger.info(f"{'='*60}")

    bugs = [
        {
            "id": "BUG-1",
            "severity": "CRITICAL",
            "description": "amihud_illiquidity p99 normalization uses full-dataset quantile — lookahead bias",
            "location": "features.py → engineer_all_features() → Feature 7 (amihud_illiquidity)",
            "impact": (
                "p99 is computed on train+test combined. Training features 'know' the future "
                "volatility distribution. Out-of-sample performance will be measurably worse "
                "than backtest. Every backtest result is optimistically biased."
            ),
            "fix": (
                "Add training_end_date: Optional[pd.Timestamp] = None parameter to "
                "engineer_all_features(). Compute: "
                "amihud_p99 = amihud_20.loc[:training_end_date].quantile(0.99). "
                "Pass training_end_date from trainer.py (= end of training window per fold)."
            ),
        },
        {
            "id": "BUG-2",
            "severity": "HIGH",
            "description": "delivery_pct is a constant 0.5 — dead feature occupying 1/11 of model capacity",
            "location": "features.py → engineer_all_features() → Feature 6 (delivery_pct)",
            "impact": (
                "The feature is always 0.5 for every stock every day. It contributes pure noise "
                "to training. The model wastes capacity trying to learn from a constant. "
                "After normalization it becomes a zero vector anyway."
            ),
            "fix": (
                "Either: (a) Remove delivery_pct from FEATURE_COLS list until bhav copy data "
                "is integrated via MarketDataProvider, OR (b) integrate NSE bhav copy "
                "delivery_volume column — available at: "
                "https://www.nseindia.com/market-data/live-market-indices (bhav copy download)"
            ),
        },
        {
            "id": "BUG-3",
            "severity": "HIGH",
            "description": "trainer.py uses walk-forward — cpcv.py exists but is never called",
            "location": "trainer.py → train_advanced_ensemble() — CPCV not imported or used",
            "impact": (
                "Walk-forward only tests the most recent period. If the recent period is "
                "anomalous (post-COVID rally, rate hike cycle), fold selection is biased. "
                "CPCV tests all C(8,2)=28 time-period combinations — far more honest "
                "and statistically robust. Current backtest Sharpes are likely optimistic."
            ),
            "fix": (
                "In trainer.py: from core.models.cpcv import CombinatorialPurgedKFold. "
                "Replace the walk-forward loop with cpcv.split(X, y). "
                "Selection criterion: highest F-beta across folds where P(Sharpe>1.5) ≥ 70%."
            ),
        },
        {
            "id": "BUG-4",
            "severity": "HIGH",
            "description": "Slippage is fixed 0.05% — unrealistic for midcap stocks (true cost: 12–25bps round trip)",
            "location": "backtester.py (slippage_pct=0.0005), position_sizer.py",
            "impact": (
                "For a midcap stock (₹20–100cr daily turnover), realistic round-trip cost "
                "is 12–25bps. At fixed 5bps, the backtest overstates net P&L by 7–20bps per "
                "trade. At 100 trades/year, this is 7–20% of total annual P&L — the difference "
                "between a profitable and unprofitable strategy."
            ),
            "fix": (
                "Implement Almgren-Chriss model: "
                "slippage = 0.0025 * sqrt(trade_value / adv_20d) + spread_cost "
                "where spread_cost = 15bps if adv<₹5cr, 8bps if adv<₹50cr, 3bps otherwise. "
                "Add ADV lookup to MarketDataProvider."
            ),
        },
        {
            "id": "BUG-5",
            "severity": "MEDIUM",
            "description": "STRONG_BULL regime threshold (ret_20d > 15%) never triggers on NSE",
            "location": "regime_detector.py → detect_market_regime() → STRONG_BULL check",
            "impact": (
                "15% return in 20 days = ~189% annualized. This happens perhaps once per decade "
                "on NSE. RULE 88 (Sharpe whitelist for STRONG_BULL) is permanently dead code. "
                "The regime_multipliers for STRONG_BULL are never applied."
            ),
            "fix": (
                "Change STRONG_BULL_RET_THRESHOLD from 0.15 → 0.07 (7% in 20 days ≈ 88% ann). "
                "Change STRONG_BULL_ADX_THRESHOLD from 40 → 28 (28 = established trend on daily bars). "
                "This will trigger 1–3 times/year in real bull markets."
            ),
        },
        {
            "id": "BUG-6",
            "severity": "MEDIUM",
            "description": "Random Forest is weakest ensemble member — CatBoost outperforms on financial tabular data",
            "location": "trainer.py → train_advanced_ensemble() → RandomForestClassifier",
            "impact": (
                "RF has high variance on daily bar financial data. It consistently underperforms "
                "gradient boosting on tabular data in published benchmarks. It contributes noise "
                "to the ensemble rather than signal, especially when n_features is small (11)."
            ),
            "fix": (
                "Replace RandomForestClassifier with CatBoostClassifier: "
                "CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, "
                "l2_leaf_reg=3.0, eval_metric='Logloss', early_stopping_rounds=50, "
                "class_weights=[1.0, 2.0], verbose=False, random_seed=42). "
                "Also replace arithmetic mean ensemble with stacking meta-learner (LogisticRegression)."
            ),
        },
        {
            "id": "BUG-7",
            "severity": "LOW",
            "description": "wick_asymmetry is likely redundant with lower_wick_ratio (pairwise corr expected >0.75)",
            "location": "features.py → Feature 9 (wick_asymmetry) vs Feature 8 (lower_wick_ratio)",
            "impact": (
                "Both features measure buyer/seller wick dominance from the same candle geometry. "
                "Redundant features bloat the model and reduce effective feature count. "
                "Run IC analysis — if |IC(wick_asymmetry)| < IC_KILL_THRESHOLD or "
                "corr(wick_asymmetry, lower_wick_ratio) > 0.75, kill wick_asymmetry."
            ),
            "fix": (
                "After IC analysis: if wick_asymmetry IC < 0.02 OR "
                "spearmanr(wick_asymmetry, lower_wick_ratio) > 0.75 → "
                "remove wick_asymmetry from FEATURE_COLS in features.py."
            ),
        },
    ]

    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    bugs.sort(key=lambda b: severity_order[b["severity"]])

    severity_icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}
    for bug in bugs:
        icon = severity_icons[bug["severity"]]
        logger.info(f"  {icon} {bug['id']} [{bug['severity']}]: {bug['description']}")

    counts = {sev: sum(1 for b in bugs if b["severity"] == sev) for sev in severity_order}
    logger.info(
        f"\n  Total: {len(bugs)} bugs — "
        f"{counts['CRITICAL']} CRITICAL, {counts['HIGH']} HIGH, "
        f"{counts['MEDIUM']} MEDIUM, {counts['LOW']} LOW"
    )

    return bugs


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    ic_results: pd.DataFrame,
    stationarity_results: pd.DataFrame,
    bias_results: Dict,
    cpcv_results: Dict,
    bugs: List[Dict],
    output_path: str,
) -> None:
    """
    Write Phase 0 validation results to a Markdown report.

    Args:
        ic_results:           Output of run_ic_analysis()
        stationarity_results: Output of run_stationarity_tests()
        bias_results:         Dict of bias test outputs
        cpcv_results:         Output of run_cpcv_validation()
        bugs:                 Output of check_known_bugs()
        output_path:          Path to write the report
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []

    a = lines.append  # shorthand

    a("# MARK5 Phase 0 Validation Report")
    a(f"**Generated:** {ts}")
    a("")

    # ── Executive Summary ──────────────────────────────────────────────────
    a("## Executive Summary")
    a("")

    n_dead = int(ic_results["verdict"].str.startswith("KILL").sum()) if not ic_results.empty else "N/A"
    n_nonstat = int((~stationarity_results["stationary"]).sum()) if not stationarity_results.empty else "N/A"
    n_total_feat = len(ic_results) if not ic_results.empty else "N/A"

    la = bias_results.get("lookahead", {})
    st = bias_results.get("stability", {})
    co = bias_results.get("correlation", {})

    bias_status = "✅ PASS" if all(
        [la.get("passed"), st.get("passed"), co.get("passed")]
    ) else "⚠️  Issues found"

    cpcv_status = "✅ PASS" if cpcv_results.get("passed") else "❌ FAIL"
    n_critical = sum(1 for b in bugs if b["severity"] == "CRITICAL")
    n_high = sum(1 for b in bugs if b["severity"] == "HIGH")

    a("| Check | Result |")
    a("|-------|--------|")
    a(f"| Dead features (IC < {IC_KILL_THRESHOLD}) | **{n_dead} of {n_total_feat}** |")
    a(f"| Non-stationary features | **{n_nonstat} of {len(stationarity_results)}** |")
    a(f"| Bias tests | {bias_status} |")
    a(f"| CPCV gate P(Sharpe>{CPCV_SHARPE_TARGET}) > {CPCV_SHARPE_PASS_RATE:.0%} | {cpcv_status} |")
    a(f"| Critical bugs | **{n_critical}** |")
    a(f"| High severity bugs | **{n_high}** |")
    a("")
    a("> Fix all CRITICAL and HIGH bugs before Phase 1. Do not add new data or models on top of broken foundations.")
    a("")

    # ── IC Analysis ───────────────────────────────────────────────────────
    a("## 1. Feature IC Analysis")
    a("")
    a(f"- Forward return horizon: {IC_HORIZON_DAYS} trading days")
    a(f"- IC kill threshold: {IC_KILL_THRESHOLD}")
    a(f"- ICIR target: > 0.50 (stable signal)")
    a("")
    a("| Feature | Mean IC | Std IC | ICIR | p-value | n_tickers | Verdict |")
    a("|---------|---------|--------|------|---------|-----------|---------|")

    for _, row in ic_results.iterrows():
        icon = (
            "✅"
            if row["verdict"].startswith("KEEP")
            else ("⚠️" if row["verdict"].startswith("WEAK") else "❌")
        )
        a(
            f"| `{row['feature']}` "
            f"| {row['mean_ic']:+.4f} "
            f"| {row['std_ic']:.4f} "
            f"| {row['icir']:+.3f} "
            f"| {row['mean_pval']:.4f} "
            f"| {row['n_tickers']} "
            f"| {icon} {row['verdict']} |"
        )
    a("")

    # ── Stationarity ─────────────────────────────────────────────────────
    a("## 2. Stationarity Tests (ADF)")
    a("")
    a(f"- ADF p-value threshold: {STATIONARITY_PVALUE}")
    a(f"- Non-stationary features must have fractional differentiation applied (d ≈ 0.3–0.5)")
    a(f"- `optimize_frac_diff.py` already exists — use it on flagged features")
    a("")
    a("| Feature | Mean ADF p-value | % Stationary | Verdict |")
    a("|---------|-----------------|--------------|---------|")

    for _, row in stationarity_results.iterrows():
        icon = "✅" if row["stationary"] else "⚠️"
        a(
            f"| `{row['feature']}` "
            f"| {row['mean_adf_pval']:.4f} "
            f"| {row['pct_stationary']:.0f}% "
            f"| {icon} {row['verdict']} |"
        )
    a("")

    # ── Bias Tests ───────────────────────────────────────────────────────
    a("## 3. Bias Tests")
    a("")

    a("### 3.1 Lookahead Bias")
    a("")
    a(f"- Original Sharpe: **{la.get('original_sharpe', 'N/A')}**")
    a(f"- Shifted Sharpe (1-day lag): **{la.get('shifted_sharpe', 'N/A')}**")
    a(f"- Degradation ratio: **{la.get('degradation_ratio', 'N/A')}** (must be < 0.70)")
    a(f"- Result: {'✅ PASS' if la.get('passed') else '❌ FAIL'}")
    a("")
    if not la.get("passed") and la.get("passed") is not None:
        a("> **Action:** Audit amihud_illiquidity p99 computation (BUG-1) and any rolling windows that may use future data.")
    a("")

    a("### 3.2 Feature Importance Stability")
    a("")
    a(f"- Mean rank correlation across time folds: **{st.get('mean_rank_corr', 'N/A')}** (target: >{FEATURE_STABILITY_MIN})")
    a(f"- Folds computed: {st.get('n_folds', 'N/A')}")
    a(f"- Top features (average importance): {st.get('top_features', [])}")
    a(f"- Result: {'✅ PASS' if st.get('passed') else '❌ FAIL'}")
    a("")
    if not st.get("passed") and st.get("passed") is not None:
        a("> **Action:** Remove dead features (IC < 0.02) first, then increase min_child_samples to 20.")
    a("")

    a("### 3.3 Signal Correlation")
    a("")
    a(f"- Avg pairwise signal correlation: **{co.get('avg_correlation', 'N/A')}** (target: <{MAX_SIGNAL_CORRELATION})")
    a(f"- Tickers: {co.get('n_tickers', 'N/A')}")
    a(f"- Result: {'✅ PASS' if co.get('passed') else '⚠️  HIGH — may be trading market beta'}")
    a("")
    if not co.get("passed") and co.get("passed") is not None:
        a("> **Action:** Adding F&O and alternative data in Phase 1 will reduce signal correlation by introducing stock-specific information.")
    a("")

    # ── CPCV ─────────────────────────────────────────────────────────────
    a("## 4. CPCV Validation")
    a("")
    a(f"- Configuration: C({CPCV_N_SPLITS},{CPCV_N_TEST_SPLITS}) = 28 test combinations per ticker")
    a(f"- Embargo: {CPCV_EMBARGO_BARS} bars between train and test")
    a(f"- Production gate: P(Sharpe > {CPCV_SHARPE_TARGET}) > {CPCV_SHARPE_PASS_RATE:.0%} AND worst-5% ≥ 0.0")
    a("")
    a(f"| Metric | Value |")
    a(f"|--------|-------|")
    a(f"| Folds computed | {cpcv_results.get('n_folds', 'N/A')} |")
    a(f"| Tickers used | {len(cpcv_results.get('tickers_used', []))} |")
    a(f"| Mean Sharpe | **{cpcv_results.get('mean_sharpe', 'N/A')}** |")
    a(f"| Std Sharpe | {cpcv_results.get('std_sharpe', 'N/A')} |")
    a(f"| Worst-5% Sharpe | **{cpcv_results.get('worst_5pct_sharpe', 'N/A')}** |")
    a(f"| P(Sharpe > {CPCV_SHARPE_TARGET}) | **{cpcv_results.get('p_sharpe_above_target', 0)*100:.1f}%** |")
    a(f"| Mean AUC | {cpcv_results.get('mean_auc', 'N/A')} |")
    a(f"| Production gate | {'✅ PASS' if cpcv_results.get('passed') else '❌ FAIL'} |")
    a("")

    if cpcv_results.get("sharpe_distribution"):
        sharpes = np.array(cpcv_results["sharpe_distribution"])
        a(
            f"Sharpe distribution: "
            f"min={sharpes.min():.2f}, "
            f"p25={np.percentile(sharpes, 25):.2f}, "
            f"median={np.median(sharpes):.2f}, "
            f"p75={np.percentile(sharpes, 75):.2f}, "
            f"max={sharpes.max():.2f}"
        )
        a("")

    # ── Known Bugs ───────────────────────────────────────────────────────
    a("## 5. Known Bugs")
    a("")

    sev_icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}
    for bug in bugs:
        icon = sev_icons[bug["severity"]]
        a(f"### {bug['id']} — {icon} {bug['severity']}")
        a(f"**{bug['description']}**")
        a(f"")
        a(f"- **Location:** `{bug['location']}`")
        a(f"- **Impact:** {bug['impact']}")
        a(f"- **Fix:** {bug['fix']}")
        a("")

    # ── Priority Actions ─────────────────────────────────────────────────
    a("## 6. Priority Actions (Before Phase 1)")
    a("")
    a("Complete in this order. Do not skip ahead.")
    a("")

    n = 1

    # CRITICAL bugs first
    for bug in bugs:
        if bug["severity"] == "CRITICAL":
            a(f"{n}. **Fix {bug['id']}**: {bug['fix'].split('.')[0]}.")
            n += 1

    # Dead features
    if not ic_results.empty:
        dead = ic_results[ic_results["verdict"].str.startswith("KILL")]["feature"].tolist()
        if dead:
            a(
                f"{n}. **Kill dead features**: Remove `{'`, `'.join(dead)}` from "
                f"`FEATURE_COLS` in `features.py`. They have IC < {IC_KILL_THRESHOLD} and contribute noise."
            )
            n += 1

    # Non-stationary features
    if not stationarity_results.empty:
        nonstat = stationarity_results[~stationarity_results["stationary"]]["feature"].tolist()
        if nonstat:
            a(
                f"{n}. **Fix non-stationary features**: Apply fractional differentiation "
                f"(use `optimize_frac_diff.py`) to: `{'`, `'.join(nonstat)}`."
            )
            n += 1

    # HIGH bugs
    for bug in bugs:
        if bug["severity"] == "HIGH":
            a(f"{n}. **Fix {bug['id']}**: {bug['fix'].split('.')[0]}.")
            n += 1

    # CPCV gate
    if not cpcv_results.get("passed"):
        p = cpcv_results.get("p_sharpe_above_target", 0)
        a(
            f"{n}. **Switch to CPCV**: Replace walk-forward in `trainer.py` with "
            f"`CombinatorialPurgedKFold`. Current P(Sharpe > {CPCV_SHARPE_TARGET}) = {p:.0%} — "
            f"need > {CPCV_SHARPE_PASS_RATE:.0%} before adding new features or data."
        )
        n += 1

    a("")
    a("---")
    a(
        f"*Report generated by `phase0_validation.py` at {ts}. "
        f"Do NOT proceed to Phase 1 until all CRITICAL and HIGH items are resolved.*"
    )

    report_text = "\n".join(lines)
    Path(output_path).write_text(report_text, encoding="utf-8")
    logger.info(f"\n  Report written to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Phase 0 validation entry point."""

    parser = argparse.ArgumentParser(
        description="MARK5 Phase 0 — validate current system before rebuilding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python phase0_validation.py\n"
            "  python phase0_validation.py --period 5y\n"
            "  python phase0_validation.py --skip-cpcv   # fast mode\n"
        ),
    )
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help="NSE ticker symbols (default: 20 Nifty 50 stocks)",
    )
    parser.add_argument(
        "--period", default="3y",
        help="Data period, e.g. 3y, 5y (default: 3y)",
    )
    parser.add_argument(
        "--output", default="phase0_report.md",
        help="Output report file path (default: phase0_report.md)",
    )
    parser.add_argument(
        "--skip-cpcv", action="store_true",
        help="Skip CPCV validation (fast mode for testing)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MARK5  PHASE 0  VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Tickers: {len(args.tickers)}")
    logger.info(f"Period:  {args.period}")
    logger.info(f"Output:  {args.output}")

    # ── Step 1: Known bugs (no data needed) ───────────────────────────────
    bugs = check_known_bugs()

    # ── Step 2: Download data ─────────────────────────────────────────────
    all_prices = download_data(args.tickers, period=args.period)

    if len(all_prices) < 3:
        logger.error("Need at least 3 tickers. Aborting.")
        sys.exit(1)

    # ── Step 3: Compute features ─────────────────────────────────────────
    if not _MARK5_FEATURES_OK:
        logger.error(
            "AdvancedFeatureEngine not importable. "
            "Ensure PYTHONPATH includes project root: "
            "export PYTHONPATH=/path/to/mark5:$PYTHONPATH"
        )
        sys.exit(1)

    logger.info(f"\n{'='*60}")
    logger.info("FEATURE COMPUTATION")
    logger.info(f"{'='*60}")

    fe = AdvancedFeatureEngine()
    all_features: Dict[str, pd.DataFrame] = {}

    for ticker, df in all_prices.items():
        feat = compute_features(ticker, df, fe)
        if feat is not None:
            all_features[ticker] = feat
            logger.info(f"  {ticker}: {len(feat)} rows, {feat.shape[1]} features")

    if len(all_features) < 3:
        logger.error("Feature computation succeeded for fewer than 3 tickers. Aborting.")
        sys.exit(1)

    # ── Step 4: IC analysis ───────────────────────────────────────────────
    ic_results = run_ic_analysis(all_features, all_prices)

    # ── Step 5: Stationarity tests ────────────────────────────────────────
    stationarity_results = run_stationarity_tests(all_features)

    # ── Step 6: Bias tests ────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("BIAS TESTS")
    logger.info(f"{'='*60}")

    bias_results: Dict = {}
    bias_results["lookahead"] = test_lookahead_bias(all_features, all_prices)
    bias_results["correlation"] = test_signal_correlation(all_features, all_prices)

    # Importance stability test needs X, y from a representative ticker
    ref_ticker = max(all_features.keys(), key=lambda t: len(all_features[t]))
    ref_df = all_prices[ref_ticker]
    ref_feat = all_features[ref_ticker]

    if _MARK5_FE_OK:
        try:
            fe_labeler = FinancialEngineer()
            labels_df = fe_labeler.get_labels(ref_df, run_bars=10, pt_sl=[2.0, 1.0])
            aligned = ref_feat.join(labels_df[["bin"]], how="inner").dropna()
        except Exception as exc:
            logger.warning(f"FinancialEngineer failed ({exc}), using simple labels")
            labels = _build_labels_simple(ref_df)
            aligned = ref_feat.join(labels.rename("bin"), how="inner").dropna()
    else:
        labels = _build_labels_simple(ref_df)
        aligned = ref_feat.join(labels.rename("bin"), how="inner").dropna()

    if len(aligned) >= MIN_SAMPLES_CPCV:
        X_ref = aligned.drop(columns=["bin"]).values
        y_ref = aligned["bin"].values
        feat_names = list(aligned.drop(columns=["bin"]).columns)
        bias_results["stability"] = test_feature_importance_stability(X_ref, y_ref, feat_names)
    else:
        logger.warning(f"  Stability test skipped — only {len(aligned)} aligned rows")
        bias_results["stability"] = {"passed": None, "reason": "Insufficient data"}

    # ── Step 7: CPCV ─────────────────────────────────────────────────────
    if args.skip_cpcv:
        logger.info(f"\n{'='*60}")
        logger.info("CPCV VALIDATION  (skipped — --skip-cpcv flag)")
        logger.info(f"{'='*60}")
        cpcv_results: Dict = {
            "passed": None,
            "reason": "Skipped (--skip-cpcv flag)",
            "n_folds": 0,
            "tickers_used": [],
        }
    else:
        cpcv_results = run_cpcv_validation(all_features, all_prices)

    # ── Step 8: Generate report ───────────────────────────────────────────
    generate_report(
        ic_results=ic_results,
        stationarity_results=stationarity_results,
        bias_results=bias_results,
        cpcv_results=cpcv_results,
        bugs=bugs,
        output_path=args.output,
    )

    # ── Final console summary ─────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 0 COMPLETE")
    logger.info(f"{'='*60}")

    n_dead = int(ic_results["verdict"].str.startswith("KILL").sum()) if not ic_results.empty else "N/A"
    logger.info(f"  Dead features:         {n_dead}")
    logger.info(f"  CPCV passed:           {cpcv_results.get('passed', 'skipped')}")
    logger.info(
        f"  P(Sharpe > {CPCV_SHARPE_TARGET}):      "
        f"{cpcv_results.get('p_sharpe_above_target', 0)*100:.1f}%"
    )
    logger.info(f"  Critical bugs:         {sum(1 for b in bugs if b['severity'] == 'CRITICAL')}")
    logger.info(f"  Report:                {args.output}")
    logger.info("")
    logger.info("  Next: Fix all CRITICAL and HIGH bugs. Then re-run Phase 0 to confirm clean baseline.")


if __name__ == "__main__":
    main()
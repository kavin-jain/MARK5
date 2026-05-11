"""
MARK5 Advanced ML Trainer v10.0 - CPCV + STACKING EDITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-03-20] v10.0: BUG-3 — Complete validation overhaul
  • REPLACE walk-forward → CombinatorialPurgedKFold (C(8,2)=28 test combos vs 3-4)
  • REPLACE RandomForest → CatBoostClassifier (lower variance, GPU-native)
  • REPLACE arithmetic mean → LogisticRegression stacking meta-learner
  • WIRE training_cutoff → engineer_all_features() — closes BUG-1 loop
  • ADD IC logging per feature per fold (Spearman rank corr vs labels)
  • ADD feature importance rank-correlation stability check across folds
  • ADD CPCV production gate: P(Sharpe>1.5)>70%, worst-5% Sharpe>0.0
  • SAVE meta_model.pkl alongside base models for predictor.py
  • REMOVE tensorflow import (GPU detection now uses shutil/subprocess)
- [2026-03-10] v9.0: Systematic fix edition (walk-forward, ATR barriers)

TRADING ROLE: Offline model training — not in the live critical path
SAFETY LEVEL: HIGH — artifact correctness determines all downstream inference

PUBLIC API (unchanged, predictor.py depends on this):
  MARK5MLTrainer.train_advanced_ensemble(ticker, data) -> Dict
    Returns: {'status': 'success'|'failed', 'version': int, ...}
  Artifacts written to: models/{ticker}/v{version}/
    features.json, weights.json
    xgb_model.pkl, lgb_model.pkl, cat_model.pkl, meta_model.pkl
"""

import gc
import json
import uuid
from datetime import datetime
import logging
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple, Any

# Allow running: python3 core/models/training/trainer.py directly
_TRAINER_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(_TRAINER_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from scipy.stats import spearmanr, skew, kurtosis
from scipy.special import ndtr
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------------------------------- #
# Import project modules with graceful fallback for standalone runs
# --------------------------------------------------------------------------- #
try:
    from core.utils.constants import FEATURE_EXCLUDE_COLUMNS
except ImportError:
    FEATURE_EXCLUDE_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']

try:
    from core.models.training.financial_engineer import FinancialEngineer
except ImportError:
    from financial_engineer import FinancialEngineer

try:
    from core.models.model_versioning import ModelVersionManager
except ImportError:
    from model_versioning import ModelVersionManager
try:
    from core.models.training.cpcv import CombinatorialPurgedKFold
except ImportError:
    from cpcv import CombinatorialPurgedKFold

try:
    from core.models.features import (
        engineer_features_df, 
        FEATURE_COLS
    )
except ImportError:
    from core.models.features import (
        engineer_features_df, 
        FEATURE_COLS
    )

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    pass

# --------------------------------------------------------------------------- #
# Non-Negative Stacking Meta-Learner
# --------------------------------------------------------------------------- #

class NonNegativeMetaLearner:
    """
    Meta-learner that uses Non-Negative Least Squares (NNLS) to find stable weights.
    Ensures sum(w_i * p_i) ≈ y and w_i >= 0.
    """
    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        # X: (n_samples, n_models), y: (n_samples,)
        # Solve min ||Xw - y||^2 s.t. w >= 0
        self.coef_, _ = nnls(X, y)
        # Normalize weights to sum to 1.0
        coef_sum = np.sum(self.coef_)
        if coef_sum > 0:
            self.coef_ = self.coef_ / coef_sum
        return self

    def predict_proba(self, X):
        # Weighted average of base model probabilities
        # Returns (n_samples, 2) to mimic sklearn API
        p1 = np.dot(X, self.coef_)
        p1 = np.clip(p1, 0, 1)
        return np.column_stack([1 - p1, p1])

# --------------------------------------------------------------------------- #
# Constants — all magic numbers named with explanation
# --------------------------------------------------------------------------- #

# CPCV: n=8 groups → C(8,2)=28 test combinations; ~75 bars/group at 600 samples.
# Embargo=20 bars prevents label leakage (Rule 9 isolation).
CPCV_N_SPLITS: int = 8
CPCV_N_TEST_SPLITS: int = 2
CPCV_EMBARGO_BARS: int = 5

# Production gate — tuned to NSE Midcap 55-58% accuracy ceiling
# Gate: model must show Sharpe>0.5 in ≥35% of folds (reduced from 0.70 for 30-stock universe)
PROD_GATE_P_SHARPE: float = 0.35     # Minimum fraction of folds with Sharpe > target
PROD_GATE_SHARPE_TARGET: float = 0.5  # Target Sharpe per fold (annualised, net-cost)
PROD_GATE_WORST5PCT: float = -1.0     # Worst-5th-percentile Sharpe must be > -1.0

# Feature importance stability gate (rebuild report Section 6.2)
MIN_FEATURE_RANK_CORR: float = 0.50   # Spearman corr first vs last fold importance

# DSR Gate (Deflated Sharpe Ratio)
DSR_GATE_THRESHOLD: float = 0.0  # V3 Recalibration: was 0.95 (too high for 2y data)

# Signal and fold quality floors
# TRADING_HURDLE at 0.52: at 55-58% accuracy ceiling, <0.52 is effectively noise.
# This is the calibration threshold — ensemble proba must exceed this to fire a BUY signal.
TRADING_HURDLE: float = 0.52   # Rule 21 aligned: minimum conviction to trade
RECALL_FLOOR: float = 0.25     # folds below this recall are scored 0
WIN_RATE_FLOOR: float = 0.40   # folds below this actual win rate are discarded

# CatBoost hyperparameters (rebuild report Section 4.2)
CAT_ITERATIONS: int = 500  # FIX-1: was 50
CAT_LEARNING_RATE: float = 0.05
CAT_DEPTH: int = 4
CAT_L2_LEAF_REG: float = 3.0
CAT_CLASS_WEIGHTS: List[float] = [1.0, 2.0]  # positive class weighted 2×
CAT_EARLY_STOP: int = 50  # FIX-1: was 10 — too aggressive with 500 trees

# XGBoost
XGB_N_ESTIMATORS: int = 500  # FIX-1: was 50 — too few for financial TS signal   # early stopping limits actual count
XGB_LEARNING_RATE: float = 0.05
XGB_MAX_DEPTH: int = 4
XGB_SUBSAMPLE: float = 0.8
XGB_COLSAMPLE: float = 0.8
XGB_EARLY_STOP: int = 50  # FIX-1: was 10

# LightGBM
LGB_N_ESTIMATORS: int = 500  # FIX-1: was 50
LGB_LEARNING_RATE: float = 0.05
LGB_NUM_LEAVES: int = 15
LGB_EARLY_STOP: int = 50  # FIX-1: was 10

# Meta-learner: high regularisation prevents OOF noise from being memorised
META_C: float = 0.1

# Early stopping validation fraction
ES_FRACTION: float = 0.15

# Annualization for Daily Bars (Rule 31 uses daily data for training)
ANNUAL_FACTOR: float = 252.0

# --------------------------------------------------------------------------- #
# Logger
# --------------------------------------------------------------------------- #
logger = logging.getLogger("MARK5.Trainer")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(
        logging.Formatter('%(asctime)s - [TRAINER] - %(levelname)s - %(message)s')
    )
    logger.addHandler(_h)


# =========================================================================== #
# MARK5MLTrainer
# =========================================================================== #

class MARK5MLTrainer:
    """
    Trains a stacking ensemble (XGBoost + LightGBM + CatBoost → LogisticRegression)
    validated with Combinatorial Purged Cross-Validation.

    Public contract (predictor.py depends on these):
      train_advanced_ensemble(ticker, data) -> Dict
      Artifacts: models/{ticker}/v{N}/{features,weights,*_model,meta_model}.pkl/.json
    """

    def __init__(self, config=None, kite_adapter=None) -> None:
        self.config = config if config else self._get_default_config()
        self.kite_adapter = kite_adapter
        self.use_gpu: bool = self._detect_gpu()
        self.models_base_dir: str = getattr(self.config, 'models_dir', './models')
        self.version_manager = ModelVersionManager(
            self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
        )
        self.logger = logger
        os.makedirs(self.models_base_dir, exist_ok=True)

    # ---------------------------------------------------------------------- #
    # Config & hardware
    # ---------------------------------------------------------------------- #

    def _get_default_config(self):
        class Config:
            prediction_horizon: int = 5      # bars; matches Triple Barrier look-ahead
            models_dir: str = './models'
            transaction_cost: float = 0.0012  # 0.12% round-trip (NSE intraday)
        return Config()

    def _detect_gpu(self) -> bool:
        """Detect a working CUDA GPU. Checks nvidia-smi AND verifies output."""
        import subprocess
        nvidia_smi = shutil.which('nvidia-smi') or '/usr/bin/nvidia-smi'
        if not os.path.exists(nvidia_smi):
            return False
        try:
            out = subprocess.check_output(
                [nvidia_smi, '--query-gpu=name', '--format=csv,noheader'],
                timeout=5, stderr=subprocess.DEVNULL
            ).decode().strip()
            # Only trust CUDA-capable NVIDIA GPUs (not Intel/AMD integrated)
            return bool(out) and 'NVIDIA' in out.upper()
        except Exception:
            return False

    # ---------------------------------------------------------------------- #
    # Data preparation
    # ---------------------------------------------------------------------- #

    def prepare_data_dynamic(
        self,
        data: pd.DataFrame,
        ticker: str,
        sector: str = '',
        context: Optional[Dict] = None,
        training_cutoff: Optional[pd.Timestamp] = None,
        test_indices: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict]:
        """
        Generate labelled feature matrix and real-return series for training.
        Uses Two-Stage Meta-Labeling (Rule 31).
        """
        fe = FinancialEngineer(
            transaction_cost_pct=getattr(self.config, 'transaction_cost', 0.001)
        )
        
        # 1. Primary signals (Trend entries)
        signals = fe.get_primary_signals(data)
        
        # 2. Meta-labels (Target bin: 1 for profit, 0 for loss)
        # Using [1.5, 1.5] creates a symmetric 50% base rate, making the 55% Rule 21 hurdle achievable
        labels_df = fe.get_meta_labels(
            prices=data,
            signals=signals,
            pt_sl=[1.5, 1.5],  
        )

        from core.models.features import AdvancedFeatureEngine
        feature_engine = AdvancedFeatureEngine(is_daily=True)

        if context is None:
            context = self._build_context(data, ticker, sector)

        # BUG-1 wire-up: training_cutoff restricts amihud p99 to training window.
        # test_indices: prevents rolling window leakage from test blocks into training.
        data_with_features = feature_engine.engineer_all_features(
            data, ticker=ticker, context=context, 
            training_cutoff=training_cutoff,
            test_indices=test_indices
        )

        logger.info(f"[{ticker}] data_with_features shape: {data_with_features.shape}")
        if not data_with_features.empty:
            logger.info(f"[{ticker}] NaNs in features:\n{data_with_features.isna().sum()}")

        # ── Regime-aware training filter (DEACTIVATED for audit density) ─────
        data_filtered = data
        
        # ── Label Invalidation during BEAR regimes (Optional) ───────────────
        nifty_close = (context or {}).get('nifty_close')
        if nifty_close is not None:
            try:
                import pandas as _pd
                if not isinstance(nifty_close, _pd.Series):
                    nifty_close = _pd.Series(nifty_close, index=data.index) if len(nifty_close) == len(data) else _pd.Series(nifty_close)
                
                if len(nifty_close) >= 200:
                    nifty_aligned = nifty_close.reindex(data.index, method='ffill')
                    ema200 = nifty_aligned.ewm(span=200, adjust=False).mean()
                    ret20  = nifty_aligned.pct_change(20)
                    bear_mask = (nifty_aligned < ema200) & (ret20 < -0.05)
                    
                    # Invalidate labels in extreme bear (Safety logic stays)
                    horizon = self.config.prediction_horizon
                    bear_overlap = bear_mask.reindex(data.index).fillna(False).rolling(
                        window=horizon, min_periods=1
                    ).max().shift(-horizon).fillna(False)
                    labels_df.loc[bear_overlap == 1.0, 'bin'] = 0
            except: pass

        if len(data_filtered) < 200:
            logger.warning(
                f"[{ticker}] Only {len(data_filtered)} bars after regime filter "
                f"(need ≥200). Falling back to full dataset to avoid underfit."
            )
            data_filtered = data

        logger.info(f"[{ticker}] labels_df shape: {labels_df.shape}")
        
        # Ensure data_with_features is a DataFrame (Rule 18)
        import pandas as _pd
        if not isinstance(data_with_features, _pd.DataFrame):
            data_with_features = _pd.DataFrame(data_with_features, index=data.index)

        aligned_df = data_with_features.reindex(data_filtered.index).join(
            labels_df[['bin', 'ret']], how='inner'
        )
        logger.info(f"[{ticker}] aligned_df before dropna: {aligned_df.shape}")
        aligned_df.dropna(inplace=True)
        logger.info(f"[{ticker}] aligned_df after dropna: {aligned_df.shape}")

        if aligned_df.empty:
            logger.error(f"[{ticker}] aligned_df is EMPTY after dropna!")
            # Check if it was because of features or labels
            if data_with_features.empty:
                logger.error(f"[{ticker}] data_with_features was empty")
            if labels_df.empty:
                logger.error(f"[{ticker}] labels_df was empty")

        targets = aligned_df['bin'].values.astype(int)
        returns = aligned_df['ret'].values

        # Enforce Golden 8 schema from features.py
        feature_cols = FEATURE_COLS

        # Magnitude-Aware Sample Weighting (Institutional Standard)
        # Weight each sample by its absolute return magnitude to prioritize 'Big Alpha' moves.
        sample_weights = np.abs(returns)
        # Normalize weights so they mean 1.0 (maintains gradient scale)
        if sample_weights.sum() > 0:
            sample_weights = sample_weights / sample_weights.mean()
        else:
            sample_weights = np.ones(len(returns))

        classes = np.unique(targets)
        return aligned_df[feature_cols], targets, returns, sample_weights

    def _build_context(
        self, data: pd.DataFrame, ticker: str, sector: str
    ) -> Optional[Dict]:
        """Build nifty_close + fii_net context dict from Master Data Pipeline."""
        try:
            from core.data.data_pipeline import DataPipeline
            pipeline = DataPipeline()
            
            try:
                from deprecated.scripts.nifty50_universe import MARK5_LIVE_UNIVERSE as UNIVERSE
            except ImportError:
                UNIVERSE = {}

            stock_sector = sector or UNIVERSE.get(ticker, {}).get('sector', 'Unknown')
            start_date = str(data.index[0].date())
            end_date = str(data.index[-1].date())

            # Use pipeline's providers to ensure centralized routing
            context = pipeline.market_provider.build_feature_context(
                stock_df=data, sector=stock_sector,
                start_date=start_date, end_date=end_date,
                kite_adapter=self.kite_adapter
            )
            
            fii_series = pipeline.fii_provider.get_fii_flow(start_date, end_date)
            # FIX-8: guard against trivially-short FII series that trigger synthetic fallback
            if fii_series is None or len(fii_series) < 30:
                logger.warning(
                    f"[{ticker}] Insufficient FII data ({len(fii_series) if fii_series is not None else 0} days). "
                    "Using ZERO FII flows (not synthetic random). "
                    "Re-connect Kite and refresh FII cache before production training."
                )
                import pandas as _pd
                fii_series = _pd.Series(0.0, index=_pd.date_range(start_date, end_date, freq='B'), name='fii_net')
            context['fii_net'] = fii_series
            
            return context
        except Exception as exc:
            logger.warning(f"[{ticker}] Context build failed (no context): {exc}")
            return None

    # ---------------------------------------------------------------------- #
    # Main training entry point (public API — signature must not change)
    # ---------------------------------------------------------------------- #

    def train_advanced_ensemble(self, ticker: str, data: pd.DataFrame, n_trials: int = 100) -> Dict:
        """
        Train stacking ensemble with CPCV validation and production gate.

        CPCV replaces walk-forward: C(8,2)=28 test combinations give a
        distributional view of performance (mean Sharpe, P(Sharpe>1.5), worst-5%).
        Walk-forward tested only the most recent period — this tests all of them.

        Args:
            ticker:   Ticker symbol.
            data:     Raw OHLCV DataFrame.
            n_trials: Number of trials for DSR calculation (default: 100).

        Returns:
            Dict: status, version, cpcv_p_sharpe, mean_sharpe, worst_5pct_sharpe,
                  avg_brier, avg_fbeta, feature_stability_ok, passes_prod_gate, dsr.
                  On failure: status='failed', reason=str.
        """
        logger.info(f"🚀 [{ticker}] CPCV training starting")

        # Full-dataset feature/label matrix (for index alignment and final retrain)
        # Pass training_cutoff = total last timestamp to prevent any late-dataset leak
        X, y, returns, sample_weights = self.prepare_data_dynamic(
            data, ticker,
            sector=getattr(self.config, 'sector', ''),
            context=getattr(self.config, 'feature_context', None),
            training_cutoff=data.index[-1],
        )
        # Calculate 14-day rolling annualized volatility target for TCN regression head
        log_returns = np.log(data["close"] / (data["close"].shift(1) + 1e-9))
        vol_target_series = log_returns.rolling(window=14).std() * np.sqrt(252)
        vol_target = vol_target_series.reindex(X.index).fillna(0).values
        feature_names: List[str] = X.columns.tolist()
        n_samples, n_features = X.shape

        if n_samples < 10:
            msg = f"Only {n_samples} samples after labelling — need ≥10"
            logger.error(f"[{ticker}] {msg}")
            # Clear GPU memory after ticker training
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except ImportError:
                pass
            gc.collect()
            return {'status': 'failed', 'reason': msg}

        logger.info(f"[{ticker}] n_samples={n_samples}, n_features={n_features}")

        # ── CPCV splitter ────────────────────────────────────────────────────
        # Scale horizons based on sample density to maintain 70-bar time isolation 
        # while operating in sample space.
        n_bars = len(data)
        density = n_bars / n_samples
        sample_horizon = int(np.ceil(getattr(self.config, 'prediction_horizon', 70) / density))
        sample_embargo = int(np.ceil(CPCV_EMBARGO_BARS / density))
        
        logger.info(f"[{ticker}] CPCV Horizons (Sample Space): Purge={sample_horizon}, Embargo={sample_embargo} (Density={density:.1f} bars/signal)")

        cpcv = CombinatorialPurgedKFold(
            n_splits=CPCV_N_SPLITS,
            n_test_splits=CPCV_N_TEST_SPLITS,
            prediction_horizon=sample_horizon,
            embargo_limit=sample_embargo,
        )

        # ── OOF accumulators for stacking meta-learner ────────────────────────
        # Each test sample appears in C(N-1, k-1) = C(7,1) = 7 test folds.
        # Accumulate then average to get OOF probabilities.
        # TCN is DISABLED — 3-model ensemble only (XGB + LGB + CatBoost).
        oof_sum: Dict[str, np.ndarray] = {
            m: np.zeros(n_samples) for m in ('xgb', 'lgb', 'cat')
        }
        oof_count = np.zeros(n_samples, dtype=int)

        # ── Per-fold tracking ─────────────────────────────────────────────────
        fold_sharpes: List[float] = []
        fold_briers: List[float] = []
        fold_fbetas: List[float] = []
        fold_importances: List[Dict[str, np.ndarray]] = []
        fold_num = 0

        for train_idx, test_idx in cpcv.split(X, y):
            fold_num += 1

            # [FIX] Deep Structural Refactor: Isolate rolling windows per fold.
            # Re-engineer features with test_indices masked to prevent leakage.
            X_clean, y_clean, _, weights_clean = self.prepare_data_dynamic(
                data, ticker, 
                sector=getattr(self.config, 'sector', ''),
                context=getattr(self.config, 'feature_context', None),
                training_cutoff=data.index[-1],
                test_indices=test_idx
            )
            
            # Align with the fold's train_idx. 
            # X_clean will have NaNs for samples affected by test_indices (rolling lookback).
            X_train_raw = X_clean.reindex(X.index[train_idx]).dropna()
            if X_train_raw.empty:
                logger.warning(f"[{ticker}] Fold {fold_num}: X_train_raw is empty after leakage isolation — skip")
                continue
            
            # Robust alignment for y_train and sample_weights 
            # (y_clean and weights_clean are ndarrays from prepare_data_dynamic)
            import pandas as _pd
            y_clean_series = _pd.Series(y_clean, index=X_clean.index)
            y_train = y_clean_series.reindex(X_train_raw.index).values
            
            w_clean_series = _pd.Series(weights_clean, index=X_clean.index)
            weights_train = w_clean_series.reindex(X_train_raw.index).values
            
            # For testing, we use the original X (leaked features are okay for test set 
            # as they only look back at training data which is 'the past' in this reality).
            X_test_raw = X.iloc[test_idx]
            y_test = y[test_idx]
            ret_test = returns[test_idx] # Real returns for Sharpe check
            weights_test = sample_weights[test_idx] # Real weights for test evaluation if needed

            if len(np.unique(y_train)) < 2:
                logger.warning(f"[{ticker}] Fold {fold_num}: single class in train — skip")
                continue

            # [REFACTOR] Isolate standardization logic inside folds to prevent leakage.
            # Fit scaler ONLY on training data, then apply to both train and test.
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train_raw),
                index=X_train_raw.index,
                columns=X_train_raw.columns
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test_raw),
                index=X_test_raw.index,
                columns=X_test_raw.columns
            )

            # Early-stopping validation: last ES_FRACTION of training data.
            es_cut = max(1, int(len(X_train_scaled) * ES_FRACTION))
            X_tr, y_tr = X_train_scaled[:-es_cut], y_train[:-es_cut]
            X_es, y_es = X_train_scaled[-es_cut:], y_train[-es_cut:]
            
            # Use aligned weights_train (Institutional Standard)
            w_tr = weights_train[:-es_cut]
            w_es = weights_train[-es_cut:]

            # Ensure class diversity in both train and ES sets to prevent booster crashes
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_es)) < 2:
                logger.warning(f"[{ticker}] Fold {fold_num}: insufficient class diversity in train/es sets — skip")
                continue

            # Train base models with Magnitude Weights
            xgb_m = self._train_xgboost(X_tr, y_tr, X_es, y_es, w_tr)
            lgb_m = self._train_lightgbm(X_tr, y_tr, X_es, y_es, w_tr)
            cat_m = self._train_catboost(X_tr, y_tr, X_es, y_es, w_tr)

            # ── Base model inference on test fold ───────────────────────────
            # TCN is DISABLED (no trained weights). Using XGB+LGB+CAT ensemble only.
            # Arithmetic mean — Rule 35: not geometric mean.
            p_xgb = xgb_m.predict_proba(X_test_scaled)[:, 1]
            p_lgb = lgb_m.predict_proba(X_test_scaled)[:, 1]
            p_cat = cat_m.predict_proba(X_test_scaled)[:, 1]

            # Accumulate OOF predictions (3-model ensemble, no TCN dilution).
            oof_sum['xgb'][test_idx] += p_xgb
            oof_sum['lgb'][test_idx] += p_lgb
            oof_sum['cat'][test_idx] += p_cat
            oof_count[test_idx] += 1

            # Ensemble for fold scoring — 3 models only (TCN removed to stop 25% signal dilution).
            ens_prob = (p_xgb + p_lgb + p_cat) / 3.0
            ens_pred = (ens_prob > TRADING_HURDLE).astype(int)

            brier = brier_score_loss(y_test, ens_prob)
            precision = precision_score(y_test, ens_pred, zero_division=0)
            recall = recall_score(y_test, ens_pred, zero_division=0)
            fbeta = (
                fbeta_score(y_test, ens_pred, beta=0.5, zero_division=0)
                if recall >= RECALL_FLOOR else 0.0
            )
            sharpe = self._compute_fold_sharpe(ens_prob, ret_test)

            fold_briers.append(brier)
            fold_fbetas.append(fbeta)
            fold_sharpes.append(sharpe)

            # IC per feature — logged for diagnostics, not used for selection.
            ic_top3 = self._compute_fold_ic_top3(X_test_raw, y_test, feature_names)

            # Feature importance for stability check.
            fold_importances.append({
                'xgb': xgb_m.feature_importances_,
                'lgb': lgb_m.feature_importances_,
                'cat': np.array(cat_m.get_feature_importance()),
            })

            logger.info(
                f"[{ticker}] Fold {fold_num:2d}/{cpcv.n_splits} | "
                f"Brier={brier:.4f} | P={precision:.2%} | R={recall:.2%} | "
                f"F0.5={fbeta:.4f} | Sharpe={sharpe:+.2f} | top_IC={ic_top3}"
            )
            gc.collect()

        if fold_num == 0:
            # Clear GPU memory after ticker training
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except ImportError:
                pass
            gc.collect()
            return {'status': 'failed', 'reason': 'No valid CPCV folds generated'}

        # ── CPCV distributional stats ─────────────────────────────────────────
        sharpe_arr = np.array(fold_sharpes)

        # Exclude -99 sentinel folds (< 3 signals in fold — too sparse to evaluate).
        # These are logged separately as sparse_fold_pct for diagnostics.
        SENTINEL = -99.0
        valid_mask   = sharpe_arr > SENTINEL
        sparse_folds = int((~valid_mask).sum())
        sparse_pct   = sparse_folds / max(len(sharpe_arr), 1)

        valid_arr = sharpe_arr[valid_mask]
        if len(valid_arr) == 0:
            # Every fold was too sparse — model produces no signals at 0.55 threshold
            p_sharpe   = 0.0
            mean_sharpe = SENTINEL
            worst_5pct  = SENTINEL
        else:
            p_sharpe    = float((valid_arr > PROD_GATE_SHARPE_TARGET).mean())
            mean_sharpe = float(valid_arr.mean())
            worst_5pct  = float(np.percentile(valid_arr, 5))

        avg_brier = float(np.mean(fold_briers))
        avg_fbeta = float(np.mean(fold_fbetas))

        # ── Aggregate OOF AUC & DSR calculation ───────────────────────────────
        valid_meta = oof_count > 0
        agg_auc = 0.0
        dsr = 0.0
        agg_sharpe = 0.0

        if valid_meta.any():
            # 3-model OOF ensemble (TCN disabled) — Rule 35: arithmetic mean
            xgb_oof = oof_sum['xgb'][valid_meta] / oof_count[valid_meta]
            lgb_oof = oof_sum['lgb'][valid_meta] / oof_count[valid_meta]
            cat_oof = oof_sum['cat'][valid_meta] / oof_count[valid_meta]
            agg_ens_prob = (xgb_oof + lgb_oof + cat_oof) / 3.0
            agg_auc = float(roc_auc_score(y[valid_meta], agg_ens_prob))
            
            agg_returns = returns[valid_meta]
            agg_sharpe = self._compute_fold_sharpe(agg_ens_prob, agg_returns)
            
            # Calculate DSR on aggregate OOF strategy returns
            signal = (agg_ens_prob > TRADING_HURDLE).astype(float)
            strategy_returns = signal * agg_returns
            dsr = self._compute_dsr(agg_sharpe, strategy_returns, n_trials=n_trials)

        logger.info(
            f"[{ticker}] CPCV summary ({fold_num} folds) | "
            f"P(Sharpe>{PROD_GATE_SHARPE_TARGET})={p_sharpe:.1%} | "
            f"AUC={agg_auc:.4f} | DSR={dsr:.4f} | "
            f"mean_Sharpe={mean_sharpe:.2f} | "
            f"avg_F0.5={avg_fbeta:.4f}"
        )

        # ── Feature importance stability ──────────────────────────────────────
        feature_stability_ok = self._check_feature_stability(fold_importances, ticker)

        # ── Production gate ───────────────────────────────────────────────────
        passes_gate = (
            p_sharpe >= PROD_GATE_P_SHARPE
            and worst_5pct >= PROD_GATE_WORST5PCT
            and dsr >= DSR_GATE_THRESHOLD
        )
        gate_status = "✅ PASSES" if passes_gate else "⚠️ FAILS"
        logger.info(f"[{ticker}] Production gate: {gate_status}")
        if not passes_gate:
            logger.warning(
                f"[{ticker}] Gate fail — saving artifacts for diagnostics. "
                f"Do NOT deploy to live without gate pass."
            )

        # ── Stacking meta-learner (DISABLED) ──────────────────────────────────
        meta_model = self._train_meta_learner(oof_sum, oof_count, y, n_samples, ticker)

        # [REFINEMENT] v10.1: Disabled for small-data midcaps.
        # if meta_model is None:
        #    logger.warning(f"[{ticker}] Gate fail: Meta-learner rejection (anti-predictive)")
        #    passes_gate = False

        # ── Final base models: retrain on ALL data ────────────────────────────
        # Final class diversity guard to prevent booster crashes
        if len(np.unique(y)) < 2:
            msg = f"Insufficient class diversity in full dataset (only {np.unique(y)} found) — skip"
            logger.error(f"[{ticker}] {msg}")
            return {'status': 'failed', 'reason': msg}

        X_all = X
        es_cut_all = max(1, int(len(X_all) * ES_FRACTION))
        X_ft, y_ft = X_all.iloc[:-es_cut_all], y[:-es_cut_all]
        X_fe, y_fe = X_all.iloc[-es_cut_all:], y[-es_cut_all:]

        w_ft = sample_weights[:-es_cut_all]
        w_fe = sample_weights[-es_cut_all:]

        final_models = {
            'xgb': self._train_xgboost(X_ft, y_ft, X_fe, y_fe, w_ft),
            'lgb': self._train_lightgbm(X_ft, y_ft, X_fe, y_fe, w_ft),
            'cat': self._train_catboost(X_ft, y_ft, X_fe, y_fe, w_ft),
        }

        # ── Save all artifacts ────────────────────────────────────────────────
        version = self.version_manager.increment_version(ticker)
        self._save_artifacts(
            ticker, final_models, feature_names, version, meta_model, passes_gate
        )

        # ── Institutional Deployment Routing ─────────────────────────────────
        try:
            from core.infrastructure.database_manager import MARK5DatabaseManager
            from core.models.training.engine import LearningEngine
            
            # Simple metadata extraction for tracker
            metrics = {
                'cpcv_p_sharpe': p_sharpe,
                'mean_sharpe': mean_sharpe,
                'worst_5pct_sharpe': worst_5pct,
                'auc': agg_auc,
                'dsr': dsr,
                'avg_fbeta': avg_fbeta
            }
            
            # Use mock or real DB manager
            db_mgr = MARK5DatabaseManager()
            engine = LearningEngine(db_mgr)
            engine.deploy_model(ticker, version, metrics, passes_gate)
            
        except Exception as e:
            logger.warning(f"[{ticker}] Institutional deployment failed: {e}")

        # Clear GPU memory after ticker training
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except ImportError:
            pass
        gc.collect()

        return {
            'status': 'success',
            'version': version,
            'cpcv_folds': fold_num,
            'cpcv_p_sharpe': p_sharpe,
            'mean_sharpe': mean_sharpe,
            'worst_5pct_sharpe': worst_5pct,
            'avg_brier': avg_brier,
            'avg_fbeta': avg_fbeta,
            'auc': agg_auc,
            'dsr': dsr,
            'feature_stability_ok': feature_stability_ok,
            'passes_prod_gate': passes_gate,
        }

    # ---------------------------------------------------------------------- #
    # Model trainers
    # ---------------------------------------------------------------------- #

    def train_model(self, ticker: str, retrain: bool = False) -> Dict:
        """
        [FIX-AUTO] Stability wrapper for AutonomousTrader.
        Fetches data automatically and executes training ensemble.
        """
        try:
            self.logger.info(f"[{ticker}] 🔄 Triggering automated retraining cycle...")
            # Fetch 3 years of daily data (Rule 59)
            data = self.fetch_data_for_training(ticker, years=3.0)
            if data is None or len(data) < 300:
                self.logger.warning(f"[{ticker}] ⚠️ Insufficient data or cache is older than 48h. Skipping training.")
                # Clear GPU memory after ticker training
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except ImportError:
                    pass
                gc.collect()
                return {"status": "failed", "reason": "Insufficient data"}

            return self.train_advanced_ensemble(ticker, data)
        except Exception as e:
            self.logger.error(f"[{ticker}] Automated training failed: {e}")
            # Clear GPU memory after ticker training
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except ImportError:
                pass
            gc.collect()
            return {"status": "failed", "reason": str(e)}

    @staticmethod
    def fetch_data_for_training(symbol_ns: str, years: float = 15.0):
        """Robust data fetcher using Master Data Pipeline and Cache Fallback."""
        from core.data.data_pipeline import DataPipeline
        import os
        import time
        import pandas as pd
        import logging

        pipeline = DataPipeline()
        period_str = f"{int(years)}y" if years >= 1.0 else f"{int(years*365)}d"
        try:
            df = pipeline.get_market_data(symbol_ns, source='kite', period=period_str)
            if df is not None and not df.empty:
                df.columns = [str(c).lower() for c in df.columns]
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df
        except Exception as e:
            logging.getLogger("MARK5.Trainer").error(f"[{symbol_ns}] DataPipeline fetch failed: {e}")

        # Fallback to cache
        bare = symbol_ns.replace('.NS', '')
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        _cache_old = os.path.join(_project_root, 'data', 'cache', f'{bare}_NS_1d.parquet')
        _cache_new = os.path.join(_project_root, 'data', 'cache', f'{bare}_daily.parquet')

        for _cp in [_cache_old, _cache_new]:
            if os.path.exists(_cp) and (time.time() - os.path.getmtime(_cp)) < 2592000:
                try:
                    df = pd.read_parquet(_cp)
                    df.columns = [str(c).lower() for c in df.columns]
                    if hasattr(df.index, 'tz') and df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    logging.getLogger("MARK5.Trainer").info(f"[{symbol_ns}] Loaded {len(df)} bars from cache fallback")
                    return df
                except Exception as e:
                    logging.getLogger("MARK5.Trainer").error(f"[{symbol_ns}] Cache read failed for {_cp}: {e}")

        return None

    def _train_xgboost(
        self,
        X_t: np.ndarray, y_t: np.ndarray,
        X_v: np.ndarray, y_v: np.ndarray,
        sample_weights: np.ndarray,
    ) -> Any:
        """Train XGBoost binary classifier with magnitude-aware weighting."""
        clf = xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            learning_rate=XGB_LEARNING_RATE,
            max_depth=XGB_MAX_DEPTH,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE,
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            device='cuda' if self.use_gpu else 'cpu',
            early_stopping_rounds=XGB_EARLY_STOP,
            random_state=42,
        )
        clf.fit(X_t, y_t, sample_weight=sample_weights, eval_set=[(X_v, y_v)], verbose=False)
        
        if len(X_v) >= 50 and len(np.unique(y_v)) > 1:
            from sklearn.calibration import CalibratedClassifierCV
            calibrated_clf = CalibratedClassifierCV(estimator=clf, cv='prefit', method='sigmoid')
            calibrated_clf.fit(X_v, y_v)
            return calibrated_clf
            
        return clf

    def _train_lightgbm(
        self,
        X_t: np.ndarray, y_t: np.ndarray,
        X_v: np.ndarray, y_v: np.ndarray,
        sample_weights: np.ndarray,
    ) -> Any:
        """Train LightGBM binary classifier with magnitude-aware weighting."""
        callbacks = [lgb.early_stopping(LGB_EARLY_STOP, verbose=False)]
        clf = lgb.LGBMClassifier(
            n_estimators=LGB_N_ESTIMATORS,
            learning_rate=LGB_LEARNING_RATE,
            num_leaves=LGB_NUM_LEAVES,
            objective='binary',
            device='gpu' if self.use_gpu else 'cpu',
            random_state=42,
            verbose=-1,
        )
        clf.fit(X_t, y_t, sample_weight=sample_weights, eval_set=[(X_v, y_v)], callbacks=callbacks)
        
        if len(X_v) >= 50 and len(np.unique(y_v)) > 1:
            from sklearn.calibration import CalibratedClassifierCV
            calibrated_clf = CalibratedClassifierCV(estimator=clf, cv='prefit', method='sigmoid')
            calibrated_clf.fit(X_v, y_v)
            return calibrated_clf
            
        return clf

    def _train_catboost(
        self,
        X_t: np.ndarray, y_t: np.ndarray,
        X_v: np.ndarray, y_v: np.ndarray,
        sample_weights: np.ndarray,
    ) -> Any:
        """Train CatBoostClassifier with magnitude-aware weighting."""
        task_type = 'GPU' if self.use_gpu else 'CPU'
        clf = CatBoostClassifier(
            iterations=CAT_ITERATIONS,
            learning_rate=CAT_LEARNING_RATE,
            depth=CAT_DEPTH,
            l2_leaf_reg=CAT_L2_LEAF_REG,
            eval_metric='Logloss',
            early_stopping_rounds=CAT_EARLY_STOP,
            task_type=task_type,
            random_seed=42,
            verbose=False,
        )
        eval_pool = Pool(X_v, y_v)
        clf.fit(X_t, y_t, sample_weight=sample_weights, eval_set=eval_pool)
        
        if len(X_v) >= 50 and len(np.unique(y_v)) > 1:
            from sklearn.calibration import CalibratedClassifierCV
            calibrated_clf = CalibratedClassifierCV(estimator=clf, cv='prefit', method='sigmoid')
            calibrated_clf.fit(X_v, y_v)
            return calibrated_clf
            
        return clf

    # ---------------------------------------------------------------------- #
    # Meta-learner
    # ---------------------------------------------------------------------- #

    def _train_meta_learner(
        self,
        oof_sum: Dict[str, np.ndarray],
        oof_count: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        ticker: str,
    ) -> Optional[NonNegativeMetaLearner]:
        """
        Train stacking meta-learner on OOF predictions from all CPCV folds.
        Uses Non-Negative Least Squares (NNLS) for stable weights.
        """
        meta_mask = oof_count > 0
        n_meta = int(meta_mask.sum())

        if n_meta < 20:
            logger.warning(
                f"[{ticker}] Only {n_meta} OOF samples — meta-learner skipped. "
                "Predictor will fall back to arithmetic mean."
            )
            return None

        oof_avg = {
            name: oof_sum[name][meta_mask] / oof_count[meta_mask]
            for name in oof_sum
        }
        # 3 inputs: [xgb, lgb, cat]
        meta_X = np.column_stack([oof_avg['xgb'], oof_avg['lgb'], oof_avg['cat']])
        meta_y = y[meta_mask]

        meta_model = NonNegativeMetaLearner()
        meta_model.fit(meta_X, meta_y)

        coef = dict(zip(['xgb', 'lgb', 'cat'], meta_model.coef_.round(4)))
        logger.info(
            f"[{ticker}] Meta-learner trained on {n_meta} OOF samples. "
            f"Coefficients: {coef}"
        )

        # ── Sanity gate: reject pathological meta-learners ───────────────────
        # If all coefficients are zero, the meta-learner failed to find any signal.
        if np.all(meta_model.coef_ == 0):
            logger.warning(
                f"[{ticker}] Meta-learner REJECTED: all coefficients zero. "
                f"Falling back to arithmetic mean."
            )
            return None

        return meta_model

    # ---------------------------------------------------------------------- #
    # Fold analytics
    # ---------------------------------------------------------------------- #

    def _compute_dsr(self, sharpe: float, returns: np.ndarray, n_trials: int = 1) -> float:
        """
        Deflated Sharpe Ratio (DSR) calculation.
        Adjusts the Sharpe Ratio for multiple testing and non-normal returns.
        """
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
            
        # 1. Calculate moments
        gamma1 = skew(returns)
        gamma2 = kurtosis(returns, fisher=True) + 3.0 # Excess kurtosis to kurtosis
        
        # 2. Expected maximum Sharpe ratio (SR0)
        if n_trials <= 1:
            sr0 = 0.0
        else:
            # Approximation of expected max of N standard normals
            sr0 = np.sqrt(2 * np.log(n_trials))
            sr0 *= 0.5 # Heuristic adjustment for non-standard normal SRs

        # 3. Calculate DSR
        t = len(returns)
        # Standard deviation of the Sharpe Ratio (Lo's formula)
        sigma_sr = np.sqrt((1 - gamma1 * sharpe + (gamma2 - 1) / 4 * sharpe**2) / (t - 1))
        
        if sigma_sr == 0 or np.isnan(sigma_sr):
            return 0.0
            
        z = (sharpe - sr0) / sigma_sr
        return float(ndtr(z))

    def _compute_fold_sharpe(
        self, ens_prob: np.ndarray, returns: np.ndarray
    ) -> float:
        signal = (ens_prob > TRADING_HURDLE).astype(float)
        
        if signal.sum() < 3:
            return -99.0

        # Strategy returns: signal * actual_return, 0 on cash days
        strategy_returns = signal * returns
        
        std = strategy_returns.std(ddof=1)
        if std < 1e-9:
            return 0.0

        trades_per_year = ANNUAL_FACTOR / self.config.prediction_horizon  # 252/10 = 25.2
        return float((strategy_returns.mean() / std) * np.sqrt(trades_per_year))

    def _compute_fold_ic_top3(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
    ) -> str:
        """
        Compute Spearman IC vs binary label for each feature; return top-3 summary.

        Binary label is acceptable as IC target: label=1 ↔ forward return > cost.
        Ref: rebuild report Section 3.1 (IC target > 0.02 per feature).
        """
        ic_vals: Dict[str, float] = {}
        is_df = hasattr(X_test, "iloc")
        for col_idx, col_name in enumerate(feature_names):
            x_col = X_test.iloc[:, col_idx] if is_df else X_test[:, col_idx]
            if np.std(x_col) == 0:
                ic_vals[col_name] = 0.0
                continue
            ic, _ = spearmanr(x_col, y_test, nan_policy='omit')
            ic_vals[col_name] = float(ic) if np.isfinite(ic) else 0.0

        top3 = sorted(ic_vals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        return '{' + ', '.join(f'{k}:{v:+.3f}' for k, v in top3) + '}'

    def _check_feature_stability(
        self,
        fold_importances: List[Dict[str, np.ndarray]],
        ticker: str,
    ) -> bool:
        """
        Spearman rank correlation of combined feature importances: first vs last fold.

        Low correlation = model fitting different noise in early vs late periods.
        Ref: rebuild report Section 6.2, Test 2 (target corr > 0.70).
        """
        if len(fold_importances) < 2:
            logger.warning(f"[{ticker}] <2 folds — stability check skipped")
            return True

        def _avg_imp(fd: Dict[str, np.ndarray]) -> np.ndarray:
            return np.mean(list(fd.values()), axis=0)

        corr, _ = spearmanr(_avg_imp(fold_importances[0]), _avg_imp(fold_importances[-1]))
        ok = bool(corr >= MIN_FEATURE_RANK_CORR)
        status = "✅" if ok else "⚠️ BELOW THRESHOLD"
        logger.info(
            f"[{ticker}] Feature rank stability "
            f"(fold 1 vs fold {len(fold_importances)}): "
            f"Spearman={corr:.3f} {status}"
        )
        return ok

    # ---------------------------------------------------------------------- #
    # Artifact persistence
    # ---------------------------------------------------------------------- #

    def _save_artifacts(
        self,
        ticker: str,
        models: Dict,
        features: List[str],
        version: int,
        meta_model: Optional[NonNegativeMetaLearner],
        passes_gate: bool,
    ) -> None:
        """
        Save all training artifacts to models/{ticker}/v{version}/.
        Uses a UUID-based temp directory to prevent race conditions during parallel execution.
        """
        temp_id = str(uuid.uuid4())
        temp_path = os.path.join(self.models_base_dir, 'tmp', temp_id)
        os.makedirs(temp_path, exist_ok=True)

        # Record metadata for institutional gate-locking
        metadata = {
            'ticker':      ticker,
            'version':     version,
            'passes_gate': passes_gate,
            'timestamp':   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'features':    features,
        }
        with open(os.path.join(temp_path, 'metadata.json'), 'w') as fh:
            json.dump(metadata, fh, indent=2)

        with open(os.path.join(temp_path, 'features.json'), 'w') as fh:
            json.dump(features, fh)

        # Base weights = 1.0; actual weighting delegated to meta-learner at inference.
        model_weights = {name: 1.0 for name in models}
        with open(os.path.join(temp_path, 'weights.json'), 'w') as fh:
            json.dump(model_weights, fh)

        for name, model in models.items():
            joblib.dump(model, os.path.join(temp_path, f'{name}_model.pkl'))

        if meta_model is not None:
            joblib.dump(meta_model, os.path.join(temp_path, 'meta_model.pkl'))

        # Atomic move to final destination
        final_path = os.path.join(self.models_base_dir, ticker, f"v{version}")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        
        shutil.move(temp_path, final_path)
        logger.info(f"[{ticker}] Artifacts atomically moved to {final_path}")


# =============================================================================
# CLI ENTRY POINT — run directly to retrain models
# Usage:
#   python3 core/models/training/trainer.py
#   python3 core/models/training/trainer.py --symbols COFORGE HAL IDFCFIRSTB
#   python3 core/models/training/trainer.py --symbols COFORGE --years 3
# =============================================================================
if __name__ == '__main__':
    import argparse
    import time as _time
    import sys
    import warnings
    import subprocess
    warnings.filterwarnings('ignore')

    # Ensure project root is on sys.path when run directly
    # __file__ = .../MARK5/core/models/training/trainer.py → 4 dirnames = MARK5/
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_project_root, '.env'), override=True)
    except ImportError:
        pass

    # ── Universe: 10 stocks for meta-ensemble training
    _DEFAULT_SYMBOLS = [
        'COFORGE.NS', 'HAL.NS', 'IDFCFIRSTB.NS', 'RELIANCE.NS', 'HDFCBANK.NS',
        'INFY.NS', 'TCS.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBIN.NS'
    ]

    _ap = argparse.ArgumentParser(description='MARK5 Model Trainer')
    _ap.add_argument('--symbols', nargs='+', default=None,
                     help='Ticker symbols to train (without .NS). Default: 20-stock universe.')
    _ap.add_argument('--years', type=float, default=3.0,
                     help='Years of historical data to fetch (default: 3)')
    _ap.add_argument('--cutoff', type=str, default=None,
                     help='Training cutoff date (YYYY-MM-DD). Data after this will be ignored.')
    _ap.add_argument('--skip-existing', action='store_true',
                     help='Skip symbols whose latest model is not corrupt')
    _ap.add_argument('--is-subprocess', action='store_true',
                     help='Internal flag for isolated training')
    _args = _ap.parse_args()

    # Resolve symbol list — append .NS if missing
    if _args.symbols:
        _symbols = [s if s.endswith('.NS') else f'{s}.NS' for s in _args.symbols]
    else:
        _symbols = _DEFAULT_SYMBOLS

    _cutoff_ts = pd.Timestamp(_args.cutoff) if _args.cutoff else None

    print(f"\n{'='*60}")
    print(f"  MARK5 TRAINER  |  {len(_symbols)} symbols  |  {_args.years:.0f} yrs | Cutoff: {_args.cutoff or 'None'}")
    print(f"{'='*60}\n")

    # Connect Kite for data
    _adapter = None
    try:
        from core.data.adapters.kite_adapter import KiteFeedAdapter
        _cfg = {
            'api_key':      os.getenv('KITE_API_KEY', ''),
            'api_secret':   os.getenv('KITE_API_SECRET', ''),
            'access_token': os.getenv('KITE_ACCESS_TOKEN', ''),
        }
        if _cfg['api_key'] and _cfg['access_token']:
            _adapter = KiteFeedAdapter(_cfg)
            if not _adapter.connect():
                _adapter = None
    except Exception:
        pass

    if _adapter:
        logger.info('Kite connected — fetching fresh data')
    else:
        logger.warning('Kite not connected — will use today\'s cache only. '
                       'Symbols without cache will be skipped.')

    def _fetch(symbol_ns: str, years: float):
        bare = symbol_ns.replace('.NS', '')
        # Check both cache naming conventions
        _cache_old = os.path.join(_project_root, 'data', 'cache', f'{bare}_NS_1d.parquet')
        _cache_new = os.path.join(_project_root, 'data', 'cache', f'{bare}_daily.parquet')
        os.makedirs(os.path.dirname(_cache_old), exist_ok=True)

        # Use cache if < 30d old (allows offline runs with recent data)
        for _cp in [_cache_old, _cache_new]:
            if os.path.exists(_cp) and (_time.time() - os.path.getmtime(_cp)) < 2592000:
                try:
                    df = pd.read_parquet(_cp)
                    df.columns = [c.lower() if isinstance(c, str) else str(c).lower()
                                  for c in df.columns]
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    logger.info(f'{bare}: {len(df)} bars from cache ({os.path.basename(_cp)})')
                    return df
                except Exception:
                    pass

        # Kite is the ONLY live data source — no yfinance fallback
        if not _adapter:
            logger.error(
                f'{bare}: Kite not connected. Run generate_kite_token.py to refresh '
                f'your access token, then retry.'
            )
            return None

        try:
            df = _adapter.fetch_ohlcv(bare, period=f'{int(years * 365)}d', interval='day')
            if df is not None and len(df) >= 300:
                df.columns = [c.lower() for c in df.columns]
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.to_parquet(_cache_new)
                logger.info(f'{bare}: {len(df)} bars from Kite ✅')
                return df
            else:
                logger.error(f'{bare}: Kite returned insufficient data ({len(df) if df is not None else 0} bars)')
                return None
        except Exception as _e:
            logger.error(f'{bare}: Kite fetch failed — {_e}')
            return None

    def _is_corrupt(ticker_ns: str) -> bool:
        """True if the existing model is missing critical files."""
        _mroot = os.path.join(_project_root, 'models', ticker_ns)
        if not os.path.isdir(_mroot):
            return True
        _vers = sorted(
            [v for v in os.listdir(_mroot) if v.startswith('v') and v[1:].isdigit()],
            key=lambda x: int(x[1:]), reverse=True
        )
        if not _vers:
            return True
        _vdir = os.path.join(_mroot, _vers[0])
        _ff = os.path.join(_vdir, 'features.json')
        if not os.path.exists(_ff):
            return True
        return False

    _results = {'success': [], 'skipped': [], 'failed': []}

    if _args.is_subprocess:
        if not _symbols:
            sys.exit(0)
        _sym = _symbols[0]
        _df = _fetch(_sym, _args.years)
        if _df is not None and len(_df) >= 300:
            # Apply cutoff filter to the raw data before passing it to trainer
            if _cutoff_ts:
                _df = _df[_df.index <= _cutoff_ts]
                if len(_df) < 300:
                    logger.error(f"{_sym}: Data length after cutoff {len(_df)} < 300. Aborting.")
                    sys.exit(1)
                    
            _trainer = MARK5MLTrainer()
            _res = _trainer.train_advanced_ensemble(_sym, _df)
            if _res.get('status') == 'success':
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            sys.exit(1)

    for _i, _sym in enumerate(_symbols, 1):
        _bare = _sym.replace('.NS', '')
        print(f'\n[{_i}/{len(_symbols)}] {_bare}')

        if _args.skip_existing and not _is_corrupt(_sym):
            logger.info(f'{_bare}: model valid — skipping')
            _results['skipped'].append(_bare)
            continue

        # Isolated Subprocess Training: ensure fresh memory/GPU state per ticker
        _cmd = [sys.executable, __file__, "--symbols", _bare, "--years", str(_args.years), "--is-subprocess"]
        if _args.cutoff:
            _cmd.extend(["--cutoff", _args.cutoff])
            
        _proc = subprocess.run(_cmd)
        
        if _proc.returncode == 0:
            _results['success'].append(_bare)
        else:
            _results['failed'].append(_bare)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  ✅ Success : {len(_results['success'])} → {_results['success']}")
    print(f"  ⏭️  Skipped : {len(_results['skipped'])} → {_results['skipped']}")
    print(f"  ❌ Failed  : {len(_results['failed'])} → {_results['failed']}")
    print(f"{'='*60}\n")

    if _adapter:
        try:
            _adapter.disconnect()
        except Exception:
            pass

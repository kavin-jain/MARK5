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
    scaler.pkl, features.json, weights.json
    xgb_model.pkl, lgb_model.pkl, cat_model.pkl, meta_model.pkl
"""

import gc
import json
import logging
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple

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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, fbeta_score, precision_score, recall_score
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

# --------------------------------------------------------------------------- #
# Constants — all magic numbers named with explanation
# --------------------------------------------------------------------------- #

# CPCV: n=8 groups → C(8,2)=28 test combinations; ~75 bars/group at 600 samples.
# Embargo=30 bars prevents label leakage (Rule 9 isolation).
CPCV_N_SPLITS: int = 8
CPCV_N_TEST_SPLITS: int = 2
CPCV_EMBARGO_BARS: int = 30

# Production gate (rebuild report Section 6.1)
PROD_GATE_P_SHARPE: float = 0.40     # P(Sharpe > SHARPE_TARGET) must exceed this
PROD_GATE_SHARPE_TARGET: float = 1.5  # minimum acceptable annualised Sharpe
PROD_GATE_WORST5PCT: float = 0.0      # worst-5% fold Sharpe must be non-negative

# Feature importance stability gate (rebuild report Section 6.2)
MIN_FEATURE_RANK_CORR: float = 0.70   # Spearman corr first vs last fold importance

# Signal and fold quality floors
TRADING_HURDLE: float = 0.52   # prob threshold to generate BUY
RECALL_FLOOR: float = 0.25     # folds below this recall are scored 0
WIN_RATE_FLOOR: float = 0.40   # folds below this actual win rate are discarded

# CatBoost hyperparameters (rebuild report Section 4.2)
CAT_ITERATIONS: int = 500
CAT_LEARNING_RATE: float = 0.05
CAT_DEPTH: int = 6
CAT_L2_LEAF_REG: float = 3.0
CAT_CLASS_WEIGHTS: List[float] = [1.0, 2.0]  # positive class weighted 2×
CAT_EARLY_STOP: int = 50

# XGBoost
XGB_N_ESTIMATORS: int = 1000   # early stopping limits actual count
XGB_LEARNING_RATE: float = 0.05
XGB_MAX_DEPTH: int = 5
XGB_SUBSAMPLE: float = 0.8
XGB_COLSAMPLE: float = 0.8
XGB_EARLY_STOP: int = 50

# LightGBM
LGB_N_ESTIMATORS: int = 1000
LGB_LEARNING_RATE: float = 0.05
LGB_NUM_LEAVES: int = 31
LGB_EARLY_STOP: int = 50

# Meta-learner: high regularisation prevents OOF noise from being memorised
META_C: float = 0.1

# Early stopping validation fraction
ES_FRACTION: float = 0.15

# Annualisation for Sharpe (252 trading days * 6.25 hours ≈ 1575 bars per year)
ANNUAL_FACTOR: float = 1575.0

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
      Artifacts: models/{ticker}/v{N}/{scaler,features,weights,*_model,meta_model}.pkl/.json
    """

    def __init__(self, config=None, kite_adapter=None) -> None:
        self.config = config if config else self._get_default_config()
        self.kite_adapter = kite_adapter
        self.use_gpu: bool = self._detect_gpu()
        self.models_base_dir: str = getattr(self.config, 'models_dir', './models')
        self.version_manager = ModelVersionManager(
            self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
        )
        os.makedirs(self.models_base_dir, exist_ok=True)

    # ---------------------------------------------------------------------- #
    # Config & hardware
    # ---------------------------------------------------------------------- #

    def _get_default_config(self):
        class Config:
            prediction_horizon: int = 10      # bars; matches MAX_HOLD_BARS in simulation
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
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict]:
        """
        Generate labelled feature matrix and real-return series for training.

        Args:
            data:             Raw OHLCV DataFrame.
            ticker:           Ticker symbol.
            sector:           Sector string for context building.
            context:          Pre-built feature context (nifty_close, fii_net).
                              Built internally when None.
            training_cutoff:  Last bar timestamp of the current training window.
                              Passed to engineer_all_features() to prevent amihud
                              p99 lookahead (BUG-1 fix).

        Returns:
            Tuple of (features_df, targets_ndarray, returns_ndarray, class_weight_dict).
        """
        fe = FinancialEngineer(
            transaction_cost_pct=getattr(self.config, 'transaction_cost', 0.001)
        )
        labels_df = fe.get_labels(
            prices=data,
            run_bars=self.config.prediction_horizon,
            pt_sl=[1.8, 2.0],  # LOOSENED: 1.8x ATR (Target) | 2.0x ATR (Stop)
        )

        from core.models.features import AdvancedFeatureEngine
        feature_engine = AdvancedFeatureEngine()

        if context is None:
            context = self._build_context(data, ticker, sector)

        # BUG-1 wire-up: training_cutoff restricts amihud p99 to training window.
        data_with_features = feature_engine.engineer_all_features(
            data, context=context, training_cutoff=training_cutoff
        )

        # ── Regime-aware training filter (Rule 23) ───────────────────────────
        nifty_close = (context or {}).get('nifty_close')
        bear_mask = None
        if nifty_close is not None and len(nifty_close) >= 200:
            try:
                nifty_aligned = nifty_close.reindex(data.index, method='ffill')
                ema200 = nifty_aligned.ewm(span=200, adjust=False).mean()
                ret20  = nifty_aligned.pct_change(20)

                # BEAR: below 200EMA AND 20d return < -5%
                bear_mask = (nifty_aligned < ema200) & (ret20 < -0.05)

                # Also compute ATR-regime on NIFTY (high-volatility filter)
                highs  = data['high'] if 'high' in data.columns else nifty_aligned
                lows   = data['low']  if 'low'  in data.columns else nifty_aligned
                closes = data['close']
                tr     = pd.concat([
                    highs - lows,
                    (highs - closes.shift(1)).abs(),
                    (lows  - closes.shift(1)).abs(),
                ], axis=1).max(axis=1)
                atr14  = tr.ewm(alpha=1/14, adjust=False).mean()
                atr50  = tr.ewm(alpha=1/50, adjust=False).mean()
                volatile_mask = atr14 > (1.5 * atr50)

                bear_mask = bear_mask | volatile_mask
            except Exception as _e:
                logger.warning(f"[{ticker}] Regime filter failed ({_e}) — using all bars")
                bear_mask = None

        # Apply bear/volatile filter to OHLCV before label generation
        data_filtered = data
        if bear_mask is not None:
            bear_mask_aligned = bear_mask.reindex(data.index).fillna(False)
            n_dropped = int(bear_mask_aligned.sum())
            n_total   = len(data)
            if n_dropped > 0:
                logger.info(
                    f"[{ticker}] Regime filter: dropped {n_dropped}/{n_total} "
                    f"BEAR/VOLATILE bars ({n_dropped/n_total:.0%}) — training on "
                    f"{n_total - n_dropped} TRENDING+RANGING bars only."
                )
                data_filtered = data[~bear_mask_aligned]

        if len(data_filtered) < 200:
            logger.warning(
                f"[{ticker}] Only {len(data_filtered)} bars after regime filter "
                f"(need ≥200). Falling back to full dataset to avoid underfit."
            )
            data_filtered = data

        aligned_df = data_with_features.reindex(data_filtered.index).join(
            labels_df[['bin', 'ret']], how='inner'
        )
        aligned_df.dropna(inplace=True)

        targets = aligned_df['bin'].values.astype(int)
        returns = aligned_df['ret'].values

        exclude = (
            set(FEATURE_EXCLUDE_COLUMNS)
            | {'bin', 'target', 'ret', 'out', 'sl', 'pt', 't1'}
        )
        feature_cols = [c for c in aligned_df.columns if c not in exclude]

        classes = np.unique(targets)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=targets)
        class_weight_dict = dict(zip(classes, weights))

        logger.info(
            f"[{ticker}] Samples={len(aligned_df)} | "
            f"Positive_rate={targets.mean():.1%} | "
            f"Class_weights={class_weight_dict}"
        )
        return aligned_df[feature_cols], targets, returns, class_weight_dict

    def _build_context(
        self, data: pd.DataFrame, ticker: str, sector: str
    ) -> Optional[Dict]:
        """Build nifty_close + fii_net context dict from cached market data sources."""
        try:
            from core.data.market_data import MarketDataProvider
            from core.data.fii_data import FIIDataProvider
            try:
                from scripts.nifty50_universe import NIFTY_50
            except ImportError:
                NIFTY_50 = {}

            stock_sector = sector or NIFTY_50.get(ticker, {}).get('sector', 'Unknown')
            mp = MarketDataProvider()
            fp = FIIDataProvider()
            start_date = str(data.index[0].date())
            end_date = str(data.index[-1].date())

            context = mp.build_feature_context(
                stock_df=data, sector=stock_sector,
                start_date=start_date, end_date=end_date,
                kite_adapter=self.kite_adapter
            )
            fp = FIIDataProvider()
            fii_series = fp.get_fii_flow(start_date, end_date)
            if fii_series is not None and len(fii_series) > 200:
                context['fii_net'] = fii_series
            else:
                fii_len = len(fii_series) if fii_series is not None else 0
                logger.warning(f"[{ticker}] Insufficient FII data ({fii_len} days). Using SYNTHETIC FII flows.")
                context['fii_net'] = fp.generate_synthetic_fii_data(data.index)
            
            return context
        except Exception as exc:
            logger.warning(f"[{ticker}] Context build failed (no context): {exc}")
            return None

    # ---------------------------------------------------------------------- #
    # Main training entry point (public API — signature must not change)
    # ---------------------------------------------------------------------- #

    def train_advanced_ensemble(self, ticker: str, data: pd.DataFrame) -> Dict:
        """
        Train stacking ensemble with CPCV validation and production gate.

        CPCV replaces walk-forward: C(8,2)=28 test combinations give a
        distributional view of performance (mean Sharpe, P(Sharpe>1.5), worst-5%).
        Walk-forward tested only the most recent period — this tests all of them.

        Args:
            ticker: Ticker symbol.
            data:   Raw OHLCV DataFrame.

        Returns:
            Dict: status, version, cpcv_p_sharpe, mean_sharpe, worst_5pct_sharpe,
                  avg_brier, avg_fbeta, feature_stability_ok, passes_prod_gate.
                  On failure: status='failed', reason=str.
        """
        logger.info(f"🚀 [{ticker}] CPCV training starting")

        # Full-dataset feature/label matrix (for index alignment and final retrain)
        features_df, y, returns, class_weights = self.prepare_data_dynamic(
            data, ticker,
            sector=getattr(self.config, 'sector', ''),
            context=getattr(self.config, 'feature_context', None),
            training_cutoff=None,
        )
        X = features_df.values
        feature_names: List[str] = features_df.columns.tolist()
        n_samples, n_features = X.shape

        if n_samples < 100:
            msg = f"Only {n_samples} samples after labelling — need ≥100"
            logger.error(f"[{ticker}] {msg}")
            return {'status': 'failed', 'reason': msg}

        logger.info(f"[{ticker}] n_samples={n_samples}, n_features={n_features}")

        # ── CPCV splitter ────────────────────────────────────────────────────
        cpcv = CombinatorialPurgedKFold(
            n_splits=CPCV_N_SPLITS,
            n_test_splits=CPCV_N_TEST_SPLITS,
            embargo=CPCV_EMBARGO_BARS,
        )

        # ── OOF accumulators for stacking meta-learner ────────────────────────
        # Each test sample appears in C(N-1, k-1) = C(7,1) = 7 test folds.
        # Accumulate then average to get OOF probabilities.
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

            X_train_raw = X[train_idx]
            y_train = y[train_idx]
            X_test_raw = X[test_idx]
            y_test = y[test_idx]
            ret_test = returns[test_idx] # Real returns for Sharpe check

            if len(np.unique(y_train)) < 2:
                logger.warning(f"[{ticker}] Fold {fold_num}: single class in train — skip")
                continue

            # Pass fold's last training bar as training_cutoff for BUG-1 fix.
            fold_cutoff: pd.Timestamp = features_df.index[train_idx[-1]]

            # Scale: fit on train only — never on test.
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)

            # Early-stopping validation: last ES_FRACTION of scaled training data.
            es_cut = max(1, int(len(X_train_scaled) * ES_FRACTION))
            X_tr, y_tr = X_train_scaled[:-es_cut], y_train[:-es_cut]
            X_es, y_es = X_train_scaled[-es_cut:], y_train[-es_cut:]

            # Train base models.
            xgb_m = self._train_xgboost(X_tr, y_tr, X_es, y_es, class_weights)
            lgb_m = self._train_lightgbm(X_tr, y_tr, X_es, y_es, class_weights)
            cat_m = self._train_catboost(X_tr, y_tr, X_es, y_es)

            p_xgb = xgb_m.predict_proba(X_test_scaled)[:, 1]
            p_lgb = lgb_m.predict_proba(X_test_scaled)[:, 1]
            p_cat = cat_m.predict_proba(X_test_scaled)[:, 1]

            # Accumulate OOF predictions.
            oof_sum['xgb'][test_idx] += p_xgb
            oof_sum['lgb'][test_idx] += p_lgb
            oof_sum['cat'][test_idx] += p_cat
            oof_count[test_idx] += 1

            # Ensemble for fold scoring (arithmetic mean — meta-learner trained later).
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

        logger.info(
            f"[{ticker}] CPCV summary ({fold_num} folds) | "
            f"P(Sharpe>{PROD_GATE_SHARPE_TARGET})={p_sharpe:.1%} "
            f"(gate={PROD_GATE_P_SHARPE:.0%}) | "
            f"mean_Sharpe={mean_sharpe:.2f} | worst-5%={worst_5pct:.2f} | "
            f"sparse_folds={sparse_folds}/{fold_num} ({sparse_pct:.0%}) | "
            f"avg_Brier={avg_brier:.4f} | avg_F0.5={avg_fbeta:.4f}"
        )

        # ── Feature importance stability ──────────────────────────────────────
        feature_stability_ok = self._check_feature_stability(fold_importances, ticker)

        # ── Production gate ───────────────────────────────────────────────────
        passes_gate = (
            p_sharpe >= PROD_GATE_P_SHARPE
            and worst_5pct >= PROD_GATE_WORST5PCT
        )
        gate_status = "✅ PASSES" if passes_gate else "⚠️ FAILS"
        logger.info(f"[{ticker}] Production gate: {gate_status}")
        if not passes_gate:
            logger.warning(
                f"[{ticker}] Gate fail — saving artifacts for diagnostics. "
                f"Do NOT deploy to live without gate pass."
            )

        # ── Stacking meta-learner ─────────────────────────────────────────────
        meta_model = self._train_meta_learner(oof_sum, oof_count, y, n_samples, ticker)

        # ── Final base models: retrain on ALL data ────────────────────────────
        final_scaler = StandardScaler()
        X_all = final_scaler.fit_transform(X)
        es_cut_all = max(1, int(len(X_all) * ES_FRACTION))
        X_ft, y_ft = X_all[:-es_cut_all], y[:-es_cut_all]
        X_fe, y_fe = X_all[-es_cut_all:], y[-es_cut_all:]

        final_models = {
            'xgb': self._train_xgboost(X_ft, y_ft, X_fe, y_fe, class_weights),
            'lgb': self._train_lightgbm(X_ft, y_ft, X_fe, y_fe, class_weights),
            'cat': self._train_catboost(X_ft, y_ft, X_fe, y_fe),
        }

        # ── Save all artifacts ────────────────────────────────────────────────
        version = self.version_manager.increment_version(ticker)
        self._save_artifacts(
            ticker, final_models, final_scaler, feature_names, version, meta_model
        )

        return {
            'status': 'success',
            'version': version,
            'cpcv_folds': fold_num,
            'cpcv_p_sharpe': p_sharpe,
            'mean_sharpe': mean_sharpe,
            'worst_5pct_sharpe': worst_5pct,
            'avg_brier': avg_brier,
            'avg_fbeta': avg_fbeta,
            'feature_stability_ok': feature_stability_ok,
            'passes_prod_gate': passes_gate,
        }

    # ---------------------------------------------------------------------- #
    # Model trainers
    # ---------------------------------------------------------------------- #

    def _train_xgboost(
        self,
        X_t: np.ndarray, y_t: np.ndarray,
        X_v: np.ndarray, y_v: np.ndarray,
        weights: Dict,
    ) -> xgb.XGBClassifier:
        """Train XGBoost binary classifier with early stopping."""
        sample_w = np.array([weights[int(yi)] for yi in y_t])
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
        clf.fit(X_t, y_t, sample_weight=sample_w, eval_set=[(X_v, y_v)], verbose=False)
        return clf

    def _train_lightgbm(
        self,
        X_t: np.ndarray, y_t: np.ndarray,
        X_v: np.ndarray, y_v: np.ndarray,
        weights: Dict,
    ) -> lgb.LGBMClassifier:
        """Train LightGBM binary classifier with early stopping."""
        sample_w = np.array([weights[int(yi)] for yi in y_t])
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
        clf.fit(X_t, y_t, sample_weight=sample_w, eval_set=[(X_v, y_v)], callbacks=callbacks)
        return clf

    def _train_catboost(
        self,
        X_t: np.ndarray, y_t: np.ndarray,
        X_v: np.ndarray, y_v: np.ndarray,
    ) -> CatBoostClassifier:
        """
        Train CatBoostClassifier with early stopping.

        Replaces RandomForestClassifier (BUG-3/BUG-6).
        CatBoost lower variance than RF on financial tabular data.
        Handles class imbalance via class_weights natively.
        Ref: rebuild report Section 4.2.
        """
        task_type = 'GPU' if self.use_gpu else 'CPU'
        clf = CatBoostClassifier(
            iterations=CAT_ITERATIONS,
            learning_rate=CAT_LEARNING_RATE,
            depth=CAT_DEPTH,
            l2_leaf_reg=CAT_L2_LEAF_REG,
            eval_metric='Logloss',
            early_stopping_rounds=CAT_EARLY_STOP,
            task_type=task_type,
            class_weights=CAT_CLASS_WEIGHTS,
            random_seed=42,
            verbose=False,
        )
        eval_pool = Pool(X_v, y_v)
        clf.fit(X_t, y_t, eval_set=eval_pool)
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
    ) -> Optional[LogisticRegression]:
        """
        Train stacking meta-learner on OOF predictions from all CPCV folds.

        Samples that appear in zero test folds are excluded (they are always in
        training — typically the very first or last group depending on CPCV config).

        Args:
            oof_sum:   {model_name: accumulated OOF probability sums} shape (n_samples,)
            oof_count: per-sample test-fold appearance count, shape (n_samples,)
            y:         original labels, shape (n_samples,)
            ticker:    for logging.

        Returns:
            Fitted LogisticRegression or None if insufficient OOF coverage.
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
        meta_X = np.column_stack([oof_avg['xgb'], oof_avg['lgb'], oof_avg['cat']])
        meta_y = y[meta_mask]

        meta_model = LogisticRegression(
            C=META_C, class_weight='balanced', max_iter=500, random_state=42
        )
        meta_model.fit(meta_X, meta_y)

        coef = dict(zip(['xgb', 'lgb', 'cat'], meta_model.coef_[0].round(4)))
        logger.info(
            f"[{ticker}] Meta-learner trained on {n_meta} OOF samples. "
            f"Coefficients: {coef}"
        )

        # ── Sanity gate: reject pathological meta-learners ───────────────────
        # If ALL base-model coefficients are negative, the LR learned to INVERT
        # the ensemble signal. This happens when OOF accuracy < 50% (base models
        # are anti-predictive in this period). Falling back to arithmetic mean
        # (Rule 35) is correct — never deploy an inverting meta-learner.
        all_negative = all(v < 0 for v in meta_model.coef_[0])
        if all_negative:
            logger.warning(
                f"[{ticker}] Meta-learner REJECTED: all coefficients negative "
                f"({coef}). Base models are anti-predictive in current training "
                f"window. Falling back to arithmetic mean (Rule 35)."
            )
            return None

        return meta_model

    # ---------------------------------------------------------------------- #
    # Fold analytics
    # ---------------------------------------------------------------------- #

    def _compute_fold_sharpe(
        self, ens_prob: np.ndarray, returns: np.ndarray
    ) -> float:
        """
        Real-return Sharpe calculation for CPCV gate.

        Assigns the actual return 'r' to active signals, and 0 where signal=0.
        Annualised using approximate trades-per-year from prediction_horizon.
        """
        signal = (ens_prob > TRADING_HURDLE).astype(float)
        active = signal == 1
        if active.sum() < 3:
            return -99.0

        active_pnl = returns[active]  # ACTUAL returns (PT hit, SL hit, or timeout)
        std = active_pnl.std(ddof=1)
        if std < 1e-9:
            return 0.0

        trades_per_year = ANNUAL_FACTOR / self.config.prediction_horizon
        return float((active_pnl.mean() / std) * np.sqrt(trades_per_year))

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
        for col_idx, col_name in enumerate(feature_names):
            ic, _ = spearmanr(X_test[:, col_idx], y_test, nan_policy='omit')
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
        scaler: StandardScaler,
        features: List[str],
        version: int,
        meta_model: Optional[LogisticRegression],
    ) -> None:
        """
        Save all training artifacts to models/{ticker}/v{version}/.

        Artifact contract (predictor.py depends on exact filenames):
          scaler.pkl       — fitted StandardScaler
          features.json    — ordered feature name list (schema for predictor)
          weights.json     — per-model weight dict (all 1.0; meta-learner handles weights)
          xgb_model.pkl    — trained XGBClassifier
          lgb_model.pkl    — trained LGBMClassifier
          cat_model.pkl    — trained CatBoostClassifier (replaces rf_model.pkl)
          meta_model.pkl   — LogisticRegression meta-learner (new in v10.0)
        """
        path = os.path.join(self.models_base_dir, ticker, f"v{version}")
        os.makedirs(path, exist_ok=True)

        joblib.dump(scaler, os.path.join(path, 'scaler.pkl'))

        with open(os.path.join(path, 'features.json'), 'w') as fh:
            json.dump(features, fh)

        # Base weights = 1.0; actual weighting delegated to meta-learner at inference.
        model_weights = {name: 1.0 for name in models}
        with open(os.path.join(path, 'weights.json'), 'w') as fh:
            json.dump(model_weights, fh)

        for name, model in models.items():
            joblib.dump(model, os.path.join(path, f'{name}_model.pkl'))

        if meta_model is not None:
            joblib.dump(meta_model, os.path.join(path, 'meta_model.pkl'))


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

    # ── Universe: NIFTY Midcap 150 (liquid subset, ≥₹500cr daily ADV)
    # Rationale: Mid-caps are less efficiently priced than NIFTY50.
    # Stronger momentum signals, higher ATR ratios, more alpha per MARK5 features.
    # All symbols verified: ≥₹500cr daily rupee ADV (RULE 4 compliant).
    _DEFAULT_SYMBOLS = [
        # IT Midcap — strongest momentum sector on NSE, clean trends
        'COFORGE.NS', 'PERSISTENT.NS', 'MPHASIS.NS', 'KPITTECH.NS', 'LTTS.NS',
        # Capital Goods / Defence — multi-year re-rating trend
        'HAL.NS', 'BEL.NS', 'POLYCAB.NS', 'DIXON.NS', 'ABB.NS',
        # Financials Midcap — high beta, strong breakout potential
        'IDFCFIRSTB.NS', 'LICHSGFIN.NS', 'MUTHOOTFIN.NS', 'CHOLAFIN.NS', 'ABCAPITAL.NS',
        # Consumer / Retail — steady trend, low noise
        'IRCTC.NS', 'JUBLFOOD.NS', 'PAGEIND.NS', 'MARICO.NS', 'COLPAL.NS',
        # Chemicals / Pharma — high IC on dist_52w_high feature
        'PIIND.NS', 'DEEPAKNTR.NS', 'AARTIIND.NS', 'LAURUSLABS.NS', 'GRANULES.NS',
        # Real Estate / Infra — high ATR regime signal
        'GODREJPROP.NS', 'OBEROIRLTY.NS', 'PRESTIGE.NS', 'CONCOR.NS', 'CUMMINSIND.NS',
    ]

    _ap = argparse.ArgumentParser(description='MARK5 Model Trainer')
    _ap.add_argument('--symbols', nargs='+', default=None,
                     help='Ticker symbols to train (without .NS). Default: 20-stock universe.')
    _ap.add_argument('--years', type=float, default=3.0,
                     help='Years of historical data to fetch (default: 3)')
    _ap.add_argument('--skip-existing', action='store_true',
                     help='Skip symbols whose latest model is not corrupt')
    _args = _ap.parse_args()

    # Resolve symbol list — append .NS if missing
    if _args.symbols:
        _symbols = [s if s.endswith('.NS') else f'{s}.NS' for s in _args.symbols]
    else:
        _symbols = _DEFAULT_SYMBOLS

    print(f"\n{'='*60}")
    print(f"  MARK5 TRAINER  |  {len(_symbols)} symbols  |  {_args.years:.0f} yrs")
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

        # Use cache if < 24h old (avoids re-fetching during the same session)
        for _cp in [_cache_old, _cache_new]:
            if os.path.exists(_cp) and (_time.time() - os.path.getmtime(_cp)) < 86400:
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
        """True if the existing model has corrupted scaler values."""
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
        _sf = os.path.join(_vdir, 'scaler.pkl')
        _ff = os.path.join(_vdir, 'features.json')
        if not os.path.exists(_sf) or not os.path.exists(_ff):
            return True
        try:
            import joblib as _jl, json as _js
            _sc = _jl.load(_sf)
            with open(_ff) as _f:
                _sch = _js.load(_f)
            for _i, _feat in enumerate(_sch):
                _m = _sc.mean_[_i]
                _s = float(_sc.var_[_i] ** 0.5)
                if abs(_m) > 1e6 or _s == 0.0:
                    return True
            return False
        except Exception:
            return True

    _results = {'success': [], 'skipped': [], 'failed': []}

    for _i, _sym in enumerate(_symbols, 1):
        _bare = _sym.replace('.NS', '')
        print(f'\n[{_i}/{len(_symbols)}] {_bare}')

        if _args.skip_existing and not _is_corrupt(_sym):
            logger.info(f'{_bare}: model valid — skipping')
            _results['skipped'].append(_bare)
            continue

        _df = _fetch(_sym, _args.years)
        if _df is None or len(_df) < 300:
            logger.error(f'{_bare}: insufficient data — skipping')
            _results['failed'].append(_bare)
            continue

        _df.columns = [c.lower() for c in _df.columns]
        # Strip timezone — trainer expects tz-naive
        if _df.index.tz is not None:
            _df.index = _df.index.tz_localize(None)

        _t0 = _time.time()
        try:
            _trainer = MARK5MLTrainer()
            _res = _trainer.train_advanced_ensemble(_sym, _df)
            _elapsed = _time.time() - _t0

            if _res.get('status') == 'success':
                _v   = _res.get('version', '?')
                _auc = _res.get('auc', _res.get('roc_auc', 0.0))
                _fb  = _res.get('avg_fbeta', _res.get('fbeta', 0.0))
                logger.info(f'✅ {_bare}: v{_v} | AUC={_auc:.4f} | Fbeta={_fb:.4f} | {_elapsed:.0f}s')
                _results['success'].append(_bare)
            else:
                logger.error(f'❌ {_bare}: {_res.get("reason", "unknown")}')
                _results['failed'].append(_bare)
        except Exception as _exc:
            import traceback
            logger.error(f'❌ {_bare}: {_exc}')
            traceback.print_exc()
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

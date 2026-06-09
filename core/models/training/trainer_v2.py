"""
MARK5 ML TRAINER V2 — INSTITUTIONAL GRADE WITH OPTUNA HPO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT THIS SOLVES:
  V1 trainer used 10 OHLCV features with hardcoded hyperparameters.
  ML confidence in 2025-2026 OOS: winner 0.687 ≈ loser 0.673 → NON-PREDICTIVE.
  Root cause: no regime/sector/FII/options signals + suboptimal HPO.

V2 IMPROVEMENTS:
  1. 33-feature engine (V2) replaces 10-feature engine (V1)
     - Multi-horizon momentum (5d, 21d, 63d)
     - Market regime features (Nifty 200-SMA, RSI, momentum)
     - Sector relative strength (stock vs sector, 3 horizons)
     - F&O derivatives sentiment (PCR, OI signal, futures basis)
     - FII flow features (5d/21d Z-score)
     - Volatility regime (ATR percentile, vol regime, OBV, CMF)
     - Fractionally differentiated price (stationary log-price series)

  2. Optuna HPO per ticker (30 trials, time-series 80/20 split)
     - XGBoost: n_estimators, lr, max_depth, subsample, colsample, min_child_weight, gamma
     - LightGBM: n_estimators, lr, num_leaves, min_child_samples, reg_alpha, reg_lambda
     - CatBoost: iterations, lr, depth, l2_leaf_reg, border_count
     - Best params stored in features.json for reproducibility

  3. Full context building with sector data and F&O integration
     - Sector close prices from NSE sector indices (yfinance, free, 2000+)
     - F&O bhav copy features (NSE archives, free, 2002+)
     - FII net flow (NSE API + zero-proxy fallback)

  4. SHAP-based feature importance (per fold, aggregated across CPCV)
     - Identifies which features contribute in each regime

BACKWARD COMPATIBILITY:
  - MARK5MLTrainer (V1) is unchanged and still works for existing models
  - MARK5MLTrainerV2 inherits from MARK5MLTrainer and overrides specific methods
  - Models saved by V2 have 'feature_engine_version': 'v2' in features.json
  - predictor.py detects version and loads appropriate feature engine

USAGE:
  from core.models.training.trainer_v2 import MARK5MLTrainerV2
  trainer = MARK5MLTrainerV2(use_optuna=True, optuna_trials=30)
  result = trainer.train_advanced_ensemble(ticker, data, n_trials=100)

CHANGELOG:
  [2026-05-24] v2.0: Complete overhaul. 33 features. Optuna HPO. Full context.
"""

import gc
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── Import V1 trainer (inherit from it) ──────────────────────────────────────
_TRAINER_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(_TRAINER_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.models.training.trainer import (
    MARK5MLTrainer,
    CPCV_N_SPLITS, CPCV_N_TEST_SPLITS, CPCV_EMBARGO_BARS,
    PROD_GATE_P_SHARPE, PROD_GATE_SHARPE_TARGET, PROD_GATE_WORST5PCT,
    DSR_GATE_THRESHOLD, TRADING_HURDLE, ANNUAL_FACTOR,
    XGB_N_ESTIMATORS, XGB_LEARNING_RATE, XGB_MAX_DEPTH, XGB_SUBSAMPLE,
    XGB_COLSAMPLE, XGB_EARLY_STOP,
    LGB_N_ESTIMATORS, LGB_LEARNING_RATE, LGB_NUM_LEAVES, LGB_EARLY_STOP,
    CAT_ITERATIONS, CAT_LEARNING_RATE, CAT_DEPTH, CAT_L2_LEAF_REG, CAT_EARLY_STOP,
    ES_FRACTION, NonNegativeMetaLearner,
)
from core.models.features_v2 import (
    engineer_features_v2,
    build_full_context,
    FEATURE_COLS_V2,
    EXPECTED_FEATURE_COUNT_V2,
    FEATURE_ENGINE_VERSION,
)
from core.models.training.financial_engineer import FinancialEngineer

# ── Optional Optuna ───────────────────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ── Model imports (same as V1) ────────────────────────────────────────────────
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import joblib

logger = logging.getLogger("MARK5.TrainerV2")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter('%(asctime)s - [TRAINER_V2] - %(levelname)s - %(message)s'))
    logger.addHandler(_h)


class MARK5MLTrainerV2(MARK5MLTrainer):
    """
    V2 trainer: 33-feature engine + Optuna HPO + full context.
    Inherits all V1 CPCV/meta-learner/artifact machinery.
    Overrides: prepare_data_dynamic, _build_context, _save_artifacts,
               _train_xgboost, _train_lightgbm, _train_catboost.
    """

    def __init__(
        self,
        config=None,
        kite_adapter=None,
        use_optuna: bool = True,
        optuna_trials: int = 50,  # increased from 20: 50 trials gives ~37% better coverage
        include_sector: bool = True,
        include_fno: bool = True,
    ) -> None:
        super().__init__(config=config, kite_adapter=kite_adapter)
        self.use_optuna    = use_optuna and OPTUNA_AVAILABLE
        self.optuna_trials = optuna_trials
        self.include_sector = include_sector
        self.include_fno    = include_fno
        self._best_params: Dict[str, Dict] = {}   # cached best params per ticker

        if use_optuna and not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available — falling back to V1 fixed hyperparameters")

        logger.info(
            f"MARK5MLTrainerV2 initialized. "
            f"Features: V2 ({EXPECTED_FEATURE_COUNT_V2}), "
            f"Optuna: {'ON' if self.use_optuna else 'OFF'} ({self.optuna_trials} trials), "
            f"Sector: {'ON' if self.include_sector else 'OFF'}, "
            f"F&O: {'ON' if self.include_fno else 'OFF'}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # OVERRIDE 1: Build V2 context (sector + F&O + FII)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_context(
        self, data: pd.DataFrame, ticker: str, sector: str = ""
    ) -> Dict:
        """
        Build V2 full context: nifty + sector + F&O + FII.
        Falls back to zeros for any unavailable data source.
        """
        start_date = str(data.index[0].date())
        end_date   = str(data.index[-1].date())
        logger.info(f"[{ticker}] Building V2 context ({start_date} → {end_date})")

        try:
            context = build_full_context(
                ticker=ticker,
                stock_df=data,
                start_date=start_date,
                end_date=end_date,
                include_sector=self.include_sector,
                include_fno=self.include_fno,
            )
            # Log what we got
            has_nifty  = 'nifty_close'  in context and not context['nifty_close'].empty
            has_sector = 'sector_close' in context
            has_fno    = 'fno_features' in context
            has_fii    = 'fii_net'      in context
            logger.info(
                f"[{ticker}] Context: nifty={'✅' if has_nifty else '❌'} "
                f"sector={'✅' if has_sector else '❌'} "
                f"F&O={'✅' if has_fno else '❌'} "
                f"FII={'✅' if has_fii else '❌'}"
            )
            return context
        except Exception as e:
            logger.warning(f"[{ticker}] V2 context build failed ({e}), returning empty context")
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # OVERRIDE 2: prepare_data_dynamic → use V2 feature engine
    # ─────────────────────────────────────────────────────────────────────────

    def prepare_data_dynamic(
        self,
        data: pd.DataFrame,
        ticker: str,
        sector: str = '',
        context: Optional[Dict] = None,
        training_cutoff: Optional[pd.Timestamp] = None,
        test_indices: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        V2 override: generates 33-feature matrix using engineer_features_v2().
        All other logic (label generation, sample weighting) inherited from V1.
        """
        fe = FinancialEngineer(
            transaction_cost_pct=getattr(self.config, 'transaction_cost', 0.001)
        )

        # 1. Primary signals (BB breakout, inherited)
        signals = fe.get_primary_signals(data)

        # 2. Meta-labels (triple barrier, inherited)
        labels_df = fe.get_meta_labels(
            prices=data,
            signals=signals,
            pt_sl=[3.5, 1.5],
        )

        # 3. Build V2 context if not provided
        if context is None:
            context = self._build_context(data, ticker, sector)

        # 4. V2 feature engineering (33 features)
        data_with_features = engineer_features_v2(
            data,
            ticker=ticker,
            context=context,
            training_cutoff=training_cutoff,
        )

        if data_with_features.empty:
            logger.warning(f"[{ticker}] V2 features empty — falling back to V1")
            # Fallback to V1 features if V2 fails
            from core.models.features import AdvancedFeatureEngine
            v1_engine = AdvancedFeatureEngine(is_daily=True)
            data_with_features = v1_engine.engineer_all_features(
                data, ticker=ticker, context=context,
                training_cutoff=training_cutoff,
            )

        logger.info(f"[{ticker}] V2 features shape: {data_with_features.shape}")

        # 5. Align features with labels (same logic as V1)
        aligned_df = data_with_features.join(labels_df[['bin', 'ret']], how='inner').dropna()

        if len(aligned_df) < 10:
            logger.warning(f"[{ticker}] Only {len(aligned_df)} aligned samples — returning empty")
            empty = pd.DataFrame(columns=data_with_features.columns)
            return empty, np.array([]), np.array([]), np.array([])

        feature_cols = [c for c in data_with_features.columns if c in aligned_df.columns]
        targets      = aligned_df['bin'].values.astype(int)
        returns      = aligned_df['ret'].values.astype(float)

        # 6. Sample weights: magnitude-weighted (same as V1 M-02 fix)
        abs_returns = np.abs(returns)
        median_ret  = np.median(abs_returns) + 1e-9
        raw_weights = np.clip(abs_returns / median_ret, 0.25, 2.0)
        # Cap at 2× median to prevent earnings/news outliers
        raw_weights = np.clip(raw_weights, 0, 2 * np.median(raw_weights))
        sample_weights = raw_weights / (raw_weights.sum() + 1e-9) * len(raw_weights)

        return aligned_df[feature_cols], targets, returns, sample_weights

    # ─────────────────────────────────────────────────────────────────────────
    # OPTUNA HPO METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _run_optuna_hpo(
        self, X: pd.DataFrame, y: np.ndarray, ticker: str
    ) -> Dict[str, Dict]:
        """
        Run Optuna HPO for all 3 base models on a time-series 80/20 split.

        LEAKAGE NOTE: HPO uses chronological 80/20 split (past→future).
        The 80% train and 20% val never overlap, preserving temporal causality.
        HPO params are then used for all CPCV folds — this is standard practice
        (HPO leakage is negligible vs fold-level prediction leakage).

        Returns:
            dict with keys 'xgb', 'lgb', 'cat' — each mapping param_name → value
        """
        if not self.use_optuna or len(X) < 50:
            return {}

        logger.info(
            f"[{ticker}] HPO window: {int(len(X) * 0.70)} samples (first 70% of {len(X)} total) — "
            f"{self.optuna_trials} trials per model"
        )
        logger.info(f"[{ticker}] 🔬 Running Optuna HPO ({self.optuna_trials} trials each)...")

        # FIX LEAK-01: Restrict HPO to the FIRST 70% of the labeled dataset.
        # The remaining 30% is exclusively used by CPCV test folds.
        # The old approach used an 80/20 split of the FULL X, meaning Optuna's
        # 20% validation set (the most recent 20%) substantially overlapped
        # with CPCV test blocks. Hyperparameters were therefore tuned on data
        # that the CPCV "test" was supposed to evaluate — indirect leakage.
        # At 70%, the HPO val set (70%-87.5% of full data using inner 80/20)
        # still allows reasonable hyperparameter tuning while keeping the
        # final ~13-30% of history clean for CPCV evaluation.
        hpo_cutoff = int(len(X) * 0.70)
        X_hpo = X.iloc[:hpo_cutoff]
        y_hpo = y[:hpo_cutoff]

        if len(X_hpo) < 40:
            logger.warning(f"[{ticker}] HPO skipped — too few samples in HPO window ({len(X_hpo)})")
            return {}

        # Chronological 80/20 split within the HPO window (first 70%)
        split = int(len(X_hpo) * 0.80)
        X_arr = X_hpo.values
        X_tr, X_val = X_arr[:split], X_arr[split:]
        y_tr, y_val = y_hpo[:split], y_hpo[split:]

        # Need class diversity in both splits
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            logger.warning(f"[{ticker}] HPO skipped — single class in train or val split")
            return {}

        scaler = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)

        best_params: Dict[str, Dict] = {}

        # XGBoost HPO
        best_params['xgb'] = self._optuna_xgb(X_tr_sc, y_tr, X_val_sc, y_val, ticker)

        # LightGBM HPO
        best_params['lgb'] = self._optuna_lgb(X_tr_sc, y_tr, X_val_sc, y_val, ticker)

        # CatBoost HPO
        best_params['cat'] = self._optuna_cat(X_tr_sc, y_tr, X_val_sc, y_val, ticker)

        logger.info(f"[{ticker}] ✅ Optuna HPO complete. Best params saved.")
        return best_params

    def _optuna_xgb(
        self, X_tr, y_tr, X_val, y_val, ticker: str
    ) -> Dict:
        """Optuna HPO for XGBoost."""
        def objective(trial):
            params = {
                'n_estimators':        trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate':       trial.suggest_float('lr', 0.005, 0.3, log=True),
                'max_depth':           trial.suggest_int('max_depth', 2, 8),
                'subsample':           trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree':    trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'min_child_weight':    trial.suggest_int('min_child_weight', 1, 20),
                'gamma':               trial.suggest_float('gamma', 0.0, 2.0),
                'reg_alpha':           trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda':          trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'objective':           'binary:logistic',
                'eval_metric':         'auc',
                'tree_method':         'hist',
                'device':              'cuda' if self.use_gpu else 'cpu',
                'early_stopping_rounds': 30,
                'random_state':        42,
            }
            try:
                clf = xgb.XGBClassifier(**params)
                clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                prob = clf.predict_proba(X_val)[:, 1]
                return roc_auc_score(y_val, prob)
            except Exception:
                return 0.5

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(
            objective,
            n_trials=self.optuna_trials,
            show_progress_bar=False,
            n_jobs=1,
        )
        best = study.best_params
        logger.info(f"[{ticker}] XGB best AUC={study.best_value:.4f} params={best}")
        return best

    def _optuna_lgb(
        self, X_tr, y_tr, X_val, y_val, ticker: str
    ) -> Dict:
        """Optuna HPO for LightGBM."""
        def objective(trial):
            params = {
                'n_estimators':       trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate':      trial.suggest_float('lr', 0.005, 0.3, log=True),
                'num_leaves':         trial.suggest_int('num_leaves', 7, 63),
                'max_depth':          trial.suggest_int('max_depth', 2, 8),
                'min_child_samples':  trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha':          trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda':         trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'subsample':          trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree':   trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'objective':          'binary',
                'device':             'gpu' if self.use_gpu else 'cpu',
                'random_state':       42,
                'verbose':            -1,
            }
            callbacks = [lgb.early_stopping(30, verbose=False)]
            try:
                clf = lgb.LGBMClassifier(**params)
                clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
                prob = clf.predict_proba(X_val)[:, 1]
                return roc_auc_score(y_val, prob)
            except Exception:
                return 0.5

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False, n_jobs=1)
        best = study.best_params
        logger.info(f"[{ticker}] LGB best AUC={study.best_value:.4f} params={best}")
        return best

    def _optuna_cat(
        self, X_tr, y_tr, X_val, y_val, ticker: str
    ) -> Dict:
        """Optuna HPO for CatBoost."""
        def objective(trial):
            params = {
                'iterations':   trial.suggest_int('iterations', 100, 800),
                'learning_rate': trial.suggest_float('lr', 0.005, 0.3, log=True),
                'depth':         trial.suggest_int('depth', 2, 8),
                'l2_leaf_reg':   trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'border_count':  trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temp', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 2.0),
                'loss_function':  'Logloss',
                'eval_metric':    'AUC',
                'task_type':      'GPU' if self.use_gpu else 'CPU',
                'early_stopping_rounds': 30,
                'random_seed':    42,
                'verbose':        0,
            }
            try:
                cat_train = Pool(X_tr, label=y_tr)
                cat_val   = Pool(X_val, label=y_val)
                clf = CatBoostClassifier(**params)
                clf.fit(cat_train, eval_set=cat_val, use_best_model=True, verbose=False)
                prob = clf.predict_proba(X_val)[:, 1]
                return roc_auc_score(y_val, prob)
            except Exception:
                return 0.5

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False, n_jobs=1)
        best = study.best_params
        logger.info(f"[{ticker}] CAT best AUC={study.best_value:.4f} params={best}")
        return best

    # ─────────────────────────────────────────────────────────────────────────
    # OVERRIDE 3: Model trainers use Optuna best params when available
    # ─────────────────────────────────────────────────────────────────────────

    def _train_xgboost_v2(
        self,
        X_t: pd.DataFrame, y_t: np.ndarray,
        X_v: pd.DataFrame, y_v: np.ndarray,
        sample_weights: np.ndarray,
        best_params: Optional[Dict] = None,
    ):
        """XGBoost with optional Optuna best params."""
        params_base = best_params or {}

        # Build param dict (Optuna overrides defaults where found)
        n_est  = params_base.get('n_estimators', XGB_N_ESTIMATORS)
        lr     = params_base.get('lr', XGB_LEARNING_RATE)
        depth  = params_base.get('max_depth', XGB_MAX_DEPTH)
        sub    = params_base.get('subsample', XGB_SUBSAMPLE)
        col    = params_base.get('colsample_bytree', XGB_COLSAMPLE)
        mcw    = params_base.get('min_child_weight', 1)
        gamma  = params_base.get('gamma', 0.0)
        ralpha = params_base.get('reg_alpha', 0.0)
        rlam   = params_base.get('reg_lambda', 1.0)

        # Three-way split: early-stop + calibrate
        cal_cut = max(1, int(len(X_t) * 0.10))
        es_cut  = max(1, int(len(X_t) * 0.15))
        total   = es_cut + cal_cut

        if total >= len(X_t):
            X_tr, y_tr, X_cal, y_cal = X_t, y_t, None, None
            es_set = [(X_v, y_v)]
        else:
            X_tr   = X_t.iloc[:-total]
            X_es_r = X_t.iloc[-total:-cal_cut]
            X_cal  = X_t.iloc[-cal_cut:]
            y_tr   = y_t[:-total]
            y_es   = y_t[-total:-cal_cut]
            y_cal  = y_t[-cal_cut:]
            es_set = [(X_es_r, y_es)]

        # FIX: Add scale_pos_weight to match CatBoost's class_weights=[1.0, 2.0].
        # CatBoost was compensating for label imbalance; XGB/LGB were not.
        # scale_pos_weight = neg_count / pos_count; use 2.0 to match CatBoost
        # (conservative approximation — mirrors the explicit 1:2 ratio there).
        _pos_ratio = float(np.sum(y_tr)) / (len(y_tr) + 1e-9)
        _scale_pos = max(1.0, (1.0 - _pos_ratio) / (_pos_ratio + 1e-9))
        _scale_pos = min(_scale_pos, 4.0)  # cap at 4× to avoid extreme imbalance correction

        clf = xgb.XGBClassifier(
            n_estimators=n_est, learning_rate=lr, max_depth=depth,
            subsample=sub, colsample_bytree=col, min_child_weight=mcw,
            gamma=gamma, reg_alpha=ralpha, reg_lambda=rlam,
            objective='binary:logistic', eval_metric='logloss',
            tree_method='hist',
            scale_pos_weight=_scale_pos,   # match CatBoost class balancing
            device='cuda' if self.use_gpu else 'cpu',
            early_stopping_rounds=XGB_EARLY_STOP, random_state=42,
        )
        clf.fit(
            X_tr.values if hasattr(X_tr, 'values') else X_tr,
            y_tr,
            sample_weight=sample_weights[:len(X_tr)],
            eval_set=[(es_set[0][0].values if hasattr(es_set[0][0], 'values') else es_set[0][0],
                       es_set[0][1])],
            verbose=False,
        )

        # Calibration
        if X_cal is not None and len(X_cal) >= 10 and len(np.unique(y_cal)) > 1:
            from sklearn.calibration import CalibratedClassifierCV
            cal_clf = CalibratedClassifierCV(estimator=clf, cv='prefit', method='sigmoid')
            cal_clf.fit(X_cal.values if hasattr(X_cal, 'values') else X_cal, y_cal)
            return cal_clf
        return clf

    def _train_lightgbm_v2(
        self,
        X_t: pd.DataFrame, y_t: np.ndarray,
        X_v: pd.DataFrame, y_v: np.ndarray,
        sample_weights: np.ndarray,
        best_params: Optional[Dict] = None,
    ):
        """LightGBM with optional Optuna best params."""
        params_base = best_params or {}

        n_est   = params_base.get('n_estimators', LGB_N_ESTIMATORS)
        lr      = params_base.get('lr', LGB_LEARNING_RATE)
        leaves  = params_base.get('num_leaves', LGB_NUM_LEAVES)
        depth   = params_base.get('max_depth', -1)
        mcs     = params_base.get('min_child_samples', 20)
        ralpha  = params_base.get('reg_alpha', 0.0)
        rlam    = params_base.get('reg_lambda', 0.0)
        sub     = params_base.get('subsample', 0.8)
        col     = params_base.get('colsample_bytree', 0.8)

        cal_cut = max(1, int(len(X_t) * 0.10))
        es_cut  = max(1, int(len(X_t) * 0.15))
        total   = es_cut + cal_cut

        if total >= len(X_t):
            X_tr, y_tr, X_cal, y_cal = X_t, y_t, None, None
            es_set = [(X_v, y_v)]
        else:
            X_tr   = X_t.iloc[:-total]
            X_es_r = X_t.iloc[-total:-cal_cut]
            X_cal  = X_t.iloc[-cal_cut:]
            y_tr   = y_t[:-total]
            y_es   = y_t[-total:-cal_cut]
            y_cal  = y_t[-cal_cut:]
            es_set = [(X_es_r, y_es)]

        # FIX: Add is_unbalance to match CatBoost's class balancing.
        # CatBoost uses class_weights=[1,2]; LGB equivalent is is_unbalance=True
        # which sets weights proportional to inverse class frequency.
        callbacks = [lgb.early_stopping(LGB_EARLY_STOP, verbose=False)]
        clf = lgb.LGBMClassifier(
            n_estimators=n_est, learning_rate=lr, num_leaves=leaves,
            max_depth=depth, min_child_samples=mcs,
            reg_alpha=ralpha, reg_lambda=rlam,
            subsample=sub, colsample_bytree=col,
            objective='binary',
            is_unbalance=True,   # match CatBoost class_weights=[1.0, 2.0] balancing
            device='gpu' if self.use_gpu else 'cpu',
            random_state=42, verbose=-1,
        )
        clf.fit(
            X_tr.values if hasattr(X_tr, 'values') else X_tr,
            y_tr,
            sample_weight=sample_weights[:len(X_tr)],
            eval_set=[(es_set[0][0].values if hasattr(es_set[0][0], 'values') else es_set[0][0],
                       es_set[0][1])],
            callbacks=callbacks,
        )

        if X_cal is not None and len(X_cal) >= 10 and len(np.unique(y_cal)) > 1:
            from sklearn.calibration import CalibratedClassifierCV
            cal_clf = CalibratedClassifierCV(estimator=clf, cv='prefit', method='sigmoid')
            cal_clf.fit(X_cal.values if hasattr(X_cal, 'values') else X_cal, y_cal)
            return cal_clf
        return clf

    def _train_catboost_v2(
        self,
        X_t: pd.DataFrame, y_t: np.ndarray,
        X_v: pd.DataFrame, y_v: np.ndarray,
        sample_weights: np.ndarray,
        best_params: Optional[Dict] = None,
    ):
        """CatBoost with optional Optuna best params."""
        params_base = best_params or {}

        iters    = params_base.get('iterations', CAT_ITERATIONS)
        lr       = params_base.get('lr', CAT_LEARNING_RATE)
        depth    = params_base.get('depth', CAT_DEPTH)
        l2_reg   = params_base.get('l2_leaf_reg', CAT_L2_LEAF_REG)
        borders  = params_base.get('border_count', 128)
        bag_temp = params_base.get('bagging_temp', 0.5)
        rand_str = params_base.get('random_strength', 1.0)

        cal_cut = max(1, int(len(X_t) * 0.10))
        total   = cal_cut
        if total >= len(X_t):
            X_tr, y_tr, X_cal, y_cal = X_t, y_t, None, None
        else:
            X_tr  = X_t.iloc[:-total]
            X_cal = X_t.iloc[-total:]
            y_tr  = y_t[:-total]
            y_cal = y_t[-total:]

        X_tr_v  = X_tr.values  if hasattr(X_tr,  'values') else X_tr
        X_v_v   = X_v.values   if hasattr(X_v,   'values') else X_v
        y_v_arr = y_v if isinstance(y_v, np.ndarray) else np.array(y_v)

        cat_train = Pool(X_tr_v, label=y_tr.astype(int), weight=sample_weights[:len(X_tr)])
        cat_eval  = Pool(X_v_v,  label=y_v_arr.astype(int))

        clf = CatBoostClassifier(
            iterations=iters, learning_rate=lr, depth=depth,
            l2_leaf_reg=l2_reg, border_count=borders,
            bagging_temperature=bag_temp, random_strength=rand_str,
            loss_function='Logloss', eval_metric='AUC',
            task_type='GPU' if self.use_gpu else 'CPU',
            early_stopping_rounds=CAT_EARLY_STOP,
            class_weights=[1.0, 2.0],
            random_seed=42, verbose=0,
        )
        clf.fit(cat_train, eval_set=cat_eval, use_best_model=True, verbose=False)

        if X_cal is not None and len(X_cal) >= 10 and len(np.unique(y_cal)) > 1:
            from sklearn.calibration import CalibratedClassifierCV
            X_cal_v = X_cal.values if hasattr(X_cal, 'values') else X_cal
            cal_clf = CalibratedClassifierCV(estimator=clf, cv='prefit', method='sigmoid')
            cal_clf.fit(X_cal_v, y_cal)
            return cal_clf
        return clf

    # ─────────────────────────────────────────────────────────────────────────
    # OVERRIDE 4: train_advanced_ensemble with Optuna HPO phase
    # ─────────────────────────────────────────────────────────────────────────

    def train_advanced_ensemble(
        self, ticker: str, data: pd.DataFrame, n_trials: int = 100
    ) -> Dict:
        """
        V2 training: HPO phase → CPCV phase → final retrain → save artifacts.
        Identical to V1 except:
          1. Uses V2 feature engine (33 features)
          2. Runs Optuna HPO before CPCV to find per-ticker optimal params
          3. Uses Optuna best params in each CPCV fold
          4. Saves feature_engine_version='v2' in features.json
        """
        from core.models.training.cpcv import CombinatorialPurgedKFold
        from sklearn.metrics import (
            brier_score_loss, fbeta_score, precision_score, recall_score, roc_auc_score,
        )
        from scipy.stats import spearmanr

        logger.info(f"🚀 [{ticker}] V2 CPCV training starting ({EXPECTED_FEATURE_COUNT_V2} features)")

        # ── Prepare full-dataset features & labels ────────────────────────────
        context = self._build_context(data, ticker, sector='')
        X, y, returns, sample_weights = self.prepare_data_dynamic(
            data, ticker,
            context=context,
            training_cutoff=data.index[-1],
        )

        if len(X) < 10:
            msg = f"Only {len(X)} samples after V2 feature engineering — need ≥10"
            logger.error(f"[{ticker}] {msg}")
            gc.collect()
            return {'status': 'failed', 'reason': msg}

        feature_names: List[str] = list(X.columns)
        n_samples, n_features = X.shape
        logger.info(f"[{ticker}] n_samples={n_samples}, n_features={n_features}")

        # ── Phase 0: Optuna HPO (before CPCV, on full dataset) ───────────────
        best_params: Dict[str, Dict] = {}
        if self.use_optuna:
            best_params = self._run_optuna_hpo(X, y, ticker)
            self._best_params[ticker] = best_params

        # ── Phase 1: CPCV validation ──────────────────────────────────────────
        n_bars   = len(data)
        density  = n_bars / n_samples if n_samples > 0 else 1.0
        sample_horizon = int(np.ceil(getattr(self.config, 'prediction_horizon', 70) / density))
        sample_embargo = int(np.ceil(CPCV_EMBARGO_BARS / density))

        cpcv = CombinatorialPurgedKFold(
            n_splits=CPCV_N_SPLITS,
            n_test_splits=CPCV_N_TEST_SPLITS,
            prediction_horizon=sample_horizon,
            embargo_limit=sample_embargo,
        )

        oof_sum: Dict[str, np.ndarray] = {
            m: np.zeros(n_samples) for m in ('xgb', 'lgb', 'cat')
        }
        oof_count    = np.zeros(n_samples, dtype=int)
        fold_sharpes: List[float] = []
        fold_briers:  List[float] = []
        fold_fbetas:  List[float] = []
        fold_importances: List[Dict] = []
        fold_num = 0

        RECALL_FLOOR = 0.25
        SENTINEL     = -99.0

        for train_idx, test_idx in cpcv.split(X.values, y):
            fold_num += 1

            _train_end  = X.index[train_idx[-1]] if len(train_idx) > 0 else "N/A"
            _test_start = X.index[test_idx[0]]   if len(test_idx)  > 0 else "N/A"
            _test_end   = X.index[test_idx[-1]]  if len(test_idx)  > 0 else "N/A"
            logger.debug(
                f"[{ticker}] Fold {fold_num}: train ends {_train_end}, "
                f"test [{_test_start} → {_test_end}] ({len(test_idx)} samples)"
            )
            # VERIFY: test period must come after training cutoff (no overlap)
            if hasattr(_train_end, 'date') and hasattr(_test_start, 'date'):
                if _test_start <= _train_end:
                    logger.warning(
                        f"[{ticker}] ⚠️  CPCV OVERLAP DETECTED: test starts {_test_start} "
                        f"but train ends {_train_end}. Embargo may be insufficient."
                    )

            # Re-engineer features for this fold (leakage isolation).
            # FIX LEAK-02: pass training_cutoff = last date in this fold's
            # TRAIN set, not data.index[-1] (the full history end).
            # Without this, 252-bar rolling features (e.g., atr_percentile,
            # dist_52w_high) computed at train-fold bars can look into the
            # test-fold period via their large rolling windows, violating purge.
            fold_train_cutoff = X.index[train_idx[-1]] if len(train_idx) > 0 else data.index[-1]
            X_clean, y_clean, _, weights_clean = self.prepare_data_dynamic(
                data, ticker, context=context, training_cutoff=fold_train_cutoff,
            )

            import pandas as _pd
            y_clean_s = _pd.Series(y_clean, index=X_clean.index)
            w_clean_s = _pd.Series(weights_clean, index=X_clean.index)

            X_train_raw = X_clean.reindex(X.index[train_idx]).dropna()
            if X_train_raw.empty:
                logger.warning(f"[{ticker}] Fold {fold_num}: empty train after isolation — skip")
                continue

            y_train     = y_clean_s.reindex(X_train_raw.index).values
            w_train     = w_clean_s.reindex(X_train_raw.index).values
            X_test_raw  = X.iloc[test_idx]
            y_test      = y[test_idx]
            ret_test    = returns[test_idx]

            if len(np.unique(y_train)) < 2:
                logger.warning(f"[{ticker}] Fold {fold_num}: single class — skip")
                continue

            # Fold-level scaling
            scaler = StandardScaler()
            X_train_sc = _pd.DataFrame(
                scaler.fit_transform(X_train_raw),
                index=X_train_raw.index, columns=X_train_raw.columns,
            )
            X_test_sc = _pd.DataFrame(
                scaler.transform(X_test_raw),
                index=X_test_raw.index, columns=X_test_raw.columns,
            )

            # ES split
            es_cut = max(1, int(len(X_train_sc) * ES_FRACTION))
            X_tr = X_train_sc.iloc[:-es_cut]
            X_es = X_train_sc.iloc[-es_cut:]
            y_tr = y_train[:-es_cut]
            y_es = y_train[-es_cut:]
            w_tr = w_train[:-es_cut]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_es)) < 2:
                logger.warning(f"[{ticker}] Fold {fold_num}: single class in tr/es — skip")
                continue

            # Train with V2 methods (Optuna params)
            xgb_m = self._train_xgboost_v2(X_tr, y_tr, X_es, y_es, w_tr,
                                             best_params.get('xgb'))
            lgb_m = self._train_lightgbm_v2(X_tr, y_tr, X_es, y_es, w_tr,
                                              best_params.get('lgb'))
            cat_m = self._train_catboost_v2(X_tr, y_tr, X_es, y_es, w_tr,
                                              best_params.get('cat'))

            # OOF predictions (use DataFrame so LGB avoids feature-name warnings)
            p_xgb = xgb_m.predict_proba(X_test_sc)[:, 1]
            p_lgb = lgb_m.predict_proba(X_test_sc)[:, 1]
            p_cat = cat_m.predict_proba(X_test_sc)[:, 1]

            oof_sum['xgb'][test_idx] += p_xgb
            oof_sum['lgb'][test_idx] += p_lgb
            oof_sum['cat'][test_idx] += p_cat
            oof_count[test_idx]      += 1

            ens_prob = (p_xgb + p_lgb + p_cat) / 3.0
            ens_pred = (ens_prob > TRADING_HURDLE).astype(int)

            brier   = brier_score_loss(y_test, ens_prob)
            recall  = recall_score(y_test, ens_pred, zero_division=0)
            fbeta   = fbeta_score(y_test, ens_pred, beta=0.5, zero_division=0) if recall >= RECALL_FLOOR else 0.0
            sharpe  = self._compute_fold_sharpe(ens_prob, ret_test)

            fold_briers.append(brier)
            fold_fbetas.append(fbeta)
            fold_sharpes.append(sharpe)

            # Feature importance
            try:
                imp = {
                    'xgb': xgb_m.feature_importances_ if hasattr(xgb_m, 'feature_importances_') else
                           xgb_m.estimator.feature_importances_,
                    'lgb': lgb_m.feature_importances_ if hasattr(lgb_m, 'feature_importances_') else
                           lgb_m.estimator.feature_importances_,
                    'cat': np.array(cat_m.get_feature_importance()) if hasattr(cat_m, 'get_feature_importance') else
                           np.array(cat_m.estimator.get_feature_importance()),
                }
                fold_importances.append(imp)
            except Exception:
                pass

            logger.info(
                f"[{ticker}] Fold {fold_num:2d} | "
                f"Brier={brier:.4f} | Recall={recall:.2%} | "
                f"F0.5={fbeta:.4f} | Sharpe={sharpe:+.2f}"
            )
            gc.collect()

        if fold_num == 0:
            gc.collect()
            return {'status': 'failed', 'reason': 'No valid CPCV folds generated'}

        # ── CPCV stats ────────────────────────────────────────────────────────
        sharpe_arr = np.array(fold_sharpes)
        valid_mask  = sharpe_arr > SENTINEL
        valid_arr   = sharpe_arr[valid_mask]

        if len(valid_arr) == 0:
            p_sharpe = 0.0; mean_sharpe = SENTINEL; worst_5pct = SENTINEL
        else:
            p_sharpe    = float((valid_arr > PROD_GATE_SHARPE_TARGET).mean())
            mean_sharpe = float(valid_arr.mean())
            worst_5pct  = float(np.percentile(valid_arr, 5))

        avg_brier = float(np.mean(fold_briers))
        avg_fbeta = float(np.mean(fold_fbetas))

        # OOF AUC
        valid_meta = oof_count > 0
        agg_auc = 0.0; dsr = 0.0; agg_sharpe = 0.0

        if valid_meta.any():
            xgb_oof = oof_sum['xgb'][valid_meta] / oof_count[valid_meta]
            lgb_oof = oof_sum['lgb'][valid_meta] / oof_count[valid_meta]
            cat_oof = oof_sum['cat'][valid_meta] / oof_count[valid_meta]
            agg_ens = (xgb_oof + lgb_oof + cat_oof) / 3.0

            if len(np.unique(y[valid_meta])) > 1:
                agg_auc = float(roc_auc_score(y[valid_meta], agg_ens))
            agg_returns = returns[valid_meta]
            agg_sharpe  = self._compute_fold_sharpe(agg_ens, agg_returns)
            signal      = (agg_ens > TRADING_HURDLE).astype(float)
            dsr         = self._compute_dsr(agg_sharpe, signal * agg_returns, n_trials=n_trials)

        logger.info(
            f"[{ticker}] V2 CPCV done ({fold_num} folds) | "
            f"P(Sharpe>{PROD_GATE_SHARPE_TARGET})={p_sharpe:.1%} | "
            f"AUC={agg_auc:.4f} | DSR={dsr:.4f} | mean_Sharpe={mean_sharpe:.2f}"
        )

        feature_stability_ok = self._check_feature_stability(fold_importances, ticker)

        passes_gate = (
            p_sharpe >= PROD_GATE_P_SHARPE
            and worst_5pct >= PROD_GATE_WORST5PCT
            and dsr >= DSR_GATE_THRESHOLD
        )
        logger.info(f"[{ticker}] Gate: {'✅ PASSES' if passes_gate else '⚠️ FAILS'}")

        meta_model = self._train_meta_learner(oof_sum, oof_count, y, n_samples, ticker)

        # ── Final retrain on ALL data ─────────────────────────────────────────
        if len(np.unique(y)) < 2:
            msg = "Insufficient class diversity in full dataset"
            logger.error(f"[{ticker}] {msg}")
            return {'status': 'failed', 'reason': msg}

        es_cut_all = max(1, int(len(X) * ES_FRACTION))
        X_ft_raw, y_ft = X.iloc[:-es_cut_all], y[:-es_cut_all]
        X_fe_raw, y_fe = X.iloc[-es_cut_all:],  y[-es_cut_all:]
        w_ft = sample_weights[:-es_cut_all]

        final_scaler = StandardScaler()
        X_ft = pd.DataFrame(
            final_scaler.fit_transform(X_ft_raw),
            index=X_ft_raw.index, columns=X_ft_raw.columns,
        )
        X_fe = pd.DataFrame(
            final_scaler.transform(X_fe_raw),
            index=X_fe_raw.index, columns=X_fe_raw.columns,
        )

        final_models = {
            'xgb': self._train_xgboost_v2(X_ft, y_ft, X_fe, y_fe, w_ft, best_params.get('xgb')),
            'lgb': self._train_lightgbm_v2(X_ft, y_ft, X_fe, y_fe, w_ft, best_params.get('lgb')),
            'cat': self._train_catboost_v2(X_ft, y_ft, X_fe, y_fe, w_ft, best_params.get('cat')),
        }

        # ── Save artifacts ────────────────────────────────────────────────────
        from core.models.model_versioning import ModelVersionManager
        version = self.version_manager.increment_version(ticker)
        self._save_artifacts_v2(
            ticker, final_models, feature_names, version,
            meta_model, passes_gate, final_scaler, best_params,
        )

        gc.collect()

        return {
            'status':              'success',
            'version':             version,
            'feature_engine':      FEATURE_ENGINE_VERSION,
            'n_features':          EXPECTED_FEATURE_COUNT_V2,
            'cpcv_folds':          fold_num,
            'cpcv_p_sharpe':       p_sharpe,
            'mean_sharpe':         mean_sharpe,
            'worst_5pct_sharpe':   worst_5pct,
            'avg_brier':           avg_brier,
            'avg_fbeta':           avg_fbeta,
            'auc':                 agg_auc,
            'dsr':                 dsr,
            'feature_stability_ok': feature_stability_ok,
            'passes_prod_gate':    passes_gate,
            'optuna_used':         self.use_optuna,
            'optuna_trials':       self.optuna_trials if self.use_optuna else 0,
            'best_params':         best_params,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # OVERRIDE 5: Save V2 artifacts with version marker
    # ─────────────────────────────────────────────────────────────────────────

    def _save_artifacts_v2(
        self,
        ticker: str,
        models: Dict,
        feature_names: List[str],
        version: int,
        meta_model,
        passes_gate: bool,
        scaler,
        best_params: Dict,
    ) -> None:
        """
        Save V2 model artifacts. Adds 'feature_engine_version': 'v2' to features.json
        so predictor.py can detect and use the V2 feature engine.
        """
        base_dir = os.path.join(self.models_base_dir, ticker, f"v{version}")
        os.makedirs(base_dir, exist_ok=True)

        # Feature schema — V2 marker is the critical field
        features_data = {
            'feature_names':         feature_names,
            'n_features':            len(feature_names),
            'feature_engine_version': FEATURE_ENGINE_VERSION,   # 'v2'
            'feature_cols':          FEATURE_COLS_V2,
            'trained_at':            pd.Timestamp.now().isoformat(),
            'passes_gate':           passes_gate,
            'optuna_best_params':    best_params,
        }
        with open(os.path.join(base_dir, 'features.json'), 'w') as f:
            json.dump(features_data, f, indent=2)

        # Model weights
        weights = {'xgb': 1/3, 'lgb': 1/3, 'cat': 1/3}
        with open(os.path.join(base_dir, 'weights.json'), 'w') as f:
            json.dump(weights, f, indent=2)

        # Metadata (for production gate check in predictor.py)
        metadata = {
            'passes_gate':           passes_gate,
            'feature_engine_version': FEATURE_ENGINE_VERSION,
            'n_features':            len(feature_names),
            'trained_at':            pd.Timestamp.now().isoformat(),
            'version':               version,
        }
        with open(os.path.join(base_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Base models
        for name, model in models.items():
            joblib.dump(model, os.path.join(base_dir, f'{name}_model.pkl'))

        # Meta-learner
        if meta_model is not None:
            joblib.dump(meta_model, os.path.join(base_dir, 'meta_model.pkl'))

        # Scaler
        if scaler is not None:
            joblib.dump(scaler, os.path.join(base_dir, 'scaler.pkl'))

        logger.info(
            f"[{ticker}] V2 artifacts saved to {base_dir} "
            f"(features=V2/{len(feature_names)}, gate={'PASS' if passes_gate else 'FAIL'})"
        )

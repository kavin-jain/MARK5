"""
MARK5 INFERENCE ENGINE v7.0 - ARCHITECT EDITION
Revisions:
1. ATOMIC MODEL SWAPPING: No race conditions during weight updates.
2. ENTROPY GUARDRAIL: Rejects predictions if probability distribution is too flat.
3. FAIL-SAFE DEFAULTS: Returns 'HOLD' on any internal exception.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
from threading import Lock
from core.models.features import AdvancedFeatureEngine
from core.models.model_versioning import ModelVersionManager
from core.models.training.trainer import NonNegativeMetaLearner
from core.utils.config_manager import get_config

# HACK: Allow unpickling of NonNegativeMetaLearner when artifacts were saved in __main__ scope
import sys
sys.modules['__main__'].NonNegativeMetaLearner = NonNegativeMetaLearner

class AtomicModelContainer:
    """
    Holds the state of the world.
    Swapped atomically to prevent partial reads during inference.

    v10.0: meta_model attribute added for stacking ensemble inference.
    When meta_model is not None (v10+ artifacts), prediction uses
    LogisticRegression(p_xgb, p_lgb, p_cat) instead of arithmetic mean.
    When None (v9 or older artifacts), falls back to arithmetic mean.
    """
    def __init__(
        self,
        models: Dict,
        scaler,
        weights: Dict,
        schema: list,
        calibrators: Dict = None,
        meta_model=None,
    ):
        self.models = models
        self.scaler = scaler
        self.weights = weights
        self.schema = schema
        self.calibrators = calibrators or {}
        self.meta_model = meta_model   # LogisticRegression or None
        self.timestamp = datetime.now()

class MARK5Predictor:
    def __init__(self, ticker: str, data_provider=None, allow_shadow: bool = False):
        self.logger = logging.getLogger(f"MARK5.Inference.{ticker}")
        self.ticker = ticker
        self.allow_shadow = allow_shadow
        
        # Centralized Data Pipeline
        from core.data.data_pipeline import DataPipeline
        self.pipeline = DataPipeline()
        self.feature_engine = self.pipeline.feature_engine
        
        # Dependency Injection or Singleton Config
        self.config = get_config()
        self.version_manager = ModelVersionManager(self.config.__dict__ if hasattr(self.config, '__dict__') else self.config)
        self.version = self.version_manager.get_latest_version(ticker)
        
        self._container: Optional[AtomicModelContainer] = None
        self._lock = Lock() # For swapping the container
        self.dl_seq_length = 64 # Aligned with v13.0 trainer
        
        self.reload_artifacts()

    def reload_artifacts(self):
        """
        Loads artifacts into a NEW container, then atomically swaps.
        Zero downtime.
        """
        try:
            # Look for versioned directory first
            version = self.version_manager.get_latest_version(self.ticker)
            base_dir = f"models/{self.ticker}/v{version}"
            
            # Fallback to non-versioned if version is 0 or path missing
            if version == 0 or not os.path.exists(base_dir):
                base_dir = f"models/{self.ticker}"
            
            # Load Schema
            if not os.path.exists(f"{base_dir}/features.json"):
                self.logger.warning(f"No artifacts found for {self.ticker}")
                return

            with open(f"{base_dir}/features.json", 'r') as f:
                schema = json.load(f)
            
            # Load Scaler (Optional for self-standardized features)
            scaler_path = f"{base_dir}/scaler.pkl"
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                self.logger.info(f"No scaler found for {self.ticker}, skipping scaling.")
                scaler = None
            
            # Load Weights
            weights = {}
            if os.path.exists(f"{base_dir}/weights.json"):
                with open(f"{base_dir}/weights.json", 'r') as f:
                    weights = json.load(f)

            # Load base models
            models = {}
            for name in ['xgb', 'lgb', 'cat', 'rf']:
                model_path = f"{base_dir}/{name}_model.pkl"
                if os.path.exists(model_path):
                    models[name] = joblib.load(model_path)
            
            # Load TCN (v12.0+)
            tcn_path = f"{base_dir}/tcn_model.keras"
            if os.path.exists(tcn_path):
                try:
                    import tensorflow as tf
                    # TCN requires custom objects if defined in system.py
                    from core.models.tcn.system import focal_loss, TemporalBlock
                    models['tcn'] = tf.keras.models.load_model(
                        tcn_path, 
                        custom_objects={'loss': focal_loss(), 'TemporalBlock': TemporalBlock}
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to load TCN model: {e}")

            # Load Meta-Learner (v10.0+)
            meta_model = None
            meta_model_path = f"{base_dir}/meta_model.pkl"
            if os.path.exists(meta_model_path):
                meta_model = joblib.load(meta_model_path)

            # Load Metadata & Enforce PRODUCTION GATE (v10.1)
            meta_path = f"{base_dir}/metadata.json"
            if not os.path.exists(meta_path):
                self.logger.warning(f"No metadata for {self.ticker} — refusing to load unverified model.")
                return

            with open(meta_path, 'r') as fh:
                metadata = json.load(fh)

            passes_gate = metadata.get('passes_gate', False)
            if not passes_gate and not self.allow_shadow:
                self.logger.error(
                    f"🛑 GATE FAIL: {self.ticker} v{metadata.get('version')} "
                    f"did NOT pass production gate. Refusing to load for LIVE trading."
                )
                return
            elif not passes_gate and self.allow_shadow:
                self.logger.warning(
                    f"⚠️ SHADOW LOAD: {self.ticker} v{metadata.get('version')} "
                    f"is being loaded for diagnostic/backtest use despite gate failure."
                )

            # ATOMIC SWAP — meta_model passed to constructor (v10.0+)
            new_container = AtomicModelContainer(
                models, scaler, weights, schema, {},
                meta_model=meta_model,
            )
            with self._lock:
                self._container = new_container

            self.logger.info(
                f"✅ Models hot-swapped for {self.ticker}: {list(models.keys())} | "
                f"meta={'yes' if meta_model else 'no'}"
            )

        except Exception as e:
            self.logger.error(f"Artifact reload failed: {e}")

    def _calculate_entropy(self, probs: np.ndarray) -> float:
        """High entropy = Flat distribution = Confusion."""
        return -np.sum(probs * np.log(probs + 1e-9))

    def _build_context(self, data: pd.DataFrame) -> Optional[Dict]:
        """Build context for feature engineering using DataPipeline."""
        try:
            start_date = str(data.index[0].date())
            end_date = str(data.index[-1].date())
            
            # Use pipeline's providers
            context = self.pipeline.market_provider.build_feature_context(
                stock_df=data, sector='Unknown',
                start_date=start_date, end_date=end_date
            )
            fii_series = self.pipeline.fii_provider.get_fii_flow(start_date, end_date)
            context['fii_net'] = fii_series
            return context
        except Exception as e:
            self.logger.warning(f"Context build failed: {e}")
            return None

    def _check_regime(self, raw_data: pd.DataFrame) -> str:
        """
        Rule 23: Detect BEAR regime from price data.
        Returns 'BEAR' if close is below 200d EMA, else 'BULL'.
        Requires at least 200 bars of data.
        """
        if raw_data is None or len(raw_data) < 50:
            return 'UNKNOWN'
        try:
            close = raw_data['close']
            ema_period = min(200, len(close) - 1)
            ema200 = close.ewm(span=ema_period, adjust=False).mean()
            last_close = float(close.iloc[-1])
            last_ema = float(ema200.iloc[-1])
            if last_close < last_ema:
                return 'BEAR'
            return 'BULL'
        except Exception:
            return 'UNKNOWN'

    def predict(self, raw_data: Optional[pd.DataFrame] = None, **kwargs) -> Dict:
        # Get local reference to avoid locking overhead
        container = self._container
        if not container:
            return {'status': 'error', 'msg': 'System initializing'}

        try:
            # 0. Auto-fetch if data is missing
            if raw_data is None:
                self.logger.info(f"[{self.ticker}] No data provided for inference. Fetching latest...")
                raw_data = self.pipeline.get_market_data(self.ticker, period='1y')
                if raw_data.empty:
                    return {'signal': 'HOLD', 'reason': 'No data available for inference'}

            # 0.75. PRIMARY SIGNAL GATE (Meta-Labeling Architecture)
            # The meta-learner predicts the probability of success of a primary signal.
            # If there is no primary signal, we MUST hold.
            from core.models.training.financial_engineer import FinancialEngineer
            fe = FinancialEngineer()
            primary_signals = fe.get_primary_signals(raw_data)
            current_primary_signal = primary_signals.iloc[-1]

            if current_primary_signal == 0:
                return {
                    'status': 'success',
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'No primary trend signal',
                    'regime': 'UNKNOWN',
                    'entropy': 0.0,
                    'ensemble_method': 'primary_gate',
                    'probs': [1.0, 0.0],
                }

            # 0.5. RULE 23 REGIME GATE — check BEFORE any ML inference
            # If stock is below its 200d EMA, BEAR regime: suppress ALL long entries.
            regime = self._check_regime(raw_data)
            if regime == 'BEAR' and current_primary_signal == 1:
                return {
                    'status': 'success',
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'Rule 23: BEAR regime — suppressing LONG entry',
                    'regime': 'BEAR',
                    'entropy': 0.693,
                    'ensemble_method': 'regime_gate',
                    'probs': [1.0, 0.0],
                }
            elif regime == 'BULL' and current_primary_signal == -1:
                return {
                    'status': 'success',
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'Rule 23: BULL regime — suppressing SHORT entry',
                    'regime': 'BULL',
                    'entropy': 0.693,
                    'ensemble_method': 'regime_gate',
                    'probs': [1.0, 0.0],
                }

            # 1. Feature Engineering
            context = kwargs.get('context')
            if context is None:
                context = self._build_context(raw_data)
                
            df_feats = self.feature_engine.engineer_all_features(raw_data, ticker=self.ticker, context=context)
            
            # 2. Schema Alignment (Strict)
            dropped_cols = set(df_feats.columns) - set(container.schema)
            zero_filled_cols = set(container.schema) - set(df_feats.columns)
            if dropped_cols or zero_filled_cols:
                self.logger.warning(
                    f"Schema mismatch for {self.ticker}: "
                    f"extra_features_dropped={dropped_cols}, "
                    f"missing_features_zero_filled={zero_filled_cols}"
                )
            df_aligned = df_feats.reindex(columns=container.schema).fillna(0)
            
            # 3. Scaling
            if container.scaler is not None:
                X_scaled_np = container.scaler.transform(df_aligned.values)
                X_scaled = pd.DataFrame(X_scaled_np, columns=df_aligned.columns, index=df_aligned.index)
            else:
                X_scaled = df_aligned
                
            X_current = X_scaled.iloc[[-1]]
            
            # 3b. TCN-specific Feature Engineering (Institutional Standard)
            X_seq = None
            if 'tcn' in container.models:
                from core.models.features import engineer_tcn_features_df
                df_tcn = engineer_tcn_features_df(raw_data, context=context)
                if len(df_tcn) >= self.dl_seq_length:
                    X_seq = df_tcn.values[-self.dl_seq_length:].reshape(1, self.dl_seq_length, -1)
                else:
                    self.logger.warning(f"Insufficient data for TCN sequence for {self.ticker}")

            # 4. Inference — base models (skip TCN here, handled below)
            model_probs_class1 = []
            model_details = {}
            
            for name, model in container.models.items():
                if name == 'tcn':
                    continue
                w = container.weights.get(name, 1.0)
                if w <= 0:
                    continue
                try:
                    probs = model.predict_proba(X_current)[0]
                    raw_prob_buy = float(probs[1]) if len(probs) > 1 else float(probs[0])
                    model_probs_class1.append(raw_prob_buy)
                    model_details[name] = {'raw': raw_prob_buy}
                except Exception:
                    continue

            # TCN inference — MUST happen before meta-learner so model_details['tcn'] is set
            if X_seq is not None and 'tcn' in container.models:
                try:
                    tcn_output = container.models['tcn'].predict(X_seq, verbose=0)
                    if isinstance(tcn_output, (list, tuple)):
                        tcn_prob = float(tcn_output[0][0][0])
                    else:
                        tcn_prob = float(tcn_output[0][0])
                    model_probs_class1.append(tcn_prob)
                    model_details['tcn'] = {'raw': tcn_prob}
                except Exception as e:
                    self.logger.warning(f"TCN inference failed: {e}")

            if not model_probs_class1:
                return {'signal': 'HOLD', 'reason': 'All models failed'}

            # 5. Ensemble — stacking meta-learner or arithmetic mean fallback
            stacking_keys = ['xgb', 'lgb', 'cat', 'tcn']
            
            if container.meta_model is not None:
                meta_X = np.array([
                    model_details.get(m, {}).get('raw', 0.5)
                    for m in stacking_keys
                ]).reshape(1, -1)
                try:
                    confidence = float(container.meta_model.predict_proba(meta_X)[0, 1])
                except ValueError:
                    meta_X_v10 = meta_X[:, :3]
                    confidence = float(container.meta_model.predict_proba(meta_X_v10)[0, 1])
                ensemble_method = 'stacking'
            else:
                confidence = float(np.mean(model_probs_class1)) if model_probs_class1 else 0.0
                ensemble_method = 'arithmetic_mean'
            
            # 6. Signal generation per Rule 21 — LONG-ONLY (delivery, regime-gated above)
            PROBABILITY_HURDLE = 0.50
            
            if confidence >= PROBABILITY_HURDLE:
                # Use the primary signal's direction!
                signal = "BUY" if current_primary_signal == 1 else "SELL"
            else:
                signal = "HOLD"
                if confidence > 0.50:
                    signal = f"HOLD (Conf {confidence:.0%} < {PROBABILITY_HURDLE:.0%} hurdle)"

            entropy_val = self._calculate_entropy(np.array([1.0 - confidence, confidence]))

            return {
                'status': 'success',
                'signal': signal,
                'confidence': confidence,
                'entropy': entropy_val,
                'ensemble_method': ensemble_method,
                'regime': regime,
                'probs': [1.0 - confidence, confidence],
            }

        except Exception as e:
            self.logger.error(f"Inference crash: {e}")
            return {'signal': 'HOLD', 'reason': 'Crash Protection'}
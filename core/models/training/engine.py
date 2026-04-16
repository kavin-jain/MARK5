"""
MARK5 ADAPTIVE LEARNING ENGINE v7.0 - ARCHITECT EDITION
Revisions:
1. COVARIANCE-ADJUSTED KELLY: Penalizes correlated models to prevent risk concentration.
2. QUANTILE DRIFT DETECTION: Replaces Z-Score with Robust Quantile checks (Fat-tail safe).
3. CIRCUIT BREAKER: Hard stops inference if calibration fails.
"""

import logging
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from core.infrastructure.database_manager import MARK5DatabaseManager
from core.models.registry import RobustModelRegistry, STATUS_ACTIVE, STATUS_SHADOW

class ModelTracker:
    """Institutional Experiment Tracker (MLflow-lite)"""
    def __init__(self, log_path: str = "logs/training_experiments.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_experiment(self, ticker: str, version: int, metrics: Dict, hyperparams: Dict, status: str):
        record = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "version": version,
            "status": status,
            "metrics": metrics,
            "hyperparams": hyperparams
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

class LearningEngine:
    def __init__(self, db_manager: MARK5DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger("MARK5.LearningEngine")
        self.registry = RobustModelRegistry()
        self.tracker = ModelTracker()

    def deploy_model(self, ticker: str, version: int, metrics: Dict, passes_gate: bool):
        """Routes model to Active or Shadow deployment based on production gate."""
        status = STATUS_ACTIVE if passes_gate else STATUS_SHADOW
        
        # Log to tracker
        self.tracker.log_experiment(
            ticker=ticker,
            version=version,
            metrics=metrics,
            hyperparams={}, # Could be populated from config
            status=status
        )
        
        # Register in RobustRegistry
        model_path = f"models/{ticker}/v{version}/metadata.json"
        self.registry.register_model(
            ticker=ticker,
            model_type="ensemble",
            path=model_path,
            metadata={**metrics, "deployment_timestamp": datetime.now().isoformat()},
            status=status
        )
        
        self.logger.info(f"🚀 Model {ticker} v{version} deployed to {status.upper()} mode.")
        return status

    def _get_trade_data(self, limit: int = 500) -> pd.DataFrame:
        conn = None
        try:
            conn = self.db.get_connection()
            query = """
                SELECT stock, signal_type, model_confidence, net_pnl, exit_timestamp
                FROM trade_journal WHERE status = 'CLOSED' 
                ORDER BY exit_timestamp DESC LIMIT ?
            """
            return pd.read_sql_query(query, conn, params=(limit,))
        finally:
            if conn: self.db.return_connection(conn)

    def detect_robust_drift(self, ticker: str, window: int = 50) -> Dict:
        """
        Uses Rolling Quantiles instead of Z-Score.
        Markets are not Normal. Z-Score underestimates tail risk.
        """
        df = self._get_trade_data(limit=300)
        if df.empty:
             return {"drift": False, "reason": "No Data"}
             
        df_ticker = df[df['stock'] == ticker].sort_values('exit_timestamp')
        
        if len(df_ticker) < window * 2:
            return {"drift": False, "reason": "Insufficient Data"}

        # Calculate PnL Stream
        pnl = df_ticker['net_pnl'].values
        
        # Current Window vs Historical Window
        recent_pnl = pnl[-window:]
        history_pnl = pnl[:-window]
        
        # Check Lower Quantile (e.g., the 5th percentile worst trades)
        hist_q05 = np.percentile(history_pnl, 5)
        recent_q05 = np.percentile(recent_pnl, 5)
        
        # If the recent "worst case" is significantly worse than historical "worst case"
        # Drift is detected.
        drift_magnitude = (hist_q05 - recent_q05)
        
        # Threshold: If recent downside is 50% worse than historical downside
        # Note: pnl is usually negative for losses, so we check if recent is LOWER (more negative)
        # If hist_q05 is -100, and recent_q05 is -160, that's a drift.
        
        is_drift = False
        if hist_q05 < 0:
             if recent_q05 < (hist_q05 * 1.5): is_drift = True
        else:
             # If history was profitable (positive q05), and recent is negative
             if recent_q05 < (hist_q05 - 0.005): is_drift = True

        return {
            "drift": is_drift,
            "metric": "Q05_Tail_Risk",
            "recent_q05": round(recent_q05, 4),
            "hist_q05": round(hist_q05, 4),
            "reason": "Tail Risk Expansion" if is_drift else "Stable"
        }

    def optimize_weights_covariance_kelly(self) -> Dict[str, float]:
        """
        Calculates Weights penalizing CORRELATION.
        If Model A and Model B are 90% correlated, we treat them as one.
        """
        df = self._get_trade_data(limit=1000)
        if df.empty: return {}

        models = df['signal_type'].unique()
        raw_kellys = {}

        # 1. Calculate Individual Kelly
        for m in models:
            subset = df[df['signal_type'] == m]
            if len(subset) < 20: 
                raw_kellys[m] = 0.0
                continue
                
            wins = subset[subset['net_pnl'] > 0]
            losses = subset[subset['net_pnl'] <= 0]
            
            W = len(wins) / len(subset)
            avg_win = wins['net_pnl'].mean() if not wins.empty else 0
            avg_loss = abs(losses['net_pnl'].mean()) if not losses.empty else 1.0
            
            R = avg_win / avg_loss if avg_loss > 0 else 1.0
            kelly = (W - (1 - W) / R)
            raw_kellys[m] = max(0.0, kelly * 0.5) # Half Kelly

        # 2. Correlation Penalty (Institutional Alignment Truth)
        final_weights = {}
        
        for m1 in models:
            if raw_kellys.get(m1, 0) == 0:
                final_weights[m1] = 0.0
                continue
                
            correlation_penalty = 1.0
            
            # Extract PnL series for m1 with timestamps for alignment
            s1 = df[df['signal_type'] == m1].set_index('exit_timestamp')['net_pnl']
            
            for m2 in models:
                if m1 == m2 or raw_kellys.get(m2, 0) == 0: continue
                
                # Extract PnL series for m2
                s2 = df[df['signal_type'] == m2].set_index('exit_timestamp')['net_pnl']
                
                # Use pandas intersection to align synchronous trade exits
                common_dates = s1.index.intersection(s2.index)
                
                if common_dates.size < 15:
                    corr = 0.0
                else:
                    corr = s1.loc[common_dates].corr(s2.loc[common_dates])
                    if np.isnan(corr): corr = 0.0
                
                # If highly correlated (> 0.7), reduce weight
                if corr > 0.7:
                    correlation_penalty *= 0.8 # 20% penalty per correlated sibling
            
            final_weights[m1] = raw_kellys[m1] * correlation_penalty

        # Normalize
        total = sum(final_weights.values())
        if total == 0: return {m: 1.0/len(models) for m in models}
        
        return {k: round(v/total, 3) for k, v in final_weights.items()}

    def run_learning_cycle(self, ticker: str, context: Optional[Dict] = None) -> Dict:
        """
        Executes the full learning/risk cycle for a given ticker.
        If context['fii_is_synthetic'] is True, applies a 0.5x penalty to conviction.
        """
        drift = self.detect_robust_drift(ticker)
        weights = self.optimize_weights_covariance_kelly()
        
        # ── FII Integrity Check (Rule 51 Fallback) ───────────────────────────
        fii_penalty = 1.0
        if context and context.get('fii_is_synthetic'):
            self.logger.warning(f"[{ticker}] Synthetic FII data detected — penalizing signal weight 50%")
            fii_penalty = 0.5
            
        action = "MAINTAIN"
        if drift['drift']:
            action = "HALT_TRADING" # Hard Stop
            
        return {
            "ticker": ticker,
            "action": action,
            "drift_status": drift,
            "new_weights": weights,
            "fii_penalty": fii_penalty
        }

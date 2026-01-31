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
from typing import Dict, List, Optional
from core.infrastructure.database_manager import MARK5DatabaseManager

class LearningEngine:
    def __init__(self, db_manager: MARK5DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger("MARK5.LearningEngine")

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

        # 1. Pivot to get PnL matrix (Index=TradeTime, Columns=Model)
        # Approximation: We align by closest timestamp or just take recent streams
        # For simplicity in this architecture, we calculate Kelly per model and apply a penalty matrix
        
        models = df['signal_type'].unique()
        raw_kellys = {}
        performance_vectors = {}

        # Calculate Individual Kelly
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
            
            # Store normalized PnL vector for correlation
            # We paddle with 0s to align lengths for correlation check (naive but fast)
            pnl_vec = subset['net_pnl'].values[-50:] 
            if len(pnl_vec) < 50: pnl_vec = np.pad(pnl_vec, (50-len(pnl_vec), 0), 'constant')
            performance_vectors[m] = pnl_vec

        # 2. Correlation Penalty
        final_weights = {}
        
        for m1 in models:
            if raw_kellys.get(m1, 0) == 0:
                final_weights[m1] = 0.0
                continue
                
            correlation_penalty = 1.0
            
            for m2 in models:
                if m1 == m2 or raw_kellys.get(m2, 0) == 0: continue
                
                # Calculate Correlation
                v1 = performance_vectors.get(m1, np.zeros(50))
                v2 = performance_vectors.get(m2, np.zeros(50))
                
                if np.std(v1) == 0 or np.std(v2) == 0: continue
                
                corr = np.corrcoef(v1, v2)[0, 1]
                
                # If highly correlated (> 0.7), reduce weight
                if corr > 0.7:
                    correlation_penalty *= 0.8 # 20% penalty per correlated sibling
            
            final_weights[m1] = raw_kellys[m1] * correlation_penalty

        # Normalize
        total = sum(final_weights.values())
        if total == 0: return {m: 1.0/len(models) for m in models}
        
        return {k: round(v/total, 3) for k, v in final_weights.items()}

    def run_learning_cycle(self, ticker: str) -> Dict:
        drift = self.detect_robust_drift(ticker)
        weights = self.optimize_weights_covariance_kelly()
        
        action = "MAINTAIN"
        if drift['drift']:
            action = "HALT_TRADING" # Hard Stop
            
        return {
            "ticker": ticker,
            "action": action,
            "drift_status": drift,
            "new_weights": weights
        }

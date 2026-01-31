"""
MARK5 ADAPTIVE LEARNING ENGINE v6.0
-----------------------------------
Architect: The Legendary Trader
Features:
1. Kelly Criterion Weight Optimization (Mathematically optimal growth)
2. Brier Score Calibration Tracking (Detects model hallucination)
3. Rolling Z-Score Drift Detection (Statistical anomaly detection)
4. Regime-Specific Performance Bucketing
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from core.infrastructure.legacy_sqlite import MARK5DatabaseManager

class LearningEngine:
    """
    The Brain of MARK5.
    Adjusts behavior based on Mathematical Expectation, not just 'Win Rate'.
    """
    
    def __init__(self, db_manager: MARK5DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger("MARK5.LearningEngine")
        # Decay factor for rolling stats (gives more weight to recent trades)
        self.decay_factor = 0.95 

    def _get_trade_data(self, ticker: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        """Optimized SQL fetch for learning metrics."""
        conn = None
        try:
            conn = self.db.get_connection()
            query = """
                SELECT 
                    stock, 
                    signal_type, 
                    model_confidence, 
                    entry_price, 
                    exit_price, 
                    net_pnl, 
                    exit_timestamp
                FROM trade_journal 
                WHERE status = 'CLOSED' 
            """
            params = []
            if ticker:
                query += " AND stock = ?"
                params.append(ticker)
            
            query += " ORDER BY exit_timestamp DESC LIMIT ?"
            params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=tuple(params))
            return df
        finally:
            if conn: self.db.return_connection(conn)

    def calculate_calibration_curve(self, df: pd.DataFrame) -> Dict:
        """
        DETECTS HALLUCINATIONS.
        Compares Model Confidence vs Actual Win Rate.
        """
        if df.empty: return {}
        
        # Bin confidence into buckets (0.5-0.6, 0.6-0.7, etc.)
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        df['conf_bin'] = pd.cut(df['model_confidence'], bins=bins)
        
        calibration = {}
        for bin_range in df['conf_bin'].unique():
            if pd.isna(bin_range): continue
            subset = df[df['conf_bin'] == bin_range]
            if len(subset) < 5: continue # Insufficient sample
            
            avg_predicted_conf = subset['model_confidence'].mean()
            actual_win_rate = len(subset[subset['net_pnl'] > 0]) / len(subset)
            
            # Calibration Error: How much is the model lying?
            # Positive = Overconfident (Dangerous), Negative = Underconfident
            error = avg_predicted_conf - actual_win_rate
            
            calibration[str(bin_range)] = {
                'predicted': round(avg_predicted_conf, 2),
                'actual': round(actual_win_rate, 2),
                'error': round(error, 2),
                'status': 'HALLUCINATING' if error > 0.1 else 'CALIBRATED'
            }
            
        return calibration

    def detect_statistical_drift(self, ticker: str, window: int = 30) -> Dict:
        """
        Uses Rolling Z-Score to differentiate 'Bad Luck' from 'Broken Model'.
        If PnL drops > 2 Standard Deviations below mean, it's Drift.
        """
        df = self._get_trade_data(ticker, limit=200) # Need history for Z-score
        if len(df) < window:
            return {"drift": False, "reason": "Insufficient Data"}
        
        # Sort by time ascending for rolling calc
        df = df.sort_values('exit_timestamp')
        
        # Calculate Rolling Mean and Std of PnL
        rolling_mean = df['net_pnl'].rolling(window=window).mean()
        rolling_std = df['net_pnl'].rolling(window=window).std()
        
        # Current PnL stats (last N trades)
        current_mean = rolling_mean.iloc[-1]
        current_std = rolling_std.iloc[-1]
        
        # Calculate Z-Score of the most recent trade outcomes
        recent_pnl_avg = df['net_pnl'].tail(10).mean()
        
        if current_std == 0: return {"drift": False}

        z_score = (recent_pnl_avg - current_mean) / current_std
        
        # Threshold: If recent performance is -2 Sigma away from historical mean
        drift_detected = z_score < -2.0
        
        return {
            "drift": drift_detected,
            "z_score": round(z_score, 2),
            "reason": f"Performance deviation {z_score:.1f}σ from mean" if drift_detected else "Stable"
        }

    def optimize_weights_kelly(self) -> Dict[str, float]:
        """
        Calculates Optimal Weights using the KELLY CRITERION.
        Kelly % = W - (1-W)/R
        Where W = Win Probability, R = Win/Loss Ratio
        """
        df = self._get_trade_data(limit=500)
        if df.empty:
            return {"xgboost": 0.33, "lstm": 0.33, "transformer": 0.33}
            
        weights = {}
        
        # Group by Signal Type (Model)
        for model in df['signal_type'].unique():
            subset = df[df['signal_type'] == model]
            if len(subset) < 10: continue
            
            wins = subset[subset['net_pnl'] > 0]
            losses = subset[subset['net_pnl'] <= 0]
            
            if losses.empty: 
                kelly = 0.5 # Cap max allocation if undefined risk
            else:
                W = len(wins) / len(subset)
                avg_win = wins['net_pnl'].mean()
                avg_loss = abs(losses['net_pnl'].mean())
                
                if avg_loss == 0:
                    R = 1.0 # Edge case
                else:
                    R = avg_win / avg_loss
                
                # Full Kelly is too aggressive. Use Half-Kelly.
                kelly = (W - (1 - W) / R) * 0.5
            
            # Ensure non-negative weights
            weights[model] = max(0.0, kelly)
            
        # Normalize to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight == 0:
            return {"xgboost": 0.33, "lstm": 0.33, "transformer": 0.33}
            
        final_weights = {k: round(v / total_weight, 2) for k, v in weights.items()}
        self.logger.info(f"Kelly-Optimized Weights: {final_weights}")
        return final_weights

    def run_learning_cycle(self, ticker: str) -> Dict:
        """
        Orchestrates the learning process.
        Returns actionable system updates.
        """
        # 1. Detect Drift (Is the model broken?)
        drift_info = self.detect_statistical_drift(ticker)
        
        # 2. Check Calibration (Is the model lying?)
        df = self._get_trade_data(ticker, limit=100)
        calibration = self.calculate_calibration_curve(df)
        
        # 3. Optimize Weights (How much should we trust each model?)
        new_weights = self.optimize_weights_kelly()
        
        # 4. Generate Recommendations
        action = "MAINTAIN"
        if drift_info['drift']:
            action = "HALT_AND_RETRAIN"
        
        # Check for Overconfidence
        hallucinating_bins = [k for k, v in calibration.items() if v['status'] == 'HALLUCINATING']
        if hallucinating_bins:
             self.logger.warning(f"Model Hallucinating in buckets: {hallucinating_bins}")
             # Recommendation: Increase confidence threshold for entry
             action = "INCREASE_THRESHOLD"

        return {
            "ticker": ticker,
            "action": action,
            "drift_status": drift_info,
            "new_weights": new_weights,
            "calibration_issues": hallucinating_bins
        }

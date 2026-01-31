"""
MARK5 PERFORMANCE TRACKER v6.0 (FINANCIAL GRADE)
------------------------------------------------
Architect: The Legendary Trader
Improvements:
1. Financial Metrics (Expectancy, SQN) over Academic Metrics (F1).
2. Statistical Decay Detection (Z-Score) instead of fixed thresholds.
3. Regime-Aware Tracking (Knows if model fails in Chop vs Trend).
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class ModelPerformanceTracker:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default path handling
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_dir = os.path.join(base_dir, 'database', 'main')
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, 'mark5.db')
        
        self.db_path = db_path
        self._init_tables()

    def _init_tables(self):
        """
        Updated Schema: Adds Expectancy, SQN, and Regime Context.
        """
        # We use ALTER TABLE in production migrations, but here is the ideal schema
        self._execute_safe('''
            CREATE TABLE IF NOT EXISTS model_performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                model_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                market_regime TEXT,  -- Crucial: Context
                accuracy REAL,
                expectancy REAL,     -- Crucial: (Win% * AvgWin) - (Loss% * AvgLoss)
                sqn_score REAL,      -- System Quality Number
                profit_factor REAL,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                total_trades INTEGER,
                metadata TEXT
            )
        ''')
        
        self._execute_safe('''
            CREATE TABLE IF NOT EXISTS performance_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                model_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                z_score REAL,        -- Statistical significance of the failure
                message TEXT,
                metrics TEXT
            )
        ''')
        
        # Schema Migration: Check for new columns and add if missing
        self._migrate_schema()

    def _migrate_schema(self):
        """Ensure existing tables have new columns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check model_performance_history columns
        cursor.execute("PRAGMA table_info(model_performance_history)")
        columns = [info[1] for info in cursor.fetchall()]
        
        new_cols = {
            'market_regime': 'TEXT',
            'expectancy': 'REAL',
            'sqn_score': 'REAL',
            'profit_factor': 'REAL'
        }
        
        for col, dtype in new_cols.items():
            if col not in columns:
                try:
                    cursor.execute(f"ALTER TABLE model_performance_history ADD COLUMN {col} {dtype}")
                except Exception as e:
                    print(f"Migration warning: Could not add {col}: {e}")
                    
        # Check performance_alerts columns
        cursor.execute("PRAGMA table_info(performance_alerts)")
        alert_columns = [info[1] for info in cursor.fetchall()]
        
        if 'z_score' not in alert_columns:
            try:
                cursor.execute("ALTER TABLE performance_alerts ADD COLUMN z_score REAL")
            except Exception as e:
                print(f"Migration warning: Could not add z_score: {e}")
                
        conn.commit()
        conn.close()

    def _execute_safe(self, query: str, params: Tuple = ()):
        """Robust SQLite execution with retry."""
        for i in range(5):
            conn = None
            try:
                conn = sqlite3.connect(self.db_path, timeout=10)
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e):
                    time.sleep(0.1 * (2 ** i))
                    continue
                raise e
            finally:
                if conn: conn.close()

    def _calculate_financial_metrics(self, metrics: Dict) -> Dict:
        """
        Enriches raw metrics with Financial Reality (Expectancy & SQN).
        """
        win_rate = metrics.get('win_rate', 0.0)
        avg_win = metrics.get('avg_win', 0.0)
        avg_loss = abs(metrics.get('avg_loss', 0.0)) # Ensure positive
        total_trades = metrics.get('total_trades', 0)
        
        # 1. Expectancy (The "Edge")
        # E = (Win% * AvgWin) - (Loss% * AvgLoss)
        loss_rate = 1.0 - win_rate
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        
        # 2. System Quality Number (SQN)
        # SQN = sqrt(N) * (Expectancy / StdDev of R-multiples)
        # Simplified approximation for streaming data
        sqn = 0.0
        if total_trades > 30 and avg_loss > 0:
            # Approx R-multiple std dev (heuristic)
            r_std = 1.5 # Placeholder if granular trade data missing
            sqn = (total_trades ** 0.5) * (expectancy / (avg_loss * r_std))
            
        metrics['expectancy'] = round(expectancy, 4)
        metrics['sqn_score'] = round(sqn, 2)
        
        return metrics

    def record_performance(self, ticker: str, model_type: str, 
                          metrics: Dict, market_regime: str = "UNKNOWN", metadata: Dict = None):
        """
        Records performance with Context (Regime) and Math (Expectancy).
        """
        # Enrich metrics
        fin_metrics = self._calculate_financial_metrics(metrics)
        
        self._execute_safe('''
            INSERT INTO model_performance_history 
            (ticker, model_type, timestamp, market_regime, accuracy, expectancy, 
             sqn_score, profit_factor, win_rate, avg_win, avg_loss, 
             max_drawdown, sharpe_ratio, total_trades, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker,
            model_type,
            datetime.now().isoformat(),
            market_regime,
            fin_metrics.get('accuracy'),
            fin_metrics.get('expectancy'),
            fin_metrics.get('sqn_score'),
            fin_metrics.get('profit_factor'),
            fin_metrics.get('win_rate'),
            fin_metrics.get('avg_win'),
            fin_metrics.get('avg_loss'),
            fin_metrics.get('max_drawdown'),
            fin_metrics.get('sharpe_ratio'),
            fin_metrics.get('total_trades'),
            json.dumps(metadata) if metadata else None
        ))

    def check_statistical_decay(self, ticker: str, model_type: str) -> Tuple[bool, str]:
        """
        Detects 'Broken' models using Z-Score.
        Distinguishes between "Bad Luck" (Noise) and "Decay" (Signal).
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Fetch last 100 performance records
            df = pd.read_sql_query('''
                SELECT timestamp, expectancy 
                FROM model_performance_history 
                WHERE ticker = ? AND model_type = ? 
                ORDER BY timestamp DESC LIMIT 100
            ''', conn, params=(ticker, model_type))
            
            if len(df) < 20: return False, "Insufficient Data"
            
            # Sort chronological
            df = df.sort_values('timestamp')
            
            # Calculate Baseline (Historic) vs Recent
            baseline_window = df['expectancy'].iloc[:-5] # Everything except last 5
            recent_window = df['expectancy'].iloc[-5:]   # Last 5 updates
            
            mean_hist = baseline_window.mean()
            std_hist = baseline_window.std()
            mean_recent = recent_window.mean()
            
            if std_hist == 0: return False, "Stable"
            
            # Z-Score Calculation
            # How many standard deviations has performance dropped?
            z_score = (mean_recent - mean_hist) / std_hist
            
            # Threshold: -2.0 Sigma (95% confidence it's broken)
            if z_score < -2.0:
                msg = f"CRITICAL DECAY: Expectancy dropped {z_score:.1f}σ (Avg: {mean_recent:.2f} vs Hist: {mean_hist:.2f})"
                
                self._record_alert(ticker, model_type, 'STATISTICAL_DECAY', 'HIGH', msg, 
                                 {'z_score': z_score, 'current_expectancy': mean_recent})
                return True, msg
                
            elif z_score < -1.5:
                msg = f"WARNING: Performance drift {z_score:.1f}σ"
                self._record_alert(ticker, model_type, 'DRIFT_WARNING', 'MEDIUM', msg, 
                                 {'z_score': z_score})
                return True, msg
                
            return False, "Stable"
            
        finally:
            conn.close()

    def get_regime_report(self, ticker: str) -> str:
        """
        Generates the 'Holy Grail' report: Performance per Market Regime.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            query = '''
                SELECT 
                    market_regime,
                    COUNT(*) as samples,
                    AVG(accuracy) as acc,
                    AVG(expectancy) as edge,
                    AVG(max_drawdown) as dd
                FROM model_performance_history
                WHERE ticker = ?
                GROUP BY market_regime
            '''
            df = pd.read_sql_query(query, conn, params=(ticker,))
            
            if df.empty: return "No regime data available."
            
            report = [f"\n🏛️ REGIME ANALYSIS FOR {ticker}"]
            report.append(f"{'Regime':<20} {'Samples':<8} {'Accuracy':<10} {'Expectancy (Edge)':<20} {'Drawdown':<10}")
            report.append("-" * 75)
            
            for _, row in df.iterrows():
                # Highlight bad regimes
                edge_str = f"{row['edge']:.2f}" if row['edge'] is not None else "N/A"
                if row['edge'] is not None and row['edge'] < 0: edge_str += " ⚠️"
                
                acc_val = row['acc'] * 100 if row['acc'] is not None else 0.0
                dd_val = row['dd'] * 100 if row['dd'] is not None else 0.0
                
                report.append(f"{row['market_regime']:<20} {row['samples']:<8} {acc_val:.1f}%     {edge_str:<20} {dd_val:.1f}%")
                
            return "\n".join(report)
        finally:
            conn.close()
            
    def generate_performance_report(self, ticker: str = None) -> str:
        """Generate comprehensive performance report (Backward Compatible Wrapper)"""
        # Re-implementing basic report functionality for compatibility
        report = []
        report.append("=" * 70)
        report.append("MODEL PERFORMANCE REPORT (v6.0 Financial Grade)")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        conn = sqlite3.connect(self.db_path)
        if ticker:
            tickers = [ticker]
        else:
            query = "SELECT DISTINCT ticker FROM model_performance_history ORDER BY ticker"
            tickers = [row[0] for row in conn.execute(query).fetchall()]
            
        for tick in tickers:
            report.append(f"\n## {tick}")
            report.append("-" * 70)
            
            # Get latest performance
            query = '''
                SELECT model_type, accuracy, win_rate, expectancy, profit_factor, timestamp
                FROM model_performance_history
                WHERE ticker = ?
                AND id IN (
                    SELECT MAX(id) FROM model_performance_history 
                    WHERE ticker = ?
                    GROUP BY model_type
                )
                ORDER BY expectancy DESC
            '''
            results = conn.execute(query, (tick, tick)).fetchall()
            
            if results:
                report.append(f"{'Model':<15} {'Accuracy':>10} {'Win Rate':>10} {'Expectancy':>12} {'Profit Factor':>14} {'Last Update'}")
                report.append("-" * 85)
                
                for model_type, acc, win_rate, exp, pf, ts in results:
                    timestamp = datetime.fromisoformat(ts).strftime('%Y-%m-%d %H:%M')
                    acc_str = f"{acc*100:.1f}%" if acc else "N/A"
                    win_str = f"{win_rate*100:.1f}%" if win_rate else "N/A"
                    exp_str = f"{exp:.4f}" if exp is not None else "N/A"
                    pf_str = f"{pf:.2f}" if pf is not None else "N/A"
                    
                    report.append(f"{model_type:<15} {acc_str:>10} {win_str:>10} {exp_str:>12} {pf_str:>14} {timestamp}")
            
            # Append Regime Report
            report.append(self.get_regime_report(tick))
            
        conn.close()
        return "\n".join(report)

    def _record_alert(self, ticker: str, model_type: str, alert_type: str,
                     severity: str, message: str, metrics: Dict):
        """Internal alert logger."""
        self._execute_safe('''
            INSERT INTO performance_alerts 
            (ticker, model_type, timestamp, alert_type, severity, message, metrics, z_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker, model_type, datetime.now().isoformat(),
            alert_type, severity, message, json.dumps(metrics), metrics.get('z_score', 0.0)
        ))

if __name__ == '__main__':
    tracker = ModelPerformanceTracker()
    
    print("Testing Statistical Decay Detection...")
    
    # 1. Train history (Good performance)
    for _ in range(50):
        tracker.record_performance('TEST_BANKNIFTY', 'xgboost', {
            'win_rate': 0.60, 'avg_win': 100, 'avg_loss': 50, # Expectancy = 40
            'total_trades': 100
        }, market_regime="LOW_VOLATILITY")
        
    # 2. Recent failure (Bad performance)
    for _ in range(5):
        tracker.record_performance('TEST_BANKNIFTY', 'xgboost', {
            'win_rate': 0.30, 'avg_win': 80, 'avg_loss': 60, # Expectancy = -18
            'total_trades': 105
        }, market_regime="HIGH_VOLATILITY")
        
    # 3. Check Decay
    is_decayed, msg = tracker.check_statistical_decay('TEST_BANKNIFTY', 'xgboost')
    print(f"Decay Detected: {is_decayed}")
    print(f"Message: {msg}")
    
    # 4. View Regime Report
    print(tracker.get_regime_report('TEST_BANKNIFTY'))

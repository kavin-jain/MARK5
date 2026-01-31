"""
MARK5 Training Analytics v6.0 (FINANCIAL GRADE)
-----------------------------------------------
Architect: The Legendary Trader
Improvements:
1. ResultsAnalyzer: Percentage Drawdown, Sortino Ratio, Equity Curve.
2. TrainingResultsAnalyzer: Regime Specialists, Drift Analysis.
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import os
from typing import Dict, List, Optional

class ResultsAnalyzer:
    def __init__(self, trades_df: pd.DataFrame, initial_capital: float = 100000.0):
        self.trades = trades_df.copy()
        self.initial_capital = initial_capital
        
        if not self.trades.empty:
            if 'timestamp' in self.trades.columns:
                self.trades['timestamp'] = pd.to_datetime(self.trades['timestamp'])
                self.trades.sort_values('timestamp', inplace=True)
            
            # 1. Generate Equity Curve immediately
            self.trades['equity'] = self.initial_capital + self.trades['pnl'].cumsum()
            self.trades['peak_equity'] = self.trades['equity'].cummax()
            
            # 2. Calculate Percentage Drawdown
            # (Peak - Current) / Peak
            self.trades['dd_pct'] = (self.trades['equity'] - self.trades['peak_equity']) / self.trades['peak_equity']

    def calculate_hft_metrics(self) -> Dict:
        if self.trades.empty:
            return {"Status": "No Trades"}

        pnls = self.trades['pnl']
        equity = self.trades['equity']
        
        # 1. Advanced Risk Metrics
        max_dd_pct = self.trades['dd_pct'].min() # Negative value
        
        # Sortino Ratio (The Sniper's Metric)
        # We assume 0% risk-free rate for intraday
        avg_return = pnls.mean()
        downside_returns = pnls[pnls < 0]
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            sortino = float('inf')
        else:
            sortino = avg_return / downside_std

        # 2. Streak Analysis (Vectorized)
        # Identify blocks of consecutive Wins/Losses
        groups = (pnls > 0).astype(int).diff().ne(0).cumsum()
        streak_lengths = pnls.groupby(groups).cumcount() + 1
        
        max_win_streak = streak_lengths[pnls > 0].max() if not streak_lengths[pnls > 0].empty else 0
        max_loss_streak = streak_lengths[pnls < 0].max() if not streak_lengths[pnls < 0].empty else 0

        # 3. Expectancy & Profit Factor
        gross_win = pnls[pnls > 0].sum()
        gross_loss = abs(pnls[pnls < 0].sum())
        profit_factor = gross_win / gross_loss if gross_loss != 0 else float('inf')
        
        win_rate = len(pnls[pnls > 0]) / len(pnls)
        avg_win = pnls[pnls > 0].mean() if gross_win > 0 else 0
        avg_loss = abs(pnls[pnls < 0].mean()) if gross_loss > 0 else 0
        
        # Real Expectancy (Van Tharp)
        # (Win% * AvgWin) - (Loss% * AvgLoss)
        expectancy_val = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # 4. HFT Specifics
        # Return on Max Drawdown (Calmar Proxy)
        # Net Profit / Max Absolute Drawdown
        abs_dd = (equity - self.trades['peak_equity']).min()
        calmar_proxy = pnls.sum() / abs(abs_dd) if abs_dd != 0 else 0.0

        return {
            "Total Trades": len(self.trades),
            "Net PnL": round(pnls.sum(), 2),
            "Final Equity": round(equity.iloc[-1], 2),
            "Return %": round(((equity.iloc[-1] - self.initial_capital) / self.initial_capital) * 100, 2),
            "Max Drawdown %": f"{max_dd_pct:.2%}",
            "Sortino Ratio": round(sortino, 2),
            "Profit Factor": round(profit_factor, 2),
            "Win Rate": f"{win_rate:.2%}",
            "Expectancy (₹)": round(expectancy_val, 2),
            "Max Win Streak": int(max_win_streak),
            "Max Loss Streak": int(max_loss_streak),
            "Recovery Factor": round(calmar_proxy, 2)
        }

class TrainingResultsAnalyzer:
    def __init__(self, db_path: str = None):
        self.logger = logging.getLogger("MARK5.TrainingAnalyzer")
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.db_path = os.path.join(base_dir, 'database', 'main', 'mark5.db')
        else:
            self.db_path = db_path

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def get_regime_specialists(self) -> pd.DataFrame:
        """
        Finds the best performing model for each market regime.
        """
        try:
            with self._get_conn() as conn:
                # Query Expectancy per Regime per Model
                query = """
                    SELECT 
                        market_regime,
                        model_type,
                        COUNT(*) as runs,
                        AVG(expectancy) as avg_edge,
                        AVG(accuracy) as avg_acc,
                        MAX(timestamp) as last_trained
                    FROM model_performance_history
                    WHERE timestamp >= date('now', '-30 days')
                    GROUP BY market_regime, model_type
                    HAVING runs > 2
                    ORDER BY market_regime, avg_edge DESC
                """
                return pd.read_sql_query(query, conn)
        except Exception as e:
            self.logger.error(f"Regime Analysis Failed: {e}")
            return pd.DataFrame()

    def compare_rounds(self, baseline_days: int = 7) -> pd.DataFrame:
        """
        Did we improve over last week?
        """
        try:
            with self._get_conn() as conn:
                # 1. Baseline Performance (Older than N days)
                base_q = """
                    SELECT model_type, AVG(expectancy) as base_edge
                    FROM model_performance_history
                    WHERE timestamp < date('now', ?)
                    GROUP BY model_type
                """
                
                # 2. Recent Performance (Last N days)
                new_q = """
                    SELECT model_type, AVG(expectancy) as new_edge
                    FROM model_performance_history
                    WHERE timestamp >= date('now', ?)
                    GROUP BY model_type
                """
                
                params = (f"-{baseline_days} days", f"-{baseline_days} days")
                
                base_df = pd.read_sql_query(base_q, conn, params=(f"-{baseline_days} days",))
                new_df = pd.read_sql_query(new_q, conn, params=(f"-{baseline_days} days",))
                
                if base_df.empty or new_df.empty: return pd.DataFrame()
                
                # Merge
                comparison = pd.merge(base_df, new_df, on='model_type', how='inner')
                comparison['improvement'] = comparison['new_edge'] - comparison['base_edge']
                comparison['status'] = comparison['improvement'].apply(
                    lambda x: "✅ IMPROVED" if x > 0 else "⚠️ REGRESSED"
                )
                return comparison
                
        except Exception as e:
            self.logger.error(f"Comparison Failed: {e}")
            return pd.DataFrame()

    def generate_report(self) -> str:
        """Generates the Architect's Report"""
        
        # 1. Specialists
        specialists = self.get_regime_specialists()
        
        # 2. Drift / Improvement
        drift = self.compare_rounds(baseline_days=7)
        
        report = ["\n🏛️ MARK5 ARCHITECT REPORT"]
        report.append("=" * 60)
        
        if not specialists.empty:
            report.append("\n🏆 REGIME SPECIALISTS (Best Model per State):")
            # Group by regime and pick top 1
            best_per_regime = specialists.groupby('market_regime').first().reset_index()
            
            report.append(f"{'Regime':<20} {'Best Model':<15} {'Expectancy':<10} {'Accuracy':<10}")
            report.append("-" * 60)
            for _, row in best_per_regime.iterrows():
                report.append(f"{row['market_regime']:<20} {row['model_type']:<15} {row['avg_edge']:.2f}       {row['avg_acc']:.1%}")
        
        if not drift.empty:
            report.append("\n📈 WEEKLY PERFORMANCE DELTA:")
            report.append(f"{'Model':<15} {'Old Edge':<10} {'New Edge':<10} {'Status':<15}")
            report.append("-" * 60)
            for _, row in drift.iterrows():
                report.append(f"{row['model_type']:<15} {row['base_edge']:.2f}       {row['new_edge']:.2f}       {row['status']}")
                
        report.append("=" * 60)
        return "\n".join(report)

if __name__ == "__main__":
    analyzer = TrainingResultsAnalyzer()
    print(analyzer.generate_report())

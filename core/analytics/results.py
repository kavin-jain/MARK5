"""
MARK5 HFT Results Analyzer v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Production hardening & standardized header

TRADING ROLE: Analyzes trade results for drawdown, streaks, profit factors
SAFETY LEVEL: MEDIUM - Informs risk decisions

FEATURES:
✅ Win/Loss streak analysis (vectorized)
✅ Drawdown duration tracking
✅ Profit factor & expectancy calculation
"""

import pandas as pd
import numpy as np
from typing import Dict

class ResultsAnalyzer:
    def __init__(self, trades_df: pd.DataFrame):
        self.trades = trades_df.copy()
        if not self.trades.empty:
            self.trades['timestamp'] = pd.to_datetime(self.trades['timestamp'])
            self.trades.sort_values('timestamp', inplace=True)

    def calculate_hft_metrics(self) -> Dict:
        if self.trades.empty:
            return {}

        pnls = self.trades['pnl']
        
        # 1. Win/Loss Streaks
        # Vectorized streak calculation
        # Group by consecutive wins/losses
        groups = (pnls > 0).astype(int).diff().ne(0).cumsum()
        # Calculate streak lengths
        streak_lengths = pnls.groupby(groups).cumcount() + 1
        
        # Filter for win streaks (pnl > 0) and loss streaks (pnl < 0)
        win_streaks = streak_lengths[pnls > 0]
        loss_streaks = streak_lengths[pnls < 0]
        
        max_win_streak = win_streaks.max() if not win_streaks.empty else 0
        max_loss_streak = loss_streaks.max() if not loss_streaks.empty else 0

        # 2. Drawdown Analysis
        cum_pnl = pnls.cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        max_drawdown = drawdown.min()
        
        # Time underwater (Drawdown Duration)
        is_underwater = drawdown < 0
        # Calculate duration of continuous underwater periods
        underwater_blocks = is_underwater.astype(int).groupby(is_underwater.astype(int).diff().ne(0).cumsum()).cumsum()
        max_drawdown_duration_trades = underwater_blocks.max()

        # 3. Profit Factor & Expectancy
        gross_profit = pnls[pnls > 0].sum()
        gross_loss = abs(pnls[pnls < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        expectancy = pnls.mean()
        
        return {
            "Total Trades": len(self.trades),
            "Net PnL": pnls.sum(),
            "Profit Factor": round(profit_factor, 2),
            "Win Rate": f"{(len(pnls[pnls > 0]) / len(pnls)):.2%}",
            "Max Win Streak": int(max_win_streak),
            "Max Loss Streak": int(abs(max_loss_streak)), # Absolute value
            "Max Drawdown": round(max_drawdown, 2),
            "Max Drawdown Duration (Trades)": int(max_drawdown_duration_trades) if not pd.isna(max_drawdown_duration_trades) else 0,
            "Expectancy per Trade": round(expectancy, 2)
        }

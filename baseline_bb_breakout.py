"""
MARK5 Baseline Strategy: Bollinger Band Breakout v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRADING ROLE: Non-ML Baseline for alpha comparison.
Calculates how much return is due to simple momentum vs. ML.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("MARK5.Baseline")

# Ensure project root is in path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.models.tcn.backtester import RobustBacktester, Trade, IndianTaxConfig

# Comparison Universe (Elite Stocks)
TICKERS = [
    "IRFC.NS", "IOB.NS", "M&M.NS", "CONCOR.NS", "AARTIIND.NS",
    "BEL.NS", "IRCON.NS", "HUDCO.NS", "BRIGADE.NS", "DLF.NS"
]

OOS_START = "2025-04-01"

class BBBreakoutBacktester(RobustBacktester):
    """Bollinger Band Breakout (Non-ML)"""
    
    def run_bb_simulation(self, df: pd.DataFrame, window: int = 20, std_dev: float = 2.0):
        # 1. Compute Bands
        df = df.copy()
        df['MA20'] = df['close'].rolling(window).mean()
        df['STD20'] = df['close'].rolling(window).std()
        df['Upper'] = df['MA20'] + (std_dev * df['STD20'])
        df['Lower'] = df['MA20'] - (std_dev * df['STD20'])
        df['ATR'] = self._calculate_atr(df)
        
        equity = float(self.initial_capital)
        position = 0
        entry_price = 0.0
        entry_idx = 0
        trades = []
        equity_curve = pd.Series(float(self.initial_capital), index=df.index, dtype='float64')
        
        # OOS Mask
        oos_mask = df.index >= pd.Timestamp(OOS_START)
        
        for i in range(1, len(df)):
            if not oos_mask[i]:
                continue
                
            curr_open = df['open'].iloc[i]
            curr_high = df['high'].iloc[i]
            curr_low = df['low'].iloc[i]
            curr_close = df['close'].iloc[i]
            curr_atr = df['ATR'].iloc[i-1]
            
            # Simple BB Logic
            prev_close = df['close'].iloc[i-1]
            prev_upper = df['Upper'].iloc[i-1]
            prev_ma    = df['MA20'].iloc[i-1]
            
            # EXIT LOGIC
            if position != 0:
                exit_now = False
                exit_p = 0.0
                reason = ""
                
                # Rule: Exit if close below MA20 or Max Hold 15 days
                hold_days = i - entry_idx
                if hold_days >= 15:
                    exit_now = True; exit_p = curr_open; reason = "MAX_HOLD"
                elif prev_close < prev_ma:
                    exit_now = True; exit_p = curr_open; reason = "MA_EXIT"
                
                if exit_now:
                    turnover = (abs(position) * entry_price) + (abs(position) * exit_p)
                    charges = self.tax_engine.calculate_charges(
                        buy_val=abs(position) * entry_price,
                        sell_val=abs(position) * exit_p,
                        turnover=turnover
                    )
                    net_pnl = ((exit_p - entry_price) * position) - charges['total']
                    equity += net_pnl
                    trades.append(Trade(df.index[entry_idx], df.index[i], "BASELINE", 'LONG', entry_price, exit_p, abs(position), (exit_p - entry_price) * position, net_pnl, charges, reason, i - entry_idx))
                    position = 0

            # ENTRY LOGIC
            if position == 0:
                # Signal: Close broke above upper band
                if prev_close > prev_upper:
                    fill_p = curr_open * (1 + self.slippage)
                    risk_amt = equity * 0.05 # 5% risk
                    stop_dist = curr_atr * 1.5
                    qty = int(risk_amt / stop_dist) if stop_dist > 0 else 0
                    if qty * fill_p > equity: qty = int(equity / fill_p)
                    
                    if qty > 0:
                        position = qty
                        entry_price = fill_p
                        entry_idx = i
            
            equity_curve.iloc[i] = equity
            
        return equity_curve, trades

def run_baseline(mark5_avg: float = None):
    results = []
    bt = BBBreakoutBacktester(initial_capital=100000, segment='EQUITY_DELIVERY')
    
    for ticker in TICKERS:
        base = ticker.replace('.NS', '')
        path = f"data/cache/{base}_NS_1d.parquet"
        if not os.path.exists(path): path = f"data/cache/{base.replace('.', '_')}_NS_1d.parquet"
        if not os.path.exists(path): continue
        
        df = pd.read_parquet(path)
        _, trades = bt.run_bb_simulation(df)
        
        ret = sum(t.net_pnl for t in trades) / 1000
        results.append({"Ticker": ticker, "Return %": ret, "Trades": len(trades)})
        
    res_df = pd.DataFrame(results)
    avg_ret = res_df["Return %"].mean()
    
    md = [
        "# MARK5 Baseline Report: Bollinger Band Breakout",
        f"**Universe:** Top 10 performant stocks",
        f"**Period:** {OOS_START} → Present",
        f"**Strategy:** Pure BB Breakout (No ML)",
        "",
        f"| Average Baseline Return | {avg_ret:.2f}% |",
        "| :--- | :--- |"
    ]
    
    if mark5_avg:
        alpha = mark5_avg - avg_ret
        md.append(f"| MARK5 Ensemble Return | {mark5_avg:.2f}% |")
        md.append(f"| **Genuine ML Alpha** | **{alpha:+.2f}%** |")
        
    md.append("\n## Per-Stock Baseline")
    md.append("| Ticker | Return % | Trades |")
    md.append("| :--- | :--- | :--- |")
    for _, row in res_df.iterrows():
        md.append(f"| {row['Ticker']} | {row['Return %']:.2f}% | {row['Trades']} |")
        
    report = "\n".join(md)
    print(report)
    with open("reports/baseline_bb_breakout.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mark5-avg", type=float, help="Known MARK5 average for alpha comparison")
    args = parser.parse_args()
    run_baseline(args.mark5_avg)

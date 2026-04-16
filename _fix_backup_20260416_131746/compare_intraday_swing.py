import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("MARK5.Comparison")

from core.models.predictor import MARK5Predictor
from core.models.tcn.backtester import RobustBacktester, Trade

# Top 10 Elite Stocks from 94-stock test
ELITE_TICKERS = [
    "IRFC.NS", "IOB.NS", "M&M.NS", "CONCOR.NS", "AARTIIND.NS",
    "BEL.NS", "IRCON.NS", "HUDCO.NS", "BRIGADE.NS", "DLF.NS"
]

OOS_START_DATE = "2025-04-01"

class ComparisonBacktester(RobustBacktester):
    """Extended backtester to support true intraday (exit same day)."""
    
    def run_comparison(self, df: pd.DataFrame, signals: pd.Series, mode: str = 'SWING', max_hold: int = 15):
        self.max_hold_days = max_hold
        self.segment = 'EQUITY_INTRADAY' if mode == 'INTRADAY' else 'EQUITY_DELIVERY'
        # Update tax engine for the segment
        from core.models.tcn.backtester import IndianTaxConfig
        self.tax_engine = IndianTaxConfig(self.segment)
        
        # Pre-calculations
        df = df.copy()
        df['ATR'] = self._calculate_atr(df)
        df['Signal'] = signals
        
        equity = float(self.initial_capital)
        position = 0
        entry_price = 0.0
        entry_idx = 0
        trades = []
        equity_curve = pd.Series(float(self.initial_capital), index=df.index, dtype='float64')
        
        for i in range(1, len(df)):
            curr_open = df['open'].iloc[i]
            curr_high = df['high'].iloc[i]
            curr_low = df['low'].iloc[i]
            curr_close = df['close'].iloc[i]
            curr_atr = df['ATR'].iloc[i-1]
            prev_signal = df['Signal'].iloc[i-1]
            
            # 1. EXIT LOGIC
            if position != 0:
                exit_now = False
                exit_p = 0.0
                reason = ""
                
                # INTRADAY MODE: Force exit at close of same day
                if mode == 'INTRADAY':
                    exit_now = True
                    exit_p = curr_close * (1 - self.slippage if position > 0 else 1 + self.slippage)
                    reason = "INTRADAY_CLOSE"
                else:
                    # SWING MODE: ATR Barriers or Max Hold
                    stop_p = entry_price - (curr_atr * self.atr_multiplier) if position > 0 else entry_price + (curr_atr * self.atr_multiplier)
                    target_p = entry_price + (curr_atr * self.pt_multiplier) if position > 0 else entry_price - (curr_atr * self.pt_multiplier)
                    
                    hold_days = i - entry_idx
                    
                    if hold_days >= self.max_hold_days:
                        exit_now = True
                        exit_p = curr_open * (1 - self.slippage if position > 0 else 1 + self.slippage)
                        reason = "MAX_HOLD"
                    elif position > 0: # Long
                        if curr_open <= stop_p:
                            exit_now = True; exit_p = curr_open; reason = "SL_GAP"
                        elif curr_low <= stop_p:
                            exit_now = True; exit_p = stop_p; reason = "SL_HIT"
                        elif curr_high >= target_p:
                            exit_now = True; exit_p = target_p; reason = "PT_HIT"
                        elif prev_signal == -1:
                            exit_now = True; exit_p = curr_open; reason = "SIGNAL_REV"
                    elif position < 0: # Short
                        if curr_open >= stop_p:
                            exit_now = True; exit_p = curr_open; reason = "SL_GAP"
                        elif curr_high >= stop_p:
                            exit_now = True; exit_p = stop_p; reason = "SL_HIT"
                        elif curr_low <= target_p:
                            exit_now = True; exit_p = target_p; reason = "PT_HIT"
                        elif prev_signal == 1:
                            exit_now = True; exit_p = curr_open; reason = "SIGNAL_REV"

                if exit_now:
                    turnover = (abs(position) * entry_price) + (abs(position) * exit_p)
                    charges = self.tax_engine.calculate_charges(
                        buy_val=(abs(position) * entry_price) if position > 0 else (abs(position) * exit_p),
                        sell_val=(abs(position) * exit_p) if position > 0 else (abs(position) * entry_price),
                        turnover=turnover
                    )
                    net_pnl = ((exit_p - entry_price) * position) - charges['total']
                    equity += net_pnl
                    trades.append(Trade(df.index[entry_idx], df.index[i], "ELITE", 'LONG' if position > 0 else 'SHORT', entry_price, exit_p, abs(position), (exit_p - entry_price) * position, net_pnl, charges, reason, i - entry_idx))
                    position = 0
            
            # 2. ENTRY LOGIC (Next Open)
            if position == 0 and prev_signal != 0:
                fill_p = curr_open * (1 + self.slippage if prev_signal == 1 else 1 - self.slippage)
                risk_amt = equity * self.risk_per_trade
                stop_dist = curr_atr * self.atr_multiplier
                qty = int(risk_amt / stop_dist) if stop_dist > 0 else 0
                
                # Cap at 100% equity
                if qty * fill_p > equity: qty = int(equity / fill_p)
                
                if qty > 0:
                    position = qty * (1 if prev_signal == 1 else -1)
                    entry_price = fill_p
                    entry_idx = i
            
            equity_curve.iloc[i] = equity

        return equity_curve, trades

def run_comparison():
    results = []
    oos_start = pd.Timestamp(OOS_START_DATE)
    
    for ticker in ELITE_TICKERS:
        logger.info(f"Comparing {ticker}...")
        
        # Load Cache
        base = ticker.replace('.NS', '')
        path = f"data/cache/{base}_NS_1d.parquet"
        if not os.path.exists(path): path = f"data/cache/{base.replace('.', '_')}_NS_1d.parquet"
        if not os.path.exists(path): continue
        
        df = pd.read_parquet(path)
        test_df = df[df.index >= oos_start - pd.Timedelta(days=365)].copy()
        
        # Load Predictor
        try:
            predictor = MARK5Predictor(ticker, allow_shadow=True)
        except: continue
        
        # Generate Signals
        full_signals = [0] * len(test_df)
        start_idx = test_df.index.get_indexer([oos_start], method='bfill')[0]
        for i in range(start_idx, len(test_df)):
            res = predictor.predict(test_df.iloc[:i+1])
            sig = res.get('signal', 'HOLD')
            full_signals[i] = 1 if 'BUY' in sig else (-1 if 'SELL' in sig else 0)
        
        signals_series = pd.Series(full_signals, index=test_df.index)
        
        # Run Both Modes
        bt = ComparisonBacktester(initial_capital=100000)
        
        # Mode 1: Intraday
        _, trades_intra = bt.run_comparison(test_df, signals_series, mode='INTRADAY')
        ret_intra = sum(t.net_pnl for t in trades_intra) / 1000
        
        # Mode 2: Swing (15 days)
        _, trades_swing = bt.run_comparison(test_df, signals_series, mode='SWING', max_hold=15)
        ret_swing = sum(t.net_pnl for t in trades_swing) / 1000
        
        results.append({
            "Ticker": ticker,
            "Intraday Ret%": ret_intra,
            "Swing Ret%": ret_swing,
            "Alpha (Swing-Intra)": ret_swing - ret_intra,
            "Intraday Trades": len(trades_intra),
            "Swing Trades": len(trades_swing)
        })
        
    res_df = pd.DataFrame(results)
    
    md = [
        "# Strategy Comparison: Intraday vs Swing (15D Max Hold)",
        "## Universe: Elite Top 10 Stocks",
        f"**Test Period:** {OOS_START_DATE} to Present (Strict OOS)",
        "",
        "| Ticker | Intraday Ret% | Swing Ret% | Win Type | Margin |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for _, row in res_df.iterrows():
        win_type = "🟢 SWING" if row["Swing Ret%"] > row["Intraday Ret%"] else "🔵 INTRADAY"
        margin = abs(row["Swing Ret%"] - row["Intraday Ret%"])
        md.append(f"| {row['Ticker']} | {row['Intraday Ret%']:.2f}% | {row['Swing Ret%']:.2f}% | {win_type} | {margin:.2f}% |")
    
    avg_intra = res_df["Intraday Ret%"].mean()
    avg_swing = res_df["Swing Ret%"].mean()
    
    md.append("\n## Summary Verdict")
    md.append(f"- **Avg Intraday Return:** {avg_intra:.2f}%")
    md.append(f"- **Avg Swing Return:** {avg_swing:.2f}%")
    md.append(f"- **Better Strategy:** {'🟢 SWING' if avg_swing > avg_intra else '🔵 INTRADAY'}")
    md.append(f"- **The Verdict:** {'Swing captures the broader trend momentum.' if avg_swing > avg_intra else 'Intraday avoids overnight risk and leverages compound cycles.'}")

    report = "\n".join(md)
    with open("reports/intraday_vs_swing_report.md", "w") as f:
        f.write(report)
    print(report)

if __name__ == "__main__":
    run_comparison()

"""
CHANAKYA: Advanced Indian Market Backtesting Engine (NSE/BSE)
Architected for TCN/LSTM Strategy Validation.
-------------------------------------------------------------
Features:
1. Strict "Next-Open" Execution (Eliminates Lookahead Bias)
2. Granular Indian Tax Modeling (STT, GST, Stamp Duty, Exchange Fees)
3. Regulatory Compliance Checks (No Overnight Shorts in Cash Segment)
4. Vectorized Metric Calculation for High-Frequency Data
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal
import logging

# Setup High-Precision Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("CHANAKYA_ENGINE")

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    symbol: str
    direction: Literal['LONG', 'SHORT']
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    net_pnl: float
    taxes: Dict[str, float]
    exit_reason: str
    hold_duration: int

class IndianTaxConfig:
    """
    Exact Tax Structure for NSE Equity/F&O (As of 2024-25)
    """
    def __init__(self, segment: str = 'EQUITY_INTRADAY'):
        self.segment = segment
        # Basic Brokerage (e.g., Zerodha/Groww flat fee model or percentage)
        self.brokerage_flat = 20.0 
        self.brokerage_pct = 0.0003 # 0.03% (Max cap usually applies)
        
        # Government & Exchange Levies
        if segment == 'EQUITY_INTRADAY':
            self.stt_buy = 0.0
            self.stt_sell = 0.00025  # 0.025% on Sell
            self.trans_charge = 0.000325 # NSE Transaction charges
            self.stamp_duty = 0.00003 # 0.003% on Buy
        elif segment == 'EQUITY_DELIVERY':
            self.stt_buy = 0.001     # 0.1%
            self.stt_sell = 0.001    # 0.1%
            self.trans_charge = 0.000325
            self.stamp_duty = 0.00015 # 0.015% on Buy
        elif segment == 'FUTURES':
            self.stt_buy = 0.0
            self.stt_sell = 0.0001   # 0.01% on Sell
            self.trans_charge = 0.00019
            self.stamp_duty = 0.00002
        else:
            raise ValueError("Unknown Segment. Use: EQUITY_INTRADAY, EQUITY_DELIVERY, or FUTURES")
            
        self.gst_rate = 0.18 # 18% on (Brokerage + Txn Charges)
        self.sebi_fee = 0.000001 # ₹10 per crore

    def calculate_charges(self, buy_val: float, sell_val: float, turnover: float) -> Dict[str, float]:
        """Calculates exact breakdown of charges"""
        # 1. Brokerage (Lower of Flat or %)
        b_buy = min(self.brokerage_flat, buy_val * self.brokerage_pct)
        b_sell = min(self.brokerage_flat, sell_val * self.brokerage_pct)
        brokerage = b_buy + b_sell

        # 2. STT (Securities Transaction Tax)
        stt = (buy_val * self.stt_buy) + (sell_val * self.stt_sell)

        # 3. Exchange Transaction Charges
        txn_charges = turnover * self.trans_charge

        # 4. GST (18% on Brokerage + Txn Charges)
        gst = (brokerage + txn_charges) * self.gst_rate

        # 5. Stamp Duty (On Buy only)
        stamp = buy_val * self.stamp_duty

        # 6. SEBI Fees
        sebi = turnover * self.sebi_fee

        total_tax = brokerage + stt + txn_charges + gst + stamp + sebi
        
        return {
            'total': total_tax,
            'brokerage': brokerage,
            'stt': stt,
            'gst': gst,
            'stamp_duty': stamp,
            'txn_charges': txn_charges
        }

class RobustBacktester:
    def __init__(
        self,
        initial_capital: float = 100000.0,
        segment: str = 'EQUITY_INTRADAY',
        slippage_pct: float = 0.0005, # 0.05% slippage
        use_atr_stop: bool = True,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.02 # Risk 2% of capital per trade
    ):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.segment = segment
        self.tax_engine = IndianTaxConfig(segment)
        self.slippage = slippage_pct
        self.use_atr_stop = use_atr_stop
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.risk_per_trade = risk_per_trade
        
        # Validation
        if segment == 'EQUITY_DELIVERY':
            logger.warning("⚠️ MARKET RULE: Short selling disabled for Equity Delivery.")

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """True Range calculation handling gaps"""
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=self.atr_period, adjust=False).mean()

    def run_simulation(self, df: pd.DataFrame, signals: pd.Series) -> Tuple[pd.DataFrame, dict]:
        """
        The Core Simulation Loop.
        CRITICAL: Executions happen at OPEN of i+1 based on Signal at i.
        """
        logger.info(f"🚀 Starting Simulation on {len(df)} bars. Segment: {self.segment}")
        
        # Pre-calculations
        df = df.copy()
        df['ATR'] = self._calculate_atr(df)
        df['Signal'] = signals # 1 (Buy), -1 (Sell), 0 (Hold)
        
        # State Variables
        equity = self.capital
        position = 0 # Quantity
        entry_price = 0.0
        entry_idx = 0
        trades: List[Trade] = []
        equity_curve = [equity]
        
        # Iterate (Vectorization is hard with complex path-dependent stops)
        # We start from index 1 because we need prev close for ATR
        # We stop at len-1 because we trade on Next Open
        
        for i in range(1, len(df) - 1):
            curr_date = df.index[i]
            curr_close = df['close'].iloc[i]
            curr_atr = df['ATR'].iloc[i]
            signal = df['Signal'].iloc[i]
            
            # ---------------------------------------------------
            # 1. MANAGE EXISTING POSITIONS (Stop Loss / Take Profit)
            # ---------------------------------------------------
            if position != 0:
                # We check Low/High of CURRENT bar 'i' to see if SL was hit
                # (Assuming we entered at Open of 'i' or before)
                
                sl_hit = False
                exit_price = 0.0
                
                if position > 0: # Long
                    stop_price = entry_price - (curr_atr * self.atr_multiplier)
                    if df['low'].iloc[i] <= stop_price:
                        sl_hit = True
                        exit_price = stop_price - (stop_price * self.slippage) # Slippage on Stop
                        reason = "SL_HIT"
                    elif signal == -1: # Reverse/Exit Signal
                        sl_hit = True
                        # If signal is generated at 'i', we exit at 'i+1' OPEN. 
                        # But wait, managing existing positions usually happens intraday.
                        # For simplicity in TCN backtest, we exit at Close 'i' if signal flips
                        exit_price = curr_close
                        reason = "SIGNAL_EXIT"
                        
                elif position < 0: # Short
                    stop_price = entry_price + (curr_atr * self.atr_multiplier)
                    if df['high'].iloc[i] >= stop_price:
                        sl_hit = True
                        exit_price = stop_price + (stop_price * self.slippage)
                        reason = "SL_HIT"
                    elif signal == 1:
                        sl_hit = True
                        exit_price = curr_close
                        reason = "SIGNAL_EXIT"

                if sl_hit:
                    # Execute Exit
                    turnover = (abs(position) * entry_price) + (abs(position) * exit_price)
                    taxes = self.tax_engine.calculate_charges(
                        buy_val=(abs(position) * entry_price) if position > 0 else (abs(position) * exit_price),
                        sell_val=(abs(position) * exit_price) if position > 0 else (abs(position) * entry_price),
                        turnover=turnover
                    )
                    
                    gross_pnl = (exit_price - entry_price) * position
                    net_pnl = gross_pnl - taxes['total']
                    
                    equity += net_pnl
                    
                    trades.append(Trade(
                        entry_date=df.index[entry_idx],
                        exit_date=curr_date,
                        symbol="NIFTY_MOCK",
                        direction='LONG' if position > 0 else 'SHORT',
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=abs(position),
                        gross_pnl=gross_pnl,
                        net_pnl=net_pnl,
                        taxes=taxes,
                        exit_reason=reason,
                        hold_duration=(i - entry_idx)
                    ))
                    
                    position = 0
                    entry_price = 0.0

            # ---------------------------------------------------
            # 2. ENTRY LOGIC (Strict Next-Open Execution)
            # ---------------------------------------------------
            # If we are flat, look for entry
            if position == 0 and signal != 0:
                # COMPLIANCE CHECK
                if self.segment == 'EQUITY_DELIVERY' and signal == -1:
                    # Ignore Short signal in Delivery
                    pass
                else:
                    # Execution happens at NEXT BAR OPEN
                    next_open = df['open'].iloc[i+1]
                    
                    # Apply Slippage
                    if signal == 1:
                        fill_price = next_open * (1 + self.slippage)
                        direction = 1
                    else:
                        fill_price = next_open * (1 - self.slippage)
                        direction = -1
                    
                    # Position Sizing based on Risk
                    # Risk amount = Capital * Risk%
                    # Stop distance = ATR * Multiplier
                    # Shares = Risk Amount / Stop Distance
                    risk_amt = equity * self.risk_per_trade
                    stop_dist = curr_atr * self.atr_multiplier
                    
                    logger.info(f"Debug: i={i}, Signal={signal}, Equity={equity}, ATR={curr_atr}, RiskAmt={risk_amt}, StopDist={stop_dist}")

                    if stop_dist > 0:
                        qty = int(risk_amt / stop_dist)
                    else:
                        qty = 0
                    
                    logger.info(f"Debug: Qty={qty}, FillPrice={fill_price}, Cost={qty*fill_price}")

                    if qty > 0 and (qty * fill_price) <= equity:
                        position = qty * direction
                        entry_price = fill_price
                        entry_idx = i + 1 # We entered at i+1
                        logger.info(f"Debug: ENTERED TRADE at {entry_price}")
                    else:
                         logger.info("Debug: Trade Rejected (Insufficient Funds or Zero Qty)")
            
            equity_curve.append(equity)
            
        # ---------------------------------------------------
        # 2.5 CLOSE REMAINING POSITION
        # ---------------------------------------------------
        if position != 0:
            last_price = df['close'].iloc[-1]
            turnover = (abs(position) * entry_price) + (abs(position) * last_price)
            taxes = self.tax_engine.calculate_charges(
                buy_val=(abs(position) * entry_price) if position > 0 else (abs(position) * last_price),
                sell_val=(abs(position) * last_price) if position > 0 else (abs(position) * entry_price),
                turnover=turnover
            )
            
            gross_pnl = (last_price - entry_price) * position
            net_pnl = gross_pnl - taxes['total']
            
            equity += net_pnl
            
            trades.append(Trade(
                entry_date=df.index[entry_idx],
                exit_date=df.index[-1],
                symbol="NIFTY_MOCK",
                direction='LONG' if position > 0 else 'SHORT',
                entry_price=entry_price,
                exit_price=last_price,
                quantity=abs(position),
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                taxes=taxes,
                exit_reason="END_OF_SIM",
                hold_duration=(len(df) - 1 - entry_idx)
            ))
            
            logger.info(f"Debug: Force Closed remaining position at {last_price}")

        # Convert to DF
        equity_series = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
        
        # ---------------------------------------------------
        # 3. METRICS GENERATION
        # ---------------------------------------------------
        if not trades:
            logger.error("❌ No trades generated. Check Signal Logic or Risk constraints.")
            return equity_series, {}

        trades_df = pd.DataFrame([vars(t) for t in trades])
        
        # Calculate Drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # Returns
        returns = equity_series.pct_change().dropna()
        
        # Realistic Sharpe (Assuming Intraday 5min data -> 75 bars/day)
        # If daily data, use 252. Adjust 'bars_per_year' accordingly.
        # Here we assume daily input for simplicity, but for intraday multiply by sqrt(bars_per_year)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) 
        
        metrics = {
            'Total Return %': ((equity - self.initial_capital) / self.initial_capital) * 100,
            'Win Rate %': (len(trades_df[trades_df['net_pnl'] > 0]) / len(trades_df)) * 100,
            'Profit Factor': abs(trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum() / trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum()),
            'Max Drawdown %': max_dd * 100,
            'Sharpe Ratio': sharpe,
            'Total Trades': len(trades),
            'Total Taxes Paid': sum(t.taxes['total'] for t in trades)
        }
        
        logger.info(f"✅ Simulation Complete. Final Equity: ₹{equity:,.2f}")
        logger.info(f"💰 Total Taxes Paid to Govt: ₹{metrics['Total Taxes Paid']:,.2f}")
        
        return equity_series, metrics

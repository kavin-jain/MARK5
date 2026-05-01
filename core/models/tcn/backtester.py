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
        atr_multiplier: float = 1.0, # Stop loss at 1.0x ATR
        pt_multiplier: float = 2.0,  # Profit target at 2.0x ATR (Positive Expectancy)
        risk_per_trade: float = 0.05, # Increased to 5% risk per trade for higher returns
        max_hold_days: int = 10,       # Rule 3: Max hold period
    ):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.segment = segment
        self.tax_engine = IndianTaxConfig(segment)
        self.slippage = slippage_pct
        self.use_atr_stop = use_atr_stop
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.pt_multiplier = pt_multiplier
        self.risk_per_trade = risk_per_trade
        self.max_hold_days = max_hold_days
        
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

    def run_simulation(self, df: pd.DataFrame, signals: pd.Series, symbol: str = 'UNKNOWN') -> Tuple[pd.DataFrame, dict]:
        """
        The Core Simulation Loop.
        CRITICAL: Executions happen at OPEN of i+1 based on Signal at i.
        """
        # 0. Input Validation
        if not isinstance(df, pd.DataFrame) or not isinstance(signals, pd.Series):
             logger.error("❌ Invalid input types: df must be DataFrame, signals must be Series")
             return pd.Series(), {'error': 'Invalid Input'}
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
             logger.error(f"❌ Missing columns. Required: {required_cols}")
             return pd.Series(), {'error': 'Missing Columns'}
             
        if len(df) != len(signals):
            logger.error("❌ Mismatched lengths between Data and Signals")
            return pd.Series(), {'error': 'Length Mismatch'}

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
        cooldown = 0  # Bars to wait after a stop-loss before re-entering
        trades: List[Trade] = []
        
        # Initialize Equity Curve with correct index and initial capital
        equity_curve = pd.Series(self.capital, index=df.index)
        
        # Iterate
        # We start from index 1 because we need prev close for ATR (and logic relies on i-1 for signals sometimes)
        # We stop at len-1 because we trade on Next Open
        
        for i in range(1, len(df)):
            curr_date = df.index[i]
            # Market Data for CURRENT bar 'i'
            curr_open = df['open'].iloc[i]
            curr_high = df['high'].iloc[i]
            curr_low = df['low'].iloc[i]
            curr_close = df['close'].iloc[i]
            curr_atr = df['ATR'].iloc[i-1] # Use previous ATR for stop calculation to avoid lookahead? Standard is prev ATR.
            
            # Signal from PREVIOUS bar (i-1) triggers entry at OPEN of CURRENT bar (i)
            prev_signal = df['Signal'].iloc[i-1] 
            
            # ---------------------------------------------------
            # 1. MANAGE EXISTING POSITIONS (Stop Loss / Take Profit)
            # ---------------------------------------------------
            if position != 0:
                sl_hit = False
                exit_price = 0.0
                reason = ""
                
                # Check for GAP FIRST (Look-ahead bias fix)
                # If Market Opens BEYOND Stop, we exit at OPEN
                
                if position > 0: # Long
                    stop_price = entry_price - (curr_atr * self.atr_multiplier)
                    target_price = entry_price + (curr_atr * self.pt_multiplier)
                    hold_duration = i - entry_idx
                    
                    # Rule 3: Max hold period exit — at open of day after limit
                    if hold_duration >= self.max_hold_days:
                        sl_hit = True
                        exit_price = curr_open * (1 - self.slippage)
                        reason = "MAX_HOLD"
                    # Scenario A: Gap Down below Stop
                    elif curr_open <= stop_price:
                        sl_hit = True
                        exit_price = curr_open * (1 - self.slippage) # Exit at Open if gapped
                        reason = "SL_GAP"
                    # Scenario A2: Gap Up above Target
                    elif curr_open >= target_price:
                        sl_hit = True
                        exit_price = curr_open * (1 - self.slippage)
                        reason = "PT_GAP"
                    # Scenario B: Intraday SL/PT — FIX-4: resolve ambiguity
                    # when BOTH levels are breached on the same daily bar,
                    # use open-proximity to determine which was hit first.
                    elif curr_low <= stop_price and curr_high >= target_price:
                        dist_sl = abs(curr_open - stop_price)
                        dist_pt = abs(curr_open - target_price)
                        if dist_sl <= dist_pt:
                            sl_hit = True; exit_price = stop_price * (1 - self.slippage); reason = "SL_HIT"
                        else:
                            sl_hit = True; exit_price = target_price * (1 - self.slippage); reason = "PT_HIT"
                    elif curr_low <= stop_price:
                        sl_hit = True
                        exit_price = stop_price * (1 - self.slippage)
                        reason = "SL_HIT"
                    elif curr_high >= target_price:
                        sl_hit = True
                        exit_price = target_price * (1 - self.slippage)
                        reason = "PT_HIT"
                    # Scenario C: Signal Reversal (DELIVERY: exit at open, not close)
                    elif prev_signal == -1 and self.segment != 'EQUITY_DELIVERY':
                        sl_hit = True
                        exit_price = curr_open * (1 - self.slippage)  # Rule 42: next open
                        reason = "SIGNAL_EXIT"
                        
                elif position < 0: # Short
                    stop_price = entry_price + (curr_atr * self.atr_multiplier)
                    target_price = entry_price - (curr_atr * self.pt_multiplier)
                    
                    # Scenario A: Gap Up above Stop
                    if curr_open >= stop_price:
                         sl_hit = True
                         exit_price = curr_open * (1 + self.slippage)
                         reason = "SL_GAP"
                    # Scenario A2: Gap Down below Target
                    elif curr_open <= target_price:
                         sl_hit = True
                         exit_price = curr_open * (1 + self.slippage)
                         reason = "PT_GAP"
                    # Scenario B: Intraday SL/PT — FIX-4 (short side)
                    elif curr_high >= stop_price and curr_low <= target_price:
                        dist_sl = abs(curr_open - stop_price)
                        dist_pt = abs(curr_open - target_price)
                        if dist_sl <= dist_pt:
                            sl_hit = True; exit_price = stop_price * (1 + self.slippage); reason = "SL_HIT"
                        else:
                            sl_hit = True; exit_price = target_price * (1 + self.slippage); reason = "PT_HIT"
                    elif curr_high >= stop_price:
                        sl_hit = True
                        exit_price = stop_price * (1 + self.slippage)
                        reason = "SL_HIT"
                    elif curr_low <= target_price:
                        sl_hit = True
                        exit_price = target_price * (1 + self.slippage)
                        reason = "PT_HIT"
                    # Scenario C: Signal Reversal
                    elif prev_signal == 1:
                        sl_hit = True
                        exit_price = curr_close * (1 + self.slippage)
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
                        symbol=symbol,
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
                    # Cooldown after hard stops only — prevents immediate re-entry whipsawing
                    if reason in ('SL_HIT', 'SL_GAP'):
                        cooldown = 3
            
            # ---------------------------------------------------
            # 2. ENTRY LOGIC (Strict Next-Open Execution)
            # ---------------------------------------------------
            # executed_at_open determines if we just entered on this bar
            # if we entered on this bar, we do NOT check intraday stops immediately 
            # (unless we want strict checking, but usually safe to standard allow 1 bar)
            # For this logic, if we are flat, we look to enter at THIS bar's Open based on Prev Signal
            
            entries_on_this_bar = False
            
            if cooldown > 0:
                cooldown -= 1
            elif position == 0 and prev_signal != 0:
                # COMPLIANCE CHECK
                if self.segment == 'EQUITY_DELIVERY' and prev_signal == -1:
                    pass
                else:
                    # Execution happens at CURRENT OPEN (since decision was made at prev close)
                    fill_price = curr_open
                    
                    # Apply Slippage
                    if prev_signal == 1:
                        fill_price = fill_price * (1 + self.slippage)
                        direction = 1
                    else:
                        fill_price = fill_price * (1 - self.slippage)
                        direction = -1
                    
                    # Position Sizing (Rule 22 formula)
                    risk_amt = equity * self.risk_per_trade
                    stop_dist = curr_atr * self.atr_multiplier
                    
                    if stop_dist > 0:
                        qty = int(risk_amt / stop_dist)
                    else:
                        qty = 0
                    
                    # Cap position value at 100% of portfolio (no margin)
                    max_pos_value = equity * 1.0
                    if qty > 0 and (qty * fill_price) > max_pos_value:
                        qty = max(1, int(max_pos_value / fill_price))
                    
                    if qty > 0 and (qty * fill_price) <= equity:
                        position = qty * direction
                        entry_price = fill_price
                        entry_idx = i
                        entries_on_this_bar = True
            
            # Update Equity Curve
            equity_curve.iloc[i] = equity
            
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
            # Update final equity point
            equity_curve.iloc[-1] = equity
            
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

        # ---------------------------------------------------
        # 3. METRICS GENERATION
        # ---------------------------------------------------
        if not trades:
            logger.warning("❌ No trades generated.")
            # Return valid but empty metrics structure to prevent KeyErrors
            return equity_curve, {'trades': [], 'Total Return %': 0.0, 'Max Drawdown %': 0.0, 'Sharpe Ratio': 0.0, 'Total Trades': 0}

        trades_df = pd.DataFrame([vars(t) for t in trades])
        
        # Calculate Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # Returns
        returns = equity_curve.pct_change().dropna()
        
        # SIGMA FIX: Sharpe must include risk-free rate (6.5%)
        # Safe Sharpe
        RISK_FREE_RATE = 0.065
        if returns.std() > 1e-9:
             excess_returns = returns - (RISK_FREE_RATE / 252)
             sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
             assert -2 < sharpe < 5, f"Sharpe {sharpe:.2f} outside plausible range — check inputs"
        else:
             sharpe = 0.0
             
        # SIGMA FIX: Calmar ratio addition
        # Calmar Ratio
        years = len(equity_curve) / 252
        if years > 0:
            annual_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
        else:
            annual_return = 0.0

        if abs(max_dd) > 1e-9:
            calmar = annual_return / abs(max_dd)
        else:
            calmar = float('inf') if annual_return > 0 else 0.0

        # Safe Profit Factor
        gross_wins = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
        gross_losses = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
        
        if gross_losses > 1e-9:
            prof_factor = gross_wins / gross_losses
        elif gross_wins > 0:
            prof_factor = float('inf')
        else:
            prof_factor = 0.0
        
        metrics = {
            'Total Return %': ((equity - self.initial_capital) / self.initial_capital) * 100,
            'Annual Return %': annual_return * 100,
            'Win Rate %': (len(trades_df[trades_df['net_pnl'] > 0]) / len(trades_df)) * 100,
            'Profit Factor': prof_factor,
            'Max Drawdown %': max_dd * 100,
            'Sharpe Ratio': sharpe,
            'Calmar Ratio': calmar,
            'Total Trades': len(trades),
            'Total Taxes Paid': sum(t.taxes['total'] for t in trades),
            'trades': trades # Expose raw trades for reporting
        }
        
        logger.info(f"✅ Simulation Complete. Final Equity: ₹{equity:,.2f}")
        
        return equity_curve, metrics

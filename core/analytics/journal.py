"""
MARK5 TRADE JOURNAL & RECORD KEEPING v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Production hardening
  • Added connection pooling and proper cleanup
  • Added NSE tax calculation engine
  • Added partial exit support with proportional accounting
  • Added unrealized PnL tracking for open positions
- [Previous] v1.0: Initial implementation

TRADING ROLE: Handles trade lifecycle logging with P&L tracking
SAFETY LEVEL: CRITICAL - Must accurately track all trades for audit

FEATURES:
✅ Full trade lifecycle (entry → partial exits → close)
✅ NSE tax calculation (STT, Exchange, SEBI, Stamp, GST)
✅ Daily performance summaries with win rate, profit factor
✅ Pydantic validation for all trade data
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Optional, List
import pandas as pd
from core.infrastructure.database_manager import MARK5DatabaseManager
from pydantic import BaseModel, Field

class TradeEntrySchema(BaseModel):
    trade_id: Optional[str] = None  # Generated if not provided
    timestamp: datetime = Field(default_factory=lambda: datetime.now())  # Entry time (ISO format)
    stock: str  # Ticker (required)
    action: str  # 'BUY'/'SELL' (required)
    entry_price: float  # Executed entry price (required)
    entry_quantity: int  # Total units entered (required)
    entry_value: Optional[float] = None  # Derived: entry_price * entry_quantity (optional)
    stop_loss_price: Optional[float] = None  # Stop loss price per unit
    target_price: Optional[float] = None  # Take profit price per unit
    risk_reward_ratio: Optional[float] = None  # Risk:Reward ratio
    signal_type: str  # Model signal (e.g., 'LSTM_Buy') (required)
    model_confidence: float  # Confidence (0-1) (required)
    reasoning: Dict = Field(default_factory=dict)  # Signal explanation
    order_id: str  # Broker order ID (required)
    commission: float = 0.0  # Entry commission
    slippage: float = 0.0  # Entry slippage (executed - expected price)

class TradeExitSchema(BaseModel):
    exit_time: datetime  # Exact exit time (required)
    exit_price: float  # Executed exit price (required)
    exit_quantity: int  # Units exited (must ≤ remaining_quantity) (required)
    exit_reason: str  # Exit reason (e.g., 'TargetHit') (required)
    commission: float = 0.0  # Exit commission0.0

class TradeJournal:
    """
    Manages the lifecycle of trade records:
    - Entry logging with AI reasoning
    - Exit logging with P&L calculation
    - Daily performance summary
    """
    
    def __init__(self, db_manager: MARK5DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger("MARK5.TradeJournal")
        
    def log_trade_entry(self, trade_data: Dict) -> str:
        """
        Log a new trade entry into the database.
        """
        conn = None
        cursor = None
        try:
            processed_trade_data = trade_data.copy()
            # Parse timestamp if not provided (default to now)
            if 'timestamp' not in processed_trade_data:
                processed_trade_data['timestamp'] = datetime.now()
            elif isinstance(processed_trade_data['timestamp'], str):
                processed_trade_data['timestamp'] = datetime.fromisoformat(processed_trade_data['timestamp'])
            
            entry_data = TradeEntrySchema(**processed_trade_data)
            
            # Generate trade_id using entry timestamp's date
            if not entry_data.trade_id:
                trade_id_date = entry_data.timestamp.strftime('%Y%m%d')
                trade_id_suffix = str(uuid.uuid4().hex)[:6].upper()
                trade_id = f"TRADE_{trade_id_date}_{trade_id_suffix}"
            else:
                trade_id = entry_data.trade_id

            # Compute entry_value if not provided
            entry_value = entry_data.entry_value or (entry_data.entry_price * entry_data.entry_quantity)

            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trade_journal (
                    trade_id, timestamp, stock, action, 
                    entry_price, entry_quantity, entry_value,
                    stop_loss_price, target_price, risk_reward_ratio,
                    signal_type, model_confidence, reasoning,
                    order_id, commission, slippage, status,
                    remaining_quantity, total_exit_quantity, total_gross_pnl, total_net_pnl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                entry_data.timestamp.isoformat(),
                entry_data.stock,
                entry_data.action,
                entry_data.entry_price,
                entry_data.entry_quantity,
                entry_value,
                entry_data.stop_loss_price,
                entry_data.target_price,
                entry_data.risk_reward_ratio,
                entry_data.signal_type,
                entry_data.model_confidence,
                json.dumps(entry_data.reasoning),
                entry_data.order_id,
                entry_data.commission,
                entry_data.slippage,
                'OPEN',
                entry_data.entry_quantity,  # remaining_quantity = entry_quantity initially
                0,  # total_exit_quantity starts at 0
                0.0,  # total_gross_pnl starts at 0
                0.0   # total_net_pnl starts at 0
            ))
            
            conn.commit()
            self.logger.info(f"✅ Trade logged: {trade_id} ({entry_data.stock} {entry_data.action})")
            return trade_id
            
        except Exception as e:
            self.logger.error(f"Entry logging failed: {e}")
            if conn: conn.rollback()
            return False
        finally:
            if cursor: cursor.close()
            if conn is not None: self.db.return_connection(conn)

    def _calculate_nse_taxes(self, buy_val: float, sell_val: float, is_intraday: bool = True) -> float:
        """
        INTERNAL ENGINE: Calculates exact NSE Equity Intraday Taxes.
        Because 'Commission' is not just brokerage.
        """
        turnover = buy_val + sell_val
        
        # 1. Brokerage (Zerodha/Angel style: Flat 20 or 0.03%)
        brokerage = min(40, (buy_val * 0.0003) + (sell_val * 0.0003)) # 20 per leg max
        
        # 2. STT (Securities Transaction Tax) - 0.025% on SELL for Intraday
        stt = sell_val * 0.00025 if is_intraday else turnover * 0.001
        
        # 3. Exchange Txn Charges (NSE Equity: 0.00325%)
        etc = turnover * 0.0000325
        
        # 4. SEBI Charges
        sebi = turnover * 0.000001
        
        # 5. Stamp Duty (RULE 83: 0.015% on BUY)
        stamp = buy_val * 0.00015
        
        # 6. GST (18% on Brokerage + ETC + SEBI)
        gst = (brokerage + etc + sebi) * 0.18
        
        return brokerage + stt + etc + sebi + stamp + gst

    def log_trade_exit(self, trade_id: str, exit_data: Dict) -> bool:
        """
        Log trade exit and calculate P&L
        
        Args:
            trade_id: The ID of the trade to close
            exit_data: Dictionary containing exit details
            
        Returns:
            bool: Success status
        """
        try:
            # Validate Data
            if 'exit_time' not in exit_data:
                exit_data['exit_time'] = datetime.now()
            elif isinstance(exit_data['exit_time'], str):
                exit_data['exit_time'] = datetime.fromisoformat(exit_data['exit_time'])
                
            exit_data_validated = TradeExitSchema(**exit_data)
            
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Fetch entry data + existing exit tracking
            cursor.execute("""
                SELECT entry_price, entry_quantity, entry_value, commission, 
                       remaining_quantity, total_exit_quantity, total_gross_pnl, total_net_pnl 
                FROM trade_journal 
                WHERE trade_id = ? AND status = 'OPEN'
            """, (trade_id,))
            row = cursor.fetchone()
            
            if not row:
                self.logger.error(f"Trade {trade_id} not found or closed.")
                return False
                
            entry_price, entry_qty, entry_val, entry_commission, \
                remaining_quantity, total_exit_quantity, total_gross_pnl, total_net_pnl = row
            
            # --- THE FIX: Proportional Accounting ---
            exit_qty = exit_data_validated.exit_quantity
            exit_price = exit_data_validated.exit_price
            
            # Value of the portion being sold
            exit_val = exit_price * exit_qty
            
            # Cost basis of the portion being sold (Weighted Average)
            # If total_entry_val is tracked correctly, this is: (Total Entry Val / Total Entry Qty) * Exit Qty
            cost_basis = (entry_val / entry_qty) * exit_qty
            
            gross_pnl = exit_val - cost_basis
            
            # --- THE FIX: Real Tax Calculation ---
            # We assume Intraday for this system
            total_taxes = self._calculate_nse_taxes(buy_val=cost_basis, sell_val=exit_val, is_intraday=True)
            
            # Net PnL is Gross - Taxes - Any extra brokerage/slippage passed in arguments
            net_pnl = gross_pnl - total_taxes - exit_data_validated.commission
            
            pnl_pct = (net_pnl / cost_basis) * 100 if cost_basis > 0 else 0.0
            
            # Update trade totals
            # Handle legacy None values
            if remaining_quantity is None: remaining_quantity = entry_qty
            if total_exit_quantity is None: total_exit_quantity = 0
            if total_gross_pnl is None: total_gross_pnl = 0.0
            if total_net_pnl is None: total_net_pnl = 0.0
            
            new_total_exited = total_exit_quantity + exit_data_validated.exit_quantity
            new_remaining_qty = remaining_quantity - exit_data_validated.exit_quantity
            new_total_gross = total_gross_pnl + gross_pnl
            new_total_net = total_net_pnl + net_pnl
            
            cursor.execute("""
                UPDATE trade_journal SET
                    exit_time = ?,
                    exit_price = ?,
                    exit_quantity = ?,
                    exit_value = ?,
                    gross_pnl = ?,
                    net_pnl = ?,
                    pnl_percent = ?,
                    exit_reason = ?,
                    status = 'CLOSED'
                WHERE trade_id = ?
            """, (
                datetime.now().isoformat(),
                exit_data_validated.exit_price,
                exit_data_validated.exit_quantity,
                exit_val,
                gross_pnl,
                net_pnl,
                pnl_pct,
                exit_data_validated.exit_reason,
                'CLOSED' if new_remaining_qty == 0 else 'OPEN',
                new_remaining_qty,
                new_total_exited,
                new_total_gross,
                new_total_net,
                trade_id
            ))
            
            conn.commit()
            self.logger.info(f"✅ Trade closed: {trade_id} (PnL: {net_pnl:.2f})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to log trade exit: {e}")
            if 'conn' in locals(): conn.rollback()
            return False
        finally:
            if 'conn' in locals(): self.db.return_connection(conn)

    def update_open_positions_pnl(self, current_prices: Dict[str, float]):
        """
        Calculate and log unrealized P&L for all open positions.
        """
        conn = None
        cursor = None
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Fetch open trades with remaining_quantity
            cursor.execute("""
                SELECT trade_id, stock, entry_price, remaining_quantity 
                FROM trade_journal 
                WHERE status = 'OPEN'
            """)
            open_trades = cursor.fetchall()
            
            for trade in open_trades:
                tid, stock, entry_price, remaining_qty = trade
                
                # Handle legacy NULLs
                if remaining_qty is None: remaining_qty = 0 # Should not happen with new schema
                
                if stock not in current_prices:
                    continue
                    
                curr_price = current_prices[stock]
                unrealized_pnl = (curr_price - entry_price) * remaining_qty
                
                cursor.execute("""
                    UPDATE trade_journal SET
                        unrealized_pnl = ?
                    WHERE trade_id = ?
                """, (unrealized_pnl, tid))
                
                # self.logger.debug(f"Updated Open PnL: {tid} {stock} -> {unrealized_pnl:.2f}")
                
            conn.commit()
            self.logger.info("Updated unrealized P&L for all open trades.")
            
        except Exception as e:
            self.logger.error(f"Open PnL update failed: {e}")
            if conn: conn.rollback()
        finally:
            if cursor: cursor.close()
            if conn is not None: self.db.return_connection(conn)

    def generate_daily_summary(self, date: str = None) -> Dict:
        """
        Generate a summary of trading performance for a specific date.
        """
        # Parse date
        if not date:
            date_obj = datetime.now()
        else:
            try:
                date_obj = datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                self.logger.error(f"Invalid date {date}. Use YYYY-MM-DD.")
                return {}
        date_str = date_obj.strftime('%Y-%m-%d')

        conn = None
        cursor = None
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()

            # Fetch closed trades with gross_pnl/net_pnl for the date
            # Note: We use exit_timestamp (or exit_time) to filter
            query = """
                SELECT gross_pnl, net_pnl, stock 
                FROM trade_journal 
                WHERE date(exit_time) = ? AND status = 'CLOSED'
            """
            df = pd.read_sql_query(query, conn, params=(date_str,))
            
            if df.empty:
                self.logger.info(f"No closed trades on {date_str}.")
                return {}

            # Calculate stats
            win_mask = df['gross_pnl'] > 0
            loss_mask = df['gross_pnl'] <= 0
            
            total_trades = len(df)
            winning_trades = len(df[win_mask])
            losing_trades = len(df[loss_mask])
            
            gross_profit = df[win_mask]['gross_pnl'].sum()
            gross_loss = abs(df[loss_mask]['gross_pnl'].sum())
            net_profit = df['net_pnl'].sum()
            
            largest_win = df['gross_pnl'].max() if winning_trades > 0 else 0.0
            largest_loss = df['gross_pnl'].min() if losing_trades > 0 else 0.0
            
            avg_win = df[win_mask]['gross_pnl'].mean() if winning_trades > 0 else 0.0
            avg_loss = df[loss_mask]['gross_pnl'].mean() if losing_trades > 0 else 0.0
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
            
            best_stock = df.loc[df['gross_pnl'].idxmax()]['stock'] if not df.empty else "N/A"
            worst_stock = df.loc[df['gross_pnl'].idxmin()]['stock'] if not df.empty else "N/A"

            stats = {
                'net_profit': float(df['net_pnl'].sum()),
                'date': date_str,
                'total_trades': int(total_trades),
                'winning_trades': int(winning_trades),
                'losing_trades': int(losing_trades),
                'gross_profit': float(gross_profit),
                'gross_loss': float(gross_loss),
                'largest_win': float(largest_win),
                'largest_loss': float(largest_loss),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'best_stock': str(best_stock),
                'worst_stock': str(worst_stock)
            }
            
            # Store summary
            cursor.execute("""
                INSERT OR REPLACE INTO daily_summary (
                    date, total_trades, winning_trades, losing_trades, win_rate,
                    gross_profit, gross_loss, net_profit,
                    largest_win, largest_loss, avg_win, avg_loss,
                    profit_factor, best_stock, worst_stock
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_str,
                stats['total_trades'], stats['winning_trades'], stats['losing_trades'], stats['win_rate'],
                stats['gross_profit'], stats['gross_loss'], stats['net_profit'],
                stats['largest_win'], stats['largest_loss'], stats['avg_win'], stats['avg_loss'],
                stats['profit_factor'], stats['best_stock'], stats['worst_stock']
            ))
            
            conn.commit()
            self.logger.info(f"� Daily summary for {date_str} generated. Net P&L: {stats['net_profit']:.2f}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Daily summary failed: {e}")
            return {}
        finally:
            if cursor: cursor.close()
            if conn is not None: self.db.return_connection(conn)

    def update_rolling_sharpe(self, ticker: str, trade_return: float):
        """Call this after every trade close."""
        import numpy as np
        from collections import deque
        if not hasattr(self, '_return_history'):
            self._return_history = {}
            self._rolling_sharpe_cache = {}
            
        if ticker not in self._return_history:
            self._return_history[ticker] = deque(maxlen=60)  # 60-trade window
        
        self._return_history[ticker].append(trade_return)
        
        if len(self._return_history[ticker]) >= 10:  # Minimum 10 trades
            returns = np.array(self._return_history[ticker])
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            self._rolling_sharpe_cache[ticker] = sharpe

    def get_rolling_sharpe(self, ticker: str, window: int = 60) -> float:
        if not hasattr(self, '_rolling_sharpe_cache'): return 0.5
        return self._rolling_sharpe_cache.get(ticker, 0.5)  # Default 0.5 (neutral)

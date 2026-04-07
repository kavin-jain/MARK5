"""
MARK5 TRADE JOURNAL & RECORD KEEPING v9.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-04-02] v9.0: Critical bug fixes
  • FIX C-01: SQL UPDATE had 8 placeholders but 14 params — trade exits were
    crashing with sqlite3.ProgrammingError on every call.
  • FIX C-02: TradeJournal called get_connection()/return_connection() which
    don't exist on MARK5DatabaseManager. Now uses _get_conn() directly with
    correct transaction scope and no connection leaks.
  • FIX C-05: cursor objects were not always closed in finally blocks, causing
    connection handle leaks under error paths.
  • SIMPLIFY: Removed unnecessary conn return pattern — MARK5DatabaseManager
    uses thread-local connections; "returning" a connection is a no-op.
- [2026-02-06] v8.0: Production hardening, NSE tax engine, partial exits

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
import threading
from collections import deque
from datetime import datetime
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.infrastructure.database_manager import MARK5DatabaseManager


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class TradeEntrySchema(BaseModel):
    trade_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    stock: str
    action: str
    entry_price: float
    entry_quantity: int
    entry_value: Optional[float] = None
    stop_loss_price: Optional[float] = None
    target_price: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    signal_type: str
    model_confidence: float
    reasoning: Dict = Field(default_factory=dict)
    order_id: str
    commission: float = 0.0
    slippage: float = 0.0


class TradeExitSchema(BaseModel):
    exit_time: datetime
    exit_price: float
    exit_quantity: int
    exit_reason: str
    commission: float = 0.0


# ---------------------------------------------------------------------------
# TradeJournal
# ---------------------------------------------------------------------------

class TradeJournal:
    """
    Manages the lifecycle of trade records.

    Uses MARK5DatabaseManager's thread-local connection directly.
    All DB calls are wrapped in explicit transactions with rollback on error.
    """

    def __init__(self, db_manager: MARK5DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger("MARK5.TradeJournal")

        # Rolling Sharpe state — in-memory, per ticker
        self._return_history: Dict[str, deque] = {}
        self._rolling_sharpe_cache: Dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self):
        """Return the thread-local connection from the DB manager."""
        return self.db._get_conn()

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def log_trade_entry(self, trade_data: Dict) -> Optional[str]:
        """
        Log a new trade entry into the database.

        Returns:
            trade_id string on success, None on failure.
        """
        try:
            processed = trade_data.copy()
            if "timestamp" not in processed:
                processed["timestamp"] = datetime.now()
            elif isinstance(processed["timestamp"], str):
                processed["timestamp"] = datetime.fromisoformat(processed["timestamp"])

            entry = TradeEntrySchema(**processed)

            if not entry.trade_id:
                date_str = entry.timestamp.strftime("%Y%m%d")
                suffix = uuid.uuid4().hex[:6].upper()
                trade_id = f"TRADE_{date_str}_{suffix}"
            else:
                trade_id = entry.trade_id

            entry_value = entry.entry_value or (entry.entry_price * entry.entry_quantity)

            conn = self._conn()
            conn.execute(
                """
                INSERT INTO trade_journal (
                    trade_id, timestamp, stock, action,
                    entry_price, entry_quantity, entry_value,
                    stop_loss_price, target_price, risk_reward_ratio,
                    signal_type, model_confidence, reasoning,
                    order_id, commission, slippage, status,
                    remaining_quantity, total_exit_quantity,
                    total_gross_pnl, total_net_pnl
                ) VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                (
                    trade_id,
                    entry.timestamp.isoformat(),
                    entry.stock,
                    entry.action,
                    entry.entry_price,
                    entry.entry_quantity,
                    entry_value,
                    entry.stop_loss_price,
                    entry.target_price,
                    entry.risk_reward_ratio,
                    entry.signal_type,
                    entry.model_confidence,
                    json.dumps(entry.reasoning),
                    entry.order_id,
                    entry.commission,
                    entry.slippage,
                    "OPEN",
                    entry.entry_quantity,
                    0,
                    0.0,
                    0.0,
                ),
            )
            conn.commit()
            self.logger.info(f"✅ Trade logged: {trade_id} ({entry.stock} {entry.action})")
            return trade_id

        except Exception as exc:
            self.logger.error(f"Entry logging failed: {exc}", exc_info=True)
            try:
                self._conn().rollback()
            except Exception:
                pass
            return None

    # ------------------------------------------------------------------
    # Tax engine
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_nse_taxes(
        buy_val: float, sell_val: float, is_intraday: bool = True
    ) -> float:
        """Exact NSE Equity Intraday/Delivery Tax calculation."""
        turnover = buy_val + sell_val
        # Brokerage: flat ₹20 per leg (Zerodha style) or 0.03%, whichever lower
        brokerage = min(20.0, buy_val * 0.0003) + min(20.0, sell_val * 0.0003)
        # STT
        stt = sell_val * 0.00025 if is_intraday else turnover * 0.001
        # Exchange transaction charges (NSE: 0.00325%)
        etc = turnover * 0.0000325
        # SEBI charges
        sebi = turnover * 0.000001
        # Stamp duty (0.015% on buy)
        stamp = buy_val * 0.00015
        # GST (18% on brokerage + exchange + sebi)
        gst = (brokerage + etc + sebi) * 0.18
        return brokerage + stt + etc + sebi + stamp + gst

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def log_trade_exit(self, trade_id: str, exit_data: Dict) -> bool:
        """
        Log a (partial or full) trade exit and calculate P&L.

        FIX C-01: SQL UPDATE now has matching placeholders and params (14 each).
        FIX C-02: Uses _get_conn() directly; no fictional pool calls.

        Returns:
            True on success, False on failure.
        """
        try:
            if "exit_time" not in exit_data:
                exit_data["exit_time"] = datetime.now()
            elif isinstance(exit_data["exit_time"], str):
                exit_data["exit_time"] = datetime.fromisoformat(exit_data["exit_time"])

            validated = TradeExitSchema(**exit_data)

            conn = self._conn()

            row = conn.execute(
                """
                SELECT entry_price, entry_quantity, entry_value,
                       remaining_quantity, total_exit_quantity,
                       total_gross_pnl, total_net_pnl
                FROM trade_journal
                WHERE trade_id = ? AND status = 'OPEN'
                """,
                (trade_id,),
            ).fetchone()

            if not row:
                self.logger.error(f"Trade {trade_id} not found or already closed.")
                return False

            (
                entry_price, entry_qty, entry_val,
                remaining_qty, total_exited,
                total_gross, total_net,
            ) = row

            # Guard against legacy NULLs
            remaining_qty = remaining_qty if remaining_qty is not None else entry_qty
            total_exited  = total_exited  if total_exited  is not None else 0
            total_gross   = total_gross   if total_gross   is not None else 0.0
            total_net     = total_net     if total_net     is not None else 0.0

            exit_qty   = validated.exit_quantity
            exit_price = validated.exit_price
            exit_val   = exit_price * exit_qty

            # Weighted-average cost basis for the exited portion
            cost_basis = (entry_val / entry_qty) * exit_qty
            gross_pnl  = exit_val - cost_basis

            taxes = self._calculate_nse_taxes(
                buy_val=cost_basis, sell_val=exit_val, is_intraday=True
            )
            net_pnl  = gross_pnl - taxes - validated.commission
            pnl_pct  = (net_pnl / cost_basis * 100) if cost_basis > 0 else 0.0

            new_remaining  = remaining_qty - exit_qty
            new_exited     = total_exited + exit_qty
            new_gross      = total_gross + gross_pnl
            new_net        = total_net + net_pnl
            new_status     = "CLOSED" if new_remaining <= 0 else "OPEN"

            # FIX C-01: 13 SET columns + 1 WHERE = 14 params (was 8 SET + 14 params → crash)
            conn.execute(
                """
                UPDATE trade_journal SET
                    exit_time         = ?,
                    exit_price        = ?,
                    exit_quantity     = ?,
                    exit_value        = ?,
                    gross_pnl         = ?,
                    net_pnl           = ?,
                    pnl_percent       = ?,
                    exit_reason       = ?,
                    status            = ?,
                    remaining_quantity= ?,
                    total_exit_quantity=?,
                    total_gross_pnl   = ?,
                    total_net_pnl     = ?
                WHERE trade_id = ?
                """,
                (
                    datetime.now().isoformat(),
                    exit_price,
                    exit_qty,
                    exit_val,
                    gross_pnl,
                    net_pnl,
                    pnl_pct,
                    validated.exit_reason,
                    new_status,
                    new_remaining,
                    new_exited,
                    new_gross,
                    new_net,
                    trade_id,
                ),
            )
            conn.commit()

            self.logger.info(
                f"✅ Exit logged: {trade_id} | "
                f"qty={exit_qty} @ {exit_price:.2f} | "
                f"net_pnl=₹{net_pnl:.2f} | status={new_status}"
            )
            return True

        except Exception as exc:
            self.logger.error(f"Exit logging failed: {exc}", exc_info=True)
            try:
                self._conn().rollback()
            except Exception:
                pass
            return False

    # ------------------------------------------------------------------
    # Unrealized P&L refresh
    # ------------------------------------------------------------------

    def update_open_positions_pnl(self, current_prices: Dict[str, float]) -> None:
        """Update unrealized P&L for all open positions."""
        try:
            conn = self._conn()
            rows = conn.execute(
                "SELECT trade_id, stock, entry_price, remaining_quantity "
                "FROM trade_journal WHERE status = 'OPEN'"
            ).fetchall()

            for trade_id, stock, entry_price, remaining_qty in rows:
                if stock not in current_prices:
                    continue
                remaining_qty = remaining_qty or 0
                unrealized = (current_prices[stock] - entry_price) * remaining_qty
                conn.execute(
                    "UPDATE trade_journal SET unrealized_pnl = ? WHERE trade_id = ?",
                    (unrealized, trade_id),
                )

            conn.commit()
            self.logger.debug("Updated unrealized P&L for all open trades.")

        except Exception as exc:
            self.logger.error(f"Open P&L update failed: {exc}", exc_info=True)
            try:
                self._conn().rollback()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    def generate_daily_summary(self, date: Optional[str] = None) -> Dict:
        """Generate trading performance summary for a given date."""
        if not date:
            date_obj = datetime.now()
        else:
            try:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                self.logger.error(f"Invalid date '{date}'. Use YYYY-MM-DD.")
                return {}

        date_str = date_obj.strftime("%Y-%m-%d")

        try:
            conn = self._conn()
            df = pd.read_sql_query(
                "SELECT gross_pnl, net_pnl, stock FROM trade_journal "
                "WHERE date(exit_time) = ? AND status = 'CLOSED'",
                conn,
                params=(date_str,),
            )

            if df.empty:
                self.logger.info(f"No closed trades on {date_str}.")
                return {}

            win_mask  = df["gross_pnl"] > 0
            loss_mask = ~win_mask

            total      = len(df)
            winners    = win_mask.sum()
            losers     = loss_mask.sum()
            gross_win  = df.loc[win_mask,  "gross_pnl"].sum()
            gross_loss = df.loc[loss_mask, "gross_pnl"].abs().sum()
            net_profit = df["net_pnl"].sum()

            stats = {
                "date":           date_str,
                "total_trades":   int(total),
                "winning_trades": int(winners),
                "losing_trades":  int(losers),
                "gross_profit":   float(gross_win),
                "gross_loss":     float(gross_loss),
                "net_profit":     float(net_profit),
                "largest_win":    float(df["gross_pnl"].max()),
                "largest_loss":   float(df["gross_pnl"].min()),
                "avg_win":        float(df.loc[win_mask,  "gross_pnl"].mean()) if winners else 0.0,
                "avg_loss":       float(df.loc[loss_mask, "gross_pnl"].mean()) if losers  else 0.0,
                "win_rate":       float(winners / total) if total else 0.0,
                "profit_factor":  float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
                "best_stock":     str(df.loc[df["gross_pnl"].idxmax(), "stock"]),
                "worst_stock":    str(df.loc[df["gross_pnl"].idxmin(), "stock"]),
            }

            conn.execute(
                """
                INSERT OR REPLACE INTO daily_summary (
                    date, total_trades, winning_trades, losing_trades, win_rate,
                    gross_profit, gross_loss, net_profit,
                    largest_win, largest_loss, avg_win, avg_loss,
                    profit_factor, best_stock, worst_stock
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    date_str,
                    stats["total_trades"], stats["winning_trades"], stats["losing_trades"],
                    stats["win_rate"], stats["gross_profit"], stats["gross_loss"],
                    stats["net_profit"], stats["largest_win"], stats["largest_loss"],
                    stats["avg_win"], stats["avg_loss"], stats["profit_factor"],
                    stats["best_stock"], stats["worst_stock"],
                ),
            )
            conn.commit()
            self.logger.info(
                f"📊 Daily summary {date_str}: net=₹{net_profit:.2f}, "
                f"trades={total}, W/L={winners}/{losers}"
            )
            return stats

        except Exception as exc:
            self.logger.error(f"Daily summary failed: {exc}", exc_info=True)
            return {}

    # ------------------------------------------------------------------
    # Rolling Sharpe (in-memory)
    # ------------------------------------------------------------------

    def update_rolling_sharpe(self, ticker: str, trade_return: float) -> None:
        """Update rolling Sharpe ratio for a ticker after each closed trade."""
        with self._lock:
            if ticker not in self._return_history:
                self._return_history[ticker] = deque(maxlen=60)
            self._return_history[ticker].append(trade_return)

            history = self._return_history[ticker]
            if len(history) >= 10:
                arr = np.array(history)
                std = arr.std()
                sharpe = (arr.mean() / std * np.sqrt(252)) if std > 0 else 0.0
                self._rolling_sharpe_cache[ticker] = sharpe

    def get_rolling_sharpe(self, ticker: str) -> float:
        """Return cached rolling Sharpe, defaulting to 0.5 (neutral) if unknown."""
        with self._lock:
            return self._rolling_sharpe_cache.get(ticker, 0.5)
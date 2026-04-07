"""
MARK5 PRECISION DATABASE MANAGER v9.0 - PRODUCTION GRADE (The Vault)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-04-02] v9.0: Bug fixes
  • FIX C-02: Added get_connection() / return_connection() public interface.
    Previously TradeJournal called these non-existent methods, crashing
    immediately on every journal write. The underlying implementation uses
    thread-local connections so return_connection() is a documented no-op.
  • FIX: Removed duplicate singleton lock acquisition pattern (deadlock risk).
  • CLEANUP: Moved gzip/hashlib imports inside methods that need them to
    avoid top-level import failures in minimal environments.
- [2026-02-06] v8.0: Production hardening

TRADING ROLE: Persistent storage for trades, risk state, market stats
SAFETY LEVEL: CRITICAL - ACID compliance for financial data
"""

import sqlite3
import logging
import os
import threading
import json
import gzip
import hashlib
from datetime import datetime, timedelta
from decimal import Decimal, getcontext, ROUND_HALF_UP
from typing import Dict, List, Optional, Any

from core.utils.config_manager import get_database_config

# Set global precision higher than market requirements
getcontext().prec = 28

logger = logging.getLogger("MARK5.Vault")


# Register Decimal ↔ SQLite adapters at module load time
def _adapt_decimal(d: Decimal) -> str:
    return str(d)


def _convert_decimal(s: bytes) -> Decimal:
    return Decimal(s.decode("ascii"))


sqlite3.register_adapter(Decimal, _adapt_decimal)
sqlite3.register_converter("DECIMAL", _convert_decimal)


class MARK5DatabaseManager:
    """
    Thread-safe SQLite manager using thread-local connections.

    Connection strategy:
    - Each thread gets its own connection via threading.local().
    - get_connection() / return_connection() provide a pool-like API
      so that higher-level classes (TradeJournal etc.) don't need to
      know about the thread-local internals.
    - return_connection() is intentionally a no-op: thread-local
      connections are reused automatically; there is no pool to return to.
    """

    _instance: Optional["MARK5DatabaseManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def __init__(self, config=None):
        if self._initialized:
            return

        self.db_config = config if config else get_database_config()
        self.db_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "../../", self.db_config.path
            )
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.logger = logging.getLogger("MARK5.Vault")
        self._local = threading.local()

        self.init_database()
        self._initialized = True
        self.logger.info(f"MARK5 Database (The Vault) initialized at {self.db_path}")

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """
        Return the thread-local SQLite connection, creating it if needed.
        This is the canonical internal accessor.
        """
        if not hasattr(self._local, "connection") or self._local.connection is None:
            conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                timeout=30.0,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.connection = conn
        return self._local.connection

    def get_connection(self) -> sqlite3.Connection:
        """
        FIX C-02: Public interface used by TradeJournal and other consumers.
        Returns the thread-local connection.
        """
        return self._get_conn()

    def return_connection(self, conn: sqlite3.Connection) -> None:
        """
        FIX C-02: Public interface — intentional no-op.
        Thread-local connections are reused automatically across calls
        in the same thread; there is no pool to return to.
        """
        pass  # No-op by design — thread-local connection stays alive

    def close(self) -> None:
        """Close this thread's connection (call at thread teardown)."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def init_database(self) -> None:
        conn = self._get_conn()

        # Helper to safely add columns (v9.1 hardening)
        def add_col(table, col, definition):
            try:
                # Check current columns
                cursor = conn.execute(f"PRAGMA table_info({table})")
                existing = [row[1] for row in cursor.fetchall()]
                if col not in existing:
                    logger.info(f"VAULT: Migrating {table} -> adding {col}")
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")
            except Exception as e:
                logger.error(f"VAULT: Migration failed for {table}.{col}: {e}")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_journal (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id         TEXT NOT NULL UNIQUE,
                timestamp        TEXT NOT NULL,
                stock            TEXT NOT NULL,
                action           TEXT NOT NULL,

                entry_price      DECIMAL NOT NULL,
                quantity         INTEGER,
                entry_quantity   INTEGER,
                entry_value      DECIMAL NOT NULL,

                stop_loss_price  DECIMAL,
                target_price     DECIMAL,
                risk_reward_ratio REAL,

                signal_type      TEXT,
                model_confidence REAL,
                reasoning        JSON,

                order_id         TEXT,
                commission       DECIMAL DEFAULT 0,
                slippage         DECIMAL DEFAULT 0,

                exit_time        TEXT,
                exit_price       DECIMAL,
                exit_quantity    INTEGER,
                exit_value       DECIMAL,
                gross_pnl        DECIMAL,
                net_pnl          DECIMAL,
                pnl_percent      REAL,
                fees             DECIMAL,
                unrealized_pnl   DECIMAL,

                remaining_quantity    INTEGER DEFAULT 0,
                total_exit_quantity   INTEGER DEFAULT 0,
                total_gross_pnl       DECIMAL DEFAULT 0,
                total_net_pnl         DECIMAL DEFAULT 0,

                exit_reason      TEXT,
                status           TEXT NOT NULL,
                strategy_name    TEXT,
                execution_meta   JSON,

                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migration 2026-04-06: Ensure extended P&L columns exist for v9.0+
        add_col("trade_journal", "remaining_quantity", "INTEGER DEFAULT 0")
        add_col("trade_journal", "total_exit_quantity", "INTEGER DEFAULT 0")
        add_col("trade_journal", "total_gross_pnl", "DECIMAL DEFAULT 0")
        add_col("trade_journal", "total_net_pnl", "DECIMAL DEFAULT 0")
        add_col("trade_journal", "exit_reason", "TEXT")
        add_col("trade_journal", "fees", "DECIMAL")
        conn.commit()

        conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_configuration (
                rule_name  TEXT PRIMARY KEY,
                value      DECIMAL NOT NULL,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "INSERT OR IGNORE INTO risk_configuration (rule_name, value) "
            "VALUES ('MAX_DAILY_LOSS', '5000.00')"
        )
        conn.execute(
            "INSERT OR IGNORE INTO risk_configuration (rule_name, value) "
            "VALUES ('MAX_PER_TRADE_LOSS', '1000.00')"
        )

        conn.execute("""
            CREATE TABLE IF NOT EXISTS circuit_breaker_state (
                ticker              TEXT PRIMARY KEY,
                consecutive_losses  INTEGER DEFAULT 0,
                daily_trades        INTEGER DEFAULT 0,
                last_reset_date     TEXT,
                updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ticker_statistics (
                ticker               TEXT,
                calc_date            TEXT,
                avg_daily_return_pct DECIMAL,
                avg_volume_inr       DECIMAL,
                market_cap_crore     DECIMAL,
                PRIMARY KEY (ticker, calc_date)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_holidays (
                date        TEXT PRIMARY KEY,
                description TEXT,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                date           TEXT PRIMARY KEY,
                total_trades   INTEGER,
                winning_trades INTEGER,
                losing_trades  INTEGER,
                win_rate       REAL,
                gross_profit   DECIMAL,
                gross_loss     DECIMAL,
                net_profit     DECIMAL,
                largest_win    DECIMAL,
                largest_loss   DECIMAL,
                avg_win        DECIMAL,
                avg_loss       DECIMAL,
                profit_factor  REAL,
                best_stock     TEXT,
                worst_stock    TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_checkpoints (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                checkpoint_name  TEXT NOT NULL,
                checkpoint_time  TEXT NOT NULL,
                state_data       BLOB NOT NULL,
                state_hash       TEXT,
                checkpoint_type  TEXT DEFAULT 'MANUAL',
                description      TEXT,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_time
            ON system_checkpoints(checkpoint_time DESC)
        """)

        conn.commit()
        self._migrate_schema(conn)

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Add missing columns to existing tables (safe ALTER TABLE)."""
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(trade_journal)").fetchall()
        }
        migrations = [
            ("entry_quantity",        "INTEGER"),
            ("risk_reward_ratio",     "REAL"),
            ("signal_type",           "TEXT"),
            ("model_confidence",      "REAL"),
            ("reasoning",             "JSON"),
            ("order_id",              "TEXT"),
            ("commission",            "DECIMAL DEFAULT 0"),
            ("slippage",              "DECIMAL DEFAULT 0"),
            ("exit_time",             "TEXT"),
            ("exit_quantity",         "INTEGER"),
            ("pnl_percent",           "REAL"),
            ("unrealized_pnl",        "DECIMAL"),
            ("remaining_quantity",    "INTEGER DEFAULT 0"),
            ("total_exit_quantity",   "INTEGER DEFAULT 0"),
            ("total_gross_pnl",       "DECIMAL DEFAULT 0"),
            ("total_net_pnl",         "DECIMAL DEFAULT 0"),
            ("exit_reason",           "TEXT"),
        ]
        for col_name, col_type in migrations:
            if col_name not in existing:
                try:
                    conn.execute(
                        f"ALTER TABLE trade_journal ADD COLUMN {col_name} {col_type}"
                    )
                    self.logger.info(f"Schema migrated: added column '{col_name}'")
                except Exception as exc:
                    self.logger.warning(f"Migration skipped for '{col_name}': {exc}")
        conn.commit()

    # ------------------------------------------------------------------
    # Business queries
    # ------------------------------------------------------------------

    def log_trade_entry(self, trade_data: Dict) -> bool:
        """Record a new trade entry. Returns True on success."""
        conn = self._get_conn()
        try:
            entry_price = Decimal(str(trade_data["entry_price"]))
            qty = int(trade_data["quantity"])
            value = entry_price * qty

            with conn:
                conn.execute(
                    """
                    INSERT INTO trade_journal
                    (trade_id, timestamp, stock, action,
                     entry_price, quantity, entry_value, status, strategy_name)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        trade_data["trade_id"],
                        datetime.now().isoformat(),
                        trade_data["ticker"],
                        trade_data["action"],
                        entry_price,
                        qty,
                        value,
                        "OPEN",
                        trade_data.get("strategy", "MANUAL"),
                    ),
                )
            return True
        except Exception as exc:
            self.logger.critical(f"FAILED TO LOG TRADE ENTRY: {exc}")
            return False

    def log_trade_exit(self, trade_id: str, exit_data: Dict) -> bool:
        """Update a trade with exit data and precise P&L. Returns True on success."""
        conn = self._get_conn()
        try:
            exit_price = Decimal(str(exit_data["exit_price"]))
            cursor = conn.execute(
                "SELECT entry_price, quantity FROM trade_journal WHERE trade_id = ?",
                (trade_id,),
            )
            row = cursor.fetchone()
            if not row:
                self.logger.error(f"Trade ID {trade_id} not found for exit.")
                return False

            entry_price, qty = row
            gross_pnl = (exit_price - entry_price) * qty
            fees = Decimal(str(exit_data.get("fees", "0.0")))
            net_pnl = gross_pnl - fees

            with conn:
                conn.execute(
                    """
                    UPDATE trade_journal SET
                        exit_price = ?, exit_value = ?,
                        gross_pnl = ?, fees = ?, net_pnl = ?,
                        status = 'CLOSED', updated_at = CURRENT_TIMESTAMP
                    WHERE trade_id = ?
                    """,
                    (exit_price, exit_price * qty, gross_pnl, fees, net_pnl, trade_id),
                )
            return True
        except Exception as exc:
            self.logger.critical(f"FAILED TO LOG TRADE EXIT: {exc}")
            return False

    def get_todays_pnl(self) -> Decimal:
        """Return net realized P&L for today as Decimal."""
        conn = self._get_conn()
        today = datetime.now().strftime("%Y-%m-%d")
        result = conn.execute(
            "SELECT SUM(net_pnl) FROM trade_journal "
            "WHERE status='CLOSED' AND timestamp LIKE ?",
            (f"{today}%",),
        ).fetchone()[0]
        return Decimal(str(result)) if result is not None else Decimal("0.00")

    def get_circuit_breaker_state(self, ticker: str) -> Dict[str, Any]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT consecutive_losses, daily_trades, last_reset_date "
            "FROM circuit_breaker_state WHERE ticker = ?",
            (ticker,),
        ).fetchone()
        if row:
            return {
                "consecutive_losses": row[0],
                "daily_trades": row[1],
                "last_reset_date": row[2],
            }
        return {}

    def update_circuit_breaker_state(
        self, ticker: str, losses: int, trades: int, date: str
    ) -> None:
        conn = self._get_conn()
        with conn:
            conn.execute(
                """
                INSERT INTO circuit_breaker_state
                    (ticker, consecutive_losses, daily_trades, last_reset_date, updated_at)
                VALUES (?,?,?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(ticker) DO UPDATE SET
                    consecutive_losses = excluded.consecutive_losses,
                    daily_trades       = excluded.daily_trades,
                    last_reset_date    = excluded.last_reset_date,
                    updated_at         = CURRENT_TIMESTAMP
                """,
                (ticker, losses, trades, date),
            )

    def get_ticker_stats(self, ticker: str) -> Dict[str, Any]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT avg_daily_return_pct, avg_volume_inr, market_cap_crore "
            "FROM ticker_statistics WHERE ticker = ? "
            "ORDER BY calc_date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        if row:
            return {
                "avg_daily_return_pct": float(row[0]),
                "avg_volume_inr":       float(row[1]),
                "market_cap_crore":     float(row[2]),
            }
        return {}

    def fetch_nse_holidays(self, start_date: str, end_date: str) -> List[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT date FROM market_holidays WHERE date BETWEEN ? AND ?",
            (start_date, end_date),
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Disaster recovery
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        checkpoint_name: str,
        state_data: Dict,
        checkpoint_type: str = "MANUAL",
        description: Optional[str] = None,
    ) -> bool:
        conn = self._get_conn()
        try:
            state_json = json.dumps(state_data, default=str)
            state_compressed = gzip.compress(state_json.encode("utf-8"))
            state_hash = hashlib.sha256(state_compressed).hexdigest()
            with conn:
                conn.execute(
                    """
                    INSERT INTO system_checkpoints
                        (checkpoint_name, checkpoint_time, state_data,
                         state_hash, checkpoint_type, description)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (
                        checkpoint_name,
                        datetime.now().isoformat(),
                        state_compressed,
                        state_hash,
                        checkpoint_type,
                        description,
                    ),
                )
            self.logger.info(
                f"Checkpoint saved: {checkpoint_name} ({len(state_compressed)} bytes)"
            )
            return True
        except Exception as exc:
            self.logger.critical(f"FAILED TO SAVE CHECKPOINT: {exc}")
            return False

    def restore_checkpoint(self, checkpoint_name: Optional[str] = None) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            if checkpoint_name:
                row = conn.execute(
                    "SELECT state_data, state_hash, checkpoint_time, checkpoint_type "
                    "FROM system_checkpoints WHERE checkpoint_name = ? "
                    "ORDER BY checkpoint_time DESC LIMIT 1",
                    (checkpoint_name,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT state_data, state_hash, checkpoint_time, checkpoint_type "
                    "FROM system_checkpoints ORDER BY checkpoint_time DESC LIMIT 1"
                ).fetchone()

            if not row:
                self.logger.warning(
                    f"Checkpoint not found: {checkpoint_name or 'latest'}"
                )
                return None

            state_compressed, stored_hash, cp_time, cp_type = row
            actual_hash = hashlib.sha256(state_compressed).hexdigest()
            if actual_hash != stored_hash:
                self.logger.critical(
                    f"CHECKPOINT INTEGRITY FAILURE: {checkpoint_name}"
                )
                return None

            state_data = json.loads(gzip.decompress(state_compressed).decode("utf-8"))
            self.logger.info(f"Checkpoint restored: {checkpoint_name or 'latest'}")
            return {
                "state": state_data,
                "checkpoint_time": cp_time,
                "checkpoint_type": cp_type,
                "verified": True,
            }
        except Exception as exc:
            self.logger.critical(f"FAILED TO RESTORE CHECKPOINT: {exc}")
            return None

    def list_checkpoints(self, limit: int = 20) -> List[Dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT checkpoint_name, checkpoint_time, checkpoint_type, "
            "LENGTH(state_data) as size_bytes, description "
            "FROM system_checkpoints ORDER BY checkpoint_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "name":        r[0],
                "time":        r[1],
                "type":        r[2],
                "size_bytes":  r[3],
                "description": r[4],
            }
            for r in rows
        ]

    def cleanup_old_checkpoints(self, keep_days: int = 7) -> int:
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat()
        with conn:
            cursor = conn.execute(
                "DELETE FROM system_checkpoints "
                "WHERE checkpoint_time < ? AND checkpoint_type != 'END_OF_DAY'",
                (cutoff,),
            )
        deleted = cursor.rowcount
        self.logger.info(f"Cleaned up {deleted} old checkpoints")
        return deleted

    def create_backup(self, backup_path: Optional[str] = None) -> str:
        if backup_path is None:
            backup_dir = os.path.join(os.path.dirname(self.db_path), "backups")
            os.makedirs(backup_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"mark5_backup_{ts}.db")

        conn = self._get_conn()
        backup_conn = sqlite3.connect(backup_path)
        try:
            conn.backup(backup_conn)
            self.logger.info(f"Database backup created: {backup_path}")
            return backup_path
        finally:
            backup_conn.close()
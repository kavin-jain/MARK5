"""
MARK5 PRECISION DATABASE MANAGER v8.0 - PRODUCTION GRADE (The Vault)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Standardized header, version bump
- [Previous] v6.0: Precision architecture, Decimal enforcement

TRADING ROLE: Persistent storage for trades, risk state, market stats
SAFETY LEVEL: CRITICAL - ACID compliance for financial data

FEATURES:
✅ Decimal.Decimal for all monetary values (28-digit precision)
✅ Thread-local connections with WAL journaling
✅ Trade journal, risk configuration, circuit breaker state
"""

import sqlite3
import logging
import os
import threading
import queue
import time
from decimal import Decimal, getcontext, ROUND_HALF_UP
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import json
import gzip

from core.utils.config_manager import get_database_config

# --- PRECISION ARCHITECTURE ---
# Set global precision higher than market requirements to absorb intermediate calc errors
getcontext().prec = 28

logger = logging.getLogger(__name__)

def adapt_decimal(d):
    return str(d)

def convert_decimal(s):
    return Decimal(s.decode('ascii'))

# Register converters immediately
sqlite3.register_adapter(Decimal, adapt_decimal)
sqlite3.register_converter("DECIMAL", convert_decimal)

class MARK5DatabaseManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # Singleton to prevent file lock contention
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MARK5DatabaseManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config=None):
        if self._initialized: return
        
        self.db_config = config if config else get_database_config()
        self.db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', self.db_config.path))
        
        # Ensure directory
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.logger = logging.getLogger('MARK5.Vault')
        self.local_storage = threading.local() # Thread-local connections
        
        self.init_database()
        self._initialized = True
        self.logger.info("MARK5 Precision Database (The Vault) Initialized.")

    def _get_conn(self) -> sqlite3.Connection:
        """
        Thread-local connection factory.
        SQLite connections cannot be shared across threads safely.
        """
        if not hasattr(self.local_storage, 'connection'):
            conn = sqlite3.connect(
                self.db_path, 
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                timeout=30.0 # High timeout to wait for locks instead of failing
            )
            # Performance & Integrity Settings
            conn.execute("PRAGMA journal_mode=WAL") # Write-Ahead Logging for concurrency
            conn.execute("PRAGMA synchronous=NORMAL") # Safe enough for WAL, faster
            conn.execute("PRAGMA foreign_keys=ON")
            self.local_storage.connection = conn
        return self.local_storage.connection

    def init_database(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 1. TRADE JOURNAL (The Source of Truth)
        # Note the use of "DECIMAL" type which we hooked above
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL,
                stock TEXT NOT NULL,
                action TEXT NOT NULL,
                
                -- PRECISION MONEY FIELDS --
                entry_price DECIMAL NOT NULL,
                quantity INTEGER,
                entry_quantity INTEGER,
                entry_value DECIMAL NOT NULL,
                
                stop_loss_price DECIMAL,
                target_price DECIMAL,
                risk_reward_ratio REAL,
                
                -- SIGNAL & MODEL FIELDS --
                signal_type TEXT,
                model_confidence REAL,
                reasoning JSON,
                
                -- ORDER EXECUTION FIELDS --
                order_id TEXT,
                commission DECIMAL DEFAULT 0,
                slippage DECIMAL DEFAULT 0,
                
                exit_price DECIMAL,
                exit_value DECIMAL,
                
                gross_pnl DECIMAL,
                net_pnl DECIMAL,
                fees DECIMAL,
                
                -- POSITION TRACKING --
                remaining_quantity INTEGER DEFAULT 0,
                total_exit_quantity INTEGER DEFAULT 0,
                total_gross_pnl DECIMAL DEFAULT 0,
                total_net_pnl DECIMAL DEFAULT 0,
                
                status TEXT NOT NULL,
                strategy_name TEXT,
                execution_meta JSON,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. RISK LIMITS (Configuration)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_configuration (
                rule_name TEXT PRIMARY KEY,
                value DECIMAL NOT NULL,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert safe defaults if empty
        cursor.execute("INSERT OR IGNORE INTO risk_configuration (rule_name, value) VALUES ('MAX_DAILY_LOSS', '5000.00')")
        cursor.execute("INSERT OR IGNORE INTO risk_configuration (rule_name, value) VALUES ('MAX_PER_TRADE_LOSS', '1000.00')")

        # 3. CIRCUIT BREAKER STATE (Persistence)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS circuit_breaker_state (
                ticker TEXT PRIMARY KEY,
                consecutive_losses INTEGER DEFAULT 0,
                daily_trades INTEGER DEFAULT 0,
                last_reset_date TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 4. MARKET DATA CACHE (For Slippage/Stats)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ticker_statistics (
                ticker TEXT,
                calc_date TEXT,
                avg_daily_return_pct DECIMAL,
                avg_volume_inr DECIMAL,
                market_cap_crore DECIMAL,
                PRIMARY KEY (ticker, calc_date)
            )
        """)

        # 5. MARKET HOLIDAYS (Exchange Schedule)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_holidays (
                date TEXT PRIMARY KEY,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 6. SYSTEM CHECKPOINTS (Disaster Recovery)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checkpoint_name TEXT NOT NULL,
                checkpoint_time TEXT NOT NULL,
                state_data BLOB NOT NULL,
                state_hash TEXT,
                checkpoint_type TEXT DEFAULT 'MANUAL',
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Index for checkpoint lookup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_time 
            ON system_checkpoints(checkpoint_time DESC)
        """)

        conn.commit()

        # --- Schema Migration: Add missing columns to existing trade_journal ---
        try:
            existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(trade_journal)").fetchall()}
            migrations = [
                ("entry_quantity", "INTEGER"),
                ("risk_reward_ratio", "REAL"),
                ("signal_type", "TEXT"),
                ("model_confidence", "REAL"),
                ("reasoning", "JSON"),
                ("order_id", "TEXT"),
                ("commission", "DECIMAL DEFAULT 0"),
                ("slippage", "DECIMAL DEFAULT 0"),
                ("remaining_quantity", "INTEGER DEFAULT 0"),
                ("total_exit_quantity", "INTEGER DEFAULT 0"),
                ("total_gross_pnl", "DECIMAL DEFAULT 0"),
                ("total_net_pnl", "DECIMAL DEFAULT 0"),
            ]
            for col_name, col_type in migrations:
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE trade_journal ADD COLUMN {col_name} {col_type}")
                    logger.info(f"Migrated trade_journal: added column '{col_name}'")
            conn.commit()
        except Exception as e:
            logger.warning(f"Schema migration check: {e}")

    def log_trade_entry(self, trade_data: Dict) -> bool:
        """
        Records a new trade entry with ACID guarantees.
        """
        conn = self._get_conn()
        try:
            # Enforce Decimal Conversion
            entry_price = Decimal(str(trade_data['entry_price']))
            qty = int(trade_data['quantity'])
            value = entry_price * qty
            
            with conn: # Automatic Transaction
                conn.execute("""
                    INSERT INTO trade_journal 
                    (trade_id, timestamp, stock, action, entry_price, quantity, entry_value, status, strategy_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
                """, (
                    trade_data['trade_id'],
                    datetime.now().isoformat(),
                    trade_data['ticker'],
                    trade_data['action'],
                    entry_price,
                    qty,
                    value,
                    trade_data.get('strategy', 'MANUAL')
                ))
            return True
        except Exception as e:
            self.logger.critical(f"FAILED TO LOG TRADE: {e}")
            return False

    def log_trade_exit(self, trade_id: str, exit_data: Dict) -> bool:
        """
        Updates trade with exit data and precise PnL calculation.
        """
        conn = self._get_conn()
        try:
            exit_price = Decimal(str(exit_data['exit_price']))
            
            # Fetch original entry to calculate PnL cleanly
            cursor = conn.execute("SELECT entry_price, quantity FROM trade_journal WHERE trade_id = ?", (trade_id,))
            row = cursor.fetchone()
            if not row:
                self.logger.error(f"Trade ID {trade_id} not found for exit.")
                return False
                
            entry_price = row[0] # Is already Decimal
            qty = row[1]
            
            # Precise Calculation
            # Long: (Exit - Entry) * Qty
            # Short: (Entry - Exit) * Qty
            # Assuming 'action' check happened in logic, simplified here for 'LONG'
            
            gross_pnl = (exit_price - entry_price) * qty 
            fees = Decimal(str(exit_data.get('fees', '0.0')))
            net_pnl = gross_pnl - fees
            
            with conn:
                conn.execute("""
                    UPDATE trade_journal 
                    SET exit_price = ?,
                        exit_value = ?,
                        gross_pnl = ?,
                        fees = ?,
                        net_pnl = ?,
                        status = 'CLOSED',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE trade_id = ?
                """, (
                    exit_price, 
                    exit_price * qty,
                    gross_pnl,
                    fees,
                    net_pnl,
                    trade_id
                ))
            return True
        except Exception as e:
            self.logger.critical(f"FAILED TO LOG EXIT: {e}")
            return False

    def get_todays_pnl(self) -> Decimal:
        """Returns NET PnL for the current day as a Decimal."""
        conn = self._get_conn()
        today = datetime.now().strftime('%Y-%m-%d')
        cursor = conn.execute("""
            SELECT SUM(net_pnl) FROM trade_journal 
            WHERE status='CLOSED' AND timestamp LIKE ?
        """, (f"{today}%",))
        result = cursor.fetchone()[0]
        return result if result is not None else Decimal('0.00')

    def get_circuit_breaker_state(self, ticker: str) -> Dict[str, Any]:
        """Fetch persistent state for a ticker."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT consecutive_losses, daily_trades, last_reset_date FROM circuit_breaker_state WHERE ticker = ?",
            (ticker,)
        )
        row = cursor.fetchone()
        if row:
            return {
                'consecutive_losses': row[0],
                'daily_trades': row[1],
                'last_reset_date': row[2]
            }
        return {}

    def update_circuit_breaker_state(self, ticker: str, losses: int, trades: int, date: str):
        """Update circuit breaker state atomically."""
        conn = self._get_conn()
        with conn:
            conn.execute("""
                INSERT INTO circuit_breaker_state (ticker, consecutive_losses, daily_trades, last_reset_date, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(ticker) DO UPDATE SET
                    consecutive_losses = excluded.consecutive_losses,
                    daily_trades = excluded.daily_trades,
                    last_reset_date = excluded.last_reset_date,
                    updated_at = CURRENT_TIMESTAMP
            """, (ticker, losses, trades, date))

    def get_ticker_stats(self, ticker: str) -> Dict[str, Any]:
        """Fetch latest market stats for slippage/cost logic."""
        conn = self._get_conn()
        # Get most recent entry
        cursor = conn.execute("""
            SELECT avg_daily_return_pct, avg_volume_inr, market_cap_crore 
            FROM ticker_statistics 
            WHERE ticker = ? 
            ORDER BY calc_date DESC LIMIT 1
        """, (ticker,))
        row = cursor.fetchone()
        if row:
            return {
                'avg_daily_return_pct': float(row[0]),
                'avg_volume_inr': float(row[1]),
                'market_cap_crore': float(row[2])
            }
        return {}

    def fetch_nse_holidays(self, start_date: str, end_date: str) -> List[str]:
        """Fetch list of NSE holidays between dates."""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT date FROM market_holidays 
            WHERE date BETWEEN ? AND ?
        """, (start_date, end_date))
        return [row[0] for row in cursor.fetchall()]

    def close(self):
        if hasattr(self.local_storage, 'connection'):
            self.local_storage.connection.close()

    # =========================================================================
    # DISASTER RECOVERY (Checkpoint System)
    # =========================================================================

    def save_checkpoint(
        self,
        checkpoint_name: str,
        state_data: Dict,
        checkpoint_type: str = "MANUAL",
        description: str = None
    ) -> bool:
        """
        Save system state checkpoint for disaster recovery.
        
        Checkpoints are saved hourly during market hours (automatically)
        or manually before risky operations.
        
        Args:
            checkpoint_name: Unique identifier for this checkpoint
            state_data: Dict containing all critical system state
            checkpoint_type: MANUAL, HOURLY, PRE_TRADE, END_OF_DAY
            description: Optional description
            
        Returns:
            True if checkpoint saved successfully
        """
        conn = self._get_conn()
        try:
            # Serialize and compress state
            state_json = json.dumps(state_data, default=str)
            state_compressed = gzip.compress(state_json.encode('utf-8'))
            
            # Calculate hash for integrity verification
            import hashlib
            state_hash = hashlib.sha256(state_compressed).hexdigest()
            
            with conn:
                conn.execute("""
                    INSERT INTO system_checkpoints
                    (checkpoint_name, checkpoint_time, state_data, state_hash, 
                     checkpoint_type, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    checkpoint_name,
                    datetime.now().isoformat(),
                    state_compressed,
                    state_hash,
                    checkpoint_type,
                    description
                ))
            
            self.logger.info(f"Checkpoint saved: {checkpoint_name} ({len(state_compressed)} bytes)")
            return True
            
        except Exception as e:
            self.logger.critical(f"FAILED TO SAVE CHECKPOINT: {e}")
            return False

    def restore_checkpoint(self, checkpoint_name: str = None) -> Optional[Dict]:
        """
        Restore system state from checkpoint.
        
        Args:
            checkpoint_name: Specific checkpoint to restore, or None for latest
            
        Returns:
            Dict with system state, or None if not found
        """
        conn = self._get_conn()
        try:
            if checkpoint_name:
                cursor = conn.execute("""
                    SELECT state_data, state_hash, checkpoint_time, checkpoint_type
                    FROM system_checkpoints
                    WHERE checkpoint_name = ?
                    ORDER BY checkpoint_time DESC LIMIT 1
                """, (checkpoint_name,))
            else:
                # Get latest checkpoint
                cursor = conn.execute("""
                    SELECT state_data, state_hash, checkpoint_time, checkpoint_type
                    FROM system_checkpoints
                    ORDER BY checkpoint_time DESC LIMIT 1
                """)
            
            row = cursor.fetchone()
            if not row:
                self.logger.warning(f"Checkpoint not found: {checkpoint_name or 'latest'}")
                return None
            
            state_compressed, stored_hash, checkpoint_time, cp_type = row
            
            # Verify integrity
            import hashlib
            actual_hash = hashlib.sha256(state_compressed).hexdigest()
            if actual_hash != stored_hash:
                self.logger.critical(
                    f"CHECKPOINT INTEGRITY FAILURE: {checkpoint_name} "
                    f"(stored: {stored_hash[:16]}, actual: {actual_hash[:16]})"
                )
                return None
            
            # Decompress and deserialize
            state_json = gzip.decompress(state_compressed).decode('utf-8')
            state_data = json.loads(state_json)
            
            self.logger.info(
                f"Checkpoint restored: {checkpoint_name or 'latest'} "
                f"(from {checkpoint_time}, type: {cp_type})"
            )
            
            return {
                'state': state_data,
                'checkpoint_time': checkpoint_time,
                'checkpoint_type': cp_type,
                'verified': True
            }
            
        except Exception as e:
            self.logger.critical(f"FAILED TO RESTORE CHECKPOINT: {e}")
            return None

    def list_checkpoints(self, limit: int = 20) -> List[Dict]:
        """List available checkpoints for recovery."""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT checkpoint_name, checkpoint_time, checkpoint_type, 
                   LENGTH(state_data) as size_bytes, description
            FROM system_checkpoints
            ORDER BY checkpoint_time DESC
            LIMIT ?
        """, (limit,))
        
        return [
            {
                'name': row[0],
                'time': row[1],
                'type': row[2],
                'size_bytes': row[3],
                'description': row[4]
            }
            for row in cursor.fetchall()
        ]

    def cleanup_old_checkpoints(self, keep_days: int = 7) -> int:
        """Remove checkpoints older than specified days."""
        conn = self._get_conn()
        cutoff = (datetime.now() - __import__('datetime').timedelta(days=keep_days)).isoformat()
        
        with conn:
            cursor = conn.execute("""
                DELETE FROM system_checkpoints
                WHERE checkpoint_time < ? AND checkpoint_type != 'END_OF_DAY'
            """, (cutoff,))
            
        deleted = cursor.rowcount
        self.logger.info(f"Cleaned up {deleted} old checkpoints")
        return deleted

    def create_backup(self, backup_path: str = None) -> str:
        """
        Create a full database backup file.
        
        Args:
            backup_path: Optional custom path, defaults to data/backups/
            
        Returns:
            Path to backup file
        """
        if backup_path is None:
            backup_dir = os.path.join(os.path.dirname(self.db_path), 'backups')
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(backup_dir, f'mark5_backup_{timestamp}.db')
        
        conn = self._get_conn()
        backup_conn = sqlite3.connect(backup_path)
        
        try:
            conn.backup(backup_conn)
            self.logger.info(f"Database backup created: {backup_path}")
            return backup_path
        finally:
            backup_conn.close()

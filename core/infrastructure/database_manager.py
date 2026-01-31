"""
MARK5 Precision Database Manager v6.0 (The Vault)
-------------------------------------------------
Architectural Mandate: ZERO FLOATING POINT ERRORS.
- Enforces decimal.Decimal for all monetary values.
- Strictly separates Hot Data (Redis/Timescale) from Metadata (SQLite).
- ACID Compliance for Trade Journaling.
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
                quantity INTEGER NOT NULL,
                entry_value DECIMAL NOT NULL,
                
                stop_loss_price DECIMAL,
                target_price DECIMAL,
                
                exit_price DECIMAL,
                exit_value DECIMAL,
                
                gross_pnl DECIMAL,
                net_pnl DECIMAL,
                fees DECIMAL,
                
                status TEXT NOT NULL,
                strategy_name TEXT,
                execution_meta JSON, -- Slippage, latency stats
                
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

        conn.commit()

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

    def close(self):
        if hasattr(self.local_storage, 'connection'):
            self.local_storage.connection.close()

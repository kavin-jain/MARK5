"""
MARK5 DATABASE LOGGER v1.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-05-11] v1.0: Initial creation for DB auditing

TRADING ROLE: Centralized Logging for Database Infrastructure
SAFETY LEVEL: HIGH
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import threading

# Ensure logs directory exists
LOG_DIR = Path("logs/database")
LOG_DIR.mkdir(parents=True, exist_ok=True)

_db_logger_instance = None
_lock = threading.Lock()

def get_db_logger() -> logging.Logger:
    """
    Returns a configured logger for database audit and error tracking.
    Uses RotatingFileHandler to prevent disk space exhaustion.
    Max 50MB per file, keeping 10 backups.
    """
    global _db_logger_instance
    with _lock:
        if _db_logger_instance is not None:
            return _db_logger_instance
            
        logger = logging.getLogger("MARK5.Database")
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Formatter: [TIMESTAMP] [LEVEL] [ACTION/DETAILS]
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Audit Log Handler (All DB operations)
        audit_file = LOG_DIR / "audit.log"
        audit_handler = RotatingFileHandler(
            audit_file, maxBytes=50*1024*1024, backupCount=10, encoding='utf-8'
        )
        audit_handler.setLevel(logging.INFO)
        audit_handler.setFormatter(formatter)
        
        # Error Log Handler (Only warnings and errors)
        error_file = LOG_DIR / "error.log"
        error_handler = RotatingFileHandler(
            error_file, maxBytes=50*1024*1024, backupCount=10, encoding='utf-8'
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(formatter)
        
        logger.addHandler(audit_handler)
        logger.addHandler(error_handler)
        
        # Prevent propagation to root to keep console clean
        logger.propagate = False
        
        _db_logger_instance = logger
        return _db_logger_instance

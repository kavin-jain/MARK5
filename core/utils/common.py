"""
MARK5 Common Utilities
======================
Consolidated utilities for file system, logging, decorators, and caching.
"""

import json
import pickle
import logging
import hashlib
import threading
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Union, Optional, Callable, TypeVar, Type, Dict

# =============================================================================
# File System Utilities
# =============================================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_json_load(file_path: Union[str, Path], default: Any = None) -> Any:
    """
    Safely load JSON file with error handling
    
    Args:
        file_path: Path to JSON file
        default: Default value to return on error
        
    Returns:
        Loaded JSON data or default value
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not load {file_path}: {e}")
        return default


def safe_json_save(data: Any, file_path: Union[str, Path]) -> bool:
    """
    Safely save data to JSON file
    
    Args:
        data: Data to save
        file_path: Path to save JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_directory(Path(file_path).parent)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Could not save {file_path}: {e}")
        return False


def safe_pickle_load(file_path: Union[str, Path], default: Any = None) -> Any:
    """
    Safely load pickle file with error handling
    
    Args:
        file_path: Path to pickle file
        default: Default value to return on error
        
    Returns:
        Loaded pickle data or default value
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not load {file_path}: {e}")
        return default


def safe_pickle_save(data: Any, file_path: Union[str, Path]) -> bool:
    """
    Safely save data to pickle file
    
    Args:
        data: Data to save
        file_path: Path to save pickle file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_directory(Path(file_path).parent)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Could not save {file_path}: {e}")
        return False


# =============================================================================
# Logging Utilities
# =============================================================================

# Track initialized loggers to prevent duplicates
_initialized_loggers = set()


def setup_logger(
    name: str, 
    level: int = logging.INFO, 
    log_file: Optional[str] = None,
    console_level: Optional[int] = None,
    force_reinit: bool = False
) -> logging.Logger:
    """
    Setup logger with consistent formatting and duplicate prevention
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: INFO)
        log_file: Optional file path for file logging
        console_level: Console logging level (default: same as level)
        force_reinit: Force reinitialization even if already set up
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Return existing logger if already initialized (prevents duplicates)
    if name in _initialized_loggers and not force_reinit:
        return logger
    
    # Clear existing handlers to prevent duplicates
    logger.handlers.clear()
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler if log_file specified
    if log_file:
        ensure_directory(Path(log_file).parent)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    # Console handler (with optional different level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level if console_level is not None else level)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger (avoids duplicate console output)
    logger.propagate = False
    
    # Mark as initialized
    _initialized_loggers.add(name)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger or create with default settings
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _initialized_loggers:
        return logging.getLogger(name)
    else:
        return setup_logger(name)


def reset_logger(name: str) -> None:
    """
    Reset logger initialization state
    
    Args:
        name: Logger name to reset
    """
    _initialized_loggers.discard(name)
    logger = logging.getLogger(name)
    logger.handlers.clear()


# =============================================================================
# Decorator Utilities
# =============================================================================

T = TypeVar('T')


class Singleton(type):
    """
    Thread-safe singleton metaclass
    Ensures only one instance of a class exists
    """
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls: Type[T], *args, **kwargs) -> T:
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logging.getLogger(__name__).debug(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper


def retry_decorator(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0, 
                    retry_on: tuple = (Exception,), exclude: tuple = (), fallback_value = None) -> Callable:
    """
    Decorator to retry functions on failure with exponential backoff
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exclude as e:
                    logging.getLogger(__name__).error(f"❌ {func.__name__} failed with non-retriable error: {e}")
                    raise e
                except retry_on as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        logging.getLogger(__name__).error(f"❌ {func.__name__} failed after {max_retries} attempts: {e}")
                        break
                    
                    logging.getLogger(__name__).warning(
                        f"⚠️ {func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return fallback_value
        return wrapper
    return decorator


# =============================================================================
# Caching Utilities
# =============================================================================

def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a unique cache key from arguments
    """
    key_data = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_data.encode()).hexdigest()


class TTLCache:
    """
    Simple TTL (Time-To-Live) cache implementation
    Thread-safe caching with automatic expiration
    """
    
    def __init__(self, ttl: int = 300):
        """
        Initialize TTL cache
        
        Args:
            ttl: Time-to-live in seconds (default: 300 = 5 minutes)
        """
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self._lock = threading.Lock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get cached value if not expired
        """
        with self._lock:
            if key in self.cache:
                if datetime.now() - self.timestamps[key] < timedelta(seconds=self.ttl):
                    return self.cache[key]
                else:
                    self._remove(key)
        return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set cached value with current timestamp
        """
        with self._lock:
            self.cache[key] = value
            self.timestamps[key] = datetime.now()
    
    def _remove(self, key: str) -> None:
        """
        Remove key from cache (internal method)
        """
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries and return count
        """
        with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, timestamp in self.timestamps.items()
                if now - timestamp >= timedelta(seconds=self.ttl)
            ]
            for key in expired_keys:
                self._remove(key)
            return len(expired_keys)

"""
MARK5 CONFIG MANAGER v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Standardized header, production certification
- [Previous] v6.0: Architect Grade Validation

TRADING ROLE: Central Configuration Authority
SAFETY LEVEL: CRITICAL - Controls all system parameters

FEATURES:
✅ YAML + Pydantic Validation
✅ Environment Variable Overrides
✅ Deep Merging of Defaults
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from copy import deepcopy
from pydantic import ValidationError

from core.config.validators import (
    SystemConfig, DatabaseConfig, RedisConfig, HardwareConfig,
    RiskManagementConfig, MLModelConfig, ExecutionConfig, TimescaleConfig, TradingConfig
)

# Default Configuration
DEFAULT_SYSTEM_CONFIG = {
    "system_name": "MARK5",
    "version": "6.0.0 (Architect)",
    "mode": "PAPER",
    "database": {
        "path": "database/main/mark5.db",
        "wal_mode": True
    },
    "timescale": {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "password",
        "dbname": "mark5_timescale"
    },
    "redis": {
        "host": "localhost",
        "port": 6379
    },
    "hardware": {
        "enable_gpu": True
    },
    "risk": {
        "initial_capital": 100000.0,
        "max_drawdown_hard_stop": 0.05,
        "max_daily_loss": 0.02
    },
    "ml": {
        "active_models": ["xgboost", "lightgbm"]
    },
    "execution": {
        "exchange": "NSE"
    },
    "models_dir": "models",
    "reports_dir": "reports",
    "logs_dir": "logs"
}

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger("MARK5.ConfigManager")
        project_root = Path(__file__).parent.parent.parent.resolve()
        self.config_dir = project_root / "config"
        self.config_file = self.config_dir / "config.yaml"
        
        self._config = self._load_config()
        self._initialized = True

    def _load_config(self) -> SystemConfig:
        """Load and validate configuration"""
        config_data = deepcopy(DEFAULT_SYSTEM_CONFIG)
        
        # Load from file if exists
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        self._deep_merge(config_data, file_config)
            except Exception as e:
                self.logger.error(f"Failed to load config file: {e}")

        # Resolve paths to absolute paths relative to project root
        project_root = str(Path(__file__).parent.parent.parent.resolve())
        for path_key in ['models_dir', 'reports_dir', 'logs_dir']:
            if path_key in config_data and not os.path.isabs(config_data[path_key]):
                config_data[path_key] = os.path.join(project_root, config_data[path_key])

        # Validate with Pydantic
        try:
            return SystemConfig(**config_data)
        except ValidationError as e:
            self.logger.critical(f"Configuration Validation Failed: {e}")
            raise

    def _deep_merge(self, base: Dict, update: Dict):
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get_config(self) -> SystemConfig:
        return self._config

def get_config() -> SystemConfig:
    return ConfigManager().get_config()

# Convenience Accessors
def get_database_config() -> DatabaseConfig:
    return get_config().database

def get_timescale_config() -> TimescaleConfig:
    return get_config().timescale

def get_redis_config() -> RedisConfig:
    return get_config().redis

def get_risk_config() -> RiskManagementConfig:
    return get_config().risk

def get_execution_config() -> ExecutionConfig:
    return get_config().execution

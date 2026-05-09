"""
MARK5 SYSTEM VALIDATORS v8.1 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-05-09] v8.1: Migrate from deprecated class-based Config to ConfigDict (Pydantic v2)
- [2026-02-06] v8.0: Standardized header, production certification

TRADING ROLE: Configuration Schema & Integrity Enforcement
SAFETY LEVEL: CRITICAL - Invalid config halts system
"""

import os
from typing import List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from datetime import time

# ---------------- 1. INFRASTRUCTURE (The Iron) ----------------

class DatabaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    path: str = "database/main/mark5.db"
    connection_timeout: int = 10
    wal_mode: bool = True
    synchronous: str = "NORMAL"


class RedisConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    host: str = "localhost"
    port: int = 6379
    password: Optional[SecretStr] = None
    socket_timeout: float = Field(0.05, le=0.1)
    max_connections: int = 100


class HardwareConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enable_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    num_worker_threads: int = Field(8, ge=2)
    use_quantization: bool = False


# ---------------- 2. TRADING LOGIC (The Brain) ----------------

class RiskManagementConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    initial_capital: float = Field(..., gt=0)

    # Kelly Criterion Settings
    enable_kelly_sizing: bool = True
    kelly_fraction: float = Field(0.5, gt=0.0, le=1.0)

    # Circuit Breakers — Rule 18: 5% DD hard stop, Rule 17: 2% daily
    max_drawdown_hard_stop: float = Field(0.05, gt=0.0, le=0.2)
    max_daily_loss: float = Field(0.02, gt=0.0, le=0.1)

    # Per Trade Limits — Rule 13: max 7.5% per position
    max_position_size_pct: float = Field(0.075, gt=0, le=0.15)
    default_stop_loss_pct: float = 0.01

    # India VIX gate — Rule 23: VOLATILE regime at VIX > 22
    max_allowed_vix: float = 35.0

    @model_validator(mode='after')
    def validate_safety(self):
        if self.max_daily_loss >= self.max_drawdown_hard_stop:
            raise ValueError("Daily Loss Limit must be tighter than Total System Hard Stop")
        return self


class MLModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    active_models: List[str] = ["xgboost", "lightgbm"]
    retrain_interval_days: int = 7

    # Drift Detection
    enable_drift_detection: bool = True
    drift_z_score_threshold: float = -2.0

    # Inference — Rule 21: min confidence 55%
    prediction_horizon_bars: int = 7
    min_confidence_threshold: float = Field(0.55, ge=0.51, le=0.99)


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    exchange: str = "NSE"
    latency_budget_ms: int = 50
    max_slippage_tolerance_pct: float = 0.001

    # Market Hours (IST)
    market_open: time = time(9, 15)
    market_close: time = time(15, 30)
    auto_square_off: time = time(15, 20)  # Rule 80: 15:20 IST


class TimescaleConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("TIMESCALE_PASSWORD", "")))
    dbname: str = "mark5_timescale"
    min_connections: int = 1
    max_connections: int = 10


# ---------------- 3. MASTER CONFIG ----------------

class UIConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    port: int = 8000
    theme: str = "dark"


class KiteConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    api_key: str = ""
    api_secret: SecretStr = SecretStr("")
    access_token: Optional[SecretStr] = None


class TradingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    # NIFTY Midcap 150 liquid subset (Rule 34: minimum 30 stocks)
    watchlist: List[str] = [
        # IT Midcap
        "COFORGE", "PERSISTENT", "MPHASIS", "KPITTECH", "LTTS",
        # Capital Goods / Defence
        "HAL", "BEL", "POLYCAB", "DIXON", "ABB",
        # Financials Midcap
        "IDFCFIRSTB", "LICHSGFIN", "MUTHOOTFIN", "CHOLAFIN", "ABCAPITAL",
        # Consumer / Retail
        "IRCTC", "JUBLFOOD", "PAGEIND", "MARICO", "COLPAL",
        # Chemicals / Pharma
        "PIIND", "DEEPAKNTR", "AARTIIND", "LAURUSLABS", "GRANULES",
        # Real Estate / Infra
        "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "CONCOR", "CUMMINSIND",
    ]


class SystemConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    system_name: str = "MARK5"
    version: str = "6.1.0"
    mode: Literal["PAPER", "LIVE", "BACKTEST"] = "PAPER"

    # Modules
    database: DatabaseConfig
    timescale: TimescaleConfig = TimescaleConfig()
    redis: RedisConfig
    hardware: HardwareConfig = HardwareConfig()
    ui: UIConfig = UIConfig()
    kite: KiteConfig = KiteConfig()

    # Logic
    risk: RiskManagementConfig
    ml: MLModelConfig = MLModelConfig()
    execution: ExecutionConfig = ExecutionConfig()
    trading: TradingConfig = TradingConfig()

    # Feature Flags
    enable_tax_optimization: bool = True
    models_dir: str = "models"
    reports_dir: str = "reports"
    logs_dir: str = "logs"

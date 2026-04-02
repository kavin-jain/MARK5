"""
MARK5 SYSTEM VALIDATORS v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Standardized header, production certification
- [Previous] v6.0: Architect Grade (The DNA)

TRADING ROLE: Configuration Schema & Integrity Enforcement
SAFETY LEVEL: CRITICAL - Invalid config halts system

FEATURES:
✅ Pydantic-based Strict Typing
✅ Tax Engine Optimization Flags
✅ HFT Latency Constraints
"""

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from datetime import time

# ---------------- 1. INFRASTRUCTURE (The Iron) ----------------

class DatabaseConfig(BaseModel):
    path: str = "database/main/mark5.db"
    connection_timeout: int = 10 # Reduced from 30
    wal_mode: bool = True
    synchronous: str = "NORMAL" # Performance optimization over 'FULL' safety
    
    class Config: frozen = True

class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    password: Optional[SecretStr] = None 
    # HFT Requirement: Fail fast. If cache is slow, skip it.
    socket_timeout: float = Field(0.05, le=0.1) 
    max_connections: int = 100
    
    class Config: frozen = True

class HardwareConfig(BaseModel):
    """New: Controls for the AI Engines"""
    enable_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    num_worker_threads: int = Field(8, ge=2)
    use_quantization: bool = False # For faster inference at slight accuracy cost

    class Config: frozen = True

# ---------------- 2. TRADING LOGIC (The Brain) ----------------

class RiskManagementConfig(BaseModel):
    initial_capital: float = Field(..., gt=0)
    
    # Kelly Criterion Settings
    enable_kelly_sizing: bool = True
    kelly_fraction: float = Field(0.5, gt=0.0, le=1.0) # Half-Kelly is industry standard safety
    
    # Circuit Breakers
    max_drawdown_hard_stop: float = Field(0.05, gt=0.0, le=0.2) # Stop system at 5% DD
    max_daily_loss: float = Field(0.02, gt=0.0, le=0.1)
    
    # Per Trade Limits
    max_position_size_pct: float = 0.1
    default_stop_loss_pct: float = 0.01
    
    # Volatility Guardrails
    max_allowed_vix: float = 35.0 # Don't trade if India VIX > 35 (System breaks)
    
    @model_validator(mode='after')
    def validate_safety(self):
        if self.max_daily_loss >= self.max_drawdown_hard_stop:
            raise ValueError("Daily Loss Limit must be tighter than Total System Hard Stop")
        return self

    class Config: frozen = True

class MLModelConfig(BaseModel):
    """Controls the MARK5 Ensemble"""
    active_models: List[str] = ["xgboost", "lightgbm", "lstm", "tcn"]
    retrain_interval_days: int = 7
    
    # Drift Detection
    enable_drift_detection: bool = True
    drift_z_score_threshold: float = -2.0 # Retrain if performance drops 2 sigma
    
    # Inference
    prediction_horizon_bars: int = 7
    min_confidence_threshold: float = Field(0.60, ge=0.51, le=0.99)
    
    class Config: frozen = True

class ExecutionConfig(BaseModel):
    exchange: str = "NSE"
    
    # HFT Settings
    latency_budget_ms: int = 50 # Max allowed time from Tick -> Order
    max_slippage_tolerance_pct: float = 0.001 # 0.1%
    
    # Market Hours (Indian Standard Time)
    market_open: time = time(9, 15)
    market_close: time = time(15, 30)
    auto_square_off: time = time(15, 15) # 3:15 PM Intraday exit
    
    class Config: frozen = True

class TimescaleConfig(BaseModel):
    enabled: bool = False # Set to True if TimescaleDB is available
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    # SECURITY: Password must be set via config.yaml or environment variable
    # Do not use default password in production
    password: SecretStr = SecretStr("")  # Requires explicit configuration
    dbname: str = "mark5_timescale"
    min_connections: int = 1
    max_connections: int = 10
    
    class Config: frozen = True

# ---------------- 3. MASTER CONFIG ----------------

class UIConfig(BaseModel):
    enabled: bool = False
    port: int = 8000
    theme: str = "dark"
    
    class Config: frozen = True

class KiteConfig(BaseModel):
    api_key: str = ""
    api_secret: SecretStr = SecretStr("")
    access_token: Optional[SecretStr] = None
    
    class Config: frozen = True

class TradingConfig(BaseModel):
    # NIFTY Midcap 150 — liquid subset (≥₹500cr daily ADV, RULE 4 compliant)
    # Mid-caps offer stronger momentum signals and less efficient pricing vs NIFTY50
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
    
    class Config: frozen = True

class SystemConfig(BaseModel):
    system_name: str = "MARK5"
    version: str = "6.0.0 (Architect)"
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
    enable_tax_optimization: bool = True # Use the IndianTaxEngine
    # Output Directories
    models_dir: str = "models"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    
    class Config: frozen = True

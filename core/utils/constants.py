"""
MARK5 SYSTEM CONSTANTS v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Standardized header, production certification
- [Previous] v5.0: Bug Fixes & Optimizations

TRADING ROLE: Global Constants & Enums
SAFETY LEVEL: CRITICAL - Hardcoded Trading Rules

FEATURES:
✅ Market Hours & Holidays (India 2025)
✅ Transaction Costs (Brokerage, STT)
✅ Signal Thresholds & Risk Limits
"""

from enum import Enum
from typing import Dict, List

class ModelType(Enum):
    """ML Model Types - PRODUCTION ONLY"""
    XGBOOST = "xgboost"           # Used in advanced ensemble
    LIGHTGBM = "lightgbm"         # Used in advanced ensemble
    RANDOM_FOREST = "randomforest"  # Used in advanced ensemble
    ENSEMBLE = "ensemble"         # Used for weighted ensemble

class SignalType(Enum):
    """Trading Signal Types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class MarketRegime(Enum):
    """Market Regime Types"""
    BULL_MARKET = "BULL_MARKET"
    BEAR_MARKET = "BEAR_MARKET"
    SIDEWAYS_MARKET = "SIDEWAYS_MARKET"
    VOLATILE_MARKET = "VOLATILE_MARKET"

class RiskLevel(Enum):
    """Risk Levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM" 
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

# System Configuration Constants
SYSTEM_CONFIG = {
    "VERSION": "5.0.0",  # Updated with bug fixes and optimizations
    "SYSTEM_NAME": "MARK5 - Advanced AI Stock Prediction System",
    "DEFAULT_MARKET": "NSE",
    "DEFAULT_CURRENCY": "INR",

    # Data Collection
    "DEFAULT_PERIOD": "1y",
    "DEFAULT_INTERVAL": "1d",
    "MAX_RETRIES": 3,
    "REQUEST_TIMEOUT": 30,
    "RATE_LIMIT_DELAY": 1.0,

    # Risk Management
    "MAX_POSITION_SIZE": 0.05,  # RULE 11: 5% of portfolio max
    "DEFAULT_STOP_LOSS": 0.05,  # 5%
    "DEFAULT_TAKE_PROFIT": 0.15,  # 15%
    "MAX_DAILY_LOSS": 0.02,  # 2%
}

# Default Watchlist - NIFTY Midcap 150 (30 stocks across sectors)
DEFAULT_WATCHLIST = [
    "COFORGE.NS", "PERSISTENT.NS", "MPHASIS.NS", "KPITTECH.NS", "LTTS.NS",
    "HAL.NS", "BEL.NS", "POLYCAB.NS", "DIXON.NS", "ABB.NS",
    "IDFCFIRSTB.NS", "LICHSGFIN.NS", "MUTHOOTFIN.NS", "CHOLAFIN.NS", "ABCAPITAL.NS",
    "IRCTC.NS", "JUBLFOOD.NS", "PAGEIND.NS", "MARICO.NS", "COLPAL.NS",
    "PIIND.NS", "DEEPAKNTR.NS", "AARTIIND.NS", "LAURUSLABS.NS", "GRANULES.NS",
    "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "CONCOR.NS", "CUMMINSIND.NS"
]

# Technical Indicators Configuration
TECHNICAL_INDICATORS = {
    "SMA_PERIODS": [5, 10, 20, 50, 200],
    "EMA_PERIODS": [5, 10, 20, 50, 200], 
    "RSI_PERIODS": [14, 21, 30],
    "MACD_CONFIG": {"fast": 12, "slow": 26, "signal": 9},
    "BB_CONFIG": {"period": 20, "std": 2},
    "STOCH_CONFIG": {"k_period": 14, "d_period": 3},
    "ATR_PERIOD": 14,
    "ADX_PERIOD": 14,
    "CCI_PERIOD": 20,
    "WILLIAMS_R_PERIOD": 14,
    "MFI_PERIOD": 14,
    "OBV_ENABLED": True,
    "SAR_CONFIG": {"acceleration": 0.02, "maximum": 0.2}
}

# Model Performance Thresholds
MODEL_THRESHOLDS = {
    "MIN_ACCURACY": 0.60,
    "GOOD_ACCURACY": 0.75,
    "EXCELLENT_ACCURACY": 0.85,
    "MIN_PRECISION": 0.65,
    "MIN_RECALL": 0.65,
    "MIN_F1_SCORE": 0.65
}

# File Paths - PRODUCTION ONLY
# File Paths - PRODUCTION ONLY
PATHS = {
    "BASE_DIR": ".",
    "DATA_DIR": "data",
    "MODELS_DIR": "models", 
    "LOGS_DIR": "logs",
    "CONFIG_DIR": "config",
    "DATABASE_DIR": "database"
}

# Signal Thresholds by Market Regime
SIGNAL_THRESHOLDS = {
    MarketRegime.BULL_MARKET.value: {
        SignalType.STRONG_BUY.value: 0.75,
        SignalType.BUY.value: 0.55, 
        SignalType.HOLD.value: 0.40,
        SignalType.SELL.value: 0.60,
        SignalType.STRONG_SELL.value: 0.75
    },
    MarketRegime.BEAR_MARKET.value: {
        SignalType.STRONG_BUY.value: 0.80,
        SignalType.BUY.value: 0.65,
        SignalType.HOLD.value: 0.35, 
        SignalType.SELL.value: 0.50,
        SignalType.STRONG_SELL.value: 0.70
    },
    MarketRegime.SIDEWAYS_MARKET.value: {
        SignalType.STRONG_BUY.value: 0.78,
        SignalType.BUY.value: 0.60,
        SignalType.HOLD.value: 0.45,
        SignalType.SELL.value: 0.60,
        SignalType.STRONG_SELL.value: 0.78
    },
    MarketRegime.VOLATILE_MARKET.value: {
        SignalType.STRONG_BUY.value: 0.85,
        SignalType.BUY.value: 0.70,
        SignalType.HOLD.value: 0.30,
        SignalType.SELL.value: 0.70,
        SignalType.STRONG_SELL.value: 0.85
    }
}

# Database Configuration
DATABASE_CONFIG = {
    "CONNECTION_POOL_SIZE": 10,
    "CONNECTION_TIMEOUT": 30,
    "QUERY_TIMEOUT": 60,
    "WAL_MODE": True,
    "CACHE_SIZE": 10000,
    "TEMP_STORE": "MEMORY",
    "BACKUP_INTERVAL": 24,  # hours
    "MAX_BACKUP_FILES": 30
}

# 🔥 BUG FIX #3: Feature Exclusion Consistency
# CRITICAL: Must be identical in training and prediction
# Updated: Added 8 extra features to match trained models (188 features)
FEATURE_EXCLUDE_COLUMNS = frozenset({
    'ticker', 'date', 'returns', 'log_returns',
    'close', 'open', 'high', 'low', 'volume',
    'stock splits', 'dividends', 'capital gains',
    # Valid technical indicators should NOT be excluded
    'days_to_month_end', 'is_jan', 'is_dec'
})

# 🔥 BUG FIX #4: Transaction Costs (Indian Stock Market - Equity Delivery)
TRANSACTION_COSTS = {
    "BROKERAGE_PCT": 0.0003,        # 0.03%
    "BROKERAGE_FLAT": 20.0,         # ₹20 flat fee
    "STT_SELL_PCT": 0.00025,        # 0.025% on sell side (intraday — RULE 6)
    "STT_BUY_PCT": 0.000,           # 0% on buy side (equity delivery has no STT on buy usually, but rule 6 says 0.1% sell side)
    "EXCHANGE_CHARGES_NSE": 0.0000325,  # 0.00325% NSE transaction charges (Rule 6)
    "EXCHANGE_CHARGES_BSE": 0.0000325,  
    "GST_PCT": 0.18,                # 18% GST on brokerage + exchange charges
    "SEBI_CHARGES": 0.000001,       # ₹10 per crore turnover
    "STAMP_DUTY": 0.00015,          # 0.015% on buy side (Rule 6)
    "SLIPPAGE_PCT": 0.0005,         # 0.05% expected slippage for Nifty 50 (Rule 7)
    "TOTAL_BUY_COST_PCT": 0.00065,  
    "TOTAL_SELL_COST_PCT": 0.0015  # 0.1% STT dominated
}

# 🔥 BUG FIX #1: Dynamic Cache TTL based on Market Conditions
CACHE_TTL_CONFIG = {
    "MARKET_OPEN_NORMAL": 60,       # 1 minute during trading hours (normal volatility)
    "MARKET_OPEN_HIGH_VOL": 30,     # 30 seconds during high volatility (>3% intraday)
    "MARKET_OPEN_EXTREME_VOL": 15,  # 15 seconds during extreme volatility (>5%)
    "MARKET_CLOSED": 300,           # 5 minutes after hours
    "WEEKEND": 600,                 # 10 minutes on weekends
    "HOLIDAY": 1800,                # 30 minutes on holidays
    "DATA_FRESHNESS_MAX_AGE_TRADING": 15,  # Max 15 minutes old during trading
    "DATA_FRESHNESS_MAX_AGE_CLOSED": 1440   # Max 24 hours old when closed
}

# 🔥 BUG FIX #5: Circuit Breaker Limits (NSE/BSE)
CIRCUIT_BREAKER_LIMITS = {
    "INDIVIDUAL_STOCK_LOWER": 0.05,  # 5% lower circuit
    "INDIVIDUAL_STOCK_UPPER": 0.05,  # 5% upper circuit
    "INDIVIDUAL_STOCK_LOWER_10": 0.10,  # 10% lower (Category II)
    "INDIVIDUAL_STOCK_UPPER_10": 0.10,  # 10% upper (Category II)
    "INDIVIDUAL_STOCK_LOWER_20": 0.20,  # 20% lower (Category III - No circuit)
    "INDIVIDUAL_STOCK_UPPER_20": 0.20,  # 20% upper (Category III - No circuit)
    "INDEX_LEVEL_1": 0.10,           # 10% index movement - 15 min halt
    "INDEX_LEVEL_2": 0.15,           # 15% index movement - 45 min halt
    "INDEX_LEVEL_3": 0.20            # 20% index movement - trading suspended
}

# 🔥 BUG FIX #9: Indian Market Hours (IST - UTC+5:30)
MARKET_HOURS = {
    "TIMEZONE": "Asia/Kolkata",
    "PRE_MARKET_OPEN": "09:00",
    "MARKET_OPEN": "09:15",
    "MARKET_CLOSE": "15:30",
    "POST_MARKET_CLOSE": "16:00",
    "TRADING_DAYS": [0, 1, 2, 3, 4],  # Monday to Friday
    "WEEKEND_DAYS": [5, 6]  # Saturday, Sunday
}

# 🔥 BUG FIX #9: NSE/BSE Holidays 2025 (Indian Stock Market)
NSE_HOLIDAYS_2025 = [
    "2025-01-26",  # Republic Day
    "2025-03-14",  # Holi
    "2025-03-31",  # Id-Ul-Fitr
    "2025-04-10",  # Mahavir Jayanti
    "2025-04-14",  # Dr. Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # Maharashtra Day
    "2025-06-07",  # Id-Ul-Adha (Bakri Id)
    "2025-07-07",  # Muharram
    "2025-08-15",  # Independence Day
    "2025-08-27",  # Ganesh Chaturthi
    "2025-09-05",  # Milad-un-Nabi
    "2025-10-02",  # Mahatma Gandhi Jayanti
    "2025-10-22",  # Dussehra
    "2025-11-01",  # Diwali (Laxmi Pujan)
    "2025-11-04",  # Guru Nanak Jayanti
    "2025-12-25"   # Christmas
]

# RULE 84: NSE Holidays 2026 (Indian Stock Market)
NSE_HOLIDAYS_2026 = [
    "2026-01-26",  # Republic Day
    "2026-03-04",  # Holi
    "2026-03-20",  # Id-Ul-Fitr (tentative)
    "2026-04-02",  # Mahavir Jayanti
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-05-28",  # Id-Ul-Adha (tentative)
    "2026-06-26",  # Muharram (tentative)
    "2026-08-15",  # Independence Day
    "2026-08-26",  # Milad-un-Nabi (tentative)
    "2026-09-16",  # Ganesh Chaturthi
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-12",  # Dussehra
    "2026-10-21",  # Diwali (Laxmi Pujan)
    "2026-10-22",  # Diwali Balipratipada
    "2026-11-24",  # Guru Nanak Jayanti
    "2026-12-25"   # Christmas
]

# Combined holiday list for active use — RULE 84: update annually
NSE_HOLIDAYS = NSE_HOLIDAYS_2025 + NSE_HOLIDAYS_2026

# Model Version Control (BUG FIX #4: Add versioning)
MODEL_VERSION_CONFIG = {
    "VERSION_FORMAT": "v{major}.{minor}.{patch}_{accuracy:.3f}_{timestamp}",
    "KEEP_VERSIONS": 5,  # Keep last 5 versions
    "AUTO_PROMOTE_THRESHOLD": 0.02,  # Promote if 2% better accuracy
    "A_B_TEST_SPLIT": 0.20  # 20% traffic to challenger model
}

# Performance Monitoring
PERFORMANCE_METRICS = {
    "TARGET_ACCURACY": 0.75,        # 75% target accuracy
    "TARGET_PREDICTION_TIME": 1.2,  # <1.2 seconds per prediction
    "TARGET_TRAINING_TIME": 180,    # <3 minutes per stock
    "MAX_MEMORY_MB": 4096,          # 4GB RAM limit
    "MAX_CPU_PERCENT": 80           # 80% CPU usage limit
}

# Alert Thresholds
ALERT_CONFIG = {
    "PRICE_CHANGE_THRESHOLD": 0.03,      # 3% price change alert
    "VOLUME_SPIKE_THRESHOLD": 2.0,       # 2x average volume
    "CONFIDENCE_HIGH": 0.85,             # High confidence alert
    "SIGNAL_REVERSAL": True,             # Alert on BUY→SELL change
    "STOP_LOSS_PROXIMITY": 0.02,         # Alert when 2% away from SL
    "CIRCUIT_BREAKER_ALERT": True,       # Alert on circuit breaker hit
    "MODEL_DISAGREEMENT_THRESHOLD": 0.30  # Alert if models disagree >30%
}

# 🔥 FIX #1: Hardcoded index instrument tokens (Moved from collector.py)
# Note: These tokens are semi-permanent but should be verified periodically
INDEX_INSTRUMENT_TOKENS = {
    'NIFTY50': 256265,      # Nifty 50 Index
    'NIFTY 50': 256265,     # Alternate format
    'NIFTY_50': 256265,     # Alternate format
    'NSEI': 256265,         # NSE code
    'BANKNIFTY': 260105,    # Bank Nifty Index
    'NIFTY BANK': 260105,   # Alternate format
    'NSEBANK': 260105,      # NSE code
    'NIFTYIT': 261641,      # Nifty IT Index
    'CNXIT': 261641,        # NSE code
    # Add more indices as needed
}

# Unified Symbol Mapping
# Maps Broker Token/Symbol -> Internal Unified Symbol
# Format: {Broker_Token_ID: "Internal_Symbol"}
BROKER_TOKENS_MAP = {
    # Indices
    256265: "NSE:NIFTY 50",
    260105: "NSE:NIFTY BANK",
    261641: "NSE:NIFTY IT",
    
    # Midcap Leaders
    1316353: "NSE:COFORGE",
    1150209: "NSE:PERSISTENT",
    257793:  "NSE:MPHASIS",
    1240577: "NSE:KPITTECH",
    35073:   "NSE:LTTS",
    1274369: "NSE:HAL",
    3861249: "NSE:BEL",
    1144833: "NSE:POLYCAB",
    5633:    "NSE:DIXON",
    1281:    "NSE:ABB",
    341249:  "NSE:HDFCBANK", # Keep key benchmarks
    1270529: "NSE:ICICIBANK"
}

# Reverse Map for DataBridge
INSTRUMENT_MAP_REV = BROKER_TOKENS_MAP

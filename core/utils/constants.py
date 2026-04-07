"""
MARK5 SYSTEM CONSTANTS v9.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-04-02] v9.0: Bug fixes
  • FIX: NSE_HOLIDAYS defaulted to 2025-only list. System running in 2026
    would never block on 2026 holidays. Now exports NSE_HOLIDAYS_2026 as
    the active list and NSE_HOLIDAYS as the union of both years.
  • CLEANUP: Removed duplicate INSTRUMENT_MAP_REV alias (same as
    BROKER_TOKENS_MAP). Consumers should use BROKER_TOKENS_MAP directly.
  • CLEANUP: Removed confusing comments about "Production Only" on paths
    that are development defaults.
- [2026-02-06] v8.0: Transaction costs, circuit limits, holiday calendar
"""

from enum import Enum
from typing import Dict, List

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ModelType(Enum):
    XGBOOST      = "xgboost"
    LIGHTGBM     = "lightgbm"
    CATBOOST     = "catboost"
    RANDOM_FOREST = "randomforest"
    ENSEMBLE     = "ensemble"


class SignalType(Enum):
    STRONG_BUY  = "STRONG_BUY"
    BUY         = "BUY"
    HOLD        = "HOLD"
    SELL        = "SELL"
    STRONG_SELL = "STRONG_SELL"


class MarketRegime(Enum):
    TRENDING = "TRENDING"  # BULL/BEAR strong trend
    RANGING  = "RANGING"   # SIDEWAYS / Mean-reversion
    VOLATILE = "VOLATILE"  # HIGH_VOLATILITY
    BEAR     = "BEAR"      # BEAR_MARKET / CRISIS


class RiskLevel(Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_EXCLUDE_COLUMNS = frozenset({
    "ticker", "date", "returns", "log_returns",
    "close", "open", "high", "low", "volume",
    "stock splits", "dividends", "capital gains",
    "days_to_month_end", "is_jan", "is_dec",
})

# ---------------------------------------------------------------------------
# Transaction costs (NSE Equity Intraday — Zerodha-style flat brokerage)
# ---------------------------------------------------------------------------

TRANSACTION_COSTS: Dict[str, float] = {
    "BROKERAGE_PCT":          0.0003,    # 0.03% per order
    "BROKERAGE_FLAT":         20.0,      # ₹20 cap per order leg
    "STT_SELL_PCT":           0.00025,   # 0.025% on sell (intraday)
    "STT_BUY_PCT":            0.000,     # No STT on buy (intraday)
    "EXCHANGE_CHARGES_NSE":   0.0000325, # 0.00325%
    "EXCHANGE_CHARGES_BSE":   0.0000325,
    "GST_PCT":                0.18,      # 18% on brokerage + exchange
    "SEBI_CHARGES":           0.000001,  # ₹10 per crore
    "STAMP_DUTY":             0.00015,   # 0.015% on buy
    "SLIPPAGE_PCT":           0.0005,    # 5bps expected slippage
}

# ---------------------------------------------------------------------------
# Circuit-breaker limits (NSE)
# ---------------------------------------------------------------------------

CIRCUIT_BREAKER_LIMITS: Dict[str, float] = {
    "INDIVIDUAL_STOCK_LOWER":    0.05,
    "INDIVIDUAL_STOCK_UPPER":    0.05,
    "INDIVIDUAL_STOCK_LOWER_10": 0.10,
    "INDIVIDUAL_STOCK_UPPER_10": 0.10,
    "INDIVIDUAL_STOCK_LOWER_20": 0.20,
    "INDIVIDUAL_STOCK_UPPER_20": 0.20,
    "INDEX_LEVEL_1":             0.10,
    "INDEX_LEVEL_2":             0.15,
    "INDEX_LEVEL_3":             0.20,
}

# ---------------------------------------------------------------------------
# Market hours (IST)
# ---------------------------------------------------------------------------

MARKET_HOURS: Dict = {
    "TIMEZONE":        "Asia/Kolkata",
    "PRE_MARKET_OPEN": "09:00",
    "MARKET_OPEN":     "09:15",
    "MARKET_CLOSE":    "15:30",
    "POST_MARKET_CLOSE":"16:00",
    "TRADING_DAYS":    [0, 1, 2, 3, 4],
    "WEEKEND_DAYS":    [5, 6],
}

# ---------------------------------------------------------------------------
# Dynamic cache TTL (seconds)
# ---------------------------------------------------------------------------

CACHE_TTL_CONFIG: Dict[str, int] = {
    "MARKET_OPEN_NORMAL":      60,
    "MARKET_OPEN_HIGH_VOL":    30,
    "MARKET_OPEN_EXTREME_VOL": 15,
    "MARKET_CLOSED":           300,
    "WEEKEND":                 600,
    "HOLIDAY":                 1800,
    "DATA_FRESHNESS_MAX_AGE_TRADING": 15,   # minutes
    "DATA_FRESHNESS_MAX_AGE_CLOSED":  1440, # minutes (24 h)
}

# ---------------------------------------------------------------------------
# NSE Holidays
# FIX: Active year is 2026. Both years are included so the system works
# correctly for data spanning 2025 and 2026.
# ---------------------------------------------------------------------------

NSE_HOLIDAYS_2025: List[str] = [
    "2025-01-26",  # Republic Day
    "2025-03-14",  # Holi
    "2025-03-31",  # Id-Ul-Fitr
    "2025-04-10",  # Mahavir Jayanti
    "2025-04-14",  # Dr. Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # Maharashtra Day
    "2025-06-07",  # Id-Ul-Adha
    "2025-07-07",  # Muharram
    "2025-08-15",  # Independence Day
    "2025-08-27",  # Ganesh Chaturthi
    "2025-09-05",  # Milad-un-Nabi
    "2025-10-02",  # Gandhi Jayanti
    "2025-10-22",  # Dussehra
    "2025-11-01",  # Diwali (Laxmi Pujan)
    "2025-11-04",  # Guru Nanak Jayanti
    "2025-12-25",  # Christmas
]

NSE_HOLIDAYS_2026: List[str] = [
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
    "2026-10-02",  # Gandhi Jayanti
    "2026-10-12",  # Dussehra
    "2026-10-21",  # Diwali (Laxmi Pujan)
    "2026-10-22",  # Diwali Balipratipada
    "2026-11-24",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
]

# Union of both years — use this everywhere
NSE_HOLIDAYS: List[str] = sorted(set(NSE_HOLIDAYS_2025 + NSE_HOLIDAYS_2026))

# ---------------------------------------------------------------------------
# Index instrument tokens
# ---------------------------------------------------------------------------

INDEX_INSTRUMENT_TOKENS: Dict[str, int] = {
    "NIFTY50":   256265,
    "NIFTY 50":  256265,
    "NIFTY_50":  256265,
    "NSEI":      256265,
    "BANKNIFTY": 260105,
    "NIFTY BANK":260105,
    "NSEBANK":   260105,
    "NIFTYIT":   261641,
    "CNXIT":     261641,
}

# Broker token → "EXCHANGE:SYMBOL" map
BROKER_TOKENS_MAP: Dict[int, str] = {
    256265:  "NSE:NIFTY 50",
    260105:  "NSE:NIFTY BANK",
    261641:  "NSE:NIFTY IT",
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
    341249:  "NSE:HDFCBANK",
    1270529: "NSE:ICICIBANK",
}

# ---------------------------------------------------------------------------
# Default watchlist — NIFTY Midcap 150 liquid subset (RULE 4 compliant)
# ---------------------------------------------------------------------------

DEFAULT_WATCHLIST: List[str] = [
    "COFORGE.NS",    "PERSISTENT.NS", "MPHASIS.NS",    "KPITTECH.NS",  "LTTS.NS",
    "HAL.NS",        "BEL.NS",        "POLYCAB.NS",    "DIXON.NS",     "ABB.NS",
    "IDFCFIRSTB.NS", "LICHSGFIN.NS",  "MUTHOOTFIN.NS", "CHOLAFIN.NS",  "ABCAPITAL.NS",
    "IRCTC.NS",      "JUBLFOOD.NS",   "PAGEIND.NS",    "MARICO.NS",    "COLPAL.NS",
    "PIIND.NS",      "DEEPAKNTR.NS",  "AARTIIND.NS",   "LAURUSLABS.NS","GRANULES.NS",
    "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS",   "CONCOR.NS",    "CUMMINSIND.NS",
]

# ---------------------------------------------------------------------------
# Signal thresholds, model performance thresholds, paths
# (kept for backwards compatibility — prefer ConfigManager for new code)
# ---------------------------------------------------------------------------

SIGNAL_THRESHOLDS: Dict[str, Dict[str, float]] = {
    MarketRegime.TRENDING.value: {
        SignalType.STRONG_BUY.value:  0.75,
        SignalType.BUY.value:         0.55,
        SignalType.HOLD.value:        0.40,
        SignalType.SELL.value:        0.60,
        SignalType.STRONG_SELL.value: 0.75,
    },
    MarketRegime.BEAR.value: {
        SignalType.STRONG_BUY.value:  0.80,
        SignalType.BUY.value:         0.65,
        SignalType.HOLD.value:        0.35,
        SignalType.SELL.value:        0.50,
        SignalType.STRONG_SELL.value: 0.70,
    },
    MarketRegime.RANGING.value: {
        SignalType.STRONG_BUY.value:  0.78,
        SignalType.BUY.value:         0.60,
        SignalType.HOLD.value:        0.45,
        SignalType.SELL.value:        0.60,
        SignalType.STRONG_SELL.value: 0.78,
    },
    MarketRegime.VOLATILE.value: {
        SignalType.STRONG_BUY.value:  0.85,
        SignalType.BUY.value:         0.70,
        SignalType.HOLD.value:        0.30,
        SignalType.SELL.value:        0.70,
        SignalType.STRONG_SELL.value: 0.85,
    },
}

MODEL_THRESHOLDS: Dict[str, float] = {
    "MIN_ACCURACY":   0.60,
    "GOOD_ACCURACY":  0.75,
    "EXCELLENT_ACCURACY": 0.85,
    "MIN_PRECISION":  0.65,
    "MIN_RECALL":     0.65,
    "MIN_F1_SCORE":   0.65,
}

PATHS: Dict[str, str] = {
    "BASE_DIR":    ".",
    "DATA_DIR":    "data",
    "MODELS_DIR":  "models",
    "LOGS_DIR":    "logs",
    "CONFIG_DIR":  "config",
    "DATABASE_DIR":"database",
}
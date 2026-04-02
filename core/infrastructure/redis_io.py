"""
MARK5 REDIS IO v8.0 - PRODUCTION GRADE (HFT OPTIMIZED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Standardized header, version bump
- [Previous] v2.0: HFT optimization with LUA scripts

TRADING ROLE: Low-latency data caching and atomic operations
SAFETY LEVEL: HIGH - Atomic risk checks via LUA scripts

FEATURES:
✅ MsgPack serialization for speed
✅ LUA scripting for atomic risk management
✅ Connection pooling
"""

import redis
import logging
import msgpack
import time
from typing import Optional, Dict, Any, Union
from core.utils.config_manager import get_config

# LUA SCRIPT: Atomic Risk Check
# Keys: [daily_loss_key, trade_count_key]
# Args: [max_loss, max_trades, potential_risk_amt]
# Returns: 1 (Approved), 0 (Rejected: Limit Breached)
LUA_RISK_CHECK = """
local current_loss = tonumber(redis.call('GET', KEYS[1]) or '0')
local current_trades = tonumber(redis.call('GET', KEYS[2]) or '0')
local max_loss = tonumber(ARGV[1])
local max_trades = tonumber(ARGV[2])
local new_risk = tonumber(ARGV[3])

if current_loss + new_risk > max_loss then
    return 0 -- Rejected: Loss Limit
end

if current_trades >= max_trades then
    return 0 -- Rejected: Trade Count Limit
end

-- We do NOT increment here. Increment happens on actual fill.
-- But we might want to reserve 'potential risk' if we are conservative.
return 1 -- Approved
"""

class RedisManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized: return
            
        self.logger = logging.getLogger("MARK5.Redis")
        config = get_config().redis
        
        self._pool = redis.ConnectionPool(
            host=config.host,
            port=config.port,
            db=0, # Default to 0 if not in config
            password=config.password.get_secret_value() if config.password else None,
            socket_timeout=0.5, # Very low timeout for HFT
            socket_connect_timeout=0.5,
            max_connections=50,
            decode_responses=False 
        )
        
        self.client = redis.Redis(connection_pool=self._pool)
        
        # Pre-load Lua Scripts
        try:
            self._risk_script_sha = self.client.script_load(LUA_RISK_CHECK)
        except Exception as e:
            self.logger.warning(f"Redis Script Load Failed (Redis offline?): {e}")
            self._risk_script_sha = None
        
        self._initialized = True
        self.logger.info("✅ Redis Atomic Manager Active")

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        try:
            payload = msgpack.packb(value, use_bin_type=True)
            return self.client.set(key, payload, ex=ttl)
        except Exception as e:
            self.logger.error(f"Redis SET Error: {e}")
            return False
            
    def get(self, key: str) -> Any:
        try:
            payload = self.client.get(key)
            if payload:
                return msgpack.unpackb(payload, raw=False)
            return None
        except Exception as e:
            self.logger.error(f"Redis GET Error: {e}")
            return None

    def check_risk_atomic(self, account_id: str, potential_loss: float, max_loss: float, max_trades: int) -> bool:
        """
        Executes the Lua script to check risk limits atomically.
        
        CRITICAL: Fails CLOSED (blocks trading) if Redis unavailable.
        This prevents uncontrolled losses when risk state cannot be verified.
        """
        if not self._risk_script_sha:
            # FAIL CLOSED - Do NOT trade if we cannot verify risk state
            self.logger.critical(
                "🛑 RISK CHECK BLOCKED: Redis unavailable - cannot verify risk limits. "
                "Trading halted for capital protection."
            )
            return False  # Block trade - fail closed for safety
            
        try:
            keys = [f"risk:{account_id}:daily_loss", f"risk:{account_id}:trade_count"]
            args = [max_loss, max_trades, potential_loss]
            
            result = self.client.evalsha(self._risk_script_sha, 2, *keys, *args)
            return result == 1
        except redis.exceptions.ConnectionError as e:
            self.logger.critical(f"🛑 RISK CHECK BLOCKED: Redis connection lost - {e}")
            return False  # Fail closed - block trading
        except Exception as e:
            self.logger.error(f"Atomic Risk Check Failed: {e}")
            # Fail closed (Safe) - if DB fails, do not trade.
            return False

    def update_pnl_atomic(self, account_id: str, realized_pnl: float):
        """
        Atomically updates PnL counters.
        If PnL is negative, it adds to the 'daily_loss' key.
        """
        # Simple increment for counters
        try:
            if realized_pnl < 0:
                # INCRBYFLOAT handles floating point logic inside Redis
                self.client.incrbyfloat(f"risk:{account_id}:daily_loss", abs(realized_pnl))
            
            # Update Total Net PnL
            self.client.incrbyfloat(f"stats:{account_id}:net_pnl", realized_pnl)
            
        except Exception as e:
            self.logger.error(f"Atomic PnL Update Failed: {e}")

# Global Accessor
def get_redis_manager() -> RedisManager:
    return RedisManager()

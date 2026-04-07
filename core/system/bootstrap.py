"""
MARK5 SYSTEM BOOTSTRAPPER v9.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-04-02] v9.0: Bug fix
  • FIX L-07: RedisStreamProducer was constructed as
    RedisStreamProducer(redis.client) — passing the Redis client object as
    a positional arg. RedisStreamProducer.__init__ takes no arguments; it
    fetches the RedisManager singleton internally via get_redis_manager().
    Passing redis.client was silently overwriting the internal redis_manager
    reference with a raw client object, causing method resolution errors
    when the producer tried to call self.redis_manager.client.xadd().
- [2026-02-06] v8.0: Dependency Injection Wiring

TRADING ROLE: Service Locator & Dependency Injection
SAFETY LEVEL: CRITICAL - Initializes all core services
"""

import logging
import os
from kiteconnect import KiteConnect

from core.system.container import container
from core.utils.config_manager import get_config
from core.infrastructure.redis_io import RedisManager
from core.infrastructure.db_io import TimescaleManager
from core.data.feed_manager import FeedManager
from core.execution.order_manager import OrderManager
from core.utils.time_sync import TimeSyncManager
from core.infrastructure.alerts import alert_manager


def bootstrap_system(env: str = "prod"):
    """
    Wire all core services into the dependency-injection container.

    Call order matters — services that depend on others must be registered
    after their dependencies.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MARK5.Boot")
    logger.info(f"Booting MARK5 in '{env}' mode...")

    # 0. Pre-flight: Kite Session Validation
    _validate_kite_session(logger)

    # 1. Configuration
    config = get_config()
    container.register("config", config)

    # 2. Time sync
    time_sync = TimeSyncManager()
    container.register("time", time_sync)

    # 3. Alerts
    container.register("alerts", alert_manager)

    # 4. Infrastructure
    redis = RedisManager()
    container.register("redis", redis)

    db = TimescaleManager()
    container.register("db", db)

    # 5. Data layer
    # FIX L-07: RedisStreamProducer takes no constructor arguments.
    # It fetches RedisManager via get_redis_manager() internally.
    from core.infrastructure.streams import RedisStreamProducer
    stream_producer = RedisStreamProducer()   # was: RedisStreamProducer(redis.client)

    feed_manager = FeedManager(stream_producer)
    container.register("feed_manager", feed_manager)

    from core.data.provider import DataProvider
    data_provider = DataProvider(config)
    container.register("data", data_provider)

    # 6. Risk layer
    from core.trading.risk_manager import RiskManager
    risk_cfg = getattr(config, "risk", {})
    if hasattr(risk_cfg, "model_dump"):
        risk_cfg = risk_cfg.model_dump()
    elif hasattr(risk_cfg, "dict"):
        risk_cfg = risk_cfg.dict()
    risk_manager = RiskManager(risk_cfg if isinstance(risk_cfg, dict) else {})
    container.register("risk_manager", risk_manager)

    # 7. Execution layer
    exec_cfg = getattr(config, "execution", {})
    if hasattr(exec_cfg, "model_dump"):
        exec_cfg = exec_cfg.model_dump()
    elif hasattr(exec_cfg, "dict"):
        exec_cfg = exec_cfg.dict()

    oms = OrderManager(exec_cfg if isinstance(exec_cfg, dict) else {})
    container.register("oms", oms)

    logger.info("✅ System bootstrapped successfully.")
    return container


def _validate_kite_session(logger):
    """Verify that Kite API credentials and access token are present and valid."""
    api_key = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    if not api_key or not access_token:
        logger.error("❌ CRITICAL: Missing KITE_API_KEY or KITE_ACCESS_TOKEN in environment.")
        logger.error("👉 Run 'python3 core/utils/tools/generate_kite_token.py' to authenticate.")
        raise RuntimeError("Authentication Missing: Kite Connect session not found.")

    try:
        # Quick validation: initialize client and check profile (if possible)
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        # Note: We don't call kite.profile() here to avoid unnecessary API hits 
        # during every bootstrap, but we verify the tokens are at least set.
        # Deep validation happens in the DataProvider/OrderManager.
        logger.info("🔐 Kite session tokens detected.")
    except Exception as e:
        logger.error(f"❌ CRITICAL: Kite session validation failed: {e}")
        raise


if __name__ == "__main__":
    bootstrap_system()
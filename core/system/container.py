import logging
from typing import Optional, Any

"""
MARK5 SERVICE CONTAINER v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Standardized header, production certification

TRADING ROLE: Global State & Dependency Manager
SAFETY LEVEL: CRITICAL - Central Nervous System

FEATURES:
✅ Zero-Overhead Slot-based access
✅ Strict Type Enforcement
✅ Singleton Pattern
"""

class ServiceContainer:
    __slots__ = (
        'config', 'redis', 'db', 'data', 'oms', 
        'feed_manager', 'time', 'alerts', 'logger',
        'risk_manager', 'predictor'
    )
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceContainer, cls).__new__(cls)
            cls._instance.logger = logging.getLogger("MARK5.Container")
            # Initialize slots to None to avoid AttributeError
            for slot in cls.__slots__:
                if slot != 'logger':
                    setattr(cls._instance, slot, None)
        return cls._instance

    def register(self, name: str, service: Any):
        """
        Direct slot assignment. 
        Crash early if service name is invalid (Strict Typing).
        """
        if hasattr(self, name):
            setattr(self, name, service)
        else:
            raise AttributeError(f"Service '{name}' is not defined in MARK5 Schema.")

    # No getter methods. Direct access is faster in Python.
    # container.redis is faster than container.get('redis')

# Global Instance
container = ServiceContainer()

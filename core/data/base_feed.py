"""
MARK5 BASE DATA FEED v9.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-04-02] v9.0: Bug fix
  • FIX L-06: _broadcast_ticks() used bare print() for a critical error —
    invisible in production log files and violates the MARK5 logging standard.
    Replaced with logging.getLogger("MARK5.Feed").error().

TRADING ROLE: Abstract interface for all data feed adapters
SAFETY LEVEL: CRITICAL - Observer pattern for tick distribution
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

_feed_logger = logging.getLogger("MARK5.Feed")


@dataclass
class TickData:
    """
    Universal tick structure.
    Every adapter MUST convert raw exchange data into this format.
    """
    symbol:    str
    token:     int
    timestamp: datetime
    ltp:       float
    volume:    int
    bid:       float
    ask:       float
    oi:        Optional[float] = 0.0


class BaseFeed(ABC):
    """
    Abstract base class for all data feed adapters.
    Implements the Observer pattern for distributing ticks to strategies.
    """

    def __init__(self, name: str, config: Dict):
        self.name         = name
        self.config       = config
        self.is_connected = False
        self._tick_observers: List[Callable[[List[TickData]], None]] = []

    def register_tick_observer(
        self, callback: Callable[[List[TickData]], None]
    ) -> None:
        """Register a strategy callback to receive tick updates."""
        self._tick_observers.append(callback)

    def _broadcast_ticks(self, ticks: List[TickData]) -> None:
        """
        Push normalized ticks to all registered observers.

        FIX L-06: Was using bare print() — now uses the named MARK5 logger
        so failures are visible in structured log output.
        """
        for callback in self._tick_observers:
            try:
                callback(ticks)
            except Exception as exc:
                # Never let a subscriber crash the feed
                _feed_logger.error(
                    f"[{self.name}] Subscriber {callback!r} failed to process tick: {exc}",
                    exc_info=True,
                )

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection and build instrument map. Returns True on success."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection and release resources."""

    @abstractmethod
    def fetch_ohlcv(
        self, symbol: str, period: str, interval: str
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Returns:
            DataFrame with tz-aware DatetimeIndex and columns
            [open, high, low, close, volume].
        """

    @abstractmethod
    def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to real-time ticks for the given symbols."""

    @abstractmethod
    def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from real-time ticks."""

    @abstractmethod
    def get_instrument_token(self, symbol: str) -> Optional[int]:
        """O(1) symbol → instrument token lookup."""

    @abstractmethod
    def get_health(self) -> Dict:
        """Return a health status dict for monitoring."""
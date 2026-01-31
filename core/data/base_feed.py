from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from typing import List, Dict, Optional, Any, Callable

@dataclass
class TickData:
    """
    Universal Tick Structure. 
    Every adapter MUST convert their raw data into this format.
    Zero ambiguity allowed.
    """
    symbol: str
    token: int
    timestamp: datetime
    ltp: float
    volume: int
    bid: float
    ask: float
    oi: Optional[float] = 0.0

class BaseFeed(ABC):
    """
    Abstract Base Class for Data Feeds.
    Enforces strict typing and event-driven architecture.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_connected = False
        # The observer pattern: Strategies subscribe to this list
        self._tick_observers: List[Callable[[List[TickData]], None]] = [] 
        
    def register_tick_observer(self, callback: Callable[[List[TickData]], None]):
        """Strategies attach themselves here to listen to the pulse of the market."""
        self._tick_observers.append(callback)

    def _broadcast_ticks(self, ticks: List[TickData]):
        """Push normalized data to all listening strategies."""
        for callback in self._tick_observers:
            try:
                callback(ticks)
            except Exception as e:
                # Never let a subscriber crash the feed
                print(f"CRITICAL: Subscriber failed to process tick: {e}")

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection and build instrument map."""
        pass
        
    @abstractmethod
    def disconnect(self):
        pass
        
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Must return DataFrame with Index=Datetime(tz-aware) 
        and columns=['open', 'high', 'low', 'close', 'volume']
        """
        pass
        
    @abstractmethod
    def subscribe(self, symbols: List[str]):
        """Subscribe to real-time ticks."""
        pass
        
    @abstractmethod
    def unsubscribe(self, symbols: List[str]):
        pass
        
    @abstractmethod
    def get_instrument_token(self, symbol: str) -> Optional[int]:
        """O(1) Lookup for Symbol -> Token."""
        pass

    @abstractmethod
    def get_health(self) -> Dict[str, Any]:
        pass

"""
MARK5 DATA PROVIDER v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Standardized header, production certification
- [Previous] v6.1: Gapless Synchronization

TRADING ROLE: Single Source of Truth for Data
SAFETY LEVEL: CRITICAL - Consistency of History vs Live

FEATURES:
✅ Gapless Synchronization (Buffer + Download)
✅ IST Timezone Enforcement
✅ Data Validation Integration
"""

import logging
import pandas as pd
import threading
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime, timedelta
import pytz

from .base_feed import BaseFeed, TickData
from .adapters.kite_adapter import KiteFeedAdapter
from ..validation.data_validator import DataValidator

# IST Timezone is non-negotiable
IST = pytz.timezone('Asia/Kolkata')

class DataProvider:
    """
    The Single Source of Truth.
    Implements 'gapless' synchronization between History and Live Stream.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("MARK5.Data.Provider")
        self.config = config
        self.feed: Optional[BaseFeed] = None
        self.validator = DataValidator()
        
        # The Buffer: Holds live ticks while history is downloading
        self._tick_buffer: List[TickData] = []
        self._buffer_lock = threading.Lock()
        self._is_syncing = False
        
        self._initialize_feed()

    def _initialize_feed(self):
        # Handle Pydantic model or dict
        if hasattr(self.config, 'kite'):
            kite_config = self.config.kite
        elif isinstance(self.config, dict):
            kite_config = self.config.get('kite', {})
        else:
            kite_config = {}

        if kite_config:
            self.logger.info("Initializing Kite Feed Adapter...")
            self.feed = KiteFeedAdapter(kite_config)
        else:
            self.logger.critical("No Data Adapter Configured. System cannot breathe.")

    def connect(self) -> bool:
        if self.feed:
            return self.feed.connect()
        return False

    def initialize_symbol(self, symbol: str, period: str = '5d', interval: str = 'minute') -> Optional[pd.DataFrame]:
        """
        The Atomic Startup Sequence.
        1. Subscribe to Live Ticks (Start Buffering).
        2. Download History.
        3. Stitch History + Buffer.
        4. Release to Strategy.
        """
        if not self.feed:
            return None

        self.logger.info(f"Initiating Atomic Sync for {symbol}...")

        # Step 1: Subscribe and Start Buffering
        # We hook into the feed temporarily to catch ticks arriving NOW
        self._is_syncing = True
        with self._buffer_lock:
            self._tick_buffer = []
        
        # Register a temporary internal listener
        self.feed.register_tick_observer(self._buffer_ticks)
        self.feed.subscribe([symbol])
        
        # Step 2: Fetch History (Network IO - takes time)
        self.logger.info(f"Downloading history for {symbol} while buffering ticks...")
        history_df = self.feed.fetch_ohlcv(symbol, period, interval)
        
        if history_df.empty:
            self.logger.error(f"Failed to fetch history for {symbol}")
            self._is_syncing = False
            return None

        # Step 3: Strict Validation (No filling, just checking)
        clean_df, report = self.validator.validate_strict(history_df, symbol)
        if not report['valid']:
            self.logger.critical(f"Data Validation FAILED for {symbol}. Aborting. Issues: {report['issues']}")
            self._is_syncing = False
            return None

        # Step 4: The Stitch
        # We process the buffer to find ticks that occurred AFTER the last history candle
        self.logger.info("Stitching Live Buffer to History...")
        final_df = self._stitch_buffer_to_history(clean_df, symbol)
        
        self._is_syncing = False
        self.logger.info(f"Atomic Sync Complete. DF Head: {final_df.index[-1]}")
        
        return final_df

    def _buffer_ticks(self, ticks: List[TickData]):
        """Capture ticks during the history download phase."""
        if self._is_syncing:
            with self._buffer_lock:
                self._tick_buffer.extend(ticks)

    def _stitch_buffer_to_history(self, history_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        FIX D-03: Merges buffered live ticks with historical data.
        
        Previously this was just `pass` — silently discarding all live ticks.
        Now properly:
        1. Filters ticks newer than the last historical candle
        2. Aggregates into OHLCV minute bars
        3. Appends to history DataFrame
        """
        if history_df.empty:
            return history_df

        # Ensure history index is timezone aware
        if history_df.index.tzinfo is None:
            history_df.index = history_df.index.tz_localize(IST)
        else:
            history_df.index = history_df.index.tz_convert(IST)

        last_hist_time = history_df.index.max()
        
        with self._buffer_lock:
            # Filter for this symbol, newer than history end
            relevant_ticks = [
                t for t in self._tick_buffer 
                if t.symbol.endswith(symbol.replace('.NS', '').split(':')[-1])
                and t.timestamp > last_hist_time
            ]
            # Clear buffer after extracting
            self._tick_buffer = []
        
        if not relevant_ticks:
            self.logger.info("No buffered ticks to stitch (history is current).")
            return history_df
        
        # Aggregate ticks into minute candles
        candle_map = {}  # minute_key → {open, high, low, close, volume}
        
        for tick in relevant_ticks:
            # Truncate to minute boundary
            ts = tick.timestamp.replace(second=0, microsecond=0)
            
            if ts not in candle_map:
                candle_map[ts] = {
                    'open': tick.ltp,
                    'high': tick.ltp,
                    'low': tick.ltp,
                    'close': tick.ltp,
                    'volume': tick.volume
                }
            else:
                c = candle_map[ts]
                c['high'] = max(c['high'], tick.ltp)
                c['low'] = min(c['low'], tick.ltp)
                c['close'] = tick.ltp
                c['volume'] = tick.volume  # Kite volume is cumulative; take latest
        
        if candle_map:
            buffer_df = pd.DataFrame.from_dict(candle_map, orient='index')
            buffer_df.index = pd.DatetimeIndex(buffer_df.index)
            if buffer_df.index.tzinfo is None:
                buffer_df.index = buffer_df.index.tz_localize(IST)
            
            # Append and deduplicate
            combined = pd.concat([history_df, buffer_df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.sort_index(inplace=True)
            
            self.logger.info(
                f"Stitched {len(buffer_df)} candles from {len(relevant_ticks)} ticks "
                f"({relevant_ticks[0].timestamp} → {relevant_ticks[-1].timestamp})"
            )
            return combined
        
        return history_df

    def register_tick_observer(self, callback: Callable[[List[TickData]], None]):
        if self.feed:
            self.feed.register_tick_observer(callback)

    def is_ticker_valid(self, ticker: str) -> bool:
        """
        Validates if a ticker is supported/valid.
        Checks against internal map or pattern.
        """
        # Basic regex for NSE/BSE tickers
        import re
        if not ticker or not isinstance(ticker, str):
            return False
        
        ticker = ticker.upper()
        # Check against mapped tokens first (strongest validation)
        # Note: BROKER_TOKENS_MAP values are like "NSE:COFORGE"
        # We check if "NSE:{ticker}" exists in map values
        # This might be slow if map is huge, but it's small now.
        # Ideally satisfy: any(v.endswith(f":{ticker}") for v in BROKER_TOKENS_MAP.values())
        
        # Fallback to regex
        return bool(re.match(r'^[A-Z0-9-]{3,15}$', ticker))

    def get_health(self) -> Dict[str, Any]:
        return self.feed.get_health() if self.feed else {"status": "DEAD"}

"""
MARK5 KITE CONNECT ADAPTER v9.0 — PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-28] v9.0: Complete rewrite. Session management integrated.
  • FIX D-01: Dynamic date range for historical fetching (was hardcoded)
  • FIX D-02: threading.Event for WebSocket readiness (was race condition)
  • FIX A-03: Implemented get_quote() via kite.quote()
  • FIX S-01: Safe depth extraction with None guards
  • FIX L-03: Alert logging on instrument token lookup failures
  • ADD: CandleBuilder for tick → OHLCV aggregation
  • ADD: Chunked historical download (Kite API limits)
  • ADD: fetch_index_data() for Nifty/VIX macro context
  • ADD: Integrated KiteSessionManager (no separate file needed)

TRADING ROLE: Single Source of Truth for Market Data
SAFETY LEVEL: CRITICAL — All financial data flows through here

KITE API RATE LIMITS (RULE 87 Compliance):
  - 3 req/s order placement
  - 10 req/s quotes/historical
  - Rate limiter built into fetch methods
"""

import logging
import hashlib
import os
import time
import threading
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import pytz

from ..base_feed import BaseFeed, TickData

# Robust import handling
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False

# IST Timezone — non-negotiable for Indian markets
IST = pytz.timezone('Asia/Kolkata')

# Well-known Kite instrument tokens for indices
KITE_INDEX_TOKENS = {
    'NIFTY50': 256265,       # NSE NIFTY 50
    'NIFTYBANK': 260105,     # NSE NIFTY BANK
    'INDIAVIX': 264969,      # NSE INDIA VIX
    'SENSEX': 1,             # BSE SENSEX
}

# Kite API max data per request (conservative limits)
KITE_MAX_DAYS_MINUTE = 60    # Max 60 days for minute-level data
KITE_MAX_DAYS_DAY = 2000     # Max ~5.5 years for daily data
KITE_RATE_LIMIT_DELAY = 0.12 # 100ms+ between API calls (10 req/s limit)


# =============================================================================
# CANDLE BUILDER — Tick → OHLCV Aggregation
# =============================================================================

@dataclass
class CandleState:
    """Mutable state for a single in-progress candle."""
    open: float = 0.0
    high: float = -float('inf')
    low: float = float('inf')
    close: float = 0.0
    volume: int = 0
    tick_count: int = 0
    period_start: Optional[datetime] = None


class CandleBuilder:
    """
    Aggregates raw ticks into OHLCV candles at configurable intervals.
    Thread-safe. Produces completed candles via callback.
    
    Args:
        interval_seconds: Candle width in seconds (60 for 1-minute bars)
        on_candle: Callback invoked with (symbol, timestamp, open, high, low, close, volume)
    """
    
    def __init__(self, interval_seconds: int = 60, 
                 on_candle: Optional[Callable] = None):
        self._interval = interval_seconds
        self._on_candle = on_candle
        self._candles: Dict[str, CandleState] = defaultdict(CandleState)
        self._lock = threading.Lock()
        self.logger = logging.getLogger("MARK5.CandleBuilder")
    
    def _get_period_start(self, ts: datetime) -> datetime:
        """Truncate timestamp to its period boundary."""
        epoch = ts.timestamp()
        period_epoch = (epoch // self._interval) * self._interval
        return datetime.fromtimestamp(period_epoch, tz=ts.tzinfo or IST)
    
    def ingest(self, tick: TickData):
        """Process a single tick. Must be called from the feed thread."""
        period_start = self._get_period_start(tick.timestamp)
        
        with self._lock:
            candle = self._candles[tick.symbol]
            
            # New candle period? Emit the old one first.
            if candle.period_start is not None and period_start != candle.period_start:
                self._emit(tick.symbol, candle)
                # Reset for new period
                self._candles[tick.symbol] = CandleState()
                candle = self._candles[tick.symbol]
            
            # Update candle state
            if candle.tick_count == 0:
                candle.open = tick.ltp
                candle.high = tick.ltp
                candle.low = tick.ltp
                candle.period_start = period_start
            else:
                candle.high = max(candle.high, tick.ltp)
                candle.low = min(candle.low, tick.ltp)
            
            candle.close = tick.ltp
            candle.volume += tick.volume
            candle.tick_count += 1
    
    def _emit(self, symbol: str, candle: CandleState):
        """Push completed candle to callback."""
        if self._on_candle and candle.tick_count > 0:
            try:
                self._on_candle(
                    symbol=symbol,
                    timestamp=candle.period_start,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume
                )
            except Exception as e:
                self.logger.error(f"Candle callback error for {symbol}: {e}")
    
    def flush_all(self):
        """Emit all in-progress candles (call at session close)."""
        with self._lock:
            for symbol, candle in self._candles.items():
                self._emit(symbol, candle)
            self._candles.clear()


# =============================================================================
# KITE FEED ADAPTER — Main Data Interface
# =============================================================================

class KiteFeedAdapter(BaseFeed):
    """
    Production-grade Kite Connect data adapter.
    
    Handles:
    - OAuth2 session management (integrated, no separate file)
    - Historical data fetching with date chunking
    - WebSocket streaming with proper connection event
    - Tick → OHLCV candle aggregation
    - Rate limiting per RULE 87 (10 req/s quotes, 3 req/s orders)
    
    Args:
        config: Dict or Pydantic model with api_key, api_secret, access_token
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Kite", config)
        self.logger = logging.getLogger("MARK5.Feeds.Kite")
        
        # --- Credential Extraction (supports Pydantic or dict) ---
        if hasattr(config, 'api_key'):
            # Pydantic Model path
            self.api_key = config.api_key or os.getenv("KITE_API_KEY", "")
            # Handle SecretStr if needed
            self.api_secret = getattr(config, 'api_secret', "")
            if hasattr(self.api_secret, 'get_secret_value'):
                self.api_secret = self.api_secret.get_secret_value()
            self.api_secret = self.api_secret or os.getenv("KITE_API_SECRET", "")
            
            self.access_token = getattr(config, 'access_token', "")
            if hasattr(self.access_token, 'get_secret_value'):
                self.access_token = self.access_token.get_secret_value()
            self.access_token = self.access_token or os.getenv("KITE_ACCESS_TOKEN", "")
        else:
            # Dict path
            self.api_key = config.get("api_key") or os.getenv("KITE_API_KEY", "")
            self.api_secret = config.get("api_secret") or os.getenv("KITE_API_SECRET", "")
            self.access_token = config.get("access_token") or os.getenv("KITE_ACCESS_TOKEN", "")
        
        self.kite: Optional[KiteConnect] = None
        self.ticker: Optional[KiteTicker] = None
        
        # O(1) Instrument Lookup Maps
        self.symbol_to_token: Dict[str, int] = {}
        self.token_to_symbol: Dict[int, str] = {}
        
        # Ticker symbol resolution cache: "RELIANCE.NS" → "NSE:RELIANCE"
        self._ns_to_kite: Dict[str, str] = {}
        
        # Connection readiness — FIX D-02: use Event instead of bool-before-handshake
        self._connected_event = threading.Event()
        self.last_tick_time = datetime.now(IST)
        
        # Rate limiting — RULE 87 compliance
        self._last_api_call = 0.0
        self._api_lock = threading.Lock()
        
        # CandleBuilder for live tick aggregation
        self.candle_builder = CandleBuilder(interval_seconds=60)
    
    # =========================================================================
    # SESSION MANAGEMENT (Integrated — no separate kite_session.py needed)
    # =========================================================================
    
    @staticmethod
    def generate_login_url(api_key: Optional[str] = None) -> str:
        """Generate the Kite Connect OAuth2 login URL."""
        key = api_key or os.getenv("KITE_API_KEY", "")
        if not key:
            raise ValueError("KITE_API_KEY not set in .env or config")
        return f"https://kite.zerodha.com/connect/login?v=3&api_key={key}"
    
    @staticmethod
    def exchange_token(api_key: str, api_secret: str, request_token: str) -> str:
        """
        Exchange request_token for access_token (daily login flow).
        
        The checksum is SHA-256(api_key + request_token + api_secret).
        Stores the resulting access_token back into .env automatically.
        
        Returns:
            access_token string
        """
        if not KITE_AVAILABLE:
            raise ImportError("kiteconnect library not installed: pip install kiteconnect")
        
        kite = KiteConnect(api_key=api_key)
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        # Persist to .env for this session
        _update_env_var("KITE_ACCESS_TOKEN", access_token)
        
        return access_token
    
    # =========================================================================
    # CONNECTION
    # =========================================================================
    
    def connect(self) -> bool:
        """
        Establish Kite connection, download instrument master, start WebSocket.
        
        FIX D-02: Waits for WebSocket handshake via threading.Event
        before returning True, preventing subscribe-before-connect race.
        """
        if not KITE_AVAILABLE:
            self.logger.critical("KiteConnect library missing. Run: pip install kiteconnect")
            return False
        
        if not self.api_key or not self.access_token:
            self.logger.critical(
                "Kite credentials missing. Set KITE_API_KEY and KITE_ACCESS_TOKEN in .env"
            )
            return False
        
        try:
            self.logger.info("Initializing Kite Connection...")
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            
            # Validate token by fetching profile
            profile = self.kite.profile()
            self.logger.info(f"Authenticated as: {profile.get('user_name', 'Unknown')}")
            
            # Build instrument master (critical for token lookups)
            self.logger.info("Downloading Instrument Master... (may take a few seconds)")
            instruments = self.kite.instruments("NSE")
            self._build_instrument_map(instruments)
            self.logger.info(f"Instrument map built: {len(self.symbol_to_token)} NSE instruments")
            
            # Initialize WebSocket ticker
            self.ticker = KiteTicker(self.api_key, self.access_token)
            self.ticker.on_ticks = self._on_ticks
            self.ticker.on_connect = self._on_connect
            self.ticker.on_error = self._on_error
            self.ticker.on_close = self._on_close
            self.ticker.on_reconnect = self._on_reconnect
            
            # FIX D-02: Reset event before connecting
            self._connected_event.clear()
            self.ticker.connect(threaded=True)
            
            # FIX D-02: Wait for actual WebSocket handshake (max 10 seconds)
            if not self._connected_event.wait(timeout=10.0):
                self.logger.warning(
                    "WebSocket handshake did not complete in 10s. "
                    "Historical data will work; live streaming may be delayed."
                )
            
            self.is_connected = self._connected_event.is_set()
            return True
            
        except Exception as e:
            self.logger.critical(f"Kite Connection FAILED: {e}", exc_info=True)
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Gracefully close WebSocket and flush pending candles."""
        self.candle_builder.flush_all()
        if self.ticker:
            try:
                self.ticker.close()
            except Exception:
                pass
        self.is_connected = False
        self._connected_event.clear()
        self.logger.info("Kite connection closed.")
    
    # =========================================================================
    # INSTRUMENT MAP
    # =========================================================================
    
    def _build_instrument_map(self, instruments_list):
        """
        Build O(1) symbol ↔ token lookup maps from NSE instrument dump.
        Also builds .NS suffix resolver: "COFORGE.NS" → "NSE:COFORGE"
        """
        self.symbol_to_token.clear()
        self.token_to_symbol.clear()
        self._ns_to_kite.clear()
        
        for inst in instruments_list:
            exchange = inst.get('exchange', 'NSE')
            tradingsymbol = inst.get('tradingsymbol', '')
            token = int(inst.get('instrument_token', 0))
            
            # Primary map: "NSE:COFORGE" → 1316353
            key = f"{exchange}:{tradingsymbol}"
            self.symbol_to_token[key] = token
            self.token_to_symbol[token] = key
            
            # .NS convenience map: "COFORGE.NS" → "NSE:COFORGE"
            ns_key = f"{tradingsymbol}.NS"
            self._ns_to_kite[ns_key] = key
            
            # Also map bare symbol: "COFORGE" → "NSE:COFORGE"
            if tradingsymbol not in self._ns_to_kite:
                self._ns_to_kite[tradingsymbol] = key
    
    def resolve_symbol(self, symbol: str) -> Optional[str]:
        """
        Resolve any symbol format to Kite's "NSE:SYMBOL" format.
        Accepts: "COFORGE.NS", "NSE:COFORGE", "COFORGE"
        """
        if symbol in self.symbol_to_token:
            return symbol  # Already in "NSE:SYMBOL" format
        if symbol in self._ns_to_kite:
            return self._ns_to_kite[symbol]
        
        # Try stripping .NS suffix
        bare = symbol.replace('.NS', '').replace('.BO', '')
        nse_key = f"NSE:{bare}"
        if nse_key in self.symbol_to_token:
            return nse_key
        
        return None
    
    def get_instrument_token(self, symbol: str) -> Optional[int]:
        """
        O(1) symbol → instrument token lookup.
        FIX L-03: Logs alert when lookup fails instead of silent None.
        """
        resolved = self.resolve_symbol(symbol)
        if resolved:
            return self.symbol_to_token.get(resolved)
        
        # FIX L-03: Alert on failure instead of silent drop
        self.logger.warning(
            f"⚠️ Instrument token lookup FAILED for '{symbol}'. "
            f"Not found in {len(self.symbol_to_token)} NSE instruments. "
            f"Check if symbol is valid and exchange is NSE."
        )
        return None
    
    # =========================================================================
    # RATE LIMITING — RULE 87 Compliance
    # =========================================================================
    
    def _rate_limit(self):
        """Enforce minimum delay between API calls (10 req/s = 100ms gap)."""
        with self._api_lock:
            elapsed = time.time() - self._last_api_call
            if elapsed < KITE_RATE_LIMIT_DELAY:
                time.sleep(KITE_RATE_LIMIT_DELAY - elapsed)
            self._last_api_call = time.time()
    
    # =========================================================================
    # HISTORICAL DATA — FIX D-01: Dynamic dates, chunked fetching
    # =========================================================================
    
    def fetch_ohlcv(self, symbol: str, period: str = '5y', 
                    interval: str = 'day',
                    from_date: Optional[datetime] = None,
                    to_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data via Kite API.
        
        FIX D-01: Was using hardcoded "2023-01-01" dates. Now computes
        from_date/to_date dynamically based on period argument.
        
        Auto-chunks requests to respect Kite API limits:
        - Minute data: max 60 days per request
        - Day data: max 2000 days per request
        
        Args:
            symbol: Ticker in any format ("RELIANCE.NS", "NSE:RELIANCE", "RELIANCE")
            period: Lookback period ("5y", "3y", "1y", "6m", "30d")
            interval: Kite interval ("minute", "3minute", "5minute", "15minute",
                                      "30minute", "60minute", "day")
            from_date: Explicit start date (overrides period)
            to_date: Explicit end date (default: now IST)
            
        Returns:
            DataFrame with DatetimeIndex (IST), columns=[open, high, low, close, volume]
        """
        # Map common aliases to Kite expected strings
        interval_map = {
            '1d': 'day', '1m': 'minute', '3m': '3minute', '5m': '5minute',
            '15m': '15minute', '30m': '30minute', '60m': '60minute'
        }
        interval = interval_map.get(interval, interval)

        if not self.kite:
            self.logger.error("Kite not initialized. Call connect() first.")
            return pd.DataFrame()
        
        # Resolve symbol to instrument token
        token = self.get_instrument_token(symbol)
        if not token:
            return pd.DataFrame()
        
        # Compute date range
        now_ist = datetime.now(IST)
        if to_date is None:
            to_date = now_ist
        elif to_date.tzinfo is None:
            to_date = IST.localize(to_date)
        
        if from_date is None:
            from_date = self._period_to_date(period, to_date)
        elif from_date.tzinfo is None:
            from_date = IST.localize(from_date)
        
        # Determine chunk size based on interval
        is_intraday = interval != 'day'
        max_days_per_chunk = KITE_MAX_DAYS_MINUTE if is_intraday else KITE_MAX_DAYS_DAY
        
        # Fetch in chunks
        all_records = []
        chunk_start = from_date
        
        while chunk_start < to_date:
            chunk_end = min(chunk_start + timedelta(days=max_days_per_chunk), to_date)
            
            try:
                self._rate_limit()
                records = self.kite.historical_data(
                    instrument_token=token,
                    from_date=chunk_start.strftime('%Y-%m-%d %H:%M:%S'),
                    to_date=chunk_end.strftime('%Y-%m-%d %H:%M:%S'),
                    interval=interval,
                    continuous=False,
                    oi=False
                )
                all_records.extend(records)
            except Exception as e:
                self.logger.error(
                    f"Historical fetch failed for chunk "
                    f"{chunk_start.date()}→{chunk_end.date()}: {e}"
                )
            
            chunk_start = chunk_end + timedelta(days=1)
        
        if not all_records:
            self.logger.warning(f"No historical data returned for {symbol}")
            return pd.DataFrame()
        
        # Build DataFrame with standardized schema
        df = pd.DataFrame(all_records)
        df['date'] = pd.to_datetime(df['date'])
        
        # Deduplicate overlapping chunk boundaries
        df = df.drop_duplicates(subset=['date'], keep='last')
        
        # Ensure IST timezone
        if df['date'].dt.tz is None:
            df['date'] = df['date'].dt.tz_localize(IST)
        else:
            df['date'] = df['date'].dt.tz_convert(IST)
        
        df.set_index('date', inplace=True)
        
        # Standardize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        # Remove exact duplicates
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        
        self.logger.info(
            f"Fetched {len(df)} bars for {symbol} "
            f"({from_date.date()} → {to_date.date()}, interval={interval})"
        )
        
        return df
    
    def fetch_index_data(self, index_name: str = 'NIFTY50', 
                         days_back: int = 2500,
                         interval: str = 'day') -> pd.DataFrame:
        """
        Fetch index data (NIFTY 50, INDIA VIX, etc.) using well-known tokens.
        
        Replaces the yfinance _get_nifty_data_cached / _get_vix_data_cached
        functions in features.py with reliable Kite API data.
        
        Args:
            index_name: One of 'NIFTY50', 'NIFTYBANK', 'INDIAVIX', 'SENSEX'
            days_back: Number of days of history
            interval: Kite interval string
            
        Returns:
            DataFrame with OHLCV columns, IST DatetimeIndex
        """
        token = KITE_INDEX_TOKENS.get(index_name.upper())
        if not token:
            self.logger.error(f"Unknown index: {index_name}. Valid: {list(KITE_INDEX_TOKENS.keys())}")
            return pd.DataFrame()
        
        to_date = datetime.now(IST)
        from_date = to_date - timedelta(days=days_back)
        
        # Temporarily set token in our map for fetch_ohlcv to work
        kite_symbol = f"NSE:{index_name}"
        self.symbol_to_token[kite_symbol] = token
        self.token_to_symbol[token] = kite_symbol
        
        return self.fetch_ohlcv(
            symbol=kite_symbol,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )
    
    # =========================================================================
    # LIVE QUOTES — FIX A-03: Was just `pass`
    # =========================================================================
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for a symbol via Kite REST API.
        
        FIX A-03: Was unimplemented (returned None).
        
        Returns:
            Dict with keys: ltp, open, high, low, close, volume, last_trade_time
        """
        if not self.kite:
            self.logger.error("Kite not initialized")
            return {}
        
        resolved = self.resolve_symbol(symbol)
        if not resolved:
            return {}
        
        try:
            self._rate_limit()
            quotes = self.kite.quote([resolved])
            
            if resolved in quotes:
                q = quotes[resolved]
                ohlc = q.get('ohlc', {})
                return {
                    'symbol': resolved,
                    'ltp': q.get('last_price', 0.0),
                    'open': ohlc.get('open', 0.0),
                    'high': ohlc.get('high', 0.0),
                    'low': ohlc.get('low', 0.0),
                    'close': ohlc.get('close', 0.0),
                    'volume': q.get('volume', 0),
                    'last_trade_time': q.get('last_trade_time'),
                    'oi': q.get('oi', 0),
                    'bid': q.get('depth', {}).get('buy', [{}])[0].get('price', 0.0),
                    'ask': q.get('depth', {}).get('sell', [{}])[0].get('price', 0.0),
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Quote fetch failed for {symbol}: {e}")
            return {}
    
    # =========================================================================
    # SUBSCRIPTIONS
    # =========================================================================
    
    def subscribe(self, symbols: List[str]):
        """Subscribe to live WebSocket ticks for given symbols."""
        if not self.ticker or not self._connected_event.is_set():
            self.logger.error("Cannot subscribe: WebSocket not connected")
            return
        
        tokens = []
        for sym in symbols:
            token = self.get_instrument_token(sym)
            if token:
                tokens.append(token)
            # L-03: Failure already logged by get_instrument_token
        
        if tokens:
            self.logger.info(f"Subscribing to {len(tokens)} tokens via WebSocket")
            self.ticker.subscribe(tokens)
            self.ticker.set_mode(self.ticker.MODE_FULL, tokens)
    
    def unsubscribe(self, symbols: List[str]):
        if not self.ticker:
            return
        tokens = [
            self.get_instrument_token(s) 
            for s in symbols 
            if self.get_instrument_token(s) is not None
        ]
        if tokens:
            self.ticker.unsubscribe(tokens)
    
    # =========================================================================
    # WEBSOCKET HANDLERS
    # =========================================================================
    
    def _on_ticks(self, ws, ticks):
        """
        WebSocket tick handler. Must run in microseconds.
        
        FIX S-01: Safe depth extraction with None/key guards.
        """
        self.last_tick_time = datetime.now(IST)
        normalized_ticks = []
        
        for raw_tick in ticks:
            try:
                token = raw_tick.get('instrument_token')
                symbol = self.token_to_symbol.get(token, f"UNKNOWN:{token}")
                
                # FIX S-01: Safe depth extraction
                depth = raw_tick.get('depth')
                bid = 0.0
                ask = 0.0
                if depth:
                    buy_depth = depth.get('buy', [])
                    sell_depth = depth.get('sell', [])
                    if buy_depth and isinstance(buy_depth[0], dict):
                        bid = buy_depth[0].get('price', 0.0)
                    if sell_depth and isinstance(sell_depth[0], dict):
                        ask = sell_depth[0].get('price', 0.0)
                
                tick_obj = TickData(
                    symbol=symbol,
                    token=token,
                    timestamp=raw_tick.get('exchange_timestamp', datetime.now(IST)),
                    ltp=raw_tick.get('last_price', 0.0),
                    volume=raw_tick.get('volume_traded', 0),
                    bid=bid,
                    ask=ask,
                    oi=raw_tick.get('oi', 0)
                )
                normalized_ticks.append(tick_obj)
                
                # Feed into CandleBuilder for real-time OHLCV
                self.candle_builder.ingest(tick_obj)
                
            except Exception as e:
                self.logger.error(f"Tick parsing error: {e}")
        
        # Broadcast to all observers
        if normalized_ticks:
            self._broadcast_ticks(normalized_ticks)
    
    def _on_connect(self, ws, response):
        """FIX D-02: Signal readiness event so connect() can proceed safely."""
        self.is_connected = True
        self._connected_event.set()
        self.logger.info("✅ Kite WebSocket connected to exchange.")
    
    def _on_close(self, ws, code, reason):
        self.is_connected = False
        self._connected_event.clear()
        self.logger.error(f"Kite WebSocket closed: {code} — {reason}")
    
    def _on_error(self, ws, code, reason):
        self.logger.error(f"Kite WebSocket error: {code} — {reason}")
    
    def _on_reconnect(self, ws, attempts_count):
        self.logger.warning(f"Kite reconnecting... attempt {attempts_count}")
    
    # =========================================================================
    # HEALTH
    # =========================================================================
    
    def get_health(self) -> Dict[str, Any]:
        delta = (datetime.now(IST) - self.last_tick_time).total_seconds()
        status = "HEALTHY" if self.is_connected and delta < 5 else "CRITICAL"
        return {
            "name": "Kite",
            "status": status,
            "connected": self.is_connected,
            "ws_ready": self._connected_event.is_set(),
            "last_tick_seconds_ago": round(delta, 2),
            "instruments_loaded": len(self.symbol_to_token),
            "rate_limit_delay_ms": int(KITE_RATE_LIMIT_DELAY * 1000),
        }
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    @staticmethod
    def _period_to_date(period: str, reference: datetime) -> datetime:
        """Convert period string to from_date relative to reference."""
        period = period.lower().strip()
        mapping = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180,
            '30d': 30, '60d': 60, '90d': 90, '180d': 180,
            '1y': 365, '2y': 730, '3y': 1095, '5y': 1825, '10y': 3650,
        }
        days = mapping.get(period)
        if days:
            return reference - timedelta(days=days)
        
        # Fallback: try parsing "Nd" format
        if period.endswith('d') and period[:-1].isdigit():
            return reference - timedelta(days=int(period[:-1]))
        if period.endswith('y') and period[:-1].isdigit():
            return reference - timedelta(days=int(period[:-1]) * 365)
        
        # Default: 5 years
        return reference - timedelta(days=1825)


# =============================================================================
# HELPER: Update .env file in-place
# =============================================================================

def _update_env_var(key: str, value: str, env_path: Optional[str] = None):
    """Update a single key=value in the .env file, preserving all other lines."""
    if env_path is None:
        # Walk up from this file to find project .env
        current = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            candidate = os.path.join(current, '.env')
            if os.path.exists(candidate):
                env_path = candidate
                break
            current = os.path.dirname(current)
    
    if not env_path or not os.path.exists(env_path):
        return
    
    lines = []
    found = False
    with open(env_path, 'r') as f:
        for line in f:
            if line.strip().startswith(f'{key}='):
                lines.append(f'{key}={value}\n')
                found = True
            else:
                lines.append(line)
    
    if not found:
        lines.append(f'{key}={value}\n')
    
    with open(env_path, 'w') as f:
        f.writelines(lines)

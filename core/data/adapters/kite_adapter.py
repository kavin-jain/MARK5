import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from ..base_feed import BaseFeed, TickData

# Robust import handling
try:
    import kiteconnect
    from kiteconnect import KiteConnect, KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False

class KiteFeedAdapter(BaseFeed):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Kite", config)
        self.logger = logging.getLogger("MARK5.Feeds.Kite")
        
        # Handle Pydantic or Dict
        if hasattr(config, 'api_key'):
            self.api_key = config.api_key
            self.access_token = config.access_token
        else:
            self.api_key = config.get("api_key")
            self.access_token = config.get("access_token")
        
        self.kite: Optional[KiteConnect] = None
        self.ticker: Optional[KiteTicker] = None
        
        # O(1) Lookup Optimization
        # format: {"NSE:RELIANCE": 738561, ...}
        self.symbol_to_token: Dict[str, int] = {} 
        # format: {738561: "NSE:RELIANCE", ...}
        self.token_to_symbol: Dict[int, str] = {} 
        
        self.last_tick_time = datetime.now()

    def connect(self) -> bool:
        if not KITE_AVAILABLE:
            self.logger.critical("KiteConnect library missing. System halted.")
            return False
            
        if not self.api_key or not self.access_token:
            self.logger.critical("Kite credentials missing.")
            return False
            
        try:
            self.logger.info("Initializing Kite Connection...")
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            
            # CRITICAL STEP: Build the Instrument Master
            self.logger.info("Downloading Instrument Master Dump... This may take a few seconds.")
            instruments = self.kite.instruments()
            self._build_instrument_map(instruments)
            self.logger.info(f"Instrument Map Built. Loaded {len(self.symbol_to_token)} instruments.")

            # Initialize Ticker
            self.ticker = KiteTicker(self.api_key, self.access_token)
            self.ticker.on_ticks = self._on_ticks
            self.ticker.on_connect = self._on_connect
            self.ticker.on_error = self._on_error
            self.ticker.on_close = self._on_close
            self.ticker.on_reconnect = self._on_reconnect
            
            # Threaded=True is mandatory, but we must not block it
            self.ticker.connect(threaded=True)
            
            # Wait for socket to actually open
            # (In production, we would use an event flag here)
            self.is_connected = True 
            return True
            
        except Exception as e:
            self.logger.critical(f"Kite Connection FAILED: {e}", exc_info=True)
            self.is_connected = False
            return False

    def _build_instrument_map(self, instruments_list):
        """
        Pre-computes symbol mapping for zero-latency lookups during live trading.
        """
        self.symbol_to_token.clear()
        self.token_to_symbol.clear()
        
        for inst in instruments_list:
            # Format: EXCHANGE:TRADINGSYMBOL (e.g., NSE:INFY)
            key = f"{inst['exchange']}:{inst['tradingsymbol']}"
            token = int(inst['instrument_token'])
            
            self.symbol_to_token[key] = token
            self.token_to_symbol[token] = key

    def get_instrument_token(self, symbol: str) -> Optional[int]:
        """High-speed O(1) lookup."""
        return self.symbol_to_token.get(symbol)

    def subscribe(self, symbols: List[str]):
        if not self.ticker or not self.is_connected:
            self.logger.error("Cannot subscribe: Feed not connected")
            return

        tokens = []
        for sym in symbols:
            token = self.get_instrument_token(sym)
            if token:
                tokens.append(token)
            else:
                self.logger.warning(f"Symbol not found in master map: {sym}")
        
        if tokens:
            self.logger.info(f"Subscribing to {len(tokens)} tokens")
            self.ticker.subscribe(tokens)
            # Set mode to Full to get Depth/OI
            self.ticker.set_mode(self.ticker.MODE_FULL, tokens) 

    def unsubscribe(self, symbols: List[str]):
        if not self.ticker: return
        tokens = [self.get_instrument_token(s) for s in symbols if self.get_instrument_token(s)]
        if tokens:
            self.ticker.unsubscribe(tokens)

    def fetch_ohlcv(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Fetches historical data and enforces standard schema.
        """
        token = self.get_instrument_token(symbol)
        if not token:
            self.logger.error(f"Cannot fetch history: Unknown symbol {symbol}")
            return pd.DataFrame()

        try:
            # Map common intervals to Kite syntax
            # TODO: Add robust interval mapper (e.g., '1m' -> 'minute')
            records = self.kite.historical_data(token, from_date="2023-01-01", to_date="2023-01-02", interval=interval)
            
            df = pd.DataFrame(records)
            if df.empty:
                return df
                
            # STANDARDISATION
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            return df
            
        except Exception as e:
            self.logger.error(f"Historical data fetch failed: {e}")
            return pd.DataFrame()

    def _on_ticks(self, ws, ticks):
        """
        The Heartbeat of the system. 
        Must run in microseconds. No complex logic here.
        """
        self.last_tick_time = datetime.now()
        normalized_ticks = []
        
        for raw_tick in ticks:
            try:
                token = raw_tick.get('instrument_token')
                symbol = self.token_to_symbol.get(token, f"UNKNOWN:{token}")
                
                # Normalize to Universal TickData
                tick_obj = TickData(
                    symbol=symbol,
                    token=token,
                    timestamp=raw_tick.get('exchange_timestamp', datetime.now()),
                    ltp=raw_tick.get('last_price', 0.0),
                    volume=raw_tick.get('volume_traded', 0),
                    bid=raw_tick['depth']['buy'][0]['price'] if 'depth' in raw_tick else 0.0,
                    ask=raw_tick['depth']['sell'][0]['price'] if 'depth' in raw_tick else 0.0,
                    oi=raw_tick.get('oi', 0)
                )
                normalized_ticks.append(tick_obj)
            except Exception as e:
                self.logger.error(f"Tick Parsing Error: {e}")
        
        # Blast data to all observers
        if normalized_ticks:
            self._broadcast_ticks(normalized_ticks)

    # --- WebSocket Lifecycle Hooks ---
    def _on_connect(self, ws, response):
        self.is_connected = True
        self.logger.info("Kite Ticker Connected to Exchange.")

    def _on_close(self, ws, code, reason):
        self.is_connected = False
        self.logger.error(f"Kite Ticker Closed: {code} - {reason}")

    def _on_error(self, ws, code, reason):
        self.logger.error(f"Kite Ticker Error: {code} - {reason}")

    def _on_reconnect(self, ws, attempts_count):
        self.logger.warning(f"Kite Reconnecting... Attempt {attempts_count}")

    def get_health(self) -> Dict[str, Any]:
        delta = (datetime.now() - self.last_tick_time).total_seconds()
        status = "HEALTHY" if self.is_connected and delta < 5 else "CRITICAL"
        return {
            "name": "Kite",
            "status": status,
            "connected": self.is_connected,
            "last_tick_seconds_ago": round(delta, 2),
            "instruments_loaded": len(self.symbol_to_token)
        }

    # Implement unimplemented abstract methods to prevent instantiation error
    def disconnect(self):
        if self.ticker:
            self.ticker.close()
        self.is_connected = False
        
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        # Implementation similar to fetch_ohlcv with standardization
        pass

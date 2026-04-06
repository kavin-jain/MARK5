"""
MARK5 Market Data Module — NIFTY50 Index + Sector ETF Download
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Downloads and caches NIFTY50 index and sector ETF daily OHLCV data
required for relative strength features in features.py.

Required for:
  - Feature 1: relative_strength_nifty (needs NIFTY50 daily close)
  - Feature 4: sector_rel_strength (needs sector ETF daily close)
"""

import os
import logging
import pandas as pd
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger("MARK5.MarketData")

# Sector ETF mapping — maps stock sectors to their benchmark ETFs
# These are real NSE-listed ETFs available on Kite Connect
SECTOR_ETF_MAP = {
    'Banking': 'BANKBEES',
    'IT': 'ITBEES',
    'Energy': 'ONGC',      # ONGC is more relevant as an energy stock proxy
    'FMCG': 'HINDUNILVR',      # No FMCG ETF; use HUL as proxy
    'Pharma': 'SUNPHARMA',     # PHARMABEES not liquid; use Sun Pharma
    'Financial Services': 'BANKBEES',  # Closest proxy
    'Automobile': 'TATAMOTORS',        # No auto ETF
    'Metals': 'TATASTEEL',             # No metals ETF
    'Consumer Durables': 'TITAN',
    'Telecom': 'BHARTIARTL',
    'Capital Goods': 'LT',
    'Power': 'NTPC',
    'Cement': 'ULTRACEMCO',
    'Infrastructure': 'LT',
    'Insurance': 'HDFCLIFE',
    'Healthcare': 'APOLLOHOSP',
    'Mining': 'COALINDIA',
    'Defence': 'BEL',
    'Diversified': None,  # No proxy — falls back to NIFTY50
}

NIFTY50_KITE   = 'NIFTY50'  # Correct internal key for KiteFeedAdapter.KITE_INDEX_TOKENS

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cache')


class MarketDataProvider:
    """
    Downloads and caches NIFTY50 index + sector ETF data.
    Used by AdvancedFeatureEngine for relative strength features.
    """
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def get_nifty50_data(
        self,
        start_date: str,
        end_date: str,
        use_kite: bool = False,
        kite_adapter=None
    ) -> Optional[pd.DataFrame]:
        """
        Get NIFTY50 index daily OHLCV data.
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
        DatetimeIndex aligned to IST trading days.
        """
        cache_path = os.path.join(self.cache_dir, 'NIFTY50_60m.parquet')
        
        # Try cache first
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            if not df.empty and 'close' in df.columns:
                logger.debug(f"📊 NIFTY50 60m loaded from cache: {len(df)} bars")
                return df
        
        if kite_adapter:
            logger.info(f"📡 Downloading NIFTY50 index (60m) via Kite ({start_date} to {end_date})...")
            # FIX: Use '60minute' interval to match stock clock
            nifty = kite_adapter.fetch_index_data(NIFTY50_KITE, days_back=2000, interval='60minute')
            if nifty is not None and not nifty.empty:
                if nifty.index.tz is not None:
                    nifty.index = nifty.index.tz_localize(None)
                nifty.to_parquet(cache_path)
                logger.info(f"✅ NIFTY50 60m: {len(nifty)} bars cached")
                return nifty

        logger.warning("⚠️ NIFTY50 data unavailable — relative strength will use stock-only returns")
        return None
    
    def get_sector_etf_data(
        self,
        sector: str,
        start_date: str,
        end_date: str,
        kite_adapter=None
    ) -> Optional[pd.Series]:
        """
        Get sector ETF daily close series for relative strength calculation.
        
        Args:
            sector: Sector name from nifty50_universe.py (e.g., 'Banking', 'IT')
            
        Returns:
            pd.Series of daily close prices, or None if unavailable.
        """
        etf_symbol = SECTOR_ETF_MAP.get(sector)
        if etf_symbol is None:
            return None
        
        cache_path = os.path.join(self.cache_dir, f'{etf_symbol.replace(".", "_")}_60m.parquet')
        
        # Try cache
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            if not df.empty and 'close' in df.columns:
                return df['close']
        
        # Kite Primary Download
        if kite_adapter:
            try:
                # FIX: Use '60minute' interval and correct date arguments
                from_dt = pd.to_datetime(start_date)
                to_dt = pd.to_datetime(end_date)
                data = kite_adapter.fetch_ohlcv(
                    etf_symbol, 
                    from_date=from_dt, 
                    to_date=to_dt, 
                    interval='60minute'
                )
                if data is not None and not data.empty:
                    data.columns = [str(c).lower() for c in data.columns]
                    data.to_parquet(cache_path)
                    logger.info(f"✅ Sector ETF {etf_symbol} (60m): {len(data)} bars cached")
                    return data['close']
            except Exception as e:
                logger.warning(f"Sector ETF {etf_symbol} download failed: {e}")
        
        return None
    
    def build_feature_context(
        self,
        stock_df: pd.DataFrame,
        sector: str,
        start_date: str,
        end_date: str,
        kite_adapter=None
    ) -> Dict:
        """
        Build the complete context dict needed by AdvancedFeatureEngine.
        
        Returns:
            dict with keys: nifty_close, fii_net, sector_etf_close
        """
        context = {}
        
        # NIFTY50 close series
        nifty_df = self.get_nifty50_data(start_date, end_date, kite_adapter=kite_adapter)
        if nifty_df is not None and 'close' in nifty_df.columns:
            context['nifty_close'] = nifty_df['close']
        
        # Sector ETF close series
        sector_close = self.get_sector_etf_data(sector, start_date, end_date, kite_adapter=kite_adapter)
        if sector_close is not None:
            context['sector_etf_close'] = sector_close
        
        # FII data (loaded separately by FIIDataProvider)
        # Will be injected by the calling code
        
        return context

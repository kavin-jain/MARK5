"""
MARK5 FII/DII Flow Data Module
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fetches and caches FII/DII activity data from NSE.
Per RULE 24: FII flow gate — check NSE FII/DII data every evening.
Per RULE 51: FII data must be available by 7 PM daily. If unavailable,
             use prior day's value with warning flag.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

# NSE publishes FII/DII activity at this endpoint
NSE_FII_API = "https://www.nseindia.com/api/fiidiiTradeReact"

logger = logging.getLogger("MARK5.FIIData")

NSE_HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.9,en-IN;q=0.8",
    "cache-control": "max-age=0",
    "sec-ch-ua": '"Chromium";v="129"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "referer": "https://www.nseindia.com/market-data/fii-dii",
    "x-requested-with": "XMLHttpRequest"
}

FII_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fii_dii')


class FIIDataProvider:
    """
    Provides FII/DII net flow data for feature engineering.
    
    Usage:
        provider = FIIDataProvider()
        fii_series = provider.get_fii_flow(start_date, end_date)
        # Returns pd.Series with index=date, values=FII net (₹cr)
    """
    
    def __init__(self, cache_dir: str = FII_CACHE_DIR):
        self.cache_dir = cache_dir
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, 'fii_dii_historical.csv')
    
    # In-memory session cache — prevents repeated NSE calls when fetch fails
    _session_cache: Optional[pd.Series] = None
    _session_fetched: bool = False

    def get_fii_flow(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Get FII net flow series for the given date range.
        
        Returns:
            pd.Series with DatetimeIndex and FII net values in crores.
            Positive = net buying, Negative = net selling.
            Gaps are filled with 0.0 (Neutral Fallback) per Phase 3 Plan.
        """
        # Try loading cached data
        df = self._load_cache()
        
        query_start = pd.Timestamp(start_date) if start_date else None
        query_end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        
        if df is not None and not df.empty:
            if query_start:
                df = df[df.index >= query_start]
            if query_end:
                df = df[df.index <= query_end]
            
            if not df.empty:
                logger.info(f"📊 Real FII data loaded: {len(df)} days")
                return df['fii_net']
        
        # Session-level cache: if we already tried and failed this session, don't retry
        if FIIDataProvider._session_fetched:
            return FIIDataProvider._session_cache if FIIDataProvider._session_cache is not None else pd.Series(dtype=float, name='fii_net')
        
        # No cached data — try fetching fresh from NSE for today
        logger.info("📡 FII cache empty/outdated. Attempting NSE fetch...")
        FIIDataProvider._session_fetched = True
        fetched = self._fetch_from_nse()
        
        if fetched is not None and not fetched.empty:
            self._save_cache(fetched)
            if query_start:
                fetched = fetched[fetched.index >= query_start]
            result = fetched['fii_net']
            FIIDataProvider._session_cache = result
            return result
        
        # Neutral Fallback (REVISED Phase 3): 
        # Return empty but typed series. The feature engine handles reindexing/filling.
        logger.warning("⚠️ FII real data unavailable. Returning empty series (Zero Fallback).")
        FIIDataProvider._session_cache = pd.Series(dtype=float, name='fii_net')
        return FIIDataProvider._session_cache

    
    def _load_cache(self) -> Optional[pd.DataFrame]:
        """Load cached FII/DII data from CSV."""
        if not os.path.exists(self.cache_file):
            return None
        
        try:
            df = pd.read_csv(self.cache_file, parse_dates=['date'], index_col='date')
            return df.sort_index()
        except Exception as e:
            logger.error(f"Cache load error: {e}")
            return None
    
    def _save_cache(self, df: pd.DataFrame):
        """Save FII/DII data to CSV cache."""
        try:
            # Merge with existing if present to avoid overwriting history
            existing = self._load_cache()
            if existing is not None:
                df = pd.concat([existing, df]).sort_index()
                df = df[~df.index.duplicated(keep='last')]
                
            df.to_csv(self.cache_file)
            logger.debug(f"💾 FII cache updated. Total records: {len(df)}")
        except Exception as e:
            logger.error(f"Cache save error: {e}")
    
    def _fetch_from_nse(self) -> Optional[pd.DataFrame]:
        """
        Fetch FII/DII data from NSE API.
        Hardened with session priming and retries.
        """
        import requests
        import time

        for attempt in range(2):
            try:
                session = requests.Session()
                session.headers.update(NSE_HEADERS)
                
                # Step 1: Prime the session
                session.get("https://www.nseindia.com", timeout=10)
                time.sleep(1)
                
                # Step 2: Fetch current snapshot
                response = session.get(NSE_FII_API, timeout=10)
                
                if response.status_code != 200:
                    continue

                data = response.json()
                records = []
                for entry in data:
                    date_str = entry.get('date', '')
                    if not date_str: continue
                    
                    date = pd.Timestamp(date_str)
                    category = entry.get('category', '')
                    # netValue can have commas if big
                    net_val = float(str(entry.get('netValue', '0')).replace(',', ''))
                    
                    if 'FII' in category.upper() or 'FPI' in category.upper():
                        records.append({'date': date, 'fii_net': net_val})
                
                if records:
                    return pd.DataFrame(records).set_index('date').sort_index()

            except Exception as e:
                logger.debug(f"NSE fetch fail: {e}")
                time.sleep(1)
        
        return None
    
    def generate_synthetic_fii_data(
        self,
        index: pd.DatetimeIndex,
        seed: int = 42
    ) -> pd.Series:
        """
        [DEPRECATED] Per Phase 3 Plan, synthetic data is disabled to prevent hallucinated alpha.
        Returns a zero-filled series to maintain architectural stability.
        """
        logger.warning("🚫 Synthetic FII data requested but DISABLED. Returning zeros.")
        return pd.Series(0.0, index=index, name='fii_net')
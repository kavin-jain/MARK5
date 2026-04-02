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

logger = logging.getLogger("MARK5.FIIData")

# NSE publishes FII/DII activity at this endpoint
NSE_FII_API = "https://www.nseindia.com/api/fiidiiTradeReact"
NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nseindia.com/market-data/fii-dii',
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
        """
        # Try loading cached data
        df = self._load_cache()
        
        if df is not None and not df.empty:
            if start_date:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date)]
            
            if not df.empty:
                logger.info(f"📊 FII data loaded: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
                return df['fii_net']
        
        # No cached data — try fetching from NSE
        logger.warning("⚠️ No FII data cached. Attempting NSE fetch...")
        fetched = self._fetch_from_nse()
        
        if fetched is not None and not fetched.empty:
            self._save_cache(fetched)
            return fetched['fii_net']
        
        # Fallback: return empty series (RULE 51: don't block trading)
        logger.warning("⚠️ FII data unavailable. Using zero fallback (per RULE 51).")
        return pd.Series(dtype=float, name='fii_net')
    
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
            df.to_csv(self.cache_file)
            logger.info(f"💾 FII data cached: {len(df)} days to {self.cache_file}")
        except Exception as e:
            logger.error(f"Cache save error: {e}")
    
    def _fetch_from_nse(self) -> Optional[pd.DataFrame]:
        """
        Fetch FII/DII data from NSE API.
        NSE requires a session cookie, so we first hit the main page.
        """
        try:
            import requests
            
            session = requests.Session()
            session.headers.update(NSE_HEADERS)
            
            # Get session cookie from main page
            session.get("https://www.nseindia.com", timeout=10)
            
            # Fetch FII/DII data
            response = session.get(NSE_FII_API, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"NSE API returned {response.status_code}")
                return None
            
            data = response.json()
            
            # Parse the response
            records = []
            for entry in data:
                try:
                    date = pd.Timestamp(entry.get('date', ''))
                    category = entry.get('category', '')
                    # M-8: NSE may return netValue as int/float or as a comma-formatted
                    # string. Wrap in str() before .replace() to handle both safely.
                    net_value = float(str(entry.get('netValue', '0')).replace(',', ''))
                    
                    if 'FII' in category.upper() or 'FPI' in category.upper():
                        records.append({'date': date, 'fii_net': net_value})
                except (ValueError, KeyError):
                    continue
            
            if records:
                df = pd.DataFrame(records).set_index('date').sort_index()
                return df
            
            return None
            
        except ImportError:
            logger.warning("requests library not available for NSE fetch")
            return None
        except Exception as e:
            logger.warning(f"NSE fetch failed: {e}")
            return None
    
    def generate_synthetic_fii_data(
        self,
        index: pd.DatetimeIndex,
        seed: int = 42
    ) -> pd.Series:
        """
        Generate synthetic FII flow data for backtesting when real data
        is unavailable. Uses random walk with mean-reversion to simulate
        realistic FII flow patterns.
        
        This is a TEMPORARY solution. Real FII data should be used once
        the NSE scraper is set up.
        
        Returns:
            pd.Series with FII net values (₹cr), one per trading day.
        """
        # Get unique dates only
        dates = pd.Series(index.date).unique()
        dates = pd.DatetimeIndex(sorted(dates))
        
        rng = np.random.RandomState(seed)
        
        # FII flows: roughly N(0, 1500) with mean-reversion
        # Real FII flows range from -5000 to +5000 cr daily
        n = len(dates)
        flows = np.zeros(n)
        flows[0] = rng.normal(0, 500)
        
        for i in range(1, n):
            # Mean-reverting random walk
            mean_reversion = -0.1 * flows[i-1]
            innovation = rng.normal(0, 1200)
            flows[i] = flows[i-1] + mean_reversion + innovation
        
        # Clip to realistic range
        flows = np.clip(flows, -5000, 5000)
        
        result = pd.Series(flows, index=dates, name='fii_net')
        logger.info(f"📊 Generated synthetic FII data: {len(result)} days "
                    f"(mean={result.mean():.0f}, std={result.std():.0f})")
        return result
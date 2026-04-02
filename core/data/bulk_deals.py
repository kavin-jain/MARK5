"""
MARK5 Bulk & Block Deals Data Provider
"""
import io
import logging
import os
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from core.data.market_data import MarketDataProvider

logger = logging.getLogger(__name__)

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

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
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}

class BulkDealsProvider:
    def __init__(self):
        self.session = self._init_session()
        
    def _init_session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update(NSE_HEADERS)
        try:
            # Prime the cookies
            s.get("https://www.nseindia.com", timeout=10)
            s.get("https://www.nseindia.com/market-data/bulk-deals", timeout=10)
        except Exception as e:
            logger.warning(f"Failed to prime NSE session: {e}")
        return s

    def _fetch_daily_snapshot(self) -> Optional[Dict]:
        """Fetch today's bulk/block deals summary."""
        url = "https://www.nseindia.com/api/snapshot-capital-market-largedeal"
        try:
            r = self.session.get(url, timeout=15)
            if r.status_code == 200:
                return r.json()
            else:
                logger.error(f"Failed to fetch bulk deals. HTTP {r.status_code}")
                # Re-prime on failure
                self.session = self._init_session()
        except Exception as e:
            logger.error(f"Error fetching bulk deals: {e}")
        return None

    def update_today(self, d: date = None):
        """Fetch daily live bulk/block deals and append to local cache."""
        if d is None:
            d = date.today()
            
        data = self._fetch_daily_snapshot()
        if not data:
            return

        api_date_str = data.get("as_on_date")
        if not api_date_str:
            return
            
        try:
            api_date = datetime.strptime(api_date_str, "%d-%b-%Y").date()
            if api_date != d:
                logger.warning(f"Bulk deals API date {api_date} doesn't match requested {d}")
                # Accept latest available if requested today
                d = api_date
        except:
            pass
            
        self._process_and_save(data, d)

    def _process_and_save(self, data: Dict, d: date):
        """Process JSON to parquet records append."""
        
        # 1. Bulk Deals
        bulk_raw = data.get("BULK_DEALS_DATA", [])
        if bulk_raw:
            df_bulk = pd.DataFrame(bulk_raw)
            # Schema: {'date': '20-Mar-2026', 'symbol': 'ACETEC', 'name': 'Acetech E-Commerce Ltd', 'clientName': '...', 'buySell': 'BUY', 'qty': '1599600', 'watp': '139', 'remarks': '-'}
            df_bulk['date'] = pd.to_datetime(df_bulk['date']).dt.date
            # Clean numeric columns
            df_bulk['qty'] = pd.to_numeric(df_bulk['qty'], errors='coerce').fillna(0)
            df_bulk['watp'] = pd.to_numeric(df_bulk['watp'], errors='coerce').fillna(0)
            df_bulk['value'] = df_bulk['qty'] * df_bulk['watp']
            
            self._append_cache("bulk_deals.parquet", df_bulk)
            
        # 2. Block Deals
        block_raw = data.get("BLOCK_DEALS_DATA", [])
        if block_raw:
            df_block = pd.DataFrame(block_raw)
            df_block['date'] = pd.to_datetime(df_block['date']).dt.date
            df_block['qty'] = pd.to_numeric(df_block['qty'], errors='coerce').fillna(0)
            self._append_cache("block_deals.parquet", df_block)
            
        logger.info(f"Saved {len(bulk_raw)} bulk and {len(block_raw)} block deals for {d}")

    def _append_cache(self, filename: str, new_df: pd.DataFrame):
        path = os.path.join(CACHE_DIR, filename)
        if os.path.exists(path):
            existing = pd.read_parquet(path)
            # Remove duplicated same-day entries
            combined = pd.concat([existing, new_df]).drop_duplicates(
                subset=['date', 'symbol', 'clientName', 'buySell', 'qty'], 
                keep='last'
            )
            combined.to_parquet(path, index=False)
        else:
            new_df.to_parquet(path, index=False)

    def backfill_historical(self, symbols: List[str], start: str, end: str):
        """
        Historical large deals endpoint on NSE is disabled/404.
        For Phase 1 validation, gracefully warn and initialize empty DataFrames.
        In production, this relies on daily accumulation.
        """
        logger.warning(
            "Historical bulk/block deals API (api/historical/bulk-deals) is disabled by NSE. "
            "System will rely on daily update accumulation going forward. "
            "Initializing empty cache to satisfy pipeline dependencies."
        )
        # Create empty cache if needed
        for name in ["bulk_deals.parquet", "block_deals.parquet"]:
            path = os.path.join(CACHE_DIR, name)
            if not os.path.exists(path):
                df = pd.DataFrame(columns=['date', 'symbol', 'buySell', 'qty', 'watp', 'value'])
                df.to_parquet(path, index=False)

    def get_features(self, symbol: str, start: str, end: str, spot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute:
        1. institutional_buy_5d: 5d rolling sum of buy-side value / 20d ADV
        2. block_deal_flag: 1 if block deal occurred, else 0
        """
        feats = pd.DataFrame(index=spot_df.index)
        feats['institutional_buy_5d'] = np.nan
        feats['block_deal_flag'] = 0.0
        
        # Load daily volume for ADV calculation
        if 'volume' in spot_df.columns:
            adv_20 = spot_df['volume'].rolling(20, min_periods=5).mean()
            # If price is available, convert volume ADV to value ADV
            if 'close' in spot_df.columns:
                adv_20_value = adv_20 * spot_df['close']
        else:
            return feats
            
        # 1. Bulk Deals (institutional_buy_5d)
        bulk_path = os.path.join(CACHE_DIR, "bulk_deals.parquet")
        if os.path.exists(bulk_path):
            try:
                bulk_df = pd.read_parquet(bulk_path)
                sym_bulk = bulk_df[bulk_df['symbol'] == symbol]
                if not sym_bulk.empty:
                    # Aggregate buy value by date
                    buys = sym_bulk[sym_bulk['buySell'] == 'BUY']
                    daily_buys = buys.groupby('date')['value'].sum()
                    daily_buys.index = pd.to_datetime(daily_buys.index)
                    
                    # Align to spot dates and fill missing with 0
                    aligned_buys = pd.Series(0.0, index=spot_df.index)
                    common_dates = aligned_buys.index.intersection(daily_buys.index)
                    aligned_buys[common_dates] = daily_buys[common_dates]
                    
                    # 5d rolling sum
                    sum_5d = aligned_buys.rolling(5, min_periods=1).sum()
                    
                    # Compute feature: buy_value_5d / ADV_20d_value
                    # Shift 1 to prevent lookahead! Data is published end-of-day.
                    raw_feat = sum_5d / adv_20_value
                    feats['institutional_buy_5d'] = raw_feat.shift(1).fillna(np.nan)
            except Exception as e:
                logger.error(f"Error computing bulk deal features for {symbol}: {e}")

        # 2. Block Deals (block_deal_flag)
        block_path = os.path.join(CACHE_DIR, "block_deals.parquet")
        if os.path.exists(block_path):
            try:
                block_df = pd.read_parquet(block_path)
                sym_block = block_df[block_df['symbol'] == symbol]
                if not sym_block.empty:
                    block_dates = pd.to_datetime(sym_block['date'].unique())
                    # Align to spot
                    raw_flag = pd.Series(0.0, index=spot_df.index)
                    common_dates = raw_flag.index.intersection(block_dates)
                    raw_flag[common_dates] = 1.0
                    
                    # Shift 1 to prevent lookahead!
                    feats['block_deal_flag'] = raw_flag.shift(1).fillna(0.0)
            except Exception as e:
                logger.error(f"Error computing block deal features for {symbol}: {e}")

        # Trim to requested dates
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        feats = feats[(feats.index >= start_ts) & (feats.index <= end_ts)]
        return feats

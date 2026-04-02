"""
MARK5 Master Data Pipeline Orchestrator.
Coordinates Daily Data Fetching and Feature Generation.
"""
import logging
import os
from datetime import date, datetime
import pandas as pd
from typing import List, Dict

from core.data.market_data import MarketDataProvider
from core.data.fii_data import FIIDataProvider
from core.data.fno_data import FNODataProvider
from core.data.bulk_deals import BulkDealsProvider
from core.models.features import AdvancedFeatureEngine

try:
    from core.data.adapters.ise_adapter import ISEAdapter
    _ISE_AVAILABLE = True
except ImportError:
    _ISE_AVAILABLE = False

logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self):
        self.market_provider = MarketDataProvider()
        self.fii_provider = FIIDataProvider()
        self.fno_provider = FNODataProvider()
        self.bulk_provider = BulkDealsProvider()
        self.feature_engine = AdvancedFeatureEngine()
        
    def fetch_daily_data(self, d: date = None, symbols: List[str] = None, spot_data: Dict = None):
        """Fetch all necessary daily data for 'd'."""
        if d is None:
            d = date.today()
        if symbols is None:
            symbols = []
        if spot_data is None:
            spot_data = {}

        logger.info(f"=== Starting Daily Data Fetch for {d} ===")

        # 1. FII Data
        try:
            logger.info("Fetching FII Data...")
            self.fii_provider.get_fii_flow()
        except Exception as exc:
            logger.warning(f"FII fetch failed: {exc}")

        # 2. Bulk & Block Deals
        try:
            logger.info("Fetching Bulk Deals...")
            self.bulk_provider.update_today(d)
        except Exception as exc:
            logger.warning(f"Bulk deals fetch failed: {exc}")

        # 3. F&O Bhav Copy (weekdays only)
        # BUGFIX: fno_provider.update_today requires symbols and spot_data, not just date
        if d.weekday() < 5:
            try:
                logger.info("Fetching F&O Bhav Copy...")
                self.fno_provider.update_today(symbols=symbols, spot_data=spot_data)
            except Exception as exc:
                logger.warning(f"F&O fetch failed: {exc}")

        # 4. ISE Premium Data refresh (if API key configured)
        if _ISE_AVAILABLE and symbols:
            try:
                from core.data.adapters.ise_adapter import ISEAdapter
                logger.info(f"Refreshing ISE fundamental data for {len(symbols)} symbols...")
                ise = ISEAdapter()
                status = ise.get_budget_status()
                if status.get("remaining", 0) >= len(symbols):
                    for sym in symbols:
                        bare = sym.replace(".NS", "").replace(".BO", "")
                        ise.fetch_stock(bare, force_refresh=False)           # daily cache
                        ise.fetch_stock_target_price(bare)                   # daily cache
                    logger.info(f"ISE refresh complete. Budget remaining: {ise.get_budget_status()['remaining']}")
                else:
                    logger.warning(f"ISE budget low ({status.get('remaining')} tokens) — ISE refresh skipped")
            except Exception as exc:
                logger.warning(f"ISE refresh failed (non-critical): {exc}")

        logger.info(f"=== Daily Data Fetch Complete for {d} ===")

    def build_feature_matrix(self, symbols: List[str], start_d: str, end_d: str) -> Dict[str, pd.DataFrame]:
        """
        Builds the unified feature dataframe for each symbol over the requested date range.
        Returns a dictionary of symbol -> features DataFrame.
        """
        logger.info(f"Building feature matrix for {len(symbols)} symbols ({start_d} → {end_d})...")
        
        # Pre-load context
        nifty_cache_path = "data/cache/NIFTY50_1d.parquet"
        if os.path.exists(nifty_cache_path):
            nifty_cache = pd.read_parquet(nifty_cache_path)
            nifty_close = nifty_cache['close']
        else:
            nifty_close = None
            
        fii_net = self.fii_provider.get_fii_flow(start_d, end_d)
            
        results = {}
        for sym in symbols:
            base_sym = sym.replace('.NS', '')
            spot_path = f"data/cache/{base_sym}_NS_1d.parquet"
            if not os.path.exists(spot_path):
                spot_path = f"data/cache/{base_sym.replace('.', '_')}_NS_1d.parquet"
                
            if os.path.exists(spot_path):
                spot_df = pd.read_parquet(spot_path)
            else:
                spot_df = None
                
            if spot_df is None or spot_df.empty:
                logger.warning(f"No spot data for {sym}. Skipping.")
                continue
                
            # Restrict spot data to the requested end date for valid processing
            end_ts = pd.Timestamp(end_d)
            spot_df_trimmed = spot_df[spot_df.index <= end_ts]
            
            # Fetch F&O features
            fno_feats = self.fno_provider.get_fno_features(sym.replace('.NS', ''), start_d, end_d, spot_df_trimmed)
            
            # Fetch Bulk features
            bulk_feats = self.bulk_provider.get_features(sym.replace('.NS', ''), start_d, end_d, spot_df_trimmed)
            
            context = {
                'nifty_close': nifty_close,
                'fii_net': fii_net,
                'fno_features': fno_feats,
                'bulk_features': bulk_feats
            }
            
            # Generate all features
            final_df = self.feature_engine.engineer_all_features(spot_df_trimmed, context=context)
            
            # Filter exactly to date range
            final_df = final_df[(final_df.index >= pd.Timestamp(start_d)) & (final_df.index <= end_ts)]
            results[sym] = final_df
            
        return results

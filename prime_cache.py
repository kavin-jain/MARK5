"""
MARK5 Pre-flight Data Extractor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Forces the download and caching of all macro context (NIFTY50,
Sector ETFs, FII/DII) using the current Kite Connect token.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("MARK5.PrimeCache")

from core.data.adapters.kite_adapter import KiteFeedAdapter
from core.data.market_data import MarketDataProvider, SECTOR_ETF_MAP
from core.data.fii_data import FIIDataProvider
from core.utils.config_manager import ConfigManager

def prime_all_data():
    logger.info("🚀 Initiating MARK5 Pre-flight Data Extraction...")
    
    # 1. Initialize Config and Kite Adapter
    config_mgr = ConfigManager()
    cfg = config_mgr.get_config()
    
    # We expect credentials in .env or config.yaml
    kite_adapter = KiteFeedAdapter({})
    success = kite_adapter.connect()
    
    if not success:
        logger.critical("❌ Kite Connection FAILED. Ensure API Key and Access Token are updated in .env.")
        sys.exit(1)
        
    logger.info("✅ Kite Connected successfully.")
    
    # 2. Fetch Macro Data (NIFTY & Sectors)
    mdp = MarketDataProvider()
    start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d') # 3 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Downloading NIFTY50 from {start_date} to {end_date}...")
    nifty_df = mdp.get_nifty50_data(start_date, end_date, use_kite=True, kite_adapter=kite_adapter)
    if nifty_df is not None and not nifty_df.empty:
        logger.info(f"✅ NIFTY50 data cached: {len(nifty_df)} bars.")
    else:
        logger.error("❌ Failed to download NIFTY50 data.")
    
    logger.info("Downloading Sector ETFs...")
    sectors = set(SECTOR_ETF_MAP.keys()) - {None}
    for sector in sectors:
        logger.info(f"Fetching ETF for sector: {sector}...")
        etf_series = mdp.get_sector_etf_data(sector, start_date, end_date, kite_adapter=kite_adapter)
        if etf_series is not None and not etf_series.empty:
            logger.info(f"✅ {sector} ETF data cached: {len(etf_series)} bars.")
        else:
            logger.warning(f"⚠️ Failed to download ETF for sector: {sector}.")
            
    # 3. Fetch FII/DII Data
    logger.info("Downloading FII/DII Flow Data from NSE...")
    fii_provider = FIIDataProvider()
    fii_series = fii_provider.get_fii_flow(start_date, end_date)
    if fii_series is not None and not fii_series.empty:
        logger.info(f"✅ FII/DII data cached: {len(fii_series)} days.")
    else:
        logger.warning("⚠️ Failed to fetch FII/DII data from NSE.")
        
    logger.info("🎉 Pre-flight Data Extraction Complete. The system is ready.")
    kite_adapter.disconnect()

if __name__ == "__main__":
    prime_all_data()

"""
MARK5 Master Data Pipeline Orchestrator.
Coordinates Daily Data Fetching, Feature Generation, and acts as the Master Data Router.
"""
import logging
import os
from datetime import date, datetime
import pandas as pd
from typing import List, Dict, Any, Optional, Union

from core.data.market_data import MarketDataProvider
from core.data.fii_data import FIIDataProvider
from core.data.fno_data import FNODataProvider
from core.data.bulk_deals import BulkDealsProvider
from core.models.features import AdvancedFeatureEngine
from core.data.adapters.ise_adapter import ISEAdapter
from core.data.provider import DataProvider

logger = logging.getLogger("MARK5.DataPipeline")

class DataPipeline:
    """
    MASTER DATA ROUTER for MARK5.
    No other module should fetch its own data. All requests flow through here.
    Enforces REQ-6: Zero Synthetic Data Policy.
    """
    
    def __init__(self):
        self.market_provider = MarketDataProvider()
        self.fii_provider = FIIDataProvider()
        self.fno_provider = FNODataProvider()
        self.bulk_provider = BulkDealsProvider()
        self.feature_engine = AdvancedFeatureEngine()
        self.ise = ISEAdapter()
        
        # Initialize DataProvider for Kite access (Single Source of Truth for OHLCV)
        try:
            self.data_provider = DataProvider(config={'kite': {}})
            # We don't call connect() here to avoid blocking; 
            # individual methods will ensure connection if needed.
        except Exception as e:
            logger.warning(f"Could not initialize DataProvider: {e}")
            self.data_provider = None

    # =========================================================================
    # MASTER ROUTING METHODS
    # =========================================================================

    def get_market_data(
        self, 
        ticker: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None, 
        interval: str = 'day', 
        source: str = 'kite',
        period: str = '1y'
    ) -> pd.DataFrame:
        """
        Master router for OHLCV market data.
        Sources: 'kite' (Primary), 'ise' (Supplementary/Historical).
        """
        df = pd.DataFrame()
        
        if source == 'kite' and self.data_provider:
            if not self.data_provider.feed or not self.data_provider.feed.is_connected:
                self.data_provider.connect()
            
            # Convert string dates to datetime if provided
            from_dt = pd.to_datetime(start_date) if start_date else None
            to_dt = pd.to_datetime(end_date) if end_date else None
            
            df = self.data_provider.feed.fetch_ohlcv(
                ticker, 
                period=period, 
                interval=interval,
                from_date=from_dt,
                to_date=to_dt
            )
        elif source == 'ise':
            # ISE historical data (Endpoint 13)
            # Map interval to ISE period if needed, or just use period
            raw = self.ise.fetch_historical_data(ticker, period=period)
            df = pd.DataFrame(raw)
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
        
        self._assert_not_synthetic(df, f"Market Data ({ticker})")
        return df

    def get_fundamental_data(self, ticker: str) -> pd.DataFrame:
        """Get comprehensive company data (Endpoint 1)."""
        raw = self.ise.fetch_stock(ticker)
        # Flatten the nested structure into a single-row DataFrame for consistency
        df = pd.json_normalize(raw)
        self._assert_not_synthetic(df, f"Fundamental Data ({ticker})")
        return df

    def get_trending_stocks(self) -> pd.DataFrame:
        """Get top gainers and losers (Endpoint 4)."""
        raw = self.ise.fetch_trending()
        gainers = pd.DataFrame(raw.get('gainers', []))
        gainers['trend_type'] = 'gainer'
        losers = pd.DataFrame(raw.get('losers', []))
        losers['trend_type'] = 'loser'
        df = pd.concat([gainers, losers], ignore_index=True)
        self._assert_not_synthetic(df, "Trending Stocks")
        return df

    def get_most_active(self, exchange: str = 'NSE') -> pd.DataFrame:
        """Get most active stocks (Endpoints 6 & 7)."""
        if exchange.upper() == 'NSE':
            raw = self.ise.fetch_NSE_most_active()
        else:
            raw = self.ise.fetch_BSE_most_active()
        df = pd.DataFrame(raw)
        self._assert_not_synthetic(df, f"Most Active ({exchange})")
        return df

    def get_analyst_recommendations(self, ticker: str) -> pd.DataFrame:
        """Get analyst price targets and recs (Endpoint 11)."""
        raw = self.ise.fetch_stock_target_price(ticker)
        df = pd.json_normalize(raw)
        self._assert_not_synthetic(df, f"Analyst Recs ({ticker})")
        return df

    def get_historical_stats(self, ticker: str, stats: str = "quarter_results") -> pd.DataFrame:
        """Get historical financial stats (Endpoint 14)."""
        raw = self.ise.fetch_historical_stats(ticker, stats=stats)
        # This usually returns a list of results in a key
        if isinstance(raw, dict) and stats in raw:
            df = pd.DataFrame(raw[stats])
        else:
            df = pd.DataFrame(raw) if isinstance(raw, list) else pd.json_normalize(raw)
        self._assert_not_synthetic(df, f"Historical Stats ({ticker}/{stats})")
        return df

    def get_price_shockers(self) -> pd.DataFrame:
        """Get price shockers (Endpoint 9)."""
        raw = self.ise.fetch_price_shockers()
        # raw is usually { "price_shockers": [...] }
        data = raw.get('price_shockers', []) if isinstance(raw, dict) else raw
        df = pd.DataFrame(data)
        self._assert_not_synthetic(df, "Price Shockers")
        return df

    def get_commodities(self) -> pd.DataFrame:
        """Get commodity futures (Endpoint 10)."""
        raw = self.ise.fetch_commodities()
        df = pd.DataFrame(raw)
        self._assert_not_synthetic(df, "Commodities")
        return df

    def search_industry(self, query: str) -> pd.DataFrame:
        """Search for stocks by industry (Endpoint 2)."""
        raw = self.ise.fetch_industry_search(query)
        df = pd.DataFrame(raw)
        self._assert_not_synthetic(df, f"Industry Search ({query})")
        return df

    def search_mutual_fund(self, query: str) -> pd.DataFrame:
        """Search for mutual funds (Endpoint 3)."""
        raw = self.ise.fetch_mutual_fund_search(query)
        df = pd.DataFrame(raw)
        self._assert_not_synthetic(df, f"MF Search ({query})")
        return df

    def get_52_week_high_low(self) -> pd.DataFrame:
        """Get 52-week high/low data (Endpoint 5)."""
        raw = self.ise.fetch_52_week_high_low()
        df = pd.json_normalize(raw)
        self._assert_not_synthetic(df, "52 Week High/Low")
        return df

    def get_mutual_funds(self) -> pd.DataFrame:
        """Get mutual funds data (Endpoint 8)."""
        raw = self.ise.fetch_mutual_funds()
        df = pd.DataFrame(raw)
        self._assert_not_synthetic(df, "Mutual Funds")
        return df

    def get_stock_forecasts(
        self, 
        ticker: str, 
        measure_code: str = "EPS", 
        period_type: str = "Annual", 
        data_type: str = "Actuals", 
        age: str = "Current"
    ) -> pd.DataFrame:
        """Get stock forecasts (Endpoint 12)."""
        raw = self.ise.fetch_stock_forecasts(
            ticker, 
            measure_code=measure_code, 
            period_type=period_type, 
            data_type=data_type, 
            age=age
        )
        df = pd.json_normalize(raw)
        self._assert_not_synthetic(df, f"Stock Forecasts ({ticker})")
        return df

    # =========================================================================
    # PIPELINE ORCHESTRATION
    # =========================================================================
        
    def fetch_daily_data(self, d: date = None, symbols: List[str] = None, spot_data: Dict = None):
        """Fetch all necessary daily data for 'd'."""
        if d is None:
            d = date.today()
        if symbols is None:
            symbols = []
        if spot_data is None:
            spot_data = {}

        # 0. Zero Synthetic Data Assertion (REQ-6)
        self._assert_not_synthetic(spot_data, "spot_data")

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
        if d.weekday() < 5:
            try:
                logger.info("Fetching F&O Bhav Copy...")
                self.fno_provider.update_today(symbols=symbols, spot_data=spot_data)
            except Exception as exc:
                logger.warning(f"F&O fetch failed: {exc}")

        # 4. ISE Premium Data refresh
        if symbols:
            try:
                status = self.ise.get_budget_status()
                if status.get("remaining", 0) >= len(symbols):
                    for sym in symbols:
                        bare = sym.replace(".NS", "").replace(".BO", "")
                        # Refresh fundamental and target data
                        self.get_fundamental_data(bare)
                        self.get_analyst_recommendations(bare)
                    logger.info(f"ISE refresh complete. Budget remaining: {self.ise.get_budget_status()['remaining']}")
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
        
        # 0. Zero Synthetic Data Assertion (REQ-6)
        fii_net = self.fii_provider.get_fii_flow(start_d, end_d)
        self._assert_not_synthetic(fii_net, "FII Data")

        # Pre-load context
        nifty_cache_path = "data/cache/NIFTY50_1d.parquet"
        if os.path.exists(nifty_cache_path):
            nifty_cache = pd.read_parquet(nifty_cache_path)
            nifty_close = nifty_cache['close']
        else:
            nifty_close = None
            
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
            
            # 0. Zero Synthetic Data Assertion (REQ-6)
            self._assert_not_synthetic(spot_df, f"Spot Data ({sym})")
                
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

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _assert_not_synthetic(self, data: Any, label: str):
        """Hard assertion against synthetic data (REQ-6)."""
        if data is None:
            return
            
        is_synthetic = False
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            is_synthetic = getattr(data, 'is_synthetic', False)
        elif isinstance(data, dict):
            is_synthetic = data.get('is_synthetic', False)
        
        if is_synthetic:
            logger.critical(f"🚫 SYNTHETIC DATA DETECTED in {label}. Blocking pipeline.")
            raise ValueError(f"MARK5 requires real historical data. Synthetic data in {label} is prohibited.")

if __name__ == "__main__":
    # Basic validation
    logging.basicConfig(level=logging.INFO)
    pipeline = DataPipeline()
    test_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    
    print("\n--- Validating Master Router ---")
    for sym in test_symbols:
        try:
            print(f"\nFetching data for {sym}...")
            # Test fundamental data
            fund = pipeline.get_fundamental_data(sym.replace(".NS", ""))
            print(f"Fundamental data columns: {fund.columns.tolist()[:5]}...")
            
            # Test trending
            trending = pipeline.get_trending_stocks()
            print(f"Trending stocks count: {len(trending)}")
            
            # Test synthetic guard
            synthetic_df = pd.DataFrame({'price': [100, 101]})
            synthetic_df.is_synthetic = True
            try:
                pipeline._assert_not_synthetic(synthetic_df, "Test Synthetic")
            except ValueError:
                print("✅ Synthetic data guard caught test synthetic data.")
                
        except Exception as e:
            print(f"❌ Error for {sym}: {e}")

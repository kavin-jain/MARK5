import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pytz
import pandas as pd
import ta
import numpy as np

from core.database_manager import MARK5DatabaseManager
from core.learning_engine import LearningEngine
from core.data.collector import MARK5DataCollector
from core.utils.config_manager import get_config

class NextDayPlanner:
    """
    Orchestrates end-of-day activities:
    1. Reviews today's performance
    2. Selects high-potential tickers for tomorrow
    3. Adjusts risk parameters (e.g., if volatility is high, reduce size)
    4. Generates a "Morning Brief"
    """
    
    def __init__(self, 
                 db_manager: MARK5DatabaseManager, 
                 learning_engine: LearningEngine,
                 collector: MARK5DataCollector):
        self.db = db_manager
        self.learner = learning_engine
        self.collector = collector
        self.logger = logging.getLogger("MARK5.NextDayPlanner")
        self.config = get_config()
        self.ist = pytz.timezone('Asia/Kolkata')
        
    def get_next_trading_day(self, current_date: datetime) -> datetime:
        """Calculate next trading day (skip weekends)."""
        next_day = current_date + timedelta(days=1)
        while next_day.weekday() >= 5: # Saturday=5, Sunday=6
            next_day += timedelta(days=1)
        return next_day

    def _get_market_regime(self) -> Dict:
        """Determine market regime using Nifty 50 data."""
        try:
            # Fetch Nifty 50 recent data (last 20 days)
            nifty_data = self.collector.get_stock_data(index_symbol='NIFTY_50', days=20)
            
            if nifty_data.empty or len(nifty_data) < 14:
                return {"regime": "Neutral", "volatility": "Medium", "trend": "Sideways"}
            
            # Calculate volatility (ATR)
            nifty_data['ATR'] = ta.volatility.AverageTrueRange(
                nifty_data['high'], nifty_data['low'], nifty_data['close'], window=14
            ).average_true_range()
            avg_nifty_atr = nifty_data['ATR'].mean()
            
            # Classify volatility
            high_vol_threshold = self.config.get('market', {}).get('high_vol_threshold', 1.5)
            if avg_nifty_atr > high_vol_threshold:
                volatility = "High"
            elif avg_nifty_atr < high_vol_threshold * 0.5:
                volatility = "Low"
            else:
                volatility = "Medium"
                
            # Determine Trend (SMA 200)
            # Fetch longer history for SMA 200 if needed, or use shorter MA for trend
            # Using EMA 20 for short-term trend proxy if data is short
            ema_20 = ta.trend.EMAIndicator(nifty_data['close'], window=20).ema_indicator().iloc[-1]
            current_close = nifty_data['close'].iloc[-1]
            
            trend = "Uptrend" if current_close > ema_20 else "Downtrend"
            
            return {
                "regime": "HighRisk" if volatility == "High" else "LowRisk",
                "volatility": volatility,
                "trend": trend
            }
        except Exception as e:
            self.logger.error(f"Market regime analysis failed: {e}")
            return {"regime": "Neutral", "volatility": "Medium", "trend": "Sideways"}

    def generate_plan(self) -> tuple[Dict, Dict]:
        """
        Generate the trading plan for the next session.
        Returns: (plan_dict, optimized_parameters_dict)
        """
        self.logger.info("Generating Next Day Plan...")
        
        # 1. Analyze Recent Performance
        performance = self.learner.analyze_recent_performance(lookback_days=5)
        self.logger.debug(f"Recent performance metrics: {performance}")
        
        # 2. Market Regime Analysis
        market_context = self._get_market_regime()
        self.logger.info(f"Market context determined: {market_context}")
        
        # 3. Watchlist Selection
        watchlist = self.select_watchlist()
        self.logger.info(f"Selected watchlist: {watchlist}")
        
        # 4. Parameter Optimization
        params = self.optimize_parameters(performance, market_context)
        self.logger.info(f"Optimized system parameters: {params}")
        
        # Calculate Plan Date
        current_time_ist = datetime.now(self.ist)
        plan_date = self.get_next_trading_day(current_time_ist)
        
        # Generate Action Items
        action_items = self._generate_action_items(market_context, performance)
            
        plan = {
            "date": plan_date.strftime('%Y-%m-%d'),
            "performance_review": performance,
            "market_context": market_context,
            "watchlist": watchlist,
            "system_parameters": params,
            "action_items": action_items
        }
        
        self._save_plan(plan)
        return plan, params
        
    def select_watchlist(self) -> List[str]:
        """
        Select top candidates for the next day based on momentum and volatility.
        """
        try:
            # Fetch universe dynamically (e.g., Nifty 50)
            # In a real scenario, collector would have a method to get constituents
            # For now, we simulate fetching or use a config list
            universe = self.config.get('universe', [
                "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", 
                "ITC", "KOTAKBANK", "LT", "AXISBANK", "HINDUNILVR", "TATAMOTORS", "MARUTI"
            ])
            
            # Fetch recent data for all universe tickers (batch)
            all_data = {}
            for ticker in universe:
                data = self.collector.get_stock_data(ticker, days=10)
                if not data.empty and len(data) > 14:
                    all_data[ticker] = data
            
            # Score tickers based on criteria (e.g., volume, RSI, momentum)
            scores = {}
            for ticker, data in all_data.items():
                try:
                    latest_close = data['close'].iloc[-1]
                    latest_vol = data['volume'].iloc[-1]
                    
                    # Liquidity Filter
                    if latest_vol * latest_close < 100_000_000: # 10 Cr turnover
                        continue
                        
                    # Calculate RSI
                    rsi = ta.momentum.RSIIndicator(data['close'], window=14).rsi().iloc[-1]
                    
                    # Score: Higher Volume + RSI in momentum zone (40-60 is chop, >60 or <40 is trend/reversal)
                    # Simple score: Volume * (1 if RSI > 60 else 0.5)
                    score = latest_vol * (1.5 if rsi > 60 else (1.2 if rsi < 40 else 1.0))
                    scores[ticker] = score
                except Exception:
                    continue
            
            # Sort by score
            sorted_tickers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [t[0] for t in sorted_tickers[:5]]
            
        except Exception as e:
            self.logger.error(f"Watchlist selection failed: {e}")
            return []

    def optimize_parameters(self, performance: Dict, market_context: Dict) -> Dict:
        """
        Dynamic parameter tuning based on recent performance and market context.
        """
        # Extract Metrics (defaulting to safe values)
        win_rate = performance.get('win_rate', 0.5)
        avg_win_pct = performance.get('avg_win_pct', 0.0)
        avg_loss_pct = performance.get('avg_loss_pct', 0.0)
        
        market_volatility = market_context.get('volatility', 'Medium')
        market_trend = market_context.get('trend', 'Sideways')
        
        # Base parameters (loaded from config)
        params = {
            "risk_strategy": "Neutral",
            "max_position_size_pct": self.config.get('risk', {}).get('max_position_size_pct', 25.0),
            "stop_loss_multiplier": 1.5,
            "confidence_threshold": 0.60
        }
        
        # Defensive Mode: Low Win Rate OR High Volatility
        if win_rate < 0.40 or market_volatility == "High":
            params["risk_strategy"] = "Defensive"
            params["max_position_size_pct"] = 15.0 # Reduce size
            params["confidence_threshold"] = 0.75 # Require higher confidence
            
        # Aggressive Mode: High Win Rate AND Uptrend AND Good R:R
        elif win_rate > 0.60 and avg_win_pct > abs(avg_loss_pct) and market_trend == "Uptrend":
            params["risk_strategy"] = "Aggressive"
            params["max_position_size_pct"] = 30.0 # Increase size
            params["confidence_threshold"] = 0.55 # Lower threshold slightly
            
        return params

    def _generate_action_items(self, market_context: Dict, performance: Dict) -> List[str]:
        action_items = []
        
        if market_context['volatility'] == 'High':
            action_items.append("Monitor stop losses closely; reduce exposure to volatile stocks")
        
        if performance.get('win_rate', 0.5) < 0.4:
            action_items.append("Review recent losing trades for pattern analysis")
            
        action_items.append("Check pre-market volume and news")
        action_items.append("Sync data at 09:00 AM")
        
        return action_items

    def _save_plan(self, plan: Dict):
        """Save the plan to a file"""
        try:
            log_dir = self.config.get('paths', {}).get('logs', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            # Verify write permission
            if not os.access(log_dir, os.W_OK):
                self.logger.error(f"Log directory {log_dir} is not writable.")
                return

            filename = os.path.join(log_dir, f"trading_plan_{plan['date']}.json")
            with open(filename, 'w') as f:
                json.dump(plan, f, indent=2, default=str)
            self.logger.info(f"✅ Trading plan saved: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save trading plan: {e}")

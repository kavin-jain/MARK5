"""
MARK5 DECISION ENGINE v8.0 - PRODUCTION GRADE (REACTIVE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Production hardening & standardized header
  • Added institutional position sizing via VolatilityAwarePositionSizer
  • Added container-based dependency injection
  • Improved ATR fallback handling

TRADING ROLE: Stateless tick processor - Market State + AI → Order
SAFETY LEVEL: CRITICAL - Generates all order execution requests

FLOW:
1. Get prediction from predictor
2. Apply signal thresholds
3. Calculate position size via institutional sizer
4. Execute via OMS if qty > 0
"""

import logging
import pandas as pd
from typing import Dict, Optional

from core.system.container import container
from core.trading.position_sizer import VolatilityAwarePositionSizer

class DecisionEngine:
    def __init__(self, collector=None, predictor=None, executor=None, journal=None, risk_analyzer=None):
        self.logger = logging.getLogger("MARK5.Decision")
        
        self.collector = collector
        # Access container slots directly if not provided (Architect's Edition compatible)
        self.predictor = predictor or getattr(container, "predictor", None)
        self.risk = risk_analyzer or getattr(container, "risk_manager", None)
        self.oms = executor or getattr(container, "oms", None)
        self.journal = journal
        
        # New Institutional Sizer
        self.sizer = VolatilityAwarePositionSizer(
            initial_capital=100000.0, # Should come from config
            default_risk_per_trade=0.01,
            max_position_size_pct=0.05  # RULE 11: 5% max
        )

    def analyze_and_act(self, ticker: str):
        """Legacy helper for autonomous.py"""
        if not self.collector:
            self.logger.error(f"No collector attached to DecisionEngine for {ticker}")
            return {'action': 'HOLD', 'reason': 'No collector'}
        
        # We need data snapshot to process tick
        try:
            # Assumes fetch_stock_data exists on collector
            df = self.collector.fetch_stock_data(ticker, period='1d', interval='15m')
            if df is None or df.empty:
                return {'action': 'HOLD', 'reason': 'No data'}
            
            result = self.process_tick(ticker, df)
            
            # Legacy fallback: autonomous.py expects a dictionary, not None
            if result is None:
                return {'action': 'HOLD', 'reason': 'Signal Rejected / Filtered'}
                
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {ticker} in analyze_and_act: {e}")
            return {'action': 'HOLD', 'reason': 'Fetch error'}

    def process_tick(self, ticker: str, data_snapshot: object):
        """
        Main Reactor Loop. Called whenever new data arrives.
        data_snapshot: DataFrame with latest OHLCV
        """
        # 1. Get Prediction
        # The predictor handles feature engineering internally
        try:
            # Check if this is a PredictionEngine or the MARK5Predictor directly
            if hasattr(self.predictor, 'predict_single'):
                ai_result = self.predictor.predict_single(data_snapshot, ticker)
            elif hasattr(self.predictor, 'predict'):
                ai_result = self.predictor.predict(data_snapshot)
            else:
                ai_result = {}
        except Exception as e:
            self.logger.error(f"Prediction failed for {ticker}: {e}")
            return
        
        # In v10.0 Meta-Labeling, the predictor directly outputs the signal
        signal = ai_result.get('signal', 'HOLD')
        confidence = ai_result.get('confidence', 0.0)
        
        # Check for errors or HOLD
        if signal == 'HOLD' or ai_result.get('status') == 'error':
            return

        # 2. Confluence Gate: Weekly Trend
        if isinstance(data_snapshot, pd.DataFrame) and 'weekly_trend' in data_snapshot.columns and not data_snapshot.empty:
            try:
                # 1 = Bullish, 0 = Bearish
                weekly_trend = data_snapshot['weekly_trend'].iloc[-1]
                weekly_rsi = data_snapshot['weekly_rsi'].iloc[-1] if 'weekly_rsi' in data_snapshot.columns else 0.0
                if signal == 'BUY' and weekly_trend == 0:
                    self.logger.info(
                        f"🚫 WEEKLY TREND VETO | {ticker} | BUY blocked | "
                        f"weekly_trend=BEARISH | weekly_rsi={weekly_rsi:.1f} | "
                        f"daily_conf={confidence:.1f}%"
                    )
                    return
                elif signal == 'SHORT' and weekly_trend == 1:
                    self.logger.info(
                        f"🚫 WEEKLY TREND VETO | {ticker} | SHORT blocked | "
                        f"weekly_trend=BULLISH | weekly_rsi={weekly_rsi:.1f} | "
                        f"daily_conf={confidence:.1f}%"
                    )
                    return
            except (IndexError, KeyError) as e:
                self.logger.warning(f"Weekly trend evaluation failed for {ticker}: {e}")

        # 2b. Confluence Gate: Weekly RSI
        if isinstance(data_snapshot, pd.DataFrame) and 'weekly_rsi' in data_snapshot.columns and not data_snapshot.empty:
            try:
                weekly_rsi = data_snapshot['weekly_rsi'].iloc[-1]
                weekly_trend = data_snapshot['weekly_trend'].iloc[-1] if 'weekly_trend' in data_snapshot.columns else 1
                trend_str = 'BULL' if weekly_trend else 'BEAR'
                
                if signal == 'BUY' and weekly_rsi > 75:
                    self.logger.info(
                        f"🚫 WEEKLY RSI VETO | {ticker} | BUY blocked | "
                        f"weekly_rsi={weekly_rsi:.1f} > 75 | weekly_trend={trend_str}"
                    )
                    return
                elif signal == 'SHORT' and weekly_rsi < 30:
                    self.logger.info(
                        f"🚫 WEEKLY RSI VETO | {ticker} | SHORT blocked | "
                        f"weekly_rsi={weekly_rsi:.1f} < 30 | weekly_trend={trend_str}"
                    )
                    return
            except (IndexError, KeyError) as e:
                self.logger.warning(f"Weekly RSI evaluation failed for {ticker}: {e}")

        # 2c. Risk:Reward Gate
        pt_pct = ai_result.get('take_profit_pct', 0.0)
        sl_pct = ai_result.get('stop_loss_pct', 0.0)
        
        if pt_pct != 0 and sl_pct != 0:
            rr_ratio = abs(pt_pct) / abs(sl_pct)
            if rr_ratio < 1.5:
                self.logger.info(f"❌ RR REJECTED: {ticker} R:R={rr_ratio:.2f} < 1.5")
                return {'action': 'HOLD', 'reason': f'Insufficient R:R {rr_ratio:.2f}'}

        # 3. Risk Check
        if isinstance(data_snapshot, pd.DataFrame) and 'close' in data_snapshot.columns and not data_snapshot.empty:
             current_price = data_snapshot['close'].iloc[-1]
             
             # Try to get ATR from ai_result or snapshot
             atr = ai_result.get('atr', current_price * 0.01) # Default 1% if missing
             if atr == 0: atr = current_price * 0.01
             
        else:
             # Fallback if data_snapshot is not a DataFrame
             self.logger.error("Invalid data snapshot format or missing close price")
             return

        # Institutional Logic: Use Sizer
        qty, details = self.sizer.calculate_size(
             symbol=ticker,
             price=current_price,
             atr=atr,
             conviction=confidence
        )
        
        if qty <= 0:
             self.logger.info(f"Size Zero {ticker}: {details.get('reason')}")
             return

        # 4. Execute
        # Pass predicted_probability down for journaling / debugging if available
        pred_prob = ai_result.get('predicted_probability', 0.0)
        self.logger.info(f"⚡ ACTION: {signal} {ticker} x {qty} (Conf: {confidence:.2f}, Prob: {pred_prob:.4f})")
        
        # Structure the result as expected by autonomous.py
        result = {
            'action': f"{signal}_EXECUTED",
            'ticker': ticker,
            'price': current_price,
            'quantity': qty,
            'confidence': confidence,
            'predicted_probability': pred_prob,
            'atr': atr,
            # Assign tentative stops/targets, which autonomous will use if active_signals
            'stop_loss': current_price - (2 * atr) if signal == 'BUY' else current_price + (2 * atr),
            'target_price': current_price + (2.5 * atr) if signal == 'BUY' else current_price - (2.5 * atr),
            'timestamp': data_snapshot.index[-1] if hasattr(data_snapshot, 'index') and not data_snapshot.empty else None
        }
        
        if self.oms:
            side_str = 'BUY' if signal == 'BUY' else 'SELL'
            self.oms.execute_order(
                symbol=ticker,
                side=side_str,
                qty=qty,
                order_type="MARKET"
            )
            
        return result

# Legacy Alias
MARK5DecisionEngine = DecisionEngine

"""
MARK5 Financial Engineering Core v2.0 - ARCHITECT REVISION
Changes:
- Parkinson Volatility (High/Low) for better intraday sensitivity.
- Transaction Cost Aware Labeling (The "Real World" filter).
- Vectorized optimizations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List

class FinancialEngineer:
    
    def __init__(self, volatility_window: int = 20, transaction_cost_pct: float = 0.001):
        """
        :param volatility_window: Window for dynamic thresholds.
        :param transaction_cost_pct: 0.001 = 0.1% round trip (covers STT + Slippage).
        """
        self.vol_window = volatility_window
        self.tc_pct = transaction_cost_pct

    # -------------------------------------------------------------------------
    # 1. STATIONARITY (Fractional Differentiation)
    # -------------------------------------------------------------------------
    def get_weights_ffd(self, d: float, thres: float, size: int) -> np.ndarray:
        w = [1.]
        for k in range(1, size):
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres: break
            w.append(w_)
        return np.array(w[::-1]).reshape(-1, 1)

    def frac_diff_ffd(self, series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
        w = self.get_weights_ffd(d, thres, len(series))
        width = len(w) - 1
        
        # Optimization: Use stride_tricks for vectorization instead of loop
        # However, for safety in variable length, we keep the robust loop but check inputs
        output = series.fillna(method='ffill').dropna()
        if width >= len(output): return output.diff().fillna(0)
        
        series_vals = output.values
        w_flat = w.flatten()
        
        # Convolve is faster than explicit loops for sliding windows
        # Valid mode ensures we don't have boundary leakage effects at the start
        res = np.convolve(series_vals, w_flat, mode='valid')
        
        # Align index: The result is shorter than input by 'width'
        return pd.Series(res, index=output.index[width:])

    # -------------------------------------------------------------------------
    # 2. VOLATILITY ESTIMATION (Parkinson > StdDev)
    # -------------------------------------------------------------------------
    def get_volatility(self, prices: pd.DataFrame, span0: int = 20) -> pd.Series:
        """
        Uses Parkinson Volatility (High/Low) if available, else standard close-to-close.
        Parkinson captures intraday range expansion better than Close Std.
        """
        if 'high' in prices.columns and 'low' in prices.columns:
            # Parkinson Volatility proxy
            # 0.361 is constant for estimator
            hl_ratio = np.log(prices['high'] / prices['low']) ** 2
            vol = np.sqrt(0.361 * hl_ratio.ewm(span=span0).mean())
            return vol
        else:
            # Fallback to standard deviation of returns
            return prices['close'].pct_change().ewm(span=span0).std()

    # -------------------------------------------------------------------------
    # 3. TRIPLE BARRIER METHOD (With Cost-Awareness)
    # -------------------------------------------------------------------------
    def apply_triple_barrier(self, 
                             close: pd.Series, 
                             events: pd.DataFrame, 
                             pt_sl: list, 
                             molecule: list) -> pd.DataFrame:
        
        events_ = events.loc[molecule]
        out = events_[['t1']].copy(deep=True)
        
        if pt_sl[0] > 0: pt = pt_sl[0] * events_['trgt']
        else: pt = pd.Series(index=events.index) # NaNs

        if pt_sl[1] > 0: sl = -pt_sl[1] * events_['trgt']
        else: sl = pd.Series(index=events.index) # NaNs

        # Vectorization is hard here due to path dependency, sticking to loop but optimized
        # We process only valid event indices
        for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
            df0 = close[loc:t1] 
            df0 = (df0 / close[loc] - 1) * 1 
            
            # Earliest stop loss hit
            sl_hit = df0[df0 < sl[loc]].index.min()
            # Earliest profit take hit
            pt_hit = df0[df0 > pt[loc]].index.min()
            
            out.loc[loc, 'sl'] = sl_hit
            out.loc[loc, 'pt'] = pt_hit
            
        return out

    def get_labels(self, prices: pd.DataFrame, t_events: list = None, run_bars: int = 5, pt_sl: list = [1, 1]) -> pd.DataFrame:
        """
        :param prices: DataFrame containing 'close', 'high', 'low'
        """
        close = prices['close']
        if t_events is None: t_events = close.index
            
        # 1. Dynamic Volatility
        vol = self.get_volatility(prices, span0=self.vol_window)
        
        # 2. Vertical Barriers
        t1 = close.index.searchsorted(t_events + pd.Timedelta(minutes=run_bars*5)) # Approx time mapping
        t1 = np.clip(t1, 0, len(close.index) - 1)
        t1_series = pd.Series(close.index[t1], index=t_events)
        
        # 3. Events
        events = pd.DataFrame({'t1': t1_series, 'trgt': vol.loc[t_events]})
        events = events.dropna(subset=['trgt'])
        
        # Filter: Skip events where volatility is too low to cover transaction costs
        # If Target < Transaction Cost, don't even label it, it's noise.
        min_vol_threshold = self.tc_pct * 1.5
        events = events[events['trgt'] > min_vol_threshold]

        if events.empty:
            return pd.DataFrame(columns=['ret', 'bin', 'out'])

        # 4. Apply Triple Barrier
        df_touches = self.apply_triple_barrier(close, events, pt_sl, events.index)
        
        # 5. Labeling
        df_touches['out'] = df_touches[['t1', 'sl', 'pt']].min(axis=1)
        df_touches = df_touches.dropna(subset=['out'])
        
        prices_out = close.loc[df_touches['out'].values].values
        prices_in = close.loc[df_touches.index].values
        
        out = pd.DataFrame(index=df_touches.index)
        out['ret'] = prices_out / prices_in - 1
        
        # --- THE REALITY CHECK (Transaction Costs) ---
        # 1 = Buy (Profit > Cost)
        # 0 = Hold/Sell (Loss OR Profit < Cost)
        
        # For Binary Classification (Signal vs Noise):
        # We only want to learn trades that clear the hurdle rate.
        out['bin'] = 0
        out.loc[out['ret'] > self.tc_pct, 'bin'] = 1 
        
        # Optional: Multiclass 
        # out.loc[out['ret'] < -self.tc_pct, 'bin'] = -1 
        
        return out

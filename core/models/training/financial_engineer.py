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
        output = series.ffill().dropna()
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
    def get_volatility(self, prices: pd.DataFrame, span0: int = 14) -> pd.Series:
        """
        ATR(14) / price — matches simulation barrier calculation exactly.
        
        CRITICAL: Training barriers MUST use the same volatility measure as
        the simulation/live system. The simulation uses ATR(14)/price for
        PT/SL barriers (test.py:669-674), so training must match.
        
        Previous bug: Used Parkinson vol (sqrt(0.361 × HL²)) which produces
        fundamentally different thresholds than ATR.
        """
        close = prices['close']
        
        if 'high' in prices.columns and 'low' in prices.columns:
            high = prices['high']
            low = prices['low']
            
            # True Range = max(H-L, |H-Cprev|, |L-Cprev|)
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR(14) smoothed, normalized by price
            atr = true_range.rolling(window=span0, min_periods=1).mean()
            vol = atr / close  # Normalize: fraction of price
            return vol
        else:
            # Fallback to standard deviation of returns
            return close.pct_change().rolling(window=span0, min_periods=1).std()

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

    def get_labels(self, prices: pd.DataFrame, t_events: list = None, run_bars: int = 7, pt_sl: list = [2, 1]) -> pd.DataFrame:
        """
        Triple barrier labeling for daily-bar swing trading.
        
        :param prices: DataFrame containing 'close', 'high', 'low'
        :param run_bars: Vertical barrier in trading days (default 7 = ~1.5 weeks)
        :param pt_sl: [profit_target_multiplier, stop_loss_multiplier] of ATR(14)
        """
        close = prices['close']
        if t_events is None: t_events = close.index
            
        # 1. Dynamic Volatility — ATR(14) normalized by price
        vol = self.get_volatility(prices, span0=14)
        
        # 2. Vertical Barriers — run_bars trading days ahead
        # For daily bars, shift by run_bars positions in the index
        t1 = close.index.searchsorted(t_events) + run_bars
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
        # C-2: NaT in sl/pt (barrier never hit) must be replaced with a sentinel
        # before min(). pandas min() over a NaT column coerces the row result to
        # NaT instead of falling back to t1, silently destroying label rows.
        for col in ['sl', 'pt']:
            if col in df_touches.columns:
                df_touches[col] = df_touches[col].fillna(pd.Timestamp.max)
        df_touches['out'] = df_touches[['t1', 'sl', 'pt']].min(axis=1)
        df_touches = df_touches.dropna(subset=['out'])

        # C-1: Filter to timestamps that exist in close.index. Holiday/half-day
        # sessions can produce 'out' timestamps not in close.index; .loc[] raises
        # KeyError and aborts the entire labeling run.
        df_touches = df_touches[df_touches['out'].isin(close.index)]

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
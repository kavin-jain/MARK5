import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
import logging

logger = logging.getLogger("MARK5.FinancialEngineer")

class FinancialEngineer:
    
    def __init__(self, volatility_window: int = 14, transaction_cost_pct: float = 0.001):
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
        
        output = series.ffill().dropna()
        if width >= len(output): return output.diff().fillna(0)
        
        series_vals = output.values
        w_flat = w.flatten()
        res = np.convolve(series_vals, w_flat, mode='valid')
        
        return pd.Series(res, index=output.index[width:])

    # -------------------------------------------------------------------------
    # 2. VOLATILITY ESTIMATION (ATR 14 / Close)
    # -------------------------------------------------------------------------
    def get_volatility(self, prices: pd.DataFrame, span0: int = 14) -> pd.Series:
        close = prices['close']
        if 'high' in prices.columns and 'low' in prices.columns:
            high = prices['high']
            low = prices['low']
            
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = true_range.ewm(alpha=1.0/span0, adjust=False).mean()
            vol = atr / close 
            return vol
        else:
            return close.pct_change().rolling(window=span0, min_periods=1).std()

    # -------------------------------------------------------------------------
    # 3. TRIPLE BARRIER METHOD (Aligned with Experiment D)
    # -------------------------------------------------------------------------
    def apply_triple_barrier(self, 
                             prices: pd.DataFrame, 
                             events: pd.DataFrame, 
                             pt_sl: list) -> pd.DataFrame:
        
        out = events[['t1', 't_day4']].copy(deep=True)
        
        # Absolute percentage barriers based on ATR
        pt = pt_sl[0] * events['trgt']
        sl = -pt_sl[1] * events['trgt']
        early_exit_thresh = 0.5 * events['trgt']

        for loc, row in events.iterrows():
            t1 = row['t1']
            t_day4 = row['t_day4']
            
            df0 = prices.loc[loc:t1] 
            if df0.empty: continue
            
            c0 = df0['close'].iloc[0]
            
            # REALITY CHECK: Use Highs for Targets, Lows for Stops
            dh = df0['high'] / c0 - 1
            dl = df0['low'] / c0 - 1
            dc = df0['close'] / c0 - 1
            
            # Find exact timestamps of barrier hits
            pt_hit = dh[dh >= pt[loc]].index.min()
            sl_hit = dl[dl <= sl[loc]].index.min()
            
            # Experiment D: Early Time-Stop Logic
            early_exit_hit = pd.NaT
            if pd.notna(t_day4) and t_day4 in dc.index:
                if dc.loc[t_day4] < early_exit_thresh[loc]:
                    early_exit_hit = t_day4
            
            out.loc[loc, 'sl'] = sl_hit
            out.loc[loc, 'pt'] = pt_hit
            out.loc[loc, 'early_exit'] = early_exit_hit
            
        return out

    def get_labels(self, prices: pd.DataFrame, t_events: list = None, run_bars: int = 70, pt_sl: list = [2.5, 2.0]) -> pd.DataFrame:
        close = prices['close']
        if t_events is None: t_events = close.index
            
        vol = self.get_volatility(prices, span0=self.vol_window)
        
        # Calculate T+70 (Hard Time Limit) and T+28 (Early Exit Limit)
        t_idx = close.index.searchsorted(t_events)
        t1_idx = np.clip(t_idx + run_bars, 0, len(close.index) - 1)
        t4_idx = np.clip(t_idx + 28, 0, len(close.index) - 1)
        
        t1_series = pd.Series(close.index[t1_idx], index=t_events)
        t4_series = pd.Series(close.index[t4_idx], index=t_events)
        
        events = pd.DataFrame({'t1': t1_series, 't_day4': t4_series, 'trgt': vol.loc[t_events]})
        events = events.dropna(subset=['trgt'])
        
        # Drop environments where transaction costs eat the whole ATR
        min_vol_threshold = self.tc_pct * 1.5
        events = events[events['trgt'] > min_vol_threshold]

        if events.empty:
            return pd.DataFrame(columns=['ret', 'bin', 'fwd_ret'])

        # Apply Physics-Accurate Barriers
        df_touches = self.apply_triple_barrier(prices, events, pt_sl)
        
        # FIX: Unified Timezone Handling for min(axis=1)
        tz = prices.index.tz
        max_ts = pd.Timestamp('2100-01-01')
        if tz is not None:
            max_ts = max_ts.tz_localize(tz)
        
        # Replace NaTs with future infinity so min() works correctly
        for col in ['sl', 'pt', 'early_exit']:
            if col in df_touches.columns:
                df_touches[col] = df_touches[col].fillna(max_ts)
                
        # Find the EARLIEST event that occurred
        df_touches['out'] = df_touches[['t1', 'sl', 'pt', 'early_exit']].min(axis=1)
        # Record WHICH barrier was hit (for accurate labeling)
        df_touches['first_touch'] = df_touches[['t1', 'sl', 'pt', 'early_exit']].idxmin(axis=1)
        
        # Ensure 'out' is comparable with index
        df_touches = df_touches[df_touches['out'].isin(close.index)]

        prices_out = close.loc[df_touches['out']].values
        prices_in = close.loc[df_touches.index].values
        
        out = pd.DataFrame(index=df_touches.index)
        out['ret'] = prices_out / prices_in - 1
        
        # Expose a clean 10-day forward return for the IC script
        prices_t1 = close.loc[events.loc[out.index, 't1']].values
        out['fwd_ret'] = prices_t1 / prices_in - 1

        # --- STRICT BINARY LABELING ---
        # 1 = True Breakout (Hit 2.5R Target OR timed out with >1.0 ATR profit)
        # 0 = Noise/Loss (Hit Stop, Early Exit, or timed out with flat/negative return)
        out['bin'] = 0
        
        hit_pt = df_touches['first_touch'] == 'pt'
        survived_with_profit = (df_touches['first_touch'] == 't1') & (out['ret'] > events.loc[out.index, 'trgt'] * 1.0)
        
        out.loc[hit_pt | survived_with_profit, 'bin'] = 1 
        
        return out
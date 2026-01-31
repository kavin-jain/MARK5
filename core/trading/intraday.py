"""
🔥 MARK5 INTRADAY TRADING UTILITIES
===================================
Helper functions for intraday trading operations

Author: MARK5 Trading System
Version: 1.0 - Intraday Ready
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple, Optional
import pytz


def generate_intraday_labels(
    data: pd.DataFrame,
    horizons: List[int] = [1, 15, 60],  # minutes
    interval: str = '15m',
    buy_threshold: float = 0.003,  # 0.3%
    sell_threshold: float = -0.003  # -0.3%
) -> pd.DataFrame:
    """
    Generate intraday trading labels based on forward returns
    
    Args:
        data: DataFrame with OHLCV data (DatetimeIndex)
        horizons: List of forward-looking periods in minutes
        interval: Data interval (e.g., '15m')
        buy_threshold: Minimum return % for BUY label
        sell_threshold: Maximum return % for SELL label
    
    Returns:
        DataFrame with added label columns
    """
    df = data.copy()
    
    # Convert interval to periods
    interval_minutes = _parse_interval_to_minutes(interval)
    
    for horizon in horizons:
        periods_ahead = horizon // interval_minutes
        
        # Calculate forward returns
        df[f'forward_return_{horizon}m'] = df['close'].pct_change(periods=periods_ahead).shift(-periods_ahead)
        
        # Generate labels
        df[f'label_{horizon}m'] = 0  # HOLD
        df.loc[df[f'forward_return_{horizon}m'] >= buy_threshold, f'label_{horizon}m'] = 1  # BUY
        df.loc[df[f'forward_return_{horizon}m'] <= sell_threshold, f'label_{horizon}m'] = -1  # SELL
        
        # Add volatility-adjusted labels (dynamic thresholds)
        rolling_std = df['returns'].rolling(window=20).std()
        dynamic_buy = rolling_std * 2  # 2 std for BUY
        dynamic_sell = -rolling_std * 2  # -2 std for SELL
        
        df[f'label_adaptive_{horizon}m'] = 0
        df.loc[df[f'forward_return_{horizon}m'] >= dynamic_buy, f'label_adaptive_{horizon}m'] = 1
        df.loc[df[f'forward_return_{horizon}m'] <= dynamic_sell, f'label_adaptive_{horizon}m'] = -1
    
    return df


def _parse_interval_to_minutes(interval: str) -> int:
    """Convert interval string to minutes"""
    interval_map = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '60m': 60
    }
    return interval_map.get(interval, 15)  # Default 15 minutes


def add_session_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add NSE session-based features
    
    NSE Sessions:
    - Opening: 09:15 - 10:00 (first 45 minutes)
    - Midday: 10:00 - 14:30 
    - Closing: 14:30 - 15:30 (last hour)
    """
    df = data.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    
    # Extract hour and minute
    hour = df.index.hour
    minute = df.index.minute
    
    # Session indicators
    df['is_opening_session'] = ((hour == 9) & (minute >= 15)) | ((hour == 10) & (minute == 0))
    df['is_closing_session'] = ((hour == 14) & (minute >= 30)) | (hour == 15)
    df['is_midday_session'] = ~(df['is_opening_session'] | df['is_closing_session'])
    
    # Time-based features
    df['minutes_since_open'] = (hour - 9) * 60 + minute - 15
    df['minutes_until_close'] = (15 - hour) * 60 + (30 - minute)
    
    # Normalize time features
    df['time_of_day'] = df['minutes_since_open'] / (6 * 60 + 15)  # Normalize to [0, 1]
    
    return df


def add_intraday_volume_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add intraday volume analysis features
    
    Features:
    - Volume participation (% of daily volume traded so far)
    - VWAP distance
    - Volume spikes
    """
    df = data.copy()
    
    if 'volume' not in df.columns:
        return df
    
    # Calculate daily cumulative volume
    df['_temp_daily_date'] = df.index.date
    df['daily_cumulative_volume'] = df.groupby('_temp_daily_date')['volume'].cumsum()
    df['daily_total_volume'] = df.groupby('_temp_daily_date')['volume'].transform('sum')
    df['volume_participation'] = df['daily_cumulative_volume'] / (df['daily_total_volume'] + 1e-10)
    
    # VWAP (Volume Weighted Average Price)
    # 🔥 FIX ISSUE #9: Prevent division by zero
    cumulative_vol = df['volume'].cumsum() + 1e-10
    df['vwap'] = (df['close'] * df['volume']).cumsum() / cumulative_vol
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
    
    # Intraday VWAP (reset daily)
    df['intraday_vwap'] = df.groupby('_temp_daily_date').apply(
        lambda x: (x['close'] * x['volume']).cumsum() / (x['volume'].cumsum() + 1e-10)
    ).reset_index(level=0, drop=True)
    df['intraday_vwap_distance'] = (df['close'] - df['intraday_vwap']) / (df['intraday_vwap'] + 1e-10)
    
    # Volume spikes
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_spike'] = df['volume'] / (df['volume_ma_20'] + 1e-10)
    df['is_volume_spike'] = (df['volume_spike'] > 2.0).astype(int)
    
    # Clean up temporary columns
    df.drop(columns=['_temp_daily_date'], inplace=True, errors='ignore')
    
    return df


def add_order_flow_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add order flow analysis features
    
    Approximations based on OHLC data:
    - Buy/sell pressure from price changes
    - Order flow imbalance
    """
    df = data.copy()
    
    # Buy/sell pressure (simplified)
    df['buy_pressure'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)).fillna(0.5)
    df['sell_pressure'] = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)).fillna(0.5)
    
    # Order flow imbalance
    df['order_flow_imbalance'] = df['buy_pressure'] - df['sell_pressure']
    df['order_flow_imbalance_ma'] = df['order_flow_imbalance'].rolling(10).mean()
    
    # Price momentum within bar
    df['intrabar_momentum'] = (df['close'] - df['open']) / (df['open'] + 1e-10)
    
    # Buying/selling strength
    df['buying_strength'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    df['selling_strength'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
    
    return df


def add_spread_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add bid-ask spread approximations
    
    Note: True spread requires Level-2 data. This uses High-Low as proxy.
    """
    df = data.copy()
    
    # High-Low spread (proxy for bid-ask spread)
    # 🔥 FIX ISSUE #9: Prevent division by zero
    df['hl_spread'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
    df['hl_spread_ma'] = df['hl_spread'].rolling(20).mean()
    df['hl_spread_ratio'] = df['hl_spread'] / (df['hl_spread_ma'] + 1e-10)
    
    # Spread volatility
    df['spread_volatility'] = df['hl_spread'].rolling(20).std()
    
    return df


def is_market_open(timestamp: Optional[datetime] = None) -> bool:
    """
    Check if NSE market is currently open
    
    Args:
        timestamp: Timestamp to check (default: current time)
    
    Returns:
        True if market is open
    """
    if timestamp is None:
        timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))
    
    # Check if it's a weekday (Mon-Fri)
    if timestamp.weekday() >= 5:
        return False
        
    # 🔥 FIX ISSUE #10: NSE Holiday Calendar
    # Format: YYYY-MM-DD
    nse_holidays = {
        # 2024
        '2024-01-22', '2024-01-26', '2024-03-08', '2024-03-25', '2024-03-29',
        '2024-04-11', '2024-04-17', '2024-05-01', '2024-05-20', '2024-06-17',
        '2024-07-17', '2024-08-15', '2024-10-02', '2024-11-01', '2024-11-15',
        '2024-12-25',
        # 2025 (Estimated/Tentative)
        '2025-01-26', '2025-02-26', '2025-03-14', '2025-03-31', '2025-04-06',
        '2025-04-10', '2025-04-14', '2025-05-01', '2025-08-15', '2025-08-27',
        '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25'
    }
    
    date_str = timestamp.strftime('%Y-%m-%d')
    if date_str in nse_holidays:
        return False
    
    # Check market hours (09:15 - 15:30 IST)
    market_open = time(9, 15)
    market_close = time(15, 30)
    
    current_time = timestamp.time()
    return market_open <= current_time <= market_close


def get_next_market_open(timestamp: Optional[datetime] = None) -> datetime:
    """Get next market open time"""
    if timestamp is None:
        timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))
    
    # Start from next day if after market close
    if timestamp.time() > time(15, 30):
        timestamp = timestamp + timedelta(days=1)
    
    # Skip weekends and holidays
    nse_holidays = {
        '2024-01-22', '2024-01-26', '2024-03-08', '2024-03-25', '2024-03-29',
        '2024-04-11', '2024-04-17', '2024-05-01', '2024-05-20', '2024-06-17',
        '2024-07-17', '2024-08-15', '2024-10-02', '2024-11-01', '2024-11-15',
        '2024-12-25',
        '2025-01-26', '2025-02-26', '2025-03-14', '2025-03-31', '2025-04-06',
        '2025-04-10', '2025-04-14', '2025-05-01', '2025-08-15', '2025-08-27',
        '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25'
    }
    
    while timestamp.weekday() >= 5 or timestamp.strftime('%Y-%m-%d') in nse_holidays:
        timestamp = timestamp + timedelta(days=1)
    
    # Set to market open time
    return timestamp.replace(hour=9, minute=15, second=0, microsecond=0)


def calculate_intraday_metrics(trades: List[Dict]) -> Dict:
    """
    Calculate intraday-specific performance metrics
    
    Args:
        trades: List of trade dictionaries with keys:
            - entry_time, exit_time, entry_price, exit_price, quantity
    
    Returns:
        Dictionary of metrics
    """
    if not trades:
        return {}
    
    # Convert to DataFrame
    df = pd.DataFrame(trades)
    
    # Calculate P&L
    df['pnl'] = (df['exit_price'] - df['entry_price']) * df['quantity']
    # 🔥 FIX ISSUE #9: Prevent division by zero
    df['return_pct'] = (df['exit_price'] - df['entry_price']) / (df['entry_price'] + 1e-10) * 100
    
    # Duration
    df['duration'] = (pd.to_datetime(df['exit_time']) - pd.to_datetime(df['entry_time'])).dt.total_seconds() / 60
    
    # Metrics
    total_trades = len(df)
    winning_trades = (df['pnl'] > 0).sum()
    losing_trades = (df['pnl'] < 0).sum()
    
    total_profit = df[df['pnl'] > 0]['pnl'].sum()
    total_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': winning_trades / total_trades * 100 if total_trades > 0 else 0,
        'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
        'avg_win': df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0,
        'avg_loss': df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0,
        'avg_duration_minutes': df['duration'].mean(),
        'total_pnl': df['pnl'].sum(),
        'avg_return_pct': df['return_pct'].mean()
    }


def filter_trading_hours(data: pd.DataFrame, timezone: str = 'Asia/Kolkata') -> pd.DataFrame:
    """
    Filter data to only include trading hours (09:15 - 15:30 IST)
    
    Args:
        data: DataFrame with DatetimeIndex
        timezone: Timezone string
    
    Returns:
        Filtered DataFrame
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        return data
    
    # Ensure timezone-aware
    if data.index.tz is None:
        data.index = data.index.tz_localize(timezone)
    else:
        data.index = data.index.tz_convert(timezone)
    
    # Filter by time
    mask = (
        (data.index.time >= time(9, 15)) &
        (data.index.time <= time(15, 30)) &
        (data.index.dayofweek < 5)  # Weekdays only
    )
    
    return data[mask]


def resample_to_interval(data: pd.DataFrame, interval: str = '15m') -> pd.DataFrame:
    """
    Resample data to specified interval
    
    Args:
        data: DataFrame with OHLCV data
        interval: Target interval (e.g., '15m', '1h')
    
    Returns:
        Resampled DataFrame
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        return data
    
    # OHLC resampling
    resampled = data.resample(interval).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled


if __name__ == '__main__':
    # Example usage
    print("🔥 MARK5 Intraday Utilities")
    print("=" * 50)
    
    # Check market status
    market_open = is_market_open()
    print(f"\n📊 Market Status: {'OPEN 🟢' if market_open else 'CLOSED 🔴'}")
    
    if not market_open:
        next_open = get_next_market_open()
        print(f"   Next Open: {next_open.strftime('%Y-%m-%d %H:%M %Z')}")
    
    # Example label generation
    print("\n📈 Label Generation Example:")
    print("   Horizons: [1, 15, 60] minutes")
    print("   Buy Threshold: +0.3%")
    print("   Sell Threshold: -0.3%")
    
    print("\n✅ Utilities loaded successfully!")

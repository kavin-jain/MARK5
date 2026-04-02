"""
scripts/validate_ranker.py
Quick validation: does top-3 ranking beat NIFTY on out-of-sample data?
Run this BEFORE integrating into test.py.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import traceback
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from core.data.adapters.kite_adapter import KiteFeedAdapter
from core.models.ranker import CrossSectionalRanker

kite = KiteFeedAdapter({})
if not kite.connect():
    print("Failed to connect to Kite. Check API credentials in .env")
    sys.exit(1)

UNIVERSE = ['RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','ICICIBANK.NS',
            'SBIN.NS','BHARTIARTL.NS','ITC.NS','AXISBANK.NS','MARUTI.NS',
            'WIPRO.NS','SUNPHARMA.NS','TECHM.NS','TITAN.NS','KOTAKBANK.NS']

TEST_START = '2022-01-01'
TEST_END   = '2025-01-01'

print("Downloading data via Kite...")
data = {}
start_dt = pd.to_datetime('2019-01-01').tz_localize('Asia/Kolkata')
end_dt = pd.to_datetime(TEST_END).tz_localize('Asia/Kolkata')

for sym in UNIVERSE:
    df = kite.fetch_ohlcv(sym, from_date=start_dt, to_date=end_dt, interval='day')
    if not df.empty:
        df.columns = [str(c).lower() for c in df.columns]
        data[sym] = df

print("Fetching NIFTY50...")
nifty = kite.fetch_index_data('NIFTY50', interval='day', days_back=2500)
if not nifty.empty:
    nifty = nifty.loc[nifty.index <= end_dt]
    nifty.columns = [str(c).lower() for c in nifty.columns]
    nifty_close = nifty['close']
else:
    print("Failed to fetch NIFTY50 data")
    sys.exit(1)

ranker = CrossSectionalRanker(top_n=3)
test_dates = pd.bdate_range(TEST_START, TEST_END, freq='W-FRI', tz='Asia/Kolkata')

# Weekly rebalancing backtest
portfolio_value = 1.0
portfolio_curve = [1.0]
nifty_curve     = [1.0]
try:
    nifty_base      = float(nifty_close.loc[nifty_close.index >= pd.Timestamp(TEST_START).tz_localize('Asia/Kolkata')].iloc[0])
except Exception:
    nifty_base = float(nifty_close.iloc[0])
    
prev_holdings = []
holding_returns = {s: 0.0 for s in UNIVERSE}

print("Running simulation...")
for i, rebal_date in enumerate(test_dates[1:], 1):
    prev_date = test_dates[i-1]
    curr_ts   = pd.Timestamp(rebal_date)
    prev_ts   = pd.Timestamp(prev_date)

    # Compute returns for existing holdings over this week
    if prev_holdings:
        week_return = 0.0
        valid_holdings = 0
        for sym in prev_holdings:
            df = data[sym]
            avail = df.loc[(df.index >= prev_ts) & (df.index <= curr_ts)]
            if len(avail) >= 2:
                week_return += float(avail['close'].iloc[-1]) / float(avail['close'].iloc[0]) - 1.0
                valid_holdings += 1
        if valid_holdings > 0:
            week_return /= valid_holdings
            
            # Recompute rankings to see turnover
            try:
                ranked = ranker.rank_universe(data, nifty_close, None, curr_ts)
                new_top_3 = [s for s, _ in ranked[:3]]
                turnover = len(set(prev_holdings) ^ set(new_top_3)) / 2.0 / 3.0  # Fraction replaced
            except Exception as e:
                turnover = 0.0
                new_top_3 = prev_holdings
                
            cost = turnover * 0.0015
            portfolio_value *= (1 + week_return - cost)
        else:
            try:
                ranked = ranker.rank_universe(data, nifty_close, None, curr_ts)
                new_top_3 = [s for s, _ in ranked[:3]]
            except Exception:
                new_top_3 = []
    else:
        try:
            ranked = ranker.rank_universe(data, nifty_close, None, curr_ts)
            new_top_3 = [s for s, _ in ranked[:3]]
        except Exception as e:
            traceback.print_exc()
            new_top_3 = []

    # Rebalance
    prev_holdings = new_top_3

    portfolio_curve.append(portfolio_value)
    
    # Debug print every 12 weeks (~quarterly)
    if i % 12 == 1:
        rankings = ranker.rank_universe(data, nifty_close, None, curr_ts)
        print(f"[{rebal_date.strftime('%Y-%m-%d')}] NIFTY: {nifty_curve[-1]*nifty_base:.1f} | PORT: {portfolio_value:.3f}")
        for r_sym, r_score in rankings[:3]:
            print(f"  + {r_sym}: {r_score:.3f}")
        for r_sym, r_score in rankings[-2:]:
            print(f"  - {r_sym}: {r_score:.3f}")

    try:
        nifty_now = float(nifty_close.loc[nifty_close.index <= curr_ts].iloc[-1])
        nifty_curve.append(nifty_now / nifty_base)
    except Exception:
        nifty_curve.append(nifty_curve[-1])

# Results
port_series  = pd.Series(portfolio_curve, index=test_dates[:len(portfolio_curve)])
nifty_series = pd.Series(nifty_curve,     index=test_dates[:len(nifty_curve)])

total_return_port  = (port_series.iloc[-1]  - 1) * 100
total_return_nifty = (nifty_series.iloc[-1] - 1) * 100
years = (test_dates[len(portfolio_curve)-1] - test_dates[0]).days / 365.25
cagr_port  = ((port_series.iloc[-1])  ** (1/max(years, 0.1)) - 1) * 100
cagr_nifty = ((nifty_series.iloc[-1]) ** (1/max(years, 0.1)) - 1) * 100

daily_port = port_series.pct_change().dropna()
sharpe = (daily_port.mean() / (daily_port.std() + 1e-9)) * np.sqrt(52)  # weekly

print(f"\n{'='*50}")
print(f"Period: {TEST_START} to {TEST_END} ({years:.1f} years)")
print(f"{'='*50}")
print(f"Portfolio total return: {total_return_port:+.1f}%")
print(f"NIFTY total return:     {total_return_nifty:+.1f}%")
print(f"Excess return:          {total_return_port - total_return_nifty:+.1f}%")
print(f"Portfolio CAGR:         {cagr_port:+.1f}%")
print(f"NIFTY CAGR:             {cagr_nifty:+.1f}%")
print(f"Weekly Sharpe:          {sharpe:.2f}")
print(f"{'='*50}")
print(f"\nGo/No-Go: {'✅ PROCEED' if cagr_port > cagr_nifty else '❌ FACTOR NOT WORKING'}")
